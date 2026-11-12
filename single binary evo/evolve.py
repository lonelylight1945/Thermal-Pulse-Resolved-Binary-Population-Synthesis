import numpy as np

from CEE import CEE
from companion import tau_MS
from pre_MT_step import bps_step, initial_spin, spin_breakup_limit, tide_step, wind_spin_step
from rlf import R_l


def orbital_period_days(M1, M2, a):
    return np.sqrt((a / 214.95) ** 3 / (M1 + M2)) * 365.24


def evolve(history, M2_i, a_i, tau_evo):
    """Trace one binary and return its full pre-CE evolution history."""
    age_track = history["age"]
    M1_track = history["star_mass"]
    M1_he_core = history["he_core_mass"]
    M1_co_core = history["co_core_mass"]
    R1_track = 10.0 ** history["log_R"]
    L1_track = 10.0 ** history["log_L"]
    M_wind = 10.0 ** history["log_abs_mdot"]
    E_grav = history["env_grav_energy"]
    E_int = history["env_internal_energy"]
    M_env_conv = history["M_env_conv"]
    R_env_conv = history["R_env_conv"]
    I_total = history["I_total"]

    age, M1, R1, R1_over_RL1, M2, a, P_orb = ([] for _ in range(7))

    def add_state(k, M2_now, a_now):
        R_L = R_l(M1_track[k], M2_now, a_now)
        age.append(age_track[k])
        M1.append(M1_track[k])
        R1.append(R1_track[k])
        R1_over_RL1.append(R1_track[k] / R_L)
        M2.append(M2_now)
        a.append(a_now)
        P_orb.append(orbital_period_days(M1_track[k], M2_now, a_now))
        return R_L

    def result(event_type, ce=None):
        return {
            "age": np.asarray(age),
            "M1": np.asarray(M1),
            "R1": np.asarray(R1),
            "R1_over_RL1": np.asarray(R1_over_RL1),
            "M2": np.asarray(M2),
            "a": np.asarray(a),
            "P_orb": np.asarray(P_orb),
            "E_grav": np.asarray(E_grav[:len(age)]),
            "E_int": np.asarray(E_int[:len(age)]),
            "I_total": np.asarray(I_total[:len(age)]),
            "M_wind": np.asarray(M_wind[:len(age)]),
            "type": event_type,
            "CEE": ce,
        }

    M2_now = float(M2_i)
    a_now = float(a_i)
    f = 0.0
    J_spin = initial_spin(M1_track[0], R1_track[0], I_total[0])
    J_excess = np.zeros_like(age_track)
    core_ratio_limit = 0.9 if float(history["zams_star_mass"]) <= 3.0 else 0.95

    R_L = add_state(0, M2_now, a_now)
    if R1_track[0] >= R_L:
        return result(1)

    for k in range(1, len(age_track)):
        k_prev = k - 1
        dt = age_track[k] - age_track[k_prev]

        if age_track[k] > tau_evo:
            return result(3)

        f += dt / tau_MS(M2_now)
        if f > 1.0:
            return result(2)

        # In MESA history, mdot[k] integrates the interval [k - 1, k].
        M2_new, a_new = bps_step(
            M1_track[k_prev], M2_now, R1_track[k_prev], a_now, M_wind[k], dt
        )

        # Additional Rejuvenation of a convective-core MS accretor
        if M2_new > 1.15 and M2_new > M2_now:
            f *= M2_now / M2_new

        J_spin = wind_spin_step(
            J_spin, I_total[k_prev], R1_track[k_prev], M_wind[k], dt
        )
        J_spin, a_new = tide_step(
            M1_track[k], M2_new, R1_track[k], L1_track[k], M_env_conv[k],
            R_env_conv[k], I_total[k], a_new, J_spin, dt
        )
        J_spin, J_excess[k] = spin_breakup_limit(
            M1_track[k], R1_track[k], I_total[k], J_spin
        )
        R_L = add_state(k, M2_new, a_new)

        if R1_track[k] >= R_L:
            # if M1_track[k] / M2_new <= 2.0:
                # return result(4)

            core_ratio = M1_co_core[k] / M1_he_core[k] if M1_he_core[k] > 0.0 else 0.0
            if core_ratio <= core_ratio_limit:
                return result(5)

            if tau_evo - age_track[k] > (1.0 - f) * tau_MS(M2_new):
                return result(6)

            M1_CE, M2_CE, a_CE, E_grav_CE, E_int_CE, E_bind_CE, event_type = CEE(
                M1_track[k], M1_he_core[k], M2_new, a_new, f, E_grav[k], E_int[k]
            )
            ce = {
                "M1": M1_CE,
                "M2": M2_CE,
                "a": a_CE,
                "E_grav": E_grav_CE,
                "E_int": E_int_CE,
                "E_bind": E_bind_CE,
                "P_orb": orbital_period_days(M1_CE, M2_CE, a_CE) if event_type in (8, 9) else None,
            }
            return result(event_type, ce)

        M2_now = M2_new
        a_now = a_new

    return result(11)
