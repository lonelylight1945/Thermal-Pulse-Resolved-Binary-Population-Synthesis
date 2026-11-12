import numpy as np

from CEE import CEE
from pre_MT_step import bps_step, initial_spin, spin_breakup_limit, tide_step, wind_spin_step
from rlf import R_l
from companion import tau_MS


def evolve(history, M2_i, a_i, tau_evo):
    """Evolve one binary along an in-memory donor track.

    Returns: M1_f, M2_f, a_f, type
    """
    age = history["age"]
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

    M2 = M2_i
    a = a_i
    f = 0.0
    J_spin = initial_spin(M1_track[0], R1_track[0], I_total[0])
    J_excess = np.zeros_like(age)
    core_ratio_limit = 0.9 if history["zams_star_mass"] <= 3.0 else 0.95

    R_L = R_l(M1_track[0], M2, a)
    if R1_track[0] >= R_L:
        return M1_track[0], M2, a, 1

    for k in range(1, len(age)):
        k_prev = k - 1
        dt = age[k] - age[k_prev]

        if age[k] > tau_evo:
            return M1_track[k_prev], M2, a, 3

        f += dt / tau_MS(M2)
        if f > 1.0:
            return M1_track[k_prev], M2, a, 2

        # In MESA history, mdot[k] integrates the interval [k - 1, k].
        M2_new, a_new = bps_step(
            M1_track[k_prev], M2, R1_track[k_prev], a, M_wind[k], dt
        )

        # Additional Rejuvenation of a convective-core MS accretor
        if M2_new > 1.15 and M2_new > M2:
            f *= M2 / M2_new

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
        R_L = R_l(M1_track[k], M2_new, a_new)

        if R1_track[k] >= R_L:
            # if M1_track[k] / M2_new <= 2.0:
            #     return M1_track[k], M2_new, a_new, 4

            core_ratio = M1_co_core[k] / M1_he_core[k] if M1_he_core[k] > 0.0 else 0.0
            if core_ratio <= core_ratio_limit:
                return M1_track[k], M2_new, a_new, 5

            if tau_evo - age[k] > (1.0 - f) * tau_MS(M2_new):
                return M1_track[k], M2_new, a_new, 6

            return CEE(
                M1_track[k], M1_he_core[k], M2_new, a_new, f, E_grav[k], E_int[k]
            )

        M2 = M2_new
        a = a_new

    return M1_track[-1], M2, a, 11
