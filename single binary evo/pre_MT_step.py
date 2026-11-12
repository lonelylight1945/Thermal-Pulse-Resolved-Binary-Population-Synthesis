import numpy as np

from rlf import R_l


AU_RSUN = 214.95
I_UNIT = 1.98840987069805e33 * 6.957e10**2  # M_sun R_sun^2, cgs
TIDE_FILL_FACTOR = 0.01


def bps_step(M_1: float,      # donor mass, M_sun
             M_2: float,      # accretor mass, M_sun
             R_1: float,      # donor radius, R_sun
             a: float,        # separation, R_sun
             M_wind: float,   # donor's wind mass loss rate, M_sun/yr, 应为正值
             dt:float,        # time step, yr
) -> tuple[float, float]:
    """
    参考 BSE/Hurley 2002 的 detached prescription 进行pre-MT的演化
    暂且只考虑的stellar wind一种双星作用,  只演化圆轨道
    M_1 为 donor，M_2 为 accretor
    """
    beta_w: float = 0.125; alpha_w: float = 1.5
    M_tot = M_1 + M_2

    # 计算星风吸积率
    term3 = (1 + M_tot * R_1 / (2.0 * beta_w * M_1 * a))**(-3/2)
    term2 = (M_2 * R_1 / (2.0 * beta_w * M_1))**2
    term1 = alpha_w / (2 * a**2)
    accf = term1 * term2 * term3

    # 防止得到一个不合理的吸积率
    accf = min(max(accf, 0.0), 0.8)
    acc_mass = accf * M_wind * dt

    # 轨道比角动量损失量，dj/j
    dj_frac = (M_wind*dt * M_2 / (M_1 * M_tot) + acc_mass / M_tot)

    # j_new/j_old
    j_ratio = 1.0 - dj_frac

    # 更新M_1、M_2 及 a
    M_1_new = M_1 - M_wind * dt
    M_2_new = M_2 + acc_mass
    M_tot_new = M_1_new + M_2_new

    a_new = a * j_ratio**2 * M_tot_new/M_tot * (M_1 * M_2 / (M_1_new * M_2_new))**2

    return M_2_new, a_new


def initial_spin(M_1, R_1, I_total):
    """Return the donor spin angular momentum in M_sun R_sun^2 yr^-1."""
    if R_1 <= 0.0 or I_total <= 0.0:
        return 0.0

    v_rot = 330.0 * M_1**3.3 / (15.0 + M_1**3.45)  # km s^-1, BSE fit
    omega = 45.35 * v_rot / R_1  # yr^-1
    return (I_total / I_UNIT) * omega


def wind_spin_step(J_spin, I_total, R_1, M_wind, dt):
    """Remove the surface specific spin angular momentum carried by the wind."""
    I = I_total / I_UNIT
    if I <= 0.0 or J_spin <= 0.0:
        return 0.0

    omega = J_spin / I
    dJ_wind = (2.0 / 3.0) * M_wind * dt * R_1**2 * omega
    return max(0.0, J_spin - dJ_wind)


def spin_breakup_limit(M_1, R_1, I_total, J_spin):
    """Apply BSE's breakup-spin cap and return retained and removed spin J."""
    I = I_total / I_UNIT
    if min(M_1, R_1, I) <= 0.0 or J_spin <= 0.0:
        return J_spin, 0.0

    omega_crit = 2.0 * np.pi * np.sqrt(M_1 * AU_RSUN**3 / R_1**3)
    J_crit = I * omega_crit
    J_excess = max(0.0, J_spin - J_crit)
    return J_spin - J_excess, J_excess


def tide_step(M_1, M_2, R_1, L_1, M_env_conv, R_env_conv, I_total,
              a, J_spin, dt):
    """Apply donor tides for a circular binary and return J_spin, a."""
    I = I_total / I_UNIT
    if min(M_1, M_2, R_1, L_1, a, I) <= 0.0:
        return J_spin, a

    R_L = R_l(M_1, M_2, a)
    if R_1 < TIDE_FILL_FACTOR * R_L:
        return J_spin, a

    omega_orb = 2.0 * np.pi / ((a / AU_RSUN) * np.sqrt(a / (AU_RSUN * (M_1 + M_2))))
    omega_spin = J_spin / I
    q = M_2 / M_1
    r_over_a_6 = (R_1 / a)**6
    rg2 = I / (M_1 * R_1**2)

    if M_env_conv > 0.0 and R_env_conv > 0.0:
        t_conv = 0.4311 * (
            M_env_conv * R_env_conv * (R_1 - 0.5 * R_env_conv) / (3.0 * L_1)
        )**(1.0 / 3.0)
        t_tide = 2.0 * np.pi / (abs(omega_orb - omega_spin) + 1.0e-10)
        f_conv = min(1.0, (t_tide / (2.0 * t_conv))**2)
        tcqr = 2.0 * f_conv * q * r_over_a_6 * M_env_conv / (21.0 * t_conv * M_1)
    else:
        t_rad = 1.592e-9 * M_1**2.84
        f_rad = 1.9782e4 * np.sqrt(M_1 * R_1**2 / a**5) * t_rad * (1.0 + q)**(5.0 / 6.0)
        tcqr = f_rad * q * r_over_a_6

    domega = 3.0 * q * tcqr * (omega_orb - omega_spin) / rg2
    delta_omega = domega * dt
    delta_omega = np.clip(delta_omega, min(0.0, omega_orb - omega_spin),
                          max(0.0, omega_orb - omega_spin))
    dJ_tide = I * delta_omega

    mu = M_1 * M_2 / (M_1 + M_2)
    J_orb = mu * a**2 * omega_orb
    if dJ_tide >= J_orb:
        dJ_tide = 0.99 * J_orb

    J_spin += dJ_tide
    a *= (1.0 - dJ_tide / J_orb)**2
    return J_spin, a
