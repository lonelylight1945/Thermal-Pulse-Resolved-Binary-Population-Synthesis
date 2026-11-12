from companion import R_ms
from rlf import R_l


def CEE(M1, M1_core, M2, a, f, E_grav, E_int, alpha_ce=0.3, alpha_th=0.5):
    """Return M1_f, M2_f, a_f, E_grav, E_int, E_bind, type."""
    G = 6.67430e-8
    M_sun = 1.98847e33
    R_sun = 6.957e10

    E_bind = E_grav + alpha_th * E_int
    if E_bind >= 0.0:
        return M1, M2, a, E_grav, E_int, E_bind, 7

    M1_cgs = M1 * M_sun
    M1_core_cgs = M1_core * M_sun
    M2_cgs = M2 * M_sun
    a_cgs = a * R_sun

    E_orb_i = -G * M1_cgs * M2_cgs / (2.0 * a_cgs)
    E_orb_f = E_orb_i + E_bind / alpha_ce
    a_f = -G * M1_core_cgs * M2_cgs / (2.0 * E_orb_f) / R_sun

    R2_L = R_l(M2, M1_core, a_f)
    if R_ms(M2, f) > R2_L:
        return M1_core, M2, a_f, E_grav, E_int, E_bind, 8
    return M1_core, M2, a_f, E_grav, E_int, E_bind, 9
