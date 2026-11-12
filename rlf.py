import numpy as np

def R_l(
    M_1: float,     # donor mass, M_sun
    M_2: float,     # accretor mass, M_sun
    a: float,       # separation, R_sun
) -> float:
    """
    计算 donor（星1）的 Roche-lobe 半径。
    """
    q = M_1 / M_2
    q13 = np.cbrt(q)    #q^(1/3)
    q23 = q13**2        #q^(2/3)

    # 计算 R_l1 / a
    f_q = 0.49 * q23 / (0.6 * q23 + np.log1p(q13))

    return a * f_q     # R_l1, R_sun