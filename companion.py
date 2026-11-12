from pathlib import Path
import numpy as np

_MS_GRID = {}
_MS_MASSES = None
_TAU_MS_C0 = 1.21162618e+01
_TAU_MS_C1 = -3.16439614e+00
_TAU_MS_C2 = 1.27791734e+00
_TAU_MS_C3 = -2.96499197e-01
_TAU_MS_C4 = 3.50925661e-02
_TAU_MS_C5 = -1.64054461e-03

def initialize_ms_grid(library_dir):
    """
    在 BPS 开始时调用一次，把所有主序 track 读入内存。
    """
    global _MS_GRID, _MS_MASSES

    library_dir = Path(library_dir)
    grid = {}

    for file in library_dir.glob("*.npz"):
        mass = float(file.stem)

        with np.load(file) as data:
            f = np.asarray(data["f"], dtype=float).copy()
            R = np.asarray(data["R"], dtype=float).copy()

        grid[mass] = {
            "f": f,
            # 保存 log(R)，后面在 log(R) 上插值
            "log_R": np.log(R),
        }

    _MS_GRID = grid
    _MS_MASSES = np.array(sorted(grid), dtype=float)

def _log_radius_on_track(mass, f):
    """
    在某一条固定质量 track 上，按主序相位插值 log(R)。
    """
    track = _MS_GRID[mass]

    return float(np.interp(f, track["f"], track["log_R"]))

def R_ms(M, f):
    """
    根据当前质量 M 和主序相位 f 计算主序星半径。

    Parameters
    ----------
    M : float
        当前恒星质量，M_sun。
    f : float
        主序相位，ZAMS=0，TAMS=1。

    Returns
    -------
    float
        半径，R_sun。
    """
    # 0.1<M<0.9 M_sun的不演化，直接用拟合公式返回ZAMS时的半径
    if M < 0.9:
        R = 0.0292087 + 0.86294848*M
        return R

    if M > 6.0:
        M = 6.0

    index = np.searchsorted(_MS_MASSES, M)

    # 恰好落在某个质量网格点
    if (
        index < len(_MS_MASSES)
        and np.isclose(M, _MS_MASSES[index])
    ):
        log_R = _log_radius_on_track(
            _MS_MASSES[index],
            f,
        )
        return float(np.exp(log_R))

    # M 位于左右两个质量网格之间
    M_left = _MS_MASSES[index - 1]
    M_right = _MS_MASSES[index]

    log_R_left = _log_radius_on_track(M_left, f)
    log_R_right = _log_radius_on_track(M_right, f)

    weight = (
        (M - M_left)
        / (M_right - M_left)
    )

    log_R = (
        (1.0 - weight) * log_R_left
        + weight * log_R_right
    )

    return float(np.exp(log_R))

def tau_MS(M):
    """
    根据当前质量 M 计算主序寿命。
    输入：M，M_sun； 输出：tau_ms，yr。
    """
    if M < 0.9:
        return 5e10

    tau_lg = ((((_TAU_MS_C5 * M + _TAU_MS_C4) * M + _TAU_MS_C3) * M
               + _TAU_MS_C2) * M + _TAU_MS_C1) * M + _TAU_MS_C0

    return 10**tau_lg
