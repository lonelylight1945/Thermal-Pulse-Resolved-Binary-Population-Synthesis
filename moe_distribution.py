"""Moe & Di Stefano correlated orbital distributions (arXiv:1606.05347).

Adapted from Maxwell Moe's IDL prescription and the 2017 Python translation
by Mads Sorensen supplied as gen_Moes_data.py. Retains its tabulation and
nearest-table-point sampling. Only binaries (including inner binaries in
triples) are sampled; the caller weights its primary-mass distribution by
the tabulated binary fraction before drawing orbital parameters.
Mass ratios are extended below 0.1 at fixed original binary fraction and
period distribution, redistributing existing binaries toward smaller q.
"""

from functools import lru_cache

import numpy as np


def _integration_weights(size, step):
    """Uniform-grid equivalent of the source's five-point idl_tabulate."""
    rules = {
        2: np.array([1, 1]) / 2,
        3: np.array([1, 4, 1]) / 3,
        4: np.array([1, 3, 3, 1]) * 3 / 8,
        5: np.array([7, 32, 12, 32, 7]) * 2 / 45,
    }
    weights = np.zeros(size)
    for start in range(0, size - 1, 4):
        count = min(5, size - start)
        weights[start:start + count] += step * rules[count]
    return weights


def _low_q_integral(lower, upper, slope):
    """Integral of (q / 0.1)**slope, including the slope = -1 limit."""
    power = slope + 1.0
    logarithmic = np.abs(power) < 1.0e-8
    log_ratio = np.log(np.asarray(upper) / lower)
    safe_power = np.where(logarithmic, 1.0, power)
    integral = (lower / 0.1)**power * np.expm1(power * log_ratio) / safe_power
    return 0.1 * np.where(logarithmic, log_ratio, integral)


@lru_cache(maxsize=1)
def _tables():
    """Build once per process; no random draws or file output."""
    masses = np.geomspace(0.8, 40.0, 101)
    logp = np.arange(158) * 0.05 + 0.15  # log10(P/day)
    q = np.linspace(0.1, 1.0, 91)
    q_low = np.arange(1, 10) * 0.01
    e = np.linspace(0.0, 0.99, 100) + 0.0001
    wq = _integration_weights(len(q), 0.01)
    large = q >= 0.3
    wlarge = _integration_weights(np.count_nonzero(large), 0.01)
    twins = (q >= 0.95).astype(float)
    twins /= wq @ twins

    period_cdf = np.empty((len(masses), len(logp)))
    binary_fractions = np.empty(len(masses))
    wp = _integration_weights(len(logp), 0.05)
    q_cdf = np.empty((len(masses), len(q_low) + len(q), len(logp)))
    small_q_slopes = np.empty((len(masses), len(logp)))
    e_cdf = np.empty((len(masses), len(e), len(logp)))

    for i, mass in enumerate(masses):
        logm = np.log10(mass)
        ptwin = 8.0 - mass if mass < 6.5 else 1.5
        ftwin = (0.3 - 0.15 * logm) * np.clip((ptwin - logp) / (ptwin - 1), 0, 1)

        gl_low = -0.5 - 0.3 * np.maximum(logp - 5.0, 0.0)
        gl_mid = np.interp(logp, [1.0, 4.5, 6.5], [-0.5, -1.2, -2.0])
        gl_high = np.interp(logp, [1.0, 2.0, 4.0], [-0.5, -1.4, -2.0])
        gs_low = np.full_like(logp, 0.3)
        gs_mid = np.interp(logp, [2.5, 5.5, 8.0], [0.2, -0.7, -1.2])
        gs_high = np.interp(logp, [1.0, 3.0, 5.6], [0.1, -0.2, -1.5])
        if mass <= 3.5:
            weight = np.clip(np.log(mass / 1.2) / np.log(3.5 / 1.2), 0, 1)
            gl = (1 - weight) * gl_low + weight * gl_mid
            gs = (1 - weight) * gs_low + weight * gs_mid
        else:
            weight = np.clip(np.log(mass / 3.5) / np.log(6.0 / 3.5), 0, 1)
            gl = (1 - weight) * gl_mid + weight * gl_high
            gs = (1 - weight) * gs_mid + weight * gs_high

        fq = q[:, None] ** gl
        fq /= wlarge @ fq[large]
        fq = fq * (1 - ftwin) + twins[:, None] * ftwin
        fq[~large] = fq[large][0] * (q[~large, None] / 0.3) ** gs
        cq = np.cumsum(fq, axis=0) - fq[0]
        original_norm = wq @ fq
        low_norm = fq[0] * _low_q_integral(0.01, 0.1, gs)
        low_cumulative = fq[0] * _low_q_integral(0.01, q_low[:, None], gs)
        q_cdf[i] = np.vstack((
            low_cumulative, low_norm + original_norm * (cq / cq[-1])
        )) / (low_norm + original_norm)
        small_q_slopes[i] = gs

        f1 = 0.020 + 0.04 * logm + 0.07 * logm**2
        f27 = 0.039 + 0.07 * logm + 0.01 * logm**2
        f55 = 0.078 - 0.05 * logm + 0.04 * logm**2
        fp = np.interp(logp, [1.0, 2.0, 3.4, 5.5],
                       [f1, f27 - 0.018 * 0.7, f27 + 0.018 * 0.7, f55])
        tail = logp > 5.5
        fp[tail] = f55 * np.exp(-0.3 * (logp[tail] - 5.5))
        prob_inner = np.maximum(
            0.0, 1 - 0.11 * np.maximum(logp - 1.5, 0.0)**1.43 * (mass / 10)**0.56
        )
        # Keep the q >= 0.1 calibration for the original period law and F_bin.
        fp *= original_norm * prob_inner
        binary_fractions[i] = wp @ fp
        cp = np.cumsum(fp) - fp[0]
        # Conditional orbital CDF; binary_fractions supplies the mass weighting.
        period_cdf[i] = cp / cp[-1]

        eta_low = 0.6 - 0.7 / (np.maximum(logp, 0.7) - 0.5)
        eta_high = 0.9 - 0.2 / (np.maximum(logp, 0.7) - 0.5)
        weight = np.clip(np.log(mass / 3.0) / np.log(7.0 / 3.0), 0, 1)
        eta = (1 - weight) * eta_low + weight * eta_high
        close = 10**logp <= 2.0
        emax = 1 - (10**logp / 2.0)**(-2.0 / 3.0)
        emax[close] = 1.0  # Close-period columns are replaced below.
        fe = e[:, None] ** eta
        idx = np.searchsorted(e, 0.8 * emax) - 1
        f_turn = fe[idx, np.arange(len(logp))]
        turnover = e[:, None] >= 0.8 * emax
        fe = np.where(turnover, f_turn * (emax - e[:, None]) / (0.2 * emax), fe)
        fe[e[:, None] >= emax] = 0.0
        fe[:, close] = e[:, None] ** -3.2
        ce = np.cumsum(fe, axis=0) - fe[0]
        e_cdf[i] = ce / ce[-1]

    return (masses, logp, np.concatenate((q_low, q)), e, period_cdf,
            q_cdf, e_cdf, small_q_slopes, binary_fractions)


def binary_fraction(M1):
    """Source's binary fraction, using the same mass table as the orbit draw."""
    masses, *_, fractions = _tables()
    return fractions[np.abs(masses - M1).argmin()]


def sample_orbit(rng, M1):
    """Return M2 (Msun) and circularized separation (Rsun) for a given M1."""
    masses, logp, q, e, period_cdf, q_cdf, e_cdf, slopes, _ = _tables()
    i = np.abs(masses - M1).argmin()
    p = np.interp(rng.random(), period_cdf[i], logp)
    j = np.abs(logp - p).argmin()
    ecc = np.interp(rng.random(), e_cdf[i, :, j], e)
    cdf = q_cdf[i, :, j]
    low_fraction = cdf[np.searchsorted(q, 0.1)]
    q_min = max(0.01, 0.1 / M1)
    slope = slopes[i, j]
    if q_min < 0.1:
        cdf_min = low_fraction * (
            _low_q_integral(0.01, q_min, slope) / _low_q_integral(0.01, 0.1, slope)
        )
    else:
        cdf_min = np.interp(q_min, q, cdf)
    u = cdf_min + (1.0 - cdf_min) * rng.random()
    if u < low_fraction:
        power = slope + 1.0
        fraction = u / low_fraction
        if abs(power) < 1.0e-8:
            mass_ratio = 0.01 * 10.0**fraction
        else:
            mass_ratio = 0.01 * np.exp(np.log1p(fraction * np.expm1(power * np.log(10.0))) / power)
    else:
        mass_ratio = np.interp(u, cdf, q)
    M2 = M1 * mass_ratio
    a = 214.95 * ((10**p / 365.24)**2 * (M1 + M2))**(1.0 / 3.0)
    return M2, a * (1 - ecc**2)
