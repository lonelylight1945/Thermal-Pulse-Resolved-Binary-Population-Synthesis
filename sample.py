import numpy as np


def sample(rng, galaxy_age, masses=None, distribution="classic"):
    """Draw M1, M2, a (Rsun), t_born (yr); select classic or Moe orbits."""

    if distribution == "moe":
        from moe_distribution import binary_fraction, sample_orbit

    while True:
        X = rng.random()
        M1 = 0.19 * X / ((1.0 - X) ** 0.75 + 0.032 * (1.0 - X) ** 0.25)
        if not 1.0 <= M1 <= 6.0:
            continue

        if masses is not None:
            M1 = masses[np.abs(masses - M1).argmin()]

        # Reject single-star trials: p(M1 | binary) is proportional to p(M1)*Fbin.
        if distribution == "moe" and rng.random() >= binary_fraction(M1):
            continue
        break

    if distribution == "moe":
        M2, a = sample_orbit(rng, M1)
        return M1, M2, a, rng.uniform(0.0, galaxy_age)

    while True:
        q = rng.random()
        M2 = M1 * q
        if 0.1 <= M2 <= 6.0:
            break

    a0 = 10.0
    a1 = 5.75e6
    m = 1.2
    p_close = 1.0 / (1.0 + m * np.log(a1 / a0))
    X = rng.random()
    if X < p_close:
        a = a0 * (X / p_close) ** (1.0 / m)
    else:
        a = a0 * (a1 / a0) ** ((X - p_close) / (1.0 - p_close))

    t_born = rng.uniform(0.0, galaxy_age)
    return M1, M2, a, t_born
