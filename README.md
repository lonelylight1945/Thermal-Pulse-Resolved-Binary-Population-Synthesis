# Thermal-Pulse-Resolved Binary Population Synthesis

This project evolves binary populations along detailed, precomputed MESA donor tracks. The tracks retain the thermal-pulse structure of the TP-AGB phase. The main application is the formation of white-dwarf + main-sequence post-common-envelope binaries (PCEBs), including long-period systems.

## Citation

If you use this code or the accompanying stellar evolution grids in your research, please cite:

Shi, Y., Ge, H., Belloni, D., Jiang, D., Chen, H., & Han, Z. (2026). [Thermal-pulse-resolved Binary Population Synthesis with Detailed Stellar Models: Implications for Long-period WD+MS Formation](https://arxiv.org/abs/2611.13502). arXiv:2611.13502.

```bibtex
@misc{shi2026thermal,
  author        = {Shi, Yanxu and Ge, Hongwei and Belloni, Diogo and Jiang, Dengkai and Chen, Hailiang and Han, Zhanwen},
  title         = {Thermal-pulse-resolved Binary Population Synthesis with Detailed Stellar Models: Implications for Long-period {WD+MS} Formation},
  year          = {2026},
  eprint        = {2611.13502},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.SR},
  url           = {https://arxiv.org/abs/2611.13502}
}
```

## Contents

- `grids_data/`: preprocessed MESA donor tracks (`1.0.npz`–`6.0.npz`, in steps of 0.1 solar masses).
- `S2_ms_grids/`: main-sequence companion radius tracks.
- `sample.py`: initial primary mass, mass ratio, orbital separation, and birth-time sampling.
- `moe_distribution.py`: cached correlated orbital distributions adapted from the supplied Moe & Di Stefano sampler.
- `main_nearest.py`: population run using the closest donor-mass grid point. It samples the companion mass *after* matching the primary to the grid.
- `main.py`: alternative population run evolving the two neighboring donor tracks and interpolating their final states when possible.
- `evolve.py`, `pre_MT_step.py`, `companion.py`, `rlf.py`, `CEE.py`: pre-contact evolution and common-envelope calculations.
- `single binary evo/`: trace one chosen binary and obtain a time series of its pre-CE evolution.

## Run

Use Python 3.10 or newer and install NumPy.

```bash
python -m pip install numpy
```

From the archive root, enter the BPS directory and run the desired driver.

```bash
cd "TP-AGB BPS/"
python main_nearest.py
python main.py
```

The population size, seed, galaxy age, and worker count are constants near the top of each main script. Each worker loads the local donor and companion grids once. For a short check, reduce `n_binaries` and `N_WORKERS`. The bundled `main_nearest.py` simulates five million initial binaries and writes `pcebs_nearest.txt`. The bundled `main.py` simulates one million and writes `pcebs.txt`. Both write in the working directory and overwrite their corresponding result table on each run.

Both drivers accept `INITIAL_DISTRIBUTION = "classic"` or `"moe"`. The bundled defaults are `"moe"` in `main_nearest.py` and `"classic"` in `main.py`.
Both use the existing primary mass law restricted to 1–6 solar masses and a uniform
birth time in `[0, GALAXY_AGE)`. The nearest-track mode matches M1 before drawing
the companion. The classic option retains the existing flat mass-ratio and
separation laws. The Moe option implements the Moe & Di Stefano (2017) prescription in `moe_distribution.py`
for correlated period, mass ratio, and eccentricity distributions, including its
approximate removal of outer tertiary companions. In this option the existing
mass law is treated as the distribution of all primaries (single stars and
multiple-system primaries). Each candidate M1 is accepted with the source's
mass-dependent binary fraction F_bin(M1), giving
`p(M1 | binary) proportional to p_primary(M1) * F_bin(M1)`.
The fraction is the integral of companion frequency times inner-binary probability
over log period, as in the supplied script. Rejected single-star trials are redrawn;
`n_binaries` remains the number of accepted binaries, not the total number of
single and binary systems. In nearest-track mode the acceptance probability is
evaluated at the matched donor mass. Orbital CDFs are conditional on being a binary.
The original `q >= 0.1` binary fraction and period distribution are retained.
The conditional mass-ratio distribution is extended to `q = 0.01` with the
same `gs` power law, continuous at `q = 0.1`, and then truncated and renormalized
at `q_min = max(0.01, 0.1 / M1)` so that `M2 >= 0.1` solar masses. In nearest-track
mode this lower bound uses the matched primary mass. Analytic integration and
inverse sampling handle the low-q interval, including `gs = -1`; no clipping to
the lower mass bound is used. This fixed-binary-fraction extension redistributes
some original high-q binaries into the low-q interval; it does not add binaries
or preserve the original absolute companion frequency at `q >= 0.1`.
The original eccentric-orbit period range is `0.15 <= log10(P/day) <= 8`.
The sampled eccentric orbit is immediately
circularized at fixed orbital angular momentum and masses using
`a_circ = a * (1 - e**2)`. The existing output `a_i` is this circularized separation;
the original eccentricity and period are not output. Subsequent evolution is circular.
This implementation combines the correlated orbital distributions with the BPS primary mass law.
Distribution tables are built once per worker using only NumPy. The random stream
remains determined by `(SEED, idx)`, independently of worker scheduling. Output
filenames are unchanged, so save each run separately when comparing distributions.

For an individual system, see the [single-binary notebook instructions](single%20binary%20evo/README.md).

## Model and output conventions

The binary is initially circular. Before Roche-lobe overflow, the model includes donor wind loss, companion wind accretion, donor spin, and donor tides. Common-envelope evolution uses the supplied MESA envelope gravitational and internal energies with default `alpha_ce = 0.3` and `alpha_th = 0.5`. The current population scripts retain only event `type = 9`, meaning that the binary passes the post-CE companion Roche-lobe check. The output orbital period is the *immediate post-CE* period in days; it does not include subsequent binary evolution or observational selection. The recorded remnant mass is the donor's helium-core mass at CE onset.

The output is tab-separated.

| Column | Meaning and units |
| --- | --- |
| `idx` | Zero-based index in the simulated binary population. |
| `t_born` | Birth time in yr from the start of the modeled Galactic history. |
| `type` | Evolution event code (9 for retained systems). |
| `M1_i`, `M2_i` | Initial donor and companion masses in solar masses. |
| `M1_f`, `M2_f` | Immediate post-CE remnant and companion masses in solar masses. |
| `a_i`, `a_f` | Initial circular and immediate post-CE separations in solar radii. |
| `P_orbf_day` | Immediate post-CE orbital period in days. |

