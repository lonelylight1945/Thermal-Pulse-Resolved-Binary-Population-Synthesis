"""BPS main program using the nearest donor-grid mass, without interpolation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from companion import initialize_ms_grid
from evolve import evolve
from sample import sample


GALAXY_AGE = 13.0e9  # yr
n_binaries = 5_000_000
SEED = 20260803
INITIAL_DISTRIBUTION = "moe"  # "classic" or "moe"
N_WORKERS = 10
BATCH_SIZE = 100
TRACK_KEYS = (
    "age", "star_mass", "he_core_mass", "co_core_mass", "log_R", "log_L",
    "log_abs_mdot", "env_grav_energy", "env_internal_energy", "zams_star_mass",
    "M_env_conv", "R_env_conv", "I_total",
)

_grids = None
_masses = None


def load_grids(grid_dir="grids_data"):
    grids = {}
    for path in sorted(Path(grid_dir).glob("*.npz"), key=lambda item: float(item.stem)):
        with np.load(path) as data:
            grids[float(path.stem)] = {key: data[key] for key in TRACK_KEYS}
    return grids


def show_progress(done, total):
    fraction = done / total
    width = 40
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress: [{bar}] {fraction:6.2%} ({done}/{total})", end="", flush=True)


def initialize_worker(grid_dir, companion_grid_dir):
    global _grids, _masses
    _grids = load_grids(grid_dir)
    _masses = np.array(sorted(_grids))
    initialize_ms_grid(companion_grid_dir)


def evolve_binary(idx):
    rng = np.random.default_rng([SEED, idx])
    M1_i, M2_i, a_i, t_born = sample(
        rng, GALAXY_AGE, _masses, distribution=INITIAL_DISTRIBUTION
    )
    history = _grids[M1_i]
    tau_evo = GALAXY_AGE - t_born

    M1_f, M2_f, a_f, event_type = evolve(history, M2_i, a_i, tau_evo)
    if event_type != 9:
        return None

    P_orbf = np.sqrt((a_f / 214.95) ** 3 / (M1_f + M2_f)) * 365.24
    return (
        f"{idx}\t{t_born:.8e}\t{event_type}\t{M1_i:.8f}\t{M1_f:.8f}\t"
        f"{M2_i:.8f}\t{M2_f:.8f}\t{a_i:.8f}\t{a_f:.8f}\t{P_orbf:.8f}\n"
    )


def evolve_batch(start, end):
    return start, [row for idx in range(start, end) if (row := evolve_binary(idx)) is not None]


def main():
    workers = min(N_WORKERS, n_binaries)
    batches = [
        (start, min(start + BATCH_SIZE, n_binaries))
        for start in range(0, n_binaries, BATCH_SIZE)
    ]
    pending_rows = {}
    next_start = 0
    completed = 0
    last_progress_update = perf_counter()

    with open("pcebs_nearest.txt", "w", encoding="utf-8") as output:
        output.write("idx\tt_born\ttype\tM1_i\tM1_f\tM2_i\tM2_f\ta_i\ta_f\tP_orbf_day\n")
        show_progress(0, n_binaries)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=("grids_data", "S2_ms_grids"),
        ) as executor:
            futures = {
                executor.submit(evolve_batch, start, end): (start, end)
                for start, end in batches
            }

            for future in as_completed(futures):
                start, end = futures[future]
                batch_start, rows = future.result()
                pending_rows[batch_start] = rows
                completed += end - start

                while next_start in pending_rows:
                    output.writelines(pending_rows.pop(next_start))
                    next_start += BATCH_SIZE

                if perf_counter() - last_progress_update >= 1.0:
                    show_progress(completed, n_binaries)
                    last_progress_update = perf_counter()

        show_progress(n_binaries, n_binaries)
        print()


if __name__ == "__main__":
    main()
