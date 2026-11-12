# Single-binary evolution notebook

This notebook follows one binary on a selected donor track and returns a time series up to the stopping event. It imports the modules in this directory, which provide the time-series version of `evolve()`.

From this directory, launch `jupyter lab main.ipynb`. Replace the two example paths in the notebook with paths to the bundled grids relative to the notebook directory.

```python
initialize_ms_grid("../S2_ms_grids")
history = np.load("../grids_data/2.4.npz")
```

Choose the donor mass through the filename and edit `M2_i` (initial companion mass in solar masses), `a_i` (initial separation in solar radii), and `tau_evo` (maximum evolution time in yr). Run the cells in order.

The result contains the stopping event and the history of age, component masses, donor radius, Roche-lobe filling factor, separation, orbital period, envelope energies, donor moment of inertia, and wind mass-loss rate. The notebook converts age to Myr for inspection. The final extraction and print cells assume a common-envelope event with a `CEE` summary. For another outcome, inspect `result["type"]` and the available history fields before running those cells.
