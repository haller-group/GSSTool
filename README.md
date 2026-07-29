# Generalized Steady State (GSS)

This repository collects MATLAB examples demonstrating Generalized Steady State (GSS) prediction
for nonlinear mechanical systems under aperiodic (earthquake, chirp,
noise, quasi-periodic, chaotic) forcing. Each example computes a unique GSS
compares it against a full-order finite-element modal time integration, and in some cases against Galerkin-ROM and
data-driven autoencoder+LSTM models [1].

**Reference**

R. S. Kaundinya, I. Thiel, B. Kaszás, S. Jain, G. Haller, "Predicting
Generalized Steady States in Aperiodically Forced Mechanical Systems,"
*Journal of Sound and Vibration* (2026).
https://www.sciencedirect.com/science/article/pii/S0022460X26003871

## Installation

Run the `installer.m` file in the main folder to add this package and its
external dependencies (see [External packages](#external-packages) below)
to the MATLAB path.

Each example folder follows the same layout: the main, runnable scripts
sit at the top level, and the helper functions they call live in a
`private/` subfolder (MATLAB automatically resolves functions in
`private/` for scripts in its parent folder, so no additional path setup
is needed beyond `installer.m`).

To reproduce the results/figures for a given example, make sure MATLAB's
**Current Folder** is set to that example's own directory, then run the
file named `*_GSS.m` (the prefix varies by example, as listed below).

```
.
├── vonKarmanBeam_GSS/
├── AxialMovingBeam_GSS/
├── OscillatorChain_GSS/
├── vonKarmanShell_GSS/
└── Partial data-driven GSS/
    ├── Oscillator Chain/
    └── vonKarman beam copy/
```

---

## Equation-based GSS examples

### `vonKarmanBeam_GSS/`
A cantilevered von Karman beam (finite element model) under recorded
earthquake ground-motion forcing.
- **`vonKarmanBeam_AnchorTool.m`** — computes the GSS
  (piecewise-exact and Newmark solves) and compares it against a
  full-order Newmark time integration.
- **`VkBeamCluster.m`** — the same GSS-vs-full-order comparison, run
  as a mesh-refinement sweep in parallel on a
  cluster.
- **`rw_calculation_vonKarmanBeam_AnchorTool.m`** — computes the
  forcing weakness ratio r_w for this example.

### `AxialMovingBeam_GSS/`
An axially moving beam with gyroscopic and nonlinear (viscoelastic)
damping forces, exhibiting 1:3 internal resonance between its first two
bending modes, subject to chirp base excitation.
- **`AxialMovingBeam_chirp_GSS.m`** — computes the GSS under the chirp forcing, compares it to a full-order time integration,
  and renders the beam's spatial deflection w(x,t) as a movie.
- **`rw_calculation_AxialMovingBeam_chirp.m`** — computes r_w for the
  chirp-forced example.

### `OscillatorChain_GSS/`
A chain of coupled oscillators with cubic and quadratic nonlinearities,
under three forcing scenarios.
- **`Osc_chain_periodic.m`** — periodic forcing; sweeps the forcing
  frequency and compares the maximum response amplitude across the
  full-order model, the exact linear solution, and the GSS under
  both periodic and aperiodic solve modes.
- **`OcillatorChain_largerforcing.m`** and
  **`Oscillator_chain_LTSM_compare_GSS.m`** — filtered-noise/recorded
  forcing; compare the GSS (at increasing truncation orders)
  against a full-order time integration and an autoencoder+LSTM model [1] trained on the same data.
- **`rw_calculation_OscillatorChain.m`** — computes r_w for this example.

### `vonKarmanShell_GSS/`
A von Karman shell-based shallow curved panel, under two forcing scenarios.
- **`Plate_QP_GSS.m`** / **`Plate_QP_GSS_Galerkin.m`** — quasi-periodic
  wind pressure fluctuations; the `_Galerkin` variant adds a
  Galerkin-projected ROM comparison (on a truncated modal basis)
  alongside the GSS-vs-full-order comparison.
- **`Plate_chaotic_GSS.m`** / **`Plate_chaotic_GSS_Galerkin.m`** —
  chaotic (Rössler-driven) wind pressure fluctuations, same GSS/full-order
  (and, for the Galerkin variant, Galerkin-ROM) comparison.
- **`rw_calculation_QP.m`** / **`rw_calculation_chaotic.m`** — r_w for
  the quasi-periodic and chaotic cases respectively.

---

## Partial data-driven GSS examples

These two examples combine a **data-driven** SSM identification step
(learned from simulated or recorded trajectories via SSMLearn/fastSSM) with a GSS correction term that accounts for the
external forcing, including comparisons with autoencoder+LSTM models [1].

### `Partial data-driven GSS/Oscillator Chain/`
- **`Oscillator_Chain_SSM_GSS.m`** — learns a 2D data-driven SSM from
  free trajectories of a stochastically-forced oscillator chain, adds a
  GSS correction to account for the forcing, and compares the resulting
  prediction against a full-order time integration and an
  autoencoder+LSTM model. Also renders a comparison movie.

### `Partial data-driven GSS/vonKarman beam copy/`
- **`vonKarman_beam_SSM_GSS.m`** — the same data-driven-SSM-plus-GSS
  workflow applied to a von Karman cantilever beam: builds the GSS
  correction on top of a precomputed data-driven SSM identification,
  constructs a vector Padé approximant of the GSS to extend its
  convergence domain, and compares the result against a full-order run
  and autoencoder+LSTM predictions (6D and 11D latent-dimension models).
- **`rw_calculation_vonKarman_beam.m`** — computes r_w for this example.

---

## Notes

- Every main script assumes the SSMTool-based toolbox classes
  (`DynamicalSystem`, `SSM`, `ImplicitNewmark`, etc.) are on the MATLAB
  path via `run ../../install.m`, relative to each example's own folder.
- A few scripts depend on external data files (recorded forcing,
  autoencoder+LSTM weights/predictions) that ship alongside the `.m`
  files in their respective folders.

## External packages

This package uses the following external open-source packages:
1. Sandia tensor toolbox: https://gitlab.com/tensors/tensor_toolbox
2. Combinator: https://www.mathworks.com/matlabcentral/fileexchange/24325-combinator-combinations-and-permutations
3. YetAnotherFECode: Zenodo, http://doi.org/10.5281/zenodo.4011281

## References

[1] Simpson, T., Dervilis, N., & Chatzi, E. (2021). Machine learning approach to model order reduction of nonlinear systems via autoencoder and LSTM networks. *Journal of Engineering Mechanics*, 147(10), Article 04021061. https://doi.org/10.1061/(ASCE)EM.1943-7889.0001971

## Contact

Please report any issues/bugs to Roshan S. Kaundinya (roshan.kaundinya@mavt.ethz.ch)
