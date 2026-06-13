# TODO

## Highest Priority

- [ ] Replace per-slice angular FFT launches with batched cuFFT.
  - Current code loops over `Ng * nyz` slices and launches multiple kernels/cuFFT calls per `11x11` slice.
  - Use `cufftPlanMany` to process all angle slices in a batch.
  - Expected impact: reduce millions of launches in small runs to a small number per solver step.

- [ ] Fix data dependency in `stepPrimary`.
  - Current code launches angular diffusion and spatial transport on `d_F` in different streams.
  - Both read/write `d_F`, so this is not a safe overlap.
  - Use sequential execution or double-buffered input/output arrays.

- [ ] Move dose reduction to GPU.
  - Current `computeDose()` copies the full distribution back to host every step.
  - Implement GPU reduction and copy back one scalar, or output depth-local dose arrays.

## Paper-Correct Output

- [ ] Implement local energy deposition according to the paper's dose formula.
  - Paper dose: `D(r) = Edep(r) / rho(r)`.
  - Primary deposition follows paper equation `(24)`.
  - Add secondary proton deposition as a corresponding term.

- [ ] Output `D(x,y,z)` or at least `IDD(x)`.
  - `IDD(x) = integral integral D(x,y,z) dy dz`.
  - Current `dose_output.txt` is only a scalar proxy and is not directly comparable with the paper.

- [ ] Add output files for paper-style comparisons.
  - `idd_output.txt`: columns `x IDD`.
  - `spot_x_<depth>.txt` or binary output for `YZ` dose planes.
  - `longitudinal_output` for integrated longitudinal data.

- [ ] Update plotting scripts to generate paper-relevant plots.
  - IDD curve.
  - Bragg peak, P90, D90, D20 markers.
  - Spot distribution heatmaps.
  - Optional comparison against FLUKA/reference data.

## Numerical Model Completeness

- [ ] Replace simplified energy attenuation with paper-level energy discretization.
  - Current `energyAttenuationKernel` uses `exp(-sigma * dt)`.
  - Implement the DG/CN energy subsystem from the paper.
  - Decide whether PCR, Thomas, cuSPARSE, or a custom batched tridiagonal solver is best.

- [ ] Complete collision integral and source term.
  - Current collision integral scaffolding is incomplete.
  - Implement catastrophic scattering source for secondary protons.
  - Load or generate realistic `sigma_c,s`, `sigma_c,t`, and transition kernels.

- [ ] Replace synthetic cross sections with paper/FLUKA-derived data.
  - Current `scripts/generate_test_data.py` generates approximate synthetic data.
  - Add data format documentation and validation.
  - Support water, bone, air, and heterogeneous material maps.

- [ ] Implement paper-level spatial transport.
  - Current transport is simplified first-order upwind.
  - Paper uses a second-order finite-volume/MUSCL-style discretization.
  - Add limiter logic and verify against convergence tests.

- [ ] Revisit angular discretization.
  - Current grid is `11x11`; paper tests use `20x20`.
  - Check boundary conditions and whether sine-transform semantics are required.
  - If keeping FFT, document the approximation clearly.

## Validation

- [ ] Add smoke tests.
  - Data generation works.
  - Build succeeds.
  - `--test` should avoid requiring a full solve but still verify CUDA availability when requested.

- [ ] Add numerical regression tests.
  - Small grid deterministic output.
  - Conservation/positivity checks.
  - No NaN/Inf checks after each subsystem.

- [ ] Reproduce paper convergence tests.
  - Depth refinement.
  - Energy refinement.
  - Compare with paper Figure 1/Figure 2 style diagnostics.

- [ ] Reproduce paper IDD benchmarks.
  - Water, bone, air.
  - Energies `50 MeV`, `100 MeV`, `230 MeV`.
  - Compare BP, P90, D90, D20 against paper Table 2.

## Performance

- [ ] Add NVTX ranges around solver phases.
  - `initialize`
  - `stepPrimary.angle`
  - `stepPrimary.transport`
  - `stepPrimary.energy`
  - `collisionSource`
  - `stepSecondary`
  - `dose`

- [ ] Add a profiling README section for standard workloads.
  - 1-step smoke profile.
  - 10-step small profile.
  - full-grid profile.

- [ ] Reduce launch overhead in angular diffusion.
  - Batched real/complex conversion.
  - Batched forward/inverse FFT.
  - Batched frequency multiplication.
  - Consider correct batched `D2Z/Z2D` half-spectrum implementation for performance.

- [ ] Optimize `spatialTransportKernel` after algorithmic fixes.
  - Reduce integer division/modulo if possible.
  - Consider layout changes for coalescing.
  - Review FP64 usage and whether FP32/mixed precision is acceptable.
  - Use double buffering to avoid in-place hazards.

- [ ] Optimize energy step after replacing the simplified model.
  - Use batched tridiagonal solve.
  - Keep energy-major vs angle-major layout under review.
  - Avoid repeated host-device transfers.

## Code Quality

- [ ] Add input validation for data files.
  - Check existence of `data_cross.txt` and `cross_total.txt`.
  - Check row counts match `Ng`.
  - Fail with clear error messages.

- [ ] Separate runtime outputs from source tree.
  - Put reports in `reports/`.
  - Put generated plots in `plots/`.
  - Put solver outputs in `outputs/`.

- [ ] Add `.gitignore`.
  - `build/`
  - `*.nsys-rep`
  - `*.ncu-rep`
  - `*.sqlite`
  - generated plots and `dose_output.txt`
  - `.DS_Store`

- [ ] Document data formats.
  - Cross section file columns.
  - Units.
  - Expected grid dimensions.
  - Material-dependent parameters.

- [ ] Clean up CMake.
  - Use target-specific include directories.
  - Use target-specific compile options instead of global `CMAKE_CUDA_FLAGS`.
  - Make CUDA architecture explicit in documentation.

