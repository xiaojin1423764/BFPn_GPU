# TODO

## Highest Priority

- [ ] Optimize angular sine-series/DST implementation.
  - Current strict Eq. `(21)` path uses direct global-memory sine transforms for
    `Nmu,Nom <= 32`.
  - Replace with a faster separable matrix-multiply implementation or a verified
    batched odd-extension FFT/DST path before scaling to paper `20x20` grids.
  - Keep the zero-boundary Eq. `(21)` semantics; do not return to periodic FFT.

- [ ] Fix data dependency in `stepPrimary`.
  - Current code launches angular diffusion and spatial transport on `d_F` in different streams.
  - Both read/write `d_F`, so this is not a safe overlap.
  - Use sequential execution or double-buffered input/output arrays.

- [ ] Move dose reduction to GPU.
  - Current IDD and scalar proxy reductions still copy block sums back to host every step.
  - Implement GPU reduction and copy back one scalar, or output depth-local dose arrays.

## Paper-Correct Output

- [ ] Implement local energy deposition according to the paper's dose formula.
  - Paper dose: `D(r) = Edep(r) / rho(r)`.
  - Primary deposition follows paper equation `(24)`.
  - Add secondary proton deposition as a corresponding term.

- [x] Output at least `IDD(x)`.
  - `IDD(x) = integral integral D(x,y,z) dy dz`.
  - Current `idd_output.txt` is the Figure 3-facing output.
  - Current `dose_output.txt` is retained only as a scalar diagnostic proxy.

- [ ] Add remaining output files for paper-style comparisons.
  - `idd_output.txt`: columns `x IDD`.
  - `spot_x_<depth>.txt` or binary output for `YZ` dose planes.
  - `longitudinal_output` for integrated longitudinal data.

- [x] Add paper-relevant IDD analysis scripts.
  - IDD curve plotting uses `scripts/plot_results.py`.
  - Bragg peak, P90, D90, D20 metrics use `scripts/analyze_idd.py`.
  - Table 2 batch runs use `scripts/run_table2.py`.
  - Spot distribution heatmaps.
  - Optional comparison against FLUKA/reference data.

## Paper Reproduction Roadmap

The current solver is a prototype, not a paper-equivalent implementation. The main mismatch is
not plotting or reduction performance; it is that the current energy step and dose output are not
the model used in the paper.

- [x] Build the first paper-style IDD output path.
  - Add depth-local buffers for primary and secondary energy deposition, e.g. `Edep_p(y,z)`,
    `Edep_s(y,z)`, and `D(y,z)`.
  - Output `idd_output.txt` with columns `x IDD`.
  - Optionally output `spot_x_<depth>.txt` for selected `YZ` dose planes.
  - Keep `dose_output.txt` only as a smoke-test scalar until it is removed or renamed.

- [x] Replace the scalar energy-group state with second-order DG energy coefficients.
  - The full path carries `F/f_F` and `F1/f_F1`, corresponding to
    `psi_g^1/psi_g^2` for primary and secondary protons.

- [x] Complete the paper DG/CN energy subsystem before expecting a validated Bragg peak.
  - Current energy update uses the Eq. `(15)` DG/CN path over `F` and `f_F`.
  - Paper equation `(15)` includes stopping power `S(E)`, straggling `T(E)`, catastrophic loss
    `sigma_c,t`, DG interface fluxes, and CN updates.
  - Fine-grid water/bone 100/230 MeV IDD positions are within the requested
    1-2% Table 2 target range.

- [x] Change `stepPrimary()` to the paper's second-order Strang depth splitting.
  - Target order: energy half step, spatial transport half step, angular diffusion full step,
    spatial transport half step, energy half step.
  - This corresponds to the paper's `L3/L4`, `L1`, `L2`, `L1`, `L3/L4` sequence.
  - Use the same structure for secondary protons once the source term is available.

- [x] Add MUSCL-style spatial transport.
  - Current transport uses the Eq. `(22)` limiter and a predictor-corrector CN
    depth update.
  - Still needs formal convergence tests against the paper diagnostics.

- [x] Replace periodic angular FFT semantics with zero-boundary sine-series semantics.
  - Current angular diffusion uses the Eq. `(20)`-`(21)` CN eigenvalues and
    DST-I/sine basis for `Nmu,Nom <= 32`.
  - Paper benchmark settings use `20x20` angular grid; current default is `11x11`.
  - Next work is performance and convergence validation, not formula replacement.

- [x] Complete catastrophic-scattering secondary source in full and streaming modes.
  - Full mode and streaming mode use the energy kernels `ker_e1/ker_e2` and
    angular transition kernel `ker_v`.
  - Paper uses `sigma_c,s` built from `sigma_c,t * P(g' -> g) * P_angle((u',v') -> (u,v))`.
  - Streaming mode is formula-aligned but slow because it evaluates the source
    through host-backed chunks.

- [ ] Validate in stages.
  - Stage 1: primary-only IDD with DG/CN energy and current transport/angular approximations.
  - Stage 2: add MUSCL transport and compare convergence in depth and energy.
  - Stage 3: add secondary source and compare water/bone/air IDD.
  - Stage 4: reproduce BP, P90, D90, and D20 positions from paper Table 2.

## Numerical Model Completeness

- [x] Replace simplified energy attenuation with paper-level energy discretization.
  - Current `energyDgCnKernel` implements the Eq. `(15)` DG/CN subsystem.
  - Further work is convergence testing and performance tuning.

- [x] Complete collision integral and source term for Figure 3 IDD validation.
  - Catastrophic scattering source for secondary protons is implemented with
    energy and angle transition kernels.
  - Load or generate realistic `sigma_c,s`, `sigma_c,t`, and transition kernels.

- [ ] Replace synthetic cross sections with paper/FLUKA-derived data.
  - Current `scripts/generate_test_data.py` generates approximate synthetic data.
  - Add data format documentation and validation.
  - Support water, bone, air, and heterogeneous material maps.

- [ ] Validate paper-level spatial transport.
  - Current transport uses the Eq. `(22)` MUSCL limiter with a CN
    predictor-corrector update.
  - Verify against convergence tests and Figure 1/Figure 2 style diagnostics.

- [ ] Validate angular discretization at larger grids.
  - Current strict sine-series path is used for `Nmu,Nom <= 32`; paper tests use
    `20x20`.
  - Add convergence tests and optimize performance for larger angular grids.

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
  - Water and bone at `100 MeV` and `230 MeV` are within the requested 1-2%
    Table 2 range on the strict fine grid.
  - Remaining: water/bone `50 MeV`, air cases, and larger paper-grid convergence.

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

- [ ] Reduce overhead in angular diffusion.
  - Optimize the direct sine-series kernels or replace them with a verified
    batched odd-extension FFT/DST.
  - Keep Eq. `(21)` zero-boundary semantics.

- [ ] Optimize `spatialTransportKernel` after algorithmic fixes.
  - Reduce integer division/modulo if possible.
  - Consider layout changes for coalescing.
  - Review FP64 usage and whether FP32/mixed precision is acceptable.
  - Use double buffering to avoid in-place hazards.

- [ ] Optimize energy step.
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
