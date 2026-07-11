# TODO

## Highest Priority

- [ ] Optimize angular sine-series/DST implementation.
  - Current Eq. `(21)` path uses direct global-memory sine transforms for
    `Nmu,Nom <= 8`, separable real DST kernels for `Nmu,Nom <= 128`, and an
    odd-extension FFT fallback above that range.
  - Further optimize the separable DST path and revisit a library-backed batched
    DST/FFT path for very large angular grids.
  - Keep the zero-boundary Eq. `(21)` semantics; do not return to periodic FFT.

- [x] Fix data dependency in `stepPrimary`.
  - Energy, spatial transport, and angular diffusion now execute sequentially on
    `stream_primary[0]`, with synchronization between dependent operators.

- [x] Move dose reduction to GPU.
  - IDD, scalar proxy, and energy-flux block sums are recursively reduced on the
    GPU; only one double is copied to the host.

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
  - Current transport uses the Eq. `(22)` limiter and an explicit predictor-corrector
    depth update.
  - Still needs formal convergence tests against the paper diagnostics.

- [x] Replace periodic angular FFT semantics with zero-boundary sine-series semantics.
  - Current angular diffusion uses the Eq. `(20)`-`(21)` CN eigenvalues and
    DST-I/sine basis. Small grids use direct sine sums, medium grids use
    separable DST kernels, and larger grids fall back to odd-extension FFT.
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
  - Current `energyDgCnPrecomputeKernel` and `energyDgCnSolveFromPrecomputeKernel`
    implement the Eq. `(15)` DG/CN subsystem as the algebra documented in
    `results/transport_formula/transport_formula.pdf` equations `(11)`-`(21)`.
  - The implemented new-value system is a 2x2 block upper-bidiagonal energy
    system solved by high-to-low energy backward substitution, not a stored
    scalar tridiagonal matrix.
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
  - Current transport uses the Eq. `(22)` MUSCL limiter with an explicit
    predictor-corrector update.
  - Verify against convergence tests and Figure 1/Figure 2 style diagnostics.

- [ ] Validate angular discretization at larger grids.
  - Validate the direct sine, separable DST, and odd-extension FFT branches
    against each other.
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

### Next Optimization Priorities

- [x] Add a cross-chunk double-buffer pipeline.
  - Overlap `pwrite(chunk N)` with `pread`, H2D, and GPU compute for `chunk N+1`.
  - Use two pinned host-buffer pairs and preserve 17-digit IDD validation within
    double-precision rounding tolerance.
  - Measure end-to-end time, not only overlapped phase time.
  - Retained implementation overlaps chunk writeback with the next chunk's read
    and GPU work using two pinned host-buffer pairs.

- [x] Reduce complete backing-store passes between split operators.
  - Investigate a blocked layout shared by the energy and transport/angle paths.
  - Do not repeat the reverted naive mmap, full-lane, or pure on-disk reorder attempts.
  - Retain a redesign only if it reduces transferred bytes or complete file scans.
  - The retained change removes `source.bin`: Scheme A recomputes the secondary
    source for the second half-step, reducing backing files from five to four.
  - A shared blocked layout for the four state files remains a separate future
    investigation.

- [x] Replace profiling-path host synchronization with CUDA events.
  - Keep `--profile-steps` phase timings without serializing the chunk pipeline.
  - Add separate storage I/O, H2D/D2H, kernel, and overlap measurements.
  - Energy, transport, and angle kernel intervals now use CUDA events and avoid
    intermediate host barriers under `--profile-steps`.

- [x] Optimize MUSCL neighborhood loading.
  - Evaluate shared-memory spatial tiles or angle-vectorized loads.
  - Do not repeat the reverted `32 angles x 8 cells` block mapping unchanged.
  - Preserve the Eq. `(22)` limiter and high-precision full-vs-streaming checks.
  - Neighbor indices and plane bases are computed once per output point; the
    limiter arithmetic and the validated block geometry are unchanged.

- [x] Move IDD and scalar reductions fully to the GPU.
  - Reduce device block sums to one scalar before copying to the host.
  - Avoid allocating and copying a host block-sum vector on every sampled step.
  - A second reduction scratch buffer supports recursive device reduction to one
    scalar without in-place races.

- [ ] Rework streaming secondary source as GEMM/sparse-SpMM.
  - The pre-Scheme-A sparse streaming source used angular transition lists and
    energy nonzero ranges, but its large-grid profile showed secondary source at
    about `19-22 s` for `Ny=Nz=40, Ng=500, Nmu=Nom=20, time=0.8`.
  - Scheme A is the validated fused tiled secondary-source implementation documented in
    `STREAMING_OPTIMIZATION.md`. It computes a small output-energy tile per lane
    to reuse angular sparse-convolution traversal without writing an intermediate
    source array.
  - Scheme A is byte-identical to full mode and the sparse streaming baseline,
    passes compute-sanitizer, reduces the measured source time from `19.38 s` to
    `2.09 s`, and reduces same-machine total time from `89.44 s` to `71.29 s`.
  - Next high-impact direction is to reformulate source generation as batched
    dense GEMM or sparse SpMM over `(cell, angle)` blocks, rather than one
    thread looping over source energy and angular neighbors.
  - A simple two-stage factorization (`angular convolution` then `energy
    accumulation`) reduced the measured source kernel to about `0.84 s`, but
    increased total time to about `109.8 s` because intermediate writes and
    memory pressure slowed I/O/transport/angle; do not retain that version.
  - Keep bitwise/full-vs-streaming IDD validation before retaining any rewrite.

- [ ] Redesign streaming energy/source I/O layout.
  - The source-cache pass is removed, and state-file writes are double-buffered.
    The remaining bottleneck is the four complete state files and their
    energy-major versus lane-slab access mismatch.
  - Scheme B is the planned layout redesign documented in
    `STREAMING_OPTIMIZATION.md`: dual layout, chunk-local transpose, or
    lane-major scratch paths for the energy/source hot path.
  - Investigate a real layout redesign or chunk-local transpose strategy for
    energy/lane access, not naive mmap or full-lane buffers.
  - Previous attempts to use mmap backing files, full-lane chunks, and simple
    angle-pair factor reuse were measured slower and should not be repeated
    without a different design.
  - A chunk-major source-cache scratch layout reduced source I/O from `5.70 s`
    to `5.22 s`, but regressed total time from `71.29 s` to `75.56 s`; it was
    reverted. A future Scheme B must eliminate a data pass or overlap I/O and
    GPU execution, not only reduce the number of `pread`/`pwrite` calls.
  - Paired `F/f_F` backing-file reads and writes now run concurrently. This
    reduced the primary-only large-grid total from `21.42 s` to `20.51 s` and is
    retained. Cross-chunk double buffering now also overlaps writeback with the
    following read/H2D/compute sequence.
  - A lane-slab tiled physical backing layout was implemented and validated,
    but regressed the adjacent large-grid run from `28.283 s` to `29.290 s`
    (3.6% slower), so it was reverted.

- [x] Fuse the primary operator boundary in streaming mode.
  - The retained path executes transport, angle, transport, and the second
    primary energy half-step while each energy chunk is resident.
  - Four lane carry buffers preserve the old/new high-energy recurrence values
    across chunks, removing one complete primary `F/f_F` backing-store write
    and reread.
  - `--no-fused-stream-boundary` provides a same-binary fallback.
  - Adjacent `30x30x500x20x20` runs improved from `27.252 s` to `24.593 s`
    (9.8%). Bone-100 output is byte-identical; water-230 differs by at most
    `8.67e-19` absolute / `1.82e-16` relative from changed reduction order;
    compute-sanitizer reports zero errors.

- [ ] Fuse the secondary source directly into the energy recurrence.
  - A four-output-energy prototype was numerically identical on the small
    secondary test, but used 128 registers and a 64-byte stack frame.
  - On the `30x30x500x20x20` workload its source kernel took `10.071 s`, versus
    `2.681 s` source plus `0.465 s` recurrence for the retained separate path.
  - The prototype was reverted. Revisit only with a formulation that preserves
    source-kernel parallelism, such as a staged GEMM/SpMM design.

- [x] Skip provably inactive high-energy streaming ranges.
  - Initialization records the highest exactly nonzero primary energy group.
  - Without straggling, the DG recurrence cannot create higher-energy values;
    the catastrophic source kernel is checked to be energy non-increasing.
  - Energy passes only read, transfer, solve, and write the active prefix, while
    transport/angle skips chunks wholly above it.
  - `--no-zero-chunk-skip` retains a same-binary A/B fallback.
  - The `30x30x500x20x20`, two-step water-100 benchmark improved from
    `34.809 s` to `24.631 s` (29.2%) with byte-identical IDD.

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
  - The paper `20x20` grid now uses the separable DST instead of the direct 2D
    transform, and the inverse writes directly to its final output. In the final
    large secondary benchmark, angle time fell from `9.20 s` to `0.55 s`, and
    direct-vs-separable IDD remained byte-identical in the validation case.

- [ ] Optimize spatial transport kernels after algorithmic fixes.
  - Reduce integer division/modulo if possible.
  - Consider layout changes for coalescing.
  - Review FP64 usage and whether FP32/mixed precision is acceptable.
  - Use double buffering to avoid in-place hazards.
  - The MUSCL corrector now writes in place after both flux arrays are complete,
    removing one full-array D2D copy and one temporary device array per solver.

- [ ] Optimize energy step.
  - Optimize the current coefficient precompute plus high-to-low energy
    backward substitution.
  - Revisit a batched tridiagonal solve only if the straggling term is changed
    to a fully implicit new-value coupling.
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
