# BFPn GPU Solver

CUDA/CMake prototype for a deterministic proton transport solver inspired by the paper `2504.00340v1.pdf`, "A deterministic solver for the linear Boltzmann model of a single mono-directional proton beam".

## Current Status

This repository builds and runs a CUDA implementation of the paper-facing IDD
path for Eq. `(15)` validation. The full in-memory path now carries the two DG
energy coefficients through the Strang depth splitting sequence and has been
checked against the paper's Table 2 IDD positions on 16 GB GPU friendly grids.

Implemented at prototype level:

- CUDA memory wrappers and solver orchestration.
- Primary and secondary proton arrays.
- Angular diffusion step using the zero-boundary sine-series/DST-I
  discretization from Eq. `(20)`-`(21)` for `Nmu,Nom <= 32`.
- Spatial transport step using the Eq. `(22)` MUSCL limiter with a
  predictor-corrector Crank-Nicolson depth update.
- Eq. `(15)` DG/CN energy step for `psi_g^1` and `psi_g^2`, including stopping
  power, catastrophic loss, and optional straggling terms.
- Catastrophic secondary source in the full in-memory path using the
  `ker_e1/ker_e2` energy kernels and `ker_v` angular kernel.
- Paper-facing integrated depth dose output in `idd_output.txt`.
- BP/P90/D90/D20 analysis script for Table 2 style comparisons.

Not yet complete:

- Formal convergence validation of the Eq. `(22)` spatial transport substep.
- Full paper-grid `80x80x20x20` full-state run on 16 GB GPUs without streaming
  performance work.
- Streaming out-of-core performance tuning for full secondary source parity.
- Air and heterogeneous material validation.

## Requirements

- CMake 3.18+
- C++17 compiler
- NVIDIA CUDA Toolkit
- NVIDIA GPU and compatible driver
- Python 3 with `numpy` and optionally `matplotlib`
- Optional profiling tools: `nsys`, `ncu`

The current CMake default uses `CMAKE_CUDA_ARCHITECTURES=75` unless overridden. For newer GPUs, override it if desired:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
```

## Build and Run

Generate synthetic data:

```bash
python3 scripts/generate_test_data.py -o data
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Or use the top-level Makefile:

```bash
make
make run
```

`make run` builds and runs a 16 GB GPU friendly Figure 3 development case:

```text
water, 230 MeV, Ny=20, Nz=20, Ng=500, Nmu=20, Nom=20, energy-model=legacy
```

Override parameters as needed:

```bash
make run TIME=1.0 MATERIAL=water ENERGY=230 NY=20 NZ=20 ENERGY_MODEL=legacy
make run-lite TIME=1.0
make run-streaming TIME=0.1
make plot
make analyze
```

Run a small smoke test:

```bash
./build/bin/bfp_solver 100 --data data --time 4.0
```

Run the default grid:

```bash
./build/bin/bfp_solver --data data --time 40.0
```

Run a Figure 3 style water case:

```bash
./build/bin/bfp_solver --time 40.0 --material water --energy 230
```

Run a Figure 3 style bone case:

```bash
./build/bin/bfp_solver --time 40.0 --material bone --energy 230
```

Run a 16 GB GPU friendly Figure 3 development case with the paper angular
resolution but reduced transverse resolution:

```bash
./build/bin/bfp_solver --time 40.0 --material water --energy 230 --ny 20 --nz 20 --nmu 20 --nom 20
```

Run the paper transverse/angular grid on a 16 GB GPU in low-memory development
mode:

```bash
./build/bin/bfp_solver --time 40.0 --material water --energy 230 --ny 80 --nz 80 --nmu 20 --nom 20 --lite-memory
```

`--lite-memory` keeps only the primary scalar fluence buffer and computes IDD
from that buffer. It intentionally disables secondary protons, the second DG
coefficient, lateral/angular transport work buffers, and full paper-equivalent
operator splitting. Use it to test the `80x80x20x20` memory footprint and the
primary Bragg-peak energy-loss path; do not treat it as a final Figure 3
reproduction.

On the RTX 5070 Ti test machine, `--time 1.0` with the paper grid and
`--lite-memory` takes about one minute, so a full `--time 40.0` run is expected
to be tens of minutes. Use shorter depths or smaller grids while tuning.

Run the full primary/secondary state in out-of-core streaming mode:

```bash
./build/bin/bfp_solver --time 0.1 --material water --energy 230 --ny 80 --nz 80 --nmu 20 --nom 20 --streaming-full --energy-chunk 4 --lane-chunk 262144 --stream-dir /tmp/bfpn_stream
```

`--streaming-full` keeps the full primary/secondary state in binary backing
files and moves only chunks through the GPU. This avoids the full-mode GPU OOM
and avoids allocating tens of GiB of RAM, but it is currently disk-I/O bound on
the paper grid. Use it first on smaller grids to validate full-state behavior.
Add `--primary-only` to halve the backing files and skip secondary evolution:

```bash
./build/bin/bfp_solver --time 0.1 --material water --energy 230 --ny 80 --nz 80 --nmu 20 --nom 20 --streaming-full --primary-only --energy-chunk 4 --lane-chunk 262144 --stream-dir /tmp/bfpn_stream
```

Energy update selection:

```bash
--energy-model eq15
--energy-model legacy
```

`eq15` is the current implementation of the paper's two-coefficient DG energy
system from equation `(15)`, including `psi_g^1/psi_g^2`, density-scaled
stopping-power terms, catastrophic loss, straggling, and the full-memory
catastrophic secondary source. It is the default for validation work. `legacy`
keeps the previous stable upwind energy approximation and is useful only for
quick comparison runs.

Validated fine-grid development run:

```bash
./build/bin/bfp_solver 2000 --time 40 --material water --energy 100 --ny 8 --nz 8 --ng 500 --nmu 7 --nom 7 --energy-model eq15 --eq15-straggling
./build/bin/bfp_solver 2000 --time 40 --material water --energy 230 --ny 8 --nz 8 --ng 500 --nmu 7 --nom 7 --energy-model eq15 --eq15-straggling
./build/bin/bfp_solver 2000 --time 40 --material bone  --energy 100 --ny 8 --nz 8 --ng 500 --nmu 7 --nom 7 --energy-model eq15 --eq15-straggling
./build/bin/bfp_solver 2000 --time 40 --material bone  --energy 230 --ny 8 --nz 8 --ng 500 --nmu 7 --nom 7 --energy-model eq15 --eq15-straggling
```

The current fine-grid outputs are stored under `results/eq15_strict_fine_*`,
with the combined normalized IDD figure at:

```text
results/eq15_strict_fine_water_bone_100_230.png
```

Fine-grid Table 2 style errors from the current run:

```text
case        BP rel. diff   P90 rel. diff   D90 rel. diff   D20 rel. diff
water 100      -0.529%        -0.234%        -0.613%        -0.916%
water 230      -0.062%        +0.315%        -0.319%        -0.854%
bone  100      +1.048%        +1.275%        +0.907%        +0.631%
bone  230      +1.520%        +1.869%        +1.212%        +0.620%
```

These are relative to the paper reference values used by
`scripts/analyze_idd.py` and stay within the requested 1-2% target range. The
bone cases use the paper Table 1 compositions with Bethe-Bloch compound
stopping power weighted by element mass fractions and `Z/A`.

Another 16 GB friendly option keeps more transverse cells but uses the current
default angular resolution:

```bash
./build/bin/bfp_solver --time 40.0 --material water --energy 230 --ny 40 --nz 40 --nmu 11 --nom 11
```

When `--material water|bone|air` is provided and `--data` is omitted, the solver uses
`BFPn_CPU_Solver/<material>/` cross-section files. Do not use the synthetic `data/`
directory for Figure 3 reproduction; it is only for build and smoke-test runs.

The optional positional argument is `Nx`. With `Lx=40`, the depth step is:

```text
dt = Lx / Nx
```

Examples:

- `Nx=100`, `--time 4.0` runs about 10 steps.
- default `Nx=4000`, `--time 40.0` runs about 4000 steps.

## Output

The solver writes the paper-facing depth-dose output:

```text
idd_output.txt
```

Format:

```text
x_cm IDD
```

The solver also writes the legacy diagnostic scalar:

```text
dose_output.txt
```

`idd_output.txt` is computed from the current primary and secondary DG coefficient buffers using the paper's dose chain `Edep -> D = Edep / rho -> IDD(x)`. It uses the available GPU model state and is the file to plot for Figure 3-style comparisons.

Important: `idd_output.txt` is the correct output path for Figure 3 style IDD
work. The full in-memory Eq. `(15)` path is validated for water/bone 100 and
230 MeV on the fine development grid above. Remaining caveats are mainly grid
fidelity and model coverage: the paper's full transverse/angular grid does not
fit the current full layout on a 16 GB GPU, air/heterogeneous cases still need
validation, and streaming mode is formula-aligned but still disk-I/O bound.

Plot the current IDD history:

```bash
python3 scripts/plot_results.py -i idd_output.txt -o idd_plot.png
```

Analyze Figure 3 / Table 2 metrics:

```bash
python3 scripts/analyze_idd.py -i idd_output.txt --material water --energy 230
```

Run all Table 2 material/energy cases:

```bash
python3 scripts/run_table2.py
```

Use the paper's transverse and angular grid sizes:

```bash
python3 scripts/run_table2.py --paper-grid
```

Important: the full paper grid `--ny 80 --nz 80 --nmu 20 --nom 20` does not fit
on a 16 GB GPU with the current full phase-space layout. For `Ng=500`, one
double phase-space array is about `9.78 GiB`; the full mode needs several full
arrays plus transport, energy, and FFT work buffers, so the preflight estimate
is roughly `120 GiB`. The solver now checks this before allocation and prints a
clear memory estimate instead of failing later with a raw `cudaMalloc`
out-of-memory error. Use `--lite-memory` for a primary-only paper-grid
development run. A full paper-equivalent run still requires a streaming/chunked
implementation.

## Relation to the Paper

The paper compares deterministic depth-splitting results to FLUKA using:

- `IDD(x) = integral integral D(x,y,z) dy dz`
- spot distributions on the `YZ` plane
- longitudinal dose data
- Bragg peak and P90/D90/D20 positions

The current `idd_output.txt` is the intended Figure 3 output path. On the fine
development grid, water/bone 100 and 230 MeV IDD positions are within the 1-2%
target range. The next reproduction milestones are broader grid/material
coverage, spot/longitudinal dose outputs, and performance work for larger
transverse/angular grids.

### Figure 3 Reproduction Formula Flow

To reproduce Figure 3, the implementation needs the following paper formulas in this order:

1. Equation `(23)`: initializes the incoming Gaussian beam at `x = 0`, including the spatial, angular, and energy spreads and the incoming beam energy `E0`.
2. Equations `(9)`, `(10a)`, and `(10b)`: split the total proton distribution into primary protons and secondary/scattering protons. Figure 3 requires primary transport plus the first catastrophic-scattering secondary contribution.
3. Equations `(11)`-`(14)`: define the second-order DG representation in energy, storing two coefficients per energy group, `psi_g^1` and `psi_g^2`.
4. Equation `(15)`: evolves the DG energy coefficients with stopping power `S(E)`, straggling `T(E)`, and catastrophic loss `sigma_c,t`. This is the core formula needed for a correct Bragg peak.
5. Equations `(16)`-`(19)`: split the depth evolution into `L1` spatial transport in `y,z`, `L2` angular diffusion in `u,v`, and `L3/L4` energy slowing-down terms.
6. Strang splitting with Crank-Nicolson substeps: advances one depth step using energy half step, spatial half step, angular full step, spatial half step, and energy half step.
7. Equations `(20)` and `(21)`: discretize and solve angular diffusion with the zero-boundary sine-series/DST-I method.
8. Equation `(22)`: updates lateral `y,z` transport with the second-order finite-volume/MUSCL scheme and a CN predictor-corrector depth step.
9. Dose definition `D(r) = Edep(r) / rho(r)` and equation `(24)`: compute local primary energy deposition; the secondary term is added analogously.
10. Section 4.2 IDD definition: output `IDD(x) = integral integral D(x,y,z) dy dz`, which is the curve plotted in Figure 3.

In short:

```text
Eq. (23) initial beam
-> Eq. (9), (10a), (10b) primary/secondary evolution
-> Eq. (11)-(15) DG energy update
-> Eq. (16)-(22) depth splitting, angular diffusion, and YZ transport
-> Eq. (24) energy deposition
-> D = Edep / rho
-> IDD(x) = integral integral D dy dz
-> Figure 3
```

## Roadmap Toward Paper-Equivalent Results

The solver now writes `idd_output.txt` through the paper-facing `Edep -> D -> IDD`
output path. Remaining work is mainly broader numerical fidelity: paper-grid
scaling, air/heterogeneous material validation, spot/longitudinal outputs, and
performance optimization.

Recommended implementation order:

1. Complete paper-style output buffers and files.
   - `idd_output.txt` with columns `x IDD` is implemented.
   - Add local `Edep_p(y,z)`, `Edep_s(y,z)`, and `D(y,z) = Edep(y,z) / rho(y,z)` outputs at each
     depth.
   - Add optional `spot_x_<depth>.txt` outputs for selected `YZ` dose planes.

2. Continue validating the energy model.
   - `--energy-model eq15` implements the current equation `(15)` DG/CN path in
     full in-memory mode.
   - `--energy-model legacy` keeps the older stable upwind path for comparison.
   - The paper uses a second-order DG discretization in energy with two coefficients per group,
     Crank-Nicolson updates, stopping power `S(E)`, straggling `T(E)`, and catastrophic loss
     `sigma_c,t`.
   - Current water/bone 100/230 MeV IDD positions are within the 1-2% target on
     the fine development grid.

3. Validate the Strang splitting order.
   - Implemented sequence: energy half step, spatial transport half step, angular diffusion full step,
     spatial transport half step, energy half step.
   - Further validation should compare convergence against the paper's Figure 1/2 diagnostics.

4. Complete transport and secondary validation.
   - The spatial transport kernel now uses the Eq. `(22)` MUSCL limiter and a
     predictor-corrector CN update.
   - The full in-memory secondary source uses energy/angle transition kernels.
   - The out-of-core streaming path uses the same energy/angle transition
     source formula but is still much slower than full in-memory mode.
   - Validate convergence toward the paper's `20x20` angular grid.

See `TODO.md` for the detailed paper reproduction roadmap.

## Profiling

Nsight Systems small run:

```bash
nsys profile --force-overwrite=true \
  --trace=cuda,cublas,cusparse,nvtx,osrt \
  --stats=true \
  -o nsys_bfpn_small \
  ./build/bin/bfp_solver 100 --data data --time 4.0
```

Nsight Compute single-kernel samples:

```bash
ncu --target-processes all --set basic \
  --kernel-name regex:spatialTransportKernel \
  --launch-count 1 \
  --force-overwrite \
  --export ncu_spatial_small \
  ./build/bin/bfp_solver 100 --data data --time 0.4

ncu --target-processes all --set basic \
  --kernel-name regex:energyDgCnKernel \
  --launch-count 1 \
  --force-overwrite \
  --export ncu_energy_small \
  ./build/bin/bfp_solver 100 --data data --time 0.4
```

Current profiling conclusion: the main performance bottleneck is excessive launch count in angular diffusion. The code performs many tiny `11x11` FFTs one slice at a time. Replace this with batched cuFFT before optimizing individual kernels.
