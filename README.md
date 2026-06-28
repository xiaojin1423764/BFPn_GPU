# BFPn GPU Solver

CUDA/CMake prototype for a deterministic proton transport solver inspired by the paper `2504.00340v1.pdf`, "A deterministic solver for the linear Boltzmann model of a single mono-directional proton beam".

## Current Status

This repository currently builds and runs a simplified GPU prototype. It is not yet a faithful reproduction of the paper's numerical results.

Implemented at prototype level:

- CUDA memory wrappers and solver orchestration.
- Primary and secondary proton arrays.
- Angular diffusion step using cuFFT.
- Spatial transport step using a CUDA kernel.
- Simplified energy attenuation step using a CUDA kernel.
- Collision integral scaffolding with cuSPARSE-oriented sparse storage.
- Basic scalar dose history output.

Not yet complete:

- Paper-equivalent dose definition `D(x,y,z) = Edep(x,y,z) / rho(x,y,z)`.
- Integrated depth dose `IDD(x) = integral integral D(x,y,z) dy dz`.
- Full DG/CN energy discretization from the paper.
- Full second-order MUSCL/CN spatial transport.
- FLUKA-fitted catastrophic scattering kernels.
- Water/bone/air and heterogeneous material cases matching the paper.

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

Run a small smoke test:

```bash
./build/bin/bfp_solver 100 --data data --time 4.0
```

Run the default grid:

```bash
./build/bin/bfp_solver --data data --time 40.0
```

The optional positional argument is `Nx`. With `Lx=40`, the depth step is:

```text
dt = Lx / Nx
```

Examples:

- `Nx=100`, `--time 4.0` runs about 10 steps.
- default `Nx=4000`, `--time 40.0` runs about 4000 steps.

## Output

The solver writes:

```text
dose_output.txt
```

Format:

```text
depth_or_time scalar_dose_proxy
```

Important: this is not the paper's `D(x,y,z)` or `IDD(x)`. The current value is computed by summing the simplified `f_F` energy-loss buffer over all space, angle, and energy:

```cpp
dose_step = sum(h_f_F) * grid.dg;
```

Use this output only as a smoke-test signal until the dose model is replaced with the paper-equivalent calculation.

Plot the current scalar history:

```bash
python3 scripts/plot_results.py -i dose_output.txt -o dose_plot.png
```

## Relation to the Paper

The paper compares deterministic depth-splitting results to FLUKA using:

- `IDD(x) = integral integral D(x,y,z) dy dz`
- spot distributions on the `YZ` plane
- longitudinal dose data
- Bragg peak and P90/D90/D20 positions

The current `dose_output.txt` is only superficially similar to an IDD curve because it has one scalar per depth step. It is not directly comparable to the paper figures.

For a faithful reproduction, the next major milestone is to compute and output local energy deposition and IDD according to the paper's dose definition.

## Roadmap Toward Paper-Equivalent Results

The current scalar `dose_output.txt` cannot be turned into the paper's IDD curve by plotting
or reduction changes alone. The solver must first implement the paper's energy deposition
and output path.

Recommended implementation order:

1. Add paper-style output buffers and files.
   - Compute local `Edep_p(y,z)`, `Edep_s(y,z)`, and `D(y,z) = Edep(y,z) / rho(y,z)` at each
     depth.
   - Write `idd_output.txt` with columns `x IDD`, where `IDD(x) = integral integral D(x,y,z) dy dz`.
   - Add optional `spot_x_<depth>.txt` outputs for selected `YZ` dose planes.

2. Replace the simplified energy attenuation model.
   - The current kernel uses `Fnew = Fold * exp(-sigma * dt)`.
   - The paper uses a second-order DG discretization in energy with two coefficients per group,
     Crank-Nicolson updates, stopping power `S(E)`, straggling `T(E)`, and catastrophic loss
     `sigma_c,t`.
   - This step is required before expecting a Bragg peak or paper-like IDD.

3. Change the depth step to the paper's Strang splitting order.
   - Target sequence: energy half step, spatial transport half step, angular diffusion full step,
     spatial transport half step, energy half step.
   - Keep changes staged so each subsystem can be validated independently.

4. Upgrade transport and secondary physics.
   - Replace first-order upwind transport with the paper's second-order MUSCL fluxes.
   - Implement catastrophic-scattering source terms for secondary protons using
     `sigma_c,s`, `sigma_c,t`, and transition matrices.
   - Validate angular diffusion semantics and move toward the paper's `20x20` angular grid.

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
  --kernel-name regex:energyAttenuationKernel \
  --launch-count 1 \
  --force-overwrite \
  --export ncu_energy_small \
  ./build/bin/bfp_solver 100 --data data --time 0.4
```

Current profiling conclusion: the main performance bottleneck is excessive launch count in angular diffusion. The code performs many tiny `11x11` FFTs one slice at a time. Replace this with batched cuFFT before optimizing individual kernels.
