# Streaming Optimization Notes

Date: 2026-07-06

This note records the current streaming optimization state, numerical
validation results, and remaining optimization directions.

## Reference Validation Target

The current numerical reference is the water 100 MeV curve in:

```text
results/eq15_strict_fine_water_bone_100_230.png
```

The reference water 100 MeV Bragg peak is:

```text
water100 BP = 7.52 cm
```

The corresponding full-mode data file is:

```text
results/eq15_strict_fine_clip_water_100/idd_output.txt
```

For every streaming optimization below, correctness is checked by comparing
IDD output against full mode and against the water100 reference prefix from
the figure data.

Required correctness criteria:

```text
full vs streaming IDD max_abs = 0
full vs streaming IDD max_rel = 0
water100 BP remains 7.52 cm when the run reaches the peak
```

`dose_output.txt` is not currently a valid streaming correctness metric because
streaming mode writes a zero diagnostic proxy there. Use `idd_output.txt`.

## Current Correctness Results

### Short water100 streaming check

Case:

```text
Nx=2000, Ny=Nz=8, Ng=500, Nmu=Nom=7
material=water, energy=100, time=0.4, energy-model=eq15
```

Result:

```text
streaming vs full IDD max_abs = 0
streaming vs full IDD max_rel = 0
streaming vs figure water100 prefix max_abs = 0
streaming vs figure water100 prefix max_rel = 0
```

### Long water100 streaming check

Case:

```text
Nx=2000, Ny=Nz=8, Ng=500, Nmu=Nom=7
material=water, energy=100, time=8.0, energy-model=eq15
```

Result:

```text
streaming vs full IDD max_abs = 0
streaming vs full IDD max_rel = 0
streaming vs figure water100 prefix max_abs = 0
streaming vs figure water100 prefix max_rel = 0
BP = 7.52 cm
```

This confirms that the optimized streaming result is still aligned with the
water100 Bragg peak in `results/eq15_strict_fine_water_bone_100_230.png`.

### Multi-lane source-cache path

Case:

```text
time=0.4, lane-chunk=980
```

This forces the path used when a lane chunk does not cover a complete
`(y,z,angle)` plane, similar to paper-grid streaming behavior.

Result:

```text
streaming vs full IDD max_abs = 0
streaming vs full IDD max_rel = 0
streaming vs figure water100 prefix max_abs = 0
streaming vs figure water100 prefix max_rel = 0
secondary source cache = on
```

## Current Performance Results

Known timings on the current machine for the water100 `time=8.0` case:

```text
full mode                                  65.02 s
streaming default ec128                   197.47 s
streaming source reuse                    157.51 s
streaming source reuse + fast I/O         148.42 s
streaming source reuse + fast I/O + pair  148.33 s
```

The pair transport/angle pass is numerically correct but gives only a very
small speedup on the small water100 test case.

Short `time=0.4` timings:

```text
streaming fast I/O                         6.02 s
streaming pair-pass                        5.82 s
multi-lane source cache, lane-chunk=980    10.95 s
multi-lane pair + source cache             11.15 s
```

The multi-lane short case is mainly a correctness test for the paper-grid path;
it is too small to prove paper-grid performance benefit.

## Optimizations Already Implemented

### Complete-lane continuous I/O fast path

When `lane_chunk` covers the full `(y,z,angle)` plane, `streamingEnergyStep`
now reads and writes each backing file as one continuous block instead of
looping over every energy group.

Expected benefit:

```text
Fewer pread/pwrite calls and less host-side syscall overhead.
```

Validated:

```text
water100 IDD remains bitwise identical to full mode.
```

### Secondary source reuse for complete-lane chunks

For complete-lane chunks, the second secondary energy half-step reuses the
secondary source already resident in device memory.

Expected benefit:

```text
Avoids one primary F/f_F read and one secondary source kernel launch.
```

Validated:

```text
water100 IDD remains bitwise identical to full mode.
```

### Secondary source cache for multi-lane chunks

When a lane chunk does not cover the full `(y,z,angle)` plane, streaming now
creates a temporary source backing file:

```text
source.bin
```

The first secondary energy half-step computes the source and writes it to this
cache. The second secondary energy half-step reads the cached source instead of
reading primary `F/f_F` again and recomputing the source.

Expected benefit:

```text
Paper-grid and other multi-lane streaming cases avoid repeated primary source
generation in the second secondary energy half-step.
```

Validated:

```text
lane-chunk=980 water100 IDD remains bitwise identical to full mode.
```

### Pair transport and angle streaming passes

Streaming now has pair versions of transport and angle steps that read both
`F` and `f_F` chunks, launch the two solver calls, and write both chunks back
under one synchronization point.

Expected benefit:

```text
Slightly fewer synchronization points and less duplicated pass orchestration.
```

Observed benefit:

```text
Small water100 time=8.0 case improves only from 148.42 s to 148.33 s.
```

This change is correct but not a major optimization by itself.

## Tried And Reverted

### Reused pinned host buffer for IDD reduction partial sums

Attempt:

```text
Replace per-call std::vector<double> h_block_sums with a reusable HostDoubleBuffer.
```

Result:

```text
Numerically correct, but the short water100 time=0.4 case slowed down.
```

Decision:

```text
Reverted. Keep the existing std::vector reduction buffer.
```

## Scheme A/B Optimization Plan

This section records the two larger streaming optimization directions currently
under discussion. Scheme A is validated and retained. A chunk-major source-cache
variant of Scheme B was tested and reverted because it regressed end-to-end time.

### Scheme A: fused tiled secondary-source kernel

Goal:

```text
Reduce repeated angular-neighbor work in the streaming secondary source without
creating a large intermediate global-memory source array.
```

Current implementation status:

```text
Implemented in src/bfpm_solver.cu and validated on an NVIDIA GeForce RTX 5070 Ti.
```

The new prototype kernel is:

```text
streamingSecondarySourceTiledEnergyKernel<OUT_TILE>
```

The current call path uses:

```text
OUT_TILE = 4
```

The old optimized sparse source kernel remains in the file as a fallback:

```text
streamingSecondarySourceKernel
```

Algorithmic idea:

```text
One thread owns one streaming lane and computes four neighboring output energy
groups for that lane. For the same input lane and angular sparse transition
list, it accumulates the catastrophic-scattering source into a small register
tile over output energy. This reuses the angular sparse-convolution traversal
across four output groups.
```

Compared with the previous sparse source kernel:

```text
previous: one thread computes one output energy group and repeats angular
          neighbor traversal independently for each group

scheme A: one thread computes OUT_TILE neighboring output groups and reuses the
          angular traversal inside the thread
```

What it deliberately avoids:

```text
No separate angular-convolution output array is written to global memory.
No two-stage source factorization is retained.
No mmap/full-lane backing-store change is included in this scheme.
```

Reason for this design:

```text
The previous two-stage factorization made the measured source kernel much
faster, but total runtime became worse because the intermediate source arrays
increased memory traffic and pressure. Scheme A keeps the fusion inside one
kernel and tries to reduce arithmetic traversal while preserving the existing
streaming I/O structure.
```

Known risk before accepting:

```text
The prototype builds, but ptxas reports about 96 registers for the tiled source
kernel versus about 60 registers for the previous sparse kernel. Occupancy may
drop enough to erase or reverse the arithmetic benefit. This must be measured.
```

Validation result (2026-07-11):

```text
small full vs streaming IDD       byte-for-byte identical
large sparse vs tiled IDD         byte-for-byte identical
compute-sanitizer memcheck        ERROR SUMMARY: 0 errors

same-machine large sparse:
  total                           89.442 s
  secondary source               19.380 s
  external elapsed               97.15 s

same-machine Scheme A:
  total                           71.287 s
  secondary source                2.087 s
  external elapsed               78.98 s

Scheme A improvement:
  internal total                  20.3%
  secondary source               89.2%

primary-only regression:
  total                           21.422 s
  previous reference             21.33 s
  difference                       0.4%
```

The higher register count did not outweigh reuse on this workload. Scheme A is
therefore the accepted source kernel.

Acceptance criteria:

```text
1. Full-vs-streaming IDD must remain bitwise identical on the small secondary
   test, or any tiny FP-order difference must be explicitly quantified.
2. compute-sanitizer must report no memory errors.
3. Large secondary streaming runtime must improve over the current sparse-source
   baseline, especially the "secondary source" profile line.
4. Primary-only streaming runtime must not regress, because Scheme A should only
   affect the secondary source path.
```

### Scheme B: streaming energy/source I/O layout redesign

Goal:

```text
Reduce copy-in, copy-out, and secondary source cache I/O cost by changing how
streaming chunks are laid out and moved between host backing storage and GPU
working buffers.
```

Current status:

```text
A source-cache-only chunk-major scratch variant was implemented and measured,
then reverted. A broader layout redesign has not been implemented.
```

Motivation from the pre-Scheme-A large-grid secondary profile:

```text
large secondary total time       about 94.341 s
energy phase                     about 66.989 s
secondary source                 about 21.731 s
copy in                          about 17.320 s
copy out                         about 18.033 s
secondary source cache I/O        about 5.418 s
```

The current streaming layout is a compromise:

```text
energy step prefers lane-major chunks for high-to-low energy recurrence
transport/angle steps prefer energy-contiguous chunks over spatial/angle lanes
secondary source needs both primary input access and secondary source cache
reuse
```

Possible Scheme B designs:

```text
1. Maintain two backing layouts:
   one energy-step-friendly layout and one transport/angle-friendly layout.

2. Use a chunk-local transpose:
   read the existing backing layout, transpose inside pinned host memory or on
   the GPU, run the operator with coalesced access, then transpose back only
   when needed.

3. Add lane-major scratch files only for source/energy hot paths:
   avoid changing all operators at once, but let the secondary source and energy
   recurrence read/write in the order they actually consume data.
```

Design constraints:

```text
Do not repeat the naive mmap backing-store attempt; it was bitwise correct but
slower.

Do not repeat full-lane/larger lane chunks as a standalone optimization; that
made transport/angle/recurrence behavior worse in previous tests.

Do not keep a layout redesign unless IDD validation remains correct and the
large-grid timing improves end-to-end, not only in one kernel.
```

Expected benefit if successful:

```text
Scheme A mainly attacks secondary source arithmetic. Scheme B attacks the
larger streaming overhead: H2D/D2H copies, host file reads/writes, and source
cache traffic.
```

Tested and reverted (2026-07-11): chunk-major source cache

```text
The source cache is private to the energy pass, so the prototype stored each
[Ng][lane chunk] slab contiguously. This reduced about Ng scattered pread/pwrite
operations per lane chunk to one contiguous operation without changing the GPU
layout.

Scheme A baseline:
  total                           71.287 s
  secondary source I/O            5.704 s
  external elapsed               78.98 s

Scheme A + chunk-major cache:
  total                           75.561 s
  secondary source I/O            5.218 s
  external elapsed               83.43 s

IDD was byte-for-byte identical, and source-cache I/O improved by 8.5%, but
internal total regressed by 6.0% and external elapsed regressed by 5.6%.
```

Decision:

```text
Do not retain the chunk-major source-cache-only variant. It reduces syscall
count but not transferred bytes or H2D/D2H traffic. Any next Scheme B attempt
must eliminate a full data pass or overlap storage I/O with GPU work; a pure
on-disk reorder is not sufficient.
```

## Optimization Audit: 2026-07-11

The post-Scheme-A audit retained three additional changes:

```text
1. Read and write F/f_F backing files concurrently to increase NVMe queue depth.
2. Write the MUSCL corrector directly to F and remove its final full-array D2D copy.
3. Use the separable DST for angular grids above 8x8 and write its inverse directly
   to F, removing another full-array D2D copy and temporary array.
```

Numerical validation:

```text
default 7x7 full reference vs optimized streaming IDD   byte-for-byte identical
20x20 direct DST vs optimized separable DST IDD         byte-for-byte identical
large Scheme A vs final optimized IDD                   byte-for-byte identical
compute-sanitizer memcheck                              ERROR SUMMARY: 0 errors
```

Measured performance on the RTX 5070 Ti:

```text
primary-only, paired backing-file I/O:
  total                         21.422 s -> 20.514 s   (4.2% faster)

primary-only, 20x20 separable DST:
  angle                          3.55 s ->  0.212 s  (about 94% faster)
  total                         21.422 s -> 17.153 s  (about 20% faster)

large secondary, final retained path:
  Scheme A total                71.287 s
  final total                   59.463 s            (16.6% faster)
  Scheme A angle                9.202 s
  final angle                    0.546 s             (94.1% faster)
  Scheme A external elapsed     78.98 s
  final external elapsed        67.62 s             (14.4% faster)
```

The MUSCL and DST in-place outputs remove deterministic device-memory traffic and
temporary storage. Their isolated timing is hidden by backing-store variance, so
the measured end-to-end numbers above should not be attributed to those copies
alone.

The audit identified a true double-buffer pipeline as the next high-impact work:

```text
overlap pwrite(chunk N) with pread/H2D/compute(chunk N+1), using two pinned host
buffer pairs and CUDA events for timing without host synchronization. After that,
investigate whether a blocked layout can reduce complete backing-store passes
between energy-major and transport/angle-major operators.
```

The writeback portion of this pipeline is implemented in the five-item pass
below. A shared blocked layout remains future work.

## Five-Item Optimization Pass: 2026-07-11

Implemented and retained:

```text
1. Two-slot pinned-host writeback pipeline for energy and fused transport passes.
2. Removed the source.bin pass; Scheme A recomputes source for the second half-step.
3. CUDA-event timing replaces intermediate profiling synchronizations.
4. MUSCL neighbor indices and plane bases are computed once per output point.
5. IDD/scalar/energy-flux partials are recursively reduced to one scalar on GPU.
```

Numerical acceptance uses 17-digit output:

```text
7x7 final full vs streaming:
  max_abs                       3.47e-18
  max_rel                       2.15e-16

20x20 final full vs streaming:
  max_abs                       0
  max_rel                       0

pre-pass vs final streaming:
  maximum observed max_abs      8.67e-18
  maximum observed max_rel      7.88e-16

compute-sanitizer memcheck      ERROR SUMMARY: 0 errors
```

Performance and storage:

```text
pre-pass final binary:
  internal profiled total       59.463 s
  external elapsed              67.62 s
  backing store                 12.52 GiB / 5 files

five-item pass:
  internal profiled total       54.635 s
  external no-profile elapsed   64.29 s
  backing store                 10.02 GiB / 4 files
```

The profiled copy-out detail is accumulated work time and can exceed wall time
because writeback overlaps subsequent chunks. Use solver total and external
elapsed for end-to-end comparisons.

## Active-Energy Prefix Optimization: 2026-07-11

The retained streaming path now avoids work above the highest energy group that
can be nonzero. The bound is established from exact initialization values and
propagated without per-step scans or GPU atomics:

```text
energy DG recurrence without straggling: high energy -> low energy only
catastrophic secondary kernel:           output energy <= source energy
transport and angular diffusion:         do not change energy group
```

At initialization the source kernel matrices are checked for this triangular
property. With straggling enabled, the bound expands conservatively by one group
per energy half-step. The optimized energy pass reads/writes only the active
prefix; fused transport/angle skips energy chunks wholly above the bound.
`--no-zero-chunk-skip` disables both parts for same-binary comparison.

Rejected variants before the retained implementation:

```text
read-ahead concurrent with writeback     regressed badly from storage contention
fused primary-generated source cache     55.474 s vs 54.635 s baseline
host scan for exact-zero chunks           59.049 s vs 54.635 s baseline
GPU atomic active-bound tracking          59.846 s vs 54.635 s baseline
```

Same-binary A/B on water 100 MeV, two steps,
`Ny=Nz=30, Ng=500, Nmu=Nom=20`:

```text
inactive skipping disabled               34.809281 s
active-energy prefix + chunk skip         24.631010 s
improvement                                  29.2%
skipped transport/angle chunks             3 / 12 (25%)
IDD SHA-256                               identical
```

Numerical checks:

```text
7x7 bone-100 full vs streaming max_abs    1.73e-18
7x7 bone-100 full vs streaming max_rel    4.03e-16
20x20 water-100 A/B IDD                    byte-identical
water-230 optimized vs disabled IDD        byte-identical
ctest                                      1/1 passed
```

## Three-Item Backing-Path Pass: 2026-07-11

Three additional changes were evaluated with spatial positivity clipping kept
enabled throughout.

### 1. Tiled backing layout: reverted

A lane-slab physical file layout and conversion helpers were implemented. The
legacy and tiled streaming IDD files were byte-for-byte identical, and the
full-vs-streaming maximum relative difference remained `4.03e-16`. However,
the adjacent large-grid timing regressed:

```text
legacy layout                           28.283244 s
tiled lane-slab layout                  29.289776 s
regression                                   3.6%
```

The layout and conversion code were removed. Lower syscall count alone did not
offset the conversion and access costs.

### 2. Primary operator-boundary fusion: retained

The streaming primary path now processes energy chunks from high to low and
executes:

```text
transport half step -> angle full step -> transport half step
-> second primary energy half step
```

while a chunk is resident. Four lane carry arrays preserve old/new `F/f_F`
values at energy-chunk boundaries. This eliminates one complete primary state
writeback and the following energy-pass reread. Straggling keeps the original
path, and `--no-fused-stream-boundary` is available for same-binary comparison.

```text
fusion disabled                         27.252030 s
fusion enabled                          24.593229 s
improvement                                  9.8%
```

Correctness checks:

```text
bone-100, two steps, fusion on/off       IDD and diagnostic byte-identical
water-230, two steps, fusion on/off      max_abs 8.67e-19
                                         max_rel 1.82e-16 (one ULP)
compute-sanitizer memcheck               ERROR SUMMARY: 0 errors
ctest                                    1/1 passed
spatial positivity clipping              enabled
```

The water-230 difference is caused by the reversed energy-chunk reduction
order; it is within double-precision rounding and does not change the computed
curve.

### 3. Secondary source/recurrence fusion: reverted

A prototype directly consumed the tiled secondary source inside the high-to-low
energy recurrence, avoiding the intermediate device source array. It preserved
the original source-energy and angular-neighbor accumulation order and produced
byte-identical small-test output. Its resource use and end-to-end result were
unfavorable:

```text
prototype kernel                         128 registers, 64-byte stack
fused secondary source                   10.071184 s
retained source + recurrence              2.680719 s + 0.465401 s
```

The prototype was removed. Assigning a whole energy recurrence to each lane
reduced GPU parallelism too much; a future attempt needs a different source
factorization rather than this per-lane serial fusion.

## Required Scheme A/B Tests

Run these before accepting Scheme A or any later Scheme B implementation.

### Build

```bash
make
```

### Small secondary full-mode reference

```bash
/home/xj/BFPn_GPU_Solver/build/bin/bfp_solver 20 \
  --data /home/xj/BFPn_GPU_Solver/BFPn_CPU_Solver/water \
  --time 4.0 --material water --energy 100 \
  --ny 8 --nz 8 --ng 64 --nmu 7 --nom 7 --energy-model eq15
```

### Small secondary streaming check

```bash
/home/xj/BFPn_GPU_Solver/build/bin/bfp_solver 20 \
  --data /home/xj/BFPn_GPU_Solver/BFPn_CPU_Solver/water \
  --time 4.0 --material water --energy 100 \
  --ny 8 --nz 8 --ng 64 --nmu 7 --nom 7 --energy-model eq15 \
  --streaming-full --energy-chunk 16 --lane-chunk 2048 \
  --stream-dir /tmp/bfpn_verify_scheme_a --idd-stride 1 --profile-steps
```

Compare `idd_output.txt` from full and streaming runs:

```text
required before accepting: 17-digit output differences remain within quantified
double-precision rounding tolerance
```

### Memory-safety check

```bash
compute-sanitizer --tool memcheck \
  /home/xj/BFPn_GPU_Solver/build/bin/bfp_solver 100 \
  --data /home/xj/BFPn_GPU_Solver/BFPn_CPU_Solver/water \
  --time 0.8 --material water --energy 100 \
  --ny 4 --nz 4 --ng 500 --nmu 20 --nom 20 --energy-model eq15 \
  --streaming-full --energy-chunk 128 --lane-chunk 2048 \
  --stream-dir /tmp/bfpn_scheme_a_sanitizer --idd-stride 999 \
  --profile-steps
```

Required result:

```text
ERROR SUMMARY: 0 errors
```

### Large secondary performance benchmark

```bash
/home/xj/BFPn_GPU_Solver/build/bin/bfp_solver 100 \
  --data /home/xj/BFPn_GPU_Solver/BFPn_CPU_Solver/water \
  --time 0.8 --material water --energy 100 \
  --ny 40 --nz 40 --ng 500 --nmu 20 --nom 20 --energy-model eq15 \
  --streaming-full --energy-chunk 128 --lane-chunk 262144 \
  --stream-dir /tmp/bfpn_scheme_a_secondary_t08 --idd-stride 999 \
  --profile-steps
```

Compare against the current sparse-source baseline:

```text
total                            94.341 s
energy                           66.989 s
secondary source                 21.731 s
copy in                          17.320 s
copy out                         18.033 s
secondary source I/O              5.418 s
```

For Scheme A, the most important line is:

```text
secondary source
```

For Scheme B, the most important lines are:

```text
copy in
copy out
secondary source I/O
total
```

### Primary-only regression check

```bash
/home/xj/BFPn_GPU_Solver/build/bin/bfp_solver 100 \
  --data /home/xj/BFPn_GPU_Solver/BFPn_CPU_Solver/water \
  --time 0.4 --material water --energy 100 \
  --ny 40 --nz 40 --ng 500 --nmu 20 --nom 20 --energy-model eq15 \
  --streaming-full --primary-only --energy-chunk 128 --lane-chunk 262144 \
  --stream-dir /tmp/bfpn_scheme_a_primary --idd-stride 999 \
  --profile-steps
```

Current reference:

```text
primary-only large streaming total is about 21.33 s
```

Scheme A should not materially change this number because it changes the
secondary source path, not the primary path.

## Remaining Optimization Opportunities

### 1. Profile the current streaming path

Before larger changes, run `nsys` on the current optimized code to identify the
dominant cost:

```text
cudaStreamSynchronize
cudaMemcpyAsync H2D/D2H
pread/pwrite
transport kernels
angle kernels
secondary source kernel
```

Recommended small profile:

```bash
nsys profile --force-overwrite=true --trace=cuda,osrt --sample=none --cpuctxsw=none \
  -o results/profiles/streaming_current/t04_water100 \
  ./build/bin/bfp_solver 2000 --time 0.4 --material water --energy 100 \
  --ny 8 --nz 8 --ng 500 --nmu 7 --nom 7 --energy-model eq15 \
  --streaming-full --energy-chunk 128 --lane-chunk 262144 \
  --stream-dir /tmp/bfpn_stream_profile_current
```

### 2. Double-buffer streaming pipeline

The original streaming chunks followed this pattern:

```text
read -> H2D -> kernel -> D2H -> synchronize -> write
```

The retained writeback pipeline overlaps:

```text
chunk N GPU work
chunk N+1 host read / H2D preparation
chunk N-1 D2H / host write
```

Energy and fused transport passes now use two pinned host-buffer slots. Device
compute remains single-buffered, so a future extension may also overlap H2D/D2H
on separate CUDA streams.

Risk:

```text
Medium. Requires careful buffer ownership and stream/event synchronization.
```

Correctness validation:

```text
Run water100 time=0.4 and time=8.0.
Require 17-digit IDD differences within double-precision rounding tolerance and
BP=7.52 cm.
```

### 3. Paper-grid multi-lane benchmark

Multi-lane paper-grid behavior should be benchmarked on a case where:

```text
effective_lane_chunk < (Ny+1)*(Nz+1)*Nmu*Nom
secondary source cache = off
```

The current path recomputes the tiled Scheme A source for the second half-step
and does not create `source.bin`.

Useful test:

```bash
./build/bin/bfp_solver 2000 --time 0.4 --material water --energy 100 \
  --ny 20 --nz 20 --ng 500 --nmu 20 --nom 20 --energy-model eq15 \
  --streaming-full --energy-chunk 128 --lane-chunk 262144 \
  --stream-dir /tmp/bfpn_stream_multilane_bench
```

Correctness should still be checked against a comparable full-mode run if the
full run fits in GPU memory.

### 4. Backing-store layout redesign

The largest remaining opportunity is probably layout-related:

```text
energy step wants lane-major chunking
transport/angle steps want energy-contiguous chunking
```

The current file layout is a compromise and causes repeated passes over the
backing store. Larger redesign options:

```text
maintain two layouts
transpose between operator phases
store operator-specific backing files
```

Risk:

```text
High. More code and more chances for indexing mistakes.
```

Correctness validation:

```text
Must preserve water100 BP=7.52 cm and full-vs-streaming IDD equality.
```

### 5. IDD stride for development runs

`--idd-stride` already exists. For performance tuning, use a larger stride to
avoid saving/reducing IDD at every step:

```bash
--idd-stride 10
```

For final figure reproduction, keep:

```bash
--idd-stride 1
```

## Current Recommendation

Next best engineering step:

```text
Prototype a shared blocked state layout that reduces complete scans of the four
remaining backing files. Keep the existing double-buffer writeback pipeline.
```

Continue to validate every change against:

```text
results/eq15_strict_fine_clip_water_100/idd_output.txt
water100 BP = 7.52 cm
17-digit full-vs-streaming IDD within double-precision rounding tolerance
```
