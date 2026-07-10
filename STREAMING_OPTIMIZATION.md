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

Current streaming chunks follow this pattern:

```text
read -> H2D -> kernel -> D2H -> synchronize -> write
```

A double-buffer pipeline would overlap:

```text
chunk N GPU work
chunk N+1 host read / H2D preparation
chunk N-1 D2H / host write
```

This is likely the next meaningful optimization after the current local
improvements.

Risk:

```text
Medium. Requires careful buffer ownership and stream/event synchronization.
```

Correctness validation:

```text
Run water100 time=0.4 and time=8.0.
Require max_abs=0, max_rel=0 and BP=7.52 cm.
```

### 3. Paper-grid multi-lane benchmark

The source cache is meant for multi-lane paper-grid behavior. It should be
benchmarked on a case where:

```text
effective_lane_chunk < (Ny+1)*(Nz+1)*Nmu*Nom
secondary source cache = on
```

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
Run nsys on the current optimized streaming code, then implement a
double-buffer pipeline only for the hottest streaming pass.
```

Continue to validate every change against:

```text
results/eq15_strict_fine_clip_water_100/idd_output.txt
water100 BP = 7.52 cm
full-vs-streaming IDD max_abs = 0
full-vs-streaming IDD max_rel = 0
```
