# Native CPU/CUDA results, 2026-10-19

The source, contracts and reproduction commands are in the
[seventh increment](seventh-increment.md). Raw measurements are committed in
`validation/results/native-baseline.json` and `native-accuracy.txt`.

## Hardware and build

- GPU: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU; compute capability
  12.0; 26 SMs; 8,151 MiB reported memory; driver 615.71.09.
- CUDA driver API reports 13.4; the compiled toolkit/runtime is 12.8.1
  (nvcc 12.8.93, cudart 12.8.90). GCC is Ubuntu 13.3.0.
- CPU: Intel Core Ultra 7 265H. The reported CPU baseline uses one scalar C++
  thread and the same mathematical source, double precision and stopping rules.
- Build settings: C++17, -O3, no fast math, no implicit FP contraction, precise
  CUDA division/sqrt, no flush-to-zero. Explicit FMA product remainders remain.
- The recorded build took 3.28 seconds: 0.46 s CPU compilation/linking and
  2.82 s CUDA compilation/linking. First library load/device query/context setup
  in the benchmark took 271.50 ms. Neither is included in warmed timings.

## Independent accuracy

All 44 standard-library reference tests and 11 explicit native integration tests
pass. The GPU is exercised directly, with no CPU emulation or fallback.
NVIDIA Compute Sanitizer reports zero memory errors, zero leaked device bytes,
and zero uninitialized-memory errors in the tested batch lifecycle cases.

The C++ scalar checks independently cover 3,541 TOF evaluations/root inversions
and 1,411 derivative evaluations, including both original scalar-lambda and
retained-q inputs. Maximum relative T error is 2.248e-15. Derivative maxima
are 7.734e-15 / 6.783e-14 / 6.260e-13 for orders 1/2/3. The maximum scaled
inverse error is 6.467e-12 on a warned flat long-arc curve; ordinary cases retain
the previous 5e-14 acceptance bound. Forward roundoff stays within the calibrated
16-ulp diagnostic model, with measured maximum 14.397 ulps in this scalar corpus.

Each backend executes 440 full transfers: 248 independently generated conics,
144 actual-input multiprecision solutions in difficult geometries, and 48
additional extreme range checks. These are numerical validations, not timing
distributions. Each backend reports one conditioning warning and six unknown
precision diagnostics, all on deliberate extreme cases. No unexpected failures
occur. Unknown diagnostics are not treated as False.

| Maximum error | C++ CPU | CUDA GPU | Acceptance |
| --- | ---: | ---: | ---: |
| Scaled velocity vs known conic, 248 | 2.623e-13 | 2.623e-13 | 2e-11 |
| Scaled velocity vs actual-input multiprecision reference, 144 | 6.672e-16 | 9.551e-16 | 2e-12 |
| Scaled velocity vs independent Gooding, 240 | 1.135e-13 | 1.136e-13 | 2e-11 |
| Scaled endpoint energy difference | 1.450e-15 | 1.381e-15 | 2e-12 |
| Scaled angular momentum vector difference | 1.486e-15 | 1.073e-15 | 2e-12 |
| Scaled propagated position error, 248 | 4.400e-11 | 4.455e-11 | 5e-10 |
| Scaled propagated velocity error, 248 | 3.252e-11 | 3.231e-11 | 5e-10 |
| Scaled velocity difference from Python, all 440 | 1.530e-15 | 3.184e-14 | 2e-11 |
| Forward-roundoff diagnostic calibration at returned full-solve roots | 6.762 ulps | 6.762 ulps | 16 ulps |

Error scales match the earlier Python validation: conic velocities use
max(sqrt(mu/p), endpoint reference speeds), arrival position uses max(p,r2),
and invariant checks use nonzero characteristic scales even at zero orbital
energy. Actual-input velocity checks use max(sqrt(mu/r1), both reference speeds).
Propagation errors include numerical integration error. A scaled error is not
an error in metres or m/s. CPU/GPU/Python agreement is secondary evidence;
the analytical, multiprecision, Gooding and propagation checks are independent.

The actual GPU checks retain the known-input rounding distinctions from the
sixth increment. Exact collinear inputs remain unsupported; none of these
results establishes a universal certificate for every finite binary64 input.

## Mars 2020 regression

The cached official Horizons states define three problems: Earth center at
launch to Mars center at entry, spacecraft Aug 1 to Feb 17, and spacecraft
Oct 15 to Dec 1. All three pass both native backends. Maximum differences are:

| Check | C++ CPU | CUDA GPU |
| --- | ---: | ---: |
| Gooding endpoint velocity comparison | 2.022e-11 m/s | 1.859e-11 m/s |
| Propagated arrival position closure | 0.694 mm | 0.669 mm |
| Propagated arrival velocity closure | 7.618e-11 m/s | 8.410e-11 m/s |

Acceptance remains 1e-6 m/s for the velocity checks and 10 m for position.
These verify the same Sun-only Lambert problem. The differences from the real
spacecraft's dynamical history in the [Mars 2020 comparison](mars2020-validation.md)
remain; a fast two-body solver does not itself reproduce that mission.

## Timing method and distributions

The benchmark uses fixed seed 958707 and prepacked aligned NumPy inputs. CPU
timing is the median of 3 calls for large batches and 7 for small batches, using
the C ABI directly with preallocated output. It includes ctypes dispatch and
single-thread C++ execution, excluding data generation and output allocation.

CUDA kernel timing is a median of three CUDA-event measurements over 5-50
launches after warmup. It measures stream elapsed time per launch, including any
host submission gaps, excluding allocation, uploads and downloads. Repetition
does not transfer data or allocate device memory. The transfer-inclusive time
is a separate median of upload+one solve+download with existing device buffers;
it includes host wrapper/stream-synchronization overhead and output allocation.
Upload/download wall times are also preserved in the raw JSON.

The synthetic workflow timing additionally includes fresh vectorized input
generation, device allocations and cleanup. It excludes compiler/import/context
initialization and real ephemeris, optimization or GUI work. This is the measured
workflow in this benchmark, not total time for a scientific mission application.
CPU compute times must not be treated as matched full CPU workflow measurements.

Four distributions are distinguished:

- Circular: identical quarter-circle transfers at an AU/Sun scale. This is a
  homogeneous control and gives the highest throughput, not a mission average.
- Transfer search: varied angles/radii, modest out-of-plane components, 100-350
  day durations, 10% long arcs, AU/Sun dimensions, with the starting radius along +X.
- Rotated search: the same transfer-search problems, each independently rotated
  into a general 3D frame to expose costs hidden by axis-aligned starting states.
- Mixed: varied geometry, both arc directions, every eighth angle 1e-12 radians,
  and log-distributed durations from 0.01 to 10,000 days. It exercises a wider
  range of iteration and series paths.

## Measured large-batch baseline

For **65,536 problems**, at the initial 128 threads/block:

| Distribution | CPU compute | GPU kernel | GPU with transfers | Synthetic workflow | Kernel throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circular control | 40.84 ms | 1.705 ms | 5.351 ms | 6.377 ms | 38.44 million/s |
| Transfer search | 58.03 ms | 8.248 ms | 11.89 ms | 15.43 ms | 7.95 million/s |
| Rotated search | 64.86 ms | 8.261 ms | 11.64 ms | 19.75 ms | 7.93 million/s |
| Mixed | 54.20 ms | 10.34 ms | 13.75 ms | 17.84 ms | 6.34 million/s |

Against this single-thread C++ compute baseline, kernel-only ratios are 23.95x,
7.04x, 7.85x and 5.24x. Transfer-inclusive ratios are 7.63x, 4.88x, 5.57x and 3.94x. This is
not a multicore CPU comparison. All benchmark solves succeed; maximum scaled
CPU/GPU velocity difference across measured batches is 4.443e-15.

Small batches can favor the CPU. A single transfer-search case takes about
2.28 microseconds in the C ABI CPU call versus 61.83 microseconds of resident
GPU stream time and 93.01 microseconds with transfers. At 1,024 mixed cases,
CPU compute is 0.860 ms versus GPU 0.951 ms kernel-only and 1.137 ms with
transfers. There is no automatic backend switch or universal break-even claim.

An initial run exposed an unnecessary cudaGetDeviceProperties call in every
host-side run. Grid limits are now cached at buffer creation. This reduced
small-batch transfer-inclusive overhead; the table above is from the final
implementation. The kernel arithmetic and accuracy contracts were unchanged.

## Resources, block sizes and divergence evidence

The compiler reports 196 registers/thread, zero spill loads/stores, a 32-byte
stack/local frame, no barriers and no shared memory. CUDA predicts 16.7%
occupancy for the tested block sizes: 8/4/2/1 resident blocks per SM at
32/64/128/256 threads. This is a resource bound, not measured SM utilization.
The compiled kernel supports at most 256 threads/block on this device.

At 16,384 problems, median event times were:

| Threads/block | Transfer search | Mixed |
| --- | ---: | ---: |
| 32 | 2.288 ms | 2.859 ms |
| 64 | 2.269 ms | 2.963 ms |
| 128 | 2.456 ms | 2.980 ms |
| 256 | 2.664 ms | 3.339 ms |

The best measured block size differs by distribution; 128 remains the explicit
initial default and users can set threads. These scans do not establish a
universal optimum. Clocks, laptop power/thermal policy and OS load were not
locked, and no statistical confidence interval is claimed.

The large circular batch has exactly three updates per solve. Transfer search
and rotated search average 2.936 with maximum three; mixed averages 3.000 with
maximum 44. A proxy computed as sum(iterations)/(32*sum(warp maximum iterations))
is 1.000, 0.979, 0.979 and 0.809 respectively for complete warps. This indicates potential imbalance
but is not a hardware branch-efficiency metric and does not account for the
different work inside each update. Nsight hardware counters remain future work.

Register live ranges, short-sum local storage, data layout, diagnostic transfer
volume, pageable host transfers and series/transcendental paths are the next
profiling targets. Actual gains must be measured while rerunning accuracy checks.
This increment establishes a functioning, validated GPU baseline with resident
batch reuse; it does not claim the engine is fully tuned.
