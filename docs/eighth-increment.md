# Eighth increment: first measured CUDA optimization

The native CPU and CUDA engines now reuse time-of-flight values during root
iteration and use one guarded addition for two-term sums. On this GPU, a matched
65,536-case search benchmark drops from 8.229 to 6.755 ms of kernel time and
from 11.168 to 9.611 ms including transfers. All independent accuracy bounds
pass. The C ABI and Python batch API are unchanged.

The previous [native results](native-results.md) remain the historical baseline.
This increment compares the saved `bc01ed0` binaries with the optimized build
in the same process, with rotating measurement order. Current source changes
are in `native/lambert.hpp`; `validation/compare_native.py` reproduces the
comparison. Raw results live in `validation/results/native-tuning-*`.

## What changed

1. Root setup evaluates T(0) and T(1) once and passes them to the starter. The
   bracket reuses the same T(0).
2. Derivative and final conditioning evaluations receive the T(x) already held
   by the root loop. They retain the existing derivative formulas, series,
   failure checks and precision estimate. The standalone derivative interface
   still computes its own T(x).
3. A two-term sum uses the correctly rounded binary64 addition and the existing
   nonfinite-result guard. Three- and four-term cancellation-sensitive sums
   retain the error-free expansions. Explicit FMA product remainders, retained
   q, tiny-arc geometry and the robust norms remain.

Root evaluation counters count the actual TOF calls and therefore decrease.
The search distribution averages 10.827 calls before and 6.194 after, while
averaging the same 2.936 updates. The circular control drops from 11 calls to 7.
All stopping tolerances, precision modes, failure statuses and diagnostics remain.
The Python mathematical reference is unchanged.

## Profiling evidence and experiments

Hardware is the NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, compute
12.0, 26 SMs, with CUDA 12.8.1 and GCC 13.3.0. The driver reports
`RmProfilingAdminOnly: 1`. Hardware performance counters require administrator
access here; driver settings were left unchanged. Evidence in this increment
comes from CUDA-event measurements, compiler resource reports, static SASS
inspection and independent numerical checks. Dynamic stall reasons and achieved
occupancy remain unmeasured.

The accepted kernel uses 197 registers/thread, a 32-byte local frame and zero
compiler-reported spill loads/stores. The baseline used 196 registers.
Predicted occupancy remains 16.7% at the explicit default of 128 threads/block.
The speed improvement comes from reduced work; register pressure remains a
profiling target. Compiler flags still disable fast math and implicit contraction
and retain precise division/sqrt and subnormal support.

Exploratory tests used seven samples of five launches at 65,536 rows, with
rotating variant order. Source patches against `bc01ed0`, compiler commands,
library hashes and raw samples are saved in `native-tuning-experiments.json`.
Each benchmark candidate passed the ordinary benchmark velocity/status checks.
Only the retained changes received the complete extreme-range validation.

| Experiment | Measured outcome | Decision |
| --- | --- | --- |
| TOF reuse alone | Search 8.252 to 7.246 ms | Keep |
| Two-term sum alone | Search 8.252 to 7.765 ms | Keep |
| Both together | Search 6.788 ms in that experiment | Keep |
| Scalar storage and unrolled expansion loops | Local frame eliminated; search/mixed times effectively unchanged | Retain the simpler existing expansion |
| Register cap 160 | About 3% faster search/mixed, about 3% slower circular; 224/180 bytes of compiler-reported spill stores/loads | Keep compiler default |
| Register caps 128 and 96 | Higher predicted occupancy with more spilling; no consistent gain across distributions | Keep compiler default |
| Power-of-two scaled vector norm | No consistent timing gain; changes numerical rounding | Retain the existing norm |
| Structure-of-arrays input | Search 6.806 versus 6.771 ms, mixed 8.268 versus 8.318 ms | Retain the public and device input layout |

The input-layout experiment transposes the eight double planes and the packed
flags before upload, leaving output layout unchanged. Its kernel timing excludes
that preparation cost. It tests coalesced input loads only; other layouts or
compact outputs remain separate experiments.

Static SASS inspection reports 18,041 non-NOP instructions in the baseline and
14,568 in the accepted kernel. The scalar-storage candidate has 13,560 and no
local load/store instructions, yet performs similarly to the accepted kernel.
These counts cover emitted library code and cold paths as well as the main
path. They do not measure executed instruction counts or locate runtime stalls.
The disassembler was NVIDIA nvdisasm 12.8.90, verified against NVIDIA's official
[redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
NVIDIA's [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
describes the distinction between static resource limits and runtime measurements.

## Matched final timing

The comparison uses seed 958707, the four previously defined distributions and
batch sizes 1, 32, 1,024, 16,384 and 65,536. Both libraries run at 128 threads/block
with identical inputs and root options. Each batch is uploaded and warmed before
seven rotating rounds. Each round measures a scalar CPU call, a CUDA-event
interval over five resident launches, and a separate upload/one launch/download
wall interval. Raw samples and library SHA-256 values are saved in
`native-tuning-comparison.json`.

For **65,536 problems**, median milliseconds are:

| Distribution | CPU before / after | Kernel before / after | With transfers before / after | New kernel throughput |
| --- | ---: | ---: | ---: | ---: |
| Circular control | 42.728 / 35.160 | 1.736 / 1.434 | 4.616 / 4.230 | 45.69 million/s |
| Transfer search | 60.211 / 46.384 | 8.229 / 6.755 | 11.168 / 9.611 | 9.70 million/s |
| Rotated search | 69.222 / 53.338 | 8.269 / 6.806 | 11.203 / 9.866 | 9.63 million/s |
| Mixed | 57.957 / 43.756 | 10.307 / 8.335 | 13.574 / 11.270 | 7.86 million/s |

Search kernel time falls 17.9%, equivalent to 21.8% higher throughput. Mixed
kernel time falls 19.1%. Transfer-inclusive search and mixed time falls 13.9%
and 17.0%. CPU time also improves because it shares the optimized mathematics.
The CPU measurements above use the scalar implementation embedded in each CUDA
library, with preallocated output; each runs on one host thread.

The kernel interval excludes uploads/downloads, allocation, initialization,
compilation and input preparation. CUDA event timing includes any host launch
submission gaps. The transfer-inclusive interval uses existing device buffers
and includes wrapper calls, synchronization and output allocation. This is not
a complete ephemeris or mission workflow measurement. Laptop clocks, thermals,
power policy and OS load were uncontrolled, and no statistical confidence
interval or universal speedup is claimed. The interleaved run is the appropriate
before/after comparison; historical absolute wall times differ between sessions.

Small batches continue to favor the CPU. One search case takes 2.77 microseconds
on the optimized scalar CPU versus 78.38 microseconds with GPU transfers.
At 1,024 mixed cases, CPU time is 0.701 ms versus 0.9036 ms with GPU transfers.
No automatic backend selection is introduced.

## Accuracy and failures

All 44 Python reference tests and 11 native integration tests pass. Both native
backends pass the complete existing validation without loosening tolerances:

- 3,541 independent scalar TOF/root cases and 1,411 derivative cases on C++ CPU.
- 440 full transfers per backend, including 144 actual-input multiprecision
  solutions and 248 independent conic/propagation cases.
- The three cached Horizons Mars 2020 boundary problems on both backends.
- Compute Sanitizer memcheck: zero errors and zero leaked bytes. Initcheck:
  zero errors, each over the 11-test native suite.

All reported worst errors match the previous build. Maximum scaled GPU velocity
error against the actual-input multiprecision references remains 9.551e-16.
The deliberately extreme full-transfer corpus retains one conditioning warning
and six unavailable precision diagnostics per backend. The Mars 2020 GPU
numerical propagation closure remains 0.669 mm in the Sun-only model.

Across the final timing corpus, all solves succeed. Every numerical result
field is exactly equal between old and optimized builds except
`root.evaluations`, which records the reduced work. This comparison treats NaNs
as equal and ignores the sign of zero; it is not a binary padding/NaN-payload
comparison. The comparison covers velocities, residuals, brackets, iteration
counters and precision flags on each backend. Maximum scaled CPU/GPU velocity
difference is 4.443e-15. Independent accuracy validation supplies the separate
correctness evidence.

## Reproduce

Use the validation dependencies and toolkit instructions from the
[seventh increment](seventh-increment.md). From the repository root, with a
trusted checkout and toolkit:

```bash
mkdir -p build/baseline-source
git archive bc01ed0 | tar -x -C build/baseline-source
python3 build/baseline-source/native/build.py --cuda-root /path/to/cuda --arch 120
python3 native/build.py --cuda-root /path/to/cuda --arch 120

python3 validation/compare_native.py \
  --library baseline build/baseline-source/build/native/liblambert_cuda.so \
  --library optimized build/native/liblambert_cuda.so \
  --sizes 1 32 1024 16384 65536 --json comparison.json

python3 -m unittest discover -s tests -v
LAMBERT_TEST_CUDA=1 python3 -m unittest discover -s native/tests -v
python3 validation/validate_native.py --gpu
```

Use the intended validation Python environment in those commands. An installed
Compute Sanitizer can run the native suite with `--tool memcheck --leak-check full`
or `--tool initcheck`, plus `--error-exitcode 99`. The comparison script accepts
`--threads`, `--rounds` and `--repetitions` for further controlled experiments.
Exploratory patches and register caps in the raw JSON can be rebuilt in separate
scratch directories using the recorded commands.

A useful next experiment is an optional compact output for large search batches.
The current full result transfers 240 bytes per problem and keeps every root
diagnostic. A compact path would need explicit status/precision semantics and
measured benefit while preserving the full diagnostic API. Hardware-counter
profiling, when available, can guide further register and series optimizations.
