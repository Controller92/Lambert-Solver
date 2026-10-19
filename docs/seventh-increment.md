# Seventh increment: shared C++ engine, CUDA kernel and Python batches

This increment ports the validated M=0 solver at `d2e3775` to C++17, executes
the same mathematical source on the CPU and GPU, and provides a minimal NumPy
interface. It includes an initial measured baseline. The existing Python core
is unchanged and remains the numerical reference. There is no legacy GUI
integration, ephemeris access inside the kernel, or multirevolution solver.

## Architecture and mathematical fidelity

| File | Responsibility |
| --- | --- |
| native/lambert.hpp | Fixed-size scalar math: geometry, TOF, derivatives, starter, safeguarded root, velocities |
| native/api.h and native/cpu.cpp | Versioned plain-struct ABI and single-thread CPU batch entry points |
| native/cuda.cu | One independent solve per thread; host-owned device buffers, stream, timing events and errors |
| src/lambert_native.py | NumPy layouts, ABI checks, input packing and owned CPU/CUDA batch operations |
| native/build.py | Explicit builds with recorded compiler flags and versions |
| validation/validate_native.py | Independent mathematical, physical and mission-state accuracy validation |
| validation/benchmark_native.py | Warmed kernel, transfer-inclusive and synthetic workflow measurements |

`LB_HD` marks mathematical functions for host and device compilation. Each
thread receives its own scalar state. There is no device heap allocation,
inter-thread communication or hidden solver state. Allocation and transfers
occur in the CUDA host layer, outside the mathematics. One problem per thread
and the initial array-of-structs layout are measured starting points, not a
claim that either is optimal.

The C++ code retains q=c/s from geometry through every TOF, derivative, starter,
root and reconstruction call. It preserves the near-parabolic series, bounded
series budgets, bisection safeguards, actual update counts, fresh residual/step
tests and tri-state conditioning diagnostic. Velocity failures return no usable
endpoint velocities. The math is based on the published
[Izzo reference](https://doi.org/10.1007/s10569-014-9587-y) already checked directly
in the earlier increments; no external Lambert implementation is called by the engine.

Two deliberate implementation adaptations are independently checked:

- Bounded short floating-point expansions replace Python's math.fsum for
  three/four terms. Since the eighth increment, two-term sums use one guarded
  binary64 addition. Explicit
  FMA product remainders implement compensated transfer-plane products. Only
  these explicit FMAs are used; compiler contraction is disabled elsewhere.
- Geometrically q<=1. Near antiparallel directions, device hypot rounding can
  produce c/s one ulp above one. The C++ geometry bounds q at one, matching the
  existing treatment of sigma and lambda. An actual failing GPU input motivated
  this correction; a regression test and multiprecision physical checks cover it.

CPU and GPU math libraries need not return bit-identical roots, velocities,
iteration paths or evaluation counts. Required stopping tests, error bounds,
statuses and truthful counters are checked independently of that agreement.
The 16-ulp conditioning model remains a calibrated diagnostic, not an interval
certificate. Genuine flat long-arc curves can still trigger warnings, and
higher-derivative range limits can leave precision unknown.

## Physical contract and result interpretation

Inputs remain three-component positions in metres, positive elapsed dynamical
seconds, and positive mu in m^3/s^2. Both endpoints must use the same central
body, origin and right-handed inertial axes. The host handles ephemerides, time
conversion and frame/origin conversion. Output velocities are m/s.

`long_way=False` selects an angle below pi with normal along r1 cross r2;
`True` selects the angle above pi with the opposite normal. This is not a
fixed-axis prograde flag. Only M=0 is implemented. Exact collinear geometry
and sin(angle)<=64*binary64_epsilon are still explicitly rejected.

Input rows occupy 72 bytes, output rows 240 bytes. The public NumPy dtypes use
aligned native-endian binary64/int32/int64 fields. The library checks ABI
version and struct sizes before use. Inputs requiring alignment/contiguity are
copied safely. Optional CPU output arrays must be separate, writable, aligned
and contiguous. Malformed Python shapes/types raise before calling native code;
numerical input and solve failures retain individual row statuses.

Native result status values match the Python LambertStatus and GeometryStatus.
`root.status=-1` means root solving was not attempted because geometry failed.
Otherwise it matches RootStatus. `root.precision_limited` is 0=False, 1=True,
-1=unknown. Failed roots and failed velocities use NaNs where Python uses None.
Inspect status before consuming numerical fields; NaNs are never substitute
velocities. Root residual and tolerances remain dimensionless. Counts are int64;
native option budgets and revolution counts must fit int32.

```python
import math
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine, pack_problems

radius, mu = 7e6, 3.986004418e14
quarter = (math.pi/2)*radius/math.sqrt(mu/radius)
inputs = pack_problems([
    LambertInput((radius, 0., 0.), (0., radius, 0.), quarter, mu),
    LambertInput((radius, 0., 0.), (0., radius, 0.), 3*quarter, mu, long_way=True),
])

cpu = NativeEngine('cpu')
cpu_result = cpu.cpu(inputs)

gpu = NativeEngine('cuda')
with gpu.cuda_batch(len(inputs)) as batch:
    batch.upload(inputs)
    kernel_ms = batch.run()
    result = batch.download()
    # No allocation or host/device transfer during another resident run:
    repeated_kernel_ms = batch.run(repetitions=10)

ok = (result['status'] == 0) & (result['root']['precision_limited'] == 0)
print(result['v1'][ok], result['v2'][ok])
print(result['root']['iterations'], result['root']['residual'])
```

For large workflows, fill an INPUT_DTYPE array with vectorized NumPy operations
instead of constructing Python objects per row. Packing, ephemerides and application
logic can cost more than the kernel. The buffer context supports repeated device
use, but this first interface does not expose external CuPy/PyTorch device arrays.

`CudaBatch` owns its buffers, stream and events. Use a context manager or explicit
close. Upload invalidates old results; download requires a successful run after
that upload. Calls on one batch are serialized with a lock. Capacity violations,
closed handles and unsupported block sizes fail explicitly. CUDA runtime errors
raise and never trigger a CPU fallback. No cudaDeviceReset is used, so other GPU
users in the process are not reset. The current device is selected for each
operation so independent batches can coexist safely.

## Build and reproduce

The implementation was built on Pop!_OS 24.04 with GCC 13.3 and NVIDIA CUDA
12.8.1 (nvcc 12.8.93, runtime 12.8.90). There was no preinstalled nvcc. The
session installed official NVIDIA redistributables into the task's `work/`
directory and verified SHA-256 values against the
[NVIDIA redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
No system packages or drivers were replaced. The toolkit and built binaries
are not committed to the repository.

From the repository root:

```bash
# CPU-only build requires a C++17 compiler:
python3 native/build.py
# Build CPU and CUDA libraries for this Blackwell GPU:
python3 native/build.py --cuda-root /path/to/cuda --arch 120

# The existing validation environment supplies NumPy and independent references:
python3 -m venv .venv-validation
.venv-validation/bin/python -m pip install -r validation/requirements.txt
python3 -m unittest discover -s tests -v
LAMBERT_TEST_CUDA=1 .venv-validation/bin/python -m unittest discover -s native/tests -v
.venv-validation/bin/python validation/validate_native.py --gpu
.venv-validation/bin/python validation/benchmark_native.py --json benchmark.json
```

The local session's CUDA root is `../../work/cuda-12.8.1/toolkit` relative to
this checkout. Use the installed toolkit root on another machine and compile
for its architecture. This first build/binding path is tested on Linux x86-64;
Windows packaging is not implemented. NumPy is the only dependency of the native
Python interface. mpmath, SciPy and LambertHub are validation-only dependencies.

Builds disable fast math, implicit contraction and flush-to-zero. CUDA division
and square roots use precise settings. SASS sm_120 and PTX compute_120 are
included. Runtime CUDA is linked statically; the installed NVIDIA driver is still
required. Compiler versions, commands and build durations are recorded in
build/native/build.json. Shared objects are replaced atomically so building
does not truncate a library mapped by a running process. Restart a Python process
to load rebuilt libraries. No build or CPU emulation happens implicitly on import.

## Validation and profiling results

The 44 Python reference tests and 11 explicit native integration tests pass.
The latter include actual GPU tail batches (257 rows), empty inputs, buffer reuse,
rejected geometry, exhausted root budgets, unknown precision, malformed layouts,
input alignment, writable output reuse, closed contexts, independent contexts,
and the nearly antiparallel device-rounding regression. NVIDIA Compute Sanitizer
memcheck reports zero errors and zero leaked bytes; initcheck also reports zero
errors. These checks are bounded evidence, not an exhaustive proof for all inputs.

The independent scalar C++ corpus has 3,541 TOF/root cases and 1,411 derivative
cases. Maximum relative T error is 2.248e-15. Maximum relative derivative errors
are 7.734e-15, 6.783e-14 and 6.260e-13. All original acceptance bounds pass.
The largest scaled inverse error, 6.467e-12, is on an already diagnosed flat
long-arc curve and is accepted under its explicit conditioning estimate, with
the warning required beyond the ordinary 5e-14 bound. There are 26 such/conservative
warnings and 88 unavailable diagnostics across this deliberately extreme scalar
corpus. The measured forward roundoff maximum is 14.397 ulps, below the retained
16-ulp diagnostic allowance.

Both CPU and GPU execute 440 complete transfer problems: 248 independently
generated conics, 144 nearly collinear actual-input multiprecision references,
and 48 additional extreme full-solve checks, including hyperbolic x up to 1e150.
Each backend is checked against the independent conic states, 240 strict Gooding
comparisons, invariant/direction checks, 248 refined two-body propagations, and
144 multiprecision solutions for the actual received input numbers. The extra
48 range cases compare with the independently validated Python core. Each backend
has one conditioning warning and six unavailable diagnostics on those extreme
full-solve inputs. No unexpected failures remain.

The native Mars 2020 check reuses the cached official Horizons data for three
boundary problems. It verifies the same Sun-only mathematical problem and does
not alter the model to fit the real spacecraft trajectory. The mission-model
limitations described in [the Mars 2020 report](mars2020-validation.md) still apply.

Numerical tables and measured timing results are recorded in
[the native results report](native-results.md).

## Next performance work

The [eighth increment](eighth-increment.md) completes the first optimization pass:
TOF reuse and simpler two-term sums reduce search kernel time by 17.9%, with
unchanged independent accuracy bounds. The resource and timing descriptions
below record this increment's original baseline.

The baseline uses 196 registers per thread, a 32-byte stack/local frame, zero
compiler-reported spill stores/loads and no shared memory. At 128 threads per
block, CUDA predicts two resident blocks per SM, or 16.7% occupancy on this GPU.
This is theoretical resource occupancy, not a hardware utilization measurement.
The compiled kernel's block-size limit is 256 threads for this build.

Block sizes 32, 64, 128 and 256 are measured rather than assuming one optimum.
The benchmark reports an iteration-count proxy for warp imbalance, explicitly
not a hardware branch-efficiency counter. Register live ranges, the short-sum
local frame, series/transcendental work, array-of-structs traffic, pageable host
transfers and the full 240-byte diagnostic result are concrete candidates for
profiling and tuning. Nsight hardware-counter profiling has not yet been done.
Reducing precision or enabling fast math is not justified by this increment.

Device-resident reuse is already functional. Further optimization should retain
the independent accuracy suite and report throughput, failures and accuracy for
each representative distribution. No uniform speedup is claimed for small
batches, other hardware, full mission workflows or a multicore CPU baseline.
