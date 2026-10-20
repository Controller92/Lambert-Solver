# Flyby encounter increment: validation and performance

This completes the Python reference, C++/CUDA encounter engine and saved CLI batch workflow. The mathematical/API contract and reproduction commands are in [flyby-solver.md](flyby-solver.md).

## Accuracy and tests

All 3,022 independent 100-digit cases passed on Python, C++ and the physical GPU. Cases include arbitrary 3D planes, unpowered inverse/forward encounters, tangential powered burns, and axis-aligned extreme turns with rp v² / GM from 1e-24 to 1e24.

| Quantity | Largest measured error |
|---|---:|
| Python outgoing velocity / speed | 1.023e-13 |
| C++ outgoing velocity / speed | 1.023e-13 |
| CUDA outgoing velocity / speed | 1.019e-13 |
| C++ periapsis radius, relative | 1.708e-13 |
| CUDA periapsis radius, relative | 1.710e-13 |
| Native periapsis position/velocity, relative | 1.781e-13 |
| Native powered delta-v, absolute | 2.030e-09 m/s |

83 cases carry the conservative inverse-conditioning warning. These are finite-grid observed errors, not universal accuracy guarantees. The complete numerical results are in [flyby-validation.json](../validation/results/flyby-validation.json).

16 new encounter/workflow tests pass, including independent central-force integration and orbital-element reconstruction. The existing 64 Python Lambert, 23 native CPU/CUDA and 13 research workflow tests also pass. CUDA Compute Sanitizer memcheck reports zero errors and zero leaked bytes; initcheck reports zero errors. Raw logs are in `validation/results/flyby-*.txt`.

## Actual mission comparison

Juno flew past Earth on 2013-10-09. The cached Horizons inputs are Earth-centred geometric ICRF states, in TDB, with exact requests and SHA-256 checks. Incoming/outgoing osculating asymptotes are inferred from finite states at 18:25 and 20:25 TDB. A five-second table samples closest approach separately.

| Comparison | Result |
|---|---:|
| Incoming excess speed | 10389.034946 m/s |
| Predicted two-body deflection | 40.677278° |
| Predicted periapsis altitude above equatorial reference radius | 554.248493 km |
| Closest five-second Horizons sample altitude on that same radius model | 554.405048 km |
| Predicted minus sampled periapsis radius | -156.556 m |
| Outgoing osculating asymptote difference | 3.045670 m/s |
| Same-conic analytical asymptote difference | 4.638e-12 m/s |
| Two-hour central-force propagation position difference | 15.650863 km |
| Two-hour central-force propagation velocity difference | 3.314220 m/s |

The asymptote comparison against another state in the same exact conic is a numerical consistency check. The comparison against the later Horizons state tests the physical approximation. The latter includes effects of non-central/multibody dynamics and finite sampling; it is not floating-point error. Equatorial-radius heights are not geodetic altitudes above the oblate Earth. This is a patched-conic model, not a navigation reconstruction.

## Performance

Measured on NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, 262,144 rows per population, 20 resident repetitions. Each homogeneous population contains successful encounters; mixed rows combine all three modes and every 101st row is deliberately invalid. CPU is compiled serial C++, not Python.

| Population | CPU solve | CUDA resident kernel | Upload + run + download | Resident throughput | Pipeline throughput |
|---|---:|---:|---:|---:|---:|
| forward | 41.697 ms | 1.716 ms | 16.999 ms | 152.79 M/s | 15.42 M/s |
| match | 71.197 ms | 2.313 ms | 17.200 ms | 113.35 M/s | 15.24 M/s |
| powered | 551.880 ms | 22.654 ms | 37.872 ms | 11.57 M/s | 6.92 M/s |
| mixed | 221.130 ms | 22.866 ms | 42.901 ms | 11.46 M/s | 6.11 M/s |

The pipeline uses persistent device allocations and pageable host arrays; it excludes allocation, object packing, JSON, ephemerides and disk. Its figure is the best of three measured pipeline runs. Resident kernel time is CUDA-event average time across 20 repetitions. These are encounter calculations, not complete transfer chains. Unpowered workloads are transfer-limited; powered root inversion takes up to 46 iterations in this population. Mixed warps inherit the slower powered branch. Group encounters by model when possible. No PyKEP performance claim is made.

The encounter kernel uses 155 registers and 72 bytes of local storage per thread, with 25% predicted occupancy at 128 threads/block. ptxas reports no register spills. Low occupancy and mixed-mode divergence leave room for a later optimization pass. All measurements and device metadata are in [flyby-benchmark.json](../validation/results/flyby-benchmark.json).

## Remaining research integration

Full date-window multi-leg searches must retain every relevant Lambert branch until encounter matching, enforce shared body/epoch/frame metadata, and add bounded candidate storage and resumable output. The new encounter solver supplies that stage’s validated numerical primitive; the old chain-search menu has not been revived.
