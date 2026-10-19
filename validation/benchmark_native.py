"""Warmed native CPU/CUDA baseline with resident and transfer-inclusive timing.

This is a scalar single-thread C++ CPU baseline, not a multicore comparison.
CUDA events measure a stream of launches, including any submission gaps, but
excluding allocation, compilation, uploads and downloads. Wall timing reports
upload+one run+download with existing device allocations. Seeds are fixed.
"""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_native import NativeEngine, INPUT_DTYPE, RESULT_DTYPE, _options
from src.lambert_root import RootOptions


def distribution(name, n):
    rng = np.random.default_rng(958707)
    rows = np.zeros(n, dtype=INPUT_DTYPE)
    radius, mu = 1.495978707e11, 1.3271244004193937e20
    rows['mu'] = mu
    rows['r1'][:, 0] = radius
    if name == 'circular':
        rows['r2'][:, 1] = radius
        rows['tof'] = math.pi/2*radius/math.sqrt(mu/radius)
    else:
        angle = rng.uniform(.2, 3., n)
        factor = rng.uniform(.8, 1.7, n)
        if name == 'mixed':
            angle[::8] = 1e-12
            rows['long_way'] = rng.integers(0, 2, n)
            rows['tof'] = 86400*10**rng.uniform(-2, 4, n)
        else:
            rows['long_way'] = rng.random(n) < .1
            rows['tof'] = rng.uniform(100, 350, n)*86400
        rows['r2'][:, 0] = radius*factor*np.cos(angle)
        rows['r2'][:, 1] = radius*factor*np.sin(angle)
        rows['r2'][:, 2] = radius*factor*.03*np.sin(angle)
        if name == 'rotated_search':
            # Rotate each complete transfer into a general 3D frame. This
            # exposes math-library costs hidden by axis-aligned starting states.
            azimuth, tilt = rng.uniform(-math.pi, math.pi, n), rng.uniform(-.2, .2, n)
            cp, sp, ci, si = np.cos(azimuth), np.sin(azimuth), np.cos(tilt), np.sin(tilt)
            for field in ('r1', 'r2'):
                v = rows[field].copy()
                x, y = v[:, 0]*cp-v[:, 1]*sp, v[:, 0]*sp+v[:, 1]*cp
                rows[field][:, 0] = x
                rows[field][:, 1] = y*ci-v[:, 2]*si
                rows[field][:, 2] = y*si+v[:, 2]*ci
    return rows


def check(cpu, gpu):
    if not np.array_equal(cpu['status'], gpu['status']):
        raise AssertionError('CPU/GPU benchmark status mismatch')
    if np.any(gpu['status']):
        raise AssertionError('Benchmark distribution unexpectedly contains failed solves')
    scale = np.maximum(1., np.maximum(np.linalg.norm(cpu['v1'], axis=1), np.linalg.norm(cpu['v2'], axis=1)))
    error = np.maximum(np.linalg.norm(cpu['v1']-gpu['v1'], axis=1), np.linalg.norm(cpu['v2']-gpu['v2'], axis=1))/scale
    maximum = float(np.max(error)) if len(error) else 0.
    if maximum > 2e-11:
        raise AssertionError(f'Benchmark CPU/GPU accuracy mismatch: {maximum}')
    return maximum


def benchmark(sizes, scan_size):
    initialized = time.perf_counter()
    cpu, gpu = NativeEngine(), NativeEngine('cuda')
    info = gpu.device_info()
    initialization_ms = (time.perf_counter()-initialized)*1000
    print('GPU:', info, flush=True)
    print('First library load and device query/context initialization ms:', initialization_ms, flush=True)
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
    print('CPU baseline: one thread,', cpu_model, flush=True)
    records = []
    for name in ('circular', 'transfer_search', 'rotated_search', 'mixed'):
        for n in sizes:
            rows = distribution(name, n)
            reference = np.empty(n, dtype=RESULT_DTYPE)
            options = _options(RootOptions())

            def cpu_call():
                status = cpu.lib.lb_cpu(rows.ctypes.data, reference.ctypes.data, n, ct.byref(options))
                if status:
                    raise RuntimeError('CPU benchmark call failed')

            cpu_call()
            cpu_samples = []
            for _ in range(3 if n >= 16384 else 7):
                start = time.perf_counter()
                cpu_call()
                cpu_samples.append((time.perf_counter()-start)*1000)
            cpu_ms = statistics.median(cpu_samples)
            with gpu.cuda_batch(n) as batch:
                batch.upload(rows)
                batch.run(repetitions=2)
                repetitions = max(5, min(50, 327680//n))
                kernel_ms = statistics.median(batch.run(repetitions=repetitions) for _ in range(3))
                pipeline, upload, download = [], [], []
                for _ in range(3):
                    start = time.perf_counter()
                    batch.upload(rows)
                    copied = time.perf_counter()
                    batch.run()
                    finished = time.perf_counter()
                    output = batch.download()
                    stop = time.perf_counter()
                    pipeline.append((stop-start)*1000)
                    upload.append((copied-start)*1000)
                    download.append((stop-finished)*1000)
                maximum = check(reference, output)
            # Synthetic full workflow, including vectorized input generation,
            # fresh device allocations, upload, solve, download and cleanup.
            # The library/context is already initialized; ephemerides are absent.
            workflows = []
            for _ in range(3):
                start = time.perf_counter()
                prepared = distribution(name, n)
                with gpu.cuda_batch(n) as fresh:
                    fresh.upload(prepared)
                    fresh.run()
                    fresh.download()
                workflows.append((time.perf_counter()-start)*1000)
            iterations = output['root']['iterations']
            full_warps = n//32
            utilization = None
            if full_warps:
                warp = iterations[:full_warps*32].reshape(-1, 32)
                utilization = float(warp.sum()/(32*warp.max(axis=1).sum()))
            record = dict(distribution=name, n=n, cpu_ms=cpu_ms, kernel_ms=kernel_ms,
                          pipeline_ms=statistics.median(pipeline), upload_wall_ms=statistics.median(upload),
                          workflow_ms=statistics.median(workflows),
                          download_wall_ms=statistics.median(download), kernel_solves_per_second=n/(kernel_ms/1000),
                          cpu_over_kernel=cpu_ms/kernel_ms, cpu_over_pipeline=cpu_ms/statistics.median(pipeline),
                          failures=int(np.count_nonzero(output['status'])), max_scaled_cpu_gpu_difference=maximum,
                          mean_iterations=float(iterations.mean()), max_iterations=int(iterations.max()),
                          warp_iteration_utilization=utilization)
            records.append(record)
            print(json.dumps(record), flush=True)
    scans = []
    for name in ('transfer_search', 'mixed'):
        rows = distribution(name, scan_size)
        for threads in (32, 64, 128, 256):
            with gpu.cuda_batch(scan_size, threads=threads) as batch:
                batch.upload(rows)
                batch.run(repetitions=2)
                ms = statistics.median(batch.run(repetitions=10) for _ in range(3))
            entry = dict(distribution=name, n=scan_size, threads=threads, kernel_ms=ms,
                         resources=gpu.device_info(threads=threads))
            scans.append(entry)
            print('BLOCK_SCAN', json.dumps(entry), flush=True)
    return dict(gpu=info, initialization_ms=initialization_ms, cpu=cpu_model, cpu_threads=1,
                records=records, block_scan=scans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1, 32, 1024, 16384, 65536])
    parser.add_argument('--scan-size', type=int, default=16384)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or args.scan_size <= 0:
        parser.error('sizes must be positive')
    result = benchmark(args.sizes, args.scan_size)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2)+'\n')
