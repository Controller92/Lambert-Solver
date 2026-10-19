"""Interleaved comparison of explicitly built CUDA libraries with the same ABI.

Each CUDA library also contains its scalar CPU implementation. Inputs, block
size, stopping options and outputs are matched. Kernel events exclude transfers;
wall pipeline timing includes upload, one launch and download on existing buffers.
No compilation, allocation, initial context setup or input generation is timed.
Libraries must be trusted local builds. No clocks or driver settings are changed.
"""
import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_native import NativeEngine, RESULT_DTYPE, _options
from src.lambert_root import RootOptions
from validation.benchmark_native import distribution, check


def field_differences(before, after, prefix=''):
    """Exact numerical comparison by field, with NaNs equal and signed zeros equal."""
    differences = {}
    for name in before.dtype.names:
        a, b = before[name], after[name]
        path = prefix+name
        if a.dtype.names:
            differences.update(field_differences(a, b, path+'.'))
        else:
            equal = (a == b)
            if a.dtype.kind == 'f':
                equal |= np.isnan(a) & np.isnan(b)
            count = int(np.count_nonzero(~equal))
            if count:
                differences[path] = count
    return differences


def summary(samples):
    return dict(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples),
                samples_ms=samples)


def compare(libraries, sizes, threads=128, rounds=7, repetitions=5):
    engines = {name: NativeEngine('cuda', path) for name, path in libraries}
    names = list(engines)
    metadata = {}
    for name, path in libraries:
        resource = engines[name].device_info(threads=threads)
        metadata[name] = dict(library=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              resources=resource)
        print(name, json.dumps(metadata[name]), flush=True)
    records = []
    for kind in ('circular', 'transfer_search', 'rotated_search', 'mixed'):
        for n in sizes:
            inputs = distribution(kind, n)
            outputs = {name: np.empty(n, dtype=RESULT_DTYPE) for name in names}
            options = _options(RootOptions())

            def cpu_call(name):
                code = engines[name].lib.lb_cpu(inputs.ctypes.data, outputs[name].ctypes.data,
                                                n, ct.byref(options))
                if code:
                    raise RuntimeError(f'{name}: C ABI CPU call failed: {code}')

            with ExitStack() as stack:
                batches = {name: stack.enter_context(engine.cuda_batch(n, threads=threads))
                           for name, engine in engines.items()}
                for name, batch in batches.items():
                    cpu_call(name)
                    batch.upload(inputs)
                    batch.run(repetitions=3)
                samples = {name: dict(cpu=[], kernel=[], pipeline=[]) for name in names}
                for turn in range(rounds):
                    offset = turn % len(names)
                    for name in names[offset:]+names[:offset]:
                        batch = batches[name]
                        start = time.perf_counter()
                        cpu_call(name)
                        samples[name]['cpu'].append((time.perf_counter()-start)*1000)
                        samples[name]['kernel'].append(batch.run(repetitions=repetitions))
                        start = time.perf_counter()
                        batch.upload(inputs)
                        batch.run()
                        batch.download()
                        samples[name]['pipeline'].append((time.perf_counter()-start)*1000)
                baseline_gpu = batches[names[0]].download()
                for name, batch in batches.items():
                    result = batch.download()
                    record = dict(distribution=kind, n=n, library=name,
                                  cpu=summary(samples[name]['cpu']),
                                  kernel=summary(samples[name]['kernel']),
                                  pipeline=summary(samples[name]['pipeline']),
                                  failures=int(np.count_nonzero(result['status'])),
                                  max_scaled_cpu_gpu_difference=check(outputs[name], result),
                                  max_scaled_baseline_gpu_difference=check(baseline_gpu, result),
                                  changed_gpu_fields=field_differences(baseline_gpu, result),
                                  changed_cpu_fields=field_differences(outputs[names[0]], outputs[name]),
                                  mean_tof_evaluations=float(result['root']['evaluations'].mean()),
                                  mean_iterations=float(result['root']['iterations'].mean()))
                    records.append(record)
                    print(json.dumps(record), flush=True)
    return dict(libraries=metadata, sizes=sizes, threads=threads, rounds=rounds,
                repetitions=repetitions, seed=958707, baseline=names[0],
                timing='Rotating library order; warmed CUDA events and separate wall/CPU samples. '
                       'CPU is scalar single-thread. Clocks/power are not controlled. '
                       'Kernel events include host submission gaps. Pipelines reuse buffers.',
                records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', nargs=2, action='append', metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1024, 65536])
    parser.add_argument('--threads', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    if len({name for name, _ in args.library}) != len(args.library):
        parser.error('Library names must be unique')
    if (any(n <= 0 for n in args.sizes) or not 1 <= args.threads <= 1024
            or args.rounds < 1 or not 1 <= args.repetitions < 2**31):
        parser.error('Invalid size, block size, round count or repetition count')
    libraries = [(name, Path(path).expanduser().resolve()) for name, path in args.library]
    result = compare(libraries, args.sizes, args.threads, args.rounds, args.repetitions)
    args.json.write_text(json.dumps(result, indent=2)+'\n')
