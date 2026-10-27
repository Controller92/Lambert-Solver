"""Matched optional-Jacobian overhead; scalar C++ CPU and warmed CUDA.

CUDA kernel timing excludes allocation/transfers. Pipeline includes upload,
solve and downloads of both outputs for Jacobians. No multicore or legacy
solver comparison is implied. Saved seeds match existing native benchmarks.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from benchmark_native import distribution
from benchmark_multirev import inputs
from src.lambert_native import NativeEngine,RESULT_DTYPE,JACOBIAN_DTYPE


def benchmark(n):
    cpu,gpu=NativeEngine(),NativeEngine('cuda')
    records=[]
    for name,rows in (('zero-revolution rotated search',distribution('rotated_search',n)),
                      ('multi-revolution M=1..5 one branch per row',inputs(n))):
        expected,ej=cpu.cpu_with_jacobian(rows)
        if np.any(expected['status']) or np.any(ej['status']):raise AssertionError('Unexpected benchmark failure')
        with gpu.cuda_batch(n) as b:
            b.upload(rows);b.run(jacobian=True,repetitions=2)
            for enabled in (False,True):
                cpu_ms=[]
                for _ in range(3):
                    start=time.perf_counter()
                    cpu.cpu_with_jacobian(rows) if enabled else cpu.cpu(rows)
                    cpu_ms.append((time.perf_counter()-start)*1000)
                kernel=[b.run(jacobian=enabled,repetitions=5) for _ in range(5)]
                pipeline=[]
                for _ in range(3):
                    start=time.perf_counter();b.upload(rows);b.run(jacobian=enabled);actual=b.download()
                    if enabled:jac=b.download_jacobian()
                    pipeline.append((time.perf_counter()-start)*1000)
                np.testing.assert_array_equal(actual['status'],expected['status'])
                np.testing.assert_allclose(actual['v1'],expected['v1'],rtol=3e-12,atol=3e-8)
                error=None
                if enabled:
                    np.testing.assert_array_equal(jac['status'],ej['status'])
                    error=float(np.max(abs(jac['values']-ej['values'])/(1e-6+abs(ej['values']))))
                    if error>2e-10:raise AssertionError(error)
                record=dict(distribution=name,rows=n,jacobian=enabled,cpu_scalar_ms=statistics.median(cpu_ms),
                    kernel_ms=statistics.median(kernel),pipeline_ms=statistics.median(pipeline),
                    kernel_samples_ms=kernel,pipeline_samples_ms=pipeline,
                    kernel_rows_per_second=n*1000/statistics.median(kernel),max_scaled_cpu_gpu_jacobian_difference=error)
                records.append(record);print(json.dumps(record),flush=True)
    return dict(device=gpu.device_info(),rows=n,result_bytes=RESULT_DTYPE.itemsize,jacobian_bytes=JACOBIAN_DTYPE.itemsize,
                cpu='single-thread scalar C++, NumPy allocation included',records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=int,default=32768);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.rows<1:p.error('rows must be positive')
    args.output.write_text(json.dumps(benchmark(args.rows),indent=2)+'\n')
