"""Reproducible native M>0 batch timing, with a scalar one-thread CPU baseline.

CUDA events exclude transfers and allocation. Pipeline wall times include the
host dispatch scan, upload, both kernels for mixed rows, and download, reusing
allocated buffers. Generation, compilation and initialization are excluded.
A row is one branch solution, not a complete pair. Clocks are uncontrolled.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lambert_native import NativeEngine
from benchmark_native import distribution, check


def inputs(n,mixed=False):
    rows=distribution('rotated_search',n)
    rng=np.random.default_rng(958711)
    rows['revolutions']=rng.integers(1,6,n)
    rows['branch']=rng.integers(0,2,n)
    scale=1.495978707e11/math.sqrt(1.3271244004193937e20/1.495978707e11)
    rows['tof']=(rows['revolutions']+2)*4*math.pi*scale
    if mixed:
        rows['revolutions'][::2]=0
        rows['tof'][::2]=200*86400
    return rows


def benchmark(sizes):
    cpu,gpu=NativeEngine(),NativeEngine('cuda')
    records=[]
    for mixed in (False,True):
        for n in sizes:
            rows=inputs(n,mixed)
            reference=cpu.cpu(rows)
            cpu_times=[]
            for _ in range(3):
                begin=time.perf_counter();cpu.cpu(rows);cpu_times.append((time.perf_counter()-begin)*1000)
            with gpu.cuda_batch(n) as b:
                b.upload(rows);b.run(repetitions=2)
                kernel=[b.run(repetitions=5) for _ in range(5)]
                pipeline=[]
                for _ in range(3):
                    begin=time.perf_counter();b.upload(rows);b.run();actual=b.download()
                    pipeline.append((time.perf_counter()-begin)*1000)
                error=check(reference,actual)
            record=dict(rows=n,distribution='mixed M=0/M>0' if mixed else 'M=1..5 both branches',
                        cpu_ms=statistics.median(cpu_times),kernel_ms=statistics.median(kernel),
                        pipeline_ms=statistics.median(pipeline),kernel_samples_ms=kernel,
                        pipeline_samples_ms=pipeline,max_scaled_cpu_gpu_difference=error,
                        failures=int(np.count_nonzero(actual['status'])),
                        mean_minimum_iterations=float(actual['minimum']['iterations'].mean()),
                        mean_root_iterations=float(actual['root']['iterations'].mean()))
            record['kernel_branch_solutions_per_second']=n*1000/record['kernel_ms']
            records.append(record);print(json.dumps(record),flush=True)
    return dict(seed=958711,device=gpu.device_info(multi=True),zero_device=gpu.device_info(),
                cpu='scalar single-thread C++',records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',type=int,nargs='+',default=[1024,32768])
    p.add_argument('--json',type=Path,required=True)
    a=p.parse_args()
    if any(n<1 for n in a.sizes):p.error('sizes must be positive')
    a.json.write_text(json.dumps(benchmark(a.sizes),indent=2)+'\n')
