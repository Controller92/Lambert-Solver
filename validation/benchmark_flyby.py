"""Time CPU, resident CUDA kernel and persistent upload/run/download separately.

Not a full mission search or disk/JSON benchmark. No Python reference timing is
used to inflate speedups. Correctness is checked outside the timed operations.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.flyby import Encounter,Mode,solve
from src.flyby_native import FlybyEngine,pack_encounters


def benchmark(count,repetitions):
    if count<1 or repetitions<1:raise ValueError('Counts must be positive')
    engine=FlybyEngine('cuda');p=Encounter((10000.,1000.,200.),3.98600435436e14,6.6e6,1e11,rp=1e7,azimuth=.4)
    f=solve(p);single=pack_encounters([p,replace(p,mode=Mode.MATCH,vout=f.vout),
        replace(p,mode=Mode.POWERED,vout=tuple(np.asarray(f.vout)*1.2))])
    report=dict(count=count,repetitions=repetitions,device=engine.device_info(),
                note='Persistent buffers, pageable host transfers; pipeline excludes allocation, Python object packing, JSON, ephemerides and disk.',runs={})
    for name,index in [('forward',0),('match',1),('powered',2),('mixed',None)]:
        rows=np.tile(single[index:index+1],count) if index is not None else np.resize(single,count)
        # Deterministic azimuth variation and a few invalid rows also exercise
        # early exits in the mixed population; homogeneous batches stay feasible.
        if index==0:rows['azimuth']=np.linspace(-np.pi,np.pi,count)
        if index is None:rows['mu'][::101]=-1
        before=time.perf_counter();cpu=engine.cpu(rows);cpu_s=time.perf_counter()-before
        with engine.cuda_batch(count) as b:
            b.upload(rows);b.run();ms=b.run(repetitions);out=b.download()
            times=[]
            for _ in range(3):
                before=time.perf_counter();b.upload(rows);b.run();out=b.download();times.append(time.perf_counter()-before)
        np.testing.assert_array_equal(cpu['status'],out['status'])
        valid=out['status']==0
        if index is not None:assert valid.all(),(name,np.unique(out['status'],return_counts=True))
        else:assert valid.sum()==count-len(rows[::101]),'Unexpected mixed failures'
        np.testing.assert_allclose(cpu['vout'][valid],out['vout'][valid],rtol=1e-11,atol=1e-8)
        report['runs'][name]=dict(cpu_seconds=cpu_s,kernel_ms=ms,pipeline_seconds=min(times),
            kernel_encounters_per_second=count/(ms*.001),pipeline_encounters_per_second=count/min(times),
            cpu_to_pipeline_speedup=cpu_s/min(times),successes=int(valid.sum()),
            max_iterations=int(out['iterations'].max()))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--count',type=int,default=262144)
    ap.add_argument('--repetitions',type=int,default=20);ap.add_argument('--output',type=Path,default=ROOT/'validation/results/flyby-benchmark.json')
    a=ap.parse_args();report=benchmark(a.count,a.repetitions)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
