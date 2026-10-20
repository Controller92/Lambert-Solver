"""Check saved research grids against independent Gooding and preserved SI states.

The GUI/CLI runtime has no Gooding dependency. Validation requires lamberthub.
This checks the declared best-C3 policy across all candidate branches at sampled
cells, including minima, rather than merely comparing a selected velocity.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from lamberthub import gooding1990

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.config import from_dict
from src.research.grid import policies
from src.research.plot import read_run


def validate(path,samples=24):
    metadata=read_run(path)
    config=from_dict(metadata['config'])
    maxima=dict(relative_C3=0.,scaled_velocity=0.,vinf_consistency=0.)
    with np.load(path,allow_pickle=False) as data:
        nd,nt=data['C3'].shape
        n=nd*nt
        rng=np.random.default_rng(958712)
        indices=set(map(int,rng.choice(n,min(n,samples),replace=False)))
        if np.isfinite(data['C3']).any():indices.add(int(np.nanargmin(data['C3'])))
        indices.update((0,n-1))
        for index in sorted(indices):
            i,j=divmod(index,nt)
            dep,arr=data['dep_states'][i],data['arrival_states'][data['arrival_index'][i,j]]
            tof=float(data['tof_days'][j])*86400.
            r1,r2=dep[:3],arr[:3]
            hz=np.cross(r1,r2)[2]
            references=[]
            for m,branch,way in policies(config):
                long=way=='long' if way in ('short','long') else hz<0 if way=='prograde' else hz>0
                if hz==0 and way in ('prograde','retrograde'):continue
                # Gooding's low_path label is not assumed to equal our x branch.
                # Gather both roots once per orientation/M, then reduce all C3.
                if branch==1:continue
                for low in ((False,) if m==0 else (False,True)):
                    try:
                        v1,v2,it,_=gooding1990(float(metadata['mu_m3_s2']),r1,r2,tof,M=m,
                            prograde=bool((hz>0)!=bool(long)),low_path=low,maxiter=100,
                            atol=1e-13,rtol=1e-13,full_output=True)
                    except ValueError:
                        # Gooding raises for infeasible revolution counts.
                        # Never allow a rejected M=0 reference to pass quietly.
                        native_feasible=any(data['candidate_status'][k,i,j]==0 for k,(cm,cb,cw) in enumerate(policies(config)) if cm==m and cw==way)
                        if m==0 or native_feasible:raise
                        continue
                    if it>=100 or not np.isfinite([v1,v2]).all():raise AssertionError('Gooding failed')
                    references.append((float(np.dot(v1-dep[3:],v1-dep[3:]))/1e6,v1,v2))
            if not references:
                if np.isfinite(data['C3'][i,j]):raise AssertionError('Unexpected feasible grid cell')
                continue
            c3,v1,v2=min(references,key=lambda r:r[0])
            actual=float(data['C3'][i,j])
            if not np.isfinite(actual):raise AssertionError('Feasible cell was omitted')
            vel_scale=max(1.,np.linalg.norm(v1),np.linalg.norm(v2))
            # Energy ties can have different velocities. Compare to any tied
            # independent candidate rather than depending on tie ordering.
            tied=[r for r in references if abs(r[0]-c3)<=1e-10*max(1.,c3)]
            dv=min(max(np.linalg.norm(data['vinf_dep'][i,j]+dep[3:]-a),
                       np.linalg.norm(data['vinf_arr'][i,j]+arr[3:]-b))/vel_scale for _,a,b in tied)
            c3_error=abs(actual-c3)/max(1.,c3)
            consistency=abs(np.dot(data['vinf_dep'][i,j],data['vinf_dep'][i,j])/1e6-actual)/max(1.,actual)
            if c3_error>1e-9 or dv>2e-10 or consistency>2e-15:
                raise AssertionError(f'Independent grid failure: {index,c3_error,dv,consistency}')
            for key,value in zip(maxima,(c3_error,dv,consistency)):maxima[key]=max(maxima[key],float(value))
    result=dict(grid=str(path),sampled_cells=len(indices),maxima=maxima,
                total_cells=metadata['total_cells'],valid_cells=metadata['valid_cells'],
                backend=config.backend,orientation=config.orientation,
                max_revolutions=config.max_revolutions,total_seconds=metadata['total_seconds'])
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('grids',nargs='+',type=Path);p.add_argument('--samples',type=int,default=24)
    p.add_argument('--json',type=Path)
    a=p.parse_args()
    if a.samples<1:p.error('samples must be positive')
    results=[validate(path,a.samples) for path in a.grids]
    if a.json:a.json.write_text(json.dumps(results,indent=2)+'\n')
