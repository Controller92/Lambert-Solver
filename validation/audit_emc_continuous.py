"""Continuous local-window EMC search; stochastic evidence, not a certificate.

Three DE seeds per prograde/retrograde leg combination. Inputs remain exactly
inside the main audit's date box. All evaluations use actual SPICE states and
the native CPU engine; final points are recomputed with independent Gooding.
The objective is an optimistic asymptotic correction after an unpowered flyby,
not a general minimum maneuver cost. A zero would identify a ballistic match.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution
from lamberthub import gooding1990
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_emc import DAY,MARS_RADIUS,KERNEL_NAMES,patch,clean,iso,norm
from src.research.ephemeris import SpiceEphemeris
from src.lambert_native import NativeEngine,INPUT_DTYPE


def check(folder,kernels):
    started=time.perf_counter();engine=NativeEngine('cpu');results=[]
    with SpiceEphemeris([kernels/n for n in KERNEL_NAMES],'ECLIPJ2000',folder/'state-cache') as e:
        start=e.epoch('2035-07-01');center=e.epoch('2036-01-17')
        mu=float(e.sp.bodvrd('MARS','GM',1)[1][0])*1e9
        rows=np.zeros(2,dtype=INPUT_DTYPE);rows['mu']=e.mu
        for pro_in,pro_out in ((True,True),(True,False),(False,True),(False,False)):
            def evaluate(x,detail=False):
                t0=start+x[0]*DAY;t1=center+x[1]*DAY;t2=t1+x[2]*DAY
                state=[]
                for body,t in [('399',t0),('499',t1),('20000001',t2)]:
                    state.append(np.asarray(e.sp.spkezr(body,float(t),e.frame,'NONE','SUN')[0])*1000)
                a,b,c=state
                rows['r1']=[a[:3],b[:3]];rows['r2']=[b[:3],c[:3]];rows['tof']=[t1-t0,t2-t1]
                rows['long_way']=[bool(np.cross(a[:3],b[:3])[2]<0) if pro_in else bool(np.cross(a[:3],b[:3])[2]>0),
                    bool(np.cross(b[:3],c[:3])[2]<0) if pro_out else bool(np.cross(b[:3],c[:3])[2]>0)]
                out=engine.cpu(rows)
                if (out['status']!=0).any():raise ValueError('Unexpected M=0 failure')
                vi=out[0]['v2']-b[3:];vo=out[1]['v1']-b[3:]
                A=norm(vi);B=norm(vo);delta=np.arctan2(norm(np.cross(vi,vo)),float(np.dot(vi,vo)))
                maximum=2*np.arcsin(1/(1+(MARS_RADIUS+200000)*A*A/mu))
                score=float(np.hypot(B-A,2*np.sqrt(A*B)*np.sin(max(0.,delta-maximum)/2)))
                if not detail:return score
                independent=[]
                for aa,bb,dt,pro in ((a,b,t1-t0,pro_in),(b,c,t2-t1,pro_out)):
                    independent.append(gooding1990(e.mu,aa[:3],bb[:3],float(dt),M=0,prograde=pro,maxiter=100,atol=1e-13,rtol=1e-13))
                errors=[max(norm(out[i]['v1']-v1),norm(out[i]['v2']-v2)) for i,(v1,v2) in enumerate(independent)]
                assert max(errors)<1e-5,errors
                return dict(launch_tdb=iso(t0),mars_tdb=iso(t1),ceres_tdb=iso(t2),
                    incoming_prograde=pro_in,outgoing_prograde=pro_out,earth_departure_C3_km2_s2=float(np.dot(out[0]['v1']-a[3:],out[0]['v1']-a[3:])/1e6),
                    gooding_max_endpoint_velocity_error_m_s=max(errors),states=state,
                    incoming_v1_m_s=out[0]['v1'],incoming_v2_m_s=out[0]['v2'],
                    outgoing_v1_m_s=out[1]['v1'],outgoing_v2_m_s=out[1]['v2'],**patch(vi,vo,mu))
            for seed in (9587,2036,4193):
                r=differential_evolution(evaluate,[(-60.,60.),(-60.,60.),(160.,340.)],seed=seed,
                    popsize=12,maxiter=120,tol=1e-9,atol=1e-5,polish=True,workers=1)
                item=evaluate(r.x,True);item.update(seed=seed,optimizer_success=bool(r.success),
                    evaluations=int(r.nfev),objective_m_s=float(r.fun),optimizer_message=str(r.message))
                results.append(item);print(pro_in,pro_out,seed,r.fun,flush=True)
    output=dict(runs=results,minimum=min(results,key=lambda r:r['objective_m_s']),seconds=time.perf_counter()-started,
        limitation='Repeated stochastic searches of the stated finite box; not a global proof. M=0 only. Objective is an asymptotic patch correction, not optimized powered-flyby delta-v.')
    (folder/'continuous-search.json').write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--kernel-dir',type=Path,default=ROOT/'data/kernels')
    a=p.parse_args();check(a.folder,a.kernel_dir)
