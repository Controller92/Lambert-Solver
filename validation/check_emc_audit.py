"""Independent Gooding checks of saved EMC search populations and roots.

Also recomputes surface/altitude sensitivity without changing either nominal
Lambert leg. A named source hash ties every check to the preserved main results.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from lamberthub import gooding1990
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_emc import DAY,MARS_RADIUS,KERNEL_NAMES,patch,clean,norm
from src.research.ephemeris import SpiceEphemeris


def check(folder,kernels):
    report=json.loads((folder/'nominal.json').read_text());mu=report['solar_mu_m3_s2'];mars_mu=report['mars_mu_m3_s2']
    maxima=dict(sampled_velocity_error_m_s=0.,speed_root_velocity_error_m_s=0.,speed_root_mismatch_m_s=0.)
    rng=np.random.default_rng(105699587);counts={}
    with np.load(folder/'search-populations.npz') as f:
        for name in ('incoming','outgoing'):
            pop=f[name];valid=np.argwhere(pop['status']==0)
            selection=rng.choice(len(valid),min(128,len(valid)),replace=False)
            for selected in selection:
                im,j,k=map(int,valid[selected]);way,m,branch=f['policies'][k]
                if name=='incoming':
                    a=f['earth_states'][j];b=f['mars_states'][im];dt=f['mars_et'][im]-f['launch_et'][j]
                else:
                    a=f['mars_states'][im];b=f['ceres_states'][f['arrival_index'][im,j]];dt=f['tof_days'][j]*DAY
                pro=bool((np.cross(a[:3],b[:3])[2]>0)!=bool(way));refs=[]
                for low in ((False,) if m==0 else (False,True)):
                    v1,v2=gooding1990(mu,a[:3],b[:3],float(dt),M=int(m),prograde=pro,low_path=low,maxiter=100,atol=1e-13,rtol=1e-13)
                    refs.append(max(norm(pop[im,j,k]['v1']-v1),norm(pop[im,j,k]['v2']-v2)))
                error=min(refs);assert error<1e-5,error
                maxima['sampled_velocity_error_m_s']=max(maxima['sampled_velocity_error_m_s'],error)
            counts[name]=len(selection)
    roots=list(csv.DictReader((folder/'speed-equality-roots.csv').open()))
    best=max(range(len(roots)),key=lambda i:float(roots[i]['turn_margin_deg']))
    choices=set(map(int,rng.choice(len(roots),min(100,len(roots)),replace=False)));choices.add(best)
    sensitivity={str(alt):dict(max_turn_margin_deg=-180.,largest_inferred_periapsis_m=0.) for alt in (0.,100000.,200000.,500000.,1000000.)}
    with SpiceEphemeris([kernels/n for n in KERNEL_NAMES],'ECLIPJ2000',folder/'state-cache') as e:
        for index,row in enumerate(roots):
            # Re-evaluate all 439 roots with the independent Lambert algorithm,
            # not merely the sampled 100 selected for endpoint-component checks.
            times=[e.epoch(row[k]) for k in ('launch_tdb','mars_tdb','ceres_tdb')]
            states=[np.asarray(e.sp.spkezr(body,float(t),e.frame,'NONE','SUN')[0])*1000 for body,t in zip(('399','499','20000001'),times)]
            a,b,c=states
            pi=ast.literal_eval(row['incoming_policy'])
            po=ast.literal_eval(row['outgoing_policy'])
            def independent(x,y,dt,policy):
                way,m,branch=policy
                pro=bool((np.cross(x[:3],y[:3])[2]>0)!=bool(way))
                if m:raise AssertionError('Unexpected multi-revolution root; map Gooding roots explicitly before use')
                return gooding1990(mu,x[:3],y[:3],float(dt),M=0,prograde=pro,maxiter=100,atol=1e-13,rtol=1e-13)
            _,v2=independent(a,b,times[1]-times[0],pi);v1,_=independent(b,c,times[2]-times[1],po)
            vi=v2-b[3:];vo=v1-b[3:]
            error=abs(norm(vo)-norm(vi));maxima['speed_root_mismatch_m_s']=max(maxima['speed_root_mismatch_m_s'],error)
            assert error<1e-5,error
            for alt in (0.,100000.,200000.,500000.,1000000.):
                metrics=patch(vi,vo,mars_mu,alt)
                assert metrics['turn_margin_deg']<0
                sensitivity[str(alt)]['max_turn_margin_deg']=max(sensitivity[str(alt)]['max_turn_margin_deg'],metrics['turn_margin_deg'])
                sensitivity[str(alt)]['largest_inferred_periapsis_m']=max(sensitivity[str(alt)]['largest_inferred_periapsis_m'],metrics['turn_only_rp_at_incoming_speed_m'])
            if index in choices:
                np.testing.assert_allclose(norm(vi),float(row['vinf_in_m_s']),rtol=0,atol=1e-5)
                np.testing.assert_allclose(norm(vo),float(row['vinf_out_m_s']),rtol=0,atol=1e-5)
                metrics=patch(vi,vo,mars_mu)
                error=max(abs(metrics[k]-float(row[k])) for k in ('vinf_in_m_s','vinf_out_m_s'))
                maxima['speed_root_velocity_error_m_s']=max(maxima['speed_root_velocity_error_m_s'],error)
    nominal_sensitivity=[]
    for scenario in report['scenarios']:
        for row in scenario['mars_encounters']:
            if not row['prograde_pair']:continue
            table=[patch(np.array(row['vinf_in_vector_m_s']),np.array(row['vinf_out_vector_m_s']),mars_mu,alt) for alt in (0.,100000.,200000.,500000.,1000000.)]
            nominal_sensitivity.append(dict(scenario=scenario['name'],encounter=row['encounter_index'],table=table))
    output=dict(sampled_population_checks=counts,independent_gooding_speed_roots_checked=len(roots),
        sampled_root_magnitude_comparisons=len(choices),maxima=maxima,root_altitude_sensitivity=sensitivity,
        nominal_altitude_sensitivity=nominal_sensitivity,mars_reference_radius_m=MARS_RADIUS,
        source_sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('nominal.json','neighborhood.json','speed-equality-roots.csv','search-populations.npz')})
    (folder/'independent-checks.json').write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean({k:v for k,v in output.items() if k!='nominal_altitude_sensitivity'}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--kernel-dir',type=Path,default=ROOT/'data/kernels')
    a=p.parse_args();check(a.folder,a.kernel_dir)
