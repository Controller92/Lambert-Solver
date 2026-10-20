"""Independent 100-digit conics, CPU/CUDA parity and cached Juno encounter.

Ordinary runs are offline. --refresh fetches public Horizons data and preserves
full requests, headers and hashes. Real-mission discrepancies are physical model
errors, distinct from floating-point error against exact two-body mathematics.
"""
import argparse
import csv
from dataclasses import replace
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.flyby import Encounter,Mode,Status,solve
from src.flyby_native import FlybyEngine,pack_encounters
DATA=ROOT/'validation/data/juno-earth-flyby'
PARAMS=dict(COMMAND="'-61'",OBJ_DATA="'YES'",MAKE_EPHEM="'YES'",EPHEM_TYPE="'VECTORS'",
    CENTER="'500@399'",REF_PLANE="'FRAME'",REF_SYSTEM="'ICRF'",OUT_UNITS="'KM-S'",
    VEC_TABLE="'2'",VEC_CORR="'NONE'",CSV_FORMAT="'YES'",TIME_TYPE="'TDB'",
    START_TIME="'2013-10-09 06:00:00'",STOP_TIME="'2013-10-10 06:00:00'",STEP_SIZE="'5 m'")

FINE=dict(PARAMS,START_TIME="'2013-10-09 19:15:00'",STOP_TIME="'2013-10-09 19:30:00'",STEP_SIZE="'180'")

def refresh():
    DATA.mkdir(parents=True,exist_ok=True)
    for name,params in [('encounter',PARAMS),('perigee',FINE),('earth',dict(COMMAND="'399'",OBJ_DATA="'YES'",MAKE_EPHEM="'NO'"))]:
        url='https://ssd.jpl.nasa.gov/api/horizons.api?'+urlencode(dict(format='json',**params))
        with urlopen(url,timeout=60) as f: response=json.load(f)
        if 'error' in response or 'result' not in response or (name!='earth' and '$$SOE' not in response['result']):
            raise RuntimeError(f'Horizons request failed: {response}')
        snapshot=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),url=url,parameters=params,
            result_sha256=hashlib.sha256(response['result'].encode()).hexdigest(),response=response)
        (DATA/f'{name}.json').write_text(json.dumps(snapshot,indent=2)+'\n')


def cached(name):
    d=json.loads((DATA/f'{name}.json').read_text());s=d['response']['result']
    if hashlib.sha256(s.encode()).hexdigest()!=d['result_sha256']: raise ValueError('Snapshot hash mismatch')
    return d,s


def mission():
    d,text=cached('encounter');c,earth=cached('earth')
    if d['parameters']!=PARAMS: raise ValueError('Unexpected Horizons request')
    header=text.split('$$SOE')[0]
    for item in ('Juno (spacecraft) (-61)','Earth (399)','BODY CENTER','GEOMETRIC cartesian states',
                 'Output units    : KM-S','Reference frame : ICRF','JDTDB','Date (TDB)'):
        if item not in header: raise ValueError(f'Unexpected Horizons header: {item}')
    if c['parameters']!=dict(COMMAND="'399'",OBJ_DATA="'YES'",MAKE_EPHEM="'NO'"):
        raise ValueError('Unexpected Earth request')
    mu=float(re.search(r'GM, km\^3/s\^2\s*=\s*([\d.]+)',earth)[1])*1e9
    radius=float(re.search(r'Equ. radius, km\s*=\s*([\d.]+)',earth)[1])*1000
    records=[]
    for row in csv.reader(io.StringIO(text.split('$$SOE')[1].split('$$EOE')[0])):
        if not row: continue
        if len(row)!=9 or row[-1].strip(): raise ValueError('Unexpected state columns')
        records.append((float(row[0]),row[1].strip(),np.array(row[2:8],float)*1000))
    jd=np.array([r[0] for r in records]);states=np.array([r[2] for r in records]);dist=np.linalg.norm(states[:,:3],axis=1)
    dates=[datetime.strptime(r[1], 'A.D. %Y-%b-%d %H:%M:%S.%f') for r in records]
    if (len(records)!=289 or any((b-a).total_seconds()!=300 for a,b in zip(dates,dates[1:]))
        or not np.allclose(np.diff(jd)*86400,300,rtol=0,atol=1e-4)
        or dates[0]!=datetime(2013,10,9,6) or dates[-1]!=datetime(2013,10,10,6)):
        raise ValueError('Unexpected epoch coverage or cadence')
    if not np.isfinite(states).all(): raise ValueError('Nonfinite states')
    fine,text_fine=cached('perigee')
    if fine['parameters']!=FINE:raise ValueError('Unexpected refined Horizons request')
    fine_rows=[r for r in csv.reader(io.StringIO(text_fine.split('$$SOE')[1].split('$$EOE')[0])) if r]
    if len(fine_rows)!=181:raise ValueError('Unexpected refined sample count')
    fine_states=np.array([r[2:8] for r in fine_rows],float)*1000
    fine_dates=[datetime.strptime(r[1].strip(),'A.D. %Y-%b-%d %H:%M:%S.%f') for r in fine_rows]
    if any((b-a).total_seconds()!=5 for a,b in zip(fine_dates,fine_dates[1:])):raise ValueError('Unexpected refined cadence')
    fine_index=int(np.argmin(np.linalg.norm(fine_states[:,:3],axis=1)))
    observed_altitude=float(np.linalg.norm(fine_states[fine_index,:3])-radius)
    closest=int(np.argmin(dist))
    # Sample one hour either side of the nearest 5-minute closest-approach bin.
    inbound,outbound=states[closest-12],states[closest+12]
    def elements(state):
        r,v=state[:3],state[3:];h=np.cross(r,v);H=np.linalg.norm(h);R=np.linalg.norm(r)
        evec=np.cross(v,h)/mu-r/R;e=np.linalg.norm(evec);P=evec/e;Q=np.cross(h/H,P)
        speed=math.sqrt(np.dot(v,v)-2*mu/R)
        return speed*(P/e+math.sqrt(e*e-1)/e*Q),speed*(-P/e+math.sqrt(e*e-1)/e*Q),H*H/(mu*(1+e))
    vin,predicted,rp=elements(inbound);_,observed,rp_out=elements(outbound)
    S=vin/np.linalg.norm(vin);B=np.cross(S,np.cross(S,predicted)/np.linalg.norm(np.cross(S,predicted)))
    ref=np.array([0.,0.,1.]);T=ref-np.dot(ref,S)*S;T/=np.linalg.norm(T);R=np.cross(S,T)
    azimuth=math.atan2(np.dot(B,R),np.dot(B,T))
    p=Encounter(tuple(vin),mu,radius+100000,1e9,rp=rp,azimuth=azimuth)
    solution=solve(p)
    if solution.status!=Status.SUCCESS: raise AssertionError(solution)
    # Independent central-force integration predicts a finite, observable state.
    def ode(t,y): return np.r_[y[3:],-mu*y[:3]/np.linalg.norm(y[:3])**3]
    def peri_event(t,y): return np.dot(y[:3],y[3:])
    peri_event.direction=1
    dt=(dates[closest+12]-dates[closest-12]).total_seconds()
    orbit=solve_ivp(ode,(0,dt),inbound,method='DOP853',rtol=3e-13,atol=1e-7,events=peri_event)
    if not orbit.success or len(orbit.t_events[0])!=1: raise AssertionError('Failed central-force integration')
    report=dict(mission='Juno Earth gravity assist, 2013-10-09',frame='ICRF',origin='Earth (399)',
        input_epoch=records[closest-12][1],output_epoch=records[closest+12][1],
        central_force_periapsis_epoch_tdb=(dates[closest-12]+timedelta(seconds=float(orbit.t_events[0][0]))).isoformat(),
        nearest_sample_epoch=records[closest][1],sample_interval_s=300,
        refined_sample_epoch=fine_rows[fine_index][1].strip(),refined_sample_interval_s=5,
        refined_sample_altitude_equatorial_radius_m=observed_altitude,
        two_body_vs_refined_periapsis_m=float(rp-radius-observed_altitude),
        refined_snapshot_sha256=fine['result_sha256'],
        nearest_sample_altitude_m=float(dist[closest]-radius),
        two_body_periapsis_altitude_m=float(rp-radius),
        inbound_vinf_m_s=float(np.linalg.norm(vin)),outbound_osculating_vinf_m_s=float(np.linalg.norm(observed)),
        speed_difference_m_s=float(np.linalg.norm(observed)-np.linalg.norm(vin)),
        predicted_turn_deg=math.degrees(solution.angle),
        outbound_asymptote_difference_m_s=float(np.linalg.norm(solution.vout-observed)),
        same_conic_asymptote_error_m_s=float(np.linalg.norm(solution.vout-predicted)),
        finite_state_position_difference_m=float(np.linalg.norm(orbit.y[:3,-1]-outbound[:3])),
        finite_state_velocity_difference_m_s=float(np.linalg.norm(orbit.y[3:,-1]-outbound[3:])),
        outbound_osculating_rp_difference_m=float(rp_out-rp),
        snapshot_sha256=d['result_sha256'],constants_sha256=c['result_sha256'],
        limitation='Horizons includes non-central/multibody dynamics. This two-body model is an encounter approximation, not a navigation reconstruction.')
    # Broad regression bounds detect unit/frame/sign mistakes, not perturbations.
    assert 4e5<report['two_body_periapsis_altitude_m']<8e5
    assert 4e5<observed_altitude<8e5
    assert abs(rp-radius-observed_altitude)<1e4
    assert 0<report['outbound_asymptote_difference_m_s']<100
    assert report['same_conic_asymptote_error_m_s']<1e-7
    assert report['finite_state_velocity_difference_m_s']<100
    return p,report


def mp_reference(p):
    """Use orbital energy and eccentricity at 100 digits, not product routines."""
    with mp.workdps(100):
        def vec(v):return mp.matrix([mp.mpf(float(x)) for x in v])
        def norm(v):return mp.sqrt(mp.fdot(v,v))
        def cross(a,b):return mp.matrix([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
        v=vec(p.vin);vi=norm(v);S=v/vi;ref=vec(p.reference);T=ref-mp.fdot(ref,S)*S;T/=norm(T);R=cross(S,T)
        mu=mp.mpf(p.mu);requested=vec(p.vout);vo=norm(requested) if p.mode==Mode.POWERED else vi
        if p.mode==Mode.FORWARD:
            rp=mp.mpf(p.rp);B=mp.cos(p.azimuth)*T+mp.sin(p.azimuth)*R
        else:
            U=requested/norm(requested);angle=mp.atan2(norm(cross(S,U)),mp.fdot(S,U))
            H=cross(S,U);H/=norm(H);B=cross(S,H)
            if p.mode==Mode.MATCH: rp=mu/vi**2*(1/mp.sin(angle/2)-1)
            else:
                lo,hi=mp.mpf(p.rp_min),mp.mpf(p.rp_max)
                for _ in range(210):
                    rp=(lo+hi)/2
                    turn=mp.asin(1/(1+rp*vi**2/mu))+mp.asin(1/(1+rp*vo**2/mu))
                    if turn>angle:lo=rp
                    else:hi=rp
                rp=(lo+hi)/2
        ein=1+rp*vi**2/mu;eout=1+rp*vo**2/mu
        ai=mp.asin(1/ein);ao=mp.asin(1/eout)
        radial=mp.sin(ai)*S+mp.cos(ai)*B;tangent=mp.cos(ai)*S-mp.sin(ai)*B
        vpi=mp.sqrt(vi**2+2*mu/rp);vpo=mp.sqrt(vo**2+2*mu/rp)
        out=vo*(mp.cos(ai+ao)*S-mp.sin(ai+ao)*B)
        return dict(rp=float(rp),vout=np.array(list(out),float),rperi=np.array(list(rp*radial),float),
                    vperi_in=np.array(list(vpi*tangent),float),vperi_out=np.array(list(vpo*tangent),float),
                    delta_v=float(abs(vpo-vpi)),angle=float(ai+ao))


def samples(count=300):
    rng=np.random.default_rng(921554);cases=[]
    for i in range(count):
        v=rng.normal(size=3);v=v/np.linalg.norm(v)*10**rng.uniform(2,5)
        mu=10**rng.uniform(10,18);rp=10**rng.uniform(4,10)
        p=Encounter(tuple(v),mu,rp*.05,rp*20,rp=rp,azimuth=rng.uniform(-math.pi,math.pi))
        r=solve(p);cases.append(p)
        cases.append(replace(p,mode=Mode.MATCH,vout=r.vout))
        # Independent outbound direction: reconstruct conic from post-burn
        # periapsis state and orbital elements, not a second forward solve.
        factor=rng.uniform(.5,2);vp=np.asarray(r.vperi_in);vp=vp/np.linalg.norm(vp)*math.sqrt((np.linalg.norm(v)*factor)**2+2*mu/rp)
        h=np.cross(r.rperi,vp);ev=np.cross(vp,h)/mu-np.asarray(r.rperi)/rp;e=np.linalg.norm(ev)
        P=ev/e;Q=np.cross(h/np.linalg.norm(h),P)
        target=np.linalg.norm(v)*factor*(-P/e+math.sqrt(e*e-1)/e*Q)
        cases.append(replace(p,mode=Mode.POWERED,vout=tuple(target)))
    # Extreme turns with exact axis-aligned incoming direction to separate
    # representation conditioning from algorithmic cancellation.
    for u in (1e-24,1e-16,1e-8,1.,1e8,1e16,1e24):
        rp=3.98600435436e14/1e8*u
        p=Encounter((10000.,0.,0.),3.98600435436e14,rp*.01,rp*100,rp=rp,azimuth=.4)
        r=solve(p);cases.extend([p,replace(p,mode=Mode.MATCH,vout=r.vout),replace(p,mode=Mode.POWERED,vout=r.vout)])
    return cases


def validate(cuda=False,count=300):
    problems=samples(count);mission_input,mission_report=mission();problems.append(mission_input)
    rows=pack_encounters(problems);eng=FlybyEngine('cuda' if cuda else 'cpu')
    cpu=eng.cpu(rows)
    if cuda:
        with eng.cuda_batch(len(rows)) as batch:
            batch.upload(rows);batch.run();gpu=batch.download()
    else:gpu=None
    report=dict(cases=len(rows),mp_cases=0,python_max_vector_relative=0.,python_max_rp_relative=0.,
                cpu_max_vector_relative=0.,cpu_max_rp_relative=0.,max_periapsis_state_relative=0.,max_delta_v_error_m_s=0.,
                gpu_max_vector_relative=0.,gpu_max_rp_relative=0.,warnings=0,mission=mission_report)
    for i,(p,row) in enumerate(zip(problems,cpu)):
        py=solve(p)
        assert row['status']==py.status,(i,p,row['status'],py.status)
        if gpu is not None:assert gpu[i]['status']==row['status'],(i,'GPU status')
        if row['status']!=0:raise AssertionError((i,p,py))
        ref=mp_reference(p);report['mp_cases']+=1;report['warnings']+=int(row['precision_limited']==1)
        speed=np.linalg.norm(row['vout'])
        pe=np.linalg.norm(np.asarray(py.vout)-ref['vout'])/speed;pr=abs(py.rp/ref['rp']-1)
        report['python_max_vector_relative']=max(report['python_max_vector_relative'],float(pe))
        report['python_max_rp_relative']=max(report['python_max_rp_relative'],float(pr))
        assert pe<2e-10 and pr<2e-8,(i,'Python',pe,pr)
        for actual in (row,gpu[i]) if gpu is not None else (row,):
            for key in ('rperi','vperi_in','vperi_out'):
                error=np.linalg.norm(actual[key]-ref[key])/np.linalg.norm(ref[key])
                report['max_periapsis_state_relative']=max(report['max_periapsis_state_relative'],float(error))
                assert error<2e-10,(i,key,error)
            error=abs(actual['delta_v']-ref['delta_v'])
            report['max_delta_v_error_m_s']=max(report['max_delta_v_error_m_s'],float(error))
            assert error<2e-9+2e-12*ref['delta_v'],(i,'delta_v',error)
        ce=np.linalg.norm(row['vout']-ref['vout'])/speed
        cr=abs(row['rp']/ref['rp']-1)
        report['cpu_max_vector_relative']=max(report['cpu_max_vector_relative'],float(ce))
        report['cpu_max_rp_relative']=max(report['cpu_max_rp_relative'],float(cr))
        assert ce<2e-10 and cr<2e-8,(i,p,ce,cr)
        if gpu is not None:
            ge=np.linalg.norm(gpu[i]['vout']-ref['vout'])/speed;gr=abs(gpu[i]['rp']/ref['rp']-1)
            report['gpu_max_vector_relative']=max(report['gpu_max_vector_relative'],float(ge))
            report['gpu_max_rp_relative']=max(report['gpu_max_rp_relative'],float(gr))
            assert ge<2e-10 and gr<2e-8,(i,'GPU',ge,gr)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true');ap.add_argument('--cuda',action='store_true')
    ap.add_argument('--count',type=int,default=300);ap.add_argument('--output',type=Path,default=ROOT/'validation/results/flyby-validation.json')
    a=ap.parse_args()
    if a.refresh:refresh()
    start=time.perf_counter();report=validate(a.cuda,a.count);report['wall_seconds']=time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
