"""Reproducible EMC paper trajectory audit; geometric SPICE, SI, TDB.

A diagnostic validation program, not a mission optimizer. Dates printed only to
one day in the paper are treated as midnight TDB assumptions. Full candidate
velocities are preserved until encounters have been checked. --search adds a
bounded neighborhood search; a sampled search is never an impossibility proof.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from lamberthub import gooding1990
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research.ephemeris import SpiceEphemeris
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine,INPUT_DTYPE,RESULT_DTYPE,pack_problems
from src.flyby import Encounter,Mode,Status,solve

DAY=86400.;MARS_RADIUS=3396200.;CERES_RADIUS=482100.
KERNEL_NAMES=('naif0012.tls','de440s.bsp','gm_de440.tpc','20000001.bsp','mar099s.bsp')


def norm(v):return float(np.linalg.norm(v))
def angle(a,b):return math.atan2(norm(np.cross(a,b)),float(np.dot(a,b)))
def normal(r,v):
    h=np.cross(r,v);return h/norm(h)
def iso(et):return (datetime(2000,1,1,12)+timedelta(seconds=float(et))).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,np.floating)):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def execute(engine,rows):
    if engine.backend=='cpu':return engine.cpu(rows)
    with engine.cuda_batch(len(rows)) as batch:
        batch.upload(rows);batch.run();return batch.download()


def patch(vi,vo,mu,altitude=200000.):
    a,b=norm(vi),norm(vo);required=angle(vi,vo);minimum=MARS_RADIUS+altitude
    maximum=2*math.asin(1/(1+minimum*a*a/mu))
    # Optimistic correction applied after any allowed unpowered turn at
    # infinity; this is not a periapsis burn or a general DSM optimum.
    angular_gap=max(0.,required-maximum)
    residual=math.hypot(b-a,2*math.sqrt(a*b)*math.sin(angular_gap/2))
    sh=math.sin(required/2);ch=math.cos(required/2)
    counterfactual_rp=(mu/a)/a*ch*ch/(sh*(1+sh)) if sh else math.inf
    reference=(0.,0.,1.) if abs(float(vi[2])/a)<.95 else (1.,0.,0.)
    p=Encounter(tuple(vi),mu,minimum,1e9,Mode.MATCH,tuple(vo),reference)
    ballistic=solve(p);powered=solve(Encounter(**dict(p.__dict__,mode=Mode.POWERED)))
    return dict(vinf_in_m_s=a,vinf_out_m_s=b,speed_mismatch_m_s=b-a,
        required_turn_deg=math.degrees(required),maximum_unpowered_turn_deg=math.degrees(maximum),
        turn_margin_deg=math.degrees(maximum-required),safety_altitude_m=altitude,
        turn_only_rp_at_incoming_speed_m=counterfactual_rp,
        optimistic_postflyby_asymptotic_correction_m_s=residual,
        ballistic_status=Status(ballistic.status).name,powered_tangential_status=Status(powered.status).name,
        powered_tangential_delta_v_m_s=powered.delta_v,
        powered_tangential_rp_m=powered.rp,
        vinf_in_vector_m_s=np.asarray(vi),vinf_out_vector_m_s=np.asarray(vo))


def make_leg(e,engine,origin,destination,t1,t2,label):
    s1=e.states(origin,[t1])[0];s2=e.states(destination,[t2])[0]
    problems=[];branches=[];policies=[]
    for long in (False,True):
        for m in (0,1,2):
            for branch in (('left',) if m==0 else ('left','right')):
                problems.append(LambertInput(tuple(s1[:3]),tuple(s2[:3]),float(t2-t1),e.mu,long,m))
                branches.append(branch);policies.append((long,m,branch))
    rows=pack_problems(problems,branches);out=execute(engine,rows)
    candidates=[];all_status=[];check_errors=[]
    for index,(r,p,policy) in enumerate(zip(out,problems,policies)):
        long,m,branch=policy
        prograde=bool((np.cross(s1[:3],s2[:3])[2]>0)!=long)
        identity=dict(long_way=long,revolutions=m,branch=branch if m else None,prograde=prograde)
        all_status.append(dict(**identity,status=int(r['status']),root_status=int(r['root']['status']),
            minimum_time_days=float(r['minimum']['time']/r['target']*(t2-t1)/DAY) if m and r['target']>0 else None))
        if r['status']!=0:continue
        refs=[]
        for low in ((False,) if m==0 else (False,True)):
            try:
                v1,v2=gooding1990(e.mu,s1[:3],s2[:3],float(t2-t1),M=m,prograde=prograde,
                    low_path=low,maxiter=100,atol=1e-13,rtol=1e-13)
                refs.append((v1,v2))
            except ValueError:pass
        if not refs:raise AssertionError('Native solution not reproduced by Gooding')
        error=min(max(norm(r['v1']-a),norm(r['v2']-b)) for a,b in refs)
        if error>1e-5:raise AssertionError(('Independent Lambert error',label,error))
        check_errors.append(error)
        vinf1=r['v1']-s1[3:];vinf2=r['v2']-s2[3:]
        h=normal(s1[:3],r['v1']);hbody=normal(s1[:3],s1[3:]);harr=normal(s2[:3],s2[3:])
        eccentricity=norm(np.cross(r['v1'],np.cross(s1[:3],r['v1']))/e.mu-s1[:3]/norm(s1[:3]))
        perihelion=norm(np.cross(s1[:3],r['v1']))**2/(e.mu*(1+eccentricity))
        candidate=dict(id=f'{label}-{index}',**identity,v1_m_s=r['v1'],v2_m_s=r['v2'],
            vinf_departure_m_s=vinf1,vinf_arrival_m_s=vinf2,C3_departure_km2_s2=float(np.dot(vinf1,vinf1)/1e6),
            arrival_vinf_m_s=norm(vinf2),heliocentric_inclination_deg=math.degrees(angle(h,[0,0,1])),
            transfer_vs_departure_body_plane_deg=math.degrees(angle(h,hbody)),
            transfer_vs_arrival_body_plane_deg=math.degrees(angle(h,harr)),
            solar_perihelion_m=perihelion,gooding_max_endpoint_velocity_error_m_s=error,
            root_precision_limited=int(r['root']['precision_limited']))
        # Independent central-force ODE closure for every feasible M=0 arc.
        if m==0:
            scale=149597870700.;unit=DAY;gm=e.mu*unit**2/scale**3
            y0=np.r_[s1[:3]/scale,r['v1']*unit/scale]
            def ode(t,y):return np.r_[y[3:],-gm*y[:3]/np.linalg.norm(y[:3])**3]
            ode_result=solve_ivp(ode,(0,(t2-t1)/DAY),y0,method='DOP853',rtol=3e-13,atol=2e-15)
            if not ode_result.success:raise AssertionError('Independent ODE failed')
            position_error=norm(ode_result.y[:3,-1]*scale-s2[:3]);velocity_error=norm(ode_result.y[3:,-1]*scale/unit-r['v2'])
            if position_error>500 or velocity_error>1e-3:raise AssertionError(('ODE closure',label,position_error,velocity_error))
            candidate.update(ode_position_closure_m=position_error,ode_velocity_closure_m_s=velocity_error)
        candidates.append(candidate)
    return dict(label=label,origin=origin,destination=destination,departure_tdb=iso(t1),arrival_tdb=iso(t2),
        tof_days=float((t2-t1)/DAY),departure_state_si=s1,arrival_state_si=s2,
        attempted_branches=all_status,candidates=candidates,maximum_gooding_error_m_s=max(check_errors,default=0.))


def nominal(e,engine,mu_mars,mu_ceres):
    launch=e.epoch('2035-07-01');mars=e.epoch('2036-01-17');cal=e.epoch('2036-09-24');back=e.epoch('2037-12-18');earth=e.epoch('2038-04-22')
    specifications=[('durations_first',[launch,mars,mars+250*DAY,mars+280*DAY,mars+730*DAY,mars+855*DAY]),
                    ('calendar_first',[launch,mars,cal,e.epoch('2036-10-24'),back,earth])]
    scenarios=[]
    for name,t in specifications:
        legs=[make_leg(e,engine,a,b,t[i],t[j],name+'-'+route) for a,b,i,j,route in
              [('399','499',0,1,'earth-mars'),('499','20000001',1,2,'mars-ceres'),
               ('20000001','499',3,4,'ceres-mars'),('499','399',4,5,'mars-earth')]]
        encounters=[]
        for index,(incoming,outgoing) in enumerate(((legs[0],legs[1]),(legs[2],legs[3]))):
            r=incoming['arrival_state_si'][:3]
            for a in incoming['candidates']:
                for b in outgoing['candidates']:
                    p=patch(a['vinf_arrival_m_s'],b['vinf_departure_m_s'],mu_mars)
                    p.update(incoming_id=a['id'],outgoing_id=b['id'],encounter_index=index,
                        prograde_pair=a['prograde'] and b['prograde'],
                        heliocentric_transfer_plane_change_deg=math.degrees(angle(normal(r,a['v2_m_s']),normal(r,b['v1_m_s']))))
                    encounters.append(p)
        # Ceres visit cannot be treated as an unpowered fly-through. Report
        # v-infinity and optimistic impulsive capture-to-just-bound lower costs.
        escape=math.sqrt(2*mu_ceres/CERES_RADIUS)
        visit=[]
        for a in legs[1]['candidates']:
            for b in legs[2]['candidates']:
                vi=a['arrival_vinf_m_s'];vo=norm(b['vinf_departure_m_s'])
                visit.append(dict(incoming_id=a['id'],outgoing_id=b['id'],prograde_pair=a['prograde'] and b['prograde'],
                    arrival_vinf_m_s=vi,departure_vinf_m_s=vo,
                    optimistic_capture_escape_sum_m_s=vi*vi/(math.hypot(vi,escape)+escape)+vo*vo/(math.hypot(vo,escape)+escape)))
        scenarios.append(dict(name=name,epochs_tdb=list(map(iso,t)),legs=legs,mars_encounters=encounters,ceres_visit=visit))
    return scenarios


POLICIES=[(long,m,b) for long in (False,True) for m in (0,1,2) for b in ((0,) if m==0 else (0,1))]


def population(e,engine,departure_states,arrival_states,tofs,shape):
    base=np.zeros(len(tofs),dtype=INPUT_DTYPE)
    base['r1']=departure_states[:,:3];base['r2']=arrival_states[:,:3]
    base['tof']=tofs;base['mu']=e.mu
    output=np.empty((len(tofs),len(POLICIES)),dtype=RESULT_DTYPE)
    for k,(long,m,b) in enumerate(POLICIES):
        base['long_way']=int(long);base['revolutions']=m;base['branch']=b
        if engine.backend=='cuda':
            with engine.cuda_batch(min(65536,len(base))) as batch:
                for lo in range(0,len(base),65536):
                    hi=min(len(base),lo+65536)
                    batch.upload(base[lo:hi]);batch.run();output[lo:hi,k]=batch.download()
        else:output[:,k]=engine.cpu(base)
    return output.reshape((*shape,len(POLICIES)))


def search(e,engine,mu,outdir):
    started=time.perf_counter()
    launches=e.epoch('2035-07-01')+np.arange(-60,61)*DAY
    encounters=e.epoch('2036-01-17')+np.arange(-60,61)*DAY
    flights=np.arange(160.,341.)
    earth=e.states('399',launches);mars=e.states('499',encounters)
    arrival_epochs,arrival_index=np.unique((encounters[:,None]+flights[None,:]*DAY).ravel(),return_inverse=True)
    ceres=e.states('20000001',arrival_epochs)
    nm,ne,nt=len(encounters),len(launches),len(flights)
    dep1=np.broadcast_to(earth[None,:,:],(nm,ne,6)).reshape(-1,6)
    arr1=np.repeat(mars,ne,axis=0)
    incoming=population(e,engine,dep1,arr1,(encounters[:,None]-launches[None,:]).ravel(),(nm,ne))
    dep2=np.repeat(mars,nt,axis=0);arr2=ceres[arrival_index]
    outgoing=population(e,engine,dep2,arr2,np.tile(flights,nm)*DAY,(nm,nt))
    print('Neighborhood Lambert populations complete',flush=True)
    # Preserve the complete branch outputs and exact states, independently of
    # any minimum-C3 reducer. Files live outside the repository's Git history.
    np.savez_compressed(outdir/'search-populations.npz',launch_et=launches,mars_et=encounters,tof_days=flights,
        earth_states=earth,mars_states=mars,ceres_et=arrival_epochs,ceres_states=ceres,
        arrival_index=arrival_index.reshape(nm,nt),policies=np.array(POLICIES,dtype='i4'),
        incoming=incoming,outgoing=outgoing)
    diagnostics={}
    for name,pop in [('incoming',incoming),('outgoing',outgoing)]:
        status,count=np.unique(pop['status'],return_counts=True)
        root,rcount=np.unique(pop['root']['status'],return_counts=True)
        diagnostics[name]=dict(count=pop.size,status_counts=dict(zip(map(int,status),map(int,count))),
            root_status_counts=dict(zip(map(int,root),map(int,rcount))),
            feasible_by_revolution={str(m):int(sum(np.sum(pop[...,k]['status']==0) for k,p in enumerate(POLICIES) if p[1]==m)) for m in (0,1,2)})
        unexpected=(pop['status']!=0)&(pop['root']['status']!=8)
        if unexpected.any():raise AssertionError(('Unexpected Lambert failures',name,np.unique(pop['root']['status'][unexpected])))
    best_grid={'prograde':None,'all':None};curve_roots=[];root_failures=[];pairs=0
    ray=np.zeros(1,dtype=INPUT_DTYPE);ray['mu']=e.mu
    counts=Counter()
    for im,et in enumerate(encounters):
        body=mars[im]
        for ki,pi in enumerate(POLICIES):
            valid_in=np.flatnonzero(incoming[im,:,ki]['status']==0)
            for ko,po in enumerate(POLICIES):
                valid_out=np.flatnonzero(outgoing[im,:,ko]['status']==0)
                if not len(valid_in) or not len(valid_out):continue
                arows=incoming[im,valid_in,ki];brows=outgoing[im,valid_out,ko]
                vi=arows['v2']-body[3:];vo=brows['v1']-body[3:]
                va=np.linalg.norm(vi,axis=1);vb=np.linalg.norm(vo,axis=1)
                cross=np.cross(vi[:,None,:],vo[None,:,:]);cosine=vi@vo.T
                required=np.arctan2(np.linalg.norm(cross,axis=2),cosine)
                maximum=2*np.arcsin(1/(1+(MARS_RADIUS+200000)*va*va/mu))
                correction=np.hypot(vb[None,:]-va[:,None],2*np.sqrt(va[:,None]*vb[None,:])*np.sin(np.maximum(0,required-maximum[:,None])/2))
                pro_in=np.cross(np.broadcast_to(body[:3],arows['v2'].shape),arows['v2'])[:,2]>0
                pro_out=np.cross(np.broadcast_to(body[:3],brows['v1'].shape),brows['v1'])[:,2]>0
                pair_pro=pro_in[:,None]&pro_out[None,:]
                pairs+=correction.size
                for category,mask in [('all',np.ones_like(pair_pro)),('prograde',pair_pro)]:
                    if not mask.any():continue
                    scored=np.where(mask,correction,np.inf);ii,jj=np.unravel_index(np.argmin(scored),scored.shape)
                    score=float(scored[ii,jj])
                    if best_grid[category] is None or score<best_grid[category]['optimistic_postflyby_asymptotic_correction_m_s']:
                        ie,it=int(valid_in[ii]),int(valid_out[jj])
                        best_grid[category]=dict(launch_tdb=iso(launches[ie]),mars_tdb=iso(et),
                            ceres_tdb=iso(et+flights[it]*DAY),tof_in_days=float((et-launches[ie])/DAY),tof_out_days=float(flights[it]),
                            incoming_policy=pi,outgoing_policy=po,**patch(vi[ii],vo[jj],mu))
                # Refine every speed-equality sign crossing between neighboring
                # feasible daily TOF samples, with exact SPICE states at each
                # continuous time and the compiled CPU Lambert core.
                b_all=outgoing[im,:,ko];vball=np.linalg.norm(b_all['v1']-body[3:],axis=1)
                for jj,ie in enumerate(valid_in):
                    g=vball-va[jj]
                    good=(b_all['status'][:-1]==0)&(b_all['status'][1:]==0)
                    brackets=np.flatnonzero(good&((g[:-1]*g[1:]<0)|(g[:-1]==0)))
                    for it in brackets:
                        ray['r1']=body[:3];ray['long_way']=int(po[0]);ray['revolutions']=po[1];ray['branch']=po[2]
                        def evaluate(days):
                            state=np.asarray(e.sp.spkezr('20000001',float(et+days*DAY),e.frame,'NONE','SUN')[0])*1000
                            ray['r2']=state[:3];ray['tof']=float(days*DAY)
                            result=engine.cpu(ray)[0]
                            if result['status']!=0:raise ValueError('Infeasible branch during speed refinement')
                            return norm(result['v1']-body[3:])-float(va[jj]),result.copy(),state
                        try:
                            root=brentq(lambda x:evaluate(x)[0],float(flights[it]),float(flights[it+1]),xtol=1e-9,rtol=1e-14)
                            mismatch,best,state=evaluate(root)
                        except ValueError as error:
                            root_failures.append(dict(mars_index=im,launch_index=int(ie),incoming_policy=pi,outgoing_policy=po,error=str(error)))
                            continue
                        metrics=patch(vi[jj],best['v1']-body[3:],mu)
                        hin=normal(body[:3],arows[jj]['v2']);hout=normal(body[:3],best['v1'])
                        row=dict(launch_tdb=iso(launches[ie]),mars_tdb=iso(et),ceres_tdb=iso(et+root*DAY),
                            tof_in_days=float((et-launches[ie])/DAY),tof_out_days=root,incoming_policy=pi,outgoing_policy=po,
                            prograde_pair=bool(pro_in[jj] and np.cross(body[:3],best['v1'])[2]>0),
                            earth_departure_C3_km2_s2=float(np.dot(arows[jj]['v1']-earth[ie,3:],arows[jj]['v1']-earth[ie,3:])/1e6),
                            heliocentric_plane_change_deg=math.degrees(angle(hin,hout)),
                            ceres_arrival_vinf_m_s=norm(best['v2']-state[3:]),**metrics)
                        curve_roots.append(row);counts['roots']+=1
        if im%10==0:print(f'Neighborhood {im+1}/{nm} encounter days, {len(curve_roots)} speed-equality roots',flush=True)
    best_roots={};feasible={}
    for category in ('all','prograde'):
        subset=[r for r in curve_roots if category=='all' or r['prograde_pair']]
        best_roots[category]=max(subset,key=lambda r:r['turn_margin_deg']) if subset else None
        feasible[category]=[r for r in subset if r['ballistic_status']=='SUCCESS']
    fields=['launch_tdb','mars_tdb','ceres_tdb','tof_in_days','tof_out_days','incoming_policy','outgoing_policy','prograde_pair',
            'earth_departure_C3_km2_s2','vinf_in_m_s','vinf_out_m_s','speed_mismatch_m_s','required_turn_deg',
            'maximum_unpowered_turn_deg','turn_margin_deg','turn_only_rp_at_incoming_speed_m','ballistic_status',
            'heliocentric_plane_change_deg','ceres_arrival_vinf_m_s']
    with (outdir/'speed-equality-roots.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for row in curve_roots:writer.writerow({k:row[k] for k in fields})
    summary=dict(domain=dict(earth_departure_start=iso(launches[0]),earth_departure_end=iso(launches[-1]),
        mars_encounter_start=iso(encounters[0]),mars_encounter_end=iso(encounters[-1]),
        outgoing_tof_min_days=float(flights[0]),outgoing_tof_max_days=float(flights[-1]),date_spacing_days=1,
        revolutions=[0,1,2],orientations=['short','long'],altitude_m=200000.),
        diagnostics=diagnostics,sampled_candidate_pairs=pairs,best_sampled_asymptotic_correction=best_grid,
        refined_speed_equality_roots=len(curve_roots),refinement_failures=root_failures,
        feasible_speed_matched_counts={k:len(v) for k,v in feasible.items()},best_speed_matched_turn_margin=best_roots,
        seconds=time.perf_counter()-started,
        limitations=['Launch and Mars epochs remain on a daily grid; TOF is continuously refined only at sign crossings.',
            'Tangential zeros, crossings outside the TOF window and disconnected feasibility islands may be missed.',
            'A bounded search without a valid solution is not a proof of global infeasibility.',
            'Encounter matching uses two-body patched conics and an assumed 200 km minimum altitude.'])
    (outdir/'neighborhood.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel-dir',type=Path,default=ROOT/'data/kernels');p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','cuda'),default='cpu');p.add_argument('--search',action='store_true')
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    engine=NativeEngine(a.backend)
    with SpiceEphemeris([a.kernel_dir/n for n in KERNEL_NAMES],'ECLIPJ2000',a.outdir/'state-cache') as e:
        mars=float(e.sp.bodvrd('MARS','GM',1)[1][0])*1e9
        ceres=float(e.sp.bodvrd('2000001','GM',1)[1][0])*1e9
        report=dict(schema_version=1,backend=a.backend,frame='ECLIPJ2000',time_scale='TDB',units='SI',
            solar_mu_m3_s2=e.mu,mars_mu_m3_s2=mars,ceres_mu_m3_s2=ceres,kernels=e.provenance,
            mars_radius_m=MARS_RADIUS,ceres_radius_assumption_m=CERES_RADIUS,
            source_pdf_sha256=hashlib.sha256(a.paper.read_bytes()).hexdigest(),
            scenarios=nominal(e,engine,mars,ceres))
        if a.search:report['neighborhood_search']=search(e,engine,mars,a.outdir)
    report['wall_seconds']=time.perf_counter()-start
    (a.outdir/'nominal.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    for scenario in report['scenarios']:
        print(scenario['name'],scenario['epochs_tdb'],flush=True)
        for row in scenario['mars_encounters']:
            if row['prograde_pair']:print('Mars',row['encounter_index'],clean(row),flush=True)
        for row in scenario['ceres_visit']:
            if row['prograde_pair']:print('Ceres',clean(row),flush=True)


if __name__=='__main__':main()
