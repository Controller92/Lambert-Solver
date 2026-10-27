"""Bounded EMC survey: all-branch chains, Mars atlases and matched benchmarks.

All accepted sampled local connections are retained in NPZ chunks. They are
distinct from complete feasible chains and from independently validated ones.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc
from .mission_dynamics import MissionEvaluator, dynamic_path, DAY, AU
from .mission_search import write_json, fingerprint, digest
from .mission_config import load_mission
from .mission_local import length
from ..lambert_native import NativeEngine, INPUT_DTYPE
from ..flyby_native import INPUT_DTYPE as FB_INPUT
from ..flyby import Mode


def angle(a,b):
    return np.degrees(np.arctan2(length(np.cross(a,b)),np.sum(a*b,axis=-1)))


def encounter_metrics(r,planet_velocity,vin,vout,ceres_normal):
    """Inertial speed gain is separate from planet-relative speed gain/burn."""
    before,after=planet_velocity+vin,planet_velocity+vout
    hi,ho=np.cross(r,before),np.cross(r,after)
    z=np.zeros_like(hi);z[...,2]=1.
    return dict(plane_change_deg=angle(hi,ho),vinf_turn_deg=angle(vin,vout),
        inclination_change_deg=angle(ho,z)-angle(hi,z),
        heliocentric_speed_gain_km_s=(length(after)-length(before))/1000,
        vinf_speed_gain_km_s=(length(vout)-length(vin))/1000,
        energy_gain_km2_s2=(np.sum(after*after,axis=-1)-np.sum(before*before,axis=-1))/2e6,
        ceres_plane_before_deg=angle(hi,ceres_normal),ceres_plane_after_deg=angle(ho,ceres_normal))


def sample_arc(r,v,end,revolutions,mu,samples=64):
    """Exact conic locus, uniformly sampled in true anomaly (not in time)."""
    h=np.cross(r,v);hn=length(h);u=r/length(r)[:,None]
    tangent=np.cross(h/hn[:,None],u)
    ev=np.cross(v,h)/mu-u;p=hn*hn/mu
    end_angle=np.mod(np.arctan2(np.sum(end*tangent,axis=1),np.sum(end*u,axis=1)),2*np.pi)
    theta=(end_angle+2*np.pi*revolutions)[:,None]*np.linspace(0,1,samples)
    direction=u[:,None,:]*np.cos(theta)[...,None]+tangent[:,None,:]*np.sin(theta)[...,None]
    denominator=1+np.sum(ev[:,None,:]*direction,axis=-1)
    if np.any(denominator<=0):raise ValueError('Conic path crossed an asymptote')
    return direction*(p[:,None]/denominator)[...,None]


def subset(block,i):
    result={}
    for k,v in block.items():
        if isinstance(v,np.ndarray):result[k]=v[i:i+1]
        elif k=='nodes':result[k]=[{name:value[i:i+1] for name,value in node.items()} for node in v]
        elif k=='frames':result[k]={j:(a[i:i+1],b[i:i+1]) for j,(a,b) in v.items()}
        else:result[k]=v
    return result


def joined_mask(block,node):
    valid=(block['legs']['status']==0)&(block['solar_margin']>=-1e-12)
    return valid[:,node-1,:,None]&valid[:,node,None,:]


def ballistic_nodes(e,block):
    nodes=[]
    for node in (1,3):
        vi=block['legs']['v2'][:,node-1]-block['arrival_states'][:,node,None,3:]
        vo=block['legs']['v1'][:,node]-block['departure_states'][:,node,None,3:]
        N,K=vi.shape[:2];rows=np.zeros((N,K,K),dtype=FB_INPUT)
        rows['vin']=vi[:,:,None,:];rows['vout']=vo[:,None,:,:];rows['reference']=[0,0,1]
        pole=np.abs(rows['vin'][...,2])>.95*length(rows['vin']);rows['reference'][pole]=[1,0,0]
        spec=e.config.visits[node]
        rows['mu']=e.mu[node];rows['rp_min']=spec['radius_m']+spec['minimum_altitude_m']
        rows['rp_max']=spec['maximum_periapsis_m'];rows['speed_atol']=e.config.speed_atol_m_s
        rows['speed_rtol']=e.config.speed_rtol;rows['mode']=int(Mode.MATCH)
        result=e._flyby(rows.ravel()).reshape(N,K,K)
        nodes.append(result)
    return nodes


class Atlas:
    def __init__(self,run,policies,mu):
        self.run,self.policies,self.mu=Path(run),policies,mu
        self.serial=0;self.counts={};self.seeds=[];self.best={};self.rows=0;self.bytes=0
        self.metric_hist={};self.metric_edges=(np.linspace(0,180,361),np.linspace(-80,80,401))
        self.track_hist={};self.track_edges=np.linspace(-8,8,601);self.clipped={}
        self.reservoir={};self.rng=np.random.default_rng(958719)
        (self.run/'chunks').mkdir(exist_ok=True)

    def record(self,block,node,model,native,mask):
        key=f'{model}-{node}';n,a,b=np.nonzero(mask);self.counts[key]=self.counts.get(key,0)+len(n)
        if not len(n):return {}
        vi=block['legs']['v2'][n,node-1,a]-block['arrival_states'][n,node,3:]
        vo=block['legs']['v1'][n,node,b]-block['departure_states'][n,node,3:]
        state=block['arrival_states'][n,node]
        ceres=(block['arrival_states'] if node==1 else block['departure_states'])[n,2]
        cn=np.cross(ceres[:,:3],ceres[:,3:])
        metrics=encounter_metrics(state[:,:3],state[:,3:],vi,vo,cn)
        metrics.update(burn_km_s=native['delta_v'][n,a,b]/1000,
                       altitude_km=(native['rp'][n,a,b]-3396200.)/1000,
                       speed_mismatch_m_s=length(vo)-length(vi))
        histogram=np.histogram2d(metrics['plane_change_deg'],metrics['heliocentric_speed_gain_km_s'],bins=self.metric_edges)[0]
        self.metric_hist[key]=self.metric_hist.get(key,0)+histogram
        # Weighted reservoir: preserve an unbiased bounded set for scatter/3D
        # highlights; histograms and on-disk records include EVERY accepted row.
        ranks=self.rng.random(len(n));keep=min(2048,len(n))
        take=np.argpartition(ranks,keep-1)[:keep]
        sample=dict(rank=ranks[take],**{k:v[take] for k,v in metrics.items()})
        for leg,indices in ((node-1,a),(node,b)):
            sample[f'r{leg}']=block['inputs']['r1'][n[take],leg,indices[take]]
            sample[f'v{leg}']=block['legs']['v1'][n[take],leg,indices[take]]
            sample[f'end{leg}']=block['inputs']['r2'][n[take],leg,indices[take]]
            sample[f'm{leg}']=block['inputs']['revolutions'][n[take],leg,indices[take]]
        if key in self.reservoir:
            sample={k:np.concatenate((self.reservoir[key][k],v)) for k,v in sample.items()}
        selected=np.argsort(sample['rank'])[:2048];self.reservoir[key]={k:v[selected] for k,v in sample.items()}
        # Every unique participating leg is drawn once per sampled epoch tuple.
        for leg,axis in ((node-1,2),(node,1)):
            ni,ki=np.nonzero(mask.any(axis=axis))
            for lo in range(0,len(ni),4096):
                ns,ks=ni[lo:lo+4096],ki[lo:lo+4096]
                points=sample_arc(block['inputs']['r1'][ns,leg,ks],block['legs']['v1'][ns,leg,ks],
                    block['inputs']['r2'][ns,leg,ks],block['inputs']['revolutions'][ns,leg,ks],self.mu)/AU
                for vertical in (1,2):
                    h=np.histogram2d(points[...,0].ravel(),points[...,vertical].ravel(),bins=(self.track_edges,self.track_edges))[0]
                    hk=f'{key}-{vertical}';self.track_hist[hk]=self.track_hist.get(hk,0)+h
                    self.clipped[hk]=self.clipped.get(hk,0)+int(points.shape[0]*points.shape[1]-h.sum())
        return dict(indices=np.column_stack((n,a,b)).astype('u4'),rp=native['rp'][n,a,b],
                    delta_v=native['delta_v'][n,a,b],**metrics)

    def save(self,block,matches,refined=False):
        data=dict(decisions=block['decisions'],arrivals=block['arrivals'],departures=block['departures'],
                  arrival_states=block['arrival_states'],departure_states=block['departure_states'],
                  v1=block['legs']['v1'],v2=block['legs']['v2'],status=block['legs']['status'],solar_margin=block['solar_margin'],
                  precision_limited=block['legs']['root']['precision_limited'],
                  relative_residual=block['legs']['root']['relative_residual'],estimated_x_error=block['legs']['root']['estimated_x_error'])
        for j,node in enumerate((1,3)):
            for model,native,mask in (
                ('powered',block['nodes'][node-1]['native'],block['nodes'][node-1]['valid']&joined_mask(block,node)),
                ('ballistic',matches[j],(matches[j]['status']==0)&joined_mask(block,node))):
                if refined and model=='powered':continue
                result=self.record(block,node,model,native,mask)
                data.update({f'{model}_{node}_{k}':v for k,v in result.items()})
        path=self.run/'chunks'/f'{self.serial:06d}.npz'
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**data);temporary.replace(path)
        self.serial+=1;self.rows+=len(block['decisions']);self.bytes+=path.stat().st_size
        return path

    def export(self):
        """Atomic aggregate snapshots; complete chunks remain the source data."""
        arrays={f'metric_{k}':v for k,v in self.metric_hist.items()}
        arrays.update({f'track_{k}':v for k,v in self.track_hist.items()})
        arrays.update(metric_plane_edges=self.metric_edges[0],metric_gain_edges=self.metric_edges[1],track_edges=self.track_edges)
        path=self.run/'atlas.tmp.npz';np.savez_compressed(path,**arrays);path.replace(self.run/'atlas.npz')
        for k,v in self.reservoir.items():
            path=self.run/f'highlights-{k}.tmp.npz';np.savez_compressed(path,**v);path.replace(self.run/f'highlights-{k}.npz')
        write_json(self.run/'counts.json',self.counts)


def select_root_seeds(records,limit=160):
    """Preserve both Mars encounters rather than letting one dominate."""
    selected=[]
    for node in (1,3):
        rows=sorted((r for r in records if r['node']==node),key=lambda r:r['gap'])
        selected.extend(rows[:limit//2])
    return selected


def porkchops(e,run,step=2.):
    from .config import GridConfig
    from .grid import compute_grid
    grids=[];elapsed=time.perf_counter()
    mins=np.array([b[0] for b in e.config.tof_days]);maxs=np.array([b[1] for b in e.config.tof_days])
    for leg in range(4):
        prior_min=mins[:leg].sum()+(e.config.visits[2]['stay_days'][0] if leg>=2 else 0)
        prior_max=maxs[:leg].sum()+(e.config.visits[2]['stay_days'][1] if leg>=2 else 0)
        remaining=mins[leg:].sum()+(e.config.visits[2]['stay_days'][0] if leg<2 else 0)
        lo=prior_min;hi=e.bounds[0,1]+min(prior_max,e.config.max_duration_days-remaining)
        departure=e.launch_origin+np.arange(lo,hi+.01,step)*DAY
        tofs=np.arange(mins[leg],maxs[leg]+.01,step)
        arrivals,inverse=np.unique(departure[:,None]+tofs[None,:]*DAY,return_inverse=True)
        a=e.e.states(e.config.visits[leg]['body'],departure);b=e.e.states(e.config.visits[leg+1]['body'],arrivals)
        cfg=GridConfig(backend='cuda',orientation=e.config.orientation,max_revolutions=e.config.max_revolutions,batch_size=65536)
        start=time.perf_counter()
        data,meta=compute_grid(cfg,departure,tofs,a,b,inverse.reshape(len(departure),len(tofs)),e.e.mu)
        path=run/f'leg-{leg+1}-grid.npz'
        np.savez_compressed(path,**data,dep_states=a,arrival_states=b,arrival_index=inverse.reshape(len(departure),len(tofs)))
        grids.append(dict(leg=leg+1,path=str(path),cells=len(departure)*len(tofs),**meta,wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(stage='porkchop',**grids[-1])),flush=True)
    return dict(grids=grids,total_seconds=time.perf_counter()-elapsed)


def benchmarks(rows,flyby_rows,budget=30.):
    cpu,gpu=NativeEngine(),NativeEngine('cuda');counts=sorted(set((1,2,4,8,len(os.sched_getaffinity(0)))))
    start=time.perf_counter();records=[];expected=cpu.cpu(rows)
    for threads in counts:
        parts=[p for p in np.array_split(rows,threads) if len(p)]
        with ThreadPoolExecutor(threads) as pool:
            samples=[]
            for _ in range(3):
                t=time.perf_counter();results=list(pool.map(cpu.cpu,parts));samples.append(time.perf_counter()-t)
            actual=np.concatenate(results)
        np.testing.assert_array_equal(actual['status'],expected['status'])
        np.testing.assert_allclose(actual['v1'],expected['v1'],rtol=2e-12,atol=1e-7,equal_nan=True)
        records.append(dict(backend='cpu',threads=threads,rows=len(rows),wall_seconds=float(np.median(samples)),
                            rows_per_second=len(rows)/np.median(samples)))
    with gpu.cuda_batch(len(rows)) as batch:
        batch.upload(rows);batch.run()
        measured=[batch.run(repetitions=10) for _ in range(3)]
        records.append(dict(backend='cuda_resident',rows=len(rows),kernel_ms=float(np.median(measured)),
                            rows_per_second=len(rows)*1000/np.median(measured)))
        t=time.perf_counter();stop=t+max(1.,budget-(time.perf_counter()-start));loops=0;kernel_ms=0.
        def burn(part):
            count=0
            while time.perf_counter()<stop:
                cpu.cpu(part);count+=len(part)
            return count
        with ThreadPoolExecutor(counts[-1]) as pool:
            workers=[pool.submit(burn,p) for p in np.array_split(rows,counts[-1]) if len(p)]
            while time.perf_counter()<stop:
                kernel_ms+=batch.run(repetitions=10)*10;loops+=10
            cpu_count=sum(f.result() for f in workers)
        stress_seconds=time.perf_counter()-t
        actual=batch.download()
        np.testing.assert_array_equal(actual['status'],expected['status'])
        good=expected['status']==0
        scale=np.maximum(1,length(expected['v1'][good]))
        error=float(np.max(length(actual['v1'][good]-expected['v1'][good])/scale))
        if error>2e-10:raise AssertionError('CPU/GPU benchmark velocity disagreement')
        records.append(dict(backend='cuda_concurrent_stress',rows=len(rows),repetitions=loops,kernel_ms=kernel_ms,
                            rows_per_second=len(rows)*loops/(kernel_ms/1000),max_scaled_velocity_error=error))
        records.append(dict(backend='cpu_concurrent_stress',threads=counts[-1],rows=cpu_count,wall_seconds=stress_seconds,
                            rows_per_second=cpu_count/stress_seconds))
    if flyby_rows is not None:
        from ..flyby_native import FlybyEngine
        fc,fg=FlybyEngine(),FlybyEngine('cuda');t=time.perf_counter();reference=fc.cpu(flyby_rows);cpu_s=time.perf_counter()-t
        with fg.cuda_batch(len(flyby_rows)) as batch:
            batch.upload(flyby_rows);batch.run();ms=batch.run(repetitions=10);actual=batch.download()
        np.testing.assert_array_equal(actual['status'],reference['status'])
        np.testing.assert_allclose(actual['delta_v'],reference['delta_v'],rtol=2e-10,atol=1e-6,equal_nan=True)
        records.append(dict(backend='flyby_cpu',rows=len(flyby_rows),wall_seconds=cpu_s,rows_per_second=len(flyby_rows)/cpu_s))
        records.append(dict(backend='flyby_cuda_resident',rows=len(flyby_rows),kernel_ms=ms,rows_per_second=len(flyby_rows)*1000/ms))
    return dict(records=records,gpu=gpu.device_info(),cpu_affinity=list(os.sched_getaffinity(0)),
                limitation='Identical rows; failed/infeasible branch attempts included. CPU is wall time; GPU is resident event time, excluding transfers.')


def run_sweep(config,outdir,seconds=900.,max_candidates=1048576,grid_step=2.,batch_size=1024,
              *,seed=958720,max_storage_gib=8.,benchmark_seconds=30.):
    if (not all(math.isfinite(v) for v in (seconds,grid_step,max_storage_gib,benchmark_seconds)) or
        seconds<30 or seconds>86400 or max_candidates<1 or grid_step<=0 or not 1<=batch_size<=4096 or
        not 0<=seed<2**32 or max_storage_gib<=0 or not 0<benchmark_seconds<=3600):
        raise ValueError('Invalid sweep limits')
    if [v['body'] for v in config.visits]!=['399','499','20000001','499','399']:
        raise ValueError('This survey requires the five-visit EMC return sequence')
    config=deepcopy(config);config.backend='cuda';config.model='powered';config.batch_candidates=batch_size;config.validate()
    run=Path(outdir)/datetime.now(timezone.utc).strftime('emc-sweep-%Y%m%dT%H%M%S');run.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+seconds
    reserve=min(max(150,benchmark_seconds+120),seconds*.35)
    root_reserve=min(max(85,benchmark_seconds+55),seconds*.2)
    write_json(run/'sweep-parameters.json',dict(seconds=seconds,max_candidates=max_candidates,grid_step=grid_step,
        batch_size=batch_size,seed=seed,max_storage_gib=max_storage_gib,benchmark_seconds=benchmark_seconds))
    write_json(run/'config.json',config.to_dict())
    monitor=None
    done=threading.Event()
    def cpu_monitor():
        with (run/'cpu-monitor.csv').open('w') as stream:
            stream.write('elapsed_s,process_equivalent_cores,system_busy_percent,rss_mb\n')
            previous_wall=time.perf_counter();previous_cpu=time.process_time()
            counters=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));previous_total=sum(counters[:8]);previous_idle=counters[3]+counters[4]
            while not done.wait(1):
                wall=time.perf_counter();cpu=time.process_time()
                counters=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));total=sum(counters[:8]);idle=counters[3]+counters[4]
                rss=next(int(line.split()[1])/1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
                busy=100*(1-(idle-previous_idle)/max(1,total-previous_total))
                stream.write(f'{wall-started:.3f},{(cpu-previous_cpu)/(wall-previous_wall):.4f},{busy:.3f},{rss:.3f}\n');stream.flush()
                previous_wall,previous_cpu,previous_total,previous_idle=wall,cpu,total,idle
    watcher=threading.Thread(target=cpu_monitor,daemon=True);watcher.start()
    try:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
            '--format=csv','-l','1'],stdout=(run/'gpu-monitor.csv').open('w'),stderr=subprocess.DEVNULL)
    except OSError:pass
    timings={};best={};near=[];sampled=0;rejected_duration=0;benchmark_rows=None
    try:
        with MissionEvaluator(config,run/'state-cache') as e:
            fp=fingerprint(config,e)
            root=Path(__file__).resolve().parents[2]
            source_names=list(fp['modules'])+['src/research/mission_sweep.py','src/research/mission_sweep_plot.py','src/research/cli.py']
            fp['sweep_sources']={name:digest(root/name) for name in source_names}
            fp['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            for name in source_names:
                destination=run/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes((root/name).read_bytes())
            write_json(run/'fingerprint.json',fp)
            write_json(run/'policies.json',e.policy_list)
            atlas=Atlas(run,e.policy_list,e.e.mu)
            timings['porkchops']=porkchops(e,run,grid_step)
            generator=qmc.Sobol(len(e.bounds),scramble=True,seed=seed)
            # Known validated basins are labelled anchors, not new discoveries.
            anchors=[]
            for model in ('ballistic','powered'):
                p=Path(__file__).resolve().parents[3]/'Mission-planner-runs/accepted'/f'validated-{model}'/'best.json'
                if p.exists():anchors.append(json.loads(p.read_text())['decisions_days'])
            anchors_used=False;bulk_started=time.perf_counter();native_seconds=0.;storage_seconds=0.;blocks=0
            while sampled<max_candidates and time.perf_counter()<deadline-reserve and atlas.bytes<max_storage_gib*1024**3:
                u=generator.random_base2(int(math.log2(batch_size))) if blocks==0 and batch_size&(batch_size-1)==0 else generator.random(batch_size)
                x=qmc.scale(u,e.bounds[:,0],e.bounds[:,1])
                if not anchors_used:
                    x=np.vstack((anchors,x)) if anchors else x;anchors_used=True
                _,departure=e.timeline(x);duration=(departure[:,-1]-departure[:,0])/DAY
                rejected_duration+=int(np.count_nonzero(duration>config.max_duration_days));x=x[duration<=config.max_duration_days]
                x=x[:max_candidates-sampled]
                # Keep native batch capacity invariant even with anchor rows.
                if len(x)>batch_size:x=x[:batch_size]
                if not len(x):continue
                t=time.perf_counter();block=e.block(x);matches=ballistic_nodes(e,block);native_seconds+=time.perf_counter()-t
                sampled+=len(x);blocks+=1
                if benchmark_rows is None:benchmark_rows=block['inputs'].ravel().copy()
                for model in ('powered','ballistic'):
                    if model=='powered':score=block['feasible_score'];paths=block['feasible_path']
                    else:
                        valid0=joined_mask(block,1)&(matches[0]['status']==0)
                        valid1=joined_mask(block,3)&(matches[1]['status']==0)
                        links=[np.where(valid0,0.,np.inf),np.where(block['nodes'][1]['valid'],block['nodes'][1]['cost'],np.inf),np.where(valid1,0.,np.inf)]
                        score,paths=dynamic_path(np.where(block['c3_margin']>=0,block['start_cost'],np.inf),links,block['end_cost'])
                    finite=np.flatnonzero(np.isfinite(score))
                    key=model+'_complete_count';atlas.counts[key]=atlas.counts.get(key,0)+len(finite)
                    if len(finite):
                        i=finite[np.argmin(score[finite])]
                        if model not in best or score[i]<best[model]['total_delta_v_m_s']:
                            old=config.model;config.model=model
                            selected=subset(block,int(i))
                            if model=='ballistic':
                                for j,node in enumerate((1,3)):
                                    selected['nodes'][node-1]['native']=matches[j][i:i+1]
                            best[model]=e.describe(selected,paths[i],True)
                            config.model=old
                # Local speed-match seeds from geometrically valid, solar-safe
                # branch pairs. Their mismatch is diagnostic, never feasibility.
                for node in (1,3):
                    mask=joined_mask(block,node)&block['nodes'][node-1]['valid']
                    n,a,b=np.nonzero(mask)
                    if len(n):
                        gap=np.abs(block['nodes'][node-1]['speed_difference'][n,a,b]);ids=np.argsort(gap)[:12]
                        for k in ids:near.append(dict(gap=float(gap[k]),node=node,x=x[n[k]].tolist(),a=int(a[k]),b=int(b[k])))
                near=select_root_seeds(near)
                t=time.perf_counter();atlas.save(block,matches);storage_seconds+=time.perf_counter()-t
                if blocks%4==0:
                    progress=dict(stage='joined_sweep',candidates=sampled,lambert_attempts=sampled*4*e.K,flyby_attempts=sampled*4*e.K*e.K,
                        connections=atlas.counts,elapsed_seconds=time.perf_counter()-started)
                    write_json(run/'progress.json',progress);print(json.dumps(progress),flush=True)
                    write_json(run/'ballistic-root-seeds.json',near)
                    for model,candidate in best.items():write_json(run/f'checkpoint-best-{model}.json',candidate)
                if blocks%32==0:
                    atlas.export()
                    write_json(run/'checkpoint.json',dict(stage='joined_sweep',completed_chunks=atlas.serial,
                        sobol_generated=int(generator.num_generated),candidates=sampled,elapsed_seconds=time.perf_counter()-started,
                        data_bytes=atlas.bytes,counts=atlas.counts))
            timings['sweep']=dict(candidates=sampled,blocks=blocks,wall_seconds=time.perf_counter()-bulk_started,
                evaluation_seconds=native_seconds,storage_atlas_seconds=storage_seconds,duration_rejected=rejected_duration,
                lambert_attempts=sampled*4*e.K,flyby_attempts=sampled*4*e.K*e.K)
            write_json(run/'ballistic-root-seeds.json',near)
            root_started=time.perf_counter();roots=0;root_attempts=0
            for root_seed in near:
                if time.perf_counter()>deadline-root_reserve:break
                cfg=deepcopy(config);cfg.backend='cpu';cfg.model='ballistic'
                choices=[e.policy_list[0]]*4;node=root_seed['node'];choices[node-1]=e.policy_list[root_seed['a']];choices[node]=e.policy_list[root_seed['b']]
                with MissionEvaluator(cfg,run/'state-cache',choices,ephemeris=e.e) as local:
                    x=np.array(root_seed['x']);col=node+1;lo,hi=local.bounds[col]
                    def value(t):
                        y=x.copy();y[col]=t;b=local.block(y[None,:])
                        if np.any(b['legs']['status'][0,[node-1,node]]):return math.nan
                        return float(b['nodes'][node-1]['speed_difference'][0,0,0])
                    ticks=np.unique(np.r_[np.linspace(lo,hi,17),x[col]])
                    values=[value(t) for t in ticks]
                    for i in range(len(ticks)-1):
                        if time.perf_counter()>deadline-root_reserve:break
                        if not np.isfinite(values[i:i+2]).all() or values[i]*values[i+1]>0:continue
                        root_attempts+=1
                        try:t=brentq(value,ticks[i],ticks[i+1],xtol=1e-10,rtol=1e-14)
                        except ValueError:continue
                        y=x.copy();y[col]=t;b=local.block(y[None,:])
                        if b['duration_days'][0]>config.max_duration_days:continue
                        if b['nodes'][node-1]['native']['status'][0,0,0]!=0 or not joined_mask(b,node)[0,0,0]:continue
                        # Remap the K=1 local block to full-policy storage IDs
                        # by a separate saved chunk with its explicit policies.
                        path=run/'ballistic-roots'/f'{roots:05d}';path.mkdir(parents=True,exist_ok=True)
                        native=b['nodes'][node-1]['native'];metrics=encounter_metrics(b['arrival_states'][0,node,:3],b['arrival_states'][0,node,3:],
                            b['legs']['v2'][0,node-1,0]-b['arrival_states'][0,node,3:],b['legs']['v1'][0,node,0]-b['departure_states'][0,node,3:],
                            np.cross((b['arrival_states'] if node==1 else b['departure_states'])[0,2,:3],
                                     (b['arrival_states'] if node==1 else b['departure_states'])[0,2,3:]))
                        write_json(path/'connection.json',dict(node=node,decisions_days=y,policies=choices,metrics=metrics,
                            native={k:native[k][0,0,0] for k in native.dtype.names},complete_mission=False))
                        np.savez_compressed(path/'data.npz',decisions=y,arrivals=b['arrivals'],departures=b['departures'],arrival_states=b['arrival_states'],
                            departure_states=b['departure_states'],v1=b['legs']['v1'],v2=b['legs']['v2'])
                        # All ballistic root connections enter the same atlas.
                        atlas.record(b,node,'ballistic',native,(native['status']==0)&joined_mask(b,node))
                        roots+=1
            timings['ballistic_refinement']=dict(accepted_local_connections=roots,brackets=root_attempts,wall_seconds=time.perf_counter()-root_started)
            fbrows=np.zeros(min(65536,len(block['decisions'])*e.K*e.K),dtype=FB_INPUT)
            vi=block['legs']['v2'][:,0]-block['arrival_states'][:,1,None,3:]
            vo=block['legs']['v1'][:,1]-block['departure_states'][:,1,None,3:]
            fbrows['vin']=np.broadcast_to(vi[:,:,None,:],(len(vi),e.K,e.K,3)).reshape(-1,3)[:len(fbrows)]
            fbrows['vout']=np.broadcast_to(vo[:,None,:,:],(len(vo),e.K,e.K,3)).reshape(-1,3)[:len(fbrows)]
            fbrows['reference']=[0,0,1];fbrows['mu']=e.mu[1];fbrows['rp_min']=3596200.;fbrows['rp_max']=3e8
            fbrows['speed_atol']=config.speed_atol_m_s;fbrows['speed_rtol']=config.speed_rtol;fbrows['mode']=int(Mode.POWERED)
            timings['benchmarks']=benchmarks(benchmark_rows,fbrows,budget=min(benchmark_seconds,max(2,deadline-time.perf_counter()-40)))
            for model,candidate in best.items():
                cfg=deepcopy(config);cfg.model=model;modelrun=run/model;modelrun.mkdir()
                write_json(modelrun/'config.json',cfg.to_dict());candidate['fingerprint']=fingerprint(cfg,e)
                write_json(modelrun/'best.json',candidate)
            atlas.export()
            write_json(run/'timings.json',timings)
            summary=dict(status='completed_bounded_sweep',seconds_budget=seconds,elapsed_seconds=time.perf_counter()-started,
                candidates=sampled,anchors=len(anchors),counts=atlas.counts,ballistic_refined_connections=roots,solar_mu_m3_s2=e.e.mu,
                seed=seed,max_storage_gib=max_storage_gib,
                best_delta_v_m_s={k:v['total_delta_v_m_s'] for k,v in best.items()},data_bytes=atlas.bytes,
                track_points_outside_display=atlas.clipped,scope='two-body impulsive patched conics; finite sampled survey, not exhaustive or N-body validated',
                graph_scope='All accepted sampled connections in metrics; all participating leg loci sampled uniformly in true anomaly, not time; 3D highlights are a subset.')
            write_json(run/'summary.json',summary)
        from .mission_sweep_plot import plot_sweep
        plot_sweep(run)
        # SPICE ownership is closed before independent validation reopens it.
        from .mission_validation import validate_candidate
        for model in best:
            if time.perf_counter()<deadline:
                result=validate_candidate(run/model/'best.json')
                summary.setdefault('independent_best_validation',{})[model]=result['status']
        summary['elapsed_seconds']=time.perf_counter()-started;write_json(run/'summary.json',summary)
        return run,summary
    except BaseException as error:
        import traceback
        write_json(run/'failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started,candidates=sampled))
        if 'atlas' in locals():
            atlas.export()
            for model,candidate in best.items():write_json(run/f'checkpoint-best-{model}.json',candidate)
        raise
    finally:
        done.set();watcher.join(timeout=3)
        if monitor is not None:monitor.terminate();monitor.wait(timeout=5)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=900);p.add_argument('--max-candidates',type=int,default=1048576)
    p.add_argument('--grid-step',type=float,default=2);p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--seed',type=int,default=958720);p.add_argument('--max-storage-gib',type=float,default=8.)
    p.add_argument('--benchmark-seconds',type=float,default=30.)
    a=p.parse_args();run,summary=run_sweep(load_mission(a.config),a.outdir,a.seconds,a.max_candidates,a.grid_step,a.batch_size,
        seed=a.seed,max_storage_gib=a.max_storage_gib,benchmark_seconds=a.benchmark_seconds)
    print(json.dumps(dict(run=str(run),**summary)),flush=True)
