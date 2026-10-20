"""Vectorized cell assembly, explicit candidate policies and bounded native batches."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
from ..lambert_native import NativeEngine, INPUT_DTYPE
from ..lambert_root import RootOptions
from ..lambert_multirev import MinimumOptions
from .ephemeris import SpiceEphemeris, sha256


def policies(config):
    orientations = ('short','long') if config.orientation == 'both' else (config.orientation,)
    return [(m,branch,way) for m in range(config.max_revolutions+1)
            for branch in ((-1,) if m==0 else (0,1)) for way in orientations]


def compute_grid(config, dep_et, tof_days, dep_states, arr_states, arrival_index, mu,
                 engine=None, progress=None):
    """Inputs and v-infinity outputs use SI. C3 output uses km^2/s^2.

    Every candidate retains status diagnostics. Only successful finite velocities
    enter the minimum-C3 reduction; ties retain the first deterministic policy.
    Warnings are preserved, never silently excluded or repaired.
    """
    engine = engine or NativeEngine(config.backend)
    nd,nt = len(dep_et),len(tof_days)
    shape = (nd,nt)
    candidates = policies(config)
    output = dict(C3=np.full(shape,np.nan),vinf_dep=np.full(shape+(3,),np.nan),
                  vinf_arr=np.full(shape+(3,),np.nan),revolutions=np.full(shape,-1,dtype='i4'),
                  branch=np.full(shape,-1,dtype='i4'),long_way=np.full(shape,-1,dtype='i4'),
                  precision_limited=np.full(shape,-1,dtype='i4'),relative_residual=np.full(shape,np.nan),
                  estimated_x_error=np.full(shape,np.nan),iterations=np.zeros(shape,dtype='i8'))
    for name in ('status','geometry_status','root_status','minimum_status','precision'):
        output['candidate_'+name] = np.full((len(candidates),nd,nt),-1,dtype='i1')
    output['candidates'] = np.array([(m,b,('short','long','prograde','retrograde').index(w)) for m,b,w in candidates],dtype='i4')
    flat = {k:v.reshape((-1,)+v.shape[2:]) for k,v in output.items() if k not in ('candidates',) and not k.startswith('candidate_')}
    capacity = min(config.batch_size,nd*nt)
    context = engine.cuda_batch(capacity) if config.backend=='cuda' else nullcontext(None)
    kernel_ms = 0.
    with context as batch:
        for start in range(0,nd*nt,capacity):
            stop = min(nd*nt,start+capacity)
            cells = np.arange(start,stop)
            i,j = cells//nt,cells%nt
            departure = dep_states[i]
            arrival = arr_states[arrival_index[i,j]]
            rows = np.zeros(len(cells),dtype=INPUT_DTYPE)
            rows['r1'],rows['r2'] = departure[:,:3],arrival[:,:3]
            rows['tof'],rows['mu'] = tof_days[j]*86400.,mu
            hz = np.cross(rows['r1'],rows['r2'])[:,2]
            for index,(m,branch,way) in enumerate(candidates):
                long = (np.full(len(cells),way=='long') if way in ('short','long')
                        else hz<0 if way=='prograde' else hz>0)
                rows['long_way'],rows['revolutions'],rows['branch'] = long,m,max(0,branch)
                if batch is None:
                    actual = engine.cpu(rows)
                else:
                    batch.upload(rows); kernel_ms += batch.run(); actual = batch.download()
                # A fixed-axis prograde/retrograde selection is undefined for a
                # plane whose normal has exactly zero z component.
                undefined = hz==0 if way in ('prograde','retrograde') else np.zeros(len(cells),dtype=bool)
                for name,values in (('status',actual['status']),('geometry_status',actual['geometry_status']),
                                    ('root_status',actual['root']['status']),('minimum_status',actual['minimum']['status']),
                                    ('precision',actual['root']['precision_limited'])):
                    output['candidate_'+name][index].flat[start:stop] = values
                output['candidate_status'][index].flat[start:stop] = np.where(undefined,4,actual['status'])
                vdep = actual['v1']-departure[:,3:]
                varr = actual['v2']-arrival[:,3:]
                with np.errstate(over='ignore',invalid='ignore'):
                    c3 = np.einsum('ij,ij->i',vdep,vdep)/1e6
                eligible = ((actual['status']==0)&~undefined&np.isfinite(c3)
                            &np.isfinite(vdep).all(axis=1)&np.isfinite(varr).all(axis=1))
                output['candidate_status'][index].flat[start:stop] = np.where(
                    (actual['status']==0)&~undefined&~eligible,5,output['candidate_status'][index].flat[start:stop])
                choose = eligible & (~np.isfinite(flat['C3'][start:stop]) | (c3<flat['C3'][start:stop]))
                selected = cells[choose]
                for name,value in (('C3',c3),('vinf_dep',vdep),('vinf_arr',varr),
                                    ('precision_limited',actual['root']['precision_limited']),
                                    ('relative_residual',actual['root']['relative_residual']),
                                    ('estimated_x_error',actual['root']['estimated_x_error']),
                                    ('iterations',actual['root']['iterations'])):
                    flat[name][selected] = value[choose]
                flat['revolutions'][selected] = m
                flat['branch'][selected] = branch
                flat['long_way'][selected] = long[choose]
            if progress: progress(stop,nd*nt)
    output['dep_et'] = dep_et
    output['tof_days'] = tof_days
    return output,dict(kernel_ms=kernel_ms if config.backend=='cuda' else None,
                       candidate_solves=nd*nt*len(candidates),policies=candidates)


def run_grid(config,out_root='results',cache_dir='.research-cache',progress=None):
    config.validate()
    import spiceypy
    started = time.perf_counter()
    with SpiceEphemeris(config.kernels,config.frame,cache_dir) as provider:
        nd,nt = config.shape
        first = provider.epoch(config.dep_start)
        dep_et = first+np.arange(nd)*config.dep_step*86400.
        tof_days = config.tof_min+np.arange(nt)*config.tof_step
        # Query each distinct arrival epoch once, not once for each grid cell.
        arrivals,inverse = np.unique(dep_et[:,None]+tof_days[None,:]*86400.,return_inverse=True)
        dep_states = provider.states(config.dep_body,dep_et)
        arr_states = provider.states(config.arr_body,arrivals)
        states_elapsed = time.perf_counter()-started
        engine = NativeEngine(config.backend)
        output,numerical = compute_grid(config,dep_et,tof_days,dep_states,arr_states,
                                         inverse.reshape(nd,nt),provider.mu,engine,progress)
        solve_elapsed = time.perf_counter()-started-states_elapsed
        provenance = provider.provenance
        calls,hits = provider.calls,provider.cache_hits
        mu = provider.mu
    # Preserve the exact SI input states so spot checks do not need SPICE/network.
    output.update(dep_states=dep_states,arrival_et=arrivals,arrival_states=arr_states,
                  arrival_index=inverse.reshape(nd,nt))
    path = Path(out_root)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    path = path.with_name(path.name+'-'+uuid.uuid4().hex[:8])
    path.mkdir(parents=True,exist_ok=False)
    data_path = path/'grid.npz'
    np.savez(data_path,**output)
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True))
    except (OSError,subprocess.CalledProcessError):
        commit,dirty = None,None
    lib_path = Path(engine.lib._name)
    valid = np.isfinite(output['C3'])
    minimum = None
    if valid.any():
        i,j = np.unravel_index(np.nanargmin(output['C3']),output['C3'].shape)
        minimum = dict(departure_et=float(dep_et[i]),tof_days=float(tof_days[j]),
                       C3_km2_s2=float(output['C3'][i,j]),revolutions=int(output['revolutions'][i,j]),
                       branch=int(output['branch'][i,j]),long_way=int(output['long_way'][i,j]))
    minimum_path = path/'minimum.csv'
    with minimum_path.open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('departure_tdb','arrival_tdb','tof_days','C3_km2_s2','M','branch','long_way','precision_limited'))
        for i,et in enumerate(dep_et):
            departure = (datetime(2000,1,1,12)+timedelta(seconds=float(et))).isoformat()
            if not valid[i].any():
                writer.writerow((departure,'','','','','','',''))
                continue
            j = int(np.nanargmin(output['C3'][i]))
            arrival = (datetime(2000,1,1,12)+timedelta(seconds=float(et+tof_days[j]*86400.))).isoformat()
            writer.writerow((departure,arrival,float(tof_days[j]),float(output['C3'][i,j]),
                             int(output['revolutions'][i,j]),int(output['branch'][i,j]),
                             int(output['long_way'][i,j]),int(output['precision_limited'][i,j])))
    metadata = dict(schema=1,config=config.to_dict(),created_utc=datetime.now(timezone.utc).isoformat(),
                    center='SUN',aberration='NONE',mu_m3_s2=mu,kernels=provenance,
                    units=dict(states='m and m/s',vinf='m/s',C3='km^2/s^2',et='TDB seconds past J2000',tof='86400-second days'),
                    native_library_sha256=sha256(lib_path),commit=commit,dirty_tree=dirty,
                    workflow_sources={str(p.relative_to(root)):sha256(p) for p in [root/'run.py',*sorted((root/'src/research').glob('*.py'))]},
                    python=platform.python_version(),numpy=np.__version__,spiceypy=spiceypy.__version__,spice_toolkit=spiceypy.tkvrsn('TOOLKIT'),root_options=asdict(RootOptions()),
                    minimum_options=asdict(MinimumOptions()),valid_cells=int(valid.sum()),total_cells=nd*nt,
                    warning_cells=int(np.sum(valid&(output['precision_limited']==1))),
                    unknown_precision_cells=int(np.sum(valid&(output['precision_limited']==-1))),minimum=minimum,
                    ephemeris_calls=calls,cached_states=hits,grid_sha256=sha256(data_path),minimum_table_sha256=sha256(minimum_path),
                    state_seconds=states_elapsed,solve_seconds=solve_elapsed,**numerical)
    if config.backend=='cuda': metadata['device'] = engine.device_info(multi=config.max_revolutions>0)
    # Write provenance before plotting so a rendering failure cannot orphan data.
    metadata['plot_status'] = 'pending' if config.plot else 'disabled'
    (path/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    manifest = path/'run.json'
    manifest.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    if config.plot:
        from .plot import plot_grid
        try:
            plot_grid(data_path,path/'porkchop.png',metadata)
            metadata['plot_status'] = 'ok'
        except Exception as error:
            metadata['plot_status'] = 'failed'
            metadata['plot_error'] = str(error)
            manifest.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            raise RuntimeError(f'Grid saved to {path}; plot failed: {error}') from error
    metadata['total_seconds'] = time.perf_counter()-started
    manifest.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return path,metadata
