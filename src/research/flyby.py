"""Saved, reproducible batches of encounters at a specified body/epoch/frame."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
from ..flyby import Encounter,Mode,Status
from ..flyby_native import FlybyEngine,pack_encounters


def load_request(path):
    raw=Path(path).read_bytes(); data=json.loads(raw)
    allowed={'schema_version','body','frame','epoch_tdb','mu_m3_s2','radius_m','min_altitude_m',
             'rp_max_m','velocity_origin','body_velocity_m_s','reference','speed_atol_m_s',
             'speed_rtol','encounters'}
    if not isinstance(data,dict) or set(data)-allowed or type(data.get('schema_version'))!=int or data.get('schema_version')!=1:
        raise ValueError('Expected flyby schema_version 1 with known keys')
    if not isinstance(data.get('body'),str) or not data['body'].strip(): raise ValueError('A body identifier is required')
    if data.get('frame') not in ('J2000','ICRF','ECLIPJ2000'): raise ValueError('Use an explicit supported inertial frame')
    epoch=datetime.fromisoformat(data['epoch_tdb'])
    if epoch.tzinfo is not None: raise ValueError('epoch_tdb must be a timezone-free TDB calendar date')
    if data.get('velocity_origin') not in ('body_relative','inertial'): raise ValueError('Specify velocity_origin')
    def scalar(key,default=None):
        x=data.get(key,default)
        if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):
            raise ValueError(f'{key} must be a finite real scalar')
        return float(x)
    mu=scalar('mu_m3_s2');radius=scalar('radius_m');alt=scalar('min_altitude_m');maximum=scalar('rp_max_m')
    if mu<=0 or radius<=0 or alt<0 or maximum<radius+alt: raise ValueError('Invalid GM, radius or periapsis limits')
    atol=scalar('speed_atol_m_s',1e-6);rtol=scalar('speed_rtol',1e-12)
    if atol<0 or rtol<0: raise ValueError('Negative speed tolerance')
    def vector(value):
        if not isinstance(value,list) or len(value)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):
            raise ValueError('Expected three finite real vector components')
        return np.asarray(value,dtype='f8')
    body=vector(data.get('body_velocity_m_s',[0.,0.,0.]))
    if data['velocity_origin']=='inertial' and 'body_velocity_m_s' not in data:
        raise ValueError('Inertial input requires body_velocity_m_s in the same frame and at the same epoch')
    reference=tuple(vector(data.get('reference',[0.,0.,1.])))
    items=data.get('encounters')
    if not isinstance(items,list) or not 1<=len(items)<=100000: raise ValueError('Expected 1..100000 encounters; use the native API for larger batches')
    problems=[]
    for item in items:
        if not isinstance(item,dict) or set(item)-{'mode','vin_m_s','vout_m_s','rp_m','azimuth_rad'}:
            raise ValueError('Unknown encounter keys')
        mode={'forward':Mode.FORWARD,'match':Mode.MATCH,'powered':Mode.POWERED}.get(item.get('mode'))
        if mode is None: raise ValueError('Mode must be forward, match or powered')
        vin=vector(item['vin_m_s'])
        if mode==Mode.FORWARD:
            if 'vout_m_s' in item: raise ValueError('Forward mode does not accept a target velocity')
            target=np.zeros(3)
            rp=item['rp_m'];azimuth=item.get('azimuth_rad',0.)
        else:
            if 'rp_m' in item or 'azimuth_rad' in item: raise ValueError('Inverse mode solves periapsis and azimuth')
            target=vector(item['vout_m_s']);rp=0.;azimuth=0.
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (rp,azimuth)):
            raise ValueError('Periapsis and azimuth must be finite real values')
        if data['velocity_origin']=='inertial':
            vin=vin-body
            if mode!=Mode.FORWARD: target=target-body
        problems.append(Encounter(tuple(vin),mu,radius+alt,maximum,mode,tuple(target),reference,rp,azimuth,atol,rtol))
    return data,pack_encounters(problems),body,hashlib.sha256(raw).hexdigest()


def run_encounters(config,outdir='results',backend='cpu',batch_size=65536):
    if type(batch_size)!=int or batch_size<1: raise ValueError('batch_size must be positive')
    started=time.perf_counter()
    data,inputs,body,config_hash=load_request(config)
    engine=FlybyEngine(backend)
    from ..flyby_native import RESULT_DTYPE
    output=np.empty(len(inputs),dtype=RESULT_DTYPE)
    kernel_ms=0.;solve_started=time.perf_counter()
    if backend=='cuda':
        with engine.cuda_batch(min(len(inputs),batch_size)) as batch:
            for lo in range(0,len(inputs),batch_size):
                hi=min(len(inputs),lo+batch_size)
                batch.upload(inputs[lo:hi]);kernel_ms+=batch.run();output[lo:hi]=batch.download()
    else:
        for lo in range(0,len(inputs),batch_size): output[lo:lo+batch_size]=engine.cpu(inputs[lo:lo+batch_size])
    solve_seconds=time.perf_counter()-solve_started
    # An inertial output is available only when the body's velocity was supplied.
    inertial=output['vout']+body if 'body_velocity_m_s' in data else np.full((len(output),3),np.nan)
    run=Path(outdir)/('flyby-'+uuid.uuid4().hex[:12]);run.mkdir(parents=True,exist_ok=False)
    artifact=run/'encounters.npz'
    np.savez_compressed(artifact,inputs=inputs,results=output,vout_inertial_m_s=inertial)
    (run/'request.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    values,counts=np.unique(output['status'],return_counts=True)
    metadata=dict(schema_version=1,model='two-body patched conic; powered = tangential common-periapsis impulse',
        backend=backend,body=data['body'],frame=data['frame'],epoch_tdb=data['epoch_tdb'],units='m, s, rad',
        source_config_sha256=config_hash,request_sha256=hashlib.sha256((run/'request.json').read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(engine.library.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        status_counts={Status(int(v)).name:int(c) for v,c in zip(values,counts)},
        count=len(inputs),precision_warning_count=int(np.sum(output['precision_limited']==1)),solve_seconds=solve_seconds,kernel_ms=kernel_ms if backend=='cuda' else None,
        total_seconds=time.perf_counter()-started,inertial_output_available='body_velocity_m_s' in data)
    (run/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(f'Flyby {backend}: {len(inputs):,} encounters; {metadata["status_counts"]}')
    print(f'Saved: {run.resolve()}')
    return run,metadata
