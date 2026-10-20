"""Explicit, cached JPL Horizons comparison at a saved candidate's event epochs.

Horizons is an independent ephemeris snapshot, not an N-body trajectory check.
Normal use is offline; --refresh in the CLI is the explicit network operation.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
from lamberthub import gooding1990
from .mission_search import digest, write_json


def crosscheck_candidate(path, refresh=False):
    path = Path(path).resolve(); c = json.loads(path.read_text())
    raw = path.parent/(path.stem+'-horizons'); raw.mkdir(exist_ok=True)
    commands = {'399':("'399'",'Earth (399)'), '499':("'499'",'Mars (499)'), '20000001':("'1;'",'1 Ceres')}
    events = {}
    for leg in c['legs']:
        for side,bodykey in (('departure','origin'),('arrival','destination')):
            events.setdefault(leg[bodykey],{})[leg[side+'_et']] = np.asarray(leg[side+'_state_si'])
    comparisons = []; states = {}; snapshots = {}
    for body,table in events.items():
        if body not in commands: raise ValueError(f'Horizons command for {body} is not configured')
        command,label = commands[body]; epochs = sorted(table)
        params = dict(COMMAND=command,OBJ_DATA="'YES'",MAKE_EPHEM="'YES'",EPHEM_TYPE="'VECTORS'",
            CENTER="'500@10'",REF_PLANE="'ECLIPTIC'" if c['frame']=='ECLIPJ2000' else "'FRAME'",
            REF_SYSTEM="'ICRF'",OUT_UNITS="'KM-S'",VEC_TABLE="'2'",VEC_CORR="'NONE'",
            CSV_FORMAT="'YES'",TIME_TYPE="'TDB'",TLIST_TYPE="'JD'",TIME_DIGITS="'FRACSEC'",
            TLIST=' '.join(f'{2451545.+t/86400.:.12f}' for t in epochs))
        file = raw/(body+'.json')
        if refresh:
            url = 'https://ssd.jpl.nasa.gov/api/horizons.api?'+urlencode(dict(format='json',**params))
            with urlopen(url,timeout=60) as response: payload = json.load(response)
            if 'error' in payload or '$$SOE' not in payload.get('result',''): raise ValueError(str(payload)[:1000])
            write_json(file,dict(url=url,parameters=params,retrieved_utc=datetime.now(timezone.utc).isoformat(),response=payload,
                result_sha256=hashlib.sha256(payload['result'].encode()).hexdigest()))
        snapshot = json.loads(file.read_text()); text = snapshot['response']['result']
        if snapshot['parameters'] != params or snapshot['result_sha256'] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('Horizons snapshot identity/hash differs')
        header = text.split('$$SOE')[0]
        expected_frame = 'Ecliptic of J2000.0' if c['frame']=='ECLIPJ2000' else 'ICRF'
        for item in (label,'Sun (10)','BODY CENTER','GEOMETRIC cartesian states','Output units    : KM-S',expected_frame,'JDTDB','Date (TDB)'):
            if item not in header: raise ValueError(f'Unexpected Horizons header: {item}')
        rows = [row for row in csv.reader(io.StringIO(text.split('$$SOE')[1].split('$$EOE')[0])) if row]
        if len(rows) != len(epochs): raise ValueError('Unexpected number of Horizons events')
        for t,row in zip(epochs,rows):
            if len(row) != 9 or row[-1].strip(): raise ValueError('Unexpected Horizons columns')
            returned_et = (float(row[0])-2451545.)*86400.; dt = returned_et-t
            if abs(dt) > .001: raise ValueError('Horizons time differs by more than 1 ms')
            state = np.asarray(row[2:8],float)*1000.
            if not np.isfinite(state).all(): raise ValueError('Nonfinite Horizons state')
            # Correct the tiny JD binary-rounding difference to the requested
            # epoch with a first-order translation (acceleration term negligible).
            state[:3] -= state[3:]*dt
            delta = state-table[t]
            comparisons.append(dict(body=body,et=t,returned_epoch_offset_s=dt,
                position_difference_m=float(np.linalg.norm(delta[:3])),velocity_difference_m_s=float(np.linalg.norm(delta[3:]))))
            states[body,t] = state
        snapshots[body] = digest(file)
    velocity_checks = []
    for i,leg in enumerate(c['legs']):
        a = states[leg['origin'],leg['departure_et']]; b = states[leg['destination'],leg['arrival_et']]
        m = leg['policy']['revolutions']; errors = []
        prograde = bool((np.cross(a[:3],b[:3])[2]>0) != leg['long_way'])
        for low in ((False,) if m==0 else (False,True)):
            try:
                v,w = gooding1990(c['solar_mu_m3_s2'],a[:3],b[:3],leg['arrival_et']-leg['departure_et'],
                    M=m,prograde=prograde,low_path=low,maxiter=100,atol=1e-13,rtol=1e-13)
                errors.append(max(np.linalg.norm(v-np.asarray(leg['v1_m_s'])),np.linalg.norm(w-np.asarray(leg['v2_m_s']))))
            except ValueError: pass
        velocity_checks.append(dict(leg=i,maximum_endpoint_velocity_change_m_s=float(min(errors,default=float('inf')))))
    passed = (all(r['position_difference_m']<=10. and r['velocity_difference_m_s']<=1e-4 for r in comparisons)
              and all(r['maximum_endpoint_velocity_change_m_s']<=.001 for r in velocity_checks))
    report = dict(status='consistent_ephemerides' if passed else 'ephemeris_difference_requires_review',candidate_sha256=digest(path),
        comparisons=comparisons,gooding_recomputed_with_horizons=velocity_checks,snapshot_sha256=snapshots,
        scope='Snapshot consistency only; does not replace conic validation or N-body correction.',
        conventions='Sun centred, geometric NONE, TDB, '+c['frame']+', SI')
    write_json(path.with_suffix('.horizons.json'),report)
    return report
