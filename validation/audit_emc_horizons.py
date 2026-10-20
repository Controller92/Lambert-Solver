"""Independent public Horizons ephemeris check for the nominal EMC itinerary.

--refresh is explicit network access; normal operation verifies cached snapshots.
Compares current Horizons to the exact SPICE states used in the main audit, and
re-solves both encounters independently with Gooding's Lambert algorithm.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
from lamberthub import gooding1990
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_emc import clean,patch,norm


def check(folder,refresh=False):
    report=json.loads((folder/'nominal.json').read_text())
    dates=sorted(set(t for s in report['scenarios'] for t in s['epochs_tdb']))
    raw=folder/'horizons';raw.mkdir(exist_ok=True)
    objects=[('earth',"'399'",'Earth (399)'),('mars',"'499'",'Mars (499)'),('ceres',"'1;'",'1 Ceres')]
    states={};maximum={}
    for label,command,body_label in objects:
        params=dict(COMMAND=command,OBJ_DATA="'YES'",MAKE_EPHEM="'YES'",EPHEM_TYPE="'VECTORS'",
            CENTER="'500@10'",REF_PLANE="'ECLIPTIC'",REF_SYSTEM="'ICRF'",OUT_UNITS="'KM-S'",VEC_TABLE="'2'",
            VEC_CORR="'NONE'",CSV_FORMAT="'YES'",TIME_TYPE="'TDB'",TLIST_TYPE="'CAL'",
            TLIST=' '.join("'"+d.replace('T',' ')+"'" for d in dates))
        file=raw/f'{label}.json'
        if refresh:
            url='https://ssd.jpl.nasa.gov/api/horizons.api?'+urlencode(dict(format='json',**params))
            with urlopen(url,timeout=60) as f:payload=json.load(f)
            if 'error' in payload or '$$SOE' not in payload.get('result',''):raise RuntimeError(str(payload)[:1000])
            file.write_text(json.dumps(dict(url=url,parameters=params,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                response=payload,result_sha256=hashlib.sha256(payload['result'].encode()).hexdigest()),indent=2)+'\n')
        snapshot=json.loads(file.read_text());text=snapshot['response']['result']
        if snapshot['parameters']!=params or snapshot['result_sha256']!=hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('Horizons snapshot mismatch')
        header=text.split('$$SOE')[0]
        for item in (body_label,'Sun (10)','BODY CENTER','GEOMETRIC cartesian states','Output units    : KM-S',
                     'Reference frame : Ecliptic of J2000.0','JDTDB','Date (TDB)'):
            if item not in header:raise ValueError(f'Unexpected {label} header: {item}')
        values={}
        for row in csv.reader(io.StringIO(text.split('$$SOE')[1].split('$$EOE')[0])):
            if not row:continue
            if len(row)!=9 or row[-1].strip():raise ValueError('Unexpected columns')
            stamp=datetime.strptime(row[1].strip(),'A.D. %Y-%b-%d %H:%M:%S.%f').isoformat()
            values[stamp]=np.asarray(row[2:8],float)*1000.
        if set(values)!=set(dates):raise ValueError('Unexpected date table')
        if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite states')
        states[label]=values
        differences=[]
        for s in report['scenarios']:
            for leg in s['legs']:
                for endpoint,key,bodykey in [('departure','departure_state_si','origin'),('arrival','arrival_state_si','destination')]:
                    if leg[bodykey]!={'earth':'399','mars':'499','ceres':'20000001'}[label]:continue
                    when=leg[endpoint+'_tdb'];delta=values[when]-np.asarray(leg[key])
                    differences.append(dict(epoch_tdb=when,position_difference_m=norm(delta[:3]),velocity_difference_m_s=norm(delta[3:])))
        maximum[label]=dict(max_position_difference_m=max(r['position_difference_m'] for r in differences),
            max_velocity_difference_m_s=max(r['velocity_difference_m_s'] for r in differences),samples=differences)
    scenarios=[]
    for s in report['scenarios']:
        legs=[]
        for leg in s['legs']:
            names={'399':'earth','499':'mars','20000001':'ceres'}
            a=states[names[leg['origin']]][leg['departure_tdb']];b=states[names[leg['destination']]][leg['arrival_tdb']]
            duration=(datetime.fromisoformat(leg['arrival_tdb'])-datetime.fromisoformat(leg['departure_tdb'])).total_seconds()
            v1,v2=gooding1990(report['solar_mu_m3_s2'],a[:3],b[:3],duration,M=0,prograde=True,maxiter=100,atol=1e-13,rtol=1e-13)
            legs.append(dict(inf1=v1-a[3:],inf2=v2-b[3:]))
        encounters=[patch(legs[i]['inf2'],legs[j]['inf1'],report['mars_mu_m3_s2']) for i,j in ((0,1),(2,3))]
        scenarios.append(dict(name=s['name'],encounters=encounters))
    output=dict(conventions='Heliocentric geometric, Ecliptic J2000, TDB, SI',state_comparisons=maximum,
        independent_gooding_encounters=scenarios,snapshot_hashes={name:json.loads((raw/f'{name}.json').read_text())['result_sha256'] for name in states})
    (folder/'horizons-crosscheck.json').write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean(output),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--refresh',action='store_true')
    a=p.parse_args();check(a.folder,a.refresh)
