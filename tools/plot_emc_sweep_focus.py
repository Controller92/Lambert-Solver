"""Readable prograde/low-burn highlights and independent local-root checks.

Adds derived artifacts without changing the original survey or its selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.mission_search import write_json
from src.research.mission_validation import asymptote


def focused_plot(run,burn_limit=5.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    evidence={};colors={'ballistic':'#2878a5','powered':'#c46b25'}
    for row,node in enumerate((1,3)):
        for model in ('ballistic','powered'):
            path=run/f'highlights-{model}-{node}.npz'
            with np.load(path,allow_pickle=False) as data:
                hi=np.cross(data[f'r{node-1}'],data[f'v{node-1}']);ho=np.cross(data[f'r{node}'],data[f'v{node}'])
                keep=(hi[:,2]>0)&(ho[:,2]>0)&(data['burn_km_s']<=burn_limit)
                evidence[f'{model}-{node}']=dict(input_highlights=len(keep),selected=int(keep.sum()),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for col,field in enumerate(('plane_change_deg','ceres_plane_after_deg')):
                    axes[row,col].scatter(data[field][keep],data['heliocentric_speed_gain_km_s'][keep],
                        c=colors[model],s=28 if model=='ballistic' else 50,marker='o' if model=='ballistic' else '^',
                        alpha=.8,edgecolors='white',linewidths=.3,label=f'{model.capitalize()} (n={keep.sum()})')
        for col,ax in enumerate(axes[row]):
            ax.axhline(0,color='.5',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=9)
            ax.set(xlabel='Heliocentric orbital-plane rotation (deg)' if col==0 else 'Post-flyby Ceres-plane misalignment (deg)',
                   ylabel='Heliocentric speed change (km/s)',title=('Outbound' if node==1 else 'Return')+' Mars encounter')
    fig.suptitle(f'Prograde on both legs | powered periapsis burn ≤ {burn_limit:g} km/s\n'
        'Filtered saved highlights: all ballistic rows, bounded uniform powered sample; local connections, not full missions',fontsize=12)
    destination=run/'prograde-low-burn-focus.png';fig.savefig(destination,dpi=180);plt.close(fig)
    write_json(run/'prograde-low-burn-focus.json',dict(burn_limit_km_s=burn_limit,
        scope='Filtered highlight set, not a complete population count or an optimality bound',groups=evidence))
    return destination


def check_ballistic_roots(run):
    """Independent hyperbola recovery and Gooding solves for every refined root."""
    from lamberthub import gooding1990
    run=Path(run);summary=json.loads((run/'summary.json').read_text());mu=summary['solar_mu_m3_s2']
    config=json.loads((run/'config.json').read_text());fp=json.loads((run/'fingerprint.json').read_text())
    import spiceypy as sp
    from src.research.ephemeris import SpiceEphemeris
    records=[]
    with SpiceEphemeris(config['kernels'],config['frame'],run/'root-validation-cache') as ephem:
        assert ephem.provenance==fp['kernels']
        mars_mu=float(sp.bodvrd(config['visits'][1]['gm_body'],'GM',1)[1][0])*1e9
        for path in sorted((run/'ballistic-roots').glob('*/connection.json')):
            c=json.loads(path.read_text());node=c['node'];native=c['native']
            with np.load(path.with_name('data.npz'),allow_pickle=False) as d:
                vin=d['v2'][0,node-1,0]-d['arrival_states'][0,node,3:]
                vout=d['v1'][0,node,0]-d['departure_states'][0,node,3:]
                incoming=asymptote(native['rperi'],native['vperi_in'],mars_mu,True)
                outgoing=asymptote(native['rperi'],native['vperi_out'],mars_mu,False)
                recovered=max(np.linalg.norm(incoming-vin),np.linalg.norm(outgoing-vout))
                gooding_errors=[]
                for leg in (node-1,node):
                    a=d['departure_states'][0,leg,:3];b=d['arrival_states'][0,leg+1,:3]
                    v1=d['v1'][0,leg,0];v2=d['v2'][0,leg,0];policy=c['policies'][leg]
                    tof=float(d['arrivals'][0,leg+1]-d['departures'][0,leg])
                    comparisons=[]
                    for low in ((False,) if policy['revolutions']==0 else (False,True)):
                        try:
                            p,q=gooding1990(mu,a,b,tof,M=policy['revolutions'],prograde=bool(np.cross(a,v1)[2]>0),
                                low_path=low,maxiter=100,atol=1e-13,rtol=1e-13)
                            comparisons.append(max(np.linalg.norm(p-v1),np.linalg.norm(q-v2)))
                        except ValueError:pass
                    gooding_errors.append(min(comparisons,default=float('inf')))
                radius_error=abs(np.linalg.norm(native['rperi'])-native['rp'])
                passed=recovered<=1e-5 and max(gooding_errors)<=1e-4 and radius_error<=1e-4
                records.append(dict(root=path.parent.name,node=node,passed=bool(passed),
                    asymptote_error_m_s=float(recovered),gooding_velocity_error_m_s=float(max(gooding_errors)),
                    periapsis_radius_error_m=float(radius_error)))
    complete=len(records)==summary['ballistic_refined_connections'] and len(records)>0
    result=dict(status='passed_local_checks' if complete and all(r['passed'] for r in records) else 'failed_local_checks',
        checked_connections=len(records),scope='Independent asymptote recovery and Gooding velocities for both adjacent legs; not full-chain or N-body certification',records=records)
    write_json(run/'ballistic-root-independent-checks.json',result)
    if result['status']!='passed_local_checks':raise AssertionError('Independent local checks failed; see saved evidence')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--check-ballistic-roots',action='store_true')
    a=p.parse_args();print(focused_plot(a.run))
    if a.check_ballistic_roots:
        result=check_ballistic_roots(a.run);print(result['status'],result['checked_connections'])
