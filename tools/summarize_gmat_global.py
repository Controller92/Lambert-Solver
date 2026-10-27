"""Preserve compact evidence and plot external GMAT mission replays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(ballistic, powered, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    summary={}
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    labels=['Earth launch','cut 1','Mars 1','cut 2','Ceres arrival',
            'Ceres departure','cut 3','Mars 2','cut 4','Earth return']
    for model,run in (('ballistic',ballistic),('powered',powered)):
        report=json.loads((run/'comparison.json').read_text())
        if 'Mission run completed.' not in (run/'console.txt').read_text():
            raise ValueError(f'{model} GMAT execution did not complete')
        if report['status'] != 'failed_gmat_complete_mission_numerical_comparison':
            raise ValueError(f'Unexpected {model} result; review before publishing')
        rows=[r for r in report['records'] if not
              (r['kind']=='event' and r['side']=='after' and r['node'] not in (0,3))]
        if len(rows)!=10:raise ValueError(f'Unexpected {model} event count')
        for name in ('comparison.json','complete-mission.script','mars-j2.cof',
                     'console.txt','phase-0.txt','phase-1.txt'):
            destination=outdir/f'{model}-{name}'
            if name.startswith('phase-'):
                # GMAT pads report rows with trailing spaces. Preserve the
                # numeric fields while keeping the committed evidence tidy.
                destination.write_text('\n'.join(line.rstrip() for line in
                    (run/name).read_text().splitlines())+'\n')
            else:
                shutil.copy2(run/name,destination)
        for axis,field in zip(axes,('position_difference_m','velocity_difference_m_s')):
            axis.plot(range(10),[max(r[field],1e-9) for r in rows],marker='o',label=model.capitalize())
        summary[model]=dict(status=report['status'],
            first_midpoint_position_difference_m=rows[1]['position_difference_m'],
            first_mars_position_difference_m=rows[2]['position_difference_m'],
            ceres_arrival_position_difference_m=rows[4]['position_difference_m'],
            second_mars_position_difference_m=rows[7]['position_difference_m'],
            earth_return_position_difference_m=rows[9]['position_difference_m'],
            maximum_epoch_offset_s=max(abs(r['epoch_offset_s']) for r in report['records']),
            source_comparison_sha256=digest(run/'comparison.json'),
            source_script_sha256=digest(run/'complete-mission.script'))
    for axis,unit,limit in zip(axes,('Position difference (m)','Velocity difference (m/s)'),(200,3e-4)):
        axis.set_yscale('log');axis.set_xticks(range(10),labels,rotation=60,ha='right')
        axis.set_ylabel(unit);axis.grid(alpha=.25,which='both')
        axis.axvline(4.5,color='.5',ls='--',lw=1)
        axis.axhline(limit,color='#b63e39',ls=':',label='Acceptance limit')
        axis.legend()
    fig.suptitle('GMAT R2026a complete-candidate replay: external numerical comparison failed\n'
                 'Dashed line marks the prescribed Ceres surface stay and launch',fontsize=12)
    fig.savefig(outdir/'gmat-replay-residuals.png',dpi=180);plt.close(fig)
    summary['plot_sha256']=digest(outdir/'gmat-replay-residuals.png')
    summary['scope']='Two continuous GMAT flight sequences per candidate, separated by a prescribed Ceres surface stay. The saved N-body corrections remain numerically validated within their native/direct-SPICE models; GMAT does not confirm them at the declared limits.'
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({m:summary[m] for m in ('ballistic','powered')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ballistic-run',required=True,type=Path)
    p.add_argument('--powered-run',required=True,type=Path)
    p.add_argument('--outdir',required=True,type=Path)
    a=p.parse_args();main(a.ballistic_run,a.powered_run,a.outdir)
