"""Supervise one bounded local EMC sweep and write a reviewable result report.

No scheduler, privilege, power-limit or clock changes. The caller may wrap this
process in a temporary systemd sleep inhibitor. The child owns its process
group so timeout cleanup cannot target unrelated work.
"""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.mission_search import write_json


def utcnow():return datetime.now(timezone.utc)


def metric_summary(run):
    """Read only saved connection columns, never all trajectory arrays at once."""
    answer={}
    def consume(key,values):
        if not len(values['plane_change_deg']):return
        item=answer.setdefault(key,dict(count=0,plane_at_least_8deg_positive_gain_ceres_alignment_within_2deg=0,ranges={}))
        item['count']+=len(values['plane_change_deg'])
        for field,array in values.items():
            good=np.asarray(array)[np.isfinite(array)]
            if not len(good):continue
            bounds=item['ranges'].setdefault(field,[float('inf'),float('-inf')])
            bounds[0]=min(bounds[0],float(good.min()));bounds[1]=max(bounds[1],float(good.max()))
        selected=(values['plane_change_deg']>=8)&(values['heliocentric_speed_gain_km_s']>0)&(values['ceres_plane_after_deg']<=2)
        item['plane_at_least_8deg_positive_gain_ceres_alignment_within_2deg']+=int(selected.sum())
    fields=('plane_change_deg','heliocentric_speed_gain_km_s','vinf_speed_gain_km_s','ceres_plane_before_deg','ceres_plane_after_deg','burn_km_s','altitude_km')
    for path in sorted((run/'chunks').glob('*.npz')):
        if path.name.endswith('.tmp.npz'):continue
        with np.load(path,allow_pickle=False) as data:
            for model in ('ballistic','powered'):
                for node in (1,3):
                    prefix=f'{model}_{node}_'
                    if prefix+'plane_change_deg' in data:
                        consume(f'{model}-{node}',{field:data[prefix+field] for field in fields})
    for path in sorted((run/'ballistic-roots').glob('*/connection.json')):
        record=json.loads(path.read_text());values={k:np.atleast_1d(record['metrics'][k]) for k in fields if k in record['metrics']}
        values['burn_km_s']=np.array([record['native']['delta_v']/1000])
        values['altitude_km']=np.array([(record['native']['rp']-3396200)/1000])
        consume(f'ballistic-{record["node"]}',values)
    return answer


def monitoring_summary(run):
    result={}
    cpu=run/'cpu-monitor.csv'
    if cpu.exists():
        with cpu.open() as stream:rows=list(csv.DictReader(stream))
        for key in ('process_equivalent_cores','system_busy_percent','rss_mb'):
            values=[float(r[key]) for r in rows]
            if values:result[key]=dict(mean=float(np.mean(values)),peak=float(max(values)))
    gpu=run/'gpu-monitor.csv'
    if gpu.exists():
        with gpu.open() as stream:rows=list(csv.reader(stream))
        for column,name in ((1,'gpu_utilization_percent'),(3,'gpu_memory_mib'),(4,'gpu_power_w'),(5,'gpu_temperature_c')):
            values=[]
            for row in rows[1:]:
                try:values.append(float(row[column].strip().split()[0]))
                except (ValueError,IndexError):pass
            if values:result[name]=dict(mean=float(np.mean(values)),peak=float(max(values)))
    return result


def make_report(job,run):
    summary=json.loads((run/'summary.json').read_text())
    timing=json.loads((run/'timings.json').read_text())
    metrics=metric_summary(run);resources=monitoring_summary(run)
    for key,item in metrics.items():
        if item['count']!=summary['counts'].get(key):raise ValueError(f'Connection count mismatch: {key}')
    write_json(job/'connection-statistics.json',metrics);write_json(job/'resource-statistics.json',resources)
    cells=sum(r['cells'] for r in timing['porkchops']['grids'])
    solves=sum(r['candidate_solves'] for r in timing['porkchops']['grids'])+timing['sweep']['lambert_attempts']
    lines=['# Overnight EMC survey','',f'Run: `{run}`','',
        f"Sampled **{summary['candidates']:,}** complete epoch tuples; **{cells:,}** independent leg-grid cells.",
        f"Executed **{solves:,}** grid/survey Lambert branch attempts and **{timing['sweep']['flyby_attempts']:,}** survey flyby branch-pair attempts.",
        'Root refinement, benchmark repetitions and validation work are additional. Failed/infeasible branch attempts are included in those counts.','',
        'Ballistic and powered results are separate. A feasible local Mars connection does not certify a complete mission or an N-body trajectory.',
        f"The sample includes {summary['anchors']} previously validated anchor tuples; retaining them is not a new discovery.",'',
        '| Local connection | Saved accepted rows | Plane rotation (deg) | Heliocentric speed change (km/s) |',
        '|---|---:|---:|---:|']
    for key,item in sorted(metrics.items()):
        plane=item['ranges']['plane_change_deg'];gain=item['ranges']['heliocentric_speed_gain_km_s']
        lines.append(f"| {key} | {item['count']:,} | {plane[0]:.4g} to {plane[1]:.4g} | {gain[0]:.4g} to {gain[1]:.4g} |")
    lines+=['','Plane rotation is the angle between heliocentric angular momenta. Powered speed gains include the charged periapsis impulse; ballistic v-infinity magnitudes are matched using the native tolerance.',
        'Connection statistics also save pre/post alignment with the Ceres orbital plane, altitude, burn and planet-relative speed change.','',
        '## Complete mission candidates','', '| Model | Best sampled total Δv (km/s) | Independent patched-conic check |','|---|---:|---|']
    for model,dv in summary['best_delta_v_m_s'].items():
        status=summary.get('independent_best_validation',{}).get(model,'not run')
        lines.append(f'| {model} | {dv/1000:.6f} | {status} |')
    lines+=['','These are finite-survey results, without a global-optimality claim. N-body correction, navigation and engineering validation remain separate.','',
        '## CPU and GPU measurements','', '| Backend | Threads | Million branch attempts/s |','|---|---:|---:|']
    for row in timing['benchmarks']['records']:
        lines.append(f"| {row['backend']} | {row.get('threads','—')} | {row['rows_per_second']/1e6:.3f} |")
    lines+=['','CPU values use wall timing. Resident CUDA timing excludes transfers/allocation. Concurrent stress measurements reflect simultaneous CPU/GPU load and may differ from standalone timings.','',
        '## Figures','']
    for path in sorted(run.glob('*.png')):lines.append(f'- [{path.stem}]({path})')
    lines+=['',f'- [Detailed connection statistics]({job}/connection-statistics.json)',f'- [Resource monitoring summary]({job}/resource-statistics.json)',
        f'- [Numerical summary]({run}/summary.json)',f'- [Timings]({run}/timings.json)',f'- [Source/kernel/library provenance]({run}/fingerprint.json)','',
        'Trajectory coverage includes every accepted sampled connection; 3D curves and scatter highlights are bounded subsets. Coverage uses uniform true-anomaly samples, not residence-time weighting.',
        'Exact numerical records and branch indices remain in the chunk files. Display clipping is reported in the numerical summary.']
    (job/'report.md').write_text('\n'.join(lines)+'\n')


def execute(job,finish_by,config,grid_step=1.,max_candidates=50000000,max_storage_gib=64.,benchmark_seconds=300.):
    job=Path(job).resolve();job.mkdir(parents=True,exist_ok=True)
    with (job/'.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (job/'status.json').exists():raise RuntimeError('Use a new job directory; an existing run will not be overwritten')
        compute_finish=finish_by-timedelta(minutes=35)
        seconds=(compute_finish-utcnow()).total_seconds()
        if seconds<120:raise ValueError('Insufficient time before the requested finish deadline')
        command=[sys.executable,str(ROOT/'run.py'),'mission-sweep','--config',str(Path(config).resolve()),'--outdir',str(job/'survey'),
            '--seconds',str(min(86400,seconds)),'--max-candidates',str(max_candidates),'--grid-step',str(grid_step),'--batch-size','2048',
            '--max-storage-gib',str(max_storage_gib),'--benchmark-seconds',str(benchmark_seconds)]
        state=dict(status='starting',supervisor_pid=os.getpid(),started_utc=utcnow().isoformat(),finish_by_utc=finish_by.isoformat(),
                   compute_finish_utc=compute_finish.isoformat(),command=command)
        write_json(job/'status.json',state)
        with (job/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',child_pid=child.pid);write_json(job/'status.json',state)
            try:code=child.wait(timeout=max(1,(finish_by-timedelta(minutes=10)-utcnow()).total_seconds()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                state.update(status='timed_out',finished_utc=utcnow().isoformat());write_json(job/'status.json',state);return 2
        runs=sorted((job/'survey').glob('emc-sweep-*'))
        run=runs[-1] if runs else None
        state.update(status='summarizing' if code==0 else 'failed',exit_code=code,run=str(run) if run else None)
        write_json(job/'status.json',state)
        if code==0 and run is not None:
            try:make_report(job,run)
            except Exception as error:
                state.update(status='report_failed',report_error=str(error));write_json(job/'status.json',state);raise
            state.update(status='completed',report=str(job/'report.md'))
        state.update(finished_utc=utcnow().isoformat(),finished_by_deadline=utcnow()<=finish_by)
        write_json(job/'status.json',state)
        return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job-dir',type=Path,required=True)
    p.add_argument('--finish-by',required=True,help='ISO8601 timestamp including timezone offset')
    p.add_argument('--config',type=Path,default=ROOT/'examples/missions/emc-ten-year.json')
    p.add_argument('--report-only',type=Path,help='Regenerate report for an already completed survey')
    a=p.parse_args();finish=datetime.fromisoformat(a.finish_by)
    if finish.tzinfo is None:p.error('finish-by must include a timezone offset')
    if a.report_only:make_report(a.job_dir,a.report_only)
    else:raise SystemExit(execute(a.job_dir,finish.astimezone(timezone.utc),a.config))
