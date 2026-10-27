"""Replay saved complete EMC N-body candidates in an external GMAT R2026a run.

The two flight sequences are continuous across Mars. Ceres arrival and departure
are separate sequences because the model prescribes a surface stay and launch.
No desired encounter state is inserted during either flight sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import spiceypy as spice


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def mars_state(et):
    return np.array(spice.spkezr('499', et, 'J2000', 'NONE', '0')[0]) * 1000


def mjd_tai(et):
    return 21545. + spice.unitim(et, 'ET', 'TAI') / 86400.


def generate(candidate, output, executable, mars_j2=True):
    run = Path(candidate)
    result = json.loads((run/'result.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    ephem = json.loads((run/'ephemeris.json').read_text())
    if summary['status'] != 'passed_global_nbody_numerical_checks':
        raise ValueError('Candidate lacks passing saved N-body validation')
    if ephem['mars_j2']['enabled'] is not True:
        raise ValueError('Expected the saved oriented Mars J2 model')
    source_hashes = json.loads((run/'artifact-hashes.json').read_text())
    for name in ('result.json', 'summary.json', 'ephemeris.json',
                 'trajectory-samples.json', 'validation.json'):
        if name in source_hashes and source_hashes[name] != sha256(run/name):
            raise ValueError(f'Saved source hash changed: {name}')
    kernel = {Path(k['path']).name: k for k in ephem['kernels']}
    for item in kernel.values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Kernel differs from saved run: ' + item['path'])
        spice.furnsh(item['path'])
    for name in ('de440s.bsp', 'mar099s.bsp', '20000001.bsp'):
        if name not in kernel:
            raise ValueError('Missing required SPK: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    nodes = result['nodes']
    samples=json.loads((run/'trajectory-samples.json').read_text())
    native_cuts={(x['leg'],x['node']):x['times'][-1] for x in samples}
    offset = np.array(ephem['constant_inertial_offset_m'])
    j2 = ephem['mars_j2']
    mu_mars = next(b['mu_m3_s2'] for b in ephem['bodies'] if b['id'] == '499')
    c20 = -j2['coefficient'] / np.sqrt(5.)
    gravity = output/'mars-j2.cof'
    gravity.write_text('COMMENT   1\n'
                       'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n'
                       f'POTFIELD  2  0  0 {mu_mars:.17g} {j2["reference_radius_m"]:.17g} 1.0\n'
                       f'RECOEF    2  0   {c20:.17g}\n')
    ids = {'10':'Sun','399':'Earth','301':'Luna','499':'Mars'}
    lines = [f"SolarSystem.SPKFilename = '{kernel['de440s.bsp']['path']}';",
             'SolarSystem.EphemerisSource = SPICE;',
             'SolarSystem.UseTTForEphemeris = false;',
             f"Mars.OrbitSpiceKernelName = {{'{kernel['mar099s.bsp']['path']}'}};"]
    for body in ephem['bodies']:
        identifier = body['id']; name = ids.setdefault(identifier, 'Body' + identifier)
        if identifier not in ('10','399','301','499'):
            files = [kernel[k]['path'] for k in ('de440s.bsp','mar099s.bsp','20000001.bsp')]
            lines.extend([f'Create Planet {name};', f'{name}.NAIFId = {identifier};',
                          f'{name}.CentralBody = Sun;',
                          f"{name}.OrbitSpiceKernelName = {{{', '.join(repr(f) for f in files)}}};"])
        lines.extend([f'{name}.Mu = {body["mu_m3_s2"]/1e9:.17g};',
                      f'{name}.EquatorialRadius = {body["collision_radius_m"]/1000:.17g};'])
    lines.extend(['Create CoordinateSystem MarsFrame;', 'MarsFrame.Origin = Mars;',
                  'MarsFrame.Axes = ICRF;'])
    # GMAT integrates a state relative to its force-model central body. Switch
    # that origin at leg midpoints; a Mars origin at low Earth parking altitude
    # lost metres in an hour despite a sub-centimetre force-body ephemeris match.
    for central in ('Earth','Mars','Body20000001'):
        force='F_'+central;prop='P_'+central
        names=[ids[b['id']] for b in ephem['bodies'] if b['id']!='499' or not (central=='Mars' and mars_j2)]
        lines.extend([f'Create ForceModel {force};',f'{force}.CentralBody = {central};',
                      f'{force}.PrimaryBodies = {{Mars}};' if central=='Mars' and mars_j2 else f'{force}.PrimaryBodies = {{}};',
                      f'{force}.PointMasses = {{{", ".join(names)}}};',
                      f'{force}.Drag = None;',f'{force}.SRP = Off;',
                      f'{force}.RelativisticCorrection = Off;'])
        if central=='Mars' and mars_j2:
            lines.extend([f'{force}.GravityField.Mars.Degree = 2;',
                          f'{force}.GravityField.Mars.Order = 0;',
                          f"{force}.GravityField.Mars.PotentialFile = '{gravity}';"])
        lines.extend([f'Create Propagator {prop};',f'{prop}.FM = {force};',
                      f'{prop}.Type = RungeKutta89;',f'{prop}.InitialStepSize = 60;',
                      f'{prop}.Accuracy = 1e-12;',f'{prop}.MinStep = 0.000001;',
                      f'{prop}.MaxStep = 10800;'])
    outputs = []
    for phase, (first, encounter, last) in enumerate(((0,1,2),(3,4,5))):
        craft = f'SC{phase}'; report = f'R{phase}'
        et = ephem['epoch_et']+nodes['times'][first]
        initial = (np.array(nodes['after'][first]) + np.r_[offset,np.zeros(3)] - mars_state(et))/1000
        lines.extend([f'Create Spacecraft {craft};', f'{craft}.DateFormat = TAIModJulian;',
                      f"{craft}.Epoch = '{mjd_tai(et):.17g}';",f'{craft}.CoordinateSystem = MarsFrame;',
                      f'{craft}.DisplayStateType = Cartesian;'])
        lines.extend(f'{craft}.{name} = {value:.17g};' for name,value in zip(('X','Y','Z','VX','VY','VZ'),initial))
        lines.extend([f'Create ReportFile {report};',
                      f"{report}.Filename = '{output/f'phase-{phase}.txt'}';",
                      f'{report}.Precision = 16;',f'{report}.WriteHeaders = false;'])
        burn = (np.array(nodes['after'][encounter])[3:]-np.array(nodes['before'][encounter])[3:])/1000
        use_burn = np.linalg.norm(burn) > 1e-12
        if use_burn:
            burner=f'B{phase}'
            lines.extend([f'Create ImpulsiveBurn {burner};',f'{burner}.CoordinateSystem = MarsFrame;',
                          f'{burner}.DecrementMass = false;'])
            lines.extend(f'{burner}.Element{i+1} = {value:.17g};' for i,value in enumerate(burn))
        outputs.append(dict(phase=phase,first=first,encounter=encounter,last=last,
                            use_burn=bool(use_burn),burn_m_s=(burn*1000).tolist()))
    lines.append('BeginMissionSequence;')
    for data in outputs:
        phase=data['phase']; craft=f'SC{phase}'; report=f'R{phase}'
        a,b,c=(data[k] for k in ('first','encounter','last'))
        fields=f'{craft}.TAIModJulian '+' '.join(f'{craft}.MarsFrame.{item}' for item in ('X','Y','Z','VX','VY','VZ'))
        start_body='Earth' if phase==0 else 'Body20000001'
        end_body='Body20000001' if phase==0 else 'Earth'
        t0,t1,t2=(nodes['times'][i] for i in (a,b,c))
        cut1=native_cuts[(phase*2,a)];cut2=native_cuts[(phase*2+1,b)]
        if not (t0<cut1<t1<cut2<t2):raise ValueError('Saved shooting cuts are not ordered')
        lines.extend([f'Report {report} {fields};',
                      f'Propagate P_{start_body}({craft}) {{{craft}.ElapsedSecs = {cut1-t0:.17g}}};',
                      f'Report {report} {fields};',
                      f'Propagate P_Mars({craft}) {{{craft}.ElapsedSecs = {t1-cut1:.17g}}};',
                      f'Report {report} {fields};'])
        if data['use_burn']:
            lines.extend([f'Maneuver B{phase}({craft});', f'Report {report} {fields};'])
        lines.extend([f'Propagate P_Mars({craft}) {{{craft}.ElapsedSecs = {cut2-t1:.17g}}};',
                      f'Report {report} {fields};',
                      f'Propagate P_{end_body}({craft}) {{{craft}.ElapsedSecs = {t2-cut2:.17g}}};',
                      f'Report {report} {fields};'])
    script=output/'complete-mission.script';script.write_text('\n'.join(lines)+'\n')
    return result,ephem,outputs,script,gravity


def compare(result, ephem, phases, output, executable, script, gravity, timeout, candidate, mars_j2):
    run=subprocess.run([str(executable),'--run',str(script)],capture_output=True,text=True,timeout=timeout)
    (output/'console.txt').write_text(run.stdout+run.stderr)
    if run.returncode or 'Mission run completed.' not in run.stdout:
        raise RuntimeError('GMAT did not complete: see console.txt')
    nodes=result['nodes']; offset=np.array(ephem['constant_inertial_offset_m']); records=[]
    samples=json.loads((Path(candidate)/'trajectory-samples.json').read_text())
    forward={(x['leg'],x['node']):x for x in samples}
    for phase in phases:
        rows=np.loadtxt(output/f'phase-{phase["phase"]}.txt',ndmin=2)
        leg=phase['phase']*2
        points=[('event',phase['first'],'after'),('midpoint',leg,None),
                ('event',phase['encounter'],'before')]
        if phase['use_burn']:points.append(('event',phase['encounter'],'after'))
        points.extend([('midpoint',leg+1,None),('event',phase['last'],'before')])
        if rows.shape != (len(points),7) or not np.isfinite(rows).all():
            raise ValueError(f'Unexpected GMAT output for phase {phase["phase"]}: {rows.shape}')
        for row,(kind,index,side) in zip(rows,points):
            actual_et=spice.unitim((row[0]-21545.)*86400.,'TAI','ET')
            if kind=='event':
                expected_et=ephem['epoch_et']+nodes['times'][index]
                target=np.array(nodes[side][index])+np.r_[offset,np.zeros(3)]
            else:
                sample=forward[(index,phase['first'] if index==leg else phase['encounter'])]
                expected_et=ephem['epoch_et']+sample['times'][-1]
                target=np.array(sample['states_shifted_si'][-1])+np.r_[offset,np.zeros(3)]
            dt=actual_et-expected_et
            target[:3]+=target[3:]*dt
            gmat_state= mars_state(actual_et)+row[1:]*1000
            diff=gmat_state-target
            records.append(dict(phase=phase['phase'],kind=kind,
                                node=index if kind=='event' else None,
                                leg=index if kind=='midpoint' else None,side=side,
                                epoch_offset_s=float(dt),
                                position_difference_m=float(np.linalg.norm(diff[:3])),
                                velocity_difference_m_s=float(np.linalg.norm(diff[3:]))))
    initial=[r for r in records if r['node'] in (0,3)]
    terminal=[r for r in records if r['kind']=='midpoint' or
              (r['node'] in (1,2,4,5) and r['side']=='before')]
    limits=dict(epoch_s=0.002,initial_position_m=100.,initial_velocity_m_s=2e-4,
                event_position_m=200.,event_velocity_m_s=3e-4)
    passed=(all(abs(r['epoch_offset_s'])<=limits['epoch_s'] for r in records)
            and all(r['position_difference_m']<=limits['initial_position_m'] and
                    r['velocity_difference_m_s']<=limits['initial_velocity_m_s'] for r in initial)
            and all(r['position_difference_m']<=limits['event_position_m'] and
                    r['velocity_difference_m_s']<=limits['event_velocity_m_s'] for r in terminal))
    report=dict(status='passed_gmat_complete_mission_numerical_comparison' if passed else 'failed_gmat_complete_mission_numerical_comparison',
                scope='Two continuous GMAT flight sequences with Mars impulses and a prescribed Ceres surface stay/launch; 14 pinned point masses. Integration origins switch at four saved leg midpoints.'+(' Degree-2 zonal Mars gravity is applied on Mars-centred segments; it is omitted on Earth/Ceres-centred cruise halves.' if mars_j2 else ' Point-mass control omits native Mars J2.'),
                model_qualification='Both trajectories use ICRF axes. GMAT body orientation and integration-origin dynamics may differ from the native SSB equations and interpolated Mars J2. The position comparison shifts target states linearly to GMAT reported epochs. This is a numerical comparison, not navigation accuracy.',
                limits=limits,records=records,
                gmat_executable_sha256=sha256(executable),gmat_script_sha256=sha256(script),
                mars_j2_cof_sha256=sha256(gravity) if mars_j2 else None,source_result_sha256=sha256(Path(candidate)/'result.json'),
                source_ephemeris_sha256=sha256(Path(candidate)/'ephemeris.json'))
    save(output/'comparison.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--gmat-console',required=True,type=Path)
    parser.add_argument('--outdir',required=True,type=Path)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--point-mass-control',action='store_true')
    args=parser.parse_args()
    executable=args.gmat_console.resolve()
    output=args.outdir.resolve()
    try:
        result,ephem,phases,script,gravity=generate(args.candidate,output,executable,not args.point_mass_control)
        report=compare(result,ephem,phases,output,executable,script,gravity,args.timeout,args.candidate,not args.point_mass_control)
        print(report['status'], 'max event position m',max(x['position_difference_m'] for x in report['records']))
        if report['status'].startswith('failed'):sys.exit(2)
    except Exception as error:
        if output.exists():save(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        spice.kclear()
