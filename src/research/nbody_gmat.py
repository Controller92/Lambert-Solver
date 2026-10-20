"""Explicit external GMAT R2026a point-mass encounter cross-check.

GMAT is never downloaded or installed implicitly. The caller supplies a local
GmatConsole executable. It has its own RK89 integrator, gravity implementation,
and SPICE/time infrastructure; all masses and SPKs are declared in the script.
"""
from pathlib import Path
import subprocess
import numpy as np
from .mission_search import digest,write_json


def check_gmat(ephem,result,executable,outdir):
 outdir=Path(outdir).resolve();outdir.mkdir(parents=True,exist_ok=True)
 executable=Path(executable).resolve()
 if not executable.is_file():raise ValueError('Supply an installed GmatConsole executable')
 ti,tm,to=[result[k] for k in ['entry_time','encounter_time','exit_time']]
 entry=(np.array(result['entry_state_shifted_si'])-ephem.body_state('499',ti))/1000
 tai=[float(ephem.e.sp.unitim(ephem.epoch+t,'ET','TAI')) for t in [ti,tm,to]]
 mjd=21545.+tai[0]/86400
 paths={Path(k['path']).name:k['path'] for k in ephem.e.provenance}
 for name in ['de440s.bsp','mar099s.bsp','20000001.bsp']:
  if name not in paths:raise ValueError('GMAT example requires the pinned kernel set')
 lines=[f"SolarSystem.SPKFilename = '{paths['de440s.bsp']}';",'SolarSystem.EphemerisSource = SPICE;',
  'SolarSystem.UseTTForEphemeris = false;',f"Mars.OrbitSpiceKernelName = {{'{paths['mar099s.bsp']}'}};"]
 names={'10':'Sun','399':'Earth','301':'Luna','499':'Mars'}
 for b in ephem.provenance()['bodies']:
  identifier=b['id'];name=names.get(identifier,'Body'+identifier);names[identifier]=name
  if identifier not in ['10','399','301','499']:
   lines += [f'Create Planet {name};',f'{name}.NAIFId = {identifier};',f'{name}.CentralBody = Sun;',
    f"{name}.OrbitSpiceKernelName = {{'{paths['de440s.bsp']}', '{paths['mar099s.bsp']}', '{paths['20000001.bsp']}'}};"]
  lines += [f'{name}.Mu = {b["mu_m3_s2"]/1e9:.17g};',f'{name}.EquatorialRadius = {b["collision_radius_m"]/1000:.17g};']
 lines += ['Create CoordinateSystem MarsFrame;','MarsFrame.Origin = Mars;','MarsFrame.Axes = MJ2000Eq;',
  'Create Spacecraft SC;','SC.DateFormat = TAIModJulian;',f"SC.Epoch = '{mjd:.17g}';",'SC.CoordinateSystem = MarsFrame;',
  'SC.DisplayStateType = Cartesian;']
 lines += [f'SC.{n} = {v:.17g};' for n,v in zip(['X','Y','Z','VX','VY','VZ'],entry)]
 lines += ['Create ForceModel F;','F.CentralBody = Mars;','F.PrimaryBodies = {};',
  'F.PointMasses = {'+', '.join(names[b[0]] for b in ephem.bodies)+'};',
  'F.Drag = None;','F.SRP = Off;','F.RelativisticCorrection = Off;',
  'Create Propagator P;','P.FM = F;','P.Type = RungeKutta89;','P.InitialStepSize = 10;',
  'P.Accuracy = 1e-12;','P.MinStep = 0.000001;','P.MaxStep = 30;']
 burn=(np.array(result['node_after_shifted_si'])[3:]-np.array(result['node_before_shifted_si'])[3:])/1000
 if np.linalg.norm(burn)>0:
  lines += ['Create ImpulsiveBurn B;','B.CoordinateSystem = MarsFrame;','B.DecrementMass = false;']
  lines += [f'B.Element{j+1} = {v:.17g};' for j,v in enumerate(burn)]
 lines += ['Create ReportFile R;',f"R.Filename = '{outdir/'states.txt'}';",'R.Precision = 16;','R.WriteHeaders = false;',
  'BeginMissionSequence;']
 fields='SC.TAIModJulian '+' '.join('SC.MarsFrame.'+n for n in ['X','Y','Z','VX','VY','VZ'])
 lines += ['Report R '+fields+';',f'Propagate P(SC) {{SC.ElapsedSecs = {tai[1]-tai[0]:.17g}}};','Report R '+fields+';']
 if np.linalg.norm(burn)>0:lines += ['Maneuver B(SC);','Report R '+fields+';']
 lines += [f'Propagate P(SC) {{SC.ElapsedSecs = {tai[2]-tai[1]:.17g}}};','Report R '+fields+';']
 script=outdir/'encounter.script';script.write_text('\n'.join(lines)+'\n')
 run=subprocess.run([str(executable),'--run',str(script)],capture_output=True,text=True,timeout=180)
 (outdir/'console.txt').write_text(run.stdout+run.stderr)
 if run.returncode or 'Mission run completed.' not in run.stdout:
  raise RuntimeError('GMAT did not complete; see '+str(outdir/'console.txt'))
 rows=np.loadtxt(outdir/'states.txt',ndmin=2)
 powered=np.linalg.norm(burn)>0
 if rows.shape!=((4 if powered else 3),7) or not np.isfinite(rows).all():raise ValueError('Unexpected GMAT state table')
 times=[float(ephem.e.sp.unitim((row[0]-21545.)*86400,'TAI','ET')-ephem.epoch) for row in rows]
 requested=[ti,tm,tm,to] if powered else [ti,tm,to]
 checks=[]
 for index,(t,requested_time) in enumerate(zip(times,requested)):
  checks.append(dict(name=f'GMAT epoch {index}',value=abs(t-requested_time),limit=.001,unit='s'))
 # Compare at GMAT's REPORTED epochs and exact initial numeric state. This
 # avoids confusing sub-microsecond MJD/A1/TDB rounding with dynamics error.
 initial=ephem.body_state('499',times[0])+np.r_[rows[0,1:4],rows[0,4:7]]*1000
 first=ephem.native.propagate(times[0],times[1],initial)['state']
 native=[initial,first]
 if powered:
  first=first.copy();first[3:]+=burn*1000;native.append(first)
 end=ephem.native.propagate(times[1],times[-1],first)['state'];native.append(end)
 residuals=[]
 for i,(row,t,state) in enumerate(zip(rows,times,native)):
  ours=state-ephem.body_state('499',t);difference=row[1:]*1000-ours
  position=float(np.linalg.norm(difference[:3]));velocity=float(np.linalg.norm(difference[3:]))
  checks += [dict(name=f'GMAT state {i} position',value=position,limit=.1,unit='m'),
             dict(name=f'GMAT state {i} velocity',value=velocity,limit=2e-5,unit='m/s')]
  residuals.append(dict(time_seconds=t,position_difference_m=position,velocity_difference_m_s=velocity))
 for c in checks:c['passed']=bool(c['value']<=c['limit'])
 report=dict(status='passed_gmat_point_mass_comparison' if all(c['passed'] for c in checks) else 'failed',checks=checks,
  states=residuals,executable=str(executable),executable_sha256=digest(executable),script_sha256=digest(script),
  native_frame='fixed inertial translation of SSB/J2000',gmat_frame='Mars-centred MJ2000Eq',
  qualification='GMAT uses its Mars-centred point-mass equations; native uses a direct barycentric force sum. Residuals include any force-origin/ephemeris consistency difference. Comparison normalizes reported epochs, not states.',
  full_mission_reattached=False)
 report['runtime_sha256']={name:digest(executable.parent/name) for name in
  ['gmat_startup_file.txt','libGmatBase.so.R2026a','libGmatUtil.so.R2026a'] if (executable.parent/name).exists()}
 write_json(outdir/'comparison.json',report);return report
