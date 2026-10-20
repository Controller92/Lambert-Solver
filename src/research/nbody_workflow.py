"""Versioned, reviewable local N-body correction runs; no implicit downloads."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import uuid
from .mission_config import load_mission
from .mission_search import digest,write_json,configuration_fingerprint
from .nbody_ephemeris import NBodyEphemeris
from .nbody_flyby import lift_candidate,shoot
from .nbody_validation import validate_local
from .nbody_gmat import check_gmat
from .nbody_plot import plot_local
from ..nbody_native import Options

ROOT=Path(__file__).resolve().parents[2]


def correct_encounter(candidate_path,outdir='results',config_path=None,node=1,boundary_radius=1e8,
                      design_margin=10000.,ephemeris_step=60.,gmat_console=None,progress=None):
 path=Path(candidate_path).resolve()
 folder=path.parent.parent if path.parent.name=='candidates' else path.parent
 config_path=Path(config_path).resolve() if config_path else folder/'config.json'
 config=load_mission(config_path);candidate=json.loads(path.read_text())
 acceptance=path.with_suffix('.validation.json')
 if not acceptance.exists():raise ValueError('First run mission-validate on the source candidate')
 record=json.loads(acceptance.read_text())
 if record.get('status')!='passed_patched_conic' or record.get('candidate_sha256')!=digest(path):
  raise ValueError('Source candidate lacks a matching passing validation record')
 if candidate.get('fingerprint',{}).get('config_sha256')!=configuration_fingerprint(config):
  raise ValueError('Source candidate configuration fingerprint differs')
 if not isinstance(ephemeris_step,(int,float)) or isinstance(ephemeris_step,bool) or not 1<=ephemeris_step<=600:
  raise ValueError('Use an explicit ephemeris spacing from 1 to 600 seconds')
 seed=lift_candidate(candidate,config,node,boundary_radius,design_margin)
 run=Path(outdir)/('nbody-'+seed['model']+'-'+uuid.uuid4().hex[:12]);run.mkdir(parents=True,exist_ok=False)
 effective=dict(schema_version=1,backend='native-cpu',source_candidate_sha256=digest(path),source_validation_sha256=digest(acceptance),
  mission_config=config.to_dict(),node=node,boundary_radius_m=boundary_radius,design_margin_m=design_margin,
  ephemeris_spacing_seconds=ephemeris_step,options=asdict(Options()),gmat_console=str(Path(gmat_console).resolve()) if gmat_console else None,
  acceptance_scope='Local Newtonian point-mass encounter only; full mission not reattached')
 write_json(run/'config.json',effective);write_json(run/'seed.json',seed)
 (run/'source-candidate.json').write_bytes(path.read_bytes())
 (run/'source-candidate.validation.json').write_bytes(acceptance.read_bytes())
 write_json(run/'source-mission-config.json',config.to_dict())
 modules=[Path('native/nbody')/n for n in ['api.h','cpu.cpp']]+[Path('src/nbody_native.py')]
 modules += [Path('src/research')/n for n in ['ephemeris.py','nbody_ephemeris.py','nbody_flyby.py','nbody_validation.py','nbody_gmat.py','nbody_plot.py','nbody_workflow.py','mission_search.py','mission_config.py']]
 fp=dict(sources={str(p):digest(ROOT/p) for p in modules},library_sha256=digest(ROOT/'build/native/libnbody_cpu.so'),
  python=platform.python_version(),versions={p:version(p) for p in ['numpy','scipy','spiceypy']})
 write_json(run/'fingerprint.json',fp)
 for relative in modules:
  destination=run/'sources'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/relative).read_bytes())
 try:
  with NBodyEphemeris(config.kernels,seed['epoch_et'],seed['entry_time']-3000,seed['exit_time']+3000,ephemeris_step,run/'state-cache') as ephem:
   write_json(run/'ephemeris.json',ephem.provenance())
   # Preserve interpolation inputs; force validation can compare exact SPICE.
   import numpy as np
   np.savez_compressed(run/'ephemeris.npz',times=ephem.times,states=ephem.states,mu=ephem.mu,radii=ephem.radii)
   if progress:progress('correcting finite encounter')
   result=shoot(ephem,seed);write_json(run/'result.json',result)
   if progress:progress('independent propagation, sensitivities and convergence checks')
   validation=validate_local(ephem,seed,result)
   validation['result_sha256']=digest(run/'result.json');validation['seed_sha256']=digest(run/'seed.json')
   write_json(run/'validation.json',validation)
   plot_local(ephem,seed,result,run/'encounter.png')
   external=None
   if gmat_console:
    if progress:progress('independent GMAT RK89 comparison')
    external=check_gmat(ephem,result,gmat_console,run/'gmat')
  passed=validation['status']=='passed_local_newtonian_nbody' and (external is None or external['status']=='passed_gmat_point_mass_comparison')
  summary=dict(status='passed_local_nbody_validation' if passed else 'failed',model=seed['model'],
   local_correction_status=result['status'],independent_validation=validation['status'],
   external_validation=external['status'] if external else 'not_run',
   full_mission_reattached=False,design_margin_added_m=design_margin,periapsis_altitude_m=validation['minimum_mars_altitude_m'],
   entry_velocity_adjustment_m_s=float(np.linalg.norm(result['entry_velocity_change_m_s'])),
   exit_velocity_adjustment_m_s=float(np.linalg.norm(result['exit_velocity_change_m_s'])),
   native_join_position_m=result['state_defect_position_m'],native_terminal_position_m=result['terminal_position_error_m'],
   check_count=len(validation['checks'])+(len(external['checks']) if external else 0),
   scope='Local finite Newtonian point-mass encounter; not a complete high-fidelity mission replacement')
  write_json(run/'summary.json',summary)
  write_json(run/'artifact-hashes.json',{str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()})
  return run,summary
 except Exception as error:
  write_json(run/'failure.json',dict(error_type=type(error).__name__,message=str(error),full_mission_reattached=False))
  raise
