"""Independent direct-SPICE DOP853, continuous encounter and STM checks."""
from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from ..nbody_native import NBodyContext,Options


def norm(v):return float(np.linalg.norm(v))


def validate_local(ephem,seed,result):
 checks=[]
 def check(name,value,limit,unit=''):
  checks.append(dict(name=name,value=float(value),limit=float(limit),unit=unit,passed=bool(np.isfinite(value) and value<=limit)))
 def ode(t,y):return np.r_[y[3:],ephem.exact_force(t,y)]
 o=Options(**result['options']);ti,tm,to=result['entry_time'],result['encounter_time'],result['exit_time']
 check('native correction accepted',0 if result['status']=='locally_corrected' else 1,0)
 entry=np.array(result['entry_state_shifted_si']);mid=np.array(result['node_before_shifted_si']);after=np.array(result['node_after_shifted_si'])
 target=np.array(result['exit_state_shifted_si']);burn=after[3:]-mid[3:]
 def integrate(a,b,state):
  sol=solve_ivp(ode,(a,b),state,method='DOP853',rtol=3e-14,atol=np.r_[np.full(3,1e-5),np.full(3,1e-9)],max_step=30.,dense_output=True)
  if not sol.success:raise RuntimeError('Independent DOP853: '+sol.message)
  return sol
 first=integrate(ti,tm,entry);second=integrate(tm,to,after)
 check('DOP853/direct SPICE node position',norm(first.y[:3,-1]-mid[:3]),.05,'m')
 check('DOP853/direct SPICE node velocity',norm(first.y[3:,-1]-mid[3:]),1e-5,'m/s')
 check('DOP853/direct SPICE exit position',norm(second.y[:3,-1]-target[:3]),.05,'m')
 check('DOP853/direct SPICE exit velocity',norm(second.y[3:,-1]-target[3:]),1e-5,'m/s')
 # Propagate the independent entry solution THROUGH the encounter, applying
 # only the declared inertial impulse. No reset to the stored shooting node.
 actual_after=first.y[:,-1].copy();actual_after[3:]+=burn
 continuous=integrate(tm,to,actual_after)
 check('continuous independent terminal position',norm(continuous.y[:3,-1]-result['exit_target_position_shifted_m']),.1,'m')
 check('continuous independent terminal velocity',norm(continuous.y[3:,-1]-target[3:]),2e-5,'m/s')
 if seed['model']=='ballistic':check('ballistic impulse',norm(burn),1e-10,'m/s')
 else:
  body=ephem.body_state('499',tm);vi=mid[3:]-body[3:];vo=after[3:]-body[3:]
  check('powered impulse magnitude',abs(norm(burn)-abs(seed['signed_tangential_burn_m_s'])),1e-8,'m/s')
  check('powered tangency with direct-SPICE Mars velocity',norm(np.cross(vi,vo))/(norm(vi)*norm(vo)),1e-7)
 # Find global closest approach on each independent arc; node tangency alone
 # is insufficient to prove minimum altitude elsewhere along the encounter.
 body_minima=[]
 for body in ephem.bodies:
  minimum=[]
  for sol,a,b in [(first,ti,tm),(continuous,tm,to)]:
   def distance(t):return norm(sol.sol(t)[:3]-ephem.body_state(body[0],t)[:3])
   grid=np.linspace(a,b,1001);dist=np.array([distance(t) for t in grid])
   candidates=[(dist[0],a),(dist[-1],b)]
   indices=list(np.flatnonzero((dist[1:-1]<=dist[:-2])&(dist[1:-1]<=dist[2:]))+1)
   indices += [int(np.argmin(dist))]
   for i in set(indices):
    m=minimize_scalar(distance,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded',options={'xatol':1e-7})
    candidates.append((float(m.fun),float(m.x)))
   minimum.append(min(candidates))
  separation,when=min(minimum)
  check('continuous collision clearance '+body[3],body[2]-separation,0.,'m')
  body_minima.append(dict(id=body[0],label=body[3],minimum_distance_m=separation,guard_radius_m=body[2],time_seconds=when))
 mars=next(item for item in body_minima if item['id']=='499')
 rp,tperi=mars['minimum_distance_m'],mars['time_seconds']
 check('minimum permitted Mars radius',seed['minimum_periapsis_radius_m']-rp,0.,'m')
 check('maximum permitted Mars radius',rp-seed['maximum_periapsis_radius_m'],0.,'m')
 check('closest approach near shooting node',abs(tperi-tm),.01,'s')
 # Variational equations versus a central finite difference of the native map.
 base=ephem.native.propagate(ti,tm,entry,o);errors=[]
 for j in range(6):
  h=1. if j<3 else .001;a=entry.copy();b=entry.copy();a[j]+=h;b[j]-=h
  da=ephem.native.propagate(ti,tm,a,o)['state'];db=ephem.native.propagate(ti,tm,b,o)['state']
  fd=(da-db)/(2*h);col=base['stm'][:,j]
  error=norm(fd-col)/max(1.,norm(col));errors.append(error)
 check('STM finite-difference column relative error',max(errors),2e-6)
 # Tighter native integration with unchanged force interpolation.
 tighter=replace(o,rtol=o.rtol/5,atol_position=o.atol_position/5,atol_velocity=o.atol_velocity/5,atol_stm=o.atol_stm/5,maximum_step=o.maximum_step/2)
 for label,a,b,state in [('entry',ti,tm,entry),('exit',tm,to,after)]:
  coarse=ephem.native.propagate(a,b,state,o);fine=ephem.native.propagate(a,b,state,tighter)
  check(label+' tighter integration position',norm(coarse['state'][:3]-fine['state'][:3]),.02,'m')
  check(label+' tighter integration velocity',norm(coarse['state'][3:]-fine['state'][3:]),5e-6,'m/s')
 # Half-spacing ephemerides, checked separately from integration refinement.
 times=np.linspace(ephem.lower,ephem.upper,2*(len(ephem.times)-1)+1)
 states=np.array([[ephem.body_state(b[0],t) for b in ephem.bodies] for t in times])
 with NBodyContext(times,states,ephem.mu,ephem.radii) as refined:
  for label,a,b,state in [('entry',ti,tm,entry),('exit',tm,to,after)]:
   coarse=ephem.native.propagate(a,b,state,tighter);fine=refined.propagate(a,b,state,tighter)
   check(label+' half-spacing ephemeris position',norm(coarse['state'][:3]-fine['state'][:3]),.03,'m')
   check(label+' half-spacing ephemeris velocity',norm(coarse['state'][3:]-fine['state'][3:]),1e-5,'m/s')
 interpolation=ephem.interpolation_check()
 check('sampled ephemeris interpolation position',interpolation['maximum_position_difference_m'],.03,'m')
 return dict(status='passed_local_newtonian_nbody' if all(c['passed'] for c in checks) else 'failed',checks=checks,
  minimum_mars_radius_m=rp,minimum_mars_altitude_m=rp-3396200.,closest_approach_time_seconds=tperi,
  interpolation=interpolation,minimum_body_separations=body_minima,stm_finite_difference_column_relative_errors=errors,
  full_mission_reattached=False,
  scope='Independent numerical acceptance for finite local Newtonian point-mass arcs; not a high-fidelity complete-mission replacement',
  independent_integrator='SciPy DOP853, separately implemented direct-SPICE point-mass force',
  versions=dict(numpy=np.__version__))
