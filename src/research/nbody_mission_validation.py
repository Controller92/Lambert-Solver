"""Independent direct-SPICE DOP853 checks for the finite complete chain."""
from dataclasses import replace
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from .nbody_mission import PAIRS,SCALE,norm
from ..nbody_native import NBodyContext


def validate_global(problem,result,progress=None):
 e=problem.e;nodes=result['nodes'];times=nodes['times'];checks=[];arcs=[];solutions=[]
 def check(name,value,limit,unit=''):
  checks.append(dict(name=name,value=float(value),limit=float(limit),unit=unit,passed=bool(np.isfinite(value) and value<=limit)))
 check('native shooting converged',0 if result['status']=='shooting_converged' else 1,0)
 for i,(a,b) in enumerate(PAIRS):
  pair=[]
  for n,after in [(a,True),(b,False)]:
   if progress:progress(f'Independent direct-SPICE leg {i+1}, node {n}')
   start=float(times[n]);end=problem.cuts[i]
   # Constant local origin anchored at a nearby stored knot. Direct-SPICE
   # positions are queried independently, without the native interpolator.
   j=[body[0] for body in e.bodies].index(nodes['ids'][n]);k=max(0,min(len(e.times)-2,np.searchsorted(e.times,start)-1))
   origin=e.states[k,j,:3].copy()
   planet,_=e.interpolated_state(nodes['ids'][n],start,origin)
   initial=np.array(nodes['relative'][n])+planet
   # Use precisely the SAME numeric inertial initial state. Replacing its
   # body centre with direct-SPICE changes the state by millimetres and
   # amplifies that different boundary condition over multi-year flight.
   if after:initial[3:]+=nodes['after'][n,3:]-nodes['before'][n,3:]
   def force(t,y):
    positions=np.array([e.body_state(body[0],t)[:3]-origin for body in e.bodies])
    d=positions-y[:3];r=np.linalg.norm(d,axis=1)
    acc=np.sum(d*(e.mu/r**3)[:,None],axis=0)
    if e.mars_j2:
     m=[body[0] for body in e.bodies].index('499');x=-d[m];pole=e.e.sp.pxform('IAU_MARS','J2000',e.epoch+t)[:,2];z=np.dot(x,pole)
     acc+=1.5*e.mu[m]*1.96045e-3*3396200.**2/r[m]**5*((5*z*z/r[m]**2-1)*x-2*z*pole)
    return np.r_[y[3:],acc]
   sol=solve_ivp(force,(start,end),initial,method='DOP853',rtol=3e-14,atol=np.r_[np.full(3,1e-5),np.full(3,1e-9)],max_step=10800.,dense_output=True)
   if not sol.success:raise RuntimeError('Independent integration: '+sol.message)
   final=sol.y[:,-1].copy();final[:3]+=origin
   native=problem.propagate_node(n,nodes,end,after)['state']
   diff=final-native
   check(f'leg {i+1} node {n} DOP853 position',norm(diff[:3]),100.,'m')
   check(f'leg {i+1} node {n} DOP853 velocity',norm(diff[3:]),2e-4,'m/s')
   pair.append(final)
   record=dict(leg=i,node=n,origin=origin,start=start,end=end,sol=sol)
   solutions.append(record)
   arcs.append(dict(leg=i,node=n,evaluations=sol.nfev,position_difference_m=norm(diff[:3]),velocity_difference_m_s=norm(diff[3:])))
  defect=pair[0]-pair[1]
  check(f'leg {i+1} independent midpoint position continuity',norm(defect[:3]),200.,'m')
  check(f'leg {i+1} independent midpoint velocity continuity',norm(defect[3:]),3e-4,'m/s')
 # Every force body, every arc: sample the independent dense output and refine
 # every bracketed minimum, including near-surface endpoints. This is numerical
 # screening, not a certified collision bound under navigation uncertainty.
 minima={b[0]:[np.inf,None] for b in e.bodies}
 solar=np.inf
 for record in solutions:
  sol,origin=record['sol'],record['origin'];lo,hi=sorted([record['start'],record['end']])
  # Include integration steps, a uniform cruise mesh, and logarithmic grids
  # close to each endpoint to resolve the finite hyperbolic excursions.
  span=hi-lo
  grid=np.unique(np.r_[np.linspace(lo,hi,1501),sol.t,lo+np.geomspace(.001,min(span,10*86400.),300),hi-np.geomspace(.001,min(span,10*86400.),300)])
  grid=grid[(grid>=lo)&(grid<=hi)]
  y=sol.sol(grid)[:3].T
  for body in e.bodies:
   def distance(t):return norm(sol.sol(t)[:3]-(e.body_state(body[0],t)[:3]-origin))
   positions=np.array([e.body_state(body[0],t)[:3]-origin for t in grid]);dist=np.linalg.norm(y-positions,axis=1)
   candidates=[(dist[0],grid[0]),(dist[-1],grid[-1])]
   for k in np.flatnonzero((dist[1:-1]<=dist[:-2])&(dist[1:-1]<=dist[2:]))+1:
    m=minimize_scalar(distance,bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-5})
    candidates.append((m.fun,m.x))
   value,when=min(candidates)
   if value<minima[body[0]][0]:minima[body[0]]=[float(value),float(when)]
 for body in e.bodies:
  minimum=minima[body[0]][0]
  # 2 cm numerical allowance only at the modelled spherical Ceres surface.
  check('collision clearance '+body[3],body[2]-minimum,.02 if body[0]=='20000001' else 0.,'m')
 check('solar clearance',problem.config.minimum_solar_radius_m-minima['10'][0],0.,'m')
 for n,k in [(1,0),(4,1)]:
  rp=norm(nodes['relative'][n][:3]);check(f'Mars {n} minimum periapsis radius',problem.seed['minimum_mars_radius_m'][k]-rp,0.,'m')
  check(f'Mars {n} maximum periapsis radius',rp-problem.seed['maximum_mars_radius_m'][k],0.,'m')
  check(f'Mars {n} radial speed',abs(np.dot(nodes['relative'][n][:3],nodes['relative'][n][3:])/rp),1e-7,'m/s')
  if problem.seed['model']=='ballistic':check(f'Mars {n} zero impulse',norm(nodes['after'][n][3:]-nodes['before'][n][3:]),1e-10,'m/s')
  elif problem.config.max_flyby_burn_m_s is not None:
   check(f'Mars {n} burn limit',norm(nodes['after'][n][3:]-nodes['before'][n][3:]),problem.config.max_flyby_burn_m_s,'m/s')
 check('continuous Mars clearance',min(problem.seed['minimum_mars_radius_m'])-minima['499'][0],0.,'m')
 # Shared surface coordinates and surface-ground velocities are independently
 # recomputed with direct SPICE transforms at actual event dates.
 sites=[]
 for n in [2,3]:
  F=e.e.sp.sxform('J2000','IAU_CERES',e.epoch+times[n]);sites.append(F@np.array(nodes['relative'][n]))
 check('same body-fixed Ceres site',norm(sites[0][:3]-sites[1][:3]),.001,'m')
 radius=problem.config.visits[2]['radius_m']
 for n,site in zip([2,3],sites):check(f'Ceres {n} spherical surface radius',abs(norm(site[:3])-radius),.001,'m')
 check('mission duration',(times[-1]-times[0])/86400.,problem.config.max_duration_days,'days')
 stay=(times[3]-times[2])/86400.;bounds=problem.config.visits[2]['stay_days']
 check('Ceres minimum stay',bounds[0]-stay,0.,'days');check('Ceres maximum stay',stay-bounds[1],0.,'days')
 for i,(a,b) in enumerate(PAIRS):
  days=(times[b]-times[a])/86400.
  check(f'leg {i+1} minimum duration',problem.config.tof_days[i][0]-days,0.,'days')
  check(f'leg {i+1} maximum duration',days-problem.config.tof_days[i][1],0.,'days')
 launch=e.epoch+times[0];check('launch lower window',e.e.epoch(problem.config.launch_window[0])-launch,0.,'s')
 check('launch upper window',launch-e.e.epoch(problem.config.launch_window[1]),0.,'s')
 earthmu=e.mu[[b[0] for b in e.bodies].index('399')]
 c3=np.dot(nodes['relative'][0][3:],nodes['relative'][0][3:])-2*earthmu/norm(nodes['relative'][0][:3])
 if problem.config.launch_c3_limit_km2_s2 is not None:check('launch C3 limit',c3/1e6,problem.config.launch_c3_limit_km2_s2,'km2/s2')
 # Integration refinement for ALL eight half-arcs.
 fine=replace(problem.o,rtol=problem.o.rtol/5,atol_position=problem.o.atol_position/5,atol_velocity=problem.o.atol_velocity/5,atol_stm=problem.o.atol_stm/5,maximum_step=problem.o.maximum_step/2)
 for i,(a,b) in enumerate(PAIRS):
  for n,after in [(a,True),(b,False)]:
   coarse=problem.propagate_node(n,nodes,problem.cuts[i],after)['state'];refined=problem.propagate_node(n,nodes,problem.cuts[i],after,fine)['state'];d=coarse-refined
   check(f'leg {i+1} node {n} integration refinement position',norm(d[:3]),25.,'m')
   check(f'leg {i+1} node {n} integration refinement velocity',norm(d[3:]),5e-5,'m/s')
 # Half-spacing interpolation test with the same pinned direct SPICE inputs,
 # independently from integration refinement. Restore all ephemeris arrays even
 # if a propagation fails; context ownership remains unambiguous.
 if progress:progress('Half-spacing ephemeris convergence, all eight arcs')
 fine_times=np.sort(np.r_[e.times,(e.times[:-1]+e.times[1:])/2])
 fine_times=np.unique((e.epoch+fine_times)-e.epoch)
 fine_states=np.array([[e.body_state(b[0],t) for b in e.bodies] for t in fine_times])
 fine_poles=np.array([e.e.sp.pxform('IAU_MARS','J2000',e.epoch+t)[:,2] for t in fine_times])
 coarse_native,coarse_times,coarse_states=e.native,e.times,e.states
 baselines={(i,n):problem.propagate_node(n,nodes,problem.cuts[i],after,fine)['state'] for i,(a,b) in enumerate(PAIRS) for n,after in [(a,True),(b,False)]}
 with NBodyContext(fine_times,fine_states,e.mu,e.radii,j2=([b[0] for b in e.bodies].index('499'),1.96045e-3,3396200.,fine_poles)) as refined:
  try:
   e.native,e.times,e.states=refined,fine_times,fine_states
   # Preserve each coarse model's exact initial INERTIAL state; changing a
   # boundary to follow a newly interpolated body would test different data.
   for i,(a,b) in enumerate(PAIRS):
    for n,after in [(a,True),(b,False)]:
     start=times[n];body=nodes['ids'][n];j=[b[0] for b in e.bodies].index(body)
     k=max(0,min(len(coarse_times)-2,np.searchsorted(coarse_times,start)-1));origin=coarse_states[k,j,:3].copy()
     e.native,e.times,e.states=coarse_native,coarse_times,coarse_states
     planet,_=e.interpolated_state(body,start,origin);initial=planet+nodes['relative'][n]
     if after:initial[3:]+=nodes['after'][n,3:]-nodes['before'][n,3:]
     e.native,e.times,e.states=refined,fine_times,fine_states
     actual=refined.propagate(start,problem.cuts[i],initial,fine,origin=origin)['state'];actual[:3]+=origin
     difference=actual-baselines[(i,n)]
     check(f'leg {i+1} node {n} half-spacing position',norm(difference[:3]),25.,'m')
     check(f'leg {i+1} node {n} half-spacing velocity',norm(difference[3:]),5e-5,'m/s')
  finally:e.native,e.times,e.states=coarse_native,coarse_times,coarse_states
 # Check declared manoeuvre charges against independently computed local
 # parking and body-fixed ground states, rather than trusting a saved sum.
 independent_costs=[]
 for n in [0,5]:
  r,v=np.array(nodes['relative'][n][:3]),np.array(nodes['relative'][n][3:]);tangent=v-np.dot(v,r)*r/np.dot(r,r);tangent/=norm(tangent)
  independent_costs.append(norm(v-np.sqrt(earthmu/norm(r))*tangent))
 for n in [1,4]:independent_costs.append(norm(nodes['after'][n][3:]-nodes['before'][n][3:]))
 for n in [2,3]:
  F=e.e.sp.sxform('IAU_CERES','J2000',e.epoch+times[n]);body_site=sites[n-2][:3]
  ground=(F@np.r_[body_site,[0.,0.,0.]])[3:]
  independent_costs.append(norm(nodes['relative'][n][3:]-ground))
 check('all impulses charged',abs(sum(independent_costs)-nodes['total_delta_v_m_s']),1e-5,'m/s')
 passed=all(c['passed'] for c in checks)
 return dict(status='passed_global_nbody_numerical_checks' if passed else 'failed',checks=checks,arcs=arcs,
  minimum_body_separations_m=minima,launch_C3_km2_s2=c3/1e6,full_mission_reattached=passed,
  acceptance_tolerances=dict(independent_half_arc_position_m=100.,independent_midpoint_position_m=200.,independent_midpoint_velocity_m_s=.0003),
  qualification='Numerical multiple-shooting closure and direct-SPICE cross-check at stated tolerances, not flight-ready navigation accuracy; no independent GMAT whole-mission acceptance yet.',
  solutions_for_plot=solutions)
