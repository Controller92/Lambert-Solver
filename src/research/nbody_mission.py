"""Joint finite-endpoint EMC shooting. Every velocity jump is an explicit burn.

The first global layer retains spherical impulsive Earth/Ceres operations and
adds ephemeris-driven point masses plus oriented Mars J2. It is not a flight
operations model. Four two-ended shooting legs avoid unstable forward shooting
through a close encounter. All six event epochs and the Ceres site are shared.
"""
from dataclasses import asdict
from pathlib import Path
import json,uuid
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .nbody_ephemeris import NBodyEphemeris
from .nbody_flyby import lift_candidate
from .ephemeris import SpiceEphemeris
from .mission_config import load_mission
from .mission_search import digest,write_json,configuration_fingerprint
from ..nbody_native import Options

DAY=86400.
PAIRS=[(0,1),(1,2),(3,4),(4,5)]
SCALE=np.r_[np.full(3,1e9),np.full(3,1e3)]


def norm(x):return float(np.linalg.norm(x))


def chart(base,q):
 u=np.asarray(base)/norm(base);ref=np.eye(3)[np.argmin(abs(u))]
 a=np.cross(u,ref);a/=norm(a);b=np.cross(u,a)
 out=u+q[0]*a+q[1]*b
 return out/norm(out)


def tangent_velocity(base,r,q):
 u=base-np.dot(base,r)*r/np.dot(r,r);u/=norm(u)
 w=np.cross(r/norm(r),u);direction=u+q[1]*w;direction/=norm(direction)
 return (norm(base)+q[0]*1000.)*direction


def seed_mission(candidate,config,margin=10000.):
 if [v['body'] for v in config.visits]!=['399','499','20000001','499','399']:
  raise ValueError('Global correction currently supports the EMC round trip')
 flybys=[lift_candidate(candidate,config,n,design_margin=margin) for n in (1,3)]
 visits=candidate['visits'];epoch=candidate['legs'][0]['departure_et']
 times=np.array([epoch,visits[1]['arrival_et'],visits[2]['arrival_et'],visits[2]['departure_et'],visits[3]['arrival_et'],candidate['legs'][-1]['arrival_et']])-epoch
 with SpiceEphemeris(config.kernels,'J2000','.research-cache') as e:
  F=e.sp.pxform(candidate['frame'],'J2000',epoch)
  relative=[np.r_[F@visits[0]['rperi_m'],F@visits[0]['hyperbolic_velocity_m_s']],None,None,None,None,
            np.r_[F@visits[4]['rperi_m'],F@visits[4]['hyperbolic_velocity_m_s']]]
  for n,f in zip((1,4),flybys):relative[n]=np.array(f['incoming_relative_periapsis_state'])
  site=np.array(visits[2]['site_body_fixed_m'])
  for n,key in [(2,'in'),(3,'out')]:
   # Candidate inertial frame is constant relative to J2000.
   relative[n]=np.r_[F@visits[2]['rperi_'+key+'_m'],F@visits[2]['vperi_'+key+'_m_s']]
 return dict(epoch_et=epoch,times=times,relative_states=relative,site_body_fixed_m=site,
  signed_burns_m_s=[f['signed_tangential_burn_m_s'] for f in flybys],model=candidate['search_model'],
  minimum_mars_radius_m=[f['minimum_periapsis_radius_m'] for f in flybys],maximum_mars_radius_m=[f['maximum_periapsis_radius_m'] for f in flybys],
  design_margin_m=margin,source_total_delta_v_m_s=candidate['total_delta_v_m_s'])


def mission_knots(times,refinement=1):
 # Dense near ALL departure/arrival bodies; slow distant-body interpolation
 # error is assessed by direct-SPICE independent propagation and refinement.
 lower=-6*DAY;upper=float(times[-1]+6*DAY)
 knots=[np.arange(lower,upper,10800./refinement),np.array([lower,upper])]
 for t in times:
  knots.append(np.arange(max(lower,t-5*DAY),min(upper,t+5*DAY),120./refinement))
 return np.unique(np.concatenate(knots))


class GlobalProblem:
 def __init__(self,ephem,seed,config,options=None,progress=None):
  self.e=ephem;self.seed=seed;self.config=config;self.progress=progress
  self.o=options or Options(rtol=4e-14,atol_position=1e-5,atol_velocity=1e-10,atol_stm=1e-12,maximum_step=10800.)
  self.base=np.array(seed['relative_states']);self.times=np.array(seed['times']);self.powered=seed['model']=='powered'
  self.cuts=[(self.times[a]+self.times[b])/2 for a,b in PAIRS]
  # PCK Ceres has a fixed pole and constant spin. Anchor SPICE once at
  # each event, then evaluate smooth relative-time rotation without adding a
  # sub-microsecond design change to a billion-second absolute ET number.
  for key in ['BODY2000001_POLE_RA','BODY2000001_POLE_DEC']:
   if np.any(ephem.e.sp.gdpool(key,0,3)[1:]!=0):raise ValueError('Ceres rotation model needs a nonuniform-spin implementation')
  if ephem.e.sp.gdpool('BODY2000001_PM',0,3)[2]!=0:raise ValueError('Ceres spin acceleration unsupported')
  self.rotations={n:ephem.e.sp.sxform('IAU_CERES','J2000',ephem.epoch+self.times[n]) for n in [2,3]}
  self.slices={};index=0
  for name,n in [('earth0',4),('mars0',6 if self.powered else 5),('ceres',6),('mars1',6 if self.powered else 5),('earth1',4),('dates',6)]:
   self.slices[name]=slice(index,index+n);index+=n
  self.size=index;self.cache={};self.history=[]
  self.lower=np.full(index,-.5);self.upper=np.full(index,.5)
  for name in ['mars0','mars1']:
   self.lower[self.slices[name].start+2]=-3.;self.upper[self.slices[name].start+2]=3.
   if self.powered:self.lower[self.slices[name].stop-1]=-2.;self.upper[self.slices[name].stop-1]=2.
  self.lower[self.slices['dates']]=-172.8;self.upper[self.slices['dates']]=172.8

 def nodes(self,q):
  t=self.times+q[self.slices['dates']]*1000.
  before=[];after=[];relative=[];ground=[]
  ids=['399','499','20000001','20000001','499','399']
  for n,body in enumerate(ids):
   if n in (0,5):
    z=q[self.slices['earth0' if n==0 else 'earth1']]
    r=norm(self.base[n,:3])*chart(self.base[n,:3],z[:2]);v=tangent_velocity(self.base[n,3:],r,z[2:4]);vout=v
    ground.append(None)
   elif n in (1,4):
    k=0 if n==1 else 1;z=q[self.slices['mars'+str(k)]]
    minimum=self.seed['minimum_mars_radius_m'][k];gap=norm(self.base[n,:3])-minimum
    r=(minimum+gap*np.exp(z[2]))*chart(self.base[n,:3],z[:2]);v=tangent_velocity(self.base[n,3:],r,z[3:5])
    burn=self.seed['signed_burns_m_s'][k]+z[5]*1000. if self.powered else 0.
    vout=v+burn*v/norm(v);ground.append(None)
   else:
    z=q[self.slices['ceres']];site=norm(self.seed['site_body_fixed_m'])*chart(self.seed['site_body_fixed_m'],z[:2])
    F=self.rotations[n];W=F[3:,:3]@F[:3,:3].T;omega=np.array([W[2,1],W[0,2],W[1,0]])
    rotation=Rotation.from_rotvec(omega*(q[self.slices['dates']][n]*1000.)).as_matrix()@F[:3,:3]
    position=rotation@site;g=np.r_[position,W@position]
    r=g[:3];v=tangent_velocity(self.base[n,3:],r,z[(2 if n==2 else 4):(4 if n==2 else 6)]);vout=v;ground.append(g[3:])
   planet,_=self.e.interpolated_state(body,float(t[n]));rel=np.r_[r,v]
   state=planet+rel;out=planet+np.r_[r,vout]
   before.append(state);after.append(out);relative.append(rel)
  radial=np.array([np.dot(r[:3],r[3:])/(norm(r[:3])*10000.) for r in relative])
  # Local circular parking planes are free, tangent at the actual Earth node.
  costs=[]
  for n in [0,5]:
   r,v=relative[n][:3],relative[n][3:];u=v-np.dot(v,r)*r/np.dot(r,r);u/=norm(u)
   mu=self.e.mu[[b[0] for b in self.e.bodies].index('399')];circular=np.sqrt(mu/norm(r))*u
   costs.append(dict(name='Earth injection' if n==0 else 'Earth capture',delta_v_m_s=norm(v-circular),vector_m_s=v-circular if n==0 else circular-v))
  for n in [1,4]:costs.append(dict(name='Mars '+str(n)+' impulse',delta_v_m_s=norm(after[n][3:]-before[n][3:]),vector_m_s=after[n][3:]-before[n][3:]))
  for n in [2,3]:
   dv=ground[n]-relative[n][3:] if n==2 else relative[n][3:]-ground[n]
   costs.append(dict(name='Ceres landing' if n==2 else 'Ceres launch',delta_v_m_s=norm(dv),vector_m_s=dv))
  return dict(times=t,before=np.array(before),after=np.array(after),relative=np.array(relative),radial=radial,costs=costs,
              total_delta_v_m_s=sum(c['delta_v_m_s'] for c in costs),ids=ids)

 def propagate_node(self,n,node,end,after=False,options=None):
  t=node['times'][n];body=node['ids'][n]
  j=[b[0] for b in self.e.bodies].index(body)
  k=min(len(self.e.times)-2,max(0,np.searchsorted(self.e.times,t,side='right')-1))
  origin=self.e.states[k,j,:3].copy()
  planet,_=self.e.interpolated_state(body,t,origin)
  relative=node['relative'][n].copy()
  if after:relative[3:]+=node['after'][n,3:]-node['before'][n,3:]
  initial=relative+planet
  result=self.e.native.propagate(t,end,initial,options or self.o,origin=origin)
  result['state'][:3]+=origin
  return result

 def evaluate(self,q):
  key=np.asarray(q,dtype='f8').tobytes()
  if key in self.cache:return self.cache[key]
  self.cache.clear();node=self.nodes(q);t=node['times'];res=[];maps=[]
  for i,(a,b) in enumerate(PAIRS):
   forward=self.propagate_node(a,node,self.cuts[i],True)
   backward=self.propagate_node(b,node,self.cuts[i])
   res.extend((forward['state']-backward['state'])/SCALE);maps.append((forward,backward))
  res=np.array(res)
  # Only endpoint coordinate/epoch derivatives use finite differences. Each
  # integration's derivative is supplied by its analytic variational equations.
  J=np.zeros((24,self.size));h=1e-5
  for k in range(self.size):
   delta=np.zeros(self.size);delta[k]=h
   plus=self.nodes(q+delta);minus=self.nodes(q-delta)
   db=(plus['before']-minus['before'])/(2*h);da=(plus['after']-minus['after'])/(2*h)
   dt=(plus['times']-minus['times'])/(2*h)
   for i,(a,b) in enumerate(PAIRS):
    f,g=maps[i]
    f0=np.r_[node['after'][a,3:],self.e.native.force(t[a],node['after'][a,:3])[0]]
    g0=np.r_[node['before'][b,3:],self.e.native.force(t[b],node['before'][b,:3])[0]]
    J[i*6:(i+1)*6,k]=(f['stm']@(da[a]-f0*dt[a])-g['stm']@(db[b]-g0*dt[b]))/SCALE
  item=dict(q=np.array(q),residual=res,jacobian=J,nodes=node,maps=maps)
  self.cache[key]=item
  self.history.append(dict(evaluation=len(self.history),residual_norm=norm(res),delta_v_m_s=node['total_delta_v_m_s']))
  if self.progress:self.progress(self.history[-1])
  return item

 def solve(self,max_evaluations=100,initial=None):
  x=np.zeros(self.size) if initial is None else np.array(initial)
  solved=least_squares(lambda x:self.evaluate(x)['residual'],x,jac=lambda x:self.evaluate(x)['jacobian'],bounds=(self.lower,self.upper),
    x_scale='jac',ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=max_evaluations)
  # Once close, hold the rapidly rotating Ceres epochs and active bounds
  # while polishing all remaining geometry and epoch variables. Column
  # scaling avoids favouring arbitrary units in the minimum-norm step.
  x=solved.x.copy();value=self.evaluate(x)
  if norm(value['residual'])<1e-4:
   for _ in range(16):
    ceres_dates=[self.slices['dates'].start+2,self.slices['dates'].start+3]
    free=[j for j in range(self.size) if j not in ceres_dates and x[j]-self.lower[j]>1e-6 and self.upper[j]-x[j]>1e-6]
    A=value['jacobian'][:,free];column_scale=1/np.maximum(np.linalg.norm(A,axis=0),1e-12)
    step=np.zeros(self.size);step[free]=column_scale*np.linalg.lstsq(A*column_scale,-value['residual'],rcond=1e-12)[0]
    accepted=False
    for factor in [1.,.5,.25,.125,.0625]:
     trial=x+factor*step
     if np.any(trial<self.lower) or np.any(trial>self.upper):continue
     new=self.evaluate(trial)
     if norm(new['residual'])<norm(value['residual']):x=trial;value=new;accepted=True;break
    if not accepted or norm(value['residual'])<1e-12:break
  node=value['nodes']
  defects=[dict(position_m=norm((a['state']-b['state'])[:3]),velocity_m_s=norm((a['state']-b['state'])[3:])) for a,b in value['maps']]
  return dict(status='shooting_converged' if max(v['position_m'] for v in defects)<1. and max(v['velocity_m_s'] for v in defects)<1e-5 and max(abs(node['radial']))<1e-8 else 'not_converged',
   model=self.seed['model'],decision_vector=x,optimizer_success=bool(solved.success),optimizer_message=str(solved.message),options=asdict(self.o),
   nodes=node,midpoint_defects=defects,residuals=value['residual'],jacobian=value['jacobian'],history=self.history,
   full_mission_reattached=False,validation_status='independent full-chain validation required',
   model_scope='Four N-body cruise legs, oriented Mars J2, instantaneous tangential Mars impulses, free-plane spherical Earth parking, same rotating spherical Ceres surface site with instantaneous landing/launch. No mass/thrust or uncertainty model.')


def correct_mission(candidate_path,outdir='results',config_path=None,max_evaluations=160,progress=None):
 """Run, independently validate and preserve a complete finite EMC correction."""
 from importlib.metadata import version
 import platform
 from .nbody_mission_validation import validate_global
 from .nbody_mission_plot import plot_global
 if type(max_evaluations) is not int or max_evaluations<1:raise ValueError('Positive evaluation budget required')
 path=Path(candidate_path).resolve();folder=path.parent.parent if path.parent.name=='candidates' else path.parent
 config=load_mission(config_path or folder/'config.json');candidate=json.loads(path.read_text())
 acceptance=path.with_suffix('.validation.json')
 if not acceptance.is_file():raise ValueError('Source requires independent patched-conic validation')
 record=json.loads(acceptance.read_text())
 if record.get('status')!='passed_patched_conic' or record.get('candidate_sha256')!=digest(path):raise ValueError('Source acceptance hash/status mismatch')
 if candidate.get('fingerprint',{}).get('config_sha256')!=configuration_fingerprint(config):raise ValueError('Source mission configuration differs')
 seed=seed_mission(candidate,config)
 run=Path(outdir)/('global-'+seed['model']+'-'+uuid.uuid4().hex[:12]);run.mkdir(parents=True,exist_ok=False)
 write_json(run/'seed.json',seed);write_json(run/'source-mission-config.json',config.to_dict())
 (run/'source-candidate.json').write_bytes(path.read_bytes());(run/'source-candidate.validation.json').write_bytes(acceptance.read_bytes())
 root=Path(__file__).resolve().parents[2]
 sources=[Path('native/nbody')/p for p in ['api.h','cpu.cpp']]+[Path('src/nbody_native.py')]
 sources += [Path('src/research')/p for p in ['nbody_mission.py','nbody_mission_validation.py','nbody_mission_plot.py','nbody_ephemeris.py','nbody_flyby.py','ephemeris.py','mission_config.py','mission_search.py']]
 write_json(run/'fingerprint.json',dict(source_sha256={str(p):digest(root/p) for p in sources},library_sha256=digest(root/'build/native/libnbody_cpu.so'),
  python=platform.python_version(),versions={p:version(p) for p in ['numpy','scipy','spiceypy']}))
 for p in sources:
  dest=run/'sources'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/p).read_bytes())
 try:
  knots=mission_knots(seed['times'])
  if progress:progress(dict(stage='building ephemeris',knots=len(knots)))
  with NBodyEphemeris(config.kernels,seed['epoch_et'],knots[0],knots[-1],times=knots,mars_j2=True,surface_guard_slack=1.,cache_dir=run/'state-cache') as e:
   write_json(run/'ephemeris.json',e.provenance())
   np.savez_compressed(run/'ephemeris.npz',times=e.times,states=e.states,mu=e.mu,radii=e.radii,poles=e.poles)
   problem=GlobalProblem(e,seed,config,progress=progress)
   result=problem.solve(max_evaluations)
   write_json(run/'result.json',result)
   validation=validate_global(problem,result,progress=(lambda s:progress(dict(stage=s))) if progress else None)
   solutions=validation.pop('solutions_for_plot')
   plot_global(e,seed,result,solutions,run/'mission.png')
   # Dense-output samples and all variational maps preserve the actual solved
   # finite trajectory, including both sides of each declared impulse.
   records=[]
   for sol in solutions:
    t=np.linspace(sol['start'],sol['end'],1201);states=sol['sol'].sol(t).T;states[:,:3]+=sol['origin']
    records.append(dict(leg=sol['leg'],node=sol['node'],times=t,states_shifted_si=states))
   write_json(run/'trajectory-samples.json',records)
   write_json(run/'shooting-maps.json',[[dict(state=m['state'],stm=m['stm']) for m in pair] for pair in problem.evaluate(result['decision_vector'])['maps']])
   validation['result_sha256']=digest(run/'result.json');validation['seed_sha256']=digest(run/'seed.json')
   write_json(run/'validation.json',validation)
  passed=validation['status']=='passed_global_nbody_numerical_checks'
  summary=dict(status='passed_global_nbody_numerical_checks' if passed else 'failed',model=seed['model'],full_mission_reattached=passed,
   duration_years=(result['nodes']['times'][-1]-result['nodes']['times'][0])/(DAY*365.25),
   total_delta_v_m_s=result['nodes']['total_delta_v_m_s'],source_total_delta_v_m_s=seed['source_total_delta_v_m_s'],
   maximum_native_midpoint_position_defect_m=max(v['position_m'] for v in result['midpoint_defects']),
   maximum_native_midpoint_velocity_defect_m_s=max(v['velocity_m_s'] for v in result['midpoint_defects']),
   maximum_independent_half_arc_position_difference_m=max(a['position_difference_m'] for a in validation['arcs']),
   check_count=len(validation['checks']),failed_checks=[c for c in validation['checks'] if not c['passed']],
   external_gmat_full_mission_validation='not_run',cost_optimization='feasibility correction of a known basin; not a minimum-delta-v or global-optimality claim',
   scope=result['model_scope'])
  write_json(run/'summary.json',summary)
  write_json(run/'artifact-hashes.json',{str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()})
  return run,summary
 except Exception as error:
  write_json(run/'failure.json',dict(type=type(error).__name__,message=str(error),full_mission_reattached=False));raise
