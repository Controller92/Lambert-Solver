"""Finite Mars encounter lifting and two-arc state/epoch multiple shooting.

This local boundary-position correction is NOT a corrected complete mission.
Changed entry/exit velocities must be reattached to interplanetary phases.
Powered burns retain a fixed signed tangential magnitude, explicitly recorded.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .mission_config import load_mission
from .ephemeris import SpiceEphemeris
from .nbody_ephemeris import NBodyEphemeris
from .mission_search import digest,write_json
from ..nbody_native import Options


def norm(v):return float(np.linalg.norm(v))


def boundary(rperi,vperi,mu,radius,incoming):
 R=norm(rperi);speed=norm(vperi);vinf2=speed*speed-2*mu/R
 if vinf2<=0 or radius<=R:raise ValueError('Hyperbolic boundary must be outside periapsis')
 a=mu/vinf2;e=1+R/a
 H=np.arccosh((radius/a+1)/e)*(-1 if incoming else 1)
 P=rperi/R;Q=vperi/speed;sh,ch=np.sinh(H),np.cosh(H);q=np.sqrt(e*e-1)
 r=a*(e-ch)*P+a*q*sh*Q
 v=np.sqrt(mu/a)/(e*ch-1)*(-sh*P+q*ch*Q)
 t=np.sqrt(a**3/mu)*(e*sh-H)
 return float(t),np.r_[r,v]


def lift_candidate(candidate,config,node=1,boundary_radius=1e8,design_margin=10000.):
 if type(node) is not int or not 0<node<len(candidate['visits'])-1:raise ValueError('Invalid encounter node')
 visit=candidate['visits'][node]
 if visit['kind']!='flyby' or visit['body']!='499':raise ValueError('This increment corrects Mars flybys')
 if not candidate['feasible']:raise ValueError('Lift requires a feasible patched-conic source')
 if not np.isfinite([boundary_radius,design_margin]).all() or design_margin<0:raise ValueError('Invalid lifting limits')
 with SpiceEphemeris(config.kernels,'J2000','.research-cache') as e:
  F=e.sp.pxform(candidate['frame'],'J2000',visit['arrival_et'])
  vin=F@np.array(visit['vinf_in_m_s']);B=F@np.array(visit['b']);B/=norm(B)
  mu=visit['mu_m3_s2'];rp=visit['rp']+design_margin;S=vin/norm(vin)
  if rp>visit['maximum_radius_m']:raise ValueError('Lifted periapsis exceeds the configured maximum')
  sine=1/(1+rp*norm(vin)**2/mu);cosine=np.sqrt(1-sine*sine)
  rperi=rp*(sine*S+cosine*B);tangent=cosine*S-sine*B
  pi=np.sqrt(norm(vin)**2+2*mu/rp)*tangent
  vo=norm(visit['vinf_out_m_s'])
  po=pi.copy() if candidate['search_model']=='ballistic' else np.sqrt(vo*vo+2*mu/rp)*tangent
  tin,entry=boundary(rperi,pi,mu,boundary_radius,True)
  tout,exit=boundary(rperi,po,mu,boundary_radius,False)
 return dict(model=candidate['search_model'],epoch_et=visit['arrival_et'],node=node,mu_m3_s2=mu,
  incoming_relative_periapsis_state=np.r_[rperi,pi],outgoing_relative_periapsis_state=np.r_[rperi,po],
  entry_relative_state=entry,exit_relative_state=exit,entry_time=tin,exit_time=tout,
  signed_tangential_burn_m_s=float(np.dot(po-pi,tangent)),boundary_radius_m=boundary_radius,
  source_periapsis_radius_m=visit['rp'],lifted_periapsis_radius_m=rp,design_margin_added_m=design_margin,
  minimum_periapsis_radius_m=visit['minimum_radius_m'],
  maximum_periapsis_radius_m=visit['maximum_radius_m'],
  scope='Explicit local seed redesign with altitude margin; required outgoing asymptote changes with this redesign. Full mission is not reattached.')


def shooting_problem(ephem,seed,options=None):
 o=options or Options();tin,tout=seed['entry_time'],seed['exit_time']
 initial=ephem.body_state('499',tin)+np.asarray(seed['entry_relative_state'])
 target=ephem.body_state('499',tout)+np.asarray(seed['exit_relative_state'])
 mid0=np.asarray(seed['incoming_relative_periapsis_state'])
 burn=float(seed['signed_tangential_burn_m_s'])
 Rscale=1e7;Vscale=1e4;Tscale=1000.;scale=np.r_[np.full(3,Rscale),np.full(3,Vscale)]
 denominator=norm(mid0[:3])*norm(mid0[3:]);cache={};evaluations=[]
 def evaluate(q):
  key=np.asarray(q,dtype='f8').tobytes()
  if key in cache:return cache[key]
  cache.clear();t=float(q[9]*Tscale);body,mars_acc=ephem.interpolated_state('499',t)
  relative=mid0+q[3:9]*scale;mid=body+relative
  start=initial.copy();start[3:]+=q[:3]*Vscale
  speed=norm(relative[3:]);u=relative[3:]/speed
  after=mid.copy();after[3:]+=burn*u
  first=ephem.native.propagate(tin,t,start,o);second=ephem.native.propagate(t,tout,after,o)
  residual=np.r_[(first['state']-mid)/scale,(second['state'][:3]-target[:3])/Rscale,
                  np.dot(relative[:3],relative[3:])/denominator]
  # Position/velocity partials use propagated variational equations. Epoch
  # partials use endpoint vector fields and derivatives of the SAME Hermite
  # Mars state used at the node. Direct-SPICE checks remain independent.
  bodydot=np.r_[body[3:],mars_acc]
  f1=np.r_[first['state'][3:],ephem.native.force(t,first['state'][:3])[0]]
  f2=np.r_[after[3:],ephem.native.force(t,after[:3])[0]]
  M=np.eye(6);M[3:,3:]+=burn/speed*(np.eye(3)-np.outer(u,u))
  J=np.zeros((10,10))
  J[:6,:3]=first['stm'][:,3:]*Vscale/scale[:,None]
  J[:6,3:9]=-np.eye(6)
  J[:6,9]=(f1-bodydot)*Tscale/scale
  J[6:9,3:9]=(second['stm'][:3]@M)*scale[None,:]/Rscale
  J[6:9,9]=second['stm'][:3]@(bodydot-f2)*Tscale/Rscale
  J[9,3:6]=relative[3:]*Rscale/denominator
  J[9,6:9]=relative[:3]*Vscale/denominator
  value=dict(residual=residual,jacobian=J,first=first,second=second,entry=start,target=target,
   mid=mid,after=after,relative=relative,time=t)
  cache[key]=value;evaluations.append(dict(norm=float(norm(residual)),time=t));return value
 bound=min(abs(tin),tout)*.4/Tscale
 lower=np.r_[np.full(3,-.2),np.full(3,-.5),np.full(3,-.2),-bound]
 upper=-lower
 return dict(evaluate=evaluate,lower=lower,upper=upper,initial=initial,target=target,options=o,
             entry_time=tin,exit_time=tout,history=evaluations)


def shoot(ephem,seed,options=None,max_evaluations=100):
 if type(max_evaluations) is not int or max_evaluations<1:raise ValueError('Invalid correction budget')
 problem=shooting_problem(ephem,seed,options)
 evaluate=problem['evaluate'];lower,upper=problem['lower'],problem['upper']
 o=problem['options'];initial,target=problem['initial'],problem['target']
 tin,tout=problem['entry_time'],problem['exit_time'];burn=float(seed['signed_tangential_burn_m_s'])
 result=least_squares(lambda q:evaluate(q)['residual'],np.zeros(10),jac=lambda q:evaluate(q)['jacobian'],
  bounds=(lower,upper),x_scale='jac',xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=max_evaluations)
 final=evaluate(result.x)
 position=norm(final['first']['state'][:3]-final['mid'][:3]);velocity=norm(final['first']['state'][3:]-final['mid'][3:])
 terminal=norm(final['second']['state'][:3]-target[:3]);rp=norm(final['relative'][:3])
 radial=float(np.dot(final['relative'][:3],final['relative'][3:])/rp)
 passed=(position<.05 and velocity<1e-6 and terminal<.05 and abs(radial)<1e-5 and
         seed['minimum_periapsis_radius_m']<=rp<=seed['maximum_periapsis_radius_m'])
 return dict(status='locally_corrected' if passed else 'local_correction_failed',optimizer_success=bool(result.success),
  optimizer_message=str(result.message),iterations=int(result.nfev),options=asdict(o),
  entry_time=tin,exit_time=tout,encounter_time=final['time'],
  entry_state_shifted_si=final['entry'],node_before_shifted_si=final['mid'],node_after_shifted_si=final['after'],
  exit_state_shifted_si=final['second']['state'],exit_target_position_shifted_m=target[:3],
  entry_velocity_change_m_s=final['entry'][3:]-initial[3:],exit_velocity_change_m_s=final['second']['state'][3:]-target[3:],
  signed_tangential_burn_m_s=burn,periapsis_radius_m=rp,periapsis_altitude_m=rp-3396200.,
  state_defect_position_m=position,state_defect_velocity_m_s=velocity,terminal_position_error_m=terminal,radial_velocity_at_node_m_s=radial,
  entry_to_node_stm=final['first']['stm'],node_to_exit_stm=final['second']['stm'],
  decision_vector=result.x,scaled_residuals=final['residual'],scaled_jacobian=final['jacobian'],evaluation_history=problem['history'],
  full_mission_reattached=False,validation_status='independent_validation_required',
  scope='Finite local boundary-position correction. Changed boundary velocities and lifting margin require global mission re-optimization.')
