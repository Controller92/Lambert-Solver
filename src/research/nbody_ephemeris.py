"""Pinned geometric J2000/SSB ephemerides for the C++ N-body propagator.

The coordinate offset is one CONSTANT barycentric vector, not a moving origin.
All positions, spacecraft velocities and forces retain inertial consistency.
Planet-system barycentres are explicit distant point-mass approximations;
Earth/Moon and Mars/Phobos/Deimos are resolved without mass double-counting.
"""
import numpy as np
from .ephemeris import SpiceEphemeris
from ..nbody_native import NBodyContext

BODIES=[('10','10',695700000.,'Sun'),('1','1',2439700.,'Mercury system'),
 ('2','2',6051800.,'Venus system'),('399','399',6378137.,'Earth'),('301','301',1737400.,'Moon'),
 ('499','499',3396200.,'Mars'),('401','401',13500.,'Phobos'),('402','402',8000.,'Deimos'),
 ('5','5',71492000.,'Jupiter system'),('6','6',60268000.,'Saturn system'),
 ('7','7',25559000.,'Uranus system'),('8','8',24764000.,'Neptune system'),
 ('9','9',1188300.,'Pluto system'),('20000001','2000001',482100.,'Ceres')]


class NBodyEphemeris:
 def __init__(self,kernels,epoch,lower,upper,step=60.,cache_dir='.research-cache',bodies=None):
  self.e=SpiceEphemeris(kernels,'J2000',cache_dir);self.native=None
  try:
   self.epoch=float(epoch);self.lower=float(lower);self.upper=float(upper);self.step=float(step)
   if not np.isfinite([epoch,lower,upper,step]).all() or lower>=upper or step<=0:raise ValueError('Invalid ephemeris interval')
   self.bodies=list(BODIES if bodies is None else bodies)
   ids=[b[0] for b in self.bodies]
   if len(ids)!=len(set(ids)):raise ValueError('Duplicate force bodies')
   for parent,children in [('1',['199']),('2',['299']),('3',['399','301']),('4',['499','401','402']),('5',['599']),('6',['699']),('7',['799']),('8',['899']),('9',['999'])]:
    if parent in ids and any(k in ids for k in children):raise ValueError('Body/system mass double-counting')
   count=int(np.ceil((upper-lower)/step))+1
   if count>2000000:raise ValueError('Ephemeris table exceeds two million knots')
   self.times=np.linspace(lower,upper,count)
   self.offset=self.direct('499',0.)[:3].copy()
   self.mu=np.array([float(self.e.sp.bodvrd(b[1],'GM',1)[1][0])*1e9 for b in self.bodies])
   self.radii=np.array([b[2] for b in self.bodies])
   self.states=np.empty((count,len(ids),6))
   for k,t in enumerate(self.times):
    for j,body in enumerate(ids):
     self.states[k,j]=self.direct(body,t);self.states[k,j,:3]-=self.offset
   self.native=NBodyContext(self.times,self.states,self.mu,self.radii)
  except Exception:self.close();raise

 def direct(self,body,time):
  return np.array(self.e.sp.spkezr(body,self.epoch+float(time),'J2000','NONE','0')[0])*1000.

 def body_state(self,body,time):
  state=self.direct(body,time);state[:3]-=self.offset;return state

 def interpolated_state(self,body,time):
  if time<self.lower or time>self.upper:raise ValueError('Ephemeris coverage')
  j=[b[0] for b in self.bodies].index(body)
  k=min(len(self.times)-2,max(0,np.searchsorted(self.times,time,side='right')-1))
  h=self.times[k+1]-self.times[k];u=(time-self.times[k])/h
  a,b=self.states[k,j],self.states[k+1,j]
  p=(2*u**3-3*u*u+1)*a[:3]+(u**3-2*u*u+u)*h*a[3:]+(-2*u**3+3*u*u)*b[:3]+(u**3-u*u)*h*b[3:]
  v=(6*u*u-6*u)/h*a[:3]+(3*u*u-4*u+1)*a[3:]+(-6*u*u+6*u)/h*b[:3]+(3*u*u-2*u)*b[3:]
  acc=(12*u-6)/h**2*a[:3]+(6*u-4)/h*a[3:]+(-12*u+6)/h**2*b[:3]+(6*u-2)/h*b[3:]
  return np.r_[p,v],acc

 def exact_force(self,time,state):
  # Independent vectorized force using direct SPICE, never native interpolation.
  r=np.asarray(state)[:3]
  positions=np.array([self.body_state(b[0],time)[:3] for b in self.bodies])
  d=positions-r;length=np.linalg.norm(d,axis=1)
  if (length<=self.radii).any():raise ValueError('Collision in independent force model')
  return np.sum(d*(self.mu/length**3)[:,None],axis=0)

 def interpolation_check(self):
  # All interval midpoints plus staggered/random phases. This measures sampled
  # interpolation error, not a rigorous continuous supremum bound.
  mid=(self.times[:-1]+self.times[1:])/2
  rng=np.random.default_rng(9587)
  extra=self.times[:-1]+rng.uniform(.1,.9,len(mid))*np.diff(self.times)
  maximum=np.zeros(len(self.bodies))
  for t in np.r_[mid,extra]:
   k=min(len(self.times)-2,np.searchsorted(self.times,t,side='right')-1)
   h=self.times[k+1]-self.times[k];u=(t-self.times[k])/h
   a,b=self.states[k],self.states[k+1]
   value=(2*u**3-3*u*u+1)*a[:,:3]+(u**3-2*u*u+u)*h*a[:,3:]+(-2*u**3+3*u*u)*b[:,:3]+(u**3-u*u)*h*b[:,3:]
   exact=np.array([self.body_state(body[0],t)[:3] for body in self.bodies])
   maximum=np.maximum(maximum,np.linalg.norm(value-exact,axis=1))
  return dict(sample_count=len(mid)+len(extra),maximum_position_difference_m=float(max(maximum)),
   bodies=[dict(id=b[0],name=b[3],maximum_position_difference_m=float(v)) for b,v in zip(self.bodies,maximum)],
   qualification='sampled midpoint/random-phase check; spacing refinement and direct-SPICE trajectory checks also required')

 def provenance(self):
  return dict(frame='J2000/ICRF axes',ephemeris_center='Solar System Barycenter (0)',aberration='NONE',
   time_scale='TDB ET seconds from J2000',epoch_et=self.epoch,constant_inertial_offset_m=self.offset,
   knots=len(self.times),lower_seconds=self.lower,upper_seconds=self.upper,maximum_spacing_seconds=float(max(np.diff(self.times))),
   interpolation='cubic Hermite, SPICE position and velocity',kernels=self.e.provenance,
   force_model='Newtonian ephemeris-driven test particle; direct barycentric point-mass sum',
   bodies=[dict(id=b[0],gm_id=b[1],mu_m3_s2=float(mu),collision_radius_m=b[2],label=b[3]) for b,mu in zip(self.bodies,self.mu)],
   limitations='No harmonics, relativity, radiation pressure, finite thrust or uncertainty model in this first correction layer')

 def close(self):
  if self.native is not None:self.native.close();self.native=None
  if getattr(self,'e',None) is not None:self.e.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
