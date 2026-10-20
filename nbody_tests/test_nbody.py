"""Dynamics/ABI/error contracts plus opt-in real finite-encounter validation."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from src.nbody_native import NBodyContext,Options

ROOT=Path(__file__).resolve().parents[1]


def circular_context():return NBodyContext([-10.,10.],np.zeros((2,1,6)),[1.],[0.])


class NativeTests(unittest.TestCase):
 def test_finite_hyperbolic_lift_energy_momentum_and_clock(self):
  from src.research.nbody_flyby import boundary
  mu=4.282837362069909e13;rp=3606200.;speed=np.sqrt(4000.**2+2*mu/rp)
  peri=np.array([rp,0.,0.,0.,speed,0.])
  def ode(t,y):return np.r_[y[3:],-mu*y[:3]/np.linalg.norm(y[:3])**3]
  for incoming in [True,False]:
   t,state=boundary(peri[:3],peri[3:],mu,1e8,incoming)
   self.assertEqual(t<0,incoming)
   self.assertAlmostEqual(np.linalg.norm(state[:3]),1e8,places=5)
   energy=np.dot(state[3:],state[3:])-2*mu/np.linalg.norm(state[:3])
   self.assertAlmostEqual(energy/4000**2,1.,places=12)
   np.testing.assert_allclose(np.cross(state[:3],state[3:]),np.cross(peri[:3],peri[3:]),rtol=3e-15,atol=1e-3)
   sol=solve_ivp(ode,(0,t),peri,method='DOP853',rtol=3e-14,atol=1e-7)
   np.testing.assert_allclose(sol.y[:3,-1],state[:3],atol=.001,rtol=0)

 def test_point_mass_force_gradient_and_symmetry(self):
  with circular_context() as c:
   r=np.array([1.2,-.7,.4]);a,G=c.force(0,r);R=np.linalg.norm(r)
   np.testing.assert_allclose(a,-r/R**3,rtol=2e-15)
   np.testing.assert_allclose(G,3*np.outer(r,r)/R**5-np.eye(3)/R**3,rtol=2e-15)
   np.testing.assert_allclose(G,G.T,atol=1e-15)
   self.assertAlmostEqual(np.trace(G),0,places=14)
   for j in range(3):
    delta=np.eye(3)[j]*1e-5
    fd=(c.force(0,r+delta)[0]-c.force(0,r-delta)[0])/2e-5
    np.testing.assert_allclose(fd,G[:,j],rtol=2e-9,atol=1e-10)

 def test_oriented_j2_force_gradient_potential_and_stm(self):
  pole=np.array([.3,.4,np.sqrt(.75)]);poles=np.array([pole,pole])
  r=np.array([1.2,-.7,.4]);J2=.002;R=.7
  with NBodyContext([-10.,10.],np.zeros((2,1,6)),[1.],[0.],j2=(0,J2,R,poles)) as c:
   a,G=c.force(0,r)
   # Independent potential derivative checks acceleration, including sign and
   # rotation, rather than reproducing the native force expression.
   def potential(x):
    d=np.linalg.norm(x);z=np.dot(x,pole)/d
    return -1/d*(1-J2*(R/d)**2*(3*z*z-1)/2)
   for j in range(3):
    d=np.eye(3)[j]*1e-5
    self.assertAlmostEqual(a[j],-(potential(r+d)-potential(r-d))/2e-5,places=9)
    fd=(c.force(0,r+d)[0]-c.force(0,r-d)[0])/2e-5
    np.testing.assert_allclose(fd,G[:,j],rtol=3e-9,atol=1e-10)
   np.testing.assert_allclose(G,G.T,atol=1e-15)
   self.assertAlmostEqual(np.trace(G),0.,places=14)
   state=np.r_[r,[.1,.8,.2]];base=c.propagate(0,.5,state)
   for j in range(6):
    delta=np.eye(6)[j]*1e-6
    fd=(c.propagate(0,.5,state+delta)['state']-c.propagate(0,.5,state-delta)['state'])/2e-6
    np.testing.assert_allclose(fd,base['stm'][:,j],rtol=3e-8,atol=3e-9)
  with self.assertRaises(ValueError):NBodyContext([-1,1],np.zeros((2,1,6)),[1],[0],j2=(0,J2,R,np.ones((2,3))))

 def test_two_body_period_reverse_and_stm(self):
  state=np.array([1.,0,0,0,1,0])
  with circular_context() as c:
   result=c.propagate(0,2*np.pi,state)
   np.testing.assert_allclose(result['state'],state,atol=2e-10,rtol=0)
   reverse=c.propagate(2*np.pi,0,result['state'])
   np.testing.assert_allclose(reverse['state'],state,atol=2e-10,rtol=0)
   np.testing.assert_allclose(reverse['stm']@result['stm'],np.eye(6),atol=2e-8,rtol=0)
   for j in range(6):
    delta=np.eye(6)[j]*1e-6
    fd=(c.propagate(0,2*np.pi,state+delta)['state']-c.propagate(0,2*np.pi,state-delta)['state'])/2e-6
    np.testing.assert_allclose(fd,result['stm'][:,j],atol=3e-8,rtol=3e-8)

 def test_moving_hermite_body_and_coordinate_translation(self):
  # A cubic moving source is represented exactly by its endpoint p/v data.
  def body(t):return np.array([4+t+.1*t**3,3-.2*t*t,2.,1+.3*t*t,-.4*t,0.])
  states=np.array([[body(0)],[body(2)]])
  with NBodyContext([0,2],states,[2.],[0.]) as c:
   r=np.array([1.,.2,0]);d=body(.7)[:3]-r
   np.testing.assert_allclose(c.force(.7,r)[0],2*d/np.linalg.norm(d)**3,rtol=2e-15)
   y=np.array([1.,.2,0,.3,.4,.1]);actual=c.propagate(0,2,y)['state']
   def ode(t,y):
    d=body(t)[:3]-y[:3];return np.r_[y[3:],2*d/np.linalg.norm(d)**3]
   expected=solve_ivp(ode,(0,2),y,method='DOP853',rtol=3e-14,atol=1e-14).y[:,-1]
   np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
   offset=np.array([100.,-20.,5.]);shifted=states.copy();shifted[:,:,:3]+=offset
   with NBodyContext([0,2],shifted,[2.],[0.]) as shifted_context:
    yy=y.copy();yy[:3]+=offset;other=shifted_context.propagate(0,2,yy)['state'];other[:3]-=offset
    np.testing.assert_allclose(other,actual,atol=2e-10,rtol=0)

 def test_exact_nonbinary_endpoints_and_exact_step_budget(self):
  with circular_context() as c:
   for start,end in [(-4.731366569,1e-3),(1e-3,-4.731366569),(0.,.0010000000002)]:
    result=c.propagate(start,end,[1,0,0,0,1,0]);self.assertEqual(result['time'],end)
   r=c.propagate(0,.000001,[1,0,0,0,1,0],Options(initial_step=.000001,maximum_steps=1))
   self.assertEqual(r['status'],0)

 def test_failure_nan_and_context_lifecycle(self):
  with circular_context() as c:
   for start,end,status in [(0,20,2),(float('nan'),1,2)]:
    result=c.propagate(start,end,[1,0,0,0,1,0],check=False)
    self.assertEqual(result['status'],status);self.assertTrue(np.isnan(result['state']).all())
   result=c.propagate(0,6,[1,0,0,0,1,0],Options(maximum_steps=1),check=False)
   self.assertEqual(result['status'],5)
   result=c.propagate(0,1,[1,0,0,0,1,0],Options(rtol=2e-15,atol_position=1e-20,atol_velocity=1e-20,atol_stm=1e-20,minimum_step=.1),check=False)
   self.assertEqual(result['status'],4)
   with self.assertRaises(ValueError):c.force(-11,[1,0,0])
  c.close()
  with self.assertRaises(RuntimeError):c.propagate(0,1,[1,0,0,0,1,0])
  with NBodyContext([0,1],np.zeros((2,1,6)),[1],[.5]) as guarded:
   result=guarded.propagate(0,0,[.1,0,0,0,1,0],check=False)
   self.assertEqual(result['status'],3)

 def test_centered_large_epoch_translation_preserves_local_state(self):
  shift=np.array([3e11,-8e10,2e11]);clock=1e9
  states=np.zeros((2,1,6));states[:,:,:3]=shift
  poles=np.array([[0,0,1.],[0,0,1.]])
  with NBodyContext([clock-10,clock+10],states,[1.],[0.],j2=(0,.002,.5,poles)) as far:
   with NBodyContext([-10,10],np.zeros((2,1,6)),[1.],[0.],j2=(0,.002,.5,poles)) as local:
    initial=np.array([1.,.01,0.,.1,1.,.1]);a=clock+1.1234;b=clock+1.6234
    actual=far.propagate(a,b,initial,origin=shift)
    expected=local.propagate(0,b-a,initial)
    np.testing.assert_allclose(actual['state'],expected['state'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(actual['stm'],expected['stm'],rtol=0,atol=1e-13)
    self.assertEqual(actual['time'],b)

 def test_reject_invalid_arrays_and_options(self):
  for times in [[1,0],[0,0],[0,float('nan')]]:
   with self.assertRaises(ValueError):NBodyContext(times,np.zeros((2,1,6)),[1],[0])
  for kwargs in [dict(rtol=True),dict(maximum_steps=True),dict(atol_velocity=0),dict(rtol=1e-17),dict(minimum_step=1000)]:
   with self.assertRaises(ValueError):Options(**kwargs).validate()


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_SPICE')=='1','explicit pinned-kernel opt-in')
class RealEncounterTests(unittest.TestCase):
 def test_ballistic_and_powered_correction_and_complete_jacobian(self):
  from src.research.mission_config import load_mission
  from src.research.nbody_ephemeris import NBodyEphemeris
  from src.research.nbody_flyby import lift_candidate,shoot,shooting_problem
  from src.research.nbody_validation import validate_local
  cases=json.loads((ROOT/'validation/mission-cases.json').read_text())['cases']
  from src.research.mission_dynamics import MissionEvaluator
  for case in cases:
   cfg=load_mission(ROOT/'examples/missions/emc-ten-year.json');cfg.backend='cpu';cfg.model=case['model']
   with tempfile.TemporaryDirectory() as d:
    with MissionEvaluator(cfg,d) as evaluator:candidate=evaluator.candidate(case['decisions_days'])
    seed=lift_candidate(candidate,cfg)
    with NBodyEphemeris(cfg.kernels,seed['epoch_et'],seed['entry_time']-3000,seed['exit_time']+3000,60.) as e:
     problem=shooting_problem(e,seed);f=problem['evaluate'];q=np.zeros(10);J=f(q)['jacobian']
     for j in range(10):
      delta=np.eye(10)[j]*1e-6
      fd=(f(q+delta)['residual']-f(q-delta)['residual'])/2e-6
      np.testing.assert_allclose(fd,J[:,j],rtol=2e-5,atol=2e-7)
     result=shoot(e,seed);self.assertEqual(result['status'],'locally_corrected')
     accepted=validate_local(e,seed,result)
     self.assertEqual(accepted['status'],'passed_local_newtonian_nbody',[x for x in accepted['checks'] if not x['passed']])
     self.assertFalse(result['full_mission_reattached'])
