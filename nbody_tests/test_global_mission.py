"""Opt-in integration coverage for shared finite mission endpoint derivatives."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.research.nbody_mission import chart,tangent_velocity,seed_mission,mission_knots,GlobalProblem

ROOT=Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
 def test_python_hermite_derivatives_ignore_large_constant_offset(self):
  from src.research.nbody_ephemeris import NBodyEphemeris
  # A small exactly representable linear body motion on a large translation.
  e=NBodyEphemeris.__new__(NBodyEphemeris);e.lower=0.;e.upper=120.;e.times=np.array([0.,120.]);e.bodies=[('499',)]
  origin=np.array([4e11,-2e11,1e11]);velocity=np.array([12500.,-25000.,6250.])
  e.states=np.array([[np.r_[origin,velocity]],[np.r_[origin+120*velocity,velocity]]])
  for t in [1.2345,37.125,119.999]:
   state,acc=e.interpolated_state('499',t)
   np.testing.assert_allclose(state[3:],velocity,rtol=0,atol=1e-10)
   np.testing.assert_allclose(acc,0,rtol=0,atol=1e-10)
   local,_=e.interpolated_state('499',t,origin)
   np.testing.assert_allclose(local[:3],t*velocity,rtol=0,atol=1e-8)

 def test_parameterized_surface_and_tangent_velocity(self):
  r=np.array([1.,2.,3.]);v=np.array([2.,-.4,.1])
  for q in [[0.,0.],[.2,-.3],[-.5,.5]]:
   direction=chart(r,q);self.assertAlmostEqual(np.linalg.norm(direction),1.,places=14)
   velocity=tangent_velocity(v,10*direction,[.001,.2])
   self.assertLess(abs(np.dot(velocity,direction)),1e-14)
   self.assertAlmostEqual(np.linalg.norm(velocity),np.linalg.norm(v)+1.,places=13)


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_SPICE')=='1','explicit pinned-kernel opt-in')
class GlobalJacobianTests(unittest.TestCase):
 def test_shared_nodes_dates_burns_and_variational_jacobian(self):
  import json
  from src.research.mission_config import load_mission
  from src.research.mission_dynamics import MissionEvaluator
  from src.research.nbody_ephemeris import NBodyEphemeris
  case=json.loads((ROOT/'validation/mission-cases.json').read_text())['cases'][1]
  config=load_mission(ROOT/'examples/missions/emc-ten-year.json');config.backend='cpu';config.model='powered'
  with tempfile.TemporaryDirectory() as d:
   with MissionEvaluator(config,d) as evaluator:candidate=evaluator.candidate(case['decisions_days'])
   seed=seed_mission(candidate,config);knots=mission_knots(seed['times'])
   with NBodyEphemeris(config.kernels,seed['epoch_et'],knots[0],knots[-1],times=knots,mars_j2=True,surface_guard_slack=1.,cache_dir=d) as e:
    problem=GlobalProblem(e,seed,config);q=np.zeros(problem.size);base=problem.evaluate(q);J=base['jacobian']
    # A derivative from every endpoint family plus powered impulse and the
    # special relative-clock Ceres rotation. Propagation is differenced here,
    # unlike the endpoint-only finite differences used by the implementation.
    columns=[0,2,6,9,10,12,26,28]
    for j in columns:
     h=1e-6;delta=np.eye(problem.size)[j]*h
     fd=(problem.evaluate(q+delta)['residual']-problem.evaluate(q-delta)['residual'])/(2*h)
     error=np.linalg.norm(fd-J[:,j])/max(1.,np.linalg.norm(J[:,j]))
     self.assertLess(error,3e-5,(j,error))
    node=problem.nodes(q);self.assertLess(max(abs(node['radial'])),1e-14)
    # Smooth relative spin must respond even below absolute-ET clock resolution.
    dt=np.eye(problem.size)[28]*1e-11
    relative_change=problem.nodes(q+dt)['relative'][2,:3]-problem.nodes(q-dt)['relative'][2,:3]
    self.assertGreater(np.linalg.norm(relative_change),1e-8)
    for n in [1,4]:
     dv=node['after'][n,3:]-node['before'][n,3:]
     self.assertLess(np.linalg.norm(np.cross(dv,node['relative'][n,3:]))/np.linalg.norm(node['relative'][n,3:]),1e-10)
