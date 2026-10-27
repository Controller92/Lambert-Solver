"""Terminal sensitivities: contracts, invariants, branches and Taylor behavior."""
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.lambert_geometry import LambertInput, prepare_geometry
from src.lambert_native import NativeEngine, pack_problems, JACOBIAN_DTYPE
from src.lambert_root import RootOptions


class JacobianTests(unittest.TestCase):
    backend = 'cpu'

    def setUp(self):
        self.engine = NativeEngine(self.backend)
        self.p = LambertInput((1.,.2,-.1),(-.3,1.1,.4),3.,1.)

    def evaluate(self,rows,options=RootOptions()):
        if self.backend == 'cpu': return self.engine.cpu_with_jacobian(rows,options)
        with self.engine.cuda_batch(len(rows)) as batch:
            batch.upload(rows); batch.run(options,jacobian=True)
            return batch.download(),batch.download_jacobian()

    def test_fixed_branch_finite_difference_and_taylor(self):
        for m in (0,1,5):
            for long_way in (False,True):
                for branch in ('left','right') if m else ('left',):
                    rows=pack_problems([replace(self.p,tof_s=3.+7*m,revolutions=m,long_way=long_way)],branch)
                    r,j=self.evaluate(rows)
                    self.assertEqual(r['status'][0],0); self.assertEqual(j['status'][0],0)
                    J=j['values'][0]
                    for k in range(7):
                        errors=[]
                        for h in (1e-3,1e-4,1e-5):
                            a,b=rows.copy(),rows.copy()
                            if k<6:
                                field='r1' if k<3 else 'r2'
                                a[field][0,k%3]+=h; b[field][0,k%3]-=h
                            else: a['tof']+=h; b['tof']-=h
                            ra,rb=NativeEngine().cpu(a)[0],NativeEngine().cpu(b)[0]
                            self.assertEqual(ra['status'],0);self.assertEqual(rb['status'],0)
                            fd=(np.r_[ra['v1'],ra['v2']]-np.r_[rb['v1'],rb['v2']])/(2*h)
                            errors.append(np.max(abs(fd-J[:,k])))
                        self.assertLess(min(errors),3e-8)
                    direction=np.array((.3,-.2,.15,-.1,.4,-.2,.25))
                    residuals=[]
                    for h in (1e-3,5e-4,2.5e-4):
                        a=rows.copy();a['r1']+=h*direction[:3];a['r2']+=h*direction[3:6];a['tof']+=h*direction[6]
                        ra=NativeEngine().cpu(a)[0]
                        residuals.append(np.linalg.norm(np.r_[ra['v1']-r['v1'][0],ra['v2']-r['v2'][0]]-h*J@direction))
                    self.assertTrue(3.8<residuals[0]/residuals[1]<4.2,residuals)
                    self.assertTrue(3.8<residuals[1]/residuals[2]<4.2,residuals)

    def test_rotation_units_reversal_and_kepler_scaling(self):
        Q,_=np.linalg.qr(np.array(((.3,.8,-.4),(-.6,.2,.5),(.7,-.1,.4))))
        if np.linalg.det(Q)<0: Q[:,0]*=-1
        for m,branch in ((0,'left'),(2,'left'),(2,'right')):
            p=replace(self.p,tof_s=3.+m*7,revolutions=m)
            r,j=self.evaluate(pack_problems([p],branch));J=j['values'][0]
            V=np.r_[r['v1'][0],r['v2'][0]]
            # r -> k*r, t -> k^(3/2)*t at fixed mu: v -> k^(-1/2)*v.
            np.testing.assert_allclose(J@np.r_[p.r1_m,p.r2_m,1.5*p.tof_s],-.5*V,atol=3e-13,rtol=3e-12)
            rotated=replace(p,r1_m=tuple(Q@p.r1_m),r2_m=tuple(Q@p.r2_m))
            _,jr=self.evaluate(pack_problems([rotated],branch))
            output_rotation=np.kron(np.eye(2),Q)
            input_rotation=np.eye(7);input_rotation[:6,:6]=output_rotation
            np.testing.assert_allclose(jr['values'][0],output_rotation@J@input_rotation.T,atol=2e-13,rtol=2e-12)
            radius,mu=7e6,3.986004418e14
            time=math.sqrt(radius**3/mu);speed=radius/time
            scaled=replace(p,r1_m=tuple(radius*np.array(p.r1_m)),r2_m=tuple(radius*np.array(p.r2_m)),tof_s=p.tof_s*time,mu_m3_s2=mu)
            _,js=self.evaluate(pack_problems([scaled],branch))
            np.testing.assert_allclose(js['values'][0],J*speed/np.array([radius]*6+[time]),rtol=3e-12,atol=2e-14)
            reverse=replace(p,r1_m=p.r2_m,r2_m=p.r1_m)
            _,jb=self.evaluate(pack_problems([reverse],branch))
            np.testing.assert_allclose(jb['values'][0],-J[[3,4,5,0,1,2]][:,[3,4,5,0,1,2,6]],atol=3e-13,rtol=3e-12)

    def test_fold_failures_empty_and_noncontiguous(self):
        rows=pack_problems([self.p,replace(self.p,mu_m3_s2=0.),replace(self.p,r2_m=(-1.,-.2,.1))])
        r,j=self.evaluate(rows)
        self.assertEqual(j.dtype,JACOBIAN_DTYPE)
        self.assertEqual(j['status'].tolist(),[0,1,1]);self.assertTrue(np.isnan(j['values'][1:]).all())
        self.assertEqual(len(self.evaluate(rows[:0])[1]),0)
        self.assertEqual(len(self.evaluate(np.tile(rows,3)[::2])[1]),5)
        _,j=self.evaluate(rows[:1],RootOptions(max_iterations=0))
        self.assertEqual(j['status'][0],1);self.assertTrue(np.isnan(j['values']).all())
        p=replace(self.p,revolutions=1,tof_s=10.)
        nominal=NativeEngine().cpu(pack_problems([p]))[0]
        g=prepare_geometry(replace(p,revolutions=0)).geometry
        for gap,expected in ((1e-8,0),(1e-12,2),(0.,1),(-1e-8,1)):
            p1=replace(p,tof_s=nominal['minimum']['time']*g.time_scale_s*(1+gap))
            for branch in ('left','right'):
                r,j=self.evaluate(pack_problems([p1],branch))
                self.assertEqual(j['status'][0],expected,(gap,branch,r,j))
                if expected: self.assertTrue(np.isnan(j['values']).all())
                if expected==2: self.assertEqual(r['status'][0],0)

    def test_tiny_arc_arbitrary_plane_regression(self):
        u=np.array((.3,-.4,.5));u/=np.linalg.norm(u)
        w=np.cross(u,(.7,.1,-.2));w/=np.linalg.norm(w)
        for angle in (1e-8,1e-12,2e-14):
            p=LambertInput(tuple(u),tuple(math.cos(angle)*u+math.sin(angle)*w),angle,1.)
            r,j=self.evaluate(pack_problems([p]))
            self.assertEqual(r['status'][0],0);self.assertEqual(j['status'][0],0)
            J=j['values'][0]
            # Independent short-time variational limit. Off-diagonal position
            # terms tend to zero, rather than eps/angle^2 cancellation noise.
            limit=np.column_stack((-np.eye(3)/angle,np.eye(3)/angle))
            np.testing.assert_allclose(J[:3,:6],limit,rtol=2e-14,atol=2e-8)
            np.testing.assert_allclose(J[3:,:6],limit,rtol=2e-14,atol=2e-8)


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit CUDA opt-in required')
class CudaJacobianTests(JacobianTests):
    backend='cuda'

    def test_optional_buffer_lifecycle_tail_and_cpu_agreement(self):
        problems=[self.p,replace(self.p,revolutions=2,tof_s=18.),replace(self.p,mu_m3_s2=0.)]*86
        rows=pack_problems(problems,['left','right','left']*86)[:257]
        with self.engine.cuda_batch(257) as b:
            with self.assertRaises(RuntimeError): b.download_jacobian()
            for subset in (rows,rows[:1],rows[:0],rows[1:2],rows):
                b.upload(subset)
                with self.assertRaises(RuntimeError): b.download_jacobian()
                b.run(jacobian=True,repetitions=2)
                r,j=b.download(),b.download_jacobian()
                cr,cj=NativeEngine().cpu_with_jacobian(subset)
                np.testing.assert_array_equal(j['status'],cj['status'])
                np.testing.assert_allclose(j['values'],cj['values'],atol=2e-12,rtol=2e-12,equal_nan=True)
                np.testing.assert_array_equal(r['status'],cr['status'])
                b.run()
                with self.assertRaises(RuntimeError): b.download_jacobian()
            with self.assertRaises(TypeError): b.run(jacobian=1)
        with self.assertRaises(RuntimeError): b.download_jacobian()


if __name__=='__main__': unittest.main()
