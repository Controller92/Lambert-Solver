"""Native M>0 contracts and actual CUDA mixed-batch execution."""
from dataclasses import replace
import math
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.lambert_geometry import LambertInput
from src.lambert_multirev import MinimumOptions, find_minimum_multi_rev
from src.lambert_native import NativeEngine, pack_problems, pack_multi_rev_pairs, INPUT_DTYPE
from src.lambert_root import RootOptions
from src.lambert_multirev_solver import solve_multi_rev_pair


class MultiTests(unittest.TestCase):
    backend='cpu'

    def setUp(self):
        self.engine=NativeEngine(self.backend)
        self.p=LambertInput((1.,0.,0.),(0.,1.,0.),2.5*math.pi,1.,revolutions=1)

    def run_rows(self,rows,options=RootOptions(),minimum_options=MinimumOptions()):
        if self.backend=='cpu': return self.engine.cpu(rows,options,minimum_options=minimum_options)
        with self.engine.cuda_batch(len(rows)) as batch:
            batch.upload(rows);batch.run(options,minimum_options=minimum_options)
            return batch.download()

    def test_minimum_rounded_zero_probe_regressions(self):
        cases=json.loads((Path(__file__).resolve().parents[2]/'validation/data/multirev-roundoff.json').read_text())
        for case in cases:
            problem=LambertInput(tuple(case['r1']),tuple(case['r2']),case['tof'],case['mu'],case['long_way'],case['revolutions'])
            result=self.run_rows(pack_multi_rev_pairs([problem]))
            self.assertEqual(result['status'].tolist(),[0,0])
            self.assertTrue(np.all(result['minimum']['first_lower']<=0))
            self.assertTrue(np.all(result['minimum']['first_upper']>=0))
            self.assertTrue(np.all(result['minimum']['upper']-result['minimum']['lower']<=
                                   1e-14+1e-14*np.abs(result['minimum']['x'])))
            for root in result['root']:
                self.assertLessEqual(abs(root['relative_residual']),5e-13)

    def test_branches_counts_scales_and_reversal(self):
        for m in (1,2,5,20,2**31-1):
            for long_way in (False,True):
                angle=1.5*math.pi if long_way else .5*math.pi
                p=replace(self.p,revolutions=m,long_way=long_way,tof_s=m*2*math.pi+angle)
                rows=self.run_rows(pack_multi_rev_pairs([p]))
                for branch,r in enumerate(rows):
                    self.assertEqual(r['status'],0)
                    self.assertEqual(r['branch'],branch)
                    self.assertEqual(r['revolutions'],m)
                    root,minimum=r['root'],r['minimum']
                    self.assertEqual(minimum['status'],0)
                    self.assertEqual(root['iterations'],root['householder']+root['bisection']+root['verification'])
                    self.assertEqual(minimum['iterations'],minimum['halley']+minimum['bisection'])
                    self.assertEqual(root['evaluations'],root['iterations']+2)
                    self.assertEqual(minimum['evaluations'],minimum['iterations']+3+2*minimum['zero_derivative']+minimum['expansions'])
                    self.assertLessEqual(abs(root['relative_residual']),5e-13)
                    self.assertEqual(root['x']<minimum['lower'],branch==0)
                    self.assertEqual(root['x']>minimum['upper'],branch==1)
                if m<100:
                    sign=-1 if long_way else 1
                    self.assertLess(min(np.linalg.norm(r['v1']-(0.,sign,0.)) for r in rows),2e-13)
        for p in (self.p,replace(self.p,r1_m=(7e6,0.,0.),r2_m=(0.,7e6,0.),
                                mu_m3_s2=3.986004418e14,tof_s=self.p.tof_s*math.sqrt(7e6**3/3.986004418e14))):
            a=self.run_rows(pack_multi_rev_pairs([p]))
            b=self.run_rows(pack_multi_rev_pairs([replace(p,r1_m=p.r2_m,r2_m=p.r1_m)]))
            np.testing.assert_allclose(a['v1'],-b['v2'],atol=2e-11,rtol=2e-13)

    def test_failures_minimum_budget_and_no_fake_velocities(self):
        minimum=find_minimum_multi_rev(.4142135623730951,1,q=.8284271247461902)
        time_scale=math.sqrt(((2+math.sqrt(2))/2)**3/2)
        for gap,status in ((-1e-8,8),(0.,9)):
            p=replace(self.p,tof_s=minimum.time_nondim*(1+gap)*time_scale)
            output=self.run_rows(pack_multi_rev_pairs([p]))
            self.assertEqual(output['root']['status'].tolist(),[status]*2)
            self.assertTrue(np.isnan(output['root']['x']).all())
            self.assertTrue(np.isnan(output['v1']).all())
        output=self.run_rows(pack_multi_rev_pairs([self.p]),minimum_options=MinimumOptions(max_iterations=0))
        self.assertTrue(np.all(output['root']['status']==7))
        self.assertTrue(np.all(output['minimum']['status']==4))
        self.assertTrue(np.isnan(output['minimum']['x']).all())
        for options in (RootOptions(max_iterations=0),RootOptions(tof_rtol=-1.)):
            output=self.run_rows(pack_multi_rev_pairs([self.p]),options)
            self.assertTrue(np.isnan(output['v1']).all())
            self.assertTrue(np.isnan(output['root']['x']).all())

    def test_mixed_tail_and_context_reuse(self):
        problems=[replace(self.p,revolutions=0,tof_s=math.pi/2),self.p,
                  replace(self.p,mu_m3_s2=0.),replace(self.p,revolutions=-1)]
        rows=pack_problems(problems*65, ['right','left','left','right']*65)[:257]
        output=self.run_rows(rows)
        self.assertEqual(len(output),257)
        for i,r in enumerate(output):
            self.assertEqual(r['status']==0,i%4<2)
            if i%4>=2:self.assertTrue(np.isnan(r['v1']).all())
        self.assertEqual(len(self.run_rows(rows[:0])),0)
        if self.backend=='cuda':
            with self.engine.cuda_batch(257) as b:
                for subset in (rows,rows[1:2],rows[:0],rows[:1],rows[:2]):
                    b.upload(subset);b.run();actual=b.download()
                    expected=NativeEngine().cpu(subset)
                    np.testing.assert_array_equal(actual['status'],expected['status'])
                    np.testing.assert_allclose(actual['v1'],expected['v1'],atol=1e-12,rtol=1e-12,equal_nan=True)

    def test_invalid_branches_and_options(self):
        rows=pack_problems([self.p]*3); rows['branch']=(2,-1,0); rows['revolutions'][2]=-1
        result=self.run_rows(rows)
        self.assertTrue(np.all(result['status']==1))
        self.assertTrue(np.all(result['root']['status']==-1))
        self.assertTrue(np.isnan(result['v1']).all())
        for mo in (MinimumOptions(x_rtol=-1.), MinimumOptions(x_atol=math.nan)):
            result=self.run_rows(pack_problems([self.p]),minimum_options=mo)
            self.assertEqual(result['root']['status'][0],7)
            self.assertEqual(result['minimum']['status'][0],1)
        with self.assertRaises(ValueError): pack_problems([self.p],branches='unknown')
        with self.assertRaises(ValueError): pack_problems([self.p],branches=[])
        with self.assertRaises(ValueError): pack_multi_rev_pairs([replace(self.p,revolutions=0)])
        with self.assertRaises(ValueError): self.engine.cpu(pack_problems([self.p]),minimum_options=MinimumOptions(max_iterations=2**40))

    def test_tiny_arc_and_partial_branch_budget(self):
        for angle in (2e-14,math.pi-1e-8):
            p=replace(self.p,r2_m=(math.cos(angle),math.sin(angle),0.),tof_s=10.)
            actual=self.run_rows(pack_multi_rev_pairs([p]))
            reference=solve_multi_rev_pair(p)
            self.assertTrue(np.all(actual['status']==0))
            for a,b in zip(actual,reference):np.testing.assert_allclose(a['v1'],b.v1_m_s,atol=1e-12,rtol=2e-12)
        # One branch success must survive a sibling that exhausts its budget.
        # Find a feasible case with sufficiently different convergence lengths.
        output=self.run_rows(pack_multi_rev_pairs([replace(self.p,tof_s=12.)]),RootOptions(max_iterations=3))
        self.assertEqual(output['root']['status'].tolist(),[3,0])
        for r in output:
            if r['root']['status']!=0:self.assertTrue(np.isnan(r['v1']).all())


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit GPU opt-in required')
class MultiCudaTests(MultiTests):
    backend='cuda'


if __name__=='__main__': unittest.main()
