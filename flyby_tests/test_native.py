from dataclasses import replace
import math
import os
from pathlib import Path
import unittest
import numpy as np
from src.flyby import Encounter,Mode,solve
from src.flyby_native import FlybyEngine,INPUT_DTYPE,pack_encounters
from test_physics import BASE


def cases():
    rng=np.random.default_rng(18548)
    problems=[]
    for i in range(500):
        vin=rng.normal(size=3);vin=vin/np.linalg.norm(vin)*10**rng.uniform(2,5)
        p=replace(BASE,vin=tuple(vin),mu=10**rng.uniform(12,17),rp=10**rng.uniform(5,10),
                  rp_min=1e3,rp_max=1e12,azimuth=rng.uniform(-math.pi,math.pi))
        r=solve(p);problems.append(p)
        if r.status==0:
            problems.append(replace(p,mode=Mode.MATCH,vout=r.vout))
            problems.append(replace(p,mode=Mode.POWERED,vout=tuple(np.asarray(r.vout)*rng.uniform(.9,1.1))))
    problems.extend([replace(BASE,mu=-1),replace(BASE,vin=(0.,0.,0.)),replace(BASE,reference=BASE.vin),
                     replace(BASE,rp=1),replace(BASE,vin=(math.nan,0.,0.)),
                     replace(BASE,mode=Mode.MATCH,vout=BASE.vin),
                     replace(BASE,mode=Mode.MATCH,vout=(-10000.,0.,0.))])
    return problems


class Native(unittest.TestCase):
    backend='cpu'

    def compute(self,rows): return FlybyEngine().cpu(rows)

    def test_reference_physics_and_status_parity(self):
        problems=cases(); result=self.compute(pack_encounters(problems))
        for p,r in zip(problems,result):
            ref=solve(p)
            self.assertEqual(r['status'],ref.status)
            self.assertEqual(r['precision_limited'],ref.precision_limited)
            if ref.status==0:
                np.testing.assert_allclose(r['vout'],ref.vout,rtol=2e-11,atol=1e-8)
                np.testing.assert_allclose(r['rp'],ref.rp,rtol=2e-10)
                np.testing.assert_allclose(r['delta_v'],ref.delta_v,rtol=1e-11,atol=1e-10)
            else: self.assertTrue(np.isnan(r['vout']).all())

    def test_empty_tail_and_invalid_layout(self):
        rows=pack_encounters([BASE]*257)
        self.assertEqual(len(self.compute(rows[::2])),129)
        self.assertEqual(len(self.compute(rows[:0])),0)
        with self.assertRaises(TypeError):self.compute(np.ones(5))
        with self.assertRaises(ValueError):pack_encounters([replace(BASE,vin=(1.,2.))])
        with self.assertRaises(TypeError):pack_encounters([replace(BASE,mu=True)])


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA')=='1','explicit GPU opt-in')
class CUDA(Native):
    backend='cuda'
    def compute(self,rows):
        with FlybyEngine('cuda').cuda_batch(len(rows)) as batch:
            batch.upload(rows);batch.run();return batch.download()

    def test_context_lifecycle_and_reupload(self):
        with FlybyEngine('cuda').cuda_batch(3) as b:
            with self.assertRaises(RuntimeError):b.download()
            with self.assertRaises(ValueError):b.upload(pack_encounters([BASE]*4))
            b.upload(pack_encounters([BASE]*3));b.run(2);self.assertTrue((b.download()['status']==0).all())
            b.upload(pack_encounters([]))
            with self.assertRaises(RuntimeError):b.download()
            self.assertEqual(b.run(),0);self.assertEqual(len(b.download()),0)
            with self.assertRaises(ValueError):b.run(0)
        with self.assertRaises(RuntimeError):b.run()
        b.close()
