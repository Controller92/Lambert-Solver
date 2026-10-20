"""Explicit native integration tests. Set LAMBERT_TEST_CUDA=1 for GPU tests."""
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine, pack_problems, INPUT_DTYPE, RESULT_DTYPE
from src.lambert_root import RootOptions


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.engine = NativeEngine()
        self.problem = LambertInput((1., 0., 0.), (0., 1., 0.), math.pi/2, 1.)

    def check_result(self, r):
        self.assertEqual(r['status'], 0)
        np.testing.assert_allclose(r['v1'], (0., 1., 0.), atol=3e-14, rtol=0)
        np.testing.assert_allclose(r['v2'], (-1., 0., 0.), atol=3e-14, rtol=0)
        root = r['root']
        self.assertGreater(root['iterations'], 0)
        self.assertEqual(root['iterations'], root['householder']+root['bisection']+root['verification'])
        self.assertLessEqual(abs(root['relative_residual']), 5e-13)

    def test_cpu_circular_and_noncontiguous_batches(self):
        rows = pack_problems([self.problem]*7)
        results = self.engine.cpu(rows[::2])
        self.assertEqual(len(results), 4)
        for r in results:
            self.check_result(r)
        self.assertEqual(len(self.engine.cpu(rows[:0])), 0)
        output = np.empty(len(rows), dtype=RESULT_DTYPE)
        self.assertIs(self.engine.cpu(rows, out=output), output)
        output.setflags(write=False)
        with self.assertRaises(ValueError):
            self.engine.cpu(rows, out=output)
        # A contiguous ndarray may still have a misaligned buffer offset.
        unaligned = np.ndarray(len(rows), dtype=INPUT_DTYPE,
                               buffer=bytearray(len(rows)*INPUT_DTYPE.itemsize+1), offset=1)
        unaligned[:] = rows
        for result in self.engine.cpu(unaligned):
            self.check_result(result)

    def test_explicit_numerical_and_input_failures(self):
        inputs = [replace(self.problem, mu_m3_s2=0.), replace(self.problem, revolutions=-1),
                  replace(self.problem, r2_m=(-1., 0., 0.)), replace(self.problem, tof_s=math.nan)]
        rows = pack_problems(inputs)
        output = self.engine.cpu(rows)
        self.assertEqual(output['geometry_status'].tolist(), [1, 1, 3, 1])
        self.assertTrue(np.all(output['root']['status'] == -1))
        self.assertTrue(np.isnan(output['v1']).all())
        self.assertTrue(np.isnan(output['v2']).all())
        output = self.engine.cpu(pack_problems([self.problem]), RootOptions(max_iterations=0))
        self.assertEqual(output['root']['status'][0], 3)
        self.assertTrue(np.isnan(output['root']['x']).all())
        self.assertTrue(np.isnan(output['v1']).all())
        output = self.engine.cpu(pack_problems([self.problem]), RootOptions(tof_rtol=-1.))
        self.assertEqual(output['root']['status'][0], 1)

    def test_bad_python_layouts_never_reach_native_code(self):
        with self.assertRaises(TypeError):
            self.engine.cpu(np.zeros((2, 8)))
        with self.assertRaises(ValueError):
            pack_problems([replace(self.problem, r1_m=(1., 2.))])
        with self.assertRaises(TypeError):
            pack_problems([replace(self.problem, long_way=1)])
        with self.assertRaises(TypeError):
            pack_problems([replace(self.problem, r1_m=(1+1j, 0., 0.))])
        with self.assertRaises(ValueError):
            pack_problems([replace(self.problem, revolutions=2**40)])
        with self.assertRaises(ValueError):
            self.engine.probe('roots', np.zeros((3, 2)))
        with self.assertRaises(RuntimeError):
            self.engine.cuda_batch(1)

    def test_root_precision_tristate_and_budget(self):
        r = self.engine.probe('roots', [[1e-150, 0., math.nan]])[0]
        self.assertEqual(r['status'], 0)
        self.assertEqual(r['precision_limited'], -1)
        self.assertTrue(math.isnan(r['estimated_x_error']))
        self.assertGreater(r['derivative_failures'], 0)
        r = self.engine.probe('roots', [[1e-310, 0., math.nan]], RootOptions(max_bracket_expansions=0))[0]
        self.assertEqual(r['status'], 2)
        self.assertTrue(math.isnan(r['x']))


@unittest.skipUnless(os.environ.get('LAMBERT_TEST_CUDA') == '1', 'explicit GPU test opt-in required')
class CudaTests(NativeTests):
    def test_nearly_antiparallel_device_roundoff(self):
        gpu = NativeEngine('cuda')
        u = np.array((.3, -.4, .5))
        u /= np.linalg.norm(u)
        v = np.cross(u, (.7, .1, -.2))
        v /= np.linalg.norm(v)
        angle, radius, mu = math.pi-1e-8, 7e6, 3.986004418e14
        r2 = radius*1.0005*(math.cos(angle)*u+math.sin(angle)*v)
        p = LambertInput(tuple(radius*u), tuple(r2), angle*radius/math.sqrt(mu/radius), mu)
        rows = pack_problems([p])
        with gpu.cuda_batch(1) as batch:
            batch.upload(rows)
            batch.run()
            result = batch.download()
        self.assertEqual(result['status'][0], 0)
        self.assertLessEqual(result['q'][0], 1.)
        np.testing.assert_allclose(result['v1'], self.engine.cpu(rows)['v1'], atol=1e-10, rtol=2e-12)

    def test_reuse_empty_tail_statuses_and_owned_lifetime(self):
        gpu = NativeEngine('cuda')
        with gpu.cuda_batch(257) as batch:
            with self.assertRaises(RuntimeError):
                batch.run()
            with self.assertRaises(RuntimeError):
                batch.download()
            rows = pack_problems([self.problem]*257)
            batch.upload(rows)
            with self.assertRaises(RuntimeError):
                batch.download()
            self.assertGreater(batch.run(repetitions=2), 0.)
            for r in batch.download():
                self.check_result(r)
            failed_inputs = [replace(self.problem, mu_m3_s2=-1.),
                             replace(self.problem, revolutions=-1),
                             replace(self.problem, r2_m=(-1., 0., 0.)),
                             replace(self.problem, tof_s=math.nan)]
            batch.upload(pack_problems(failed_inputs))
            batch.run()
            result = batch.download()
            self.assertEqual(result['geometry_status'].tolist(), [1, 1, 3, 1])
            self.assertTrue(np.all(result['root']['status'] == -1))
            self.assertTrue(np.isnan(result['v1']).all())
            self.assertTrue(np.isnan(result['v2']).all())
            batch.upload(rows[:0])
            self.assertEqual(batch.run(), 0.)
            self.assertEqual(len(batch.download()), 0)
            with self.assertRaises(ValueError):
                batch.upload(pack_problems([self.problem]*258))
        batch.close()
        with self.assertRaises(RuntimeError):
            batch.upload(rows)

    def test_independent_contexts_and_zero_iteration_status(self):
        gpu = NativeEngine('cuda')
        with gpu.cuda_batch(1) as a, gpu.cuda_batch(1, threads=64) as b:
            rows = pack_problems([self.problem])
            a.upload(rows)
            b.upload(rows)
            a.run(RootOptions(max_iterations=0))
            b.run()
            failed = a.download()
            self.assertEqual(failed['root']['status'][0], 3)
            self.assertTrue(np.isnan(failed['v1']).all())
            self.assertTrue(np.isnan(failed['v2']).all())
            self.check_result(b.download()[0])
            a.run()
            self.check_result(a.download()[0])


if __name__ == '__main__':
    unittest.main()
