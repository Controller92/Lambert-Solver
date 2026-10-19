"""Analytical full-transfer and explicit failure tests, standard library only."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from src.lambert_geometry import GeometryStatus, LambertInput
from src.lambert_root import RootOptions, RootStatus
from src.lambert_solver import LambertStatus, solve_zero_rev


class SolverTests(unittest.TestCase):
    def assert_vector(self, actual, expected, scale=1., tolerance=3e-13):
        self.assertLessEqual(math.dist(actual, expected)/scale, tolerance)

    def solved(self, problem, options=RootOptions()):
        result = solve_zero_rev(problem, options)
        self.assertEqual(result.status, LambertStatus.OK, result)
        self.assertEqual(result.root.status, RootStatus.OK)
        self.assertGreaterEqual(result.root.iterations, 1)
        self.assertLessEqual(abs(result.root.relative_residual), options.tof_rtol)
        return result

    def test_circular_arcs_scales_and_planes(self):
        maximum = 0.
        for radius, mu in ((1., 1.), (7e6, 3.986004418e14),
                           (1.495978707e11, 1.32712440018e20), (1e150, 1e300)):
            speed = math.sqrt(mu)/math.sqrt(radius)
            quarter = (math.pi/2)*radius/speed
            for long_way in (False, True):
                sense = -1 if long_way else 1
                for rotate in (lambda v: v, lambda v: (v[2], v[0], v[1])):
                    p = LambertInput(rotate((radius, 0., 0.)), rotate((0., radius, 0.)),
                                     (3 if long_way else 1)*quarter, mu, long_way)
                    result = self.solved(p)
                    v1, v2 = rotate((0., sense*speed, 0.)), rotate((-sense*speed, 0., 0.))
                    self.assert_vector(result.v1_m_s, v1, speed)
                    self.assert_vector(result.v2_m_s, v2, speed)
                    maximum = max(maximum, math.dist(result.v1_m_s, v1)/speed,
                                  math.dist(result.v2_m_s, v2)/speed)
        print(f'Analytical circular maximum velocity error / circular speed: {maximum:.3e}')

    def test_parabolic_barker_time(self):
        # p=2, nu=0 to pi/2, mu=1. Barker's equation gives t=4*sqrt(2)/3.
        p = LambertInput((1., 0., 0.), (0., 2., 0.), 4*math.sqrt(2)/3, 1.)
        result = self.solved(p)
        self.assertAlmostEqual(result.root.x, 1., delta=2e-14)
        self.assert_vector(result.v1_m_s, (0., math.sqrt(2), 0.))
        self.assert_vector(result.v2_m_s, (-1/math.sqrt(2), 1/math.sqrt(2), 0.))

    def test_elliptic_anomaly_long_arc(self):
        a, e, first, second = 2., .6, -.7, 4.
        def state(E):
            c, s = math.cos(E), math.sin(E)
            return ((a*(c-e), a*math.sqrt(1-e*e)*s, 0.),
                    (-s/(math.sqrt(a)*(1-e*c)),
                     math.sqrt(1-e*e)*c/(math.sqrt(a)*(1-e*c)), 0.))
        r1, v1 = state(first)
        r2, v2 = state(second)
        tof = a**1.5*(second-first-e*(math.sin(second)-math.sin(first)))
        result = self.solved(LambertInput(r1, r2, tof, 1., True))
        self.assert_vector(result.v1_m_s, v1)
        self.assert_vector(result.v2_m_s, v2)

    def test_hyperbolic_anomaly_time(self):
        a, e, first, second = 2., 1.5, -.4, .7
        def state(H):
            c, s = math.cosh(H), math.sinh(H)
            return ((a*(e-c), a*math.sqrt(e*e-1)*s, 0.),
                    (-s/(math.sqrt(a)*(e*c-1)),
                     math.sqrt(e*e-1)*c/(math.sqrt(a)*(e*c-1)), 0.))
        r1, v1 = state(first)
        r2, v2 = state(second)
        tof = a**1.5*(e*(math.sinh(second)-math.sinh(first))-second+first)
        result = self.solved(LambertInput(r1, r2, tof, 1.))
        self.assertGreater(result.root.x, 1.)
        self.assert_vector(result.v1_m_s, v1)
        self.assert_vector(result.v2_m_s, v2)

    def test_time_reversal(self):
        for long_way in (False, True):
            p = LambertInput((3., -2., .4), (-1., 4., 2.), 10., 2., long_way)
            forward = self.solved(p)
            reverse = self.solved(replace(p, r1_m=p.r2_m, r2_m=p.r1_m))
            self.assert_vector(reverse.v1_m_s, tuple(-v for v in forward.v2_m_s))
            self.assert_vector(reverse.v2_m_s, tuple(-v for v in forward.v1_m_s))

    def test_failures_never_have_velocities(self):
        p = LambertInput((1., 0., 0.), (0., 1., 0.), 1., 1.)
        for problem, expected in ((None, GeometryStatus.INVALID_INPUT),
                                  (replace(p, revolutions=1), GeometryStatus.UNSUPPORTED_REVOLUTIONS),
                                  (replace(p, r2_m=(-1., 0., 0.)), GeometryStatus.DEGENERATE_GEOMETRY)):
            result = solve_zero_rev(problem)
            self.assertEqual(result.status, LambertStatus.GEOMETRY_FAILURE)
            self.assertEqual(result.geometry.status, expected)
            self.assertIsNone(result.root)
            self.assertIsNone(result.v1_m_s)
            self.assertIsNone(result.v2_m_s)
        for options, expected in ((RootOptions(max_iterations=0), RootStatus.ITERATION_LIMIT),
                                  (None, RootStatus.INVALID_INPUT)):
            result = solve_zero_rev(p, options)
            self.assertEqual(result.status, LambertStatus.ROOT_FAILURE)
            self.assertEqual(result.root.status, expected)
            self.assertIsNone(result.v1_m_s)
            self.assertIsNone(result.v2_m_s)
        # Inject only the dimensional range failure to check no partial output.
        with patch('src.lambert_solver._velocity_scale', side_effect=OverflowError):
            result = solve_zero_rev(p)
        self.assertEqual(result.status, LambertStatus.NUMERICAL_RANGE)
        self.assertEqual(result.root.status, RootStatus.OK)
        self.assertIsNone(result.v1_m_s)
        self.assertIsNone(result.v2_m_s)

    def test_root_conditioning_diagnostics_are_preserved(self):
        p = LambertInput((1., 0., 0.), (0., 1., 0.), 1., 1.)
        root = self.solved(p).root
        for flag, estimate in ((True, 1e-10), (None, None), (False, 1e-16)):
            root.x_precision_limited, root.estimated_x_error = flag, estimate
            with patch('src.lambert_solver.solve_x_zero_rev', return_value=root):
                result = self.solved(p)
            self.assertIs(result.root, root)
            self.assertIs(result.root.x_precision_limited, flag)
            self.assertEqual(result.root.estimated_x_error, estimate)

    def test_nearly_coincident_geometry_has_separate_warning(self):
        regular = self.solved(LambertInput((1., 0., 0.), (0., 1., 0.), math.pi/2, 1.))
        self.assertFalse(regular.geometry_precision_warning)
        self.assertIsNone(solve_zero_rev(None).geometry_precision_warning)
        for angle in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            p = LambertInput((1., 0., 0.), (math.cos(angle), math.sin(angle), 0.), angle, 1.)
            result = self.solved(p)
            # Geometry conditioning is distinct from the scalar equation's
            # residual/step diagnostics. Both must remain available to callers.
            self.assertTrue(result.geometry_precision_warning)
            self.assertFalse(result.root.x_precision_limited)
            self.assertGreater(result.geometry_q_relative_mismatch, 1e-14)


if __name__ == '__main__':
    unittest.main()
