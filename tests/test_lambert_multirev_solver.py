"""Complete multi-revolution branch contracts and analytical orbit checks."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from src.lambert_geometry import LambertInput
from src.lambert_multirev import MinimumOptions, evaluate_tof_multi_rev, find_minimum_multi_rev
from src.lambert_multirev_root import (
    MultiRootStatus, RootBranch, solve_x_multi_rev, solve_x_multi_rev_pair,
)
from src.lambert_multirev_solver import solve_multi_rev, solve_multi_rev_pair
from src.lambert_root import RootOptions
from src.lambert_root_preparation import DerivativeResult
from src.lambert_solver import LambertStatus, solve_zero_rev
from src.lambert_tof import TofStatus
import src.lambert_multirev as forward_module


class MultiRevolutionSolverTests(unittest.TestCase):
    def setUp(self):
        self.circle = LambertInput((1., 0., 0.), (0., 1., 0.), 2*math.pi+math.pi/2,
                                   1., revolutions=1)

    def check_root(self, root, target):
        self.assertEqual(root.status, MultiRootStatus.OK, root)
        self.assertTrue(-1 < root.x < 1)
        self.assertLessEqual(abs(root.relative_residual), RootOptions().tof_rtol)
        self.assertLessEqual(abs(root.last_step), 1e-14+1e-14*abs(root.x))
        self.assertEqual(root.iterations, root.householder_steps+root.bisection_steps+root.verification_steps)
        if root.branch == RootBranch.LEFT:
            self.assertLess(root.x, root.minimum.lower)
        else:
            self.assertGreater(root.x, root.minimum.upper)
        value = evaluate_tof_multi_rev(root.x, root.minimum.lambda_, root.revolutions, q=root.minimum.q)
        self.assertEqual(value.time_nondim, root.time_nondim)
        self.assertEqual(root.residual, value.time_nondim-target)

    def test_circular_orbits_multiple_counts_and_orientations(self):
        for m in (1, 2, 5, 20):
            for long_way in (False, True):
                angle = 3*math.pi/2 if long_way else math.pi/2
                p = replace(self.circle, tof_s=2*math.pi*m+angle, long_way=long_way, revolutions=m)
                pair = solve_multi_rev_pair(p)
                self.assertTrue(all(r.status == LambertStatus.OK for r in pair), pair)
                sign = -1 if long_way else 1
                error = [max(math.dist(r.v1_m_s, (0., sign, 0.)), math.dist(r.v2_m_s, (-sign, 0., 0.))) for r in pair]
                self.assertLess(min(error), 3e-13)
                self.assertEqual(tuple(r.branch for r in pair), tuple(RootBranch))
                self.assertIs(pair[0].geometry, pair[1].geometry)
                self.assertIs(pair[0].root.minimum, pair[1].root.minimum)
                for result in pair:
                    self.check_root(result.root, result.geometry.geometry.time_nondim)
                    single = solve_multi_rev(p, result.branch)
                    self.assertEqual(single.v1_m_s, result.v1_m_s)
                    self.assertEqual(single.v2_m_s, result.v2_m_s)

    def test_branch_identity_does_not_depend_on_sign_of_x(self):
        # Juno scalar regression, both roots are positive.
        pair = solve_x_multi_rev_pair(4.174730431091548, .8151730690824357, 1,
                                      q=.3354928674427224)
        self.assertTrue(0 < pair[0].x < pair[0].minimum.x_min < pair[1].x)
        self.assertAlmostEqual(pair[0].x, .03211164043831082, delta=1e-14)
        self.assertAlmostEqual(pair[1].x, .24793848301682458, delta=1e-14)

    def test_near_minimum_is_explicit_and_resolved_neighbors_are_separate(self):
        minimum = find_minimum_multi_rev(.5, 1)
        for target in (minimum.time_nondim, math.nextafter(minimum.time_nondim, math.inf)):
            for r in solve_x_multi_rev_pair(target, .5, 1):
                self.assertEqual(r.status, MultiRootStatus.NEAR_MINIMUM)
                self.assertIsNone(r.x)
                self.assertEqual(r.tof_evaluations, 0)
        for r in solve_x_multi_rev_pair(minimum.time_nondim-1e-8, .5, 1):
            self.assertEqual(r.status, MultiRootStatus.NO_SOLUTION)
            self.assertIsNone(r.x)
        target = minimum.time_nondim*(1+1e-12)
        pair = solve_x_multi_rev_pair(target, .5, 1)
        for r in pair:
            self.check_root(r, target)
            self.assertTrue(r.x_precision_limited)
        self.assertLess(pair[0].x, pair[1].x)

    def test_all_actual_forward_calls_are_counted_once(self):
        original, count = forward_module._forward, 0
        def counted(*args):
            nonlocal count
            count += 1
            return original(*args)
        with patch('src.lambert_multirev._forward', side_effect=counted):
            left, right = solve_x_multi_rev_pair(8., .5, 2)
        self.assertEqual(left.status, MultiRootStatus.OK)
        self.assertEqual(right.status, MultiRootStatus.OK)
        self.assertIs(left.minimum, right.minimum)
        self.assertEqual(count, left.tof_evaluations+right.tof_evaluations+left.minimum.tof_evaluations)

    def test_disabled_derivatives_use_bisection_and_report_unknown_precision(self):
        unavailable = DerivativeResult(TofStatus.NUMERICAL_RANGE)
        with patch('src.lambert_multirev_root.evaluate_derivatives_multi_rev',
                   return_value=unavailable) as failures:
            pair = solve_multi_rev_pair(self.circle)
        self.assertEqual(sum(r.root.derivative_failures for r in pair), failures.call_count)
        for r in pair:
            self.assertEqual(r.status, LambertStatus.OK)
            self.assertGreater(r.root.bisection_steps, 0)
            self.assertGreater(r.root.derivative_failures, 0)
            self.assertIsNone(r.root.x_precision_limited)
            self.assertIsNone(r.root.estimated_x_error)

    def test_time_reversal_and_dimensional_similarity(self):
        p = LambertInput((3., -2., .4), (-1., 4., 2.), 200., 2., revolutions=2)
        for original, reverse in zip(solve_multi_rev_pair(p), solve_multi_rev_pair(replace(p, r1_m=p.r2_m, r2_m=p.r1_m))):
            self.assertEqual(original.status, LambertStatus.OK)
            self.assertLess(math.dist(reverse.v1_m_s, tuple(-v for v in original.v2_m_s)), 5e-13)
            self.assertLess(math.dist(reverse.v2_m_s, tuple(-v for v in original.v1_m_s)), 5e-13)
        scale, mass, tscale = 7e6, 3.986004418e14, math.sqrt((7e6)**3/3.986004418e14)
        dimensional = replace(self.circle, r1_m=(scale, 0., 0.), r2_m=(0., scale, 0.),
                              tof_s=self.circle.tof_s*tscale, mu_m3_s2=mass)
        for a, b in zip(solve_multi_rev_pair(self.circle), solve_multi_rev_pair(dimensional)):
            self.assertLess(math.dist(a.v1_m_s, tuple(v/(scale/tscale) for v in b.v1_m_s)), 5e-14)

    def test_failures_never_publish_velocities_or_substitute_a_branch(self):
        for p in (None, replace(self.circle, revolutions=0), replace(self.circle, revolutions=True),
                  replace(self.circle, revolutions=2**31), replace(self.circle, r2_m=(-1., 0., 0.))):
            for result in solve_multi_rev_pair(p):
                self.assertEqual(result.status, LambertStatus.GEOMETRY_FAILURE)
                self.assertIsNone(result.root)
                self.assertIsNone(result.v1_m_s)
        for options in (RootOptions(max_iterations=0), None):
            for result in solve_multi_rev_pair(self.circle, options):
                self.assertEqual(result.status, LambertStatus.ROOT_FAILURE)
                self.assertIsNone(result.v1_m_s)
                self.assertIsNone(result.root.x)
        failed = solve_multi_rev(self.circle, 'both')
        self.assertEqual(failed.root.status, MultiRootStatus.INVALID_INPUT)
        with patch('src.lambert_multirev_solver._reconstruct_velocities', side_effect=OverflowError):
            failed = solve_multi_rev(self.circle, 'right')
        self.assertEqual(failed.status, LambertStatus.NUMERICAL_RANGE)
        self.assertIsNone(failed.v1_m_s)
        self.assertIsNone(failed.v2_m_s)
        self.assertEqual(failed.root.status, MultiRootStatus.OK)
        for r in solve_x_multi_rev_pair(8., .5, 1, minimum_options=MinimumOptions(max_iterations=0)):
            self.assertEqual(r.status, MultiRootStatus.MINIMUM_FAILURE)
            self.assertIsNone(r.x)
        self.assertEqual(solve_zero_rev(self.circle).status, LambertStatus.GEOMETRY_FAILURE)

    def test_open_endpoints_and_invalid_root_requests(self):
        for branch in RootBranch:
            r = solve_x_multi_rev(1e300, 0., 1, branch)
            self.assertEqual(r.status, MultiRootStatus.RESOLUTION_LIMIT)
            self.assertIsNone(r.x)
            self.assertGreater(abs(r.relative_residual), RootOptions().tof_rtol)
        for target, lam, m, branch in ((True, .5, 1, 'left'), (8., True, 1, 'left'),
                                       (8., .5, 0, 'left'), (8., .5, 1, False)):
            self.assertEqual(solve_x_multi_rev(target, lam, m, branch).status, MultiRootStatus.INVALID_INPUT)
        for options in (RootOptions(x_atol=-1), RootOptions(tof_rtol=1), RootOptions(max_iterations=True)):
            self.assertEqual(solve_x_multi_rev(8., .5, 1, 'right', options).status, MultiRootStatus.INVALID_INPUT)

    def test_pair_retains_successful_sibling_when_one_branch_exhausts_budget(self):
        def starter(target, m, branch):
            return math.cos(3*math.pi/8) if branch == RootBranch.RIGHT else -.8
        with patch('src.lambert_multirev_root._starter', side_effect=starter):
            left, right = solve_multi_rev_pair(self.circle, RootOptions(max_iterations=1))
        self.assertEqual(left.root.status, MultiRootStatus.ITERATION_LIMIT)
        self.assertIsNone(left.v1_m_s)
        self.assertEqual(right.status, LambertStatus.OK)
        self.assertLess(math.dist(right.v1_m_s, (0., 1., 0.)), 3e-14)

    def test_infeasible_and_unresolved_full_transfers_publish_no_velocities(self):
        g = solve_multi_rev_pair(self.circle)[0].geometry.geometry
        minimum = find_minimum_multi_rev(g.lambda_, 1, q=g.q)
        for factor, expected in ((.5, MultiRootStatus.NO_SOLUTION), (1., MultiRootStatus.NEAR_MINIMUM)):
            p = replace(self.circle, tof_s=factor*minimum.time_nondim*g.time_scale_s)
            for result in solve_multi_rev_pair(p):
                self.assertEqual(result.status, LambertStatus.ROOT_FAILURE)
                self.assertEqual(result.root.status, expected)
                self.assertIsNone(result.root.x)
                self.assertIsNone(result.v1_m_s)
                self.assertIsNone(result.v2_m_s)


if __name__ == '__main__':
    unittest.main()
