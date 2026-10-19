"""Root recovery, safeguard behavior and truthful failure diagnostics for M=0."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from src.lambert_geometry import LambertInput, prepare_geometry
from src.lambert_root import RootOptions, RootStatus, solve_x_zero_rev
from src.lambert_root_preparation import DerivativeResult
from src.lambert_tof import TofResult, TofStatus, evaluate_tof_zero_rev


class RootTests(unittest.TestCase):
    def check_success(self, r, target, options=RootOptions()):
        self.assertEqual(r.status, RootStatus.OK, r)
        self.assertTrue(math.isfinite(r.x) and r.x > -1)
        self.assertTrue(math.isfinite(r.y) and r.y > 0)
        self.assertEqual(r.x, r.last_x)
        self.assertLessEqual(abs(r.residual), options.tof_atol+options.tof_rtol*target)
        self.assertLessEqual(abs(r.last_step), options.x_atol+options.x_rtol*abs(r.x))
        self.assertLessEqual(r.lower, r.x)
        self.assertLessEqual(r.x, r.upper)
        self.assertEqual(r.iterations, r.householder_steps+r.bisection_steps+r.verification_steps)

    def test_analytic_points_and_actual_zero_step(self):
        for lam in (-0.9, 0., 0.9):
            for x in (0., 1.):
                target = evaluate_tof_zero_rev(x, lam).time_nondim
                r = solve_x_zero_rev(target, lam)
                self.check_success(r, target)
                self.assertEqual(r.x, x)
                self.assertEqual(r.iterations, 1)
                self.assertEqual(r.verification_steps, 1)
                self.assertEqual(r.last_step, 0.)

    def test_analytical_circular_short_and_long_arcs(self):
        R, mu = 7e6, 3.986004418e14
        quarter = math.pi/2*math.sqrt(R**3/mu)
        for long_way, factor, sign in ((False, 1, 1), (True, 3, -1)):
            g = prepare_geometry(LambertInput((R, 0., 0.), (0., R, 0.), factor*quarter, mu, long_way)).geometry
            expected = sign*math.sqrt(1-g.semiperimeter_m/(2*R))
            r = solve_x_zero_rev(g.time_nondim, g.lambda_)
            self.check_success(r, g.time_nondim)
            self.assertAlmostEqual(r.x, expected, delta=2e-14)

    def test_five_unrepresentable_starters_recover_known_x(self):
        for lam in (0.99, 0.999, 0.999999, math.nextafter(1., 0.)):
            target = 9.494881184090847e23
            r = solve_x_zero_rev(target, lam)
            self.check_success(r, target)
            self.assertEqual(r.x, math.nextafter(-1., 0.))
            self.assertTrue(r.used_fallback_guess)
            self.assertGreater(r.bisection_steps, 0)
        target = 1.1107575921394486e18
        r = solve_x_zero_rev(target, math.nextafter(1., 0.))
        self.check_success(r, target)
        self.assertEqual(r.x, -0.999999999999)
        self.assertTrue(r.used_fallback_guess)

    def test_poor_starter_and_large_hyperbolic_derivative_underflow(self):
        target = 2.2204460492490802e-14
        r = solve_x_zero_rev(target, math.nextafter(1., 0.))
        self.check_success(r, target)
        self.assertAlmostEqual(r.x, 0.01, delta=1e-14)
        # The forward function is valid here even though all three derivatives
        # are not simultaneously representable. Bisection remains available.
        target = 1e-150
        r = solve_x_zero_rev(target, 0.)
        self.check_success(r, target)
        self.assertLessEqual(abs(r.x/1e150-1), 2e-14)
        self.assertGreater(r.derivative_failures, 0)
        self.assertGreater(r.bisection_steps, 0)

    def test_unsafe_steps_and_unavailable_derivatives_use_bisection(self):
        target = evaluate_tof_zero_rev(0.3, 0.5).time_nondim
        with patch('src.lambert_root._householder_step', return_value=math.inf):
            r = solve_x_zero_rev(target, 0.5)
        self.check_success(r, target)
        self.assertGreater(r.rejected_householder_steps, 0)
        self.assertEqual(r.householder_steps, 0)
        with patch('src.lambert_root.evaluate_derivatives_zero_rev',
                   return_value=DerivativeResult(TofStatus.NUMERICAL_RANGE)):
            r = solve_x_zero_rev(target, 0.5)
        self.check_success(r, target)
        self.assertGreater(r.derivative_failures, 0)

    def test_actual_tof_evaluation_count(self):
        target = evaluate_tof_zero_rev(0.3, -0.5).time_nondim
        count = 0
        def counted(x, lam, *, q=None):
            nonlocal count
            count += 1
            return evaluate_tof_zero_rev(x, lam, q=q)
        with patch('src.lambert_root.evaluate_tof_zero_rev', side_effect=counted), \
                patch('src.lambert_root_preparation.evaluate_tof_zero_rev', side_effect=counted):
            r = solve_x_zero_rev(target, -0.5)
        self.check_success(r, target)
        self.assertEqual(r.tof_evaluations, count)

    def test_flat_long_arc_reports_x_conditioning(self):
        lam = math.nextafter(-1., 0.)
        # Generate the exact target used by this test from the validated forward
        # layer; independent root-error checks live in validation/validate_root.py.
        target = evaluate_tof_zero_rev(-1e-5, lam).time_nondim
        r = solve_x_zero_rev(target, lam)
        self.check_success(r, target)
        self.assertTrue(r.x_precision_limited)
        self.assertGreater(r.estimated_x_error, RootOptions().x_atol+RootOptions().x_rtol*abs(r.x))

    def test_iteration_budget_never_claims_success(self):
        target = evaluate_tof_zero_rev(0.3, -0.5).time_nondim
        for budget in (0, 1):
            r = solve_x_zero_rev(target, -0.5, replace(RootOptions(), max_iterations=budget))
            self.assertEqual(r.status, RootStatus.ITERATION_LIMIT)
            self.assertEqual(r.iterations, budget)
            self.assertIsNone(r.x)
            self.assertIsNone(r.y)
            self.assertIsNotNone(r.last_x)
            self.assertIsNotNone(r.residual)

    def test_zero_step_without_small_residual_is_resolution_failure(self):
        left = math.nextafter(-1., 0.)
        right = math.nextafter(left, 0.)
        a = evaluate_tof_zero_rev(left, 0.).time_nondim
        b = evaluate_tof_zero_rev(right, 0.).time_nondim
        for target in ((a+b)/2, 2*a):
            r = solve_x_zero_rev(target, 0.)
            self.assertEqual(r.status, RootStatus.RESOLUTION_LIMIT, r)
            self.assertIsNone(r.x)
            self.assertIsNone(r.y)
            self.assertGreater(abs(r.relative_residual), RootOptions().tof_rtol)

    def test_forward_failure_is_reported(self):
        with patch('src.lambert_root.evaluate_tof_zero_rev',
                   return_value=TofResult(TofStatus.NUMERICAL_RANGE)):
            r = solve_x_zero_rev(1., 0.)
        self.assertEqual(r.status, RootStatus.EVALUATION_FAILURE)
        self.assertEqual(r.evaluation_status, TofStatus.NUMERICAL_RANGE)
        self.assertIsNone(r.x)

    def test_invalid_inputs_options_and_unrepresentable_upper_range(self):
        for target, lam in ((0., 0.), (-1., 0.), (float('nan'), 0.), (1., 1.),
                            (True, 0.), (1., True), (None, 0.), (1., float('inf'))):
            self.assertEqual(solve_x_zero_rev(target, lam).status, RootStatus.INVALID_INPUT)
        for options in (None, replace(RootOptions(), x_atol=-1), replace(RootOptions(), x_atol=True),
                        replace(RootOptions(), tof_rtol=1), replace(RootOptions(), x_rtol=float('nan')),
                        replace(RootOptions(), max_iterations=True),
                        replace(RootOptions(), max_bracket_expansions=-1)):
            self.assertEqual(solve_x_zero_rev(1., 0., options).status, RootStatus.INVALID_INPUT)
        r = solve_x_zero_rev(5e-324, 0.)
        self.assertEqual(r.status, RootStatus.NUMERICAL_RANGE)
        self.assertIsNone(r.x)
        r = solve_x_zero_rev(5e-324, 0., replace(RootOptions(), max_bracket_expansions=0))
        self.assertEqual(r.status, RootStatus.BRACKET_LIMIT)
        self.assertEqual(r.bracket_expansions, 0)


if __name__ == '__main__':
    unittest.main()
