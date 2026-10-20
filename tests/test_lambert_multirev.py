"""Multi-revolution forward identities, minimum bracketing and explicit failures."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from src.lambert_multirev import (
    MAX_REVOLUTIONS, FeasibilityStatus, MinimumOptions, MinimumResult, MinimumStatus,
    classify_multirev_time, evaluate_derivatives_multi_rev, evaluate_tof_multi_rev,
    find_minimum_multi_rev,
)
from src.lambert_tof import TofStatus, evaluate_tof_zero_rev


class MultiRevolutionTests(unittest.TestCase):
    def test_minimum_energy_identities_for_every_revolution_count(self):
        for m in (1, 2, 10, MAX_REVOLUTIONS):
            r = evaluate_derivatives_multi_rev(0., 0., m)
            self.assertEqual(r.status, TofStatus.OK)
            self.assertAlmostEqual(r.time_nondim/((m+.5)*math.pi), 1., delta=3e-16)
            self.assertEqual(r.first, -2.)
            self.assertEqual(r.third, -16.)
            self.assertEqual(r.second, 3*r.time_nondim)

    def test_revolution_increment_and_reflection_identities(self):
        for lam in (-.999, -.3, 0., .8, .999):
            for x in (-.95, -.2, 0., .4, .95):
                d = (1-x)*(1+x)
                increment = math.pi/d/math.sqrt(d)
                a = evaluate_tof_multi_rev(x, lam, 1)
                b = evaluate_tof_multi_rev(x, lam, 2)
                z = evaluate_tof_zero_rev(x, lam)
                reflected = evaluate_tof_multi_rev(-x, -lam, 1)
                self.assertEqual(a.status, TofStatus.OK)
                self.assertAlmostEqual((b.time_nondim-a.time_nondim)/increment, 1., delta=2e-15)
                self.assertAlmostEqual((a.time_nondim-z.time_nondim)/increment, 1., delta=2e-15)
                self.assertAlmostEqual((a.time_nondim+reflected.time_nondim)/(3*increment), 1., delta=8e-16)

    def test_elliptic_domain_and_count_contract(self):
        for x in (-1., 1., 2., math.inf, math.nan, True, 1j):
            self.assertEqual(evaluate_tof_multi_rev(x, .5, 1).status, TofStatus.INVALID_INPUT)
            self.assertEqual(evaluate_derivatives_multi_rev(x, .5, 1).status, TofStatus.INVALID_INPUT)
        for m in (0, -1, True, 1., MAX_REVOLUTIONS+1):
            self.assertEqual(evaluate_tof_multi_rev(.2, .5, m).status, TofStatus.INVALID_INPUT)
            self.assertEqual(find_minimum_multi_rev(.5, m).status, MinimumStatus.INVALID_INPUT)
        for x in (math.nextafter(-1., 0.), math.nextafter(1., 0.)):
            self.assertEqual(evaluate_derivatives_multi_rev(x, .5, 1).status, TofStatus.OK)

    def test_retained_shape_and_explicit_derivative_range_failure(self):
        for lam in (-1., 1.):
            for q in (1e-16, 1e-100, math.ulp(0.)):
                for x in (0., .25, math.nextafter(1., 0.)):
                    r = evaluate_derivatives_multi_rev(x, lam, 1, q=q)
                    self.assertEqual(r.status, TofStatus.OK, (x, lam, q, r))
                self.assertEqual(find_minimum_multi_rev(lam, 1, q=q).status, MinimumStatus.OK)
            self.assertEqual(evaluate_tof_multi_rev(.2, lam, 1).status, TofStatus.INVALID_INPUT)
        self.assertEqual(evaluate_tof_multi_rev(.2, .5, 1, q=.01).status, TofStatus.INVALID_INPUT)
        x, q = 1e-162, math.ulp(0.)
        self.assertEqual(evaluate_tof_multi_rev(x, 1., 1, q=q).status, TofStatus.OK)
        failed = evaluate_derivatives_multi_rev(x, 1., 1, q=q)
        self.assertEqual(failed.status, TofStatus.NUMERICAL_RANGE)
        self.assertIsNone(failed.first)
        self.assertIsNone(failed.time_nondim)

    def test_rounded_zero_quarter_probe_regressions(self):
        # Independent Lagrange stationary points, reproduced by validate_multirev.
        for lam,q,x,time in (
            (.5361549015100815,.7125379215867147,.1457245762499736,4.453634581006508),
            (-.8035642912088069,.3542844298940877,.1498819294456678,5.017559990323802),
            (.39965722053412844,.840274106074935,.14593459557999497,4.521924203692727),
            (.39965722053412844,.8402741060749351,.14593459557999497,4.521924203692727)):
            r=find_minimum_multi_rev(lam,1,q=q)
            self.assertEqual(r.status,MinimumStatus.OK)
            self.assertLessEqual(abs(r.x_min-x),1e-14+1e-14*abs(x))
            self.assertLessEqual(abs(r.time_nondim-time),r.estimated_time_error)
            self.assertLessEqual(r.first_lower,0.)
            self.assertGreaterEqual(r.first_upper,0.)

    def test_minimum_against_independent_high_precision_fixture(self):
        # Lagrange TOF, 160-digit arithmetic and 220 derivative bisections;
        # reproduced by validation/validate_multirev.py (not this implementation).
        result = find_minimum_multi_rev(0., 1)
        self.assertEqual(result.status, MinimumStatus.OK)
        self.assertAlmostEqual(result.x_min, .14598550926652533, delta=1e-14)
        self.assertAlmostEqual(result.time_nondim, 4.566663294296799, delta=3e-15)
        self.assertLess(result.first_lower, 0.)
        self.assertGreater(result.first_upper, 0.)
        self.assertGreater(result.iterations, 0)
        self.assertEqual(result.iterations, result.halley_steps+result.bisection_steps)
        self.assertGreaterEqual(result.tof_evaluations, result.iterations+3)

    def test_minimum_bracket_and_stationarity_over_shapes_and_counts(self):
        for lam in (-.9999, -.91, -.9, 0., .9, .9999):
            previous = 0.
            for m in (1, 2, 5, 100):
                r = find_minimum_multi_rev(lam, m)
                self.assertEqual(r.status, MinimumStatus.OK, r)
                self.assertTrue(0 <= r.lower <= r.x_min <= r.upper < 1)
                self.assertLessEqual(r.first_lower, 0.)
                self.assertGreaterEqual(r.first_upper, 0.)
                self.assertLessEqual(r.upper-r.lower, 1e-14+1e-14*abs(r.x_min))
                self.assertGreater(r.time_nondim, previous)
                previous = r.time_nondim
                for x in (r.x_min-.001, r.x_min+.001):
                    self.assertGreater(evaluate_tof_multi_rev(x, lam, m).time_nondim, r.time_nondim)

    def test_bisection_safeguard_with_disabled_halley(self):
        with patch('src.lambert_multirev._halley_candidate', return_value=math.nan):
            r = find_minimum_multi_rev(.5, 1)
        self.assertEqual(r.status, MinimumStatus.OK)
        self.assertEqual(r.halley_steps, 0)
        self.assertEqual(r.iterations, r.bisection_steps)
        self.assertLess(r.iterations, 64)

    def test_exhausted_budget_and_bad_options_have_no_minimum(self):
        r = find_minimum_multi_rev(.5, 1, options=MinimumOptions(max_iterations=0))
        self.assertEqual(r.status, MinimumStatus.ITERATION_LIMIT)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.tof_evaluations, 2)
        self.assertIsNone(r.x_min)
        self.assertIsNone(r.time_nondim)
        r = find_minimum_multi_rev(0., 1, options=MinimumOptions(x_atol=0., x_rtol=0.))
        self.assertEqual(r.status, MinimumStatus.RESOLUTION_LIMIT)
        self.assertIsNone(r.x_min)
        for options in (None, MinimumOptions(x_atol=-1), MinimumOptions(x_rtol=math.nan),
                        MinimumOptions(max_iterations=True)):
            r = find_minimum_multi_rev(.5, 1, options=options)
            self.assertEqual(r.status, MinimumStatus.INVALID_INPUT)
            self.assertEqual(r.tof_evaluations, 0)

    def test_feasibility_does_not_invent_a_merged_root(self):
        minimum = find_minimum_multi_rev(.5, 1)
        t = minimum.time_nondim
        self.assertEqual(classify_multirev_time(t, minimum).status, FeasibilityStatus.NEAR_MINIMUM)
        self.assertEqual(classify_multirev_time(math.nextafter(t, 0.), minimum).status,
                         FeasibilityStatus.NEAR_MINIMUM)
        self.assertEqual(classify_multirev_time(math.nextafter(t, math.inf), minimum).status,
                         FeasibilityStatus.NEAR_MINIMUM)
        self.assertEqual(classify_multirev_time(t-1e-8, minimum).status, FeasibilityStatus.NO_SOLUTION)
        self.assertEqual(classify_multirev_time(t+1e-8, minimum).status, FeasibilityStatus.TWO_SOLUTIONS)
        for target in (True, 0., -1., math.nan, math.inf, 1j):
            self.assertEqual(classify_multirev_time(target, minimum).status, FeasibilityStatus.INVALID_INPUT)
        self.assertEqual(classify_multirev_time(t, MinimumResult()).status,
                         FeasibilityStatus.MINIMUM_FAILURE)
        malformed = replace(minimum, estimated_time_error=math.nan)
        self.assertEqual(classify_multirev_time(t, malformed).status, FeasibilityStatus.INVALID_INPUT)


if __name__ == '__main__':
    unittest.main()
