"""Exact identities, endpoint conventions and explicit root-preparation failures."""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch

from src.lambert_root_preparation import (
    DerivativeMethod, GuessRegion, evaluate_derivatives_zero_rev, initial_guess_zero_rev,
)
from src.lambert_tof import TofStatus, evaluate_tof_zero_rev


class RootPreparationTests(unittest.TestCase):
    def derivative(self, x, lam):
        result = evaluate_derivatives_zero_rev(x, lam)
        self.assertEqual(result.status, TofStatus.OK, (x, lam, result))
        return result

    def guess(self, target, lam):
        result = initial_guess_zero_rev(target, lam)
        self.assertEqual(result.status, TofStatus.OK, (target, lam, result))
        return result

    def test_regular_minimum_energy_point(self):
        for lam in (-math.nextafter(1., 0.), -0.9, 0., 0.9, math.nextafter(1., 0.)):
            r = self.derivative(0., lam)
            self.assertEqual(r.first, -2.)
            self.assertEqual(r.third, -16.)
            self.assertEqual(r.method, DerivativeMethod.RECURRENCE)
        self.assertAlmostEqual(self.derivative(0., 0.).second, 3*math.pi/2, delta=1e-15)

    def test_exact_parabolic_derivatives_against_decimal_limits(self):
        with localcontext() as context:
            context.prec = 80
            for lam in (-math.nextafter(1., 0.), -0.9, -0.5, 0., 0.5, 0.9,
                        math.nextafter(1., 0.)):
                l = Decimal.from_float(lam)
                c5, c7, c9 = 1-l**5, 1-l**7, 1-l**9
                expected = (-Decimal(2)/5*c5,
                            Decimal(6)/7*c7-Decimal(2)/5*c5,
                            Decimal(18)/7*c7-Decimal(10)/3*c9)
                r = self.derivative(1., lam)
                self.assertEqual(r.method, DerivativeMethod.PARABOLIC)
                for actual, reference in zip((r.first, r.second, r.third), expected):
                    self.assertLessEqual(abs(actual/float(reference)-1), 1e-14)

    def test_parabolic_neighbors_and_series_switches(self):
        for lam in (-0.999, -0.5, 0., 0.5, 0.999):
            exact = self.derivative(1., lam)
            for x in (math.nextafter(1., 0.), math.nextafter(1., math.inf)):
                r = self.derivative(x, lam)
                self.assertEqual(r.method, DerivativeMethod.SERIES)
                for a, b in zip((r.first, r.second, r.third), (exact.first, exact.second, exact.third)):
                    self.assertLessEqual(abs(a/b-1), 1e-14)
            for edge in (0.7, 1.3):
                a = self.derivative(math.nextafter(edge, -math.inf), lam)
                b = self.derivative(math.nextafter(edge, math.inf), lam)
                for u, v in zip((a.first, a.second, a.third), (b.first, b.second, b.third)):
                    self.assertLessEqual(abs(u/v-1), 1e-12)

    def test_starter_thresholds_and_correct_middle_inversion(self):
        for lam in (-math.nextafter(1., 0.), -0.9, 0., 0.9, math.nextafter(1., 0.)):
            T0 = evaluate_tof_zero_rev(0., lam).time_nondim
            T1 = evaluate_tof_zero_rev(1., lam).time_nondim
            self.assertEqual(self.guess(T0, lam).x0, 0.)
            self.assertEqual(self.guess(T1, lam).x0, 1.)
            middle = self.guess(math.sqrt(T0*T1), lam)
            self.assertEqual(middle.region, GuessRegion.INTERMEDIATE)
            # Halfway in log-time must mean xi=log(2)/2, hence x=sqrt(2)-1.
            self.assertAlmostEqual(middle.x0, math.sqrt(2)-1, delta=7e-16)
            self.assertTrue(-1 < self.guess(2*T0, lam).x0 < 0)
            self.assertTrue(self.guess(T1/2, lam).x0 > 1)

    def test_starter_continuity_and_monotonicity(self):
        for lam in (-0.999, 0., 0.999):
            T0 = evaluate_tof_zero_rev(0., lam).time_nondim
            T1 = evaluate_tof_zero_rev(1., lam).time_nondim
            for threshold, expected_x in ((T0, 0.), (T1, 1.)):
                for direction in (0., math.inf):
                    value = self.guess(math.nextafter(threshold, direction), lam).x0
                    self.assertAlmostEqual(value, expected_x, delta=5e-15)
            targets = (T1/10, T1/2, T1, math.sqrt(T0*T1), T0, 2*T0, 10*T0)
            values = [self.guess(T, lam).x0 for T in targets]
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_short_time_starter_matches_inverse_parabolic_slope(self):
        for lam in (-0.999, 0., 0.999):
            T1 = evaluate_tof_zero_rev(1., lam).time_nondim
            target = T1*(1-1e-5)
            estimate = (self.guess(target, lam).x0-1)/(target-T1)
            inverse_slope = 1/self.derivative(1., lam).first
            self.assertLessEqual(abs(estimate/inverse_slope-1), 1.1e-5)

    def test_all_failure_paths_return_no_partial_values(self):
        for args in ((-1., 0.), (True, 0.), (0., 1.), (0., float('nan'))):
            r = evaluate_derivatives_zero_rev(*args)
            self.assertEqual(r.status, TofStatus.INVALID_INPUT)
            self.assertIsNone(r.first)
        # T is representable here, but its third derivative underflows binary64.
        r = evaluate_derivatives_zero_rev(1e100, 0.)
        self.assertEqual(r.status, TofStatus.NUMERICAL_RANGE)
        self.assertIsNone(r.time_nondim)
        with patch('src.lambert_root_preparation.MAX_DERIVATIVE_TERMS', 2):
            r = evaluate_derivatives_zero_rev(0.9, -0.9)
        self.assertEqual(r.status, TofStatus.SERIES_NONCONVERGENCE)
        self.assertEqual(r.series_terms, 2)
        self.assertIsNone(r.third)
        for args in ((0., 0.), (-1., 0.), (float('nan'), 0.), (True, 0.), (1., 1.), ('bad', 0.)):
            r = initial_guess_zero_rev(*args)
            self.assertEqual(r.status, TofStatus.INVALID_INPUT)
            self.assertIsNone(r.x0)
        for target in (1e300, 5e-324):
            r = initial_guess_zero_rev(target, 0.)
            self.assertEqual(r.status, TofStatus.NUMERICAL_RANGE)
            self.assertIsNone(r.x0)


if __name__ == '__main__':
    unittest.main()
