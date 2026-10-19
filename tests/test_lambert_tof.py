"""Analytical boundary cases and failure reporting for forward M=0 TOF."""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch

from src.lambert_geometry import LambertInput, prepare_geometry
from src.lambert_tof import TofMethod, TofStatus, evaluate_tof_zero_rev


class TimeOfFlightTests(unittest.TestCase):
    def value(self, x, lam):
        result = evaluate_tof_zero_rev(x, lam)
        self.assertEqual(result.status, TofStatus.OK, (x, lam, result))
        self.assertTrue(math.isfinite(result.time_nondim) and result.time_nondim > 0)
        self.assertTrue(math.isfinite(result.y) and result.y > 0)
        return result

    def test_minimum_energy_identity(self):
        for lam in (-0.999, -0.9, -0.5, 0., 0.5, 0.9, 0.999):
            result = self.value(0., lam)
            expected = math.acos(lam) + lam*math.sqrt((1-lam)*(1+lam))
            self.assertAlmostEqual(result.time_nondim, expected, delta=8e-16)
            self.assertEqual(result.method, TofMethod.ELLIPTIC)
        self.assertEqual(self.value(0., 0.).time_nondim, math.pi/2)

    def test_parabolic_identity_with_independent_decimal_arithmetic(self):
        with localcontext() as context:
            context.prec = 80
            for lam in (-math.nextafter(1., 0.), -0.9, -0.5, 0., 0.5, 0.9,
                        math.nextafter(1., 0.)):
                result = self.value(1., lam)
                expected = float(Decimal(2)/3*(1-Decimal.from_float(lam)**3))
                self.assertLessEqual(abs(result.time_nondim/expected-1), 4e-16)
                self.assertEqual(result.method, TofMethod.PARABOLIC)
                self.assertEqual(result.y, 1.)
                self.assertEqual(result.series_terms, 0)

    def test_both_circular_arcs_reproduce_elapsed_time(self):
        R, mu = 7e6, 3.986004418e14
        quarter_time = math.pi/2*math.sqrt(R**3/mu)
        errors = []
        for long_way, factor, sign in ((False, 1, 1), (True, 3, -1)):
            g = prepare_geometry(LambertInput((R, 0., 0.), (0., R, 0.),
                                             factor*quarter_time, mu, long_way)).geometry
            # a=R for a circle and 1-x^2=s/(2a); the arc selects the x sign.
            x = sign*math.sqrt(1-g.semiperimeter_m/(2*R))
            T = self.value(x, g.lambda_).time_nondim
            errors.append(abs(T-g.time_nondim))
            self.assertLessEqual(errors[-1], 2e-15)
        print(f"Circular short/long T absolute errors: {errors[0]:.3e}, {errors[1]:.3e}")

    def test_closed_hyperbolic_case(self):
        # lambda=0, x=sqrt(2): alpha/2=asinh(1), sqrt(x^2-1)=1.
        result = self.value(math.sqrt(2), 0.)
        self.assertAlmostEqual(result.time_nondim, math.sqrt(2)-math.asinh(1.), delta=4e-16)
        self.assertEqual(result.method, TofMethod.HYPERBOLIC)

    def test_parabolic_neighbors_and_method_switches(self):
        for lam in (-math.nextafter(1., 0.), -0.5, 0., 0.5, math.nextafter(1., 0.)):
            at_one = self.value(1., lam).time_nondim
            for x in (math.nextafter(1., 0.), math.nextafter(1., math.inf)):
                result = self.value(x, lam)
                self.assertEqual(result.method, TofMethod.BATTIN)
                self.assertLessEqual(abs(result.time_nondim/at_one-1), 5e-15)
            for edge in (0.8, 1.2):
                left = self.value(math.nextafter(edge, -math.inf), lam).time_nondim
                right = self.value(math.nextafter(edge, math.inf), lam).time_nondim
                self.assertLessEqual(abs(left/right-1), 1e-14)

    def test_positive_decreasing_curve_on_representative_domains(self):
        xs = (-0.999999, -0.99, -0.8, -0.1, 0., 0.5, 0.8, 0.99,
              1., 1.01, 1.2, 2., 10., 1e3, 1e10, 1e100, 1e150)
        for lam in (-0.999, -0.5, 0., 0.5, 0.999):
            values = [self.value(x, lam).time_nondim for x in xs]
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_invalid_and_range_failures_have_no_values(self):
        invalid = [(x, 0.) for x in (-2., -1., float('inf'), float('-inf'), float('nan'), True, None, 'bad')]
        invalid += [(0., lam) for lam in (-1., 1., 2., float('inf'), float('nan'), True, None)]
        for pair in invalid:
            result = evaluate_tof_zero_rev(*pair)
            self.assertEqual(result.status, TofStatus.INVALID_INPUT, pair)
            self.assertIsNone(result.time_nondim)
            self.assertIsNone(result.y)
        # A representable mathematical T may still exceed the implementation's
        # intermediate range. Report that limitation instead of returning a value.
        result = evaluate_tof_zero_rev(1e300, -0.9)
        self.assertEqual(result.status, TofStatus.NUMERICAL_RANGE)
        self.assertIsNone(result.time_nondim)
        self.assertIsNone(result.y)

    def test_exhausted_series_reports_failure(self):
        with patch('src.lambert_tof.MAX_SERIES_TERMS', 1):
            result = evaluate_tof_zero_rev(0.8, -0.9)
        self.assertEqual(result.status, TofStatus.SERIES_NONCONVERGENCE)
        self.assertEqual(result.series_terms, 1)
        self.assertIsNone(result.time_nondim)
        self.assertIsNone(result.y)


if __name__ == '__main__':
    unittest.main()
