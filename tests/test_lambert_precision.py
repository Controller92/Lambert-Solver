"""Retained shape precision and arbitrary-plane geometry regression checks."""
from decimal import Decimal, localcontext
import math
import unittest

from src.lambert_geometry import GeometryStatus, LambertInput, prepare_geometry
from src.lambert_root import RootStatus, solve_x_zero_rev
from src.lambert_root_preparation import evaluate_derivatives_zero_rev, initial_guess_zero_rev
from src.lambert_tof import TofStatus, evaluate_tof_zero_rev


class PrecisionTests(unittest.TestCase):
    def test_invalid_or_inconsistent_q_is_rejected_at_every_layer(self):
        for q in (0., -1., 1.1, float('nan'), float('inf'), True, 'bad', .5):
            for evaluate in (evaluate_tof_zero_rev, evaluate_derivatives_zero_rev,
                             initial_guess_zero_rev):
                result = evaluate(1., .5, q=q)
                self.assertEqual(result.status, TofStatus.INVALID_INPUT, (evaluate, q))
            root = solve_x_zero_rev(1., .5, q=q)
            self.assertEqual(root.status, RootStatus.INVALID_INPUT)
            self.assertIsNone(root.x)
            self.assertIsNone(root.y)
        for lam in (-1., 1.):
            self.assertEqual(evaluate_tof_zero_rev(1., lam).status, TofStatus.INVALID_INPUT)

    def test_parabolic_limits_retain_q_when_lambda_rounds(self):
        with localcontext() as context:
            context.prec = 100
            for q in (1e-12, 1e-16, 1e-20):
                for sign in (-1, 1):
                    lam = sign*(Decimal(1)-Decimal.from_float(q)).sqrt()
                    actual = evaluate_derivatives_zero_rev(1., float(lam), q=q)
                    self.assertEqual(actual.status, TofStatus.OK)
                    c3, c5, c7, c9 = (1-lam**p for p in (3, 5, 7, 9))
                    expected = (Decimal(2)/3*c3, -Decimal(2)/5*c5,
                                Decimal(6)/7*c7-Decimal(2)/5*c5,
                                Decimal(18)/7*c7-Decimal(10)/3*c9)
                    for value, reference in zip((actual.time_nondim, actual.first,
                                                 actual.second, actual.third), expected):
                        self.assertLessEqual(abs(value/float(reference)-1), 1e-14)
                    guess = initial_guess_zero_rev(actual.time_nondim, float(lam), q=q)
                    root = solve_x_zero_rev(actual.time_nondim, float(lam), q=q)
                    self.assertEqual(guess.x0, 1.)
                    self.assertEqual(root.status, RootStatus.OK)
                    self.assertEqual(root.x, 1.)

    def test_general_3d_geometry_against_actual_decimal_inputs(self):
        u = (.3, -.4, .5)
        u = tuple(v/math.hypot(*u) for v in u)
        v = (-.2, .5, .52)
        projection = sum(a*b for a, b in zip(u, v))
        v = tuple(b-projection*a for a, b in zip(u, v))
        v = tuple(b/math.hypot(*v) for b in v)
        for angle in (1e-8, 1e-12, 2e-14, math.pi-2e-14):
            for ratio in (1., 1+5e-14, 2.):
                b = tuple(ratio*(math.cos(angle)*a+math.sin(angle)*t) for a, t in zip(u, v))
                result = prepare_geometry(LambertInput(u, b, 1., 1.))
                self.assertEqual(result.status, GeometryStatus.OK)
                g = result.geometry
                with localcontext() as context:
                    context.prec = 100
                    a, b = [tuple(map(Decimal.from_float, r)) for r in (u, b)]
                    norm = lambda r: sum(t*t for t in r).sqrt()
                    r1, r2 = norm(a), norm(b)
                    c = norm(tuple(y-x for x, y in zip(a, b)))
                    h = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
                    normal = tuple(t/norm(h) for t in h)
                    rho = (r1-r2)/c
                    sigma = (1-rho*rho).sqrt()
                    self.assertLessEqual(math.dist(g.normal, tuple(map(float, normal))), 1e-15)
                    self.assertLessEqual(abs(g.rho-float(rho)), 1e-15)
                    self.assertLessEqual(abs(g.sigma-float(sigma)), 1e-15)
                    self.assertLessEqual(abs(g.q/float(2*c/(r1+r2+c))-1), 1e-15)


if __name__ == '__main__':
    unittest.main()
