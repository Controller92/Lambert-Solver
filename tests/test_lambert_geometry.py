"""Analytical and independent high-precision checks, no ephemerides or CUDA."""
from dataclasses import replace
from decimal import Decimal, localcontext
import math
import random
import unittest

from src.lambert_geometry import (
    COLLINEAR_SIN_TOL, GeometryStatus, LambertInput, prepare_geometry,
)


def norm(vector):
    return math.hypot(*vector)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


class GeometryTests(unittest.TestCase):
    def geometry(self, problem):
        result = prepare_geometry(problem)
        self.assertEqual(result.status, GeometryStatus.OK)
        self.assertEqual(result.iterations, 0)
        self.assertIsNone(result.residual)
        return result.geometry

    def test_three_four_five_triangle(self):
        g = self.geometry(LambertInput((3., 0., 0.), (0., 4., 0.), 2., 1.))
        self.assertEqual((g.r1_norm_m, g.r2_norm_m, g.chord_m, g.semiperimeter_m),
                         (3., 4., 5., 6.))
        self.assertAlmostEqual(g.lambda_, 1/math.sqrt(6), delta=2e-16)
        self.assertAlmostEqual(g.time_nondim, 2/math.sqrt(108), delta=1e-16)
        self.assertEqual(g.rho, -0.2)
        self.assertAlmostEqual(g.sigma, math.sqrt(0.96), delta=3e-16)
        self.assertEqual(g.normal, (0., 0., 1.))
        self.assertEqual(g.transverse1, (0., 1., 0.))
        self.assertEqual(g.transverse2, (-1., 0., 0.))

    def test_analytical_circular_quarter_and_three_quarter_arcs(self):
        R, mu = 7e6, 3.986004418e14
        quarter_time = (math.pi/2)*math.sqrt(R**3/mu)
        p = LambertInput((R, 0., 0.), (0., R, 0.), quarter_time, mu)
        short = self.geometry(p)
        long = self.geometry(replace(p, tof_s=3*quarter_time, long_way=True))
        s_over_R = 1 + 1/math.sqrt(2)
        expected_T = (math.pi/2)*math.sqrt(2/s_over_R**3)
        self.assertAlmostEqual(short.lambda_, math.sqrt(2)-1, delta=2e-16)
        self.assertAlmostEqual(short.time_nondim, expected_T, delta=5e-16)
        self.assertAlmostEqual(long.lambda_, -short.lambda_, delta=0.)
        self.assertAlmostEqual(long.time_nondim, 3*expected_T, delta=2e-15)
        self.assertEqual(long.normal, (0., 0., -1.))
        self.assertAlmostEqual(long.transfer_angle_rad, 3*math.pi/2, delta=0.)
        # These are geometry checks on known circular boundary data, not a solve.

    def test_rotation_and_endpoint_reversal(self):
        p = LambertInput((3., 0., 0.), (0., 4., 0.), 2., 1.)
        g = self.geometry(p)
        # Proper 3D rotation cyclically permutes axes and moves the plane off XY.
        rotate = lambda v: (v[2], v[0], v[1])
        rotated = self.geometry(replace(p, r1_m=rotate(p.r1_m), r2_m=rotate(p.r2_m)))
        self.assertEqual(rotated.normal, rotate(g.normal))
        self.assertEqual(rotated.transverse2, rotate(g.transverse2))
        self.assertEqual(rotated.time_nondim, g.time_nondim)
        reversed_g = self.geometry(replace(p, r1_m=p.r2_m, r2_m=p.r1_m))
        self.assertEqual(reversed_g.normal, tuple(-v for v in g.normal))
        self.assertEqual(reversed_g.rho, -g.rho)
        self.assertEqual(reversed_g.lambda_, g.lambda_)

    def test_similarity_scaling(self):
        p = LambertInput((3., 0., 0.), (0., 4., 0.), 2., 1.)
        base = self.geometry(p)
        errors = []
        for k in (1e-8, 1e-4, 1., 1e4, 1e8):
            g = self.geometry(replace(p, r1_m=tuple(k*x for x in p.r1_m),
                                      r2_m=tuple(k*x for x in p.r2_m), mu_m3_s2=k**3))
            errors.extend((abs(g.lambda_-base.lambda_), abs(g.time_nondim-base.time_nondim)))
        self.assertLessEqual(max(errors), 4e-16)
        print(f"L-similarity max absolute lambda/T error: {max(errors):.3e}")

    def test_nearly_antiparallel_and_unequal_nearly_parallel(self):
        # Direct 1-c/s rounds to zero in the first case; half-angle identity survives.
        e = 1e-9
        g = self.geometry(LambertInput((1., 0., 0.), (-math.cos(e), math.sin(e), 0.), 1., 1.))
        self.assertAlmostEqual(g.lambda_, e/4, delta=1e-24)
        g = self.geometry(LambertInput((1., 0., 0.), (2*math.cos(e), 2*math.sin(e), 0.), 1., 1.))
        self.assertAlmostEqual(g.sigma, math.sqrt(2)*e, delta=1e-23)

    def test_invalid_unsupported_and_degenerate_statuses(self):
        p = LambertInput((1., 0., 0.), (0., 1., 0.), 1., 1.)
        invalid = [replace(p, tof_s=0), replace(p, tof_s=float('nan')),
                   replace(p, mu_m3_s2=-1), replace(p, mu_m3_s2=float('inf')),
                   replace(p, r1_m=(0., 0., 0.)), replace(p, r2_m=(1., 2.)),
                   replace(p, r2_m=(0., float('nan'), 1.)), replace(p, r1_m=None),
                   replace(p, long_way=1), replace(p, revolutions=-1),
                   replace(p, revolutions=0.5), replace(p, revolutions=True),
                   replace(p, tof_s=True), replace(p, mu_m3_s2='bad'), None]
        for problem in invalid:
            self.assertEqual(prepare_geometry(problem).status, GeometryStatus.INVALID_INPUT)
        self.assertEqual(prepare_geometry(replace(p, revolutions=1)).status,
                         GeometryStatus.UNSUPPORTED_REVOLUTIONS)
        for r2 in ((1., 0., 0.), (2., 0., 0.), (-1., 0., 0.),
                   (1., COLLINEAR_SIN_TOL/2, 0.)):
            result = prepare_geometry(replace(p, r2_m=r2))
            self.assertEqual(result.status, GeometryStatus.DEGENERATE_GEOMETRY)
            self.assertIsNone(result.geometry)
        for scale in (1e-300, 1e300):
            result = prepare_geometry(replace(p, r1_m=(scale, 0., 0.), r2_m=(0., scale, 0.)))
            self.assertEqual(result.status, GeometryStatus.NUMERICAL_RANGE)
            self.assertIsNone(result.geometry)
        result = prepare_geometry(replace(p, r1_m=(1e-300, 0., 0.), r2_m=(0., 1e100, 0.)))
        self.assertEqual(result.status, GeometryStatus.NUMERICAL_RANGE)

    def test_independent_decimal_geometry_and_basis(self):
        rng = random.Random(9587)
        maxima = dict(length_relative=0., lambda_absolute=0., T_relative=0.,
                      rho_absolute=0., sigma_absolute=0., basis_absolute=0.)
        with localcontext() as ctx:
            ctx.prec = 80
            for i in range(500):
                length = (1e-3, 1., 1e11, 1e110)[i % 4]
                r1 = tuple(rng.uniform(-2, 2)*length for _ in range(3))
                r2 = tuple(rng.uniform(-2, 2)*length for _ in range(3))
                mu, tof = 3.986004418e14, 12345.6789
                long_way = bool(i % 2)
                g = self.geometry(LambertInput(r1, r2, tof, mu, long_way))
                a, b = [tuple(Decimal.from_float(v) for v in r) for r in (r1, r2)]
                n1 = sum(v*v for v in a).sqrt()
                n2 = sum(v*v for v in b).sqrt()
                c = sum((y-x)**2 for x, y in zip(a, b)).sqrt()
                s = (n1+n2+c)/2
                lam = (1-c/s).sqrt() * (-1 if long_way else 1)
                T = Decimal.from_float(tof)*(2*Decimal.from_float(mu)/s**3).sqrt()
                rho = (n1-n2)/c
                sigma = (1-rho*rho).sqrt()
                for actual, expected in zip((g.r1_norm_m, g.r2_norm_m, g.chord_m, g.semiperimeter_m), (n1, n2, c, s)):
                    maxima['length_relative'] = max(maxima['length_relative'], abs(actual/float(expected)-1))
                maxima['lambda_absolute'] = max(maxima['lambda_absolute'], abs(g.lambda_-float(lam)))
                maxima['T_relative'] = max(maxima['T_relative'], abs(g.time_nondim/float(T)-1))
                maxima['rho_absolute'] = max(maxima['rho_absolute'], abs(g.rho-float(rho)))
                maxima['sigma_absolute'] = max(maxima['sigma_absolute'], abs(g.sigma-float(sigma)))
                for radial, transverse in ((g.radial1, g.transverse1), (g.radial2, g.transverse2)):
                    errors = (abs(norm(radial)-1), abs(norm(transverse)-1), abs(norm(g.normal)-1),
                              abs(dot(radial, transverse)), abs(dot(g.normal, radial)), abs(dot(g.normal, transverse)))
                    maxima['basis_absolute'] = max(maxima['basis_absolute'], *errors)
        for name, error in maxima.items():
            self.assertLessEqual(error, 5e-14, name)
            print(f"500-case Decimal/basis max {name}: {error:.3e}")


if __name__ == '__main__':
    unittest.main()
