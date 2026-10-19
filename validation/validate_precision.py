"""Independent high-precision checks of retained q and nearly collinear inputs.

Scalar references use the Lagrange anomaly equation. Physical references
rebuild geometry from the actual binary64 inputs in multiprecision, invert
Lagrange with bisection, then reconstruct velocities in multiprecision.
No production helper is used to construct either reference.
"""
from collections import Counter
import math
from pathlib import Path
import random
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_geometry import LambertInput
from src.lambert_root import RootStatus, TOF_ROUNDOFF_ULPS, solve_x_zero_rev
from src.lambert_root_preparation import evaluate_derivatives_zero_rev
from src.lambert_solver import LambertStatus, solve_zero_rev
from src.lambert_tof import TofStatus, evaluate_tof_zero_rev
from validate_root import bisection_reference, reference_tof
from validate_root_preparation import numerical_derivatives
from validate_tof import lagrange_reference


def norm(vector):
    return mp.sqrt(sum(v*v for v in vector))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def physical_reference(problem):
    """Reference for the received numbers, including their endpoint rounding."""
    with mp.workdps(120):
        a, b = [tuple(map(mp.mpf, v)) for v in (problem.r1_m, problem.r2_m)]
        mu, tof = mp.mpf(problem.mu_m3_s2), mp.mpf(problem.tof_s)
        r1, r2 = norm(a), norm(b)
        c = norm(tuple(y-x for x, y in zip(a, b)))
        s = (r1+r2+c)/2
        lam = mp.sqrt(1-c/s)*(-1 if problem.long_way else 1)
        target = tof*mp.sqrt(2*mu/s**3)
        lo, hi = mp.mpf(-1), mp.mpf(1)
        while reference_tof(hi, lam) > target:
            hi = 2*hi+1
        for _ in range(180):
            middle = (lo+hi)/2
            if reference_tof(middle, lam) > target:
                lo = middle
            else:
                hi = middle
        x = (lo+hi)/2
        y = mp.sqrt(1-lam**2*(1-x*x))
        rho, gamma = (r1-r2)/c, mp.sqrt(mu*s/2)
        sigma = mp.sqrt(1-rho*rho)
        normal = cross(a, b)
        normal = tuple(v/norm(normal)*(-1 if problem.long_way else 1) for v in normal)
        u1, u2 = tuple(v/r1 for v in a), tuple(v/r2 for v in b)
        t1, t2 = cross(normal, u1), cross(normal, u2)
        radial1 = gamma/r1*((lam*y-x)-rho*(lam*y+x))
        radial2 = -gamma/r2*((lam*y-x)+rho*(lam*y+x))
        transverse = gamma*sigma*(y+lam*x)
        v1 = tuple(radial1*r+transverse/r1*t for r, t in zip(u1, t1))
        v2 = tuple(radial2*r+transverse/r2*t for r, t in zip(u2, t2))
        return v1, v2, x, normal, rho, sigma, c/s


def scalar_cases():
    xs = (-.9, -1e-5, 0., 1e-12, .01, .3, .7, .8, .9,
          math.nextafter(1., 0.), 1., math.nextafter(1., math.inf),
          1.2, 1.3, 2., 10., 1e4, 1e40, 1e150)
    for q in (.25, .01, 1e-4, 1e-8, 1e-12, 1e-15, 1e-16):
        for sign in (-1, 1):
            for x in xs:
                yield x, q, sign
    rng = random.Random(958706)
    for _ in range(100):
        yield rng.uniform(-.99, 10.), 10**rng.uniform(-16, -.6), rng.choice((-1, 1))


def validate_scalar():
    counts = Counter()
    maxima = dict(tof_relative=0., tof_ulps=0., first=0., second=0., third=0., root_scaled=0.)
    for known, q, sign in scalar_cases():
        # q is authoritative in this small-q corpus. Derive lambda from q in
        # multiprecision before rounding only the production lambda input.
        reference_lam = sign*mp.sqrt(1-mp.mpf(q))
        lam = float(reference_lam)
        forward = evaluate_tof_zero_rev(known, lam, q=q)
        if forward.status != TofStatus.OK:
            raise AssertionError(f'Forward q evaluation failed: {(known, q, sign, forward)}')
        expected = lagrange_reference(known, reference_lam, 160)
        error = abs(mp.mpf(forward.time_nondim)/expected-1)
        if error > 5e-14:
            raise AssertionError(f'Forward q error: {(known, q, sign, error)}')
        maxima['tof_relative'] = max(maxima['tof_relative'], float(error))
        derivative = evaluate_derivatives_zero_rev(known, lam, q=q)
        if abs(known) < 1e100:
            if derivative.status != TofStatus.OK:
                raise AssertionError(f'Derivative q evaluation failed: {(known, q, sign, derivative)}')
            references = numerical_derivatives(known, reference_lam, '1e-12', 160)
            for name, value, reference, bound in zip(
                    ('first', 'second', 'third'),
                    (derivative.first, derivative.second, derivative.third),
                    references, (5e-14, 2e-13, 2e-12)):
                error = float(abs(mp.mpf(value)/reference-1))
                if error > bound:
                    raise AssertionError(f'{name} q error: {(known, q, sign, error)}')
                maxima[name] = max(maxima[name], error)
            counts['derivatives'] += 1
        elif derivative.status != TofStatus.NUMERICAL_RANGE:
            raise AssertionError(f'Expected higher-derivative range limit: {(known, q, sign, derivative)}')
        target = float(expected)
        root = solve_x_zero_rev(target, lam, q=q)
        if root.status != RootStatus.OK:
            raise AssertionError(f'Root q evaluation failed: {(known, q, sign, root)}')
        reference, _ = bisection_reference(target, reference_lam, known)
        error = abs(mp.mpf(root.x)-reference)
        scaled = error/(1+abs(reference))
        if error > max(mp.mpf('5e-14')*(1+abs(reference)), root.estimated_x_error or 0.):
            raise AssertionError(f'Root q error beyond conditioning model: {(known, q, sign, error, root)}')
        if scaled > mp.mpf('5e-14') and root.x_precision_limited is not True:
            raise AssertionError(f'Unflagged root q conditioning: {(known, q, sign, root)}')
        exact_time = lagrange_reference(root.x, reference_lam, 160)
        ulps = abs(mp.mpf(root.time_nondim)-exact_time)/mp.mpf(math.ulp(root.time_nondim))
        if ulps > TOF_ROUNDOFF_ULPS:
            raise AssertionError(f'Root q roundoff allowance exceeded: {(known, q, sign, ulps)}')
        maxima['root_scaled'] = max(maxima['root_scaled'], float(scaled))
        maxima['tof_ulps'] = max(maxima['tof_ulps'], float(ulps))
        counts['roots'] += 1
        counts['root_precision_warning'] += root.x_precision_limited is True
        counts['root_precision_unknown'] += root.x_precision_limited is None
    print('Retained-q scalar corpus:', dict(counts))
    print('Retained-q maxima:', maxima)


def physical_cases():
    first = (.3, -.4, .5)
    first = tuple(v/math.hypot(*first) for v in first)
    second = cross(first, (.7, .1, -.2))
    second = tuple(v/math.hypot(*second) for v in second)
    axes = (((1., 0., 0.), (0., 1., 0.)), (first, second))
    angles = (1e-4, 1e-8, 1e-12, 2e-14, math.pi-1e-8, math.pi-2e-14)
    for angle in angles:
        for ratio in (1., 1+5*min(angle, 1e-4), 2.):
            for plane, (u, v) in enumerate(axes):
                for radius, mu in ((1., 1.), (7e6, 3.986004418e14)):
                    r1 = tuple(radius*x for x in u)
                    r2 = tuple(radius*ratio*(math.cos(angle)*x+math.sin(angle)*y) for x, y in zip(u, v))
                    for long_way in (False, True):
                        tof = (2*math.pi-angle if long_way else angle)*radius/(math.sqrt(mu)/math.sqrt(radius))
                        yield (angle, ratio, plane, radius, long_way), LambertInput(r1, r2, tof, mu, long_way)


def validate_physical():
    maxima = dict(velocity=0., normal=0., rho=0., sigma=0., q_relative=0.)
    worst = None
    count = 0
    for label, problem in physical_cases():
        result = solve_zero_rev(problem)
        if result.status != LambertStatus.OK:
            raise AssertionError(f'Near-collinear solve failed: {label}: {result}')
        v1, v2, x, normal, rho, sigma, q = physical_reference(problem)
        scale = max(mp.sqrt(mp.mpf(problem.mu_m3_s2)/norm(tuple(map(mp.mpf, problem.r1_m)))), norm(v1), norm(v2))
        velocity_error = float(max(norm(tuple(mp.mpf(a)-b for a, b in zip(result.v1_m_s, v1))),
                                   norm(tuple(mp.mpf(a)-b for a, b in zip(result.v2_m_s, v2))))/scale)
        if velocity_error > 2e-12:
            raise AssertionError(f'Near-collinear physical velocity error: {label}: {velocity_error}')
        g = result.geometry.geometry
        errors = dict(velocity=velocity_error, normal=float(norm(tuple(mp.mpf(a)-b for a, b in zip(g.normal, normal)))),
                      rho=float(abs(g.rho-rho)), sigma=float(abs(g.sigma-sigma)), q_relative=float(abs(g.q/q-1)))
        for name, error in errors.items():
            if error > 5e-14 and name != 'velocity':
                raise AssertionError(f'Near-collinear geometry error: {label}: {name}={error}')
            if error > maxima[name]:
                maxima[name] = error
                if name == 'velocity':
                    worst = label
        count += 1
    print(f'Actual-rounded-input physical cases: {count}; unexpected failures=0')
    print('Physical maxima:', maxima, '; worst velocity case:', worst)


if __name__ == '__main__':
    mp.mp.dps = 120
    validate_scalar()
    validate_physical()
