"""Independent Lagrange/multiprecision validation of the M>0 preparation layer.

No mission states or runtime implementation of Lambert roots are used here.
The reference differentiates Lagrange's anomaly equation with mpmath and
bisects that independent derivative to find each stationary point.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_multirev import (
    MAX_REVOLUTIONS, FeasibilityStatus, MinimumStatus, classify_multirev_time,
    evaluate_derivatives_multi_rev, evaluate_tof_multi_rev, find_minimum_multi_rev,
)
from src.lambert_tof import TofStatus


def reference_lambda(lam, q):
    return (mp.sign(lam)*mp.sqrt(1-mp.mpf(q)) if q is not None and q < .5
            else mp.mpf(lam))


def precision(q):
    return max(100, int(-math.log10(q))+80) if q is not None else 100


def lagrange(x, lam, m):
    d = (1-x)*(1+x)
    alpha = 2*mp.acos(x)
    beta = 2*mp.asin(lam*mp.sqrt(d))
    return ((alpha-mp.sin(alpha))-(beta-mp.sin(beta))+2*m*mp.pi)/(2*d*mp.sqrt(d))


def minimum_reference(lam, m, q=None):
    with mp.workdps(max(mp.mp.dps, precision(q))):
        l = reference_lambda(lam, q)
        f = lambda x: lagrange(x, l, m)
        lo, hi = mp.mpf(0), mp.mpf('.5')
        if mp.diff(f, hi) <= 0:
            raise AssertionError('Independent reference minimum not bracketed')
        for _ in range(500 if q is not None and q < 1e-60 else 220):
            middle = (lo+hi)/2
            if mp.diff(f, middle) < 0:
                lo = middle
            else:
                hi = middle
        x = (lo+hi)/2
        return +x, +f(x)


def forward_cases():
    for m in (1, 2, 5, 20, 100, MAX_REVOLUTIONS):
        for lam in (-math.nextafter(1., 0.), -.99, -.5, 0., .5, .99, math.nextafter(1., 0.)):
            for x in (math.nextafter(-1., 0.), -.99, -.5, -1e-8, 0., .1, .5, .7, .8,
                      .95, .999999, math.nextafter(1., 0.)):
                yield x, lam, m, None
    for q in (1e-12, 1e-16, 1e-30, 1e-100, 1e-300, math.ulp(0.)):
        for sign in (-1, 1):
            lam = sign*math.sqrt(1-q)
            for m in (1, 5, MAX_REVOLUTIONS):
                for x in (-.99, 0., 1e-8, .5, math.nextafter(1., 0.)):
                    yield x, lam, m, q
    # A genuinely unrepresentable third derivative with a representable TOF.
    yield 1e-162, 1., 1, math.ulp(0.)


def validate():
    maximum_time = 0.
    derivative_maxima = [0., 0., 0.]
    forward_ulps = 0.
    count = expected_ranges = 0
    for x, lam, m, q in forward_cases():
        count += 1
        with mp.workdps(precision(q)):
            l = reference_lambda(lam, q)
            f = lambda z: lagrange(z, l, m)
            expected = f(mp.mpf(x))
            actual = evaluate_tof_multi_rev(x, lam, m, q=q)
            derivatives = evaluate_derivatives_multi_rev(x, lam, m, q=q)
            error = float(abs(mp.mpf(actual.time_nondim)/expected-1)) if actual.status == TofStatus.OK else math.inf
            if error > 5e-14:
                raise AssertionError(f'Independent TOF failure: {x,lam,m,q,actual,error}')
            maximum_time = max(maximum_time, error)
            ulps = float(abs(mp.mpf(actual.time_nondim)-expected)/math.ulp(actual.time_nondim))
            forward_ulps = max(forward_ulps, ulps)
            if ulps > 16:
                raise AssertionError(f'Forward-roundoff model exceeded: {x,lam,m,q,ulps}')
            references = [mp.diff(f, mp.mpf(x), n) for n in (1, 2, 3)]
            if any(abs(v) > sys.float_info.max for v in references):
                if derivatives.status != TofStatus.NUMERICAL_RANGE:
                    raise AssertionError('Expected explicit unrepresentable-derivative status')
                expected_ranges += 1
                continue
            if derivatives.status != TofStatus.OK:
                raise AssertionError(f'Unexpected derivative failure: {x,lam,m,q,derivatives}')
            for i, (value, reference, bound) in enumerate(zip(
                    (derivatives.first, derivatives.second, derivatives.third), references,
                    (5e-13, 5e-13, 2e-12))):
                error = float(abs(mp.mpf(value)-reference)/max(1, abs(reference)))
                if error > bound:
                    raise AssertionError(f'Independent derivative failure: {x,lam,m,q,i,error}')
                derivative_maxima[i] = max(derivative_maxima[i], error)
    print(f'Independent Lagrange forward cases={count}; max relative T error={maximum_time:.17g}; '
          f'max forward error={forward_ulps:.17g} ulps; expected derivative range failures={expected_ranges}', flush=True)
    print('Numerical differentiation max errors scaled by max(1,abs(reference)):', derivative_maxima, flush=True)

    cases = [(lam, m, None) for lam in (-.9999, -.9, 0., .9, .9999)
             for m in (1, 2, 5, 20, 100, MAX_REVOLUTIONS)]
    cases += [(sign*math.sqrt(1-q), m, q) for q in (1e-16, 1e-100, math.ulp(0.))
              for sign in (-1, 1) for m in (1, 5, MAX_REVOLUTIONS)]
    cases.append((-.91, 5, None))  # Exercises the rounded-zero derivative probes on this libm.
    x_error = time_ulps = calibration = 0.
    max_iterations = 0
    for lam, m, q in cases:
        actual = find_minimum_multi_rev(lam, m, q=q)
        if actual.status != MinimumStatus.OK:
            raise AssertionError(f'Minimum failed: {lam,m,q,actual}')
        with mp.workdps(precision(q)):
            x, time = minimum_reference(lam, m, q)
            xe = float(abs(mp.mpf(actual.x_min)-x))
            te = abs(mp.mpf(actual.time_nondim)-time)
            if xe > 1e-14+1e-14*float(abs(x)) or te > actual.estimated_time_error:
                raise AssertionError(f'Minimum position/time calibration failed: {lam,m,q,actual,xe,te}')
            x_error = max(x_error, xe)
            time_ulps = max(time_ulps, float(te/math.ulp(actual.time_nondim)))
            calibration = max(calibration, float(te/actual.estimated_time_error))
            max_iterations = max(max_iterations, actual.iterations)
            for factor in (-2, -1, 0, 1, 2):
                target = actual.time_nondim+factor*actual.estimated_time_error
                outcome = classify_multirev_time(target, actual)
                if outcome.status == FeasibilityStatus.NO_SOLUTION and not mp.mpf(target) < time:
                    raise AssertionError('Incorrect infeasible classification outside uncertainty band')
                if outcome.status == FeasibilityStatus.TWO_SOLUTIONS and not mp.mpf(target) > time:
                    raise AssertionError('Incorrect two-solution classification outside uncertainty band')
    result = dict(forward_cases=count, max_relative_time_error=maximum_time,
                  max_forward_ulps=forward_ulps, derivative_maxima=derivative_maxima,
                  expected_derivative_range_failures=expected_ranges, minimum_cases=len(cases),
                  max_absolute_minimum_x_error=x_error, max_minimum_time_error_ulps=time_ulps,
                  max_time_error_over_estimate=calibration, max_minimum_iterations=max_iterations)
    print('Independent minimum and feasibility validation:', json.dumps(result), flush=True)
    with mp.workdps(160):
        x, t = minimum_reference(0., 1)
        print('lambda=0, M=1 fixture:', mp.nstr(x, 40), mp.nstr(t, 40), flush=True)
    print('All multi-revolution scalar preparation checks passed.', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = validate()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2)+'\n')
