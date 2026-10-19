"""Independent high-precision forward-TOF validation, no root finder or CUDA.

Run with the requirements in validation/requirements.txt installed. Evaluates
the Lagrange alpha/beta equation (Izzo Eqs. 9-15) at 100 and 160 decimal digits,
not production Eq. (18) or its Battin-series implementation. Original source:
https://doi.org/10.1007/s10569-014-9587-y
"""
from collections import Counter
import math
from pathlib import Path
import random
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_tof import MAX_SERIES_TERMS, TofStatus, evaluate_tof_zero_rev

RELATIVE_TOLERANCE = 5e-14
REFERENCE_RELATIVE_TOLERANCE = mp.mpf('1e-70')


def lagrange_reference(x, lam, precision=100):
    """Independent anomaly-based formula; floats converted without decimal rounding."""
    with mp.workdps(precision):
        x, lam = mp.mpf(x), mp.mpf(lam)
        if x == 1:
            return mp.mpf(2)/3*(1-lam**3)
        d = abs(1-x*x)
        if x < 1:
            alpha = 2*mp.acos(x)
            beta = 2*mp.asin(lam*mp.sqrt(d))
            numerator = (alpha-mp.sin(alpha))-(beta-mp.sin(beta))
        else:
            alpha = 2*mp.acosh(x)
            beta = 2*mp.asinh(lam*mp.sqrt(d))
            numerator = (mp.sinh(alpha)-alpha)-(mp.sinh(beta)-beta)
        return numerator/(2*d**mp.mpf('1.5'))


def cases():
    # Includes representable neighbors of x=-1, x=1, lambda=+/-1 and both switches.
    xs = [-1+10**(-k) for k in range(1, 16)] + [math.nextafter(-1., 0.)]
    xs += [-.9, -.5, -.01, 0., 1e-14, .01, .1, .3, .7, .8, .9, 1., 1.1,
           1.2, 1.3, 2., 10., 1e3, 1e10, 1e50, 1e100, 1e150]
    xs += [1+sign*10**(-k) for k in range(1, 17) for sign in (-1, 1)]
    xs += [math.nextafter(v, direction) for v in (.8, 1., 1.2)
           for direction in (-math.inf, math.inf)]
    lambdas = [0., -.1, .1, -.5, .5, -.9, .9, -.99, .99]
    lambdas += [sign*(1-10**(-k)) for k in range(3, 16) for sign in (-1, 1)]
    lambdas += [math.nextafter(1., 0.), math.nextafter(-1., 0.)]
    # Deduplicate inputs that decimal powers round to the same binary64 values.
    pairs = list(dict.fromkeys((x, lam) for x in xs for lam in lambdas))
    rng = random.Random(9587)
    pairs += [(rng.uniform(-.999, 4.), rng.uniform(-.999, .999)) for _ in range(500)]
    return pairs


def validate():
    mp.mp.dps = 100
    counts = Counter()
    worst_by_method = {}
    worst_relative, worst_absolute, worst_reference = (mp.mpf(-1), None), (mp.mpf(-1), None), mp.mpf(0)
    max_terms = 0
    max_y_relative = mp.mpf(0)
    min_T, max_T = mp.inf, mp.mpf(0)
    pairs = cases()
    for x, lam in pairs:
        result = evaluate_tof_zero_rev(x, lam)
        if result.status != TofStatus.OK:
            raise AssertionError(f'Forward evaluation failed: x={x!r}, lambda={lam!r}: {result}')
        expected = lagrange_reference(x, lam)
        reference_check = lagrange_reference(x, lam, 160)
        reference_error = abs(expected/reference_check-1)
        if reference_error > REFERENCE_RELATIVE_TOLERANCE:
            raise AssertionError(f'Insufficient reference precision: {(x, lam, reference_error)}')
        relative = abs(mp.mpf(result.time_nondim)/expected-1)
        absolute = abs(mp.mpf(result.time_nondim)-expected)
        if relative > RELATIVE_TOLERANCE:
            raise AssertionError(f'T error: {(x, lam, float(relative), result)}')
        y_expected = mp.sqrt(1-mp.mpf(lam)**2*(1-mp.mpf(x)**2))
        y_relative = abs(mp.mpf(result.y)/y_expected-1)
        if y_relative > mp.mpf('1e-15'):
            raise AssertionError(f'y error: {(x, lam, result.y)}')
        max_y_relative = max(max_y_relative, y_relative)
        case = (x, lam, result.method.name, float(expected))
        if relative > worst_relative[0]:
            worst_relative = (relative, case)
        if absolute > worst_absolute[0]:
            worst_absolute = (absolute, case)
        name = result.method.name
        worst_by_method[name] = max(worst_by_method.get(name, 0.), float(relative))
        worst_reference = max(worst_reference, reference_error)
        min_T, max_T = min(min_T, expected), max(max_T, expected)
        max_terms = max(max_terms, result.series_terms)
        counts[name] += 1
    print(f'Python {sys.version.split()[0]}, mpmath {mp.__version__}')
    print(f'Validated {len(pairs)} distinct inputs; failures=0; counts={dict(counts)}')
    print(f'T range: {float(min_T):.17g} .. {float(max_T):.17g}')
    print(f'Maximum relative T error: {float(worst_relative[0]):.17g}; case={worst_relative[1]}')
    print(f'Maximum absolute T error: {float(worst_absolute[0]):.17g}; case={worst_absolute[1]}')
    print(f'Maximum relative T error by method: {worst_by_method}')
    print(f'Maximum relative y error: {float(max_y_relative):.17g}')
    print(f'Maximum relative reference change, 100 to 160 digits: {float(worst_reference):.3e}')
    print(f'Maximum nonconstant Battin-series terms: {max_terms}/{MAX_SERIES_TERMS}')
    print(f'Acceptance: relative T error <= {RELATIVE_TOLERANCE:g}; reference change <= 1e-70; y relative error <= 1e-15')


if __name__ == '__main__':
    validate()
