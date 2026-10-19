"""Independent numerical differentiation and starter validation, without solving.

Requires validation/requirements.txt. Derivatives are central finite differences
of the high-precision Lagrange anomaly expression, not the production analytic
recurrence or differentiated series. Step and precision are changed independently.
"""
from collections import Counter
import math
from pathlib import Path
import random
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_root_preparation import (
    MAX_DERIVATIVE_TERMS, evaluate_derivatives_zero_rev, initial_guess_zero_rev,
)
from src.lambert_tof import TofStatus
from validate_tof import lagrange_reference

DERIVATIVE_RELATIVE_TOLERANCES = (5e-14, 2e-13, 2e-12)
REFERENCE_RELATIVE_TOLERANCE = mp.mpf('1e-17')
STARTER_SCALED_TOLERANCE = 5e-14


def numerical_derivatives(x, lam, step_scale, precision):
    with mp.workdps(precision):
        x, lam = mp.mpf(x), mp.mpf(lam)
        characteristic = min(1+x, max(1, abs(x)))
        if lam:
            # Distance to complex zeros of y limits the local analytic scale.
            characteristic = min(characteristic, mp.sqrt(x*x+(1-lam*lam)/(lam*lam)))
        h = characteristic*mp.mpf(step_scale)
        f = lambda z: lagrange_reference(z, lam, precision=mp.mp.dps)
        return tuple(mp.diff(f, x, order, method='step', h=h) for order in (1, 2, 3))


def starter_reference(target, lam):
    """Direct paper formula in high precision; corrected log-line middle branch."""
    target, lam = mp.mpf(target), mp.mpf(lam)
    T0 = lagrange_reference(0, lam, mp.mp.dps)
    T1 = mp.mpf(2)/3*(1-lam**3)
    if target >= T0:
        return mp.expm1(mp.mpf(2)/3*mp.log(T0/target))
    if target <= T1:
        return 1+mp.mpf(5)/2*T1*(T1-target)/(target*(1-lam**5))
    return mp.expm1(mp.log(2)*mp.log(T0/target)/mp.log(T0/T1))


def cases():
    xs = [math.nextafter(-1., 0.), -1+1e-12, -.999999, -.99, -.9, -.5,
          -1e-8, 0., 1e-12, 1e-8, .01, .1, .3, .5, .69, .7, .8, .9,
          .99, 1., 1.01, 1.2, 1.3, 1.31, 2., 10., 1e4, 1e20, 1e40]
    xs += [1+sign*10**(-k) for k in (4, 8, 12, 15) for sign in (-1, 1)]
    xs += [math.nextafter(edge, direction) for edge in (.7, 1., 1.3)
           for direction in (-math.inf, math.inf)]
    lambdas = [-.999999, -.999, -.99, -.9, -.5, 0., .5, .9, .99, .999,
               .999999, math.nextafter(1., 0.), math.nextafter(-1., 0.)]
    pairs = list(dict.fromkeys((x, lam) for x in xs for lam in lambdas))
    rng = random.Random(9587)
    pairs += [(rng.uniform(-.99, 10.), rng.uniform(-.999, .999)) for _ in range(500)]
    return pairs


def validate():
    mp.mp.dps = 100
    pairs = cases()
    counts = Counter()
    worst = [(mp.mpf(-1), None) for _ in range(3)]
    worst_by_method = {}
    max_reference_change = mp.mpf(0)
    max_terms = 0
    starter_counts = Counter()
    starter_max_formula_error = mp.mpf(0)
    seed_errors = []
    worst_seed = (0., None)
    range_cases = []
    for x, lam in pairs:
        result = evaluate_derivatives_zero_rev(x, lam)
        if result.status != TofStatus.OK:
            raise AssertionError(f'Derivative evaluation failed: {(x, lam, result)}')
        coarse = numerical_derivatives(x, lam, '1e-10', 100)
        refined = numerical_derivatives(x, lam, '1e-12', 160)
        for order, (actual, reference, previous) in enumerate(zip(
                (result.first, result.second, result.third), refined, coarse)):
            change = abs(previous/reference-1)
            if change > REFERENCE_RELATIVE_TOLERANCE:
                raise AssertionError(f'Reference did not stabilize: {(x, lam, order+1, change)}')
            error = abs(mp.mpf(actual)/reference-1)
            if error > DERIVATIVE_RELATIVE_TOLERANCES[order]:
                raise AssertionError(f'Derivative error: {(x, lam, order+1, error, result)}')
            if error > worst[order][0]:
                worst[order] = error, (x, lam, result.method.name)
            max_reference_change = max(max_reference_change, change)
            per_method = worst_by_method.setdefault(result.method.name, [0., 0., 0.])
            per_method[order] = max(per_method[order], float(error))
        max_terms = max(max_terms, result.series_terms)
        counts[result.method.name] += 1

        target = float(lagrange_reference(x, lam))
        guess = initial_guess_zero_rev(target, lam)
        expected = starter_reference(target, lam)
        if not math.isfinite(float(expected)) or float(expected) <= -1:
            if guess.status != TofStatus.NUMERICAL_RANGE:
                raise AssertionError(f'Unrepresentable starter not rejected: {(x, lam, guess)}')
            range_cases.append((x, lam, target))
            continue
        if guess.status != TofStatus.OK:
            raise AssertionError(f'Starter evaluation failed: {(x, lam, target, guess)}')
        formula_error = abs(mp.mpf(guess.x0)-expected)/(1+abs(expected))
        if formula_error > STARTER_SCALED_TOLERANCE:
            raise AssertionError(f'Starter formula mismatch: {(x, lam, formula_error)}')
        starter_max_formula_error = max(starter_max_formula_error, formula_error)
        seed_error = abs(guess.x0-x)/(1+abs(x))
        seed_errors.append(seed_error)
        if seed_error > worst_seed[0]:
            worst_seed = seed_error, (x, lam, guess.x0, target)
        starter_counts[guess.region.name] += 1

    print(f'Python {sys.version.split()[0]}, mpmath {mp.__version__}')
    print(f'Validated {len(pairs)} derivative inputs; failures=0; methods={dict(counts)}')
    for order, (error, case) in enumerate(worst, 1):
        print(f'Maximum relative derivative error, order {order}: {float(error):.17g}; case={case}')
    print(f'Maximum relative errors by method (orders 1,2,3): {worst_by_method}')
    print(f'Maximum relative reference change, h/100 and 160 digits: {float(max_reference_change):.3e}')
    print(f'Maximum derivative-series index: {max_terms}/{MAX_DERIVATIVE_TERMS}')
    print(f'Validated representable starters: {len(seed_errors)}; regions={dict(starter_counts)}')
    print(f'Maximum scaled error against high-precision starter formula: {float(starter_max_formula_error):.17g}')
    print(f'Known-x starter error |x0-x|/(1+|x|): mean={sum(seed_errors)/len(seed_errors):.17g}, max={worst_seed[0]:.17g}; case={worst_seed[1]}')
    print(f'Expected starter NUMERICAL_RANGE cases: {len(range_cases)}; inputs={range_cases}')
    print(f'Acceptance: derivative relative bounds={DERIVATIVE_RELATIVE_TOLERANCES}; reference change <= 1e-17; scaled starter formula error <= {STARTER_SCALED_TOLERANCE:g}')
    print('Starter quality is descriptive only. No root iteration or convergence claim is made.')


if __name__ == '__main__':
    validate()
