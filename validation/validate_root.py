"""Independent known-x and high-precision bisection checks for the M=0 root.

Requires validation/requirements.txt. The reference uses the Lagrange anomaly
formula and bisection only, without the production derivatives, starter or step.
"""
from collections import Counter
import math
from pathlib import Path
import sys

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lambert_root import TOF_ROUNDOFF_ULPS, RootOptions, RootStatus, solve_x_zero_rev
from validate_tof import cases, lagrange_reference

ROOT_SCALED_TOLERANCE = 5e-14
INDEPENDENT_TOF_RELATIVE_TOLERANCE = 6e-13


def reference_tof(x, lam, precision=100):
    # A bisection midpoint can approach x=1 much more closely than a binary64
    # input. Lagrange's subtractive formula then needs extra precision. Retain
    # its independent formulation instead of sharing production's Battin series.
    distance = abs((1-x)*(1+x))
    extra = max(0, int(-mp.log10(distance))+1) if distance else 0
    return lagrange_reference(x, lam, precision+extra+40)


def bisection_reference(target, lam, generating_x):
    """Invert the rounded input T independently, in a generating-x neighborhood.

    The known generating x locates an independent bracket, not a production
    root estimate. 100 bisections make its scaled width <= about 1e-40.
    """
    target, lam, known = mp.mpf(target), mp.mpf(lam), mp.mpf(generating_x)
    radius = min((known+1)/4, mp.mpf('1e-10')*max(1, abs(known)))
    lo, hi = known-radius, known+radius
    f_lo = reference_tof(lo, lam)-target
    f_hi = reference_tof(hi, lam)-target
    if not (f_lo >= 0 and f_hi <= 0):
        raise AssertionError(f'Independent bracket failed: {(generating_x, lam, f_lo, f_hi)}')
    for _ in range(100):
        middle = (lo+hi)/2
        f = reference_tof(middle, lam)-target
        if f > 0:
            lo = middle
        elif f < 0:
            hi = middle
        else:
            lo = hi = middle
            break
    # Confirm final reference signs using a higher-precision equation evaluation.
    if not (reference_tof(lo, lam, 160) >= target
            and reference_tof(hi, lam, 160) <= target):
        raise AssertionError(f'Independent final bracket did not stabilize: {(lo, hi, lam)}')
    return (lo+hi)/2, (hi-lo)/(1+abs(known))


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction*len(values))-1]


def conditioning_cases():
    return [(-10.**(-k), lam) for k in range(3, 15)
            for lam in (-.999999, -.9999999999, -1+1e-14, math.nextafter(-1., 0.))]


def validate_conditioning():
    warnings = 0
    max_ratio = mp.mpf(0)
    worst = (mp.mpf(-1), None)
    pairs = conditioning_cases()
    for known, lam in pairs:
        target = float(lagrange_reference(known, lam))
        result = solve_x_zero_rev(target, lam)
        if result.status != RootStatus.OK:
            raise AssertionError(f'Conditioning case failed: {(known, lam, result)}')
        reference, _ = bisection_reference(target, lam, known)
        error = abs(mp.mpf(result.x)-reference)
        scaled = error/(1+abs(reference))
        independent_residual = abs(lagrange_reference(result.x, lam, 160)/target-1)
        if independent_residual > INDEPENDENT_TOF_RELATIVE_TOLERANCE:
            raise AssertionError(f'Conditioned residual failed: {(known, lam, independent_residual)}')
        if result.estimated_x_error is None or error > max(
                ROOT_SCALED_TOLERANCE*(1+abs(reference)), result.estimated_x_error):
            raise AssertionError(f'Conditioned error exceeds reported model: {(known, lam, error, result)}')
        if scaled > ROOT_SCALED_TOLERANCE and result.x_precision_limited is not True:
            raise AssertionError(f'Unflagged loss of x accuracy: {(known, lam, scaled, result)}')
        if result.estimated_x_error:
            max_ratio = max(max_ratio, error/result.estimated_x_error)
        warnings += result.x_precision_limited is True
        if scaled > worst[0]:
            worst = scaled, (known, lam, result.x, result.estimated_x_error)
    print(f'Additional conditioning cases: {len(pairs)}; flagged={warnings}; unexpected failures=0')
    print(f'Conditioning-case maximum scaled x error: {float(worst[0]):.17g}; case={worst[1]}')
    print(f'Maximum actual absolute x error / reported first-order estimate: {float(max_ratio):.6g}')
    print('Conditioning-case acceptance uses the reported error model when the ordinary scaled x bound cannot be met; it is not a uniform 5e-14 x-accuracy claim.')


def validate_roundoff():
    worst = (mp.mpf(0), None)
    pairs = cases()+conditioning_cases()
    for known, lam in pairs:
        target = float(lagrange_reference(known, lam))
        result = solve_x_zero_rev(target, lam)
        if result.status != RootStatus.OK:
            raise AssertionError(f'Roundoff calibration root failed: {(known, lam, result)}')
        exact = lagrange_reference(result.x, lam, 160)
        ulps = abs(mp.mpf(result.time_nondim)-exact)/mp.mpf(math.ulp(result.time_nondim))
        if ulps > worst[0]:
            worst = ulps, (known, lam, result.x)
    if worst[0] > TOF_ROUNDOFF_ULPS:
        raise AssertionError(f'Diagnostic roundoff model exceeded: {worst}')
    print(f'Forward roundoff calibration at {len(pairs)} returned roots: max={float(worst[0]):.17g} ulps; case={worst[1]}; allowance={TOF_ROUNDOFF_ULPS}')


def validate():
    mp.mp.dps = 100
    options = RootOptions()
    iterations = []
    expansion_counts = []
    methods = Counter()
    fallback_cases = 0
    derivative_fallback_cases = 0
    bisection_cases = 0
    precision_warnings = 0
    unknown_precision = 0
    worst_root = (mp.mpf(-1), None)
    worst_known = (mp.mpf(-1), None)
    worst_independent_residual = (mp.mpf(-1), None)
    max_reported_relative_residual = 0.
    max_reference_width = mp.mpf(0)
    pairs = cases()
    for known, lam in pairs:
        target = float(lagrange_reference(known, lam))
        result = solve_x_zero_rev(target, lam, options)
        if result.status != RootStatus.OK:
            raise AssertionError(f'Unexpected root failure: {(known, lam, target, result)}')
        if not (result.lower <= result.x <= result.upper and result.x > -1):
            raise AssertionError(f'Root outside maintained bracket: {result}')
        if (abs(result.residual) > options.tof_atol+options.tof_rtol*target
                or abs(result.last_step) > options.x_atol+options.x_rtol*abs(result.x)):
            raise AssertionError(f'Success violates residual/step tolerances: {result}')
        if result.iterations != result.householder_steps+result.bisection_steps+result.verification_steps:
            raise AssertionError(f'Iteration counts disagree: {result}')
        reference, width = bisection_reference(target, lam, known)
        root_error = abs(mp.mpf(result.x)-reference)/(1+abs(reference))
        known_error = abs(mp.mpf(result.x)-mp.mpf(known))/(1+abs(mp.mpf(known)))
        independent_residual = abs(lagrange_reference(result.x, lam, 160)/mp.mpf(target)-1)
        if root_error > ROOT_SCALED_TOLERANCE:
            raise AssertionError(f'Independent root error: {(known, lam, root_error, result)}')
        if independent_residual > INDEPENDENT_TOF_RELATIVE_TOLERANCE:
            raise AssertionError(f'Independent TOF error: {(known, lam, independent_residual, result)}')
        case = (known, lam, result.x, target)
        if root_error > worst_root[0]:
            worst_root = root_error, case
        if known_error > worst_known[0]:
            worst_known = known_error, case
        if independent_residual > worst_independent_residual[0]:
            worst_independent_residual = independent_residual, case
        max_reported_relative_residual = max(max_reported_relative_residual, abs(result.relative_residual))
        max_reference_width = max(max_reference_width, width)
        iterations.append(result.iterations)
        expansion_counts.append(result.bracket_expansions)
        fallback_cases += result.used_fallback_guess
        derivative_fallback_cases += result.derivative_failures > 0
        bisection_cases += result.bisection_steps > 0
        precision_warnings += result.x_precision_limited is True
        unknown_precision += result.x_precision_limited is None
        methods['householder'] += result.householder_steps
        methods['bisection'] += result.bisection_steps
        methods['verification'] += result.verification_steps
    print(f'Python {sys.version.split()[0]}, mpmath {mp.__version__}')
    print(f'Validated {len(pairs)} root problems; unexpected failures=0')
    print(f'Maximum scaled error against independent bisection: {float(worst_root[0]):.17g}; case={worst_root[1]}')
    print(f'Maximum scaled error against generating x: {float(worst_known[0]):.17g}; case={worst_known[1]}')
    print(f'Maximum independent relative TOF residual: {float(worst_independent_residual[0]):.17g}; case={worst_independent_residual[1]}')
    print(f'Maximum reported absolute relative TOF residual: {max_reported_relative_residual:.17g}')
    print(f'Maximum scaled reference bracket width: {float(max_reference_width):.3e}')
    print(f'Root iterations: min={min(iterations)}, mean={sum(iterations)/len(iterations):.6f}, p50={percentile(iterations,.5)}, p95={percentile(iterations,.95)}, p99={percentile(iterations,.99)}, max={max(iterations)}')
    print(f'Completed step counts: {dict(methods)}')
    print(f'Cases using bisection={bisection_cases}, fallback starter={fallback_cases}, derivative failure fallback={derivative_fallback_cases}, maximum bracket expansions={max(expansion_counts)}')
    print(f'Root precision diagnostics: flagged={precision_warnings}, unavailable={unknown_precision}')
    print(f'Acceptance: scaled root error <= {ROOT_SCALED_TOLERANCE:g}; independent relative TOF residual <= {INDEPENDENT_TOF_RELATIVE_TOLERANCE:g}; both production stopping conditions checked')
    print('All cases use M=0. No velocity, physical propagation or GPU performance claim.')
    validate_conditioning()
    validate_roundoff()


if __name__ == '__main__':
    validate()
