"""M>0 scalar preparation for Izzo's Lambert solver, with no velocity roots.

T uses sqrt(2*mu/s^3)*elapsed_seconds; shape/retained-q conventions match
lambert_tof. Minimum-time uncertainty is an estimated roundoff diagnostic,
not a certified interval. The zero-revolution solver and native ABI are unchanged.
Reference: Izzo (2015), Eqs. (18), (22), DOI 10.1007/s10569-014-9587-y.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import sys

from .lambert_tof import TofMethod, TofResult, TofStatus, _eta, _resolve_geometry
from .lambert_root_preparation import DerivativeMethod, DerivativeResult

MAX_REVOLUTIONS = 2**31-1
MINIMUM_ROUNDOFF_ULPS = 16


def _arguments(x, lambda_, revolutions, q):
    if type(revolutions) is not int or not 1 <= revolutions <= MAX_REVOLUTIONS:
        raise ValueError('revolutions must be a positive signed-int32 count')
    if isinstance(x, bool):
        raise ValueError('boolean x')
    x = float(x)
    lam, q, _ = _resolve_geometry(lambda_, q)
    if not math.isfinite(x) or not -1 < x < 1:
        raise ValueError('multi-revolution x must lie strictly between -1 and 1')
    return x, lam, q


def _forward(x, lam, q, revolutions):
    d = (1-x)*(1+x)
    k = math.sqrt(d)
    y = math.hypot(math.sqrt(q), lam*x)
    eta = _eta(x, lam, q, y)
    # The M*pi term prevents the M=0 near-parabolic cancellation. eta may
    # underflow to zero for a retained subnormal q; the full M>0 time stays finite.
    psi = math.atan2(eta*k, math.fsum((lam, x*eta)))
    time = math.fsum(((psi+revolutions*math.pi)/k, lam*eta, -q*x))/d
    if not (math.isfinite(time) and time > 0 and math.isfinite(y) and y > 0):
        return TofResult(TofStatus.NUMERICAL_RANGE)
    return TofResult(TofStatus.OK, time, y, TofMethod.ELLIPTIC)


def evaluate_tof_multi_rev(x: float, lambda_: float, revolutions: int, *,
                           q: float | None = None) -> TofResult:
    """Evaluate T(x,lambda,M), M>=1 and -1<x<1, without finding a root.

    Counts are signed-int32 compatible; bool/fractional/zero/negative M are
    rejected. Retained q has the zero-revolution shape contract. Failures have
    no partial time/y. No ephemeris access or third-party numerics is used.
    """
    try:
        x, lam, q = _arguments(x, lambda_, revolutions, q)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return TofResult(TofStatus.INVALID_INPUT)
    try:
        return _forward(x, lam, q, revolutions)
    except (ValueError, OverflowError, ZeroDivisionError):
        return TofResult(TofStatus.NUMERICAL_RANGE)


def evaluate_derivatives_multi_rev(x: float, lambda_: float, revolutions: int, *,
                                   q: float | None = None) -> DerivativeResult:
    """T and its first three x derivatives at fixed shape and M.

    Genuine zero derivatives are allowed. Scaled q/y products avoid intermediate
    overflow from forming (lambda/y)**3 or **5. Derivative overflow can fail even
    when T is representable; no substitute derivatives are supplied.
    """
    try:
        x, lam, q = _arguments(x, lambda_, revolutions, q)
    except (TypeError, ValueError, OverflowError):
        return DerivativeResult(TofStatus.INVALID_INPUT)
    try:
        value = _forward(x, lam, q, revolutions)
        if value.status != TofStatus.OK:
            return DerivativeResult(value.status)
        time, y = value.time_nondim, value.y
        d = (1-x)*(1+x)
        eta = _eta(x, lam, q, y)
        w, ratio = math.sqrt(q)/y, lam/y
        a = w*ratio*lam
        q_ratio3 = (w*a)*lam
        q_ratio5_x = a*(a*((lam*x)/y))
        first = -2. if x == 0. else math.fsum((3*time*x, -2*q, -2*lam*lam*(eta/y)))/d
        second = math.fsum((3*time, 5*x*first, 2*q_ratio3))/d
        third = -16. if x == 0. else math.fsum((7*x*second, 8*first, -6*q_ratio5_x))/d
    except (OverflowError, ValueError, ZeroDivisionError):
        return DerivativeResult(TofStatus.NUMERICAL_RANGE)
    if not all(math.isfinite(v) for v in (first, second, third)):
        return DerivativeResult(TofStatus.NUMERICAL_RANGE)
    return DerivativeResult(TofStatus.OK, time, y, first, second, third,
                            DerivativeMethod.RECURRENCE)


class MinimumStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    EVALUATION_FAILURE = 2
    BRACKET_LIMIT = 3
    ITERATION_LIMIT = 4
    RESOLUTION_LIMIT = 5


@dataclass(frozen=True)
class MinimumOptions:
    x_atol: float = 1e-14
    x_rtol: float = 1e-14
    max_iterations: int = 128
    max_bracket_expansions: int = 32


@dataclass
class MinimumResult:
    """Minimum values exist only on OK; brackets/counters diagnose failures.

    The bracket is maintained from computed derivative signs. It is not an
    outward-rounded interval proof. iterations counts successfully evaluated
    candidates, selected by Halley or bisection. A computed zero derivative
    uses two extra quarter probes. tof_evaluations includes those probes and
    the final midpoint evaluation. Failures never populate a minimum value.
    """
    status: MinimumStatus = MinimumStatus.INVALID_INPUT
    revolutions: int | None = None
    lambda_: float | None = None
    q: float | None = None
    x_min: float | None = None
    time_nondim: float | None = None
    estimated_time_error: float | None = None
    first: float | None = None
    lower: float | None = None
    upper: float | None = None
    first_lower: float | None = None
    first_upper: float | None = None
    iterations: int = 0
    halley_steps: int = 0
    bisection_steps: int = 0
    zero_derivative_steps: int = 0
    bracket_expansions: int = 0
    tof_evaluations: int = 0
    evaluation_status: TofStatus | None = None


def _halley_candidate(x, value):
    try:
        u = value.first/value.second
        correction = .5*u*(value.third/value.second)
        denominator = 1-correction
        if abs(denominator) <= 64*sys.float_info.epsilon*max(1., abs(correction)):
            return math.nan
        return x-u/denominator
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan


def find_minimum_multi_rev(lambda_: float, revolutions: int, *, q: float | None = None,
                           options: MinimumOptions = MinimumOptions()) -> MinimumResult:
    """Locate the unique M>0 minimum by solving T'=0 on 0<x<1.

    Halley proposals stay inside the computed sign bracket; every fourth update
    bisects. Termination requires the full bracket width to meet the requested
    x tolerance. The estimated time error combines endpoint time spread and a
    calibrated 16-ulp forward allowance. It is explicitly not a certificate.
    """
    result = MinimumResult()
    try:
        _, lam, q = _arguments(0., lambda_, revolutions, q)
        if (not isinstance(options, MinimumOptions)
                or any(isinstance(v, bool) or not math.isfinite(v) or v < 0
                       for v in (options.x_atol, options.x_rtol))
                or options.x_rtol >= 1
                or type(options.max_iterations) is not int or options.max_iterations < 0
                or type(options.max_bracket_expansions) is not int
                or options.max_bracket_expansions < 0):
            return result
    except (TypeError, ValueError, OverflowError):
        return result
    result.revolutions, result.lambda_, result.q = revolutions, lam, q

    def evaluate(x):
        result.tof_evaluations += 1
        value = evaluate_derivatives_multi_rev(x, lam, revolutions, q=q)
        if value.status != TofStatus.OK:
            result.status = MinimumStatus.EVALUATION_FAILURE
            result.evaluation_status = value.status
            return None
        return value

    def bracket(lo, hi, left, right):
        result.lower, result.upper = lo, hi
        result.first_lower, result.first_upper = left.first, right.first

    lo, hi = 0., .5
    left, right = evaluate(lo), evaluate(hi)
    if left is None or right is None:
        return result
    bracket(lo, hi, left, right)
    while right.first <= 0:
        if result.bracket_expansions >= options.max_bracket_expansions:
            result.status = MinimumStatus.BRACKET_LIMIT
            return result
        hi = (1+hi)/2
        if hi == 1.:
            result.status = MinimumStatus.RESOLUTION_LIMIT
            return result
        right = evaluate(hi)
        if right is None:
            return result
        result.bracket_expansions += 1
        bracket(lo, hi, left, right)
    x, kind = lo+(hi-lo)/2, 'bisection'
    for _ in range(options.max_iterations):
        value = evaluate(x)
        if value is None:
            return result
        result.iterations += 1
        if kind == 'halley':
            result.halley_steps += 1
        else:
            result.bisection_steps += 1
        if value.first < 0:
            lo, left = x, value
        elif value.first > 0:
            hi, right = x, value
        else:
            # A rounded zero does not justify inventing a zero-width bracket.
            a, b = lo+(x-lo)/2, x+(hi-x)/2
            if a == lo and b == hi:
                result.status = MinimumStatus.RESOLUTION_LIMIT
                return result
            va, vb = evaluate(a), evaluate(b)
            if va is None or vb is None:
                return result
            if va.first > 0 or vb.first < 0:
                result.status = MinimumStatus.RESOLUTION_LIMIT
                return result
            lo, hi, left, right = a, b, va, vb
            result.zero_derivative_steps += 1
        bracket(lo, hi, left, right)
        middle = lo+(hi-lo)/2
        if hi-lo <= options.x_atol+options.x_rtol*abs(middle):
            final = evaluate(middle)
            if final is None:
                return result
            result.status = MinimumStatus.OK
            result.x_min, result.time_nondim, result.first = middle, final.time_nondim, final.first
            result.estimated_time_error = (MINIMUM_ROUNDOFF_ULPS*math.ulp(final.time_nondim)
                                          + max(abs(left.time_nondim-final.time_nondim),
                                                abs(right.time_nondim-final.time_nondim)))
            return result
        if middle == lo or middle == hi:
            result.status = MinimumStatus.RESOLUTION_LIMIT
            return result
        proposed = _halley_candidate(x, value)
        if (result.iterations % 4 != 0 and math.isfinite(proposed)
                and lo < proposed < hi and proposed != x
                and abs(proposed-x) <= .75*(hi-lo)):
            x, kind = proposed, 'halley'
        else:
            x, kind = middle, 'bisection'
    result.status = MinimumStatus.ITERATION_LIMIT
    return result


class FeasibilityStatus(IntEnum):
    TWO_SOLUTIONS = 0
    NO_SOLUTION = 1
    NEAR_MINIMUM = 2
    INVALID_INPUT = 3
    MINIMUM_FAILURE = 4


@dataclass(frozen=True)
class FeasibilityResult:
    status: FeasibilityStatus
    margin: float | None = None
    estimated_time_error: float | None = None


def classify_multirev_time(target_time: float, minimum: MinimumResult) -> FeasibilityResult:
    """Classify relative to a precomputed minimum and its estimated uncertainty.

    NEAR_MINIMUM deliberately leaves multiplicity unresolved, including targets
    equal to the reported minimum. No merged root or two velocity roots are
    fabricated. Labels outside that band are numerical estimates, not proofs.
    """
    try:
        if isinstance(target_time, bool):
            raise ValueError('boolean target')
        target = float(target_time)
        if not math.isfinite(target) or target <= 0 or not isinstance(minimum, MinimumResult):
            raise ValueError('invalid target or minimum')
    except (TypeError, ValueError, OverflowError):
        return FeasibilityResult(FeasibilityStatus.INVALID_INPUT)
    if minimum.status != MinimumStatus.OK:
        return FeasibilityResult(FeasibilityStatus.MINIMUM_FAILURE)
    try:
        if (isinstance(minimum.time_nondim, bool) or isinstance(minimum.estimated_time_error, bool)
                or not math.isfinite(minimum.time_nondim) or minimum.time_nondim <= 0
                or not math.isfinite(minimum.estimated_time_error)
                or minimum.estimated_time_error < 0):
            raise ValueError('invalid minimum values')
    except (TypeError, ValueError, OverflowError):
        return FeasibilityResult(FeasibilityStatus.INVALID_INPUT)
    margin, uncertainty = target-minimum.time_nondim, minimum.estimated_time_error
    status = (FeasibilityStatus.TWO_SOLUTIONS if margin > uncertainty else
              FeasibilityStatus.NO_SOLUTION if margin < -uncertainty else
              FeasibilityStatus.NEAR_MINIMUM)
    return FeasibilityResult(status, margin, uncertainty)
