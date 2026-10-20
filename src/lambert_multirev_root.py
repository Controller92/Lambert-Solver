"""Safeguarded M>0 left/right roots, with explicit near-minimum uncertainty.

Branch identity is relative to the stationary point, independent of the sign of
x or the geometric short/long orientation. No velocities or ephemerides here.
Izzo (2015), Eqs. (18), (22), (31), with added branch brackets and safeguards.
"""
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
import math

from .lambert_multirev import (
    FeasibilityStatus, MAX_REVOLUTIONS, MinimumOptions, MinimumResult,
    classify_multirev_time, evaluate_derivatives_multi_rev, evaluate_tof_multi_rev,
    find_minimum_multi_rev,
)
from .lambert_root import RootOptions, RootResult, TOF_ROUNDOFF_ULPS, _householder_step
from .lambert_tof import TofStatus, _resolve_geometry


class RootBranch(str, Enum):
    LEFT = 'left'
    RIGHT = 'right'


class MultiRootStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    BRACKET_LIMIT = 2
    ITERATION_LIMIT = 3
    RESOLUTION_LIMIT = 4
    EVALUATION_FAILURE = 5
    NUMERICAL_RANGE = 6
    MINIMUM_FAILURE = 7
    NO_SOLUTION = 8
    NEAR_MINIMUM = 9
    BRANCH_FAILURE = 10


@dataclass
class MultiRootResult(RootResult):
    """Only OK publishes x/y. Inherited residual/counter fields keep their units.

    tof_evaluations counts actual root-stage calls, including TOFs nested in
    derivatives. Minimum work is reported separately in minimum.tof_evaluations.
    A pair shares one minimum; count it only once when totaling pair work.
    derivative_failures includes any failed final conditioning evaluation.
    estimated_x_error uses twice the slope-based 16-ulp estimate to allow local
    inverse curvature. It is a calibrated diagnostic, not a certified interval.
    """
    status: MultiRootStatus = MultiRootStatus.INVALID_INPUT
    branch: RootBranch | None = None
    revolutions: int | None = None
    minimum: MinimumResult | None = None


def _prepare(target_time, lambda_, revolutions, options, q, minimum_options):
    if (not isinstance(options, RootOptions) or isinstance(target_time, bool)
            or type(revolutions) is not int or not 1 <= revolutions <= MAX_REVOLUTIONS):
        return None
    try:
        target = float(target_time)
        lam, retained, _ = _resolve_geometry(lambda_, q)
        values = (options.x_atol, options.x_rtol, options.tof_atol, options.tof_rtol)
        if any(isinstance(v, bool) for v in values):
            return None
        xa, xr, ta, tr = map(float, values)
        if (not math.isfinite(target) or target <= 0
                or not all(math.isfinite(v) and v >= 0 for v in (xa, xr, ta, tr))
                or xr >= 1 or tr >= 1
                or type(options.max_iterations) is not int or options.max_iterations < 0
                or type(options.max_bracket_expansions) is not int or options.max_bracket_expansions < 0):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    normalized = replace(options, x_atol=xa, x_rtol=xr, tof_atol=ta, tof_rtol=tr)
    minimum = find_minimum_multi_rev(lam, revolutions, q=retained, options=minimum_options)
    return target, lam, retained, normalized, minimum


def _starter(target, revolutions, branch):
    # Eq. (31), evaluated in log space to avoid intermediate ratio overflow.
    logarithm = ((math.log((revolutions+1)*math.pi/8)-math.log(target)) if branch == RootBranch.LEFT
                 else (math.log(target)-math.log(revolutions*math.pi/8)))
    return math.tanh(logarithm/3)


def _solve(prepared, revolutions, branch):
    count = revolutions if type(revolutions) is int and 1 <= revolutions <= MAX_REVOLUTIONS else None
    result = MultiRootResult(branch=branch, revolutions=count)
    if prepared is None:
        return result
    target, lam, q, o, minimum = prepared
    result.minimum = minimum
    feasibility = classify_multirev_time(target, minimum)
    if feasibility.status != FeasibilityStatus.TWO_SOLUTIONS:
        result.status = {FeasibilityStatus.NO_SOLUTION: MultiRootStatus.NO_SOLUTION,
                         FeasibilityStatus.NEAR_MINIMUM: MultiRootStatus.NEAR_MINIMUM}.get(
                             feasibility.status, MultiRootStatus.MINIMUM_FAILURE)
        return result
    decreasing = branch == RootBranch.LEFT

    def evaluate(point):
        result.tof_evaluations += 1
        return evaluate_tof_multi_rev(point, lam, revolutions, q=q)

    def derivative(point):
        result.tof_evaluations += 1
        return evaluate_derivatives_multi_rev(point, lam, revolutions, q=q)

    def record(point, value):
        result.last_x, result.time_nondim = point, value.time_nondim
        result.residual = value.time_nondim-target
        result.relative_residual = result.residual/target

    def residual_ok(f):
        return abs(f) <= o.tof_atol+o.tof_rtol*target

    def finish(status):
        result.status = status
        return result

    def update(point, value):
        nonlocal lo, hi, lower_value, upper_value
        f = value.time_nondim-target
        if f != 0:
            if (f > 0) == decreasing:
                lo, lower_value = point, value
            else:
                hi, upper_value = point, value
        result.lower, result.upper = lo, hi

    # Exclude the entire computed minimum bracket. Its endpoint times must
    # independently lie below the requested time before either root is pursued.
    inner = minimum.lower if decreasing else minimum.upper
    inner_value = evaluate(inner)
    if inner_value.status != TofStatus.OK:
        result.evaluation_status = inner_value.status
        return finish(MultiRootStatus.EVALUATION_FAILURE)
    record(inner, inner_value)
    if inner_value.time_nondim >= target:
        return finish(MultiRootStatus.BRACKET_LIMIT)
    if decreasing:
        lo, hi, lower_value, upper_value = -1., inner, None, inner_value
    else:
        lo, hi, lower_value, upper_value = inner, 1., inner_value, None
    result.lower, result.upper = lo, hi
    guess = _starter(target, revolutions, branch)
    if lo < guess < hi:
        x = guess
    else:
        result.used_fallback_guess = True
        x = lo+(hi-lo)/2
    value = evaluate(x)
    if value.status != TofStatus.OK:
        result.evaluation_status = value.status
        return finish(MultiRootStatus.EVALUATION_FAILURE)
    record(x, value)
    cycle_residual = abs(value.time_nondim-target)

    for iteration in range(1, o.max_iterations+1):
        f = value.time_nondim-target
        # Retain periodic bisection when accelerated steps stop making progress.
        # Forcing it after three rapidly converging steps can throw a nearly
        # solved branch back across a wide bracket, especially near the minimum.
        force_bisection = False
        if iteration % 4 == 0:
            force_bisection = abs(f) >= .25*cycle_residual and not residual_ok(f)
            cycle_residual = abs(f)
        update(x, value)
        middle = lo+(hi-lo)/2
        candidate, kind = None, 'bisection'
        if f == 0:
            candidate, kind = x, 'verification'
        elif middle == lo or middle == hi:
            for endpoint, endpoint_value in ((lo, lower_value), (hi, upper_value)):
                if endpoint_value is not None and residual_ok(endpoint_value.time_nondim-target):
                    candidate, kind = endpoint, 'verification' if endpoint == x else 'bisection'
                    break
            if candidate is None:
                return finish(MultiRootStatus.RESOLUTION_LIMIT)
        elif not force_bisection:
            d = derivative(x)
            if d.status != TofStatus.OK:
                result.derivative_failures += 1
            step = _householder_step(f, d, decreasing=decreasing)
            if step is not None:
                proposed = x-step
                if math.isfinite(proposed) and lo < proposed < hi and abs(step) <= .75*(hi-lo):
                    candidate, kind = proposed, 'householder'
                elif proposed == x and residual_ok(f):
                    candidate, kind = x, 'verification'
            if candidate is None:
                result.rejected_householder_steps += 1
        if candidate is None:
            candidate = middle
        next_value = evaluate(candidate)
        if next_value.status != TofStatus.OK and kind == 'householder':
            result.rejected_householder_steps += 1
            candidate, kind = middle, 'bisection'
            next_value = evaluate(candidate)
        if next_value.status != TofStatus.OK:
            result.evaluation_status = next_value.status
            return finish(MultiRootStatus.EVALUATION_FAILURE)
        result.last_step = candidate-x
        result.iterations += 1
        if kind == 'householder':
            result.householder_steps += 1
        elif kind == 'verification':
            result.verification_steps += 1
        else:
            result.bisection_steps += 1
        x, value = candidate, next_value
        record(x, value)
        update(x, value)
        if abs(result.last_step) <= o.x_atol+o.x_rtol*abs(x) and residual_ok(result.residual):
            d = derivative(x)
            if d.status == TofStatus.OK:
                if d.first == 0 or (d.first < 0) != decreasing:
                    return finish(MultiRootStatus.BRANCH_FAILURE)
                time_error = abs(result.residual)+TOF_ROUNDOFF_ULPS*math.ulp(value.time_nondim)
                estimate = 2*time_error/abs(d.first)
                if math.isfinite(estimate):
                    result.estimated_x_error = estimate
                    result.x_precision_limited = estimate > o.x_atol+o.x_rtol*abs(x)
            else:
                result.derivative_failures += 1
            result.x, result.y = x, value.y
            return finish(MultiRootStatus.OK)
    return finish(MultiRootStatus.ITERATION_LIMIT)


def solve_x_multi_rev(target_time: float, lambda_: float, revolutions: int,
                       branch: RootBranch | str, options: RootOptions = RootOptions(), *,
                       q: float | None = None,
                       minimum_options: MinimumOptions = MinimumOptions()) -> MultiRootResult:
    """Solve one explicitly selected branch for M>=1.

    Both step and time residual stopping tests must pass. Analytic +/-1 limits
    are open and never evaluated. NO_SOLUTION and NEAR_MINIMUM return no roots.
    Bracket expansions in RootOptions are unused: branch domains are finite;
    MinimumOptions separately controls minimum-time search and its expansion.
    """
    try:
        branch = RootBranch(branch)
    except (ValueError, TypeError):
        return MultiRootResult()
    prepared = _prepare(target_time, lambda_, revolutions, options, q, minimum_options)
    return _solve(prepared, revolutions, branch)


def solve_x_multi_rev_pair(target_time: float, lambda_: float, revolutions: int,
                            options: RootOptions = RootOptions(), *, q: float | None = None,
                            minimum_options: MinimumOptions = MinimumOptions()
                            ) -> tuple[MultiRootResult, MultiRootResult]:
    """Return fixed (left,right) results, sharing a single minimum evaluation.

    Inspect each status; a failed branch never suppresses a successful sibling.
    """
    prepared = _prepare(target_time, lambda_, revolutions, options, q, minimum_options)
    return tuple(_solve(prepared, revolutions, branch) for branch in RootBranch)
