"""Safeguarded scalar M=0 inversion of T(x, lambda), with no velocity output.

All inputs, residuals and tolerances are dimensionless. Physical units and arc
orientation remain those of lambert_geometry. Reference: Izzo (2015), section
4.1, DOI 10.1007/s10569-014-9587-y. Bracketing safeguards are added here.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import sys

from .lambert_root_preparation import evaluate_derivatives_zero_rev, initial_guess_zero_rev
from .lambert_tof import TofStatus, _resolve_geometry, evaluate_tof_zero_rev

# Diagnostic roundoff allowance, calibrated against the independently verified
# forward evaluator. Its slope-based error estimate is not an interval proof.
TOF_ROUNDOFF_ULPS = 16


class RootStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    BRACKET_LIMIT = 2
    ITERATION_LIMIT = 3
    RESOLUTION_LIMIT = 4
    EVALUATION_FAILURE = 5
    NUMERICAL_RANGE = 6


@dataclass(frozen=True)
class RootOptions:
    x_atol: float = 1e-14
    x_rtol: float = 1e-14
    tof_atol: float = 0.0
    tof_rtol: float = 5e-13
    max_iterations: int = 128
    max_bracket_expansions: int = 1024


@dataclass
class RootResult:
    """x/y are populated only on OK. Other values are diagnostic on failure.

    iterations counts completed x updates, including a zero step verified by a
    fresh TOF evaluation. Their kinds sum to iterations. TOF counts include the
    two starter evaluations and one in each derivative evaluation. lower=-1 is
    an open analytic limit, never evaluated. State is local to one call.
    """
    status: RootStatus = RootStatus.INVALID_INPUT
    x: float | None = None
    y: float | None = None
    last_x: float | None = None
    time_nondim: float | None = None
    residual: float | None = None
    relative_residual: float | None = None
    last_step: float | None = None
    lower: float | None = None
    upper: float | None = None
    iterations: int = 0
    householder_steps: int = 0
    bisection_steps: int = 0
    verification_steps: int = 0
    rejected_householder_steps: int = 0
    derivative_failures: int = 0
    bracket_expansions: int = 0
    tof_evaluations: int = 0
    used_fallback_guess: bool = False
    evaluation_status: TofStatus | None = None
    estimated_x_error: float | None = None
    x_precision_limited: bool | None = None


def _householder_step(f, derivatives, *, decreasing=True):
    """Algebraically scaled form of the paper's update, avoiding f'^2/f'^3."""
    if (derivatives.status != TofStatus.OK
            or (derivatives.first >= 0 if decreasing else derivatives.first <= 0)):
        return None
    try:
        u = f/derivatives.first
        a = u*(derivatives.second/derivatives.first)
        b = (u*(derivatives.third/derivatives.first))*u/6
        denominator = math.fsum((1., -a, b))
        if abs(denominator) <= 64*sys.float_info.epsilon*max(1., abs(a), abs(b)):
            return None
        step = u*(1-a/2)/denominator
        return step if math.isfinite(step) else None
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def solve_x_zero_rev(target_time: float, lambda_: float,
                     options: RootOptions = RootOptions(), *,
                     q: float | None = None) -> RootResult:
    """Solve M=0 for positive finite target T and nondegenerate shape.

    Optional q=c/s follows evaluate_tof_zero_rev's contract and is retained
    through every forward, derivative, starter and diagnostic evaluation.

    Require BOTH |dx|<=x_atol+x_rtol*|x| and
    |T(x)-target|<=tof_atol+tof_rtol*target. No velocities, ephemerides, GPU
    operations or hidden substitute roots. Derivative failures may safely use
    bisection; forward failures are explicit if bisection also cannot evaluate.
    """
    result = RootResult()
    if (not isinstance(options, RootOptions) or isinstance(target_time, bool)
            or isinstance(lambda_, bool)):
        return result
    if any(isinstance(v, bool) for v in (options.x_atol, options.x_rtol,
                                        options.tof_atol, options.tof_rtol)):
        return result
    try:
        target = float(target_time)
        lam, _, _ = _resolve_geometry(lambda_, q)
        xa, xr, ta, tr = (float(v) for v in (options.x_atol, options.x_rtol,
                                             options.tof_atol, options.tof_rtol))
    except (TypeError, ValueError, OverflowError):
        return result
    if (not math.isfinite(target) or target <= 0
            or not all(math.isfinite(v) and v >= 0 for v in (xa, xr, ta, tr))
            or xr >= 1 or tr >= 1
            or type(options.max_iterations) is not int or options.max_iterations < 0
            or type(options.max_bracket_expansions) is not int or options.max_bracket_expansions < 0):
        return result

    def evaluate(point):
        result.tof_evaluations += 1
        return evaluate_tof_zero_rev(point, lam, q=q)

    def record(point, value):
        result.last_x, result.time_nondim = point, value.time_nondim
        result.residual = value.time_nondim-target
        result.relative_residual = result.residual/target

    def residual_ok(f):
        return abs(f) <= ta+tr*target

    def finish(status):
        result.status = status
        return result

    guess = initial_guess_zero_rev(target, lam, q=q)
    result.tof_evaluations += 2
    t0 = evaluate(0.)
    if t0.status != TofStatus.OK:
        result.evaluation_status = t0.status
        return finish(RootStatus.EVALUATION_FAILURE)
    record(0., t0)
    lower_value = None
    if target >= t0.time_nondim:
        # T tends to +infinity at the excluded -1 endpoint.
        lo, hi, upper_value = -1., 0., t0
    else:
        lo, lower_value = 0., t0
        hi = max(1., guess.x0) if guess.status == TofStatus.OK else 1.
        upper_value = evaluate(hi)
        result.lower, result.upper = lo, hi
        while True:
            if upper_value.status != TofStatus.OK:
                result.evaluation_status = upper_value.status
                return finish(RootStatus.EVALUATION_FAILURE)
            record(hi, upper_value)
            if upper_value.time_nondim <= target:
                break
            if result.bracket_expansions >= options.max_bracket_expansions:
                return finish(RootStatus.BRACKET_LIMIT)
            lo, lower_value = hi, upper_value
            hi = 2*hi+1
            result.lower, result.upper = lo, hi
            if not math.isfinite(hi):
                return finish(RootStatus.NUMERICAL_RANGE)
            result.bracket_expansions += 1
            upper_value = evaluate(hi)
    result.lower, result.upper = lo, hi
    if guess.status == TofStatus.OK and lo <= guess.x0 <= hi:
        x = guess.x0
    else:
        result.used_fallback_guess = True
        x = lo+(hi-lo)/2
    value = t0 if x == 0 else upper_value if x == hi else evaluate(x)
    if value.status != TofStatus.OK:
        result.evaluation_status = value.status
        return finish(RootStatus.EVALUATION_FAILURE)
    record(x, value)

    for iteration in range(1, options.max_iterations+1):
        f = value.time_nondim-target
        if f > 0:
            lo, lower_value = x, value
        elif f < 0:
            hi, upper_value = x, value
        result.lower, result.upper = lo, hi
        middle = lo+(hi-lo)/2
        candidate, kind = None, 'bisection'
        if f == 0:
            # An actual zero update, independently re-evaluated below.
            candidate, kind = x, 'verification'
        elif middle == lo or middle == hi:
            # Exhausted binary64 x values. Try a qualifying endpoint, otherwise
            # report unattainable residual tolerance, even when dx would be zero.
            for endpoint, endpoint_value in ((lo, lower_value), (hi, upper_value)):
                if endpoint_value is not None and residual_ok(endpoint_value.time_nondim-target):
                    candidate, kind = endpoint, 'verification' if endpoint == x else 'bisection'
                    break
            if candidate is None:
                return finish(RootStatus.RESOLUTION_LIMIT)
        elif iteration % 4 != 0 or residual_ok(f):
            derivatives = evaluate_derivatives_zero_rev(x, lam, q=q)
            result.tof_evaluations += 1
            if derivatives.status != TofStatus.OK:
                result.derivative_failures += 1
            step = _householder_step(f, derivatives)
            if step is not None:
                proposed = x-step
                if (math.isfinite(proposed) and lo < proposed < hi and abs(step) <= 0.75*(hi-lo)):
                    candidate, kind = proposed, 'householder'
                elif proposed == x and residual_ok(f):
                    candidate, kind = x, 'verification'
            if candidate is None:
                result.rejected_householder_steps += 1
        if candidate is None:
            candidate = middle
        new_value = evaluate(candidate)
        if new_value.status != TofStatus.OK and kind == 'householder':
            result.rejected_householder_steps += 1
            candidate, kind = middle, 'bisection'
            new_value = evaluate(candidate)
        if new_value.status != TofStatus.OK:
            result.evaluation_status = new_value.status
            return finish(RootStatus.EVALUATION_FAILURE)
        result.last_step = candidate-x
        result.iterations += 1
        if kind == 'householder':
            result.householder_steps += 1
        elif kind == 'verification':
            result.verification_steps += 1
        else:
            result.bisection_steps += 1
        x, value = candidate, new_value
        record(x, value)
        if result.residual > 0:
            lo, lower_value = x, value
        elif result.residual < 0:
            hi, upper_value = x, value
        result.lower, result.upper = lo, hi
        if abs(result.last_step) <= xa+xr*abs(x) and residual_ok(result.residual):
            result.x, result.y = x, value.y
            # A small step is not a forward x-error certificate on flat curves.
            # Unknown diagnostics stay None when the combined derivative API
            # cannot represent its higher derivatives, even if bisection worked.
            final_derivatives = evaluate_derivatives_zero_rev(x, lam, q=q)
            result.tof_evaluations += 1
            if final_derivatives.status == TofStatus.OK and final_derivatives.first < 0:
                time_error = abs(result.residual)+TOF_ROUNDOFF_ULPS*math.ulp(value.time_nondim)
                result.estimated_x_error = time_error/abs(final_derivatives.first)
                result.x_precision_limited = result.estimated_x_error > xa+xr*abs(x)
            return finish(RootStatus.OK)
    return finish(RootStatus.ITERATION_LIMIT)
