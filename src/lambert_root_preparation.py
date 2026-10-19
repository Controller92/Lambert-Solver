"""M=0 TOF derivatives and initial guesses, without any root iteration.

All inputs are dimensionless and retain lambert_geometry/lambert_tof conventions.
Derivatives hold lambda fixed. Reference: Izzo (2015), Eqs. (22)-(23), section 4.2.1,
DOI 10.1007/s10569-014-9587-y. See docs/third-increment.md for derivations.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

from .lambert_tof import SERIES_REL_TOL, TofStatus, evaluate_tof_zero_rev

# Higher derivatives amplify recurrence cancellation beyond the TOF series window.
DERIVATIVE_X_MIN = 0.7
DERIVATIVE_X_MAX = 1.3
MAX_DERIVATIVE_TERMS = 256


class DerivativeMethod(IntEnum):
    RECURRENCE = 0
    SERIES = 1
    PARABOLIC = 2


@dataclass(frozen=True)
class DerivativeResult:
    status: TofStatus
    time_nondim: float | None = None
    y: float | None = None
    first: float | None = None
    second: float | None = None
    third: float | None = None
    method: DerivativeMethod | None = None
    series_terms: int = 0


class GuessRegion(IntEnum):
    LONG_TIME = 0
    INTERMEDIATE = 1
    SHORT_TIME = 2


@dataclass(frozen=True)
class InitialGuessResult:
    status: TofStatus
    x0: float | None = None
    T0: float | None = None
    T1: float | None = None
    region: GuessRegion | None = None


def _odd_complement(lam: float, power: int) -> float:
    """1-lambda^odd_power without subtracting near equals at positive lambda=1."""
    return -math.expm1(power*math.log(lam)) if lam > 0 else 1.0+(-lam)**power


def _derivative_series(x: float, lam: float) -> tuple[tuple[float, float, float] | None, int]:
    """Differentiate T=sum a_n*d^n, d=1-x^2, in the positive-x local region.

    a_n=2*c_n*(1-lambda^(2*n+3))/(2*n+3), c_n=binomial(2*n,n)/4^n.
    The three sums are derivatives with respect to d; then apply the chain rule.
    """
    d, q, l2 = (1-x)*(1+x), (1-lam)*(1+lam), lam*lam
    complement = (1-lam)*(1+lam+l2)
    lam_power, coefficient = lam*l2, 1.0
    s1 = s2 = s3 = e1 = e2 = e3 = 0.0
    p1 = p2 = p3 = 1.0
    for n in range(1, MAX_DERIVATIVE_TERMS + 1):
        complement = math.fsum((complement, lam_power*q))
        lam_power *= l2
        coefficient *= (n-0.5)/n
        a = 2*coefficient*complement/(2*n+3)
        t1 = n*a*p1
        v = t1-e1
        updated = s1+v
        e1, s1 = (updated-s1)-v, updated
        p1 *= d
        if n >= 2:
            t2 = n*(n-1)*a*p2
            v = t2-e2
            updated = s2+v
            e2, s2 = (updated-s2)-v, updated
            p2 *= d
        if n >= 3:
            t3 = n*(n-1)*(n-2)*a*p3
            v = t3-e3
            updated = s3+v
            e3, s3 = (updated-s3)-v, updated
            p3 *= d
            # Bound all three remaining tails by the largest next term ratio.
            ratio = abs(d)*(n+0.5)/(n-2)
            if ratio < 1:
                bound = ratio/(1-ratio)
                if (abs(t1)*bound <= SERIES_REL_TOL*abs(s1)
                        and abs(t2)*bound <= SERIES_REL_TOL*abs(s2)
                        and abs(t3)*bound <= SERIES_REL_TOL*abs(s3)):
                    return (-2*x*s1, math.fsum((4*x*x*s2, -2*s1)),
                            math.fsum((-8*x*x*x*s3, 12*x*s2))), n
    return None, MAX_DERIVATIVE_TERMS


def evaluate_derivatives_zero_rev(x: float, lambda_: float) -> DerivativeResult:
    """Return T, y, and first three x derivatives; never perform a root step.

    Reuses TofStatus. Failure has no partial values. NUMERICAL_RANGE includes
    derivative underflow even if T alone remains representable. series_terms
    counts a_n terms through n (n=0 is constant and does not affect derivatives).
    """
    tof = evaluate_tof_zero_rev(x, lambda_)
    if tof.status != TofStatus.OK:
        return DerivativeResult(tof.status)
    x, lam = float(x), float(lambda_)
    T, y = tof.time_nondim, tof.y
    terms = 0
    try:
        if x == 1:
            a1 = _odd_complement(lam, 5)/5
            a2 = (3/28)*_odd_complement(lam, 7)
            a3 = (5/72)*_odd_complement(lam, 9)
            derivatives = (-2*a1, math.fsum((8*a2, -2*a1)), math.fsum((24*a2, -48*a3)))
            method = DerivativeMethod.PARABOLIC
        elif DERIVATIVE_X_MIN <= x <= DERIVATIVE_X_MAX:
            derivatives, terms = _derivative_series(x, lam)
            if derivatives is None:
                return DerivativeResult(TofStatus.SERIES_NONCONVERGENCE, series_terms=terms)
            method = DerivativeMethod.SERIES
        else:
            q = (1-lam)*(1+lam)
            eta = q/(y+lam*x) if lam*x > 0 else y-lam*x
            d = (1-x)*(1+x)
            # -2+2*lambda^3*x/y = -2*(q+lambda^2*eta/y), avoiding cancellation.
            first = math.fsum((3*T*x, -2*q, -2*lam*lam*(eta/y)))/d
            ratio = lam/y
            second = math.fsum((3*T, 5*x*first, 2*q*ratio**3))/d
            third = math.fsum((7*x*second, 8*first, -6*q*ratio**5*x))/d
            derivatives = first, second, third
            method = DerivativeMethod.RECURRENCE
    except (OverflowError, ValueError, ZeroDivisionError):
        return DerivativeResult(TofStatus.NUMERICAL_RANGE)
    if not all(math.isfinite(v) and v != 0 for v in derivatives):
        return DerivativeResult(TofStatus.NUMERICAL_RANGE)
    return DerivativeResult(TofStatus.OK, T, y, *derivatives, method, terms)


def _log_ratio(a: float, b: float) -> float:
    ratio = a/b
    return math.log1p((a-b)/b) if 0.5 <= ratio <= 2.0 else math.log(a)-math.log(b)


def initial_guess_zero_rev(target_time: float, lambda_: float) -> InitialGuessResult:
    """Paper's M=0 starter with the middle exponent rederived from log-time lines.

    T0=T(0), T1=T(1). target_time must be positive and finite. x0 is only an
    approximation. NUMERICAL_RANGE includes an x0 rounded to the excluded -1
    endpoint; no clipping or substitute guess is returned.
    """
    if isinstance(target_time, bool):
        return InitialGuessResult(TofStatus.INVALID_INPUT)
    try:
        target = float(target_time)
    except (TypeError, ValueError, OverflowError):
        return InitialGuessResult(TofStatus.INVALID_INPUT)
    if not (math.isfinite(target) and target > 0):
        return InitialGuessResult(TofStatus.INVALID_INPUT)
    t0, t1 = evaluate_tof_zero_rev(0., lambda_), evaluate_tof_zero_rev(1., lambda_)
    if t0.status != TofStatus.OK or t1.status != TofStatus.OK:
        return InitialGuessResult(t0.status if t0.status != TofStatus.OK else t1.status)
    T0, T1, lam = t0.time_nondim, t1.time_nondim, float(lambda_)
    try:
        if target >= T0:
            x0 = math.expm1((2/3)*_log_ratio(T0, target))
            region = GuessRegion.LONG_TIME
        elif target <= T1:
            # Cancel (1-lambda) analytically in T1 / ((2/5)*(1-lambda^5)).
            l2 = lam*lam
            scale = (5/3)*(1+lam+l2)/(1+lam+l2+lam*l2+l2*l2)
            x0 = 1+scale*((T1-target)/target)
            region = GuessRegion.SHORT_TIME
        else:
            exponent = _log_ratio(T0, target)*math.log(2)/_log_ratio(T0, T1)
            x0 = math.expm1(exponent)
            region = GuessRegion.INTERMEDIATE
    except (OverflowError, ValueError, ZeroDivisionError):
        return InitialGuessResult(TofStatus.NUMERICAL_RANGE)
    if not (math.isfinite(x0) and x0 > -1):
        return InitialGuessResult(TofStatus.NUMERICAL_RANGE)
    return InitialGuessResult(TofStatus.OK, x0, T0, T1, region)
