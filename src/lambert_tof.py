"""Forward Izzo T(x, lambda) for zero complete revolutions, binary64 CPU math.

x and lambda are dimensionless; T = sqrt(2*mu/s^3)*elapsed_seconds, matching
lambert_geometry. lambda's sign retains the chosen short/long arc convention.
This function evaluates a supplied x; it neither finds x nor returns velocities.
Reference: Izzo (2015), DOI 10.1007/s10569-014-9587-y, Eqs. (17)-(21).
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import sys

# Over this interval |S1| <= 0.44 for |lambda| < 1, safely within series radius 1.
BATTIN_X_MIN = 0.8
BATTIN_X_MAX = 1.2
SERIES_REL_TOL = sys.float_info.epsilon
MAX_SERIES_TERMS = 128


class TofStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    NUMERICAL_RANGE = 2
    SERIES_NONCONVERGENCE = 3


class TofMethod(IntEnum):
    ELLIPTIC = 0
    BATTIN = 1
    PARABOLIC = 2
    HYPERBOLIC = 3


@dataclass(frozen=True)
class TofResult:
    """Failure has no partial T or y. series_terms counts nonconstant terms.

    OK indicates successful forward evaluation, not root convergence. No root
    iteration count or residual is invented; neither exists at this layer.
    """

    status: TofStatus
    time_nondim: float | None = None
    y: float | None = None
    method: TofMethod | None = None
    series_terms: int = 0


def _gauss_series(z: float) -> tuple[float | None, int]:
    """2F1(3,1;5/2;z), compensated sum with a bound on the remaining tail.

    t_0=1, t_(n+1)=t_n*z*(n+3)/(n+5/2). The ratio magnitudes decrease to
    |z|. After term n, its next ratio bounds all subsequent ratios, so the
    remaining absolute tail is <= |t_n|*ratio/(1-ratio).
    """
    total, term, correction = 1.0, 1.0, 0.0
    for n in range(1, MAX_SERIES_TERMS + 1):
        term *= z * (n+2) / (n+1.5)
        adjusted = term - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
        ratio = abs(z) * (n+3) / (n+2.5)
        tail_bound = abs(term) * ratio / (1.0-ratio)
        if tail_bound <= SERIES_REL_TOL * abs(total):
            return total, n
    return None, MAX_SERIES_TERMS


def evaluate_tof_zero_rev(x: float, lambda_: float) -> TofResult:
    """Evaluate for finite x > -1 and -1 < lambda < 1; M is fixed to zero.

    x=-1 has infinite TOF and |lambda|=1 is degenerate; neither is admitted.
    Invalid values return INVALID_INPUT. Intermediate overflow/underflow or
    a nonpositive/nonfinite result returns NUMERICAL_RANGE. A bounded series
    that exhausts its term budget returns SERIES_NONCONVERGENCE. No fallback
    value is substituted. All calculations are scalar with fixed-size state.
    """
    if isinstance(x, bool) or isinstance(lambda_, bool):
        return TofResult(TofStatus.INVALID_INPUT)
    try:
        x, lam = float(x), float(lambda_)
    except (TypeError, ValueError, OverflowError):
        return TofResult(TofStatus.INVALID_INPUT)
    if not (math.isfinite(x) and x > -1.0 and math.isfinite(lam) and abs(lam) < 1.0):
        return TofResult(TofStatus.INVALID_INPUT)

    # Factor 1-lambda^2 to retain accuracy near either geometry limit.
    q = (1.0-lam)*(1.0+lam)
    y = math.hypot(math.sqrt(q), lam*x)
    terms = 0
    try:
        if x == 1.0:
            # Eq. (21), factored so 1-lambda^3 does not cancel near lambda=1.
            T = (2.0/3.0) * (1.0-lam) * (1.0+lam+lam*lam)
            y, method = 1.0, TofMethod.PARABOLIC
        else:
            # eta=y-lambda*x. Rationalize only when that subtraction cancels.
            eta = q/(y+lam*x) if lam*x > 0.0 else y-lam*x
            if not (math.isfinite(eta) and eta > 0.0):
                return TofResult(TofStatus.NUMERICAL_RANGE)
            if BATTIN_X_MIN <= x <= BATTIN_X_MAX:
                z = math.fsum((1.0-lam, -x*eta))/2.0
                F, terms = _gauss_series(z)
                if F is None:
                    return TofResult(TofStatus.SERIES_NONCONVERGENCE, series_terms=terms)
                T = math.fsum(((2.0/3.0)*eta*eta*eta*F, 2.0*lam*eta))
                method = TofMethod.BATTIN
            elif x < 1.0:
                d = (1.0-x)*(1.0+x)
                k = math.sqrt(d)
                # Eq. (17): sin(psi)=eta*k, cos(psi)=lambda+x*eta.
                # atan2 retains the quadrant and avoids acos near +/-1.
                psi = math.atan2(eta*k, math.fsum((lam, x*eta)))
                # lambda*y-x = lambda*eta-q*x, without subtracting near equals.
                T = math.fsum((psi/k, lam*eta, -q*x))/d
                method = TofMethod.ELLIPTIC
            else:
                k = math.sqrt(x-1.0)*math.sqrt(x+1.0)
                # Eq. (17): sinh(psi)=eta*k, no subtractive acosh argument.
                psi = math.asinh(eta*k)
                T = math.fsum((q*x, -lam*eta, -psi/k))/k/k
                method = TofMethod.HYPERBOLIC
    except (OverflowError, ValueError, ZeroDivisionError):
        return TofResult(TofStatus.NUMERICAL_RANGE)
    if not (math.isfinite(y) and math.isfinite(T) and T > 0.0):
        return TofResult(TofStatus.NUMERICAL_RANGE)
    return TofResult(TofStatus.OK, T, y, method, terms)
