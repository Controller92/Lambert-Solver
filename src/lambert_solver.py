"""Complete scalar, binary64, zero-revolution two-body Lambert reference.

Metres, seconds, mu in m^3/s^2; one central-body inertial frame. Direction and
degeneracy conventions are those of LambertInput. No ephemerides or CUDA.
Velocity reconstruction: Izzo (2015), Algorithm 1, published p. 12,
DOI 10.1007/s10569-014-9587-y, using the corrected unit bases from geometry.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import sys

from .lambert_geometry import GeometryResult, GeometryStatus, LambertInput, Vector3, prepare_geometry
from .lambert_root import RootOptions, RootResult, RootStatus, solve_x_zero_rev

# Diagnostic consistency threshold, not a certified velocity error bound.
GEOMETRY_Q_RTOL = 64*sys.float_info.epsilon


class LambertStatus(IntEnum):
    OK = 0
    GEOMETRY_FAILURE = 1
    ROOT_FAILURE = 2
    NUMERICAL_RANGE = 3


@dataclass(frozen=True)
class LambertResult:
    """Velocities exist only on OK; inspect nested statuses on failure.

    root retains actual iterations, nondimensional residual, estimated_x_error
    and the tri-state x_precision_limited flag. OK means root stopping tests
    passed and velocities are representable. It is not a velocity error bound.
    True/None conditioning diagnostics must not be treated as False.
    geometry_precision_warning separately exposes loss when rounded lambda
    cannot accurately retain the geometric identity 1-lambda^2=c/s.
    """
    status: LambertStatus
    geometry: GeometryResult
    root: RootResult | None = None
    v1_m_s: Vector3 | None = None
    v2_m_s: Vector3 | None = None

    @property
    def geometry_q_relative_mismatch(self) -> float | None:
        g = self.geometry.geometry
        if g is None:
            return None
        geometric_q = g.chord_m/g.semiperimeter_m
        return abs((1-g.lambda_)*(1+g.lambda_)-geometric_q)/geometric_q

    @property
    def geometry_precision_warning(self) -> bool | None:
        mismatch = self.geometry_q_relative_mismatch
        return None if mismatch is None else mismatch > GEOMETRY_Q_RTOL


def _velocity_scale(mu, s, radius):
    """sqrt(mu*s/2)/radius without overflowing an intermediate mu*s."""
    mm, me = math.frexp(mu)
    sm, se = math.frexp(s)
    rm, re = math.frexp(radius)
    half, odd = divmod(me+se, 2)
    mantissa = math.sqrt(mm*sm/2)/rm
    if odd:
        mantissa *= math.sqrt(2)
    return math.ldexp(mantissa, half-re)


def solve_zero_rev(problem: LambertInput,
                   options: RootOptions = RootOptions()) -> LambertResult:
    """Prepare geometry, solve x, reconstruct both inertial velocities in m/s.

    Failure returns no partial or substituted velocities. Root residuals and
    step tolerances remain dimensionless, as documented in lambert_root.
    Collinear inputs and nonzero revolution counts are explicitly unsupported.
    """
    geometry = prepare_geometry(problem)
    if geometry.status != GeometryStatus.OK:
        return LambertResult(LambertStatus.GEOMETRY_FAILURE, geometry)
    g = geometry.geometry
    root = solve_x_zero_rev(g.time_nondim, g.lambda_, options)
    if root.status != RootStatus.OK:
        return LambertResult(LambertStatus.ROOT_FAILURE, geometry, root)

    x, y, lam = root.x, root.y, g.lambda_
    try:
        q = (1-lam)*(1+lam)
        # (y-lambda*x)*(y+lambda*x)=1-lambda^2. Rationalize the
        # subtractive member, particularly important for long hyperbolic arcs.
        if lam*x > 0:
            plus = y+lam*x
            minus = q/plus
        else:
            minus = y-lam*x
            plus = q/minus
        # A=lambda*y-x, B=lambda*y+x. The rewritten member avoids
        # first subtracting nearly equal lambda*y and +/-x near |lambda|=1.
        if lam*x > 0:
            a, b = math.fsum((lam*minus, -q*x)), lam*y+x
        else:
            a, b = lam*y-x, math.fsum((lam*plus, q*x))
        scale1 = _velocity_scale(float(problem.mu_m3_s2), g.semiperimeter_m, g.r1_norm_m)
        scale2 = _velocity_scale(float(problem.mu_m3_s2), g.semiperimeter_m, g.r2_norm_m)
        radial1 = scale1*math.fsum((a, -g.rho*b))
        radial2 = -scale2*math.fsum((a, g.rho*b))
        transverse = g.sigma*plus
        vt1, vt2 = scale1*transverse, scale2*transverse
        v1 = tuple(math.fsum((radial1*r, vt1*t))
                   for r, t in zip(g.radial1, g.transverse1))
        v2 = tuple(math.fsum((radial2*r, vt2*t))
                   for r, t in zip(g.radial2, g.transverse2))
        if (not all(math.isfinite(v) for v in (*v1, *v2))
                or scale1 <= 0 or scale2 <= 0 or vt1 <= 0 or vt2 <= 0):
            return LambertResult(LambertStatus.NUMERICAL_RANGE, geometry, root)
    except (OverflowError, ValueError, ZeroDivisionError):
        return LambertResult(LambertStatus.NUMERICAL_RANGE, geometry, root)
    return LambertResult(LambertStatus.OK, geometry, root, v1, v2)
