"""CPU double-precision input contract and Izzo geometry.

Positions: metres, relative to one central body in one right-handed inertial
frame. mu: that body's positive gravitational parameter in m^3/s^2. tof:
positive elapsed dynamical seconds (e.g. TDB interval), not a UTC date difference.
The caller handles ephemerides, frame/origin changes, and time conversions.
Only zero complete revolutions are supported. long_way chooses the arc above
pi radians; False chooses the arc below pi. Neither flag means prograde relative
to a fixed external axis. No low/high multi-revolution branch exists here.

Reference: Izzo (2015), DOI 10.1007/s10569-014-9587-y, sections 2.2 and 4,
published pp. 4-5 and Algorithm 1 (with corrected, normalized vector bases).
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import sys

Vector3 = tuple[float, float, float]
# Reject ill-conditioned plane orientation at either collinear limit.
COLLINEAR_SIN_TOL = 64.0 * sys.float_info.epsilon


class GeometryStatus(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    UNSUPPORTED_REVOLUTIONS = 2
    DEGENERATE_GEOMETRY = 3
    NUMERICAL_RANGE = 4


@dataclass(frozen=True)
class LambertInput:
    """One two-body boundary problem; vectors must have exactly three elements."""

    r1_m: Vector3
    r2_m: Vector3
    tof_s: float
    mu_m3_s2: float
    long_way: bool = False
    revolutions: int = 0


@dataclass(frozen=True)
class LambertGeometry:
    r1_norm_m: float
    r2_norm_m: float
    chord_m: float
    semiperimeter_m: float
    lambda_: float
    q: float  # c/s = 1-lambda^2, retained independently to resolve small chords
    time_scale_s: float
    time_nondim: float
    rho: float
    sigma: float
    transfer_angle_rad: float
    radial1: Vector3
    radial2: Vector3
    normal: Vector3
    transverse1: Vector3
    transverse2: Vector3


@dataclass(frozen=True)
class GeometryResult:
    """OK means geometry prepared, not a converged Lambert solution.

    Failure has geometry=None, never replacement velocities. iterations is zero
    and residual is None because no time-of-flight equation has been evaluated.
    """

    status: GeometryStatus
    geometry: LambertGeometry | None = None
    iterations: int = 0
    residual: float | None = None


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _product_error(a: float, b: float, product: float) -> float:
    """Dekker product remainder for the bounded, power-of-two-scaled inputs.

    Python 3.12 has no math.fma. Keeping the remainder prevents near-parallel
    cross products from losing the transfer plane in rounded multiplications.
    """
    splitter = 134217729.0  # 2^27+1 for binary64
    split_a, split_b = splitter*a, splitter*b
    high_a = split_a-(split_a-a)
    high_b = split_b-(split_b-b)
    low_a, low_b = a-high_a, b-high_b
    return ((high_a*high_b-product)+high_a*low_b+low_a*high_b)+low_a*low_b


def _difference_of_products(a: float, b: float, c: float, d: float) -> float:
    ab, cd = a*b, c*d
    return math.fsum((ab, -cd, _product_error(a, b, ab), -_product_error(c, d, cd)))


def _scaled_vector(vector: Vector3) -> Vector3:
    """Exact binary scaling, avoiding overflow and new normalization rounding."""
    exponent = math.frexp(max(abs(v) for v in vector))[1]
    return tuple(math.ldexp(v, -exponent) for v in vector)


def _plane(a: Vector3, b: Vector3) -> tuple[Vector3, float]:
    a, b = _scaled_vector(a), _scaled_vector(b)
    cross = (_difference_of_products(a[1], b[2], a[2], b[1]),
             _difference_of_products(a[2], b[0], a[0], b[2]),
             _difference_of_products(a[0], b[1], a[1], b[0]))
    magnitude = math.hypot(*cross)
    return cross, magnitude/(math.hypot(*a)*math.hypot(*b))


def _vector(value) -> Vector3:
    values = tuple(float(v) for v in value)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("expected three finite components")
    return values


def prepare_geometry(problem: LambertInput) -> GeometryResult:
    """Pure scalar reference. No I/O, device state, ephemerides, or root solving.

    Zero radius, invalid shapes/scalars/flags: INVALID_INPUT. Nonzero integer
    revolutions: UNSUPPORTED_REVOLUTIONS. Coincident, parallel, antiparallel or
    sin(angle) <= COLLINEAR_SIN_TOL: DEGENERATE_GEOMETRY. Nonrepresentable
    derived positive lengths/time scales: NUMERICAL_RANGE. Uses binary64 Python
    floats; this fixed-size reference makes no GPU throughput claim.
    """
    if not isinstance(problem, LambertInput):
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    if (type(problem.long_way) is not bool
            or type(problem.revolutions) is not int
            or problem.revolutions < 0
            or isinstance(problem.tof_s, bool)
            or isinstance(problem.mu_m3_s2, bool)):
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    try:
        r1, r2 = _vector(problem.r1_m), _vector(problem.r2_m)
        tof, mu = float(problem.tof_s), float(problem.mu_m3_s2)
    except (TypeError, ValueError, OverflowError):
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    if not (math.isfinite(tof) and tof > 0 and math.isfinite(mu) and mu > 0):
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    r1n, r2n = math.hypot(*r1), math.hypot(*r2)
    if r1n == 0 or r2n == 0:
        return GeometryResult(GeometryStatus.INVALID_INPUT)
    if problem.revolutions:
        return GeometryResult(GeometryStatus.UNSUPPORTED_REVOLUTIONS)
    c = math.hypot(*(b-a for a, b in zip(r1, r2)))
    s = 0.5*r1n + 0.5*r2n + 0.5*c
    if not all(math.isfinite(v) for v in (r1n, r2n, c, s)):
        return GeometryResult(GeometryStatus.NUMERICAL_RANGE)
    u1 = tuple(v/r1n for v in r1)
    u2 = tuple(v/r2n for v in r2)
    cross, sin_angle = _plane(r1, r2)
    if c == 0 or sin_angle <= COLLINEAR_SIN_TOL:
        return GeometryResult(GeometryStatus.DEGENERATE_GEOMETRY)
    if r1n/s == 0 or r2n/s == 0:
        return GeometryResult(GeometryStatus.NUMERICAL_RANGE)

    sign = -1.0 if problem.long_way else 1.0
    cross_norm = math.hypot(*cross)
    normal = tuple(sign*v/cross_norm for v in cross)
    cosine = math.fsum(a*b for a, b in zip(u1, u2))
    # Use the well-resolved half-angle norm, then sin(theta)=2*sin(h)*cos(h).
    # Subtracting normalized radials would lose tiny angles in a general frame.
    if cosine >= 0:
        cos_half = math.hypot(*(a+b for a, b in zip(u1, u2)))/2
        sin_half = sin_angle/(2*cos_half)
    else:
        sin_half = math.hypot(*(a-b for a, b in zip(u1, u2)))/2
        cos_half = sin_angle/(2*sin_half)
    lam = math.sqrt(r1n/s)*math.sqrt(r2n/s)*cos_half
    # (r1-r2)/c = dot((R1-R2)/c, (R1+R2)/(r1+r2)). This avoids
    # subtracting rounded norms, which amplifies error by 1/c on tiny chords.
    radius_scale = max(r1n, r2n)
    rho = math.fsum(((a-b)/c)*(a/radius_scale+b/radius_scale)
                    for a, b in zip(r1, r2))/(r1n/radius_scale+r2n/radius_scale)
    sigma = 2*math.sqrt(r1n/c)*math.sqrt(r2n/c)*sin_half
    # Only bound roundoff in quantities geometrically in [0,1] or [-1,1].
    lam, sigma = min(1.0, lam), min(1.0, sigma)
    rho = max(-1.0, min(1.0, rho))
    # sqrt(s^3/(2*mu)) without forming s^3, which overflows unnecessarily.
    time_scale = (s / math.sqrt(mu)) * math.sqrt(s / 2.0)
    if not math.isfinite(time_scale) or time_scale <= 0:
        return GeometryResult(GeometryStatus.NUMERICAL_RANGE)
    T = tof / time_scale
    if not math.isfinite(T) or T <= 0:
        return GeometryResult(GeometryStatus.NUMERICAL_RANGE)
    short_angle = math.atan2(sin_angle, cosine)
    angle = 2.0*math.pi - short_angle if problem.long_way else short_angle
    return GeometryResult(GeometryStatus.OK, LambertGeometry(
        r1n, r2n, c, s, sign*lam, c/s, time_scale, T, rho, sigma, angle,
        u1, u2, normal, _cross(normal, u1), _cross(normal, u2)))
