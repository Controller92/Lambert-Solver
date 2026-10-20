"""Planet-centred, two-body hyperbolic encounters; SI units, radians.

No ephemeris access here. Subtract the body's inertial velocity before calling,
then add it back afterwards. B points from the body to the incoming asymptote.
T is the normalized projection of `reference` into the incoming B plane;
R = S cross T; B/rB = cos(azimuth) T + sin(azimuth) R.

POWERED means a tangential impulse at one common periapsis. It is a restricted
physical model, not a general minimum-fuel powered flyby optimizer.
"""
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral, Real
import math


class Mode(IntEnum):
    FORWARD = 0
    MATCH = 1
    POWERED = 2


class Status(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    ZERO_SPEED = 2
    BAD_BASIS = 3
    SPEED_MISMATCH = 4
    TURN_OUT_OF_RANGE = 5
    DEGENERATE_TURN = 6
    NO_CONVERGENCE = 7
    NUMERICAL_RANGE = 8


NAN3 = (math.nan,)*3


@dataclass(frozen=True)
class Encounter:
    vin: tuple
    mu: float
    rp_min: float
    rp_max: float
    mode: Mode = Mode.FORWARD
    vout: tuple = (0., 0., 0.)
    reference: tuple = (0., 0., 1.)
    rp: float = 0.
    azimuth: float = 0.
    speed_atol: float = 1e-6
    speed_rtol: float = 1e-12


@dataclass
class Result:
    status: Status = Status.INVALID_INPUT
    iterations: int = 0
    precision_limited: int = -1
    vout: tuple = NAN3
    b: tuple = NAN3
    rperi: tuple = NAN3
    vperi_in: tuple = NAN3
    vperi_out: tuple = NAN3
    rp: float = math.nan
    angle: float = math.nan
    bmag: float = math.nan
    speed_in: float = math.nan
    target_speed: float = math.nan
    speed_error: float = math.nan
    delta_v: float = math.nan
    angle_residual: float = math.nan
    lower: float = math.nan
    upper: float = math.nan
    azimuth: float = math.nan


def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(a,k): return tuple(x*k for x in a)
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.hypot(*a)
def unit(a):
    n=norm(a)
    return tuple(x/n for x in a)


def exact_product(a,b):
    # Dekker split: inputs here are power-of-two scaled to magnitude <= 1.
    p=a*b
    ta=134217729.*a; tb=134217729.*b
    ah=ta-(ta-a); al=a-ah; bh=tb-(tb-b); bl=b-bh
    return p,((ah*bh-p)+ah*bl+al*bh)+al*bl


def cross_precise(a,b):
    def det(x,y,z,w):
        p,e=exact_product(x,y);q,f=exact_product(z,w)
        return math.fsum((p,-q,e,-f))
    return (det(a[1],b[2],a[2],b[1]),det(a[2],b[0],a[0],b[2]),det(a[0],b[1],a[1],b[0]))


def scale_exact(v):
    exponent=math.frexp(max(abs(x) for x in v))[1]
    return tuple(math.ldexp(x,-exponent) for x in v)


def half_turn(rp,v,mu):
    # k = sqrt(rp v² / mu) avoids unnecessary overflow from v².
    k = (v/math.sqrt(mu))*math.sqrt(rp)
    u = k*k
    if not math.isfinite(u) or u <= 0: raise ArithmeticError('Dimensionless encounter outside binary64 range')
    if u <= 1:
        s = 1/(1+u)
        c = k*math.sqrt(u+2)/(1+u)
    else:
        s = (1/u)/(1+1/u)
        c = math.sqrt(1+2/u)/(1+1/u)
    return s,c,u


def solve(p: Encounter) -> Result:
    """Return explicit per-encounter status; failed velocities are NaN.

    rp_min/max are inclusive centre distances, not altitudes. MATCH accepts
    the caller's explicit speed tolerance; output retains the incoming speed.
    angle_residual is model turn minus requested turn (zero for FORWARD).
    Parallel/near-parallel B-plane references are rejected, never replaced.
    """
    result = Result()
    if not isinstance(p,Encounter): raise TypeError('Expected Encounter')
    vectors = (p.vin,p.vout,p.reference)
    if any(len(v)!=3 for v in vectors): raise ValueError('Expected three-component vectors')
    values=[x for v in vectors for x in v]+[p.mu,p.rp_min,p.rp_max,p.rp,p.azimuth,p.speed_atol,p.speed_rtol]
    if any(isinstance(x,bool) or not isinstance(x,Real) for x in values): raise TypeError('Expected real numeric components and scalars')
    if isinstance(p.mode,bool) or not isinstance(p.mode,Integral): raise TypeError('Expected an integer Mode')
    if (p.mode not in (Mode.FORWARD,Mode.MATCH,Mode.POWERED)
        or not all(math.isfinite(x) for v in vectors for x in v)
        or not all(math.isfinite(x) for x in (p.mu,p.rp_min,p.rp_max,p.rp,p.azimuth,p.speed_atol,p.speed_rtol))
        or not 0 < p.rp_min <= p.rp_max or p.mu<=0 or p.speed_atol<0 or p.speed_rtol<0):
        return result
    try:
        vi = norm(p.vin)
        if vi==0: result.status=Status.ZERO_SPEED; return result
        if not math.isfinite(vi): raise ArithmeticError()
        s = unit(p.vin)
        refnorm = norm(p.reference)
        if not refnorm or not math.isfinite(refnorm): result.status=Status.BAD_BASIS; return result
        ref = tuple(x/refnorm for x in p.reference)
        t = add(ref,mul(s,-dot(ref,s)))
        if norm(t)<=1e-12: result.status=Status.BAD_BASIS; return result
        t = unit(t); r = unit(cross(s,t)); t = cross(r,s)
        vo = vi if p.mode==Mode.FORWARD else norm(p.vout)
        if vo==0: result.status=Status.ZERO_SPEED; return result
        if not math.isfinite(vo): raise ArithmeticError()
        result.speed_in=vi; result.target_speed=vo; result.speed_error=vo-vi
        required = 0.
        if p.mode==Mode.FORWARD:
            rp=p.rp
            if not p.rp_min<=rp<=p.rp_max: result.status=Status.TURN_OUT_OF_RANGE; return result
            bdir=add(mul(t,math.cos(p.azimuth)),mul(r,math.sin(p.azimuth)))
        else:
            a,b=scale_exact(p.vin),scale_exact(p.vout)
            h = cross_precise(a,b)
            cosine=dot(a,b)
            required = math.atan2(norm(h),cosine)
            result.angle=required
            if p.mode==Mode.MATCH and abs(vo-vi)>p.speed_atol+p.speed_rtol*max(vi,vo):
                result.status=Status.SPEED_MISMATCH; return result
            if cosine>=0:
                sh=math.sin(required/2); ch=math.cos(required/2)
            else:
                complement_angle=math.atan2(norm(h),-cosine)
                sh=math.cos(complement_angle/2); ch=math.sin(complement_angle/2)
            if sh==0: result.status=Status.DEGENERATE_TURN; return result
            if ch==0 or norm(h)==0: result.status=Status.TURN_OUT_OF_RANGE; return result
            bdir=unit(cross(s,unit(h)))
            if p.mode==Mode.MATCH:
                # No subtraction 1/sin(a)-1 near a=pi/2.
                rp=(p.mu/vi)/vi * (ch*ch/(sh*(1+sh)))
                if not math.isfinite(rp): raise ArithmeticError()
                if rp<p.rp_min*(1-2e-14) or rp>p.rp_max*(1+2e-14):
                    result.status=Status.TURN_OUT_OF_RANGE; return result
                rp=max(p.rp_min,min(p.rp_max,rp))
                result.lower=result.upper=rp
            else:
                complement = ch < sh
                metric = 2*math.atan2(ch,sh) if complement else required
                def turn(radius):
                    si,ci,_=half_turn(radius,vi,p.mu); so,co,_=half_turn(radius,vo,p.mu)
                    return math.atan2(ci,si)+math.atan2(co,so) if complement else math.atan2(si,ci)+math.atan2(so,co)
                lo,hi=p.rp_min,p.rp_max
                low_turn,high_turn=turn(lo),turn(hi)
                if metric<min(low_turn,high_turn)*(1-2e-14) or metric>max(low_turn,high_turn)*(1+2e-14):
                    result.status=Status.TURN_OUT_OF_RANGE; return result
                rp=lo
                for iteration in range(1,161):
                    rp=max(lo,min(hi,math.sqrt(lo)*math.sqrt(hi)))
                    residual=turn(rp)-metric
                    result.iterations=iteration
                    if (residual>0) != complement: lo=rp
                    else: hi=rp
                    if hi-lo<=2e-13*rp: break
                result.lower=lo; result.upper=hi
                if abs(turn(rp)-metric)>3e-13*metric or hi-lo>3e-13*rp:
                    result.status=Status.NO_CONVERGENCE; return result
        si,ci,ui=half_turn(rp,vi,p.mu)
        so,co,_=half_turn(rp,vo,p.mu) if p.mode==Mode.POWERED else (si,ci,ui)
        cosd=ci*co-si*so; sind=si*co+ci*so
        angle=math.atan2(sind,cosd)
        bmag=rp*math.hypot(1.,math.sqrt(2.)/math.sqrt(ui))
        escape=(math.sqrt(p.mu)/math.sqrt(rp))*math.sqrt(2.)
        vpi=math.hypot(vi,escape); vpo=math.hypot(vo if p.mode==Mode.POWERED else vi,escape)
        radial=add(mul(s,si),mul(bdir,ci)); tangent=add(mul(s,ci),mul(bdir,-si))
        result.vout=mul(add(mul(s,cosd),mul(bdir,-sind)),vo if p.mode==Mode.POWERED else vi)
        result.b=mul(bdir,bmag); result.rperi=mul(radial,rp)
        result.vperi_in=mul(tangent,vpi); result.vperi_out=mul(tangent,vpo)
        result.rp=rp; result.angle=angle; result.bmag=bmag
        scale=max(vpi,vpo)
        result.delta_v=abs(vo-vi)*((vo/scale+vi/scale)/(vpi/scale+vpo/scale)) if p.mode==Mode.POWERED else 0.
        result.angle_residual=angle-required if p.mode!=Mode.FORWARD else 0.
        result.azimuth=math.atan2(dot(bdir,r),dot(bdir,t))
        if not all(math.isfinite(x) for v in (result.vout,result.b,result.rperi,result.vperi_in,result.vperi_out) for x in v):
            raise ArithmeticError()
        if not all(math.isfinite(x) for x in (result.rp,result.angle,result.bmag,result.delta_v,result.angle_residual)):
            raise ArithmeticError()
        result.precision_limited=int(p.mode!=Mode.FORWARD and 32*math.ulp(1.)/min(sh,ch)>1e-8)
        result.status=Status.SUCCESS
        return result
    except (ArithmeticError,OverflowError,ValueError):
        return Result(status=Status.NUMERICAL_RANGE)
