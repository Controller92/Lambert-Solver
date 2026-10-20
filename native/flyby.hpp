#pragma once
// Independent encounter ABI; no changes to the Lambert numerical/ABI contract.
#include "lambert.hpp"
namespace flyby {
using namespace lambert;
enum Status { SUCCESS=0, INVALID_INPUT=1, ZERO_SPEED=2, BAD_BASIS=3, SPEED_MISMATCH=4,
              TURN_OUT_OF_RANGE=5, DEGENERATE_TURN=6, NO_CONVERGENCE=7, NUMERICAL_RANGE=8 };
struct Input {
    Vec3 vin, vout, reference;
    double mu, rp_min, rp_max, rp, azimuth, speed_atol, speed_rtol;
    int32_t mode, reserved=0;
};
struct Result {
    int32_t status=INVALID_INPUT, iterations=0, precision_limited=-1, reserved=0;
    Vec3 vout{missing,missing,missing}, b{missing,missing,missing}, rperi{missing,missing,missing};
    Vec3 vperi_in{missing,missing,missing}, vperi_out{missing,missing,missing};
    double rp=missing, angle=missing, bmag=missing, speed_in=missing, target_speed=missing;
    double speed_error=missing, delta_v=missing, angle_residual=missing;
    double lower=missing, upper=missing, azimuth=missing;
};
LB_HD double dot3(Vec3 a,Vec3 b) { const double x[3]={a.x*b.x,a.y*b.y,a.z*b.z}; return small_sum(x); }
LB_HD Vec3 unit3(Vec3 v) { return divide(v,norm(v)); }

LB_HD double det2(double a,double b,double c,double d) {
    // Explicit FMA retains the product-rounding error during cancellation;
    // compiler contraction remains disabled everywhere else.
    double p=c*d;
    return ::fma(a,b,-p)-::fma(c,d,-p);
}
LB_HD Vec3 precise_cross(Vec3 a,Vec3 b) {
    return {det2(a.y,b.z,a.z,b.y),det2(a.z,b.x,a.x,b.z),det2(a.x,b.y,a.y,b.x)};
}
LB_HD Vec3 scale_exact(Vec3 a) {
    int exponent=0;
    ::frexp(max2(::fabs(a.x),max2(::fabs(a.y),::fabs(a.z))),&exponent);
    return {::ldexp(a.x,-exponent),::ldexp(a.y,-exponent),::ldexp(a.z,-exponent)};
}
struct Half { double s,c,u; bool valid; };
LB_HD Half half(double rp,double v,double mu) {
    double k=(v/::sqrt(mu))*::sqrt(rp), u=k*k;
    if (!lambert::finite(u)||u<=0) return {missing,missing,u,false};
    if (u<=1) return {1/(1+u), k*::sqrt(u+2)/(1+u),u,true};
    return {(1/u)/(1+1/u),::sqrt(1+2/u)/(1+1/u),u,true};
}
LB_HD double turn(double rp,double vi,double vo,double mu,bool complement=false) {
    auto a=half(rp,vi,mu), b=half(rp,vo,mu);
    return a.valid&&b.valid?(complement?::atan2(a.c,a.s)+::atan2(b.c,b.s):
                            ::atan2(a.s,a.c)+::atan2(b.s,b.c)):missing;
}
LB_HD Result solve(const Input &p) {
    Result o;
    if (p.mode<0||p.mode>2||p.reserved!=0||!lambert::finite(p.vin)||!lambert::finite(p.vout)||!lambert::finite(p.reference)
        ||!lambert::finite(p.mu)||p.mu<=0||!lambert::finite(p.rp_min)||p.rp_min<=0||!lambert::finite(p.rp_max)||p.rp_max<p.rp_min
        ||!lambert::finite(p.rp)||!lambert::finite(p.azimuth)||!lambert::finite(p.speed_atol)||p.speed_atol<0
        ||!lambert::finite(p.speed_rtol)||p.speed_rtol<0) return o;
    double vi=norm(p.vin), vo=p.mode==0?vi:norm(p.vout);
    if (vi==0||vo==0) {o.status=ZERO_SPEED; return o;}
    if (!lambert::finite(vi)||!lambert::finite(vo)) {o.status=NUMERICAL_RANGE; return o;}
    Vec3 s=unit3(p.vin);
    double refn=norm(p.reference);
    if (!lambert::finite(refn)||refn==0) {o.status=BAD_BASIS; return o;}
    Vec3 ref=divide(p.reference,refn), t=sub(ref,mul(s,dot3(ref,s)));
    if (norm(t)<=1e-12) {o.status=BAD_BASIS;return o;}
    t=unit3(t); Vec3 r=unit3(cross(s,t)); t=cross(r,s);
    o.speed_in=vi; o.target_speed=vo; o.speed_error=vo-vi;
    double required=0, rp=p.rp, sh=1, ch=1; Vec3 bdir{};
    if (p.mode==0) {
        if (rp<p.rp_min||rp>p.rp_max) {o.status=TURN_OUT_OF_RANGE;return o;}
        bdir=add(mul(t,::cos(p.azimuth)),mul(r,::sin(p.azimuth)));
    } else {
        Vec3 a=scale_exact(p.vin), b=scale_exact(p.vout), h=precise_cross(a,b);
        double cosine=dot3(a,b);
        required=::atan2(norm(h),cosine); o.angle=required;
        if (p.mode==1&&::fabs(vo-vi)>p.speed_atol+p.speed_rtol*max2(vi,vo)) {o.status=SPEED_MISMATCH;return o;}
        if (cosine>=0) { sh=::sin(required/2); ch=::cos(required/2); }
        else { double complement_angle=::atan2(norm(h),-cosine);
               sh=::cos(complement_angle/2); ch=::sin(complement_angle/2); }
        if (sh==0) {o.status=DEGENERATE_TURN;return o;}
        if (ch==0||norm(h)==0) {o.status=TURN_OUT_OF_RANGE;return o;}
        bdir=unit3(cross(s,unit3(h)));
        if (p.mode==1) {
            rp=(p.mu/vi)/vi*(ch*ch/(sh*(1+sh)));
            if (!lambert::finite(rp)) {o.status=NUMERICAL_RANGE;return o;}
            if (rp<p.rp_min*(1-2e-14)||rp>p.rp_max*(1+2e-14)) {o.status=TURN_OUT_OF_RANGE;return o;}
            rp=max2(p.rp_min,min2(p.rp_max,rp));
            o.lower=o.upper=rp;
        } else {
            bool complement=ch<sh;
            double metric=complement?2*::atan2(ch,sh):required;
            double lo=p.rp_min, hi=p.rp_max;
            double tl=turn(lo,vi,vo,p.mu,complement), th=turn(hi,vi,vo,p.mu,complement);
            if (!lambert::finite(tl)||!lambert::finite(th)) {o.status=NUMERICAL_RANGE;return o;}
            if (metric<min2(tl,th)*(1-2e-14)||metric>max2(tl,th)*(1+2e-14)) {o.status=TURN_OUT_OF_RANGE;return o;}
            double residual=missing;
            for (int i=1;i<=160;++i) {
                rp=max2(lo,min2(hi,::sqrt(lo)*::sqrt(hi))); residual=turn(rp,vi,vo,p.mu,complement)-metric; o.iterations=i;
                if (!lambert::finite(residual)) {o.status=NUMERICAL_RANGE;return o;}
                if ((residual>0)!=complement) lo=rp; else hi=rp;
                if (hi-lo<=2e-13*rp) break;
            }
            o.lower=lo; o.upper=hi;
            if (::fabs(residual)>3e-13*metric||hi-lo>3e-13*rp) {o.status=NO_CONVERGENCE;return o;}
        }
    }
    auto a=half(rp,vi,p.mu), b=p.mode==2?half(rp,vo,p.mu):a;
    if (!a.valid||!b.valid) {o.status=NUMERICAL_RANGE;return o;}
    double cosd=a.c*b.c-a.s*b.s, sind=a.s*b.c+a.c*b.s;
    double escape=(::sqrt(p.mu)/::sqrt(rp))*::sqrt(2.), vpi=::hypot(vi,escape), vpo=::hypot(p.mode==2?vo:vi,escape);
    o.rp=rp; o.angle=::atan2(sind,cosd); o.bmag=rp*::hypot(1.,::sqrt(2.)/::sqrt(a.u));
    Vec3 radial=add(mul(s,a.s),mul(bdir,a.c)), tangent=sub(mul(s,a.c),mul(bdir,a.s));
    o.vout=mul(sub(mul(s,cosd),mul(bdir,sind)),p.mode==2?vo:vi);
    o.b=mul(bdir,o.bmag); o.rperi=mul(radial,rp); o.vperi_in=mul(tangent,vpi); o.vperi_out=mul(tangent,vpo);
    double scale=max2(vpi,vpo);
    o.delta_v=p.mode==2?::fabs(vo-vi)*((vo/scale+vi/scale)/(vpi/scale+vpo/scale)):0;
    o.angle_residual=p.mode?o.angle-required:0;
    o.azimuth=::atan2(dot3(bdir,r),dot3(bdir,t));
    if (!lambert::finite(o.vout)||!lambert::finite(o.b)||!lambert::finite(o.rperi)||!lambert::finite(o.vperi_in)||!lambert::finite(o.vperi_out)
        ||!lambert::finite(o.rp)||!lambert::finite(o.angle)||!lambert::finite(o.bmag)||!lambert::finite(o.delta_v)||!lambert::finite(o.angle_residual)) {
        Result bad; bad.status=NUMERICAL_RANGE; return bad;
    }
    o.precision_limited=p.mode&&32*eps/min2(sh,ch)>1e-8?1:0;
    o.status=SUCCESS; return o;
}
} // namespace flyby
