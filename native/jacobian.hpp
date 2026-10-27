#pragma once
#include "multirev.hpp"

namespace lambert {
// Separate optional result: the ordinary velocity ABI and kernel are unchanged.
// Rows [v1xyz,v2xyz], columns [r1xyz,r2xyz,tof], fixed mu and discrete branch.
struct Jacobian {
    int32_t status = 1, reserved = 0; // 0 valid, 1 solve failed, 2 ill-conditioned, 3 arithmetic
    double root_condition = missing, slope_ratio = missing, estimated_relative_slope_error = missing;
    double values[6][7];
    LB_HD Jacobian() {
        for (int a = 0; a < 6; ++a)
            for (int i = 0; i < 7; ++i) values[a][i] = missing;
    }
};

namespace sensitivity {
// Forward differentials only through smooth geometry/reconstruction. Root
// iterations, stopping rules and discrete choices are never differentiated.
struct Dual {
    double v, d[7];
    LB_HD Dual(double value = 0.) : v(value), d{} {}
};
LB_HD Dual variable(double v, int i) { Dual a(v); a.d[i] = 1.; return a; }
LB_HD Dual primal(Dual a, double v) { a.v = v; return a; }
LB_HD Dual operator+(Dual a, Dual b) {
    Dual c(sum2(a.v, b.v));
    for (int i = 0; i < 7; ++i) c.d[i] = sum2(a.d[i], b.d[i]);
    return c;
}
LB_HD Dual operator-(Dual a, Dual b) {
    Dual c(a.v - b.v);
    for (int i = 0; i < 7; ++i) c.d[i] = a.d[i] - b.d[i];
    return c;
}
LB_HD Dual operator-(Dual a) { return Dual(0.) - a; }
LB_HD Dual operator*(Dual a, Dual b) {
    Dual c(a.v * b.v);
    for (int i = 0; i < 7; ++i) c.d[i] = sum2(a.d[i]*b.v, a.v*b.d[i]);
    return c;
}
LB_HD Dual operator/(Dual a, Dual b) {
    Dual c(a.v / b.v);
    for (int i = 0; i < 7; ++i) c.d[i] = (a.d[i] - c.v*b.d[i]) / b.v;
    return c;
}
LB_HD Dual sqrt(Dual a) {
    Dual c(::sqrt(a.v));
    for (int i = 0; i < 7; ++i) c.d[i] = .5*a.d[i]/c.v;
    return c;
}
struct Vector { Dual x, y, z; };
LB_HD Vector operator+(Vector a, Vector b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
LB_HD Vector operator-(Vector a, Vector b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
LB_HD Vector operator*(Vector a, Dual b) { return {a.x*b, a.y*b, a.z*b}; }
LB_HD Vector operator/(Vector a, Dual b) { return {a.x/b, a.y/b, a.z/b}; }
LB_HD Vector primal(Vector a, Vec3 b) {
    return {primal(a.x,b.x), primal(a.y,b.y), primal(a.z,b.z)};
}
LB_HD Dual dot(Vector a, Vector b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
LB_HD Dual norm(Vector a) {
    Dual c(lambert::norm(Vec3{a.x.v,a.y.v,a.z.v}));
    for (int i = 0; i < 7; ++i)
        c.d[i] = sum3((a.x.v/c.v)*a.x.d[i], (a.y.v/c.v)*a.y.d[i], (a.z.v/c.v)*a.z.d[i]);
    return c;
}
LB_HD Vector cross(Vector a, Vector b) {
    Vector c{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
    c.x.v = difference_products(a.y.v,b.z.v,a.z.v,b.y.v);
    c.y.v = difference_products(a.z.v,b.x.v,a.x.v,b.z.v);
    c.z.v = difference_products(a.x.v,b.y.v,a.y.v,b.x.v);
    return c;
}
} // namespace sensitivity

LB_HD Jacobian terminal_jacobian(const Input &p, const Result &r) {
    Jacobian out;
    if (r.status) return out;
    out.status = 3;
    const Geometry g = prepare(p, p.revolutions != 0);
    if (g.status) return out;
    const Shape shape_ = shape(g.lambda,g.q);
    const Tof t = p.revolutions ? multi_tof(r.root.x,shape_,p.revolutions) : tof(r.root.x,shape_);
    const Derivatives dt = p.revolutions ? multi_derivatives(r.root.x,shape_,p.revolutions,t)
                                         : derivatives(r.root.x,shape_,t);
    if (dt.status) return out;
    out.root_condition = ::fabs(t.time/dt.first);
    out.slope_ratio = ::fabs(dt.first)/max2(t.time,::fabs(dt.second));
    // A fold makes dx/dz singular. Also refuse a slope whose uncertainty from
    // the root error estimate is material. Thresholds are diagnostics, not a
    // bound on every entry of the terminal Jacobian.
    double error = 32.*eps*max2(1.,::fabs(r.root.x));
    if (finite(r.root.estimated_x_error)) error = max2(error,r.root.estimated_x_error);
    out.estimated_relative_slope_error = ::fabs(dt.second/dt.first)*error;
    if (!finite(out.root_condition) || out.slope_ratio <= 64.*::sqrt(eps) ||
        out.estimated_relative_slope_error >= 1e-4) {
        out.status = 2;
        return out;
    }
    using namespace sensitivity;
    // Seed normalized coordinates to avoid SI-scale overflow in differentials.
    // L and tau are fixed units for this evaluation, not additional variables.
    const double L = g.s, tau = (L/::sqrt(p.mu))*::sqrt(L/2.);
    const double V = velocity_scale(p.mu,L,L)*2.;
    Vector a{variable(p.r1.x/L,0),variable(p.r1.y/L,1),variable(p.r1.z/L,2)};
    Vector b{variable(p.r2.x/L,3),variable(p.r2.y/L,4),variable(p.r2.z/L,5)};
    Vector chord = primal(a-b, divide(sub(p.r1,p.r2),L));
    Dual R1 = primal(norm(a),g.r1/L), R2 = primal(norm(b),g.r2/L), C = norm(chord);
    Dual S = primal((R1+R2+C)*.5,1.);
    Vector radial1 = primal(a/R1,g.radial1), radial2 = primal(b/R2,g.radial2);
    Vector h = cross(radial1,radial2);
    const Vec3 pa = scaled(p.r1), pb = scaled(p.r2);
    const Vec3 ph{difference_products(pa.y,pb.z,pa.z,pb.y),
                  difference_products(pa.z,pb.x,pa.x,pb.z),
                  difference_products(pa.x,pb.y,pa.y,pb.x)};
    h = primal(h,divide(divide(ph,lambert::norm(pa)),lambert::norm(pb)));
    Dual sine = norm(h);
    Vector normal = h/(sine*(p.long_way ? -1. : 1.));
    Dual ch, sh;
    if (lambert::dot(g.radial1,g.radial2) >= 0.) {
        ch = norm(radial1+radial2)*.5; sh = sine/(ch*2.);
    } else {
        sh = norm(radial1-radial2)*.5; ch = sine/(sh*2.);
    }
    Dual lambda = primal(sqrt(R1/S)*sqrt(R2/S)*ch*(p.long_way ? -1. : 1.),g.lambda);
    Dual q = primal(C/S,g.q);
    // q carries tiny-arc shape information even when lambda rounds to +/-1.
    // Enforce the mathematical differential dq=-2 lambda dlambda using the
    // well-conditioned coordinate in each region; preserve both primal values.
    if (::fabs(g.lambda) > .5)
        for (int i = 0; i < 7; ++i) lambda.d[i] = -q.d[i]/(2.*g.lambda);
    else
        for (int i = 0; i < 7; ++i) q.d[i] = -2.*g.lambda*lambda.d[i];
    Dual rho = primal(dot(chord/C,a+b)/(R1+R2),g.rho);
    Dual sigma = primal(sqrt(R1/C)*sqrt(R2/C)*sh*2.,g.sigma);
    Vector transverse1 = primal(cross(normal,radial1),g.transverse1);
    Vector transverse2 = primal(cross(normal,radial2),g.transverse2);
    Dual target = variable(p.tof/tau,6)/(S*sqrt(S));
    Dual x(r.root.x);
    for (int i = 0; i < 7; ++i) {
        const double shape_partial = ::fabs(g.lambda) > .5 ?
            (g.lambda/t.y)*q.d[i] : (-2.*g.lambda*g.lambda/t.y)*lambda.d[i];
        x.d[i] = (target.d[i]-shape_partial)/dt.first;
    }
    // y=hypot(sqrt(q),lambda*x), differentiated without squaring huge x.
    Dual lx = lambda*x, y(r.root.y);
    for (int i = 0; i < 7; ++i) y.d[i] = .5*q.d[i]/y.v + (lx.v/y.v)*lx.d[i];
    Dual minus = lx.v > 0. ? q/(y+lx) : y-lx;
    Dual plus = lx.v > 0. ? y+lx : q/minus;
    Dual A = lx.v > 0. ? lambda*minus-q*x : lambda*y-x;
    Dual B = lx.v > 0. ? lambda*y+x : lambda*plus+q*x;
    Dual scale1 = sqrt(S)/R1*.5, scale2 = sqrt(S)/R2*.5;
    Dual vr1 = scale1*(A-rho*B), vr2 = -scale2*(A+rho*B);
    Dual vt1 = scale1*sigma*plus, vt2 = scale2*sigma*plus;
    Vector v1, v2;
    Dual anomaly_argument = (Dual(1.)-x)*(Dual(1.)+x)*minus*minus;
    if (p.revolutions == 0 && ::fabs(anomaly_argument.v) < .1 &&
        x.v*y.v+lambda.v*(1.-x.v*x.v) > 0. && ::fabs(minus.v) < .01) {
        // Local short arcs: differentiating a normalized plane creates large
        // terms that cancel like eps/angle^2 in arbitrary orientations.
        // Use the Lagrange f,g identity instead. g = dt - Kepler's anomaly
        // term; its small-anomaly hypergeometric series is regular at x=1.
        // H = 2F1(1/2,3/2;5/2;u), u=(1-x^2)*eta^2.
        Dual H(1.), term(1.);
        for (int n = 1; n <= 64; ++n) {
            term = term*anomaly_argument*((n-.5)*(n+.5)/(n*(n+1.5)));
            H = H+term;
            if (::fabs(term.v) < eps*::fabs(H.v)) break;
        }
        Dual elapsed = variable(p.tof/tau,6);
        Dual kepler = S*sqrt(S)*minus*minus*minus*H*(2./3.);
        Dual inverse_g = Dual(1.)/(elapsed-kepler);
        Dual radial = S*minus*minus;
        Vector delta = chord*Dual(-1.);
        v1 = (delta+radial1*radial)*inverse_g;
        v2 = (delta-radial2*radial)*inverse_g;
    } else {
        v1 = radial1*vr1+transverse1*vt1;
        v2 = radial2*vr2+transverse2*vt2;
    }
    const Dual components[6] = {v1.x,v1.y,v1.z,v2.x,v2.y,v2.z};
    double values[6][7];
    for (int k = 0; k < 6; ++k)
        for (int i = 0; i < 7; ++i) {
            values[k][i] = components[k].d[i]*(V/(i == 6 ? tau : L));
            if (!finite(values[k][i])) return out;
        }
    for (int k = 0; k < 6; ++k)
        for (int i = 0; i < 7; ++i) out.values[k][i] = values[k][i];
    out.status = 0;
    return out;
}
} // namespace lambert
