#pragma once
#include "lambert.hpp"

namespace lambert {
// M>0 elliptic formula. Retained subnormal q and genuine zero derivatives are
// allowed. No M=0 near-parabolic series is needed because M*pi stays positive.
LB_HD Tof multi_tof(double x, Shape s, int32_t m) {
    Tof r;
    if (!s.valid || m < 1 || !finite(x) || x <= -1. || x >= 1.)
        return r;
    r.status = 2;
    double d = (1. - x) * (1. + x), k = ::sqrt(d);
    double y = ::hypot(::sqrt(s.q), s.lambda * x), e = eta(x, s, y);
    double psi = ::atan2(e * k, sum2(s.lambda, x * e));
    double time = sum3((psi + double(m) * pi) / k, s.lambda * e, -s.q * x) / d;
    if (!finite(time) || time <= 0. || !finite(y) || y <= 0.)
        return r;
    r.status = 0;
    r.time = time;
    r.y = y;
    r.method = 0;
    return r;
}
LB_HD Derivatives multi_derivatives(double x, Shape s, int32_t m, Tof t) {
    Derivatives r;
    r.status = t.status;
    if (t.status)
        return r;
    r.status = 2;
    double d = (1. - x) * (1. + x), e = eta(x, s, t.y);
    double w = ::sqrt(s.q) / t.y, ratio = s.lambda / t.y, a = w * ratio * s.lambda;
    double q3 = (w * a) * s.lambda, q5x = a * (a * ((s.lambda * x) / t.y));
    double first = x == 0. ? -2. : sum3(3. * t.time * x, -2. * s.q,
                                       -2. * s.lambda * s.lambda * (e / t.y)) / d;
    double second = sum3(3. * t.time, 5. * x * first, 2. * q3) / d;
    double third = x == 0. ? -16. : sum3(7. * x * second, 8. * first, -6. * q5x) / d;
    if (!finite(first) || !finite(second) || !finite(third))
        return r;
    r.status = 0;
    r.time = t.time;
    r.y = t.y;
    r.first = first;
    r.second = second;
    r.third = third;
    r.method = 0;
    (void)m;
    return r;
}
LB_HD Derivatives minimum_evaluate(Minimum &r, double x, Shape s, int32_t m) {
    ++r.evaluations;
    Derivatives d = multi_derivatives(x, s, m, multi_tof(x, s, m));
    if (d.status) {
        r.status = 2;
        r.evaluation_status = d.status;
    }
    return d;
}
LB_HD bool valid(MinimumOptions o) {
    return finite(o.x_atol) && o.x_atol >= 0. && finite(o.x_rtol) && o.x_rtol >= 0. &&
           o.x_rtol < 1. && o.max_iterations >= 0 && o.max_bracket_expansions >= 0;
}
LB_HD Minimum find_minimum(Shape s, int32_t m, MinimumOptions o = MinimumOptions{}) {
    Minimum r;
    r.status = 1;
    if (!s.valid || m < 1 || !valid(o))
        return r;
    double lo = 0., hi = .5;
    Derivatives left = minimum_evaluate(r, lo, s, m), right = minimum_evaluate(r, hi, s, m);
    if (left.status || right.status)
        return r;
    for (;;) {
        r.lower = lo; r.upper = hi; r.first_lower = left.first; r.first_upper = right.first;
        if (right.first > 0.) break;
        if (r.expansions >= o.max_bracket_expansions) { r.status = 3; return r; }
        hi = (1. + hi) / 2.;
        if (hi == 1.) { r.status = 5; return r; }
        right = minimum_evaluate(r, hi, s, m);
        if (right.status) return r;
        ++r.expansions;
    }
    double x = lo + (hi - lo) / 2.;
    bool halley = false;
    for (int64_t iteration = 0; iteration < o.max_iterations; ++iteration) {
        Derivatives value = minimum_evaluate(r, x, s, m);
        if (value.status) return r;
        ++r.iterations;
        if (halley) ++r.halley; else ++r.bisection;
        if (value.first < 0.) { lo = x; left = value; }
        else if (value.first > 0.) { hi = x; right = value; }
        else {
            double a = lo + (x - lo) / 2., b = x + (hi - x) / 2.;
            if (a == lo && b == hi) { r.status = 5; return r; }
            Derivatives va = minimum_evaluate(r, a, s, m), vb = minimum_evaluate(r, b, s, m);
            if (va.status || vb.status) return r;
            // A rounded zero is not an exact stationary point. A quarter
            // probe can cross the computed sign change by a few ulps. Keep
            // the existing opposite-sign endpoint rather than failing early.
            if (va.first > 0.) { hi = a; right = va; }
            else if (vb.first < 0.) { lo = b; left = vb; }
            else { lo = a; hi = b; left = va; right = vb; }
            ++r.zero_derivative;
        }
        r.lower = lo; r.upper = hi; r.first_lower = left.first; r.first_upper = right.first;
        double middle = lo + (hi - lo) / 2.;
        if (hi - lo <= o.x_atol + o.x_rtol * ::fabs(middle)) {
            Derivatives final = minimum_evaluate(r, middle, s, m);
            if (final.status) return r;
            r.status = 0; r.x = middle; r.time = final.time; r.first = final.first;
            r.estimated_time_error = 16. * ulp(final.time) +
                max2(::fabs(left.time - final.time), ::fabs(right.time - final.time));
            return r;
        }
        if (middle == lo || middle == hi) { r.status = 5; return r; }
        double u = value.first / value.second, correction = .5 * u * (value.third / value.second);
        double denominator = 1. - correction;
        double proposed = ::fabs(denominator) <= 64. * eps * max2(1., ::fabs(correction)) ?
                          missing : x - u / denominator;
        halley = r.iterations % 4 != 0 && finite(proposed) && lo < proposed && proposed < hi &&
                 proposed != x && ::fabs(proposed - x) <= .75 * (hi - lo);
        x = halley ? proposed : middle;
    }
    r.status = 4;
    return r;
}
LB_HD Tof counted_multi(Root &r, double x, Shape s, int32_t m) {
    ++r.evaluations;
    return multi_tof(x, s, m);
}
LB_HD void multi_update(Root &r, double x, Tof t, double target, bool decreasing,
                       double &lo, double &hi, Tof &lower, Tof &upper) {
    double f = t.time - target;
    if (f != 0.) {
        if ((f > 0.) == decreasing) { lo = x; lower = t; }
        else { hi = x; upper = t; }
    }
    r.lower = lo; r.upper = hi;
}
// Root-stage derivatives reuse the owned TOF; counters report executed native
// evaluations rather than copying Python's intentionally repeated TOF calls.
LB_HD Root multi_root(double target, Shape s, int32_t m, int32_t branch, Options o,
                      const Minimum &minimum) {
    Root r;
    if (!s.valid || m < 1 || branch < 0 || branch > 1 || !finite(target) || target <= 0. || !valid(o))
        return r;
    if (minimum.status) { r.status = 7; return r; }
    double margin = target - minimum.time;
    if (margin <= minimum.estimated_time_error) {
        r.status = margin < -minimum.estimated_time_error ? 8 : 9;
        return r;
    }
    bool decreasing = branch == 0;
    double inner = decreasing ? minimum.lower : minimum.upper;
    Tof inside = counted_multi(r, inner, s, m), lower, upper;
    if (inside.status) { r.status = 5; r.evaluation_status = inside.status; return r; }
    record(r, inner, inside, target);
    if (inside.time >= target) { r.status = 2; return r; }
    double lo = decreasing ? -1. : inner, hi = decreasing ? inner : 1.;
    if (decreasing) upper = inside; else lower = inside;
    r.lower = lo; r.upper = hi;
    // Widen m before m+1 so INT32_MAX never overflows.
    double logarithm = decreasing ? ::log((double(m) + 1.) * pi / 8.) - ::log(target) :
                                    ::log(target) - ::log(double(m) * pi / 8.);
    double x = ::tanh(logarithm / 3.);
    if (!(lo < x && x < hi)) { r.fallback_guess = 1; x = lo + (hi - lo) / 2.; }
    Tof value = counted_multi(r, x, s, m);
    if (value.status) { r.status = 5; r.evaluation_status = value.status; return r; }
    record(r, x, value, target);
    double cycle_residual = ::fabs(value.time - target);
    for (int64_t iteration = 1; iteration <= o.max_iterations; ++iteration) {
        double f = value.time - target;
        bool force = false;
        if (iteration % 4 == 0) {
            force = ::fabs(f) >= .25 * cycle_residual && !residual_ok(f, target, o);
            cycle_residual = ::fabs(f);
        }
        multi_update(r, x, value, target, decreasing, lo, hi, lower, upper);
        double middle = lo + (hi - lo) / 2., candidate = missing;
        int kind = 1;
        if (f == 0.) { candidate = x; kind = 2; }
        else if (middle == lo || middle == hi) {
            if (!lower.status && residual_ok(lower.time - target, target, o)) {
                candidate = lo; kind = lo == x ? 2 : 1;
            } else if (!upper.status && residual_ok(upper.time - target, target, o)) {
                candidate = hi; kind = hi == x ? 2 : 1;
            } else { r.status = 4; return r; }
        } else if (!force) {
            Derivatives d = multi_derivatives(x, s, m, value);
            if (d.status) ++r.derivative_failures;
            double step = householder(f, d, decreasing), proposed = x - step;
            if (finite(step)) {
                if (finite(proposed) && lo < proposed && proposed < hi && ::fabs(step) <= .75 * (hi - lo)) {
                    candidate = proposed; kind = 0;
                } else if (proposed == x && residual_ok(f, target, o)) { candidate = x; kind = 2; }
            }
            if (!finite(candidate)) ++r.rejected_householder;
        }
        if (!finite(candidate)) candidate = middle;
        Tof next = counted_multi(r, candidate, s, m);
        if (next.status && kind == 0) {
            ++r.rejected_householder; candidate = middle; kind = 1;
            next = counted_multi(r, candidate, s, m);
        }
        if (next.status) { r.status = 5; r.evaluation_status = next.status; return r; }
        r.last_step = candidate - x;
        ++r.iterations;
        if (kind == 0) ++r.householder; else if (kind == 1) ++r.bisection; else ++r.verification;
        x = candidate; value = next;
        record(r, x, value, target);
        multi_update(r, x, value, target, decreasing, lo, hi, lower, upper);
        if (::fabs(r.last_step) <= o.x_atol + o.x_rtol * ::fabs(x) && residual_ok(r.residual, target, o)) {
            Derivatives d = multi_derivatives(x, s, m, value);
            if (!d.status) {
                if (d.first == 0. || (d.first < 0.) != decreasing) { r.status = 10; return r; }
                double estimate = 2. * (::fabs(r.residual) + 16. * ulp(value.time)) / ::fabs(d.first);
                if (finite(estimate)) {
                    r.estimated_x_error = estimate;
                    r.precision_limited = estimate > o.x_atol + o.x_rtol * ::fabs(x) ? 1 : 0;
                }
            } else ++r.derivative_failures;
            r.x = x; r.y = value.y; r.status = 0;
            return r;
        }
    }
    r.status = 3;
    return r;
}
LB_HD Result solve_multi(const Input &p, Options o, MinimumOptions mo = MinimumOptions{}) {
    Result r;
    r.branch = p.branch; r.revolutions = p.revolutions; r.root.status = -1;
    if (p.revolutions < 1 || p.branch < 0 || p.branch > 1) return r;
    Geometry g = prepare(p, true);
    r.geometry_status = g.status;
    if (g.status) return r;
    r.lambda = g.lambda; r.q = g.q; r.target = g.time;
    r.status = 2;
    if (!valid(o)) { r.root.status = 1; return r; }
    Shape s = shape(g.lambda, g.q);
    r.minimum = find_minimum(s, p.revolutions, mo);
    r.root = multi_root(g.time, s, p.revolutions, p.branch, o, r.minimum);
    if (r.root.status) return r;
    return reconstruct(p, g, r);
}
LB_HD Result solve_selected(const Input &p, Options o, MinimumOptions mo = MinimumOptions{}) {
    return p.revolutions == 0 ? solve(p, o) : solve_multi(p, o, mo);
}
} // namespace lambert
