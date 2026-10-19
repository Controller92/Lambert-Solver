#pragma once
// Binary64, M=0 Izzo reference shared by host C++ and CUDA. No heap, I/O or
// device runtime calls. Units and status values match the Python reference.
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define LB_HD __host__ __device__ inline
#else
#define LB_HD inline
#endif

namespace lambert {
constexpr double eps = DBL_EPSILON;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();

struct Vec3 {
    double x, y, z;
};
struct Input {
    Vec3 r1, r2;
    double tof, mu;
    int32_t long_way, revolutions;
};
struct Options {
    double x_atol = 1e-14, x_rtol = 1e-14, tof_atol = 0., tof_rtol = 5e-13;
    int32_t max_iterations = 128, max_bracket_expansions = 1024;
};
struct Root {
    int32_t status = 1, evaluation_status = -1, precision_limited = -1, fallback_guess = 0;
    double x = missing, y = missing, last_x = missing, time = missing, residual = missing;
    double relative_residual = missing, last_step = missing, lower = missing, upper = missing;
    double estimated_x_error = missing;
    int64_t iterations = 0, householder = 0, bisection = 0, verification = 0;
    int64_t rejected_householder = 0, derivative_failures = 0, expansions = 0, evaluations = 0;
};
struct Result {
    int32_t status = 1, geometry_status = 1;
    Root root;
    Vec3 v1{missing, missing, missing}, v2{missing, missing, missing};
    double lambda = missing, q = missing, target = missing;
};
struct Tof {
    int32_t status = 1, method = -1, terms = 0, reserved = 0;
    double time = missing, y = missing;
};
struct Derivatives {
    int32_t status = 1, method = -1, terms = 0, reserved = 0;
    double time = missing, y = missing, first = missing, second = missing, third = missing;
};
struct Guess {
    int32_t status = 1;
    double x = missing;
};
struct Geometry {
    int32_t status = 1;
    double r1 = missing, r2 = missing, s = missing, lambda = missing, q = missing, time = missing;
    double rho = missing, sigma = missing;
    Vec3 radial1{}, radial2{}, transverse1{}, transverse2{};
};
struct Shape {
    bool valid = false;
    double lambda = missing, q = missing, complement = missing;
};

LB_HD bool finite(double x) {
    return std::isfinite(x);
}
LB_HD double max2(double a, double b) {
    return a > b ? a : b;
}
LB_HD double min2(double a, double b) {
    return a < b ? a : b;
}
LB_HD bool finite(Vec3 v) {
    return finite(v.x) && finite(v.y) && finite(v.z);
}
LB_HD Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
LB_HD Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
LB_HD Vec3 mul(Vec3 a, double k) {
    return {a.x * k, a.y * k, a.z * k};
}
LB_HD Vec3 divide(Vec3 a, double k) {
    return {a.x / k, a.y / k, a.z / k};
}
LB_HD double norm(Vec3 a) {
    return ::hypot(::hypot(a.x, a.y), a.z);
}
LB_HD Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Short error-free expansions retain cancellation-sensitive sums. The number
// of terms is bounded at compile time (at most four), never dynamically allocated.
template <int N> LB_HD double small_sum(const double (&values)[N]) {
    double partial[N];
    int count = 0;
    for (int j = 0; j < N; ++j) {
        double x = values[j];
        int next = 0;
        for (int k = 0; k < count; ++k) {
            double y = partial[k];
            if (::fabs(x) < ::fabs(y)) {
                double temp = x;
                x = y;
                y = temp;
            }
            double high = x + y, low = y - (high - x);
            if (!finite(high))
                return missing;
            if (low != 0.)
                partial[next++] = low;
            x = high;
        }
        partial[next] = x;
        count = next + 1;
    }
    double total = 0.;
    for (int k = 0; k < count; ++k)
        total += partial[k];
    return total;
}
LB_HD double sum2(double a, double b) {
    const double x[2] = {a, b};
    return small_sum(x);
}
LB_HD double sum3(double a, double b, double c) {
    const double x[3] = {a, b, c};
    return small_sum(x);
}
LB_HD double sum4(double a, double b, double c, double d) {
    const double x[4] = {a, b, c, d};
    return small_sum(x);
}
LB_HD double dot(Vec3 a, Vec3 b) {
    return sum3(a.x * b.x, a.y * b.y, a.z * b.z);
}
LB_HD double difference_products(double a, double b, double c, double d) {
    const double ab = a * b, cd = c * d;
    // Explicit FMA supplies the product remainders. Implicit contraction is
    // disabled by the build, so other arithmetic follows the reference order.
    return sum4(ab, -cd, ::fma(a, b, -ab), -::fma(c, d, -cd));
}
LB_HD Vec3 scaled(Vec3 a) {
    int e = 0;
    ::frexp(max2(::fabs(a.x), max2(::fabs(a.y), ::fabs(a.z))), &e);
    return {::ldexp(a.x, -e), ::ldexp(a.y, -e), ::ldexp(a.z, -e)};
}
LB_HD Shape shape(double lambda, double q) {
    Shape s;
    if (!finite(lambda) || ::fabs(lambda) > 1.)
        return s;
    const double inferred = (1. - lambda) * (1. + lambda);
    if (!finite(q) || q <= 0. || q > 1. || ::fabs(q - inferred) > 64. * eps)
        return s;
    s.complement = lambda >= 0. ? q / (1. + lambda) : 1. - lambda;
    s.valid = true;
    s.lambda = lambda;
    s.q = q;
    return s;
}
LB_HD Shape shape(double lambda) {
    Shape s = shape(lambda, (1. - lambda) * (1. + lambda));
    if (s.valid)
        s.complement = 1. - lambda;
    return s;
}
LB_HD double eta(double x, Shape s, double y) {
    return s.lambda * x > 0. ? s.q / (y + s.lambda * x) : y - s.lambda * x;
}
LB_HD Geometry prepare(const Input &p) {
    Geometry g;
    if (!finite(p.r1) || !finite(p.r2) || !finite(p.tof) || p.tof <= 0. || !finite(p.mu) ||
        p.mu <= 0. || p.long_way < 0 || p.long_way > 1 || p.revolutions < 0)
        return g;
    g.r1 = norm(p.r1);
    g.r2 = norm(p.r2);
    if (g.r1 == 0. || g.r2 == 0.)
        return g;
    if (p.revolutions != 0) {
        g.status = 2;
        return g;
    }
    Vec3 chord = sub(p.r1, p.r2);
    double c = norm(chord);
    g.s = .5 * g.r1 + .5 * g.r2 + .5 * c;
    g.status = 4;
    if (!finite(g.r1) || !finite(g.r2) || !finite(c) || !finite(g.s))
        return g;
    Vec3 a = scaled(p.r1), b = scaled(p.r2);
    Vec3 h{difference_products(a.y, b.z, a.z, b.y), difference_products(a.z, b.x, a.x, b.z),
           difference_products(a.x, b.y, a.y, b.x)};
    double hn = norm(h), sine = hn / (norm(a) * norm(b));
    if (c == 0. || sine <= 64. * eps) {
        g.status = 3;
        return g;
    }
    if (g.r1 / g.s == 0. || g.r2 / g.s == 0.)
        return g;
    Vec3 normal = divide(h, p.long_way ? -hn : hn);
    g.radial1 = divide(p.r1, g.r1);
    g.radial2 = divide(p.r2, g.r2);
    double cosine = dot(g.radial1, g.radial2), ch, sh;
    if (cosine >= 0.) {
        ch = norm(add(g.radial1, g.radial2)) / 2.;
        sh = sine / (2. * ch);
    } else {
        sh = norm(sub(g.radial1, g.radial2)) / 2.;
        ch = sine / (2. * sh);
    }
    double lambda = ::sqrt(g.r1 / g.s) * ::sqrt(g.r2 / g.s) * ch;
    g.lambda = (p.long_way ? -1. : 1.) * min2(1., lambda);
    // c<=s geometrically. Device hypot rounding can put their ratio one ulp
    // above one near the antiparallel limit, just as sigma can round above one.
    g.q = min2(1., c / g.s);
    double scale = max2(g.r1, g.r2);
    g.rho = dot(divide(chord, c), add(divide(p.r1, scale), divide(p.r2, scale))) /
            (g.r1 / scale + g.r2 / scale);
    g.rho = max2(-1., min2(1., g.rho));
    g.sigma = min2(1., 2. * ::sqrt(g.r1 / c) * ::sqrt(g.r2 / c) * sh);
    double time_scale = (g.s / ::sqrt(p.mu)) * ::sqrt(g.s / 2.);
    if (!finite(time_scale) || time_scale <= 0.)
        return g;
    g.time = p.tof / time_scale;
    if (!finite(g.time) || g.time <= 0.)
        return g;
    g.transverse1 = cross(normal, g.radial1);
    g.transverse2 = cross(normal, g.radial2);
    g.status = 0;
    return g;
}

LB_HD Tof tof(double x, Shape s) {
    Tof r;
    if (!s.valid || !finite(x) || x <= -1.)
        return r;
    r.status = 2;
    double lambda = s.lambda, q = s.q, y = ::hypot(::sqrt(q), lambda * x), time;
    if (x == 1.) {
        time = (2. / 3.) * s.complement * (1. + lambda + lambda * lambda);
        y = 1.;
        r.method = 2;
    } else {
        double e = eta(x, s, y);
        if (!finite(e) || e <= 0.)
            return r;
        if (x >= .8 && x <= 1.2) {
            double z = sum2(s.complement, -x * e) / 2.;
            double total = 1., term = 1., correction = 0.;
            bool converged = false;
            for (int n = 1; n <= 128; ++n) {
                term *= z * (n + 2.) / (n + 1.5);
                double adjusted = term - correction, updated = total + adjusted;
                correction = (updated - total) - adjusted;
                total = updated;
                double ratio = ::fabs(z) * (n + 3.) / (n + 2.5);
                r.terms = n;
                if (::fabs(term) * ratio / (1. - ratio) <= eps * ::fabs(total)) {
                    converged = true;
                    break;
                }
            }
            if (!converged) {
                r.status = 3;
                return r;
            }
            time = sum2((2. / 3.) * e * e * e * total, 2. * lambda * e);
            r.method = 1;
        } else if (x < 1.) {
            double d = (1. - x) * (1. + x), k = ::sqrt(d);
            double psi = ::atan2(e * k, sum2(lambda, x * e));
            time = sum3(psi / k, lambda * e, -q * x) / d;
            r.method = 0;
        } else {
            double k = ::sqrt(x - 1.) * ::sqrt(x + 1.), psi = ::asinh(e * k);
            time = sum3(q * x, -lambda * e, -psi / k) / k / k;
            r.method = 3;
        }
    }
    if (!finite(y) || !finite(time) || time <= 0.)
        return r;
    r.status = 0;
    r.time = time;
    r.y = y;
    return r;
}
LB_HD double odd_complement(Shape s, int power) {
    if (s.lambda <= 0.) {
        double product = 1.;
        for (int k = 0; k < power; ++k)
            product *= -s.lambda;
        return 1. + product;
    }
    double polynomial = 1.;
    for (int k = 1; k < power; ++k)
        polynomial = 1. + s.lambda * polynomial;
    return s.complement * polynomial;
}
LB_HD void compensated_add(double term, double &sum, double &correction) {
    double v = term - correction, updated = sum + v;
    correction = (updated - sum) - v;
    sum = updated;
}
LB_HD Derivatives derivatives(double x, Shape s) {
    Derivatives r;
    Tof t = tof(x, s);
    r.status = t.status;
    if (t.status != 0)
        return r;
    r.status = 2;
    double a, b, c, lambda = s.lambda, q = s.q;
    if (x == 1.) {
        double a1 = odd_complement(s, 5) / 5., a2 = (3. / 28.) * odd_complement(s, 7),
               a3 = (5. / 72.) * odd_complement(s, 9);
        a = -2. * a1;
        b = sum2(8. * a2, -2. * a1);
        c = sum2(24. * a2, -48. * a3);
        r.method = 2;
    } else if (x >= .7 && x <= 1.3) {
        double d = (1. - x) * (1. + x), l2 = lambda * lambda;
        double complement = s.complement * (1. + lambda + l2), lp = lambda * l2, coefficient = 1.;
        double s1 = 0., s2 = 0., s3 = 0., e1 = 0., e2 = 0., e3 = 0., p1 = 1., p2 = 1., p3 = 1.;
        bool converged = false;
        for (int n = 1; n <= 256; ++n) {
            complement = sum2(complement, lp * q);
            lp *= l2;
            coefficient *= (n - .5) / n;
            double an = 2. * coefficient * complement / (2 * n + 3);
            double t1 = n * an * p1, t2 = 0., t3 = 0.;
            compensated_add(t1, s1, e1);
            p1 *= d;
            if (n >= 2) {
                t2 = n * (n - 1) * an * p2;
                compensated_add(t2, s2, e2);
                p2 *= d;
            }
            if (n >= 3) {
                t3 = n * (n - 1) * (n - 2) * an * p3;
                compensated_add(t3, s3, e3);
                p3 *= d;
                double ratio = ::fabs(d) * (n + .5) / (n - 2), bound = ratio / (1. - ratio);
                if (ratio < 1. && ::fabs(t1) * bound <= eps * ::fabs(s1) &&
                    ::fabs(t2) * bound <= eps * ::fabs(s2) &&
                    ::fabs(t3) * bound <= eps * ::fabs(s3)) {
                    converged = true;
                    r.terms = n;
                    break;
                }
            }
            r.terms = n;
        }
        if (!converged) {
            r.status = 3;
            return r;
        }
        a = -2. * x * s1;
        b = sum2(4. * x * x * s2, -2. * s1);
        c = sum2(-8. * x * x * x * s3, 12. * x * s2);
        r.method = 1;
    } else {
        double e = eta(x, s, t.y), d = (1. - x) * (1. + x), ratio = lambda / t.y;
        double ratio3 = ratio * ratio * ratio, ratio5 = ratio3 * ratio * ratio;
        a = sum3(3. * t.time * x, -2. * q, -2. * lambda * lambda * (e / t.y)) / d;
        b = sum3(3. * t.time, 5. * x * a, 2. * q * ratio3) / d;
        c = sum3(7. * x * b, 8. * a, -6. * q * ratio5 * x) / d;
        r.method = 0;
    }
    if (!finite(a) || !finite(b) || !finite(c) || a == 0. || b == 0. || c == 0.)
        return r;
    r.status = 0;
    r.time = t.time;
    r.y = t.y;
    r.first = a;
    r.second = b;
    r.third = c;
    return r;
}
LB_HD double log_ratio(double a, double b) {
    double ratio = a / b;
    return ratio >= .5 && ratio <= 2. ? ::log1p((a - b) / b) : ::log(a) - ::log(b);
}
LB_HD Guess guess(double target, Shape s) {
    Guess r;
    if (!finite(target) || target <= 0.)
        return r;
    Tof t0 = tof(0., s), t1 = tof(1., s);
    if (t0.status || t1.status) {
        r.status = t0.status ? t0.status : t1.status;
        return r;
    }
    double x;
    if (target >= t0.time)
        x = ::expm1((2. / 3.) * log_ratio(t0.time, target));
    else if (target <= t1.time) {
        double l = s.lambda, l2 = l * l,
               scale = (5. / 3.) * (1. + l + l2) / (1. + l + l2 + l * l2 + l2 * l2);
        x = 1. + scale * ((t1.time - target) / target);
    } else
        x = ::expm1(log_ratio(t0.time, target) * ::log(2.) / log_ratio(t0.time, t1.time));
    r.status = 2;
    if (finite(x) && x > -1.) {
        r.status = 0;
        r.x = x;
    }
    return r;
}
LB_HD bool valid(Options o) {
    return finite(o.x_atol) && o.x_atol >= 0. && finite(o.x_rtol) && o.x_rtol >= 0. &&
           o.x_rtol < 1. && finite(o.tof_atol) && o.tof_atol >= 0. && finite(o.tof_rtol) &&
           o.tof_rtol >= 0. && o.tof_rtol < 1. && o.max_iterations >= 0 &&
           o.max_bracket_expansions >= 0;
}
LB_HD void record(Root &r, double x, Tof t, double target) {
    r.last_x = x;
    r.time = t.time;
    r.residual = t.time - target;
    r.relative_residual = r.residual / target;
}
LB_HD bool residual_ok(double residual, double target, Options o) {
    return ::fabs(residual) <= o.tof_atol + o.tof_rtol * target;
}
LB_HD Tof counted_tof(Root &r, double x, Shape s) {
    ++r.evaluations;
    return tof(x, s);
}
LB_HD double householder(double f, Derivatives d) {
    if (d.status || d.first >= 0.)
        return missing;
    double u = f / d.first, a = u * (d.second / d.first), b = (u * (d.third / d.first)) * u / 6.;
    double den = sum3(1., -a, b);
    if (::fabs(den) <= 64. * eps * max2(1., max2(::fabs(a), ::fabs(b))))
        return missing;
    double step = u * (1. - a / 2.) / den;
    return finite(step) ? step : missing;
}
LB_HD double ulp(double value) {
    if (::fabs(value) < DBL_MIN)
        return DBL_TRUE_MIN;
    int exponent = 0;
    ::frexp(value, &exponent);
    return ::ldexp(1., exponent - 53);
}
LB_HD Root solve_root(double target, Shape s, Options o = Options{}) {
    Root r;
    if (!s.valid || !finite(target) || target <= 0. || !valid(o))
        return r;
    Guess starter = guess(target, s);
    r.evaluations += 2;
    Tof t0 = counted_tof(r, 0., s);
    if (t0.status) {
        r.status = 5;
        r.evaluation_status = t0.status;
        return r;
    }
    record(r, 0., t0, target);
    double lo, hi;
    Tof lower, upper;
    if (target >= t0.time) {
        lo = -1.;
        hi = 0.;
        upper = t0;
    } else {
        lo = 0.;
        lower = t0;
        hi = starter.status == 0 ? max2(1., starter.x) : 1.;
        upper = counted_tof(r, hi, s);
        r.lower = lo;
        r.upper = hi;
        for (;;) {
            if (upper.status) {
                r.status = 5;
                r.evaluation_status = upper.status;
                return r;
            }
            record(r, hi, upper, target);
            if (upper.time <= target)
                break;
            if (r.expansions >= o.max_bracket_expansions) {
                r.status = 2;
                return r;
            }
            lo = hi;
            lower = upper;
            hi = 2. * hi + 1.;
            r.lower = lo;
            r.upper = hi;
            if (!finite(hi)) {
                r.status = 6;
                return r;
            }
            ++r.expansions;
            upper = counted_tof(r, hi, s);
        }
    }
    r.lower = lo;
    r.upper = hi;
    double x;
    if (starter.status == 0 && starter.x >= lo && starter.x <= hi)
        x = starter.x;
    else {
        r.fallback_guess = 1;
        x = lo + (hi - lo) / 2.;
    }
    Tof value = x == 0. ? t0 : x == hi ? upper : counted_tof(r, x, s);
    if (value.status) {
        r.status = 5;
        r.evaluation_status = value.status;
        return r;
    }
    record(r, x, value, target);
    for (int64_t iteration = 1; iteration <= o.max_iterations; ++iteration) {
        double f = value.time - target;
        if (f > 0.) {
            lo = x;
            lower = value;
        } else if (f < 0.) {
            hi = x;
            upper = value;
        }
        r.lower = lo;
        r.upper = hi;
        double middle = lo + (hi - lo) / 2., candidate = missing;
        int kind = 1;
        if (f == 0.) {
            candidate = x;
            kind = 2;
        } else if (middle == lo || middle == hi) {
            if (lower.status == 0 && residual_ok(lower.time - target, target, o)) {
                candidate = lo;
                kind = lo == x ? 2 : 1;
            } else if (upper.status == 0 && residual_ok(upper.time - target, target, o)) {
                candidate = hi;
                kind = hi == x ? 2 : 1;
            } else {
                r.status = 4;
                return r;
            }
        } else if (iteration % 4 != 0 || residual_ok(f, target, o)) {
            Derivatives d = derivatives(x, s);
            ++r.evaluations;
            if (d.status)
                ++r.derivative_failures;
            double step = householder(f, d), proposed = x - step;
            if (finite(step)) {
                if (finite(proposed) && lo < proposed && proposed < hi &&
                    ::fabs(step) <= .75 * (hi - lo)) {
                    candidate = proposed;
                    kind = 0;
                } else if (proposed == x && residual_ok(f, target, o)) {
                    candidate = x;
                    kind = 2;
                }
            }
            if (!finite(candidate))
                ++r.rejected_householder;
        }
        if (!finite(candidate))
            candidate = middle;
        Tof next = counted_tof(r, candidate, s);
        if (next.status && kind == 0) {
            ++r.rejected_householder;
            candidate = middle;
            kind = 1;
            next = counted_tof(r, candidate, s);
        }
        if (next.status) {
            r.status = 5;
            r.evaluation_status = next.status;
            return r;
        }
        r.last_step = candidate - x;
        ++r.iterations;
        if (kind == 0)
            ++r.householder;
        else if (kind == 1)
            ++r.bisection;
        else
            ++r.verification;
        x = candidate;
        value = next;
        record(r, x, value, target);
        if (r.residual > 0.) {
            lo = x;
            lower = value;
        } else if (r.residual < 0.) {
            hi = x;
            upper = value;
        }
        r.lower = lo;
        r.upper = hi;
        if (::fabs(r.last_step) <= o.x_atol + o.x_rtol * ::fabs(x) &&
            residual_ok(r.residual, target, o)) {
            r.x = x;
            r.y = value.y;
            Derivatives d = derivatives(x, s);
            ++r.evaluations;
            if (d.status == 0 && d.first < 0.) {
                r.estimated_x_error =
                    (::fabs(r.residual) + 16. * ulp(value.time)) / ::fabs(d.first);
                r.precision_limited = r.estimated_x_error > o.x_atol + o.x_rtol * ::fabs(x) ? 1 : 0;
            }
            r.status = 0;
            return r;
        }
    }
    r.status = 3;
    return r;
}
LB_HD double velocity_scale(double mu, double s, double radius) {
    int me = 0, se = 0, re = 0;
    double mm = ::frexp(mu, &me), sm = ::frexp(s, &se), rm = ::frexp(radius, &re);
    int half = (me + se) / 2, odd = (me + se) % 2;
    if (odd < 0) {
        --half;
        odd += 2;
    } // Match floor divmod for negative exponents.
    double m = ::sqrt(mm * sm / 2.) / rm;
    if (odd)
        m *= ::sqrt(2.);
    return ::ldexp(m, half - re);
}
LB_HD Vec3 velocity(Vec3 radial, Vec3 transverse, double vr, double vt) {
    return {sum2(vr * radial.x, vt * transverse.x), sum2(vr * radial.y, vt * transverse.y),
            sum2(vr * radial.z, vt * transverse.z)};
}
LB_HD Result solve(const Input &p, Options o = Options{}) {
    Result r;
    r.root.status = -1;
    Geometry g = prepare(p);
    r.geometry_status = g.status;
    if (g.status)
        return r;
    r.lambda = g.lambda;
    r.q = g.q;
    r.target = g.time;
    r.root = solve_root(g.time, shape(g.lambda, g.q), o);
    if (r.root.status) {
        r.status = 2;
        return r;
    }
    double x = r.root.x, y = r.root.y, l = g.lambda, q = g.q;
    double minus = eta(x, shape(l, q), y), plus = l * x > 0. ? y + l * x : q / minus;
    double a = l * x > 0. ? sum2(l * minus, -q * x) : l * y - x;
    double b = l * x > 0. ? l * y + x : sum2(l * plus, q * x);
    double scale1 = velocity_scale(p.mu, g.s, g.r1), scale2 = velocity_scale(p.mu, g.s, g.r2);
    double vr1 = scale1 * sum2(a, -g.rho * b), vr2 = -scale2 * sum2(a, g.rho * b);
    double transverse = g.sigma * plus, vt1 = scale1 * transverse, vt2 = scale2 * transverse;
    Vec3 v1 = velocity(g.radial1, g.transverse1, vr1, vt1),
         v2 = velocity(g.radial2, g.transverse2, vr2, vt2);
    r.status = 3;
    if (!finite(v1) || !finite(v2) || scale1 <= 0. || scale2 <= 0. || vt1 <= 0. || vt2 <= 0.)
        return r;
    r.status = 0;
    r.v1 = v1;
    r.v2 = v2;
    return r;
}
} // namespace lambert
