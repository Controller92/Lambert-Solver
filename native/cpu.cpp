#include "api.h"
#include <type_traits>

static_assert(std::is_standard_layout<lambert::Input>::value, "Input must be a plain ABI struct");
static_assert(std::is_standard_layout<lambert::Result>::value, "Result must be a plain ABI struct");

extern "C" LambertAbi lb_abi() {
    return {2,
            sizeof(lambert::Input),
            sizeof(lambert::Options),
            sizeof(lambert::Root),
            sizeof(lambert::Result),
            sizeof(lambert::Tof),
            sizeof(lambert::Derivatives)};
}
extern "C" LambertMultiAbi lb_multi_abi() {
    return {sizeof(lambert::Minimum), sizeof(lambert::MinimumOptions)};
}
extern "C" uint32_t lb_jacobian_size() { return sizeof(lambert::Jacobian); }
extern "C" int lb_cpu_jacobian(const lambert::Input *input, lambert::Result *output,
    lambert::Jacobian *jacobian, size_t n, const lambert::Options *options,
    const lambert::MinimumOptions *minimum_options) {
    if (!options || !minimum_options || (n && (!input || !output || !jacobian)) ||
        n > SIZE_MAX / sizeof(lambert::Result) || n > SIZE_MAX / sizeof(lambert::Jacobian)) return 1;
    for (size_t i = 0; i < n; ++i) {
        output[i] = lambert::solve_selected(input[i], *options, *minimum_options);
        jacobian[i] = lambert::terminal_jacobian(input[i], output[i]);
    }
    return 0;
}
extern "C" int lb_cpu(const lambert::Input *input, lambert::Result *output, size_t n,
                      const lambert::Options *options, const lambert::MinimumOptions *minimum_options) {
    if (!options || !minimum_options || (n && (!input || !output)) || n > SIZE_MAX / sizeof(lambert::Result))
        return 1;
    for (size_t i = 0; i < n; ++i)
        output[i] = lambert::solve_selected(input[i], *options, *minimum_options);
    return 0;
}
static lambert::Shape probe_shape(const double *row) {
    return std::isnan(row[2]) ? lambert::shape(row[1]) : lambert::shape(row[1], row[2]);
}
extern "C" int lb_roots(const double *input, lambert::Root *output, size_t n,
                        const lambert::Options *options) {
    if (!options || (n && (!input || !output)) || n > SIZE_MAX / (3 * sizeof(double)) ||
        n > SIZE_MAX / sizeof(lambert::Root))
        return 1;
    for (size_t i = 0; i < n; ++i)
        output[i] = lambert::solve_root(input[3 * i], probe_shape(input + 3 * i), *options);
    return 0;
}
extern "C" int lb_tof(const double *input, lambert::Tof *output, size_t n) {
    if ((n && (!input || !output)) || n > SIZE_MAX / (3 * sizeof(double)) ||
        n > SIZE_MAX / sizeof(lambert::Tof))
        return 1;
    for (size_t i = 0; i < n; ++i)
        output[i] = lambert::tof(input[3 * i], probe_shape(input + 3 * i));
    return 0;
}
extern "C" int lb_derivatives(const double *input, lambert::Derivatives *output, size_t n) {
    if ((n && (!input || !output)) || n > SIZE_MAX / (3 * sizeof(double)) ||
        n > SIZE_MAX / sizeof(lambert::Derivatives))
        return 1;
    for (size_t i = 0; i < n; ++i)
        output[i] = lambert::derivatives(input[3 * i], probe_shape(input + 3 * i));
    return 0;
}

static bool valid_multi_row(const double *row) {
    return lambert::finite(row[3]) && row[3] >= 1. && row[3] <= INT32_MAX && std::floor(row[3]) == row[3]
        && (row[4] == 0. || row[4] == 1.);
}
extern "C" int lb_multi_roots(const double *input, lambert::Result *output, size_t n,
    const lambert::Options *o, const lambert::MinimumOptions *mo) {
    if (!o || !mo || (n && (!input || !output)) || n > SIZE_MAX / sizeof(lambert::Result)) return 1;
    for (size_t i = 0; i < n; ++i) {
        const double *row = input + 5*i;
        lambert::Result r;
        if (valid_multi_row(row) && lambert::valid(*o) && lambert::finite(row[0]) && row[0] > 0.) {
            auto s = probe_shape(row);
            if (s.valid) {
                r.minimum = lambert::find_minimum(s, int32_t(row[3]), *mo);
                r.root = lambert::multi_root(row[0], s, int32_t(row[3]), int32_t(row[4]), *o, r.minimum);
                r.branch = int32_t(row[4]); r.revolutions = int32_t(row[3]);
                r.status = r.root.status ? 2 : 0;
            }
        }
        output[i] = r;
    }
    return 0;
}
extern "C" int lb_multi_tof(const double *input, lambert::Tof *output, size_t n) {
    if ((n && (!input || !output)) || n > SIZE_MAX / (5*sizeof(double))) return 1;
    for (size_t i = 0; i < n; ++i) {
        const double *row = input + 5*i;
        output[i] = valid_multi_row(row) ? lambert::multi_tof(row[0], probe_shape(row), int32_t(row[3])) : lambert::Tof{};
    }
    return 0;
}
extern "C" int lb_multi_derivatives(const double *input, lambert::Derivatives *output, size_t n) {
    if ((n && (!input || !output)) || n > SIZE_MAX / sizeof(lambert::Derivatives)) return 1;
    for (size_t i = 0; i < n; ++i) {
        const double *row = input + 5*i;
        output[i] = valid_multi_row(row) ? lambert::multi_derivatives(row[0], probe_shape(row), int32_t(row[3]),
                     lambert::multi_tof(row[0], probe_shape(row), int32_t(row[3]))) : lambert::Derivatives{};
    }
    return 0;
}
