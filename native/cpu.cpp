#include "api.h"
#include <type_traits>

static_assert(std::is_standard_layout<lambert::Input>::value, "Input must be a plain ABI struct");
static_assert(std::is_standard_layout<lambert::Result>::value, "Result must be a plain ABI struct");

extern "C" LambertAbi lb_abi() {
    return {1,
            sizeof(lambert::Input),
            sizeof(lambert::Options),
            sizeof(lambert::Root),
            sizeof(lambert::Result),
            sizeof(lambert::Tof),
            sizeof(lambert::Derivatives)};
}
extern "C" int lb_cpu(const lambert::Input *input, lambert::Result *output, size_t n,
                      const lambert::Options *options) {
    if (!options || (n && (!input || !output)) || n > SIZE_MAX / sizeof(lambert::Result))
        return 1;
    for (size_t i = 0; i < n; ++i)
        output[i] = lambert::solve(input[i], *options);
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
