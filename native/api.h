#pragma once
#include "jacobian.hpp"
#include <cstddef>

// ABI version and size checks protect ctypes clients against stale libraries.
struct LambertAbi {
    uint32_t version, input_size, options_size, root_size, result_size, tof_size, derivative_size;
};
struct LambertMultiAbi {
    uint32_t minimum_size, minimum_options_size;
};
struct LambertDeviceInfo {
    char name[256];
    int32_t major, minor, driver, runtime, multiprocessors, registers, local_bytes;
    int32_t active_blocks, threads_per_sm, max_threads_per_block;
};
extern "C" {
LambertAbi lb_abi();
LambertMultiAbi lb_multi_abi();
int lb_cpu(const lambert::Input *, lambert::Result *, size_t, const lambert::Options *, const lambert::MinimumOptions *);
// Independent optional ABI; existing velocity-only structs stay version 2.
uint32_t lb_jacobian_size();
int lb_cpu_jacobian(const lambert::Input *, lambert::Result *, lambert::Jacobian *, size_t,
                    const lambert::Options *, const lambert::MinimumOptions *);
// Validation probes: triples [x or target, lambda, q]; NaN q requests scalar lambda.
int lb_roots(const double *, lambert::Root *, size_t, const lambert::Options *);
// Scalar M>0 probes: rows [x/target, lambda, q, M, branch].
int lb_multi_roots(const double *, lambert::Result *, size_t, const lambert::Options *, const lambert::MinimumOptions *);
int lb_multi_tof(const double *, lambert::Tof *, size_t);
int lb_multi_derivatives(const double *, lambert::Derivatives *, size_t);
int lb_tof(const double *, lambert::Tof *, size_t);
int lb_derivatives(const double *, lambert::Derivatives *, size_t);
int lb_cuda_info(int, int, int, LambertDeviceInfo *, char *, size_t);
int lb_cuda_create(size_t, int, int, void **, char *, size_t);
int lb_cuda_destroy(void *, char *, size_t);
int lb_cuda_upload(void *, const lambert::Input *, size_t, char *, size_t);
int lb_cuda_run(void *, const lambert::Options *, const lambert::MinimumOptions *, int, float *, char *, size_t);
int lb_cuda_run_jacobian(void *, const lambert::Options *, const lambert::MinimumOptions *, int, float *, char *, size_t);
int lb_cuda_download(void *, lambert::Result *, size_t, char *, size_t);
int lb_cuda_download_jacobian(void *, lambert::Jacobian *, size_t, char *, size_t);
}
