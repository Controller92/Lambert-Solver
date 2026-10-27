#include <cuda_runtime.h>
#include <cstdio>
#include <new>
#include "api.h"

namespace {
__global__ void solve_kernel(const lambert::Input *input, lambert::Result *output, size_t count,
                             lambert::Options options) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index].revolutions == 0)
        output[index] = lambert::solve(input[index], options);
}
__global__ void multi_kernel(const lambert::Input *input, lambert::Result *output, size_t count,
                             lambert::Options options, lambert::MinimumOptions minimum_options) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index].revolutions != 0)
        output[index] = lambert::solve_multi(input[index], options, minimum_options);
}
struct Context {
    lambert::Input *input = nullptr;
    lambert::Result *output = nullptr;
    lambert::Jacobian *jacobian = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    size_t capacity = 0, count = 0;
    bool zero = false, multi = false;
    bool jacobian_ready = false;
    int device = 0, threads = 128, max_blocks = 0;
};
__global__ void jacobian_kernel(const lambert::Input *input, const lambert::Result *output,
                               lambert::Jacobian *jacobian, size_t count) {
    size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index < count) jacobian[index] = lambert::terminal_jacobian(input[index],output[index]);
}
int report(cudaError_t status, char *error, size_t capacity) {
    if (error && capacity)
        std::snprintf(error, capacity, "%s", cudaGetErrorString(status));
    return static_cast<int>(status);
}
cudaError_t release(Context *c) {
    cudaError_t first = cudaSetDevice(c->device);
    auto keep = [&](cudaError_t status) {
        if (first == cudaSuccess && status != cudaSuccess)
            first = status;
    };
    if (c->stream)
        keep(cudaStreamSynchronize(c->stream));
    if (c->input)
        keep(cudaFree(c->input));
    if (c->output)
        keep(cudaFree(c->output));
    if (c->jacobian)
        keep(cudaFree(c->jacobian));
    if (c->begin)
        keep(cudaEventDestroy(c->begin));
    if (c->end)
        keep(cudaEventDestroy(c->end));
    if (c->stream)
        keep(cudaStreamDestroy(c->stream));
    delete c;
    return first;
}
} // namespace

extern "C" int lb_cuda_info(int device, int threads, int multi, LambertDeviceInfo *info, char *error,
                            size_t capacity) {
    if (!info)
        return report(cudaErrorInvalidValue, error, capacity);
    cudaError_t status = cudaSetDevice(device);
    cudaDeviceProp prop{};
    cudaFuncAttributes attributes{};
    int blocks = 0;
    if (status == cudaSuccess)
        status = cudaGetDeviceProperties(&prop, device);
    if (status == cudaSuccess && (threads < 1 || threads > prop.maxThreadsPerBlock))
        status = cudaErrorInvalidValue;
    if (status == cudaSuccess)
        status = multi ? cudaFuncGetAttributes(&attributes, multi_kernel) : cudaFuncGetAttributes(&attributes, solve_kernel);
    if (status == cudaSuccess)
        status = multi ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, multi_kernel, threads, 0) : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, solve_kernel, threads, 0);
    if (status != cudaSuccess)
        return report(status, error, capacity);
    *info = LambertDeviceInfo{};
    std::snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->major = prop.major;
    info->minor = prop.minor;
    info->multiprocessors = prop.multiProcessorCount;
    info->registers = attributes.numRegs;
    info->local_bytes = attributes.localSizeBytes;
    info->active_blocks = blocks;
    info->threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info->max_threads_per_block = attributes.maxThreadsPerBlock;
    status = cudaDriverGetVersion(&info->driver);
    if (status == cudaSuccess)
        status = cudaRuntimeGetVersion(&info->runtime);
    return report(status, error, capacity);
}
extern "C" int lb_cuda_create(size_t n, int device, int threads, void **handle, char *error,
                              size_t capacity) {
    if (!handle)
        return report(cudaErrorInvalidValue, error, capacity);
    *handle = nullptr;
    if (n > SIZE_MAX / sizeof(lambert::Result))
        return report(cudaErrorInvalidValue, error, capacity);
    LambertDeviceInfo info{};
    int code = lb_cuda_info(device, threads, 0, &info, error, capacity);
    if (code)
        return code;
    LambertDeviceInfo multi_info{};
    code = lb_cuda_info(device, threads, 1, &multi_info, error, capacity);
    if (code) return code;
    if (threads > info.max_threads_per_block || threads > multi_info.max_threads_per_block)
        return report(cudaErrorInvalidValue, error, capacity);
    Context *c = new (std::nothrow) Context;
    if (!c)
        return report(cudaErrorMemoryAllocation, error, capacity);
    c->device = device;
    c->threads = threads;
    c->capacity = n;
    cudaError_t status = cudaDeviceGetAttribute(&c->max_blocks, cudaDevAttrMaxGridDimX, device);
    if (status == cudaSuccess)
        status = cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess)
        status = cudaEventCreate(&c->begin);
    if (status == cudaSuccess)
        status = cudaEventCreate(&c->end);
    if (status == cudaSuccess && n)
        status = cudaMalloc(reinterpret_cast<void **>(&c->input), n * sizeof(lambert::Input));
    if (status == cudaSuccess && n)
        status = cudaMalloc(reinterpret_cast<void **>(&c->output), n * sizeof(lambert::Result));
    if (status != cudaSuccess) {
        release(c);
        return report(status, error, capacity);
    }
    *handle = c;
    return report(cudaSuccess, error, capacity);
}
extern "C" int lb_cuda_destroy(void *handle, char *error, size_t capacity) {
    if (!handle)
        return report(cudaSuccess, error, capacity);
    return report(release(static_cast<Context *>(handle)), error, capacity);
}
extern "C" int lb_cuda_upload(void *handle, const lambert::Input *input, size_t n, char *error,
                              size_t capacity) {
    auto c = static_cast<Context *>(handle);
    if (!c || n > c->capacity || (n && !input))
        return report(cudaErrorInvalidValue, error, capacity);
    c->count = 0;
    c->jacobian_ready = false;
    c->zero = false; c->multi = false;
    for (size_t i = 0; i < n; ++i) {
        if (input[i].revolutions == 0) c->zero = true;
        else c->multi = true;
    }
    cudaError_t status = cudaSetDevice(c->device);
    if (status == cudaSuccess && n)
        status = cudaMemcpyAsync(c->input, input, n * sizeof(lambert::Input),
                                 cudaMemcpyHostToDevice, c->stream);
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(c->stream);
    if (status == cudaSuccess)
        c->count = n;
    return report(status, error, capacity);
}
static int run(void *handle, const lambert::Options *options, const lambert::MinimumOptions *minimum_options, int repetitions,
               float *milliseconds, char *error, size_t capacity, bool with_jacobian) {
    auto c = static_cast<Context *>(handle);
    if (!c || !options || !minimum_options || !milliseconds || repetitions < 1)
        return report(cudaErrorInvalidValue, error, capacity);
    c->jacobian_ready = false;
    cudaError_t status = cudaSetDevice(c->device);
    if (status != cudaSuccess)
        return report(status, error, capacity);
    if (with_jacobian && c->count && !c->jacobian) {
        if (c->capacity > SIZE_MAX/sizeof(lambert::Jacobian))
            return report(cudaErrorInvalidValue,error,capacity);
        status = cudaMalloc(reinterpret_cast<void **>(&c->jacobian),c->capacity*sizeof(lambert::Jacobian));
        if (status != cudaSuccess) return report(status,error,capacity);
    }
    if (c->count == 0) {
        *milliseconds = 0.;
        c->jacobian_ready = with_jacobian;
        return report(cudaSuccess, error, capacity);
    }
    size_t blocks = (c->count + c->threads - 1) / c->threads;
    if (blocks > size_t(c->max_blocks))
        status = cudaErrorInvalidValue;
    if (status == cudaSuccess)
        status = cudaEventRecord(c->begin, c->stream);
    if (status != cudaSuccess)
        return report(status, error, capacity);
    for (int i = 0; i < repetitions; ++i) {
        if (c->zero) {
            solve_kernel<<<static_cast<unsigned>(blocks), c->threads, 0, c->stream>>>(
                c->input, c->output, c->count, *options);
            status = cudaGetLastError();
            if (status != cudaSuccess) return report(status, error, capacity);
        }
        if (c->multi) {
            multi_kernel<<<static_cast<unsigned>(blocks), c->threads, 0, c->stream>>>(
                c->input, c->output, c->count, *options, *minimum_options);
            status = cudaGetLastError();
            if (status != cudaSuccess) return report(status, error, capacity);
        }
        if (with_jacobian) {
            jacobian_kernel<<<static_cast<unsigned>(blocks),c->threads,0,c->stream>>>(
                c->input,c->output,c->jacobian,c->count);
            status = cudaGetLastError();
            if (status != cudaSuccess) return report(status,error,capacity);
        }
    }
    status = cudaEventRecord(c->end, c->stream);
    if (status == cudaSuccess)
        status = cudaEventSynchronize(c->end);
    float elapsed = 0.;
    if (status == cudaSuccess)
        status = cudaEventElapsedTime(&elapsed, c->begin, c->end);
    if (status == cudaSuccess)
        *milliseconds = elapsed / repetitions;
    c->jacobian_ready = status == cudaSuccess && with_jacobian;
    return report(status, error, capacity);
}
extern "C" int lb_cuda_run(void *h, const lambert::Options *o, const lambert::MinimumOptions *mo,
    int repetitions, float *ms, char *error, size_t capacity) {
    return run(h,o,mo,repetitions,ms,error,capacity,false);
}
extern "C" int lb_cuda_run_jacobian(void *h, const lambert::Options *o, const lambert::MinimumOptions *mo,
    int repetitions, float *ms, char *error, size_t capacity) {
    return run(h,o,mo,repetitions,ms,error,capacity,true);
}
extern "C" int lb_cuda_download_jacobian(void *handle, lambert::Jacobian *output, size_t n,
    char *error, size_t capacity) {
    auto c = static_cast<Context *>(handle);
    if (!c || !c->jacobian_ready || n != c->count || (n && !output))
        return report(cudaErrorInvalidValue,error,capacity);
    cudaError_t status = cudaSetDevice(c->device);
    if (status == cudaSuccess && n)
        status = cudaMemcpyAsync(output,c->jacobian,n*sizeof(lambert::Jacobian),cudaMemcpyDeviceToHost,c->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(c->stream);
    return report(status,error,capacity);
}
extern "C" int lb_cuda_download(void *handle, lambert::Result *output, size_t n, char *error,
                                size_t capacity) {
    auto c = static_cast<Context *>(handle);
    if (!c || n != c->count || (n && !output))
        return report(cudaErrorInvalidValue, error, capacity);
    cudaError_t status = cudaSetDevice(c->device);
    if (status == cudaSuccess && n)
        status = cudaMemcpyAsync(output, c->output, n * sizeof(lambert::Result),
                                 cudaMemcpyDeviceToHost, c->stream);
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(c->stream);
    return report(status, error, capacity);
}
