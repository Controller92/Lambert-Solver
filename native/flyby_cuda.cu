#include <cuda_runtime.h>
#include <cstdio>
#include <new>
#include "flyby_api.h"
namespace {
__global__ void encounter_kernel(const flyby::Input *in,flyby::Result *out,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) out[i]=flyby::solve(in[i]);
}
struct Context {
    flyby::Input *in=nullptr; flyby::Result *out=nullptr;
    cudaStream_t stream=nullptr; cudaEvent_t begin=nullptr,end=nullptr;
    size_t capacity=0,count=0; int device=0,threads=128,max_blocks=0;
    bool ran=false;
};
int report(cudaError_t s,char *error,size_t cap) {
    if (error&&cap) std::snprintf(error,cap,"%s",cudaGetErrorString(s));
    return int(s);
}
cudaError_t release(Context *c) {
    cudaError_t first=cudaSetDevice(c->device);
    auto keep=[&](cudaError_t s) { if (first==cudaSuccess&&s!=cudaSuccess) first=s; };
    if(c->stream) keep(cudaStreamSynchronize(c->stream));
    if(c->in) keep(cudaFree(c->in));
    if(c->out) keep(cudaFree(c->out));
    if(c->begin) keep(cudaEventDestroy(c->begin));
    if(c->end) keep(cudaEventDestroy(c->end));
    if(c->stream) keep(cudaStreamDestroy(c->stream));
    delete c; return first;
}
}
extern "C" int fb_cuda_info(int device,int threads,FlybyDeviceInfo *info,char *error,size_t cap) {
    if(!info) return report(cudaErrorInvalidValue,error,cap);
    auto s=cudaSetDevice(device); cudaDeviceProp prop{}; cudaFuncAttributes attr{}; int blocks=0;
    if(s==cudaSuccess) s=cudaGetDeviceProperties(&prop,device);
    if(s==cudaSuccess) s=cudaFuncGetAttributes(&attr,encounter_kernel);
    if(s==cudaSuccess&&(threads<1||threads>prop.maxThreadsPerBlock||threads>attr.maxThreadsPerBlock)) s=cudaErrorInvalidValue;
    if(s==cudaSuccess) s=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,encounter_kernel,threads,0);
    if(s!=cudaSuccess) return report(s,error,cap);
    *info=FlybyDeviceInfo{};
    std::snprintf(info->name,sizeof(info->name),"%s",prop.name);
    info->major=prop.major; info->minor=prop.minor; info->multiprocessors=prop.multiProcessorCount;
    info->registers=attr.numRegs; info->local_bytes=attr.localSizeBytes;
    info->active_blocks=blocks; info->threads_per_sm=prop.maxThreadsPerMultiProcessor;
    info->max_threads_per_block=attr.maxThreadsPerBlock;
    s=cudaDriverGetVersion(&info->driver);
    if(s==cudaSuccess) s=cudaRuntimeGetVersion(&info->runtime);
    return report(s,error,cap);
}
extern "C" int fb_cuda_create(size_t n,int device,int threads,void **handle,char *error,size_t cap) {
    if (!handle) return report(cudaErrorInvalidValue,error,cap);
    *handle=nullptr;
    if (n>SIZE_MAX/sizeof(flyby::Result)||n>SIZE_MAX/sizeof(flyby::Input)) return report(cudaErrorInvalidValue,error,cap);
    cudaError_t s=cudaSetDevice(device);
    cudaDeviceProp prop{}; cudaFuncAttributes attr{};
    if(s==cudaSuccess) s=cudaGetDeviceProperties(&prop,device);
    if(s==cudaSuccess) s=cudaFuncGetAttributes(&attr,encounter_kernel);
    if(s==cudaSuccess&&(threads<1||threads>prop.maxThreadsPerBlock||threads>attr.maxThreadsPerBlock)) s=cudaErrorInvalidValue;
    if(s!=cudaSuccess) return report(s,error,cap);
    auto c=new(std::nothrow) Context;
    if(!c) return report(cudaErrorMemoryAllocation,error,cap);
    c->device=device; c->threads=threads; c->capacity=n;
    s=cudaDeviceGetAttribute(&c->max_blocks,cudaDevAttrMaxGridDimX,device);
    if(s==cudaSuccess) s=cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking);
    if(s==cudaSuccess) s=cudaEventCreate(&c->begin);
    if(s==cudaSuccess) s=cudaEventCreate(&c->end);
    if(s==cudaSuccess&&n) s=cudaMalloc(reinterpret_cast<void**>(&c->in),n*sizeof(flyby::Input));
    if(s==cudaSuccess&&n) s=cudaMalloc(reinterpret_cast<void**>(&c->out),n*sizeof(flyby::Result));
    if(s!=cudaSuccess) { release(c); return report(s,error,cap); }
    *handle=c; return report(cudaSuccess,error,cap);
}
extern "C" int fb_cuda_destroy(void *h,char *error,size_t cap) {
    return report(h?release(static_cast<Context*>(h)):cudaSuccess,error,cap);
}
extern "C" int fb_cuda_upload(void *h,const flyby::Input *in,size_t n,char *error,size_t cap) {
    auto c=static_cast<Context*>(h);
    if(!c||n>c->capacity||(n&&!in)) return report(cudaErrorInvalidValue,error,cap);
    c->count=0; c->ran=false;
    auto s=cudaSetDevice(c->device);
    if(s==cudaSuccess&&n) s=cudaMemcpyAsync(c->in,in,n*sizeof(flyby::Input),cudaMemcpyHostToDevice,c->stream);
    if(s==cudaSuccess) s=cudaStreamSynchronize(c->stream);
    if(s==cudaSuccess) c->count=n;
    return report(s,error,cap);
}
extern "C" int fb_cuda_run(void *h,int repetitions,float *milliseconds,char *error,size_t cap) {
    auto c=static_cast<Context*>(h);
    if(!c||repetitions<1||!milliseconds) return report(cudaErrorInvalidValue,error,cap);
    c->ran=false;
    auto s=cudaSetDevice(c->device);
    if(s!=cudaSuccess) return report(s,error,cap);
    if(!c->count) { *milliseconds=0; c->ran=true; return report(cudaSuccess,error,cap); }
    size_t blocks=(c->count+c->threads-1)/c->threads;
    if(blocks>size_t(c->max_blocks)) return report(cudaErrorInvalidValue,error,cap);
    s=cudaEventRecord(c->begin,c->stream);
    for(int i=0;s==cudaSuccess&&i<repetitions;++i) {
        encounter_kernel<<<unsigned(blocks),c->threads,0,c->stream>>>(c->in,c->out,c->count);
        s=cudaGetLastError();
    }
    if(s==cudaSuccess) s=cudaEventRecord(c->end,c->stream);
    if(s==cudaSuccess) s=cudaEventSynchronize(c->end);
    float ms=0;
    if(s==cudaSuccess) s=cudaEventElapsedTime(&ms,c->begin,c->end);
    if(s==cudaSuccess) { *milliseconds=ms/repetitions; c->ran=true; }
    return report(s,error,cap);
}
extern "C" int fb_cuda_download(void *h,flyby::Result *out,size_t n,char *error,size_t cap) {
    auto c=static_cast<Context*>(h);
    if(!c||!c->ran||n!=c->count||(n&&!out)) return report(cudaErrorInvalidValue,error,cap);
    auto s=cudaSetDevice(c->device);
    if(s==cudaSuccess&&n) s=cudaMemcpyAsync(out,c->out,n*sizeof(flyby::Result),cudaMemcpyDeviceToHost,c->stream);
    if(s==cudaSuccess) s=cudaStreamSynchronize(c->stream);
    return report(s,error,cap);
}
