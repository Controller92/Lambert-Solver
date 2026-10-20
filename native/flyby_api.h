#pragma once
#include "flyby.hpp"
#include <cstddef>
struct FlybyAbi { uint32_t version, input_size, result_size; };
struct FlybyDeviceInfo {
    char name[256];
    int32_t major,minor,driver,runtime,multiprocessors,registers,local_bytes;
    int32_t active_blocks,threads_per_sm,max_threads_per_block;
};
extern "C" {
int fb_cuda_info(int,int,FlybyDeviceInfo*,char*,size_t);
FlybyAbi fb_abi();
int fb_cpu(const flyby::Input*,flyby::Result*,size_t);
int fb_cuda_create(size_t,int,int,void**,char*,size_t);
int fb_cuda_destroy(void*,char*,size_t);
int fb_cuda_upload(void*,const flyby::Input*,size_t,char*,size_t);
int fb_cuda_run(void*,int,float*,char*,size_t);
int fb_cuda_download(void*,flyby::Result*,size_t,char*,size_t);
}
