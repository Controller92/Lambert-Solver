#include "flyby_api.h"
extern "C" FlybyAbi fb_abi() { return {1,sizeof(flyby::Input),sizeof(flyby::Result)}; }
extern "C" int fb_cpu(const flyby::Input *in,flyby::Result *out,size_t n) {
    if ((n&&(!in||!out))||n>SIZE_MAX/sizeof(flyby::Result)) return 1;
    for (size_t i=0;i<n;++i) out[i]=flyby::solve(in[i]);
    return 0;
}
