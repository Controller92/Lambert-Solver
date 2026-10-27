#include "api.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>
int main() {
    static_assert(std::is_standard_layout<lambert::Jacobian>::value,"Jacobian ABI must be plain layout");
    lambert::Input inputs[4] = {
        {{1,0,0},{0,1,0},1.5707963267948966,1,0,0,0},
        {{1,0,0},{0,1,0},15.,1,0,2,1},
        {{1,0,0},{1,2e-14,0},2e-14,1,0,0,0},
        {{1,0,0},{-1,0,0},3.,1,0,0,0}};
    lambert::Result results[4];lambert::Jacobian jac[4];
    lambert::Options options;lambert::MinimumOptions minimum;
    assert(lb_jacobian_size()==sizeof(lambert::Jacobian));
    assert(lb_cpu_jacobian(nullptr,nullptr,nullptr,0,&options,&minimum)==0);
    assert(lb_cpu_jacobian(inputs,results,jac,4,nullptr,&minimum)==1);
    assert(lb_cpu_jacobian(inputs,results,nullptr,4,&options,&minimum)==1);
    assert(lb_cpu_jacobian(inputs,results,jac,SIZE_MAX,&options,&minimum)==1);
    for (int repeat=0;repeat<20;++repeat) {
        assert(lb_cpu_jacobian(inputs,results,jac,4,&options,&minimum)==0);
        for (int i=0;i<3;++i) {
            assert(results[i].status==0 && jac[i].status==0);
            for (int a=0;a<6;++a) for(int b=0;b<7;++b) assert(std::isfinite(jac[i].values[a][b]));
        }
        assert(results[3].status!=0 && jac[3].status==1 && std::isnan(jac[3].values[0][0]));
    }
    std::cout<<"Jacobian address/undefined-behaviour sanitizer checks passed\n";
}
