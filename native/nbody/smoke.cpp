#include "api.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 double times[2]={-10,10},states[12]={},mu[1]={1},radii[1]={0};int32_t status;
 void* c=nbody_create(2,1,times,states,mu,radii,&status);assert(c&&status==0);
 NBodyOptions options{2e-13,1e-4,1e-8,1e-11,10.,1e-7,300.,1000000};
 double y[6]={1,0,0,0,1,0};NBodyResult result;
 for(int i=0;i<10;++i){
  nbody_propagate(c,-1.23456,.0010000002,y,&options,&result);assert(result.status==0);
  nbody_propagate(c,0,11,y,&options,&result);assert(result.status==2&&std::isnan(result.state[0]));
 }
 nbody_destroy(c);
 c=nbody_create(1,1,times,states,mu,radii,&status);assert(!c&&status==1);
 std::cout<<"N-body address/undefined-behaviour sanitizer checks passed\n";
}
