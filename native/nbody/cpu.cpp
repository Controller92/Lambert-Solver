#include "api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
namespace {
struct Context {
 int64_t knots; int bodies;
 std::vector<double> times,states,mu,radii;
 int force(double t,const double* r,double* acc,double* G,double& distance) const {
  if(!std::isfinite(t)||t<times.front()||t>times.back())return 2;
  auto it=std::upper_bound(times.begin(),times.end(),t);
  int64_t k=std::min<int64_t>(knots-2,std::max<int64_t>(0,it-times.begin()-1));
  double h=times[k+1]-times[k],u=(t-times[k])/h;
  double h00=(2*u-3)*u*u+1,h10=((u-2)*u+1)*u,h01=(-2*u+3)*u*u,h11=(u-1)*u*u;
  std::fill(acc,acc+3,0.);std::fill(G,G+9,0.);
  for(int b=0;b<bodies;++b){
   const double* a=&states[(k*bodies+b)*6];const double* c=&states[((k+1)*bodies+b)*6];
   double d[3],d2=0;
   for(int i=0;i<3;++i){d[i]=h00*a[i]+h10*h*a[i+3]+h01*c[i]+h11*h*c[i+3]-r[i];d2+=d[i]*d[i];}
   double dist=std::sqrt(d2);distance=std::min(distance,dist);
   if(!(dist>radii[b])||!std::isfinite(dist))return 3;
   double q=mu[b]/(d2*dist);
   for(int i=0;i<3;++i){acc[i]+=q*d[i];for(int j=0;j<3;++j)G[i*3+j]+=q*(3*d[i]*d[j]/d2-(i==j));}
  }
  for(int i=0;i<3;++i)if(!std::isfinite(acc[i]))return 6;
  for(int i=0;i<9;++i)if(!std::isfinite(G[i]))return 6;
  return 0;
 }
 int rhs(double t,const double* y,double* f,double& distance) const {
  double G[9];int s=force(t,y,f+3,G,distance);if(s)return s;
  std::copy(y+3,y+6,f);
  for(int j=0;j<6;++j)for(int i=0;i<3;++i){
   f[6+i*6+j]=y[6+(i+3)*6+j];double value=0;
   for(int k=0;k<3;++k)value+=G[i*3+k]*y[6+k*6+j];
   f[6+(i+3)*6+j]=value;
  }
  return 0;
 }
};
// Dormand-Prince 5(4) tableau; no fast-math, no contracted operations.
constexpr double C[7]={0,1./5,3./10,4./5,8./9,1,1};
constexpr double A[7][7]={{},{1./5},{3./40,9./40},{44./45,-56./15,32./9},
 {19372./6561,-25360./2187,64448./6561,-212./729},
 {9017./3168,-355./33,46732./5247,49./176,-5103./18656},
 {35./384,0,500./1113,125./192,-2187./6784,11./84}};
constexpr double B[7]={35./384,0,500./1113,125./192,-2187./6784,11./84,0};
constexpr double L[7]={5179./57600,0,7571./16695,393./640,-92097./339200,187./2100,1./40};
}
extern "C" {
int32_t nbody_abi_version(){return 1;}
int32_t nbody_options_size(){return sizeof(NBodyOptions);}
int32_t nbody_result_size(){return sizeof(NBodyResult);}
void* nbody_create(int64_t knots,int32_t bodies,const double* times,const double* states,
                   const double* mu,const double* radii,int32_t* status){
 if(!status)return nullptr;
 *status=1;
 if(knots<2||knots>10000000||bodies<1||bodies>128||!times||!states||!mu||!radii)return nullptr;
 for(int64_t i=0;i<knots;++i)if(!std::isfinite(times[i])||(i&&times[i]<=times[i-1]))return nullptr;
 for(int i=0;i<bodies;++i)if(!std::isfinite(mu[i])||mu[i]<=0||!std::isfinite(radii[i])||radii[i]<0)return nullptr;
 for(int64_t i=0;i<knots*bodies*6;++i)if(!std::isfinite(states[i]))return nullptr;
 try{
  auto* c=new Context{knots,bodies,std::vector<double>(times,times+knots),
   std::vector<double>(states,states+knots*bodies*6),std::vector<double>(mu,mu+bodies),std::vector<double>(radii,radii+bodies)};
  *status=0;return c;
 }catch(...){*status=6;return nullptr;}
}
void nbody_destroy(void* context){delete static_cast<Context*>(context);}
int32_t nbody_force(void* context,double time,const double* position,double* acc,double* G){
 if(!context||!position||!acc||!G)return 1;
 double distance=INFINITY;
 for(int i=0;i<3;++i)if(!std::isfinite(position[i]))return 1;
 int s=static_cast<Context*>(context)->force(time,position,acc,G,distance);
 if(s){std::fill(acc,acc+3,NAN);std::fill(G,G+9,NAN);}return s;
}
void nbody_propagate(void* context,double start,double end,const double* initial,const NBodyOptions* o,NBodyResult* result){
 if(!result)return;
 NBodyResult& out=*result;out={};out.status=1;out.time=start;out.smallest_distance=INFINITY;
 std::fill(out.state,out.state+6,NAN);std::fill(out.stm,out.stm+36,NAN);
 if(!context||!initial||!o)return;
 double positive[]={o->rtol,o->atol_position,o->atol_velocity,o->atol_stm,o->initial_step,o->minimum_step,o->maximum_step};
 for(double v:positive)if(!std::isfinite(v)||v<=0)return;
 if(o->rtol<2e-15||o->minimum_step>o->maximum_step||o->maximum_steps<1)return;
 auto& c=*static_cast<Context*>(context);
 if(!std::isfinite(start)||!std::isfinite(end)||std::min(start,end)<c.times.front()||std::max(start,end)>c.times.back()){out.status=2;return;}
 std::array<double,42> y{},stage{},trial{};double K[7][42];
 for(int i=0;i<6;++i){if(!std::isfinite(initial[i]))return;y[i]=initial[i];}
 for(int i=0;i<6;++i)y[6+i*6+i]=1;
 double initial_derivative[42];
 int initial_status=c.rhs(start,y.data(),initial_derivative,out.smallest_distance);++out.evaluations;
 if(initial_status){out.status=initial_status;return;}
 double t=start,direction=end>=start?1.:-1.;double h=direction*std::min(o->initial_step,o->maximum_step);
 for(int64_t step=0;step<o->maximum_steps;++step){
  if(t==end){out.status=0;std::copy(y.begin(),y.begin()+6,out.state);std::copy(y.begin()+6,y.end(),out.stm);out.time=t;return;}
  double remaining=std::abs(end-t);bool final_step=std::abs(h)>=remaining;
  h=direction*std::min(std::abs(h),remaining);
  if(t+h==t||(std::abs(h)<o->minimum_step&&remaining>=o->minimum_step)){out.status=4;out.time=t;return;}
  int stage_status=0;
  for(int s=0;s<7;++s){
   for(int i=0;i<42;++i){double v=0;for(int j=0;j<s;++j)v+=A[s][j]*K[j][i];stage[i]=y[i]+h*v;}
   double ts=C[s]==1?(final_step?end:t+h):t+C[s]*h;
   if(direction>0&&ts>end)ts=end;
   if(direction<0&&ts<end)ts=end;
   stage_status=c.rhs(ts,stage.data(),K[s],out.smallest_distance);++out.evaluations;
   if(stage_status){if(stage_status==3&&s>0)break;out.status=stage_status;out.time=t;return;}
  }
  if(stage_status==3){++out.rejected;h*=.5;if(std::abs(h)<o->minimum_step){out.status=3;out.time=t;return;}continue;}
  double error=0;
  for(int i=0;i<42;++i){
   double high=0,err=0;for(int s=0;s<7;++s){high+=B[s]*K[s][i];err+=(B[s]-L[s])*K[s][i];}
   trial[i]=y[i]+h*high;double atol=i<3?o->atol_position:i<6?o->atol_velocity:o->atol_stm;
   double scale=atol+o->rtol*std::max(std::abs(y[i]),std::abs(trial[i]));error=std::max(error,std::abs(h*err)/scale);
   if(!std::isfinite(trial[i])||!std::isfinite(error)){out.status=6;out.time=t;return;}
  }
  if(error<=1){
   y=trial;t=final_step?end:t+h;++out.accepted;
   if(final_step){out.status=0;std::copy(y.begin(),y.begin()+6,out.state);std::copy(y.begin()+6,y.end(),out.stm);out.time=end;return;}
  }else ++out.rejected;
  double factor=error==0?5.:std::clamp(.9*std::pow(error,-.2),.1,5.);
  if(error>1)factor=std::min(1.,factor);
  h=direction*std::min(o->maximum_step,std::abs(h)*factor);
 }
 out.status=5;out.time=t;
}
}
