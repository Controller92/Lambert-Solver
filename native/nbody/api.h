#pragma once
#include <cstdint>
extern "C" {
// SI, seconds from explicit TDB ET epoch, fixed inertial coordinate offset.
// Status: success=0, input=1, coverage=2, collision=3, step_floor=4,
// step_budget=5, nonfinite=6. Failed result states are NaN.
struct NBodyOptions {
 double rtol, atol_position, atol_velocity, atol_stm;
 double initial_step, minimum_step, maximum_step;
 int64_t maximum_steps;
};
struct NBodyResult {
 // smallest_distance includes rejected stages; it is NOT minimum altitude.
 double state[6], stm[36], time, smallest_distance;
 int64_t accepted, rejected, evaluations;
 int32_t status;
};
int32_t nbody_abi_version();
int32_t nbody_options_size();
int32_t nbody_result_size();
void* nbody_create(int64_t knots,int32_t bodies,const double* times,const double* states,
                   const double* mu,const double* radii,int32_t* status);
void nbody_destroy(void* context);
int32_t nbody_force(void* context,double time,const double* position,double* acceleration,double* gradient);
void nbody_propagate(void* context,double start,double end,const double* initial,
                     const NBodyOptions* options,NBodyResult* result);
}
