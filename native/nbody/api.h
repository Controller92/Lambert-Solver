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
// Optional zonal J2: pole unit vectors at each existing ephemeris knot.
// Copies input; one J2 body per context. Zero disables.
int32_t nbody_set_j2(void* context,int32_t body,double j2,double radius,const double* poles);
void nbody_destroy(void* context);
int32_t nbody_force(void* context,double time,const double* position,double* acceleration,double* gradient);
// Integrate positions relative to a supplied constant inertial origin. This
// prevents loss of small periapsis coordinates when adding planetary distances.
void nbody_propagate_centered(void* context,double start,double end,const double* initial,
                             const double* origin,const NBodyOptions* options,NBodyResult* result);
void nbody_propagate(void* context,double start,double end,const double* initial,
                     const NBodyOptions* options,NBodyResult* result);
}
