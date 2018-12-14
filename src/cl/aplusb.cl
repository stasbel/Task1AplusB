#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 8

__kernel void aplusb(__global const float *as, __global const float *bs, __global float *cs, unsigned int n) {
    size_t g_id = get_global_id(0);

    if (g_id >= n) {
        return;
    }

    cs[g_id] = as[g_id] + bs[g_id];
}
