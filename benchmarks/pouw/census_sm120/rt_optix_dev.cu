// Device programs for rt_optix.cu: one ray a launch index along +z through the scene's shared triangle footprint.
// Mode 0 (miss): the ray passes inside every triangle's bounding box and outside every triangle, so the RT core tests
// all K triangles and reports a miss. Mode 1 (hit): the ray passes inside the triangles and hits the nearest.
// A ray whose outcome isn't the mode's counts an error.
#include <optix.h>

struct Params {
  OptixTraversableHandle handle;
  unsigned int* errors;
  unsigned int width, height, mode;
};

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
  const uint3 i = optixGetLaunchIndex();
  const float u = (i.x + 0.5f) / params.width, v = (i.y + 0.5f) / params.height;
  const float x = params.mode ? -0.9f + 0.4f * u : 0.05f + 0.9f * u;
  const float y = params.mode ? -0.9f + 0.4f * v : 0.05f + 0.9f * v;
  unsigned int p0 = 0;
  optixTrace(params.handle, make_float3(x, y, 0.f), make_float3(0.f, 0.f, 1.f), 0.f, 1e16f, 0.f, OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0);
  if (p0 != (params.mode ? 2u : 1u)) atomicAdd(params.errors, 1u);
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(1u); }

extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(2u); }
