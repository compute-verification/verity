// RT-core census (round 3): triangle tests a second through OptiX (the driver's libnvoptix; headers from optix-dev).
// Scene "stack" is K triangles in the planes z = 1..K, all with the footprint (-1,-1), (1,-1), (-1,1), so every
// triangle's bounding box is [-1,1]^2 and no box excludes a ray inside it; scene "dup" is K copies of one triangle at
// z = 1, which no builder can separate into boxes, so its traversal has the fewest box tests a triangle test.
// A 2,048 x 2,048 launch of rays along +z from z = 0: in miss mode the rays start in (0.05, 0.95)^2, inside every box
// and outside every triangle, so each ray costs K triangle tests and the BVH's box tests and ends in the miss program;
// in hit mode they start inside the triangles and stop at the nearest. One TRACE instruction a ray (a warp instruction
// for 32 rays) starts the work, so K sets the tests a traced instruction drives. Prints one JSON line a (scene, build,
// mode, K): best of 5 launches by CUDA events. Two builds (fast trace, fast build) of each scene. With "clas", the
// stack scene is built as cluster acceleration structures instead (build_clas), on a pipeline that allows them.
// Usage: rt DEVICE_PROGRAMS.optixir [tri|clas]
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)
#define OX(x)                                                                    \
  do {                                                                           \
    OptixResult r_ = (x);                                                        \
    if (r_ != OPTIX_SUCCESS) {                                                   \
      fprintf(stderr, "ERR %s:%d %s: %s\n%s\n", __FILE__, __LINE__, #x, optixGetErrorString(r_), g_log); \
      exit(3);                                                                   \
    }                                                                            \
  } while (0)

static char g_log[4096];
static size_t g_logn;

struct Params {
  OptixTraversableHandle handle;
  unsigned int* errors;
  unsigned int width, height, mode;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Rec {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

static void log_cb(unsigned int level, const char* tag, const char* msg, void*) {
  if (level <= 2) fprintf(stderr, "optix [%u][%s] %s\n", level, tag, msg);
}

struct Gas {
  OptixTraversableHandle handle = 0;
  void* buf = nullptr;
  size_t bytes = 0;
};

static Gas build(OptixDeviceContext ctx, int k, unsigned int build_flags, bool dup) {
  std::vector<float3> v;
  for (int i = 0; i < k; i++) {
    const float z = dup ? 1.f : 1.f + i;
    v.push_back(make_float3(-1.f, -1.f, z));
    v.push_back(make_float3(1.f, -1.f, z));
    v.push_back(make_float3(-1.f, 1.f, z));
  }
  void* dv;
  CK(cudaMalloc(&dv, v.size() * sizeof(float3)));
  CK(cudaMemcpy(dv, v.data(), v.size() * sizeof(float3), cudaMemcpyHostToDevice));
  CUdeviceptr vb = (CUdeviceptr)dv;
  const unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput in = {};
  in.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  in.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  in.triangleArray.vertexStrideInBytes = sizeof(float3);
  in.triangleArray.numVertices = (unsigned int)v.size();
  in.triangleArray.vertexBuffers = &vb;
  in.triangleArray.flags = &flags;
  in.triangleArray.numSbtRecords = 1;
  OptixAccelBuildOptions opt = {};
  opt.buildFlags = build_flags;
  opt.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz;
  OX(optixAccelComputeMemoryUsage(ctx, &opt, &in, 1, &sz));
  void *tmp, *out;
  CK(cudaMalloc(&tmp, sz.tempSizeInBytes));
  CK(cudaMalloc(&out, sz.outputSizeInBytes));
  Gas g;
  OX(optixAccelBuild(ctx, 0, &opt, &in, 1, (CUdeviceptr)tmp, sz.tempSizeInBytes, (CUdeviceptr)out, sz.outputSizeInBytes,
                     &g.handle, nullptr, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(tmp));
  CK(cudaFree(dv));
  g.buf = out;
  g.bytes = sz.outputSizeInBytes;
  return g;
}

// The stack scene as cluster acceleration structures (Blackwell's cluster intersection engine): clusters of at most
// 85 triangles (255 unshared vertices; a cluster holds 256), then one GAS over the clusters.
static Gas build_clas(OptixDeviceContext ctx, int k, unsigned int cluster_flags) {
  const int per = 85, nc = (k + per - 1) / per;
  std::vector<float3> v;
  for (int i = 0; i < k; i++) {
    const float z = 1.f + i;
    v.push_back(make_float3(-1.f, -1.f, z));
    v.push_back(make_float3(1.f, -1.f, z));
    v.push_back(make_float3(-1.f, 1.f, z));
  }
  std::vector<unsigned int> idx(3 * per);
  for (int i = 0; i < 3 * per; i++) idx[i] = i;
  float3* dv;
  unsigned int* di;
  CK(cudaMalloc(&dv, v.size() * sizeof(float3)));
  CK(cudaMemcpy(dv, v.data(), v.size() * sizeof(float3), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&di, idx.size() * sizeof(unsigned int)));
  CK(cudaMemcpy(di, idx.data(), idx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  std::vector<OptixClusterAccelBuildInputTrianglesArgs> args(nc);
  for (int c = 0; c < nc; c++) {
    const int n = std::min(per, k - c * per);
    OptixClusterAccelBuildInputTrianglesArgs a = {};
    a.clusterId = c;
    a.triangleCount = n;
    a.vertexCount = 3 * n;
    a.indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_32BIT;
    a.basePrimitiveInfo.primitiveFlags = OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT;
    a.indexBuffer = (CUdeviceptr)di;
    a.vertexBuffer = (CUdeviceptr)(dv + 3 * per * c);
    args[c] = a;
  }
  void* dargs;
  CK(cudaMalloc(&dargs, nc * sizeof(args[0])));
  CK(cudaMemcpy(dargs, args.data(), nc * sizeof(args[0]), cudaMemcpyHostToDevice));
  OptixClusterAccelBuildInput in = {};
  in.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES;
  in.triangles.flags = (OptixClusterAccelBuildFlags)cluster_flags;
  in.triangles.maxArgCount = nc;
  in.triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  in.triangles.maxUniqueSbtIndexCountPerArg = 1;
  in.triangles.maxTriangleCountPerArg = per;
  in.triangles.maxVertexCountPerArg = 3 * per;
  in.triangles.maxTotalTriangleCount = k;
  in.triangles.maxTotalVertexCount = 3 * k;
  OptixAccelBufferSizes sz;
  OX(optixClusterAccelComputeMemoryUsage(ctx, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS, &in, &sz));
  void *clas, *tmp, *handles;
  CK(cudaMalloc(&clas, sz.outputSizeInBytes));
  CK(cudaMalloc(&tmp, sz.tempSizeInBytes));
  CK(cudaMalloc(&handles, nc * sizeof(CUdeviceptr)));
  OptixClusterAccelBuildModeDesc md = {};
  md.mode = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
  md.implicitDest.outputBuffer = (CUdeviceptr)clas;
  md.implicitDest.outputBufferSizeInBytes = sz.outputSizeInBytes;
  md.implicitDest.tempBuffer = (CUdeviceptr)tmp;
  md.implicitDest.tempBufferSizeInBytes = sz.tempSizeInBytes;
  md.implicitDest.outputHandlesBuffer = (CUdeviceptr)handles;
  md.implicitDest.outputHandlesStrideInBytes = sizeof(CUdeviceptr);
  OX(optixClusterAccelBuild(ctx, 0, &md, &in, (CUdeviceptr)dargs, 0, 0));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(tmp));
  const size_t clas_bytes = sz.outputSizeInBytes;

  OptixClusterAccelBuildInputClustersArgs ga = {};
  ga.clusterHandlesCount = nc;
  ga.clusterHandlesBufferStrideInBytes = sizeof(CUdeviceptr);
  ga.clusterHandlesBuffer = (CUdeviceptr)handles;
  void* dga;
  CK(cudaMalloc(&dga, sizeof ga));
  CK(cudaMemcpy(dga, &ga, sizeof ga, cudaMemcpyHostToDevice));
  OptixClusterAccelBuildInput gin = {};
  gin.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
  gin.clusters.flags = (OptixClusterAccelBuildFlags)cluster_flags;
  gin.clusters.maxArgCount = 1;
  gin.clusters.maxTotalClusterCount = nc;
  gin.clusters.maxClusterCountPerArg = nc;
  OX(optixClusterAccelComputeMemoryUsage(ctx, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS, &gin, &sz));
  void *gas, *ghandle;
  CK(cudaMalloc(&gas, sz.outputSizeInBytes));
  CK(cudaMalloc(&tmp, sz.tempSizeInBytes));
  CK(cudaMalloc(&ghandle, sizeof(OptixTraversableHandle)));
  md = {};
  md.mode = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
  md.implicitDest.outputBuffer = (CUdeviceptr)gas;
  md.implicitDest.outputBufferSizeInBytes = sz.outputSizeInBytes;
  md.implicitDest.tempBuffer = (CUdeviceptr)tmp;
  md.implicitDest.tempBufferSizeInBytes = sz.tempSizeInBytes;
  md.implicitDest.outputHandlesBuffer = (CUdeviceptr)ghandle;
  md.implicitDest.outputHandlesStrideInBytes = sizeof(OptixTraversableHandle);
  OX(optixClusterAccelBuild(ctx, 0, &md, &gin, (CUdeviceptr)dga, 0, 0));
  CK(cudaDeviceSynchronize());
  Gas g;
  CK(cudaMemcpy(&g.handle, ghandle, sizeof g.handle, cudaMemcpyDeviceToHost));
  CK(cudaFree(tmp));
  CK(cudaFree(ghandle));
  CK(cudaFree(dga));
  CK(cudaFree(dargs));
  CK(cudaFree(di));
  CK(cudaFree(dv));
  // The GAS references the clusters, so both stay allocated for the process's life.
  g.buf = gas;
  g.bytes = clas_bytes + sz.outputSizeInBytes;
  return g;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: rt DEVICE_PROGRAMS.optixir [tri|clas]\n");
    return 2;
  }
  const bool clas = argc > 2 && std::string(argv[2]) == "clas";
  std::ifstream f(argv[1], std::ios::binary);
  std::string ir((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (ir.empty()) {
    fprintf(stderr, "empty %s\n", argv[1]);
    return 2;
  }
  CK(cudaFree(0));
  OX(optixInit());
  OptixDeviceContextOptions co = {};
  co.logCallbackFunction = log_cb;
  co.logCallbackLevel = 2;
  OptixDeviceContext ctx;
  OX(optixDeviceContextCreate(0, &co, &ctx));

  OptixModuleCompileOptions mco = {};
  mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions pco = {};
  pco.usesMotionBlur = 0;
  pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pco.numPayloadValues = 1;
  pco.numAttributeValues = 2;
  pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pco.pipelineLaunchParamsVariableName = "params";
  pco.usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  pco.allowClusteredGeometry = clas;
  OptixModule mod;
  g_logn = sizeof g_log;
  OX(optixModuleCreate(ctx, &mco, &pco, ir.data(), ir.size(), g_log, &g_logn, &mod));

  OptixProgramGroupDesc d[3] = {};
  d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  d[0].raygen.module = mod;
  d[0].raygen.entryFunctionName = "__raygen__rg";
  d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  d[1].miss.module = mod;
  d[1].miss.entryFunctionName = "__miss__ms";
  d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  d[2].hitgroup.moduleCH = mod;
  d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OptixProgramGroupOptions pgo = {};
  OptixProgramGroup pg[3];
  g_logn = sizeof g_log;
  OX(optixProgramGroupCreate(ctx, d, 3, &pgo, g_log, &g_logn, pg));
  OptixPipelineLinkOptions plo = {};
  plo.maxTraceDepth = 1;
  OptixPipeline pipe;
  g_logn = sizeof g_log;
  OX(optixPipelineCreate(ctx, &pco, &plo, pg, 3, g_log, &g_logn, &pipe));
  OptixStackSizes ss = {};
  for (auto& p : pg) OX(optixUtilAccumulateStackSizes(p, &ss, pipe));
  unsigned int dct, dcs, cont;
  OX(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dct, &dcs, &cont));
  OX(optixPipelineSetStackSize(pipe, dct, dcs, cont, 1));

  Rec rec[3];
  for (int i = 0; i < 3; i++) OX(optixSbtRecordPackHeader(pg[i], &rec[i]));
  Rec* drec;
  CK(cudaMalloc(&drec, sizeof rec));
  CK(cudaMemcpy(drec, rec, sizeof rec, cudaMemcpyHostToDevice));
  OptixShaderBindingTable sbt = {};
  sbt.raygenRecord = (CUdeviceptr)&drec[0];
  sbt.missRecordBase = (CUdeviceptr)&drec[1];
  sbt.missRecordStrideInBytes = sizeof(Rec);
  sbt.missRecordCount = 1;
  sbt.hitgroupRecordBase = (CUdeviceptr)&drec[2];
  sbt.hitgroupRecordStrideInBytes = sizeof(Rec);
  sbt.hitgroupRecordCount = 1;

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  unsigned int* derr;
  CK(cudaMalloc(&derr, sizeof(unsigned int)));
  Params* dpar;
  CK(cudaMalloc(&dpar, sizeof(Params)));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const unsigned int W = 2048, H = 2048;
  const int ks[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  // A spatial-split builder could fit a triangle with boxes that exclude the miss rays; the fast-build variant
  // doesn't split, so agreement between the two says the K tests a ray happen.
  const struct {
    const char* name;
    unsigned int flags, cluster_flags;
  } builds[] = {{"fast_trace", OPTIX_BUILD_FLAG_PREFER_FAST_TRACE, OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE},
                {"fast_build", OPTIX_BUILD_FLAG_PREFER_FAST_BUILD, OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_BUILD}};
  const std::vector<std::string> scenes = clas ? std::vector<std::string>{"clas"}
                                               : std::vector<std::string>{"stack", "dup"};
  for (const std::string& scene_s : scenes)
  for (const auto& b : builds)
  for (int k : ks) {
    const char* scene = scene_s.c_str();
    Gas g = clas ? build_clas(ctx, k, b.cluster_flags) : build(ctx, k, b.flags, scene_s == "dup");
    for (unsigned int mode = 0; mode <= 1; mode++) {
      Params p{g.handle, derr, W, H, mode};
      CK(cudaMemcpy(dpar, &p, sizeof p, cudaMemcpyHostToDevice));
      double best = 1e30;
      unsigned int errs = 0;
      for (int r = 0; r < 6; r++) {
        CK(cudaMemset(derr, 0, sizeof(unsigned int)));
        CK(cudaEventRecord(e0));
        OX(optixLaunch(pipe, 0, (CUdeviceptr)dpar, sizeof(Params), &sbt, W, H, 1));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(&errs, derr, sizeof errs, cudaMemcpyDeviceToHost));
        if (r > 0 && ms < best) best = ms;
      }
      const double rays = (double)W * H, rps = rays / (best * 1e-3);
      printf("{\"probe\":\"rt\",\"scene\":\"%s\",\"build\":\"%s\",\"mode\":\"%s\",\"k\":%d,\"rays\":%.0f,\"ms\":%.4f,"
             "\"rays_per_s\":%.5g,\"triangle_tests_per_s\":%.5g,\"rays_per_sm_per_s\":%.5g,\"errors\":%u,"
             "\"gas_bytes\":%zu,\"sms\":%d,\"optix_abi\":%d}\n",
             scene, b.name, mode ? "hit" : "miss", k, rays, best, rps, mode ? rps : rps * k,
             rps / prop.multiProcessorCount, errs, g.bytes, prop.multiProcessorCount, OPTIX_ABI_VERSION);
      fflush(stdout);
    }
    CK(cudaFree(g.buf));
  }
  return 0;
}
