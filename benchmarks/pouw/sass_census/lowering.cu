// The lowering probe's cases (lowering.sh): one kernel a case, each the PTX whose SASS a coverage row
// (benchmarks/pouw/machine_model/data/cover/rows.toml) cites. Compiled, never run.
#include <cstdint>
typedef uint32_t u32;

#ifdef CASE_f32x2
extern "C" __global__ void k(float2* o, const float2* a, const float2* b) {
  float2 x = a[threadIdx.x], y = b[threadIdx.x], r;
  asm("{.reg .b64 p,q,s; mov.b64 p,{%2,%3}; mov.b64 q,{%4,%5}; fma.rn.f32x2 s,p,q,p; mov.b64 {%0,%1},s;}"
      : "=f"(r.x), "=f"(r.y) : "f"(x.x), "f"(x.y), "f"(y.x), "f"(y.y));
  o[threadIdx.x] = r;
}
#endif

#ifdef CASE_multimem
extern "C" __global__ void k(float* o, float* a) {
  float r;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(r) : "l"(a));
  o[threadIdx.x] = r;
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(a), "r"(1));
}
#endif

#ifdef CASE_vabsdiff4
extern "C" __global__ void k(u32* o, u32 x, u32 y) {
  u32 r;
  asm("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(threadIdx.x), "r"(y));
  o[threadIdx.x] = r;
}
#endif

#ifdef CASE_texture
extern "C" __global__ void k(float4* o, cudaTextureObject_t t, float u, float v) {
  o[threadIdx.x] = tex2D<float4>(t, u + threadIdx.x, v);
  o[threadIdx.x + 64] = tex2Dgather<float4>(t, u, v, 1);
  o[threadIdx.x + 128] = tex2DGrad<float4>(t, u, v, make_float2(1, 0), make_float2(0, 1));
  o[threadIdx.x + 192] = tex2DLod<float4>(t, u, v, u);
  int w;
  asm("txq.width.b32 %0, [%1];" : "=r"(w) : "l"(t));
  o[0].x = w;
}
#endif

#ifdef CASE_surface
extern "C" __global__ void k(int* o, cudaSurfaceObject_t s) {
  int v;
  surf2Dread(&v, s, threadIdx.x * 4, 0);
  surf2Dwrite(v + 1, s, threadIdx.x * 4, 1);
  asm volatile("sured.b.add.2d.u32.trap [%0, {%1,%2}], %3;" :: "l"(s), "r"(0), "r"(0), "r"(v));
}
#endif

#ifdef CASE_match
extern "C" __global__ void k(u32* o, u32 x) {
  u32 m, a;
  asm("match.any.sync.b32 %0, %1, 0xffffffff;" : "=r"(m) : "r"(x + threadIdx.x));
  asm("match.all.sync.b32 %0, %1, 0xffffffff;" : "=r"(a) : "r"(x ^ threadIdx.x));
  o[threadIdx.x] = m ^ a;
}
#endif

#ifdef CASE_f16acc
extern "C" __global__ void k(u32* o, const u32* a) {
  u32 d0 = a[6], d1 = a[7];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d0), "+r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]));
  o[threadIdx.x * 2] = d0; o[threadIdx.x * 2 + 1] = d1;
}
#endif

#ifdef CASE_b1xor
extern "C" __global__ void k(int* o, const u32* a) {
  int d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif

#ifdef CASE_fp6
extern "C" __global__ void k(float* o, const u32* a) {
  float d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif

#ifdef CASE_qmma_sf
extern "C" __global__ void k(float* o, const u32* a) {
  float d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m3.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif

#ifdef CASE_omma_sp
extern "C" __global__ void k(float* o, const u32* a) {
  float d[4] = {0, 0, 0, 0};
  asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%14}, {0, 0};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[9]), "r"(a[10]), "r"(a[8]),
                 "r"(a[6]), "r"(a[7]));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif

#ifdef CASE_tcgen05
extern "C" __global__ void k(u32* o) {
  u32 v;
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(0));
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(0));
  o[threadIdx.x] = v;
}
#endif

#ifdef CASE_wgmma
extern "C" __global__ void k(float* o, uint64_t da, uint64_t db) {
  float d[4] = {0, 0, 0, 0};
  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif

#ifdef CASE_dmma
extern "C" __global__ void k(double* o, const double* a) {
  double d0 = 0, d1 = 0;
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d0), "+d"(d1) : "d"(a[0]), "d"(a[1]));
  o[threadIdx.x * 2] = d0; o[threadIdx.x * 2 + 1] = d1;
}
#endif

#ifdef CASE_s4
extern "C" __global__ void k(int* o, const u32* a) {
  int d[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]));
  for (int i = 0; i < 4; i++) o[threadIdx.x * 4 + i] = d[i];
}
#endif
