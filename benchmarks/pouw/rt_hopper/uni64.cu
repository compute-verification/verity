// Warp-uniform 64-bit arithmetic, to see which uniform-datapath opcodes sm_90a writes a UR pair with
// (A6's ω for a uniform instruction). Every value but the store's lane offset is uniform across the warp.
__global__ void uni_wide(unsigned long long* out, unsigned a, unsigned b, int n) {
  unsigned long long acc = 0;
  for (int i = 0; i < n; i++) acc += (unsigned long long)(a + i) * (unsigned long long)(b ^ (unsigned)i);
  out[threadIdx.x] = acc + threadIdx.x;
}

__global__ void uni_add64(unsigned long long* out, unsigned long long a, unsigned long long b, int n) {
  unsigned long long acc = a;
  for (int i = 0; i < n; i++) acc = acc + (b << (i & 7)) + (unsigned long long)i;
  out[threadIdx.x] = acc + threadIdx.x;
}
