// RT2-HOPPER's (M′) probe on the H200 (sm_90a), for SLOT-LEMMA (#2755, note:20261019T0745Z-draft-fp-joint): does the
// legacy shallow sparse shape mma.sp m16n8k32 (s8.sp, IMMA.SP.16832) hold a full slot step, alone and beside an
// IADD3+FFMA lane? tensor_kinds/sm80.cu's own forms, kernels and timing, built for sm_90a: the dense m16n8k32 and the
// deep sparse m16n8k64 beside it, each alone at 2, 4 and 8 warps a SMSP (and with the loop thinned, U = 4) and
// interleaved with LANE_MIX (IADD3 then FFMA) at 8 to 64 lane ops a MMA (k_mixu, which has no block-time artifact:
// every warp issues both). A full slot step holds when the shallow form's cycles a MMA equal the deep form's.
#define GROUP 99
#define main sm80_main
#include "../tensor_kinds/sm80.cu"
#undef main

template <class F>
static void msp() {
  exact<F>();
  alone<F, 2>();
  alone<F, 4>();
  alone<F, 8>();
  alone<F, 4, 4>();
  alone<F, 8, 4>();
  mixu<F, 8, LANE_MIX>();
  mixu<F, 16, LANE_MIX>();
  mixu<F, 32, LANE_MIX>();
  mixu<F, 64, LANE_MIX>();
  mixu<F, 16, LANE_MIX, 8>();
  mixu<F, 32, LANE_MIX, 8>();
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  CK(cudaStreamCreate(&g_s));
  gk::Nvml nv(dev);
  g_nv = &nv;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  printf("{\"probe\":\"tensor_kinds_sm80\",\"group\":%d,\"device\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"gpu_uuid\":\"%s\","
         "\"nvml\":%s}\n",
         GROUP, p.name, p.major, p.minor, g_sms, nv.uuid().c_str(), nv.ok() ? "true" : "false");
  opalone<LANE_MIX>();
  msp<ImmaS8k32>();
  msp<ImmaSpS8k64>();
  msp<ImmaSpS8k32>();
  printf("{\"probe\":\"tensor_kinds_sm80\",\"group\":%d,\"done\":true,\"num_sms\":%d}\n", GROUP, g_sms);
  return 0;
}
