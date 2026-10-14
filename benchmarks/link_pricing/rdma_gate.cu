// RDMA through node 2's NIC (mlx5_0) between two RC QPs of one process in loopback, gated byte for byte:
//
//   rdma_gate --op write|read --src host|gpu --dst host|gpu [--mib M] [--chunk BYTES] [--seconds S] [--inline N]
//             [--gpu G] [--gid I] [--dev NAME] [--beside-read GIB [--read-hint normal|stream] [--discard 1 | --rewrite 1]]
//
// write: QP a writes src (its local memory) into dst (b's memory).  read: QP a reads src (b's memory) into dst (its
// local memory), so with --src gpu the NIC reads the GPU (GPUDirect RDMA TX) and with --dst gpu it writes the GPU.
// --inline N: N-byte writes whose payload the CPU copies into the work request (no DMA read of src; host src only).
// GPU memory is registered through nvidia_peermem (ibv_reg_mr), else as a dma-buf (ibv_reg_dmabuf_mr).
// The whole of dst is written once and compared with src (exit 3 on a mismatch), then the same operations repeat for
// `seconds` and the sustained rate is reported.  --beside-read GIB [--read-hint normal|stream]: meanwhile a kernel on
// --gpu reads GIB GiB over and over (plain or evict-first loads), and the operations repeat for as long as it runs;
// --discard 1: the reader also drops dst's lines from L2 without a write-back as it goes (--dst gpu); --rewrite 1: it
// keeps storing into src instead (--src gpu), so the NIC reads lines the SMs have just written.
// Output: JSON lines on stdout.
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define DIE(...)                      \
  do {                                \
    fprintf(stderr, __VA_ARGS__);     \
    fprintf(stderr, "\n");            \
    exit(2);                          \
  } while (0)
#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_ != cudaSuccess) DIE("CUDA %s at line %d: %s", cudaGetErrorString(e_), __LINE__, #x); \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// paths.cu's reader: with a ring, every 16th load drops one of its 128-byte lines from L2 without a write-back, or
// (REWRITE) stores 16 bytes into it, as a relay writing z just ahead of the NIC would
template <bool STREAM, bool REWRITE = false>
__global__ void k_read_passes(const uint4* __restrict__ src, size_t n, int passes, uint32_t* sink, char* ring,
                              size_t ring_lines) {
  uint32_t acc = 0, k = 0;
  size_t T = (size_t)gridDim.x * blockDim.x, d = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (int p = 0; p < passes; p++)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += T) {
      uint4 v = STREAM ? __ldcs(src + i) : src[i];
      acc ^= (v.x ^ v.y ^ v.z ^ v.w) + p;
      if (ring && (++k & 15) == 0) {
        if (REWRITE) ((uint4*)ring)[d % (ring_lines * 8)] = v;
        else asm volatile("discard.global.L2 [%0], 128;" ::"l"(ring + (d % ring_lines) * 128) : "memory");
        d += T;
      }
    }
  if (acc == 0x12345678u) *sink = acc;
}

struct Buf {
  void* p = nullptr;
  bool gpu = false;
  ibv_mr* mr = nullptr;
  const char* reg = "";
};

static Buf alloc(ibv_pd* pd, bool gpu, size_t B) {
  Buf b;
  b.gpu = gpu;
  int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (gpu) {
    CK(cudaMalloc(&b.p, B));
    b.mr = ibv_reg_mr(pd, b.p, B, acc);
    b.reg = "peermem";
    if (!b.mr) {
      int fd = -1;
      CUresult r = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)b.p, B, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
      if (r != CUDA_SUCCESS) DIE("cuMemGetHandleForAddressRange: %d", (int)r);
      b.mr = ibv_reg_dmabuf_mr(pd, 0, B, (uint64_t)b.p, fd, acc);
      b.reg = "dmabuf";
    }
  } else {
    if (posix_memalign(&b.p, 1 << 21, B)) DIE("posix_memalign");
    memset(b.p, 0, B);
    b.mr = ibv_reg_mr(pd, b.p, B, acc);
    b.reg = "host";
  }
  if (!b.mr) DIE("ibv_reg_mr (%s) failed: %s", b.reg, strerror(errno));
  return b;
}

static void fill(Buf& b, size_t B, uint32_t seed) {
  std::vector<uint32_t> h(B / 4);
  uint32_t x = seed;
  for (auto& w : h) w = (x = x * 1664525u + 1013904223u);
  if (b.gpu) CK(cudaMemcpy(b.p, h.data(), B, cudaMemcpyHostToDevice));
  else memcpy(b.p, h.data(), B);
}
static void zero(Buf& b, size_t B) {
  if (b.gpu) CK(cudaMemset(b.p, 0, B));
  else memset(b.p, 0, B);
}
static std::vector<uint8_t> host_copy(const Buf& b, size_t B) {
  std::vector<uint8_t> h(B);
  if (b.gpu) CK(cudaMemcpy(h.data(), b.p, B, cudaMemcpyDeviceToHost));
  else memcpy(h.data(), b.p, B);
  return h;
}

static void to_rts(ibv_qp* qp, uint32_t dqpn, const ibv_gid& gid, int gidx, ibv_mtu mtu) {
  ibv_qp_attr a;
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_INIT;
  a.port_num = 1;
  a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    DIE("INIT: %s", strerror(errno));
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = mtu;
  a.dest_qp_num = dqpn;
  a.rq_psn = 0;
  a.max_dest_rd_atomic = 16;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  a.ah_attr.grh.dgid = gid;
  a.ah_attr.grh.sgid_index = gidx;
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.port_num = 1;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    DIE("RTR: %s", strerror(errno));
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = 0;
  a.max_rd_atomic = 16;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                IBV_QP_MAX_QP_RD_ATOMIC))
    DIE("RTS: %s", strerror(errno));
}

int main(int argc, char** argv) {
  std::string op = "write", src_s = "host", dst_s = "host", dev = "mlx5_0", hint = "normal";
  size_t mib = 64, chunk = 1 << 20, read_gib = 0;
  bool discard = false, rewrite = false;
  double seconds = 3.0;
  int inl = 0, gpu = 0, gidx = 3;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--op") op = v;
    else if (k == "--src") src_s = v;
    else if (k == "--dst") dst_s = v;
    else if (k == "--mib") mib = strtoull(v.c_str(), 0, 10);
    else if (k == "--chunk") chunk = strtoull(v.c_str(), 0, 10);
    else if (k == "--seconds") seconds = atof(v.c_str());
    else if (k == "--inline") inl = atoi(v.c_str());
    else if (k == "--gpu") gpu = atoi(v.c_str());
    else if (k == "--gid") gidx = atoi(v.c_str());
    else if (k == "--dev") dev = v;
    else if (k == "--beside-read") read_gib = strtoull(v.c_str(), 0, 10);
    else if (k == "--read-hint") hint = v;
    else if (k == "--discard") discard = atoi(v.c_str()) != 0;
    else if (k == "--rewrite") rewrite = atoi(v.c_str()) != 0;
    else DIE("unknown %s", k.c_str());
  }
  if (hint != "normal" && hint != "stream") DIE("--read-hint normal|stream");
  if (inl && (op != "write" || src_s != "host")) DIE("--inline is a host-source write");
  size_t B = mib << 20, C = inl ? (size_t)inl : chunk;
  if (B % C) DIE("--mib must be a multiple of the chunk");
  CK(cudaSetDevice(gpu));
  CK(cudaFree(0));

  int n;
  ibv_device** list = ibv_get_device_list(&n);
  ibv_context* ctx = nullptr;
  for (int i = 0; i < n && !ctx; i++)
    if (dev == ibv_get_device_name(list[i])) ctx = ibv_open_device(list[i]);
  if (!ctx) DIE("no device %s", dev.c_str());
  ibv_pd* pd = ibv_alloc_pd(ctx);
  ibv_cq* cq = ibv_create_cq(ctx, 4096, nullptr, nullptr, 0);
  ibv_cq* rcq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);
  if (!pd || !cq || !rcq) DIE("pd/cq");
  ibv_port_attr pa;
  if (ibv_query_port(ctx, 1, &pa)) DIE("query_port");
  ibv_gid gid;
  if (ibv_query_gid(ctx, 1, gidx, &gid)) DIE("query_gid");
  ibv_qp_init_attr qa;
  memset(&qa, 0, sizeof qa);
  qa.qp_type = IBV_QPT_RC;
  qa.send_cq = cq;
  qa.recv_cq = rcq;
  qa.cap.max_send_wr = 1024;
  qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1;
  qa.cap.max_recv_sge = 1;
  qa.cap.max_inline_data = inl;
  ibv_qp* qa_ = ibv_create_qp(pd, &qa);
  qa.send_cq = rcq;
  qa.cap.max_inline_data = 0;
  ibv_qp* qb_ = ibv_create_qp(pd, &qa);
  if (!qa_ || !qb_) DIE("create_qp: %s", strerror(errno));
  to_rts(qa_, qb_->qp_num, gid, gidx, pa.active_mtu);
  to_rts(qb_, qa_->qp_num, gid, gidx, pa.active_mtu);

  // write: a's local = src, b's = dst.  read: a's local = dst, b's = src.
  Buf src = alloc(pd, src_s == "gpu", B), dst = alloc(pd, dst_s == "gpu", B);
  Buf& local = op == "write" ? src : dst;
  Buf& remote = op == "write" ? dst : src;
  int opcode = op == "write" ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
  int depth = op == "write" ? 512 : 16 * 8, sig = 16;
  if (op != "write" && op != "read") DIE("--op write|read");

  size_t posted = 0, done = 0;
  size_t chunks = B / C;
  auto post = [&](size_t i) {
    ibv_sge sge;
    sge.addr = (uint64_t)local.p + (i % chunks) * C;
    sge.length = C;
    sge.lkey = local.mr->lkey;
    ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = i;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = (ibv_wr_opcode)opcode;
    wr.send_flags = ((i + 1) % sig == 0 ? IBV_SEND_SIGNALED : 0) | (inl ? IBV_SEND_INLINE : 0);
    wr.wr.rdma.remote_addr = (uint64_t)remote.p + (i % chunks) * C;
    wr.wr.rdma.rkey = remote.mr->rkey;
    if (ibv_post_send(qa_, &wr, &bad)) DIE("post_send: %s", strerror(errno));
  };
  // run `total` operations (multiple of sig), keeping at most `depth` in flight
  auto run = [&](size_t total) {
    ibv_wc wc[64];
    size_t start = posted;
    while (done < start + total) {
      while (posted < start + total && posted - done < (size_t)depth) post(posted++);
      int k = ibv_poll_cq(cq, 64, wc);
      if (k < 0) DIE("poll_cq");
      for (int j = 0; j < k; j++) {
        if (wc[j].status != IBV_WC_SUCCESS) DIE("wc status %s", ibv_wc_status_str(wc[j].status));
        done += sig;
      }
    }
  };
  auto round_up = [&](size_t x) { return (x + sig - 1) / sig * sig; };

  fill(src, B, 0x2545f491u);
  zero(dst, B);
  run(round_up(chunks));
  std::vector<uint8_t> a = host_copy(src, B), b = host_copy(dst, B);
  bool ok = memcmp(a.data(), b.data(), B) == 0;
  printf("{\"gate\": \"rdma %s %s->%s\", \"ok\": %s, \"bytes\": %zu, \"src_reg\": \"%s\", \"dst_reg\": \"%s\"}\n",
         op.c_str(), src_s.c_str(), dst_s.c_str(), ok ? "true" : "false", B, src.reg, dst.reg);
  fflush(stdout);
  if (!ok) return 3;

  size_t RB = read_gib << 30;
  uint4* rb = nullptr;
  uint32_t* rsink = nullptr;
  cudaStream_t rs = nullptr;
  cudaEvent_t r0, r1;
  int passes = 0, sms = 0;
  if (RB) {
    CK(cudaMalloc(&rb, RB));
    CK(cudaMalloc(&rsink, 4));
    CK(cudaMemset(rb, 3, RB));
    CK(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));
    CK(cudaEventCreate(&r0));
    CK(cudaEventCreate(&r1));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, gpu));
    CK(cudaDeviceSynchronize());
    passes = std::max(1, (int)(seconds * 1.2e12 / RB));
    CK(cudaEventRecord(r0, rs));
    if (discard && !dst.gpu) DIE("--discard drops the GPU destination's lines: --dst gpu");
    if (rewrite && !src.gpu) DIE("--rewrite stores into the GPU source: --src gpu");
    char* ring = discard ? (char*)dst.p : rewrite ? (char*)src.p : nullptr;
    if (rewrite) k_read_passes<false, true><<<2 * sms, 512, 0, rs>>>(rb, RB / 16, passes, rsink, ring, B / 128);
    else if (hint == "stream") k_read_passes<true><<<2 * sms, 512, 0, rs>>>(rb, RB / 16, passes, rsink, ring, B / 128);
    else k_read_passes<false><<<2 * sms, 512, 0, rs>>>(rb, RB / 16, passes, rsink, ring, B / 128);
    CK(cudaEventRecord(r1, rs));
  }
  size_t per = round_up(std::max<size_t>(chunks, 4096));
  double t0 = now_s(), t;
  size_t ops = 0;
  do {
    run(per);
    ops += per;
    t = now_s();
  } while (RB ? cudaEventQuery(r1) == cudaErrorNotReady : t - t0 < seconds);
  printf("{\"path\": \"rdma_%s\", \"src\": \"%s\", \"dst\": \"%s\", \"mib\": %zu, \"chunk\": %zu, \"inline\": %d, "
         "\"mtu\": %d, \"seconds\": %.3f, \"gbps\": %.3f, \"mops\": %.3f",
         op.c_str(), src_s.c_str(), dst_s.c_str(), mib, C, inl, 128 << pa.active_mtu, t - t0,
         ops * (double)C / (t - t0) / 1e9, ops / (t - t0) / 1e6);
  if (RB) {
    float ms;
    CK(cudaEventSynchronize(r1));
    CK(cudaEventElapsedTime(&ms, r0, r1));
    printf(", \"beside_read_gib\": %zu, \"read_hint\": \"%s\", \"discard\": %s, \"rewrite\": %s, \"read_gbps\": %.2f",
           read_gib, hint.c_str(), discard ? "true" : "false", rewrite ? "true" : "false",
           (double)passes * RB / (ms * 1e-3) / 1e9);
  }
  printf("}\n");
  return 0;
}
