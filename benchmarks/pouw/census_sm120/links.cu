// Link census (item 6): PCIe host<->device for each visible GPU (pinned and pageable, each direction and both at once),
// both GPUs' host transfers at the same time (do two GPUs share a root port or switch?), and GPU<->GPU copies (P2P when
// cudaDeviceCanAccessPeer allows it; otherwise cudaMemcpyPeer, which the driver stages through the host). Run under
// `gpu-lease 2`, so CUDA devices 0 and 1 are the leased pair. One JSON line a constant: constant, value, unit, gpu_uuid,
// method.
#include "common.cuh"

#include <chrono>
#include <cstring>
#include <string>

static constexpr size_t BYTES = 256ull << 20;
static constexpr int REPS = 5;

static std::string uuid_of(int d) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, d));
  char s[64];
  const unsigned char* u = (const unsigned char*)p.uuid.bytes;
  snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3],
           u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
  return s;
}

static void emit(const std::string& constant, double value, const char* unit, const std::string& uuid,
                 const std::string& method) {
  printf("{\"constant\":\"%s\",\"value\":%.4g,\"unit\":\"%s\",\"gpu_uuid\":\"%s\",\"method\":\"%s\"}\n", constant.c_str(),
         value, unit, uuid.c_str(), method.c_str());
  fflush(stdout);
}

struct Dev {
  int id;
  std::string uuid;
  void *d_a, *d_b, *h_pin_a, *h_pin_b, *h_page;
  cudaStream_t s1, s2;
};

// Best of REPS of `issue` (which enqueues copies on the given streams), timed from host-side events on device d.
template <class F>
static double best_seconds(const std::vector<Dev*>& devs, F issue) {
  double best = 1e30;
  for (int r = 0; r <= REPS; r++) {
    for (auto* d : devs) {
      CK(cudaSetDevice(d->id));
      CK(cudaDeviceSynchronize());
    }
    auto t0 = std::chrono::steady_clock::now();
    issue();
    for (auto* d : devs) {
      CK(cudaSetDevice(d->id));
      CK(cudaDeviceSynchronize());
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (r > 0 && s < best) best = s;  // r == 0 warms up
  }
  return best;
}

static void per_gpu(Dev& d) {
  std::vector<Dev*> v{&d};
  CK(cudaSetDevice(d.id));
  auto h2d = best_seconds(v, [&] { CK(cudaMemcpyAsync(d.d_a, d.h_pin_a, BYTES, cudaMemcpyHostToDevice, d.s1)); });
  emit("pcie.h2d_pinned.bytes_per_s", BYTES / h2d, "B/s", d.uuid, "cudaMemcpyAsync 256 MiB pinned host -> device, best of 5");
  auto d2h = best_seconds(v, [&] { CK(cudaMemcpyAsync(d.h_pin_a, d.d_a, BYTES, cudaMemcpyDeviceToHost, d.s1)); });
  emit("pcie.d2h_pinned.bytes_per_s", BYTES / d2h, "B/s", d.uuid, "cudaMemcpyAsync 256 MiB device -> pinned host, best of 5");
  auto bi = best_seconds(v, [&] {
    CK(cudaMemcpyAsync(d.d_a, d.h_pin_a, BYTES, cudaMemcpyHostToDevice, d.s1));
    CK(cudaMemcpyAsync(d.h_pin_b, d.d_b, BYTES, cudaMemcpyDeviceToHost, d.s2));
  });
  emit("pcie.bidir_pinned.bytes_per_s", 2.0 * BYTES / bi, "B/s", d.uuid,
       "256 MiB H2D and 256 MiB D2H at once on two streams (two copy engines), pinned, total bytes / wall time");
  auto h2dp = best_seconds(v, [&] { CK(cudaMemcpy(d.d_a, d.h_page, BYTES, cudaMemcpyHostToDevice)); });
  emit("pcie.h2d_pageable.bytes_per_s", BYTES / h2dp, "B/s", d.uuid, "cudaMemcpy 256 MiB pageable (malloc) host -> device");
  auto d2hp = best_seconds(v, [&] { CK(cudaMemcpy(d.h_page, d.d_a, BYTES, cudaMemcpyDeviceToHost)); });
  emit("pcie.d2h_pageable.bytes_per_s", BYTES / d2hp, "B/s", d.uuid, "cudaMemcpy 256 MiB device -> pageable (malloc) host");
}

int main() {
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  std::vector<Dev> devs(n);
  for (int i = 0; i < n; i++) {
    Dev& d = devs[i];
    d.id = i;
    d.uuid = uuid_of(i);
    CK(cudaSetDevice(i));
    CK(cudaMalloc(&d.d_a, BYTES));
    CK(cudaMalloc(&d.d_b, BYTES));
    CK(cudaMemset(d.d_a, 0x5a, BYTES));
    CK(cudaMemset(d.d_b, 0xa5, BYTES));
    CK(cudaMallocHost(&d.h_pin_a, BYTES));
    CK(cudaMallocHost(&d.h_pin_b, BYTES));
    d.h_page = malloc(BYTES);
    memset(d.h_pin_a, 1, BYTES);
    memset(d.h_pin_b, 2, BYTES);
    memset(d.h_page, 3, BYTES);
    CK(cudaStreamCreateWithFlags(&d.s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&d.s2, cudaStreamNonBlocking));
    char bus[32];
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, i));
    printf("{\"probe\":\"device\",\"index\":%d,\"gpu_uuid\":\"%s\",\"pci_bus_id\":\"%s\"}\n", i, d.uuid.c_str(), bus);
  }
  for (auto& d : devs) per_gpu(d);
  if (n < 2) {
    printf("{\"probe\":\"links\",\"done\":true}\n");
    return 0;
  }
  Dev &a = devs[0], &b = devs[1];
  std::vector<Dev*> both{&a, &b};
  std::string pair = a.uuid + "+" + b.uuid;
  auto h2d2 = best_seconds(both, [&] {
    CK(cudaSetDevice(a.id));
    CK(cudaMemcpyAsync(a.d_a, a.h_pin_a, BYTES, cudaMemcpyHostToDevice, a.s1));
    CK(cudaSetDevice(b.id));
    CK(cudaMemcpyAsync(b.d_a, b.h_pin_a, BYTES, cudaMemcpyHostToDevice, b.s1));
  });
  emit("pcie.two_gpus.h2d_pinned.total_bytes_per_s", 2.0 * BYTES / h2d2, "B/s", pair,
       "both leased GPUs copy 256 MiB pinned host -> device at once; total bytes / wall time");
  auto d2h2 = best_seconds(both, [&] {
    CK(cudaSetDevice(a.id));
    CK(cudaMemcpyAsync(a.h_pin_a, a.d_a, BYTES, cudaMemcpyDeviceToHost, a.s1));
    CK(cudaSetDevice(b.id));
    CK(cudaMemcpyAsync(b.h_pin_a, b.d_a, BYTES, cudaMemcpyDeviceToHost, b.s1));
  });
  emit("pcie.two_gpus.d2h_pinned.total_bytes_per_s", 2.0 * BYTES / d2h2, "B/s", pair,
       "both leased GPUs copy 256 MiB device -> pinned host at once; total bytes / wall time");
  auto bi2 = best_seconds(both, [&] {
    for (Dev* d : both) {
      CK(cudaSetDevice(d->id));
      CK(cudaMemcpyAsync(d->d_a, d->h_pin_a, BYTES, cudaMemcpyHostToDevice, d->s1));
      CK(cudaMemcpyAsync(d->h_pin_b, d->d_b, BYTES, cudaMemcpyDeviceToHost, d->s2));
    }
  });
  emit("pcie.two_gpus.bidir_pinned.total_bytes_per_s", 4.0 * BYTES / bi2, "B/s", pair,
       "both leased GPUs copy 256 MiB each way at once (4 copies); total bytes / wall time");

  int ab = 0, ba = 0;
  CK(cudaDeviceCanAccessPeer(&ab, a.id, b.id));
  CK(cudaDeviceCanAccessPeer(&ba, b.id, a.id));
  emit("p2p.can_access_peer", ab && ba ? 1 : 0, "bool", pair, "cudaDeviceCanAccessPeer both ways");
  bool peer = false;
  if (ab && ba) {
    CK(cudaSetDevice(a.id));
    peer = cudaDeviceEnablePeerAccess(b.id, 0) == cudaSuccess;
    CK(cudaSetDevice(b.id));
    peer = (cudaDeviceEnablePeerAccess(a.id, 0) == cudaSuccess) && peer;
    cudaGetLastError();
  }
  std::string how = peer ? "peer access enabled (direct P2P)" : "no peer access: the driver stages through host memory";
  auto p01 = best_seconds(both, [&] {
    CK(cudaSetDevice(a.id));
    CK(cudaMemcpyPeerAsync(b.d_a, b.id, a.d_a, a.id, BYTES, a.s1));
  });
  emit("p2p.copy_0_to_1.bytes_per_s", BYTES / p01, "B/s", pair, "cudaMemcpyPeerAsync 256 MiB GPU0 -> GPU1; " + how);
  auto p10 = best_seconds(both, [&] {
    CK(cudaSetDevice(b.id));
    CK(cudaMemcpyPeerAsync(a.d_b, a.id, b.d_b, b.id, BYTES, b.s1));
  });
  emit("p2p.copy_1_to_0.bytes_per_s", BYTES / p10, "B/s", pair, "cudaMemcpyPeerAsync 256 MiB GPU1 -> GPU0; " + how);
  auto pbi = best_seconds(both, [&] {
    CK(cudaSetDevice(a.id));
    CK(cudaMemcpyPeerAsync(b.d_a, b.id, a.d_a, a.id, BYTES, a.s1));
    CK(cudaSetDevice(b.id));
    CK(cudaMemcpyPeerAsync(a.d_b, a.id, b.d_b, b.id, BYTES, b.s1));
  });
  emit("p2p.bidir.total_bytes_per_s", 2.0 * BYTES / pbi, "B/s", pair, "256 MiB each way between the pair at once; " + how);
  printf("{\"probe\":\"links\",\"done\":true}\n");
  return 0;
}
