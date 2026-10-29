// The H200's JPEG decode engines (NVJPG) as a word source: nvJPEG's hardware backend decodes a batch of baseline 4:2:0
// JPEGs into interleaved 8-bit RGB, and that output rate bounds what the engines could add to the card's words. Two
// images, encoded on the card at quality 90: a smooth gradient and uniform noise (the Huffman-heaviest). Image 0 and
// image B-1 of the batch are checked against the CUDA backend's decode of the same bitstream.
// Usage: nvjpg [W] [B] [ITERS]; prints one JSON object an image.
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));     \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)
#define NJ(x)                                                                         \
  do {                                                                                \
    nvjpegStatus_t s_ = (x);                                                          \
    if (s_ != NVJPEG_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "%s:%d nvjpeg status %d\n", __FILE__, __LINE__, (int)s_);       \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

__global__ void fill(unsigned char* p, int w, int h, int noise) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)w * h * 3) return;
  const int c = (int)(i % 3), x = (int)(i / 3 % w), y = (int)(i / 3 / w);
  unsigned v;
  if (noise) {
    unsigned z = (unsigned)i * 2654435761u + 12345u;
    z ^= z >> 15, z *= 2246822519u, z ^= z >> 13, z *= 3266489917u, z ^= z >> 16;
    v = z & 255u;
  } else {
    v = c == 0 ? x * 255 / (w - 1) : c == 1 ? y * 255 / (h - 1) : (x + y) * 255 / (w + h - 2);
  }
  p[i] = (unsigned char)v;
}

static std::vector<unsigned char> encode(int w, int h, int noise, unsigned char* d_src, cudaStream_t st) {
  nvjpegHandle_t eh;
  NJ(nvjpegCreateSimple(&eh));
  nvjpegEncoderState_t es;
  nvjpegEncoderParams_t ep;
  NJ(nvjpegEncoderStateCreate(eh, &es, st));
  NJ(nvjpegEncoderParamsCreate(eh, &ep, st));
  NJ(nvjpegEncoderParamsSetQuality(ep, 90, st));
  NJ(nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_420, st));
  NJ(nvjpegEncoderParamsSetOptimizedHuffman(ep, 0, st));
  fill<<<(unsigned)(((size_t)w * h * 3 + 255) / 256), 256, 0, st>>>(d_src, w, h, noise);
  CK(cudaGetLastError());
  nvjpegImage_t src;
  memset(&src, 0, sizeof src);
  src.channel[0] = d_src;
  src.pitch[0] = (size_t)w * 3;
  NJ(nvjpegEncodeImage(eh, es, ep, &src, NVJPEG_INPUT_RGBI, w, h, st));
  size_t len = 0;
  NJ(nvjpegEncodeRetrieveBitstream(eh, es, nullptr, &len, st));
  std::vector<unsigned char> jpg(len);
  NJ(nvjpegEncodeRetrieveBitstream(eh, es, jpg.data(), &len, st));
  CK(cudaStreamSynchronize(st));
  jpg.resize(len);
  nvjpegEncoderParamsDestroy(ep);
  nvjpegEncoderStateDestroy(es);
  nvjpegDestroy(eh);
  return jpg;
}

int main(int argc, char** argv) {
  const int W = argc > 1 ? atoi(argv[1]) : 2048, B = argc > 2 ? atoi(argv[2]) : 32, ITERS = argc > 3 ? atoi(argv[3]) : 20;
  const size_t img_bytes = (size_t)W * W * 3;
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  unsigned char *d_src, *d_out, *d_ref;
  CK(cudaMalloc(&d_src, img_bytes));
  CK(cudaMalloc(&d_out, img_bytes * B));
  CK(cudaMalloc(&d_ref, img_bytes));

  nvjpegHandle_t hw, sw;
  const nvjpegStatus_t hs = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT, &hw);
  if (hs != NVJPEG_STATUS_SUCCESS) {
    printf("{\"hardware_backend\": false, \"status\": %d}\n", (int)hs);
    return 3;
  }
  NJ(nvjpegCreateSimple(&sw));
#ifdef HWINFO
  unsigned engines = 0, cores = 0;
  NJ(nvjpegGetHardwareDecoderInfo(hw, &engines, &cores));
  printf("{\"hardware_backend\": true, \"engines\": %u, \"cores_per_engine\": %u}\n", engines, cores);
#else
  printf("{\"hardware_backend\": true}\n");
#endif

  std::vector<unsigned char> h_src(img_bytes), h_out(img_bytes), h_ref(img_bytes);
  for (int noise = 0; noise < 2; noise++) {
    const std::vector<unsigned char> jpg = encode(W, W, noise, d_src, st);
    CK(cudaMemcpy(h_src.data(), d_src, img_bytes, cudaMemcpyDeviceToHost));

    nvjpegJpegStream_t js;
    NJ(nvjpegJpegStreamCreate(hw, &js));
    NJ(nvjpegJpegStreamParse(hw, jpg.data(), jpg.size(), 0, 0, js));
    int unsupported = 1;
    NJ(nvjpegDecodeBatchedSupported(hw, js, &unsupported));
    nvjpegJpegStreamDestroy(js);
    if (unsupported) {
      printf("{\"image\": \"%s\", \"hardware_supported\": false}\n", noise ? "noise" : "smooth");
      continue;
    }

    nvjpegJpegState_t hst;
    NJ(nvjpegJpegStateCreate(hw, &hst));
    NJ(nvjpegDecodeBatchedInitialize(hw, hst, B, 1, NVJPEG_OUTPUT_RGBI));
    std::vector<const unsigned char*> data(B, jpg.data());
    std::vector<size_t> lens(B, jpg.size());
    std::vector<nvjpegImage_t> img(B);
    for (int i = 0; i < B; i++) {
      memset(&img[i], 0, sizeof img[i]);
      img[i].channel[0] = d_out + img_bytes * i;
      img[i].pitch[0] = (size_t)W * 3;
    }
    CK(cudaMemset(d_out, 0xA5, img_bytes * B));
    NJ(nvjpegDecodeBatched(hw, hst, data.data(), lens.data(), img.data(), st));
    CK(cudaStreamSynchronize(st));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, st));
    for (int it = 0; it < ITERS; it++) NJ(nvjpegDecodeBatched(hw, hst, data.data(), lens.data(), img.data(), st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));

    nvjpegJpegState_t sst;
    NJ(nvjpegJpegStateCreate(sw, &sst));
    nvjpegImage_t ref;
    memset(&ref, 0, sizeof ref);
    ref.channel[0] = d_ref;
    ref.pitch[0] = (size_t)W * 3;
    NJ(nvjpegDecode(sw, sst, jpg.data(), jpg.size(), NVJPEG_OUTPUT_RGBI, &ref, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpy(h_ref.data(), d_ref, img_bytes, cudaMemcpyDeviceToHost));

    size_t diff[2] = {0, 0};
    int maxd[2] = {0, 0};
    double mae_src = 0;
    for (int k = 0; k < 2; k++) {
      CK(cudaMemcpy(h_out.data(), d_out + img_bytes * (k ? B - 1 : 0), img_bytes, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < img_bytes; i++) {
        const int d = abs((int)h_out[i] - (int)h_ref[i]);
        diff[k] += d != 0;
        maxd[k] = d > maxd[k] ? d : maxd[k];
        if (k == 0) mae_src += abs((int)h_out[i] - (int)h_src[i]);
      }
    }
    const double pix = (double)W * W * B * ITERS, s = ms / 1e3;
    printf("{\"image\": \"%s\", \"w\": %d, \"h\": %d, \"jpeg_bytes\": %zu, \"batch\": %d, \"iters\": %d, \"ms\": %.3f, "
           "\"pixels_per_s\": %.6e, \"out_bytes_per_s\": %.6e, \"jpeg_bytes_per_s\": %.6e, "
           "\"hw_vs_cuda_bytes_differ\": [%zu, %zu], \"hw_vs_cuda_max\": [%d, %d], \"hw_vs_source_mae\": %.4f}\n",
           noise ? "noise" : "smooth", W, W, jpg.size(), B, ITERS, ms, pix / s, pix * 3 / s,
           (double)jpg.size() * B * ITERS / s, diff[0], diff[1], maxd[0], maxd[1], mae_src / img_bytes);
    fflush(stdout);
    nvjpegJpegStateDestroy(sst);
    nvjpegJpegStateDestroy(hst);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }
  nvjpegDestroy(sw);
  nvjpegDestroy(hw);
  return 0;
}
