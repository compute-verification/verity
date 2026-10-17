// mont_core.h's arithmetic on the host (carry flags emulated), with mont.cu's gate commands, so rw.py can check the
// forms without a GPU:  g++ -O1 -std=c++17 -DMB_BITS=1024 -o hc host_check.cpp
//   hc chain FORM K params.txt N        N words below N, K squarings: "x0 xK" in hex
//   hc decode FORM params.txt records.txt   each record's decoded payload in hex
#include <cstring>

#include "mont_io.h"

using namespace mb;

template <int FORM>
static void chain(u32 (&x)[NL], int K) {
  for (int r = 0; r < K; ++r) {
    u32 y[NL];
    mont_sq<FORM>(x, y);
    for (int k = 0; k < NL; ++k) x[k] = y[k];
  }
}

template <int FORM>
static void run(int argc, char **argv) {
  if (!strcmp(argv[1], "chain")) {
    const int K = atoi(argv[3]), n = atoi(argv[5]);
    std::vector<u32> h;
    mbio::Params p = mbio::read_params(argv[4]);
    mbio::fill_below(p, h, n, 0x5eedull);
    for (int w = 0; w < n; ++w) {
      u32 x[NL], x0[NL];
      for (int k = 0; k < NL; ++k) x0[k] = x[k] = h[(size_t)k * n + w];
      chain<FORM>(x, K);
      mbio::print_hex(x0);
      printf(" ");
      mbio::print_hex(x);
      printf("\n");
    }
  } else {
    mbio::Records r = mbio::read_records(argv[4]);
    for (int j = 0; j < r.n; ++j) {
      u32 c[NL], m[NL];
      for (int k = 0; k < NL; ++k) c[k] = r.c[(size_t)k * r.n + j];
      decode<FORM>(c, r.tag[j], (u32)j, m);
      mbio::print_hex(m);
      printf("\n");
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 5) return 2;
  const char *pp = !strcmp(argv[1], "chain") ? argv[4] : argv[3];
  mbio::Params p = mbio::read_params(pp);
  memcpy(cN, p.N, sizeof cN);
  cNp = p.np;
  memcpy(cK, p.K, sizeof cK);
  memcpy(cT, p.T, sizeof cT);
  switch (atoi(argv[2])) {
    case 0: run<0>(argc, argv); break;
    case 1: run<1>(argc, argv); break;
    case 2: run<2>(argc, argv); break;
    case 3: run<3>(argc, argv); break;
    default: return 2;
  }
  return 0;
}
