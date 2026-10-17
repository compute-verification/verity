// Host-side parsing and printing for mont.cu and host_check.cpp: rw.py's params.txt (N, n', K, t in hex, one a line)
// and records.txt ("c_hex tag m_hex"), words as NL little-endian limbs.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "mont_core.h"

namespace mbio {

using mb::NL;
using mb::u32;

struct Params {
  u32 N[NL], np, K[NL], T[NL];
};

inline void from_hex(const std::string &s, u32 *x) {
  for (int k = 0; k < NL; ++k) x[k] = 0;
  int bit = 0;
  for (int i = (int)s.size() - 1; i >= 0; --i, bit += 4) {
    const char ch = s[i];
    const u32 v = ch <= '9' ? ch - '0' : (ch | 32) - 'a' + 10;
    if (bit >= 32 * NL) {
      if (v) { fprintf(stderr, "value wider than %d bits\n", 32 * NL); exit(2); }
      continue;
    }
    x[bit / 32] |= v << (bit % 32);
  }
}

inline void print_hex(const u32 *x) {
  int k = NL - 1;
  while (k > 0 && !x[k]) --k;
  printf("%x", x[k]);
  for (--k; k >= 0; --k) printf("%08x", x[k]);
}

inline Params read_params(const char *path) {
  std::ifstream f(path);
  std::string a, b, c, d;
  if (!(f >> a >> b >> c >> d)) { fprintf(stderr, "bad params %s\n", path); exit(2); }
  Params p;
  from_hex(a, p.N);
  p.np = (u32)strtoul(b.c_str(), nullptr, 16);
  from_hex(c, p.K);
  from_hex(d, p.T);
  return p;
}

struct Records {
  std::vector<u32> c;  // limb-major: c[k * n + j]
  std::vector<uint8_t> tag;
  int n = 0;
};

inline Records read_records(const char *path) {
  std::ifstream f(path);
  std::vector<std::vector<u32>> cs;
  Records r;
  std::string c, m;
  int tag;
  while (f >> c >> tag >> m) {
    cs.emplace_back(NL);
    from_hex(c, cs.back().data());
    r.tag.push_back((uint8_t)tag);
  }
  r.n = (int)cs.size();
  r.c.assign((size_t)NL * r.n, 0);
  for (int j = 0; j < r.n; ++j)
    for (int k = 0; k < NL; ++k) r.c[(size_t)k * r.n + j] = cs[j][k];
  return r;
}

inline uint64_t splitmix(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// n words below N, limb-major: random limbs under a top limb below N's.
inline void fill_below(const Params &p, std::vector<u32> &h, int n, uint64_t seed) {
  h.assign((size_t)NL * n, 0);
  for (int w = 0; w < n; ++w) {
    for (int k = 0; k < NL - 1; ++k) h[(size_t)k * n + w] = (u32)splitmix(seed);
    h[(size_t)(NL - 1) * n + w] = (u32)(splitmix(seed) % p.N[NL - 1]);
  }
}

}  // namespace mbio
