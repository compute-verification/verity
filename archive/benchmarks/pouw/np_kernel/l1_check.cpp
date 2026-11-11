// l1.h's seed path on the host, both ways (the byte-level reference and the kernels' template path), one JSON line: the
// values l1_ref.py prints from the tree's Python for the same inputs, and whether the two paths agree.
//   g++ -O2 -std=c++17 benchmarks/pouw/np_kernel/l1_check.cpp -o l1_check && ./l1_check SEED ROWS K CALL
#include <cstdio>
#include <cstdlib>

#include "l1.h"

static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
static std::string hex(const l1::Bytes& b) {
  std::string s;
  char t[3];
  for (uint8_t v : b) {
    snprintf(t, sizeof t, "%02x", v);
    s += t;
  }
  return s;
}
static std::string hexw(const uint64_t* w) { return hex(l1::digest_bytes(w)); }

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: l1_check SEED ROWS K CALL\n");
    return 2;
  }
  const uint64_t seed = strtoull(argv[1], nullptr, 10), rows = strtoull(argv[2], nullptr, 10), K = strtoull(argv[3], nullptr, 10),
                 call = strtoull(argv[4], nullptr, 10);
  const l1::Window w = l1::window_of(seed, 3, 64);
  std::vector<uint8_t> a(rows * K);
  for (size_t i = 0; i < a.size(); i++) a[i] = (uint8_t)mix64(seed * 1000003 + i);
  const l1::Ref r = l1::reference(w, call, a.data(), a.size());
  const uint64_t n = a.size() / l1::SEG;
  const l1::Call k = l1::call_of(w, call, n);
  std::vector<std::vector<uint64_t>> lv(n, std::vector<uint64_t>(8));
  int bad_leaves = 0;
  for (uint64_t i = 0; i < n; i++) {
    l1::a_leaf(k, i, a.data() + l1::SEG * i, lv[i].data());
    bad_leaves += l1::digest_bytes(lv[i].data()) != r.leaves[i];
  }
  for (int level = 0; lv.size() > 1; level++) {
    std::vector<std::vector<uint64_t>> up(lv.size() / 2, std::vector<uint64_t>(8));
    for (size_t i = 0; i < up.size(); i++) l1::node_of(k, level, i, lv[2 * i].data(), lv[2 * i + 1].data(), up[i].data());
    lv = up;
  }
  uint64_t h[8], ra[8], sd[8], nz[8];
  l1::instance(k, lv[0].data(), h, ra);
  l1::seed_of(k, ra, sd);
  l1::noise_of(k, sd, 0, nz);
  const bool same = !bad_leaves && l1::digest_bytes(lv[0].data()) == r.x && l1::digest_bytes(h) == r.h &&
                    l1::digest_bytes(ra) == r.root_a && l1::digest_bytes(sd) == r.seed &&
                    l1::digest_bytes(nz) == l1::noise_block(r.seed, 0);
  printf("{\"leaves\": %llu, \"dom\": \"%s\", \"leaf0\": \"%s\", \"leaf_last\": \"%s\", \"x\": \"%s\", \"y\": \"%s\", "
         "\"bc\": \"%s\", \"h\": \"%s\", \"root_a\": \"%s\", \"seed\": \"%s\", \"noise0\": \"%s\", \"noise1\": \"%s\", "
         "\"paths_agree\": %s, \"leaf_mismatches\": %d}\n",
         (unsigned long long)n, hex(r.dom).c_str(), hex(r.leaves[0]).c_str(), hex(r.leaves.back()).c_str(), hex(r.x).c_str(),
         hex(r.y).c_str(), hex(r.bc).c_str(), hex(r.h).c_str(), hex(r.root_a).c_str(), hex(r.seed).c_str(),
         hexw(nz).c_str(), hex(l1::noise_block(r.seed, 1)).c_str(), same ? "true" : "false", bad_leaves);
  return same ? 0 : 1;
}
