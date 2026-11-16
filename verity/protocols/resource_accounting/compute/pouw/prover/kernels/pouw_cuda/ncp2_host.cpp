// The host build of `ncp2.cuh` (`pouw_device.host`): the same unit functions the device kernels call, run on the CPU so that
// the gate against `pouw_native` needs no GPU. It needs only the C library.
#include <stdlib.h>

#include "ncp2_fast.cuh"

using namespace ncp2;

template <class T>
struct Buf {
  T* p;
  size_t n;
  explicit Buf(size_t count) : p((T*)calloc(count ? count : 1, sizeof(T))), n(count) {}
  ~Buf() { free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  T& operator[](size_t i) { return p[i]; }
  T* data() { return p; }
  size_t size() const { return n; }
};

// Hash ``n`` messages (message i is data[offsets[i] .. offsets[i + 1]]), ``outlen`` bytes each (32 or 64 for the SHA-2s).
// kind: 0 SHA-256, 1 SHA-512, 2 SHAKE256, 3 TurboSHAKE128 with domain byte ``ds``.
extern "C" int ncp2_hash(int kind, int ds, const uint8_t* data, const uint64_t* offsets, uint64_t n, uint64_t outlen, uint8_t* out) {
  for (uint64_t i = 0; i < n; ++i) {
    int rc = hash_one(kind, ds, data + offsets[i], (size_t)(offsets[i + 1] - offsets[i]), (size_t)outlen, out + i * outlen);
    if (rc) return rc;
  }
  return 0;
}

// One call's every unit leaf, in the partition's numbering (`pouw_native.Native.base`), its weight side's Y unit leaves
// (``yleaves``, one per column strip) and its served y. The weight's rows are passed whole. Returns 0, or 1 for a shape
// outside 16-row and 16-column strips.
extern "C" int ncp2_commit(int m, int k, int n, const int8_t* a, const uint32_t* s, const int8_t* b, const uint32_t* t,
                           const uint16_t* bias, const uint8_t* salt, uint64_t forward, uint64_t position, uint64_t weight,
                           const uint32_t* rowids, const uint32_t* colids, const uint32_t* pi, const uint32_t* inv,
                           uint8_t* leaves, uint8_t* yleaves, uint16_t* y_out) {
  if (m <= 0 || n <= 0 || k <= 0 || m % D || n % D || (3 * k) % D || m / D > (1 << 19)) return 1;
  const int ms = m / D, ns = n / D, steps = 3 * k / D, k3 = 3 * k;
  const size_t digest = 0, node = ms, key = 2 * ms - 1, form_x = 2 * ms, tile = 3 * ms, deq = 3 * ms + (size_t)ms * ns;
  auto leaf = [&](size_t u) { return leaves + 32 * u; };

  Buf<uint8_t> rows((size_t)m * 64), strips((size_t)ms * 64), nodes((size_t)(ms > 1 ? ms - 1 : 1) * 64);
  for (int i = 0; i < m; ++i) row_digest(a + (size_t)i * k, k, &rows[(size_t)i * 64]);
  for (int st = 0; st < ms; ++st) {
    strip_digest(&rows[(size_t)st * D * 64], &strips[(size_t)st * 64]);
    unit_leaf_digest(&strips[(size_t)st * 64], leaf(digest + st));
  }
  int count = 0;
  const uint8_t* da = rfc6962(strips.data(), 0, ms, nodes.data(), &count);
  for (int j = 0; j < count; ++j) unit_leaf_digest(&nodes[(size_t)j * 64], leaf(node + j));
  uint8_t kk[32];
  call_key(salt, forward, position, weight, da, kk);
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i) kw[i] = kk[4 * i] | (kk[4 * i + 1] << 8) | (kk[4 * i + 2] << 16) | ((uint32_t)kk[4 * i + 3] << 24);
  unit_leaf(kw, 8, 1, leaf(key));

  Buf<int8_t> noise((size_t)k);
  Buf<uint8_t> x((size_t)m * k3), y((size_t)n * k3);
  Buf<uint32_t> c0((size_t)n);
  for (int i = 0; i < m; ++i) {
    e1_row(kk, rowids[i], k, noise.data());
    form_x_row(a + (size_t)i * k, noise.data(), pi, k, &x[(size_t)i * k3]);
  }
  for (int j = 0; j < n; ++j) {
    f1_col(salt, weight, colids[j], k, noise.data());
    c0[j] = form_y_col(b + (size_t)j * k, noise.data(), inv, k, &y[(size_t)j * k3]);
  }
  for (int rs = 0; rs < ms; ++rs) {
    Sha256 h;
    unit_leaf_begin(h);
    unit_leaf_bytes(h, &x[(size_t)rs * D * k3], (size_t)D * k3);
    sha256_final(h, leaf(form_x + rs));
  }
  for (int cs = 0; cs < ns; ++cs) {
    Sha256 h;
    unit_leaf_begin(h);
    unit_leaf_bytes(h, &y[(size_t)cs * D * k3], (size_t)D * k3);
    for (int c = 0; c < D; ++c) sha256_u32le(h, c0[cs * D + c]);
    sha256_final(h, yleaves + 32 * (size_t)cs);
  }

  Buf<uint32_t> ck((size_t)steps * D * D);
  for (int rs = 0; rs < ms; ++rs)
    for (int cs = 0; cs < ns; ++cs) {
      for (int r = 0; r < D; ++r)
        for (int c = 0; c < D; ++c)
          tile_entry(&x[(size_t)(rs * D + r) * k3], &y[(size_t)(cs * D + c) * k3], c0[cs * D + c], steps, &ck[r * D + c], D * D);
      size_t u = (size_t)rs * ns + cs;
      unit_leaf(ck.data(), ck.size(), 1, leaf(tile + u));
      for (int r = 0; r < D; ++r)
        for (int c = 0; c < D; ++c) {
          int row = rs * D + r, col = cs * D + c;
          uint16_t v = dequant(ck[(size_t)(steps - 1) * D * D + r * D + c], s[row], t[col], bias[col]);
          y_out[(size_t)row * n + col] = v;
          uint32_t w = v;
          unit_leaf(&w, 1, 1, leaf(deq + u * D * D + r * D + c));
        }
    }
  return 0;
}

struct ArrRun {
  uint32_t* a;
  uint32_t& operator()(int j) const { return a[j]; }
};

// `ncp2_commit` computed the serving kernel's way (`ncp2_fast.cuh`): the same leaves and y.
extern "C" int ncp2_commit_fast(int m, int k, int n, const int8_t* a, const uint32_t* s, const int8_t* b, const uint32_t* t,
                                const uint16_t* bias, const uint8_t* salt, uint64_t forward, uint64_t position, uint64_t weight,
                                const uint32_t* rowids, const uint32_t* colids, const uint32_t* pi, const uint32_t* inv,
                                uint8_t* leaves, uint8_t* yleaves, uint16_t* y_out) {
  if (m <= 0 || n <= 0 || k <= 0 || m % D || n % D || (3 * k) % D || m / D > (1 << 19)) return 1;
  const int ms = m / D, ns = n / D, steps = 3 * k / D, k3 = 3 * k;
  const size_t node = ms, key = 2 * ms - 1, form_x = 2 * ms, tile = 3 * ms, deq = 3 * ms + (size_t)ms * ns;
  auto leaf = [&](size_t u) { return leaves + 32 * u; };
  Buf<uint8_t> rows((size_t)m * 64), strips((size_t)ms * 64), nodes((size_t)(ms > 1 ? ms - 1 : 1) * 64);
  for (int i = 0; i < m; ++i) row_digest_fast(a + (size_t)i * k, k, &rows[(size_t)i * 64]);
  for (int st = 0; st < ms; ++st) {
    strip_digest_fast(&rows[(size_t)st * D * 64], &strips[(size_t)st * 64]);
    leaf_digest_fast(&strips[(size_t)st * 64], leaf(st));
  }
  int count = 0;
  const uint8_t* da = rfc6962_fast(strips.data(), 0, ms, nodes.data(), &count);
  for (int j = 0; j < count; ++j) leaf_digest_fast(&nodes[(size_t)j * 64], leaf(node + j));
  uint8_t kk[32];
  call_key_fast(salt, forward, position, weight, da, kk);
  uint32_t kw[8];
  for (int i = 0; i < 8; ++i) kw[i] = kk[4 * i] | (kk[4 * i + 1] << 8) | (kk[4 * i + 2] << 16) | ((uint32_t)kk[4 * i + 3] << 24);
  unit_leaf(kw, 8, 1, leaf(key));

  Buf<int8_t> noise_((size_t)k);
  Buf<uint8_t> x((size_t)m * k3), y((size_t)n * k3);
  Buf<uint32_t> c0((size_t)n);
  for (int i = 0; i < m; ++i) {
    e1_row_fast(kk, rowids[i], k, noise_.data());
    form_x_row(a + (size_t)i * k, noise_.data(), pi, k, &x[(size_t)i * k3]);
  }
  for (int j = 0; j < n; ++j) {
    f1_col_fast(salt, weight, colids[j], k, noise_.data());
    c0[j] = form_y_col(b + (size_t)j * k, noise_.data(), inv, k, &y[(size_t)j * k3]);
  }
  for (int rs = 0; rs < ms; ++rs) {
    WordLeaf wl;
    wl_init(wl);
    wl_bytes(wl, &x[(size_t)rs * D * k3], (size_t)D * k3);
    wl_final(wl, leaf(form_x + rs));
  }
  for (int cs = 0; cs < ns; ++cs) {
    WordLeaf wl;
    wl_init(wl);
    wl_bytes(wl, &y[(size_t)cs * D * k3], (size_t)D * k3);
    wl_block(wl, &c0[(size_t)cs * D]);
    wl_final(wl, yleaves + 32 * (size_t)cs);
  }
  uint32_t run[D * D];
  for (int rs = 0; rs < ms; ++rs)
    for (int cs = 0; cs < ns; ++cs) {
      for (int j = 0; j < D * D; ++j) run[j] = c0[cs * D + (j % D)];
      size_t u = (size_t)rs * ns + cs;
      tile_fused<true>(&x[(size_t)rs * D * k3], &y[(size_t)cs * D * k3], k3, steps, ArrRun{run}, leaf(tile + u));
      for (int j = 0; j < D * D; ++j) {
        int row = rs * D + j / D, col = cs * D + j % D;
        uint16_t v = dequant(run[j], s[row], t[col], bias[col]);
        y_out[(size_t)row * n + col] = v;
        leaf_word1(v, leaf(deq + u * D * D + j));
      }
    }
  return 0;
}

// `pouw_circuit.units_root` with the fast node: ``count`` leaves of 32 bytes.
extern "C" int ncp2_units_root_fast(const uint8_t* leaves, uint64_t count, uint8_t* root) {
  Buf<uint32_t> level((size_t)count * 8);
  for (uint64_t i = 0; i < count * 8; ++i) level[i] = be32(leaves + 4 * i);
  for (uint64_t c = count; c > 1; c = (c + 1) / 2) {
    for (uint64_t i = 0; 2 * i + 1 < c; ++i) units_node_fast(&level[16 * i], &level[16 * i + 8], &level[8 * i]);
    if (c % 2)
      for (int w = 0; w < 8; ++w) level[8 * (c / 2) + w] = level[8 * (c - 1) + w];
  }
  for (int i = 0; i < 32; ++i) root[i] = (uint8_t)(level[i >> 2] >> (24 - 8 * (i & 3)));
  return 0;
}

// One tile's low-byte leaf (`tile_fused_lowbyte`) from its X and Y strips and c₀, for the variant's check.
extern "C" int ncp2_tile_lowbyte(const uint8_t* xs, const uint8_t* ys, const uint32_t* c0, int k3, int steps, uint8_t* leaf) {
  uint32_t run[D * D];
  for (int j = 0; j < D * D; ++j) run[j] = c0[j % D];
  tile_fused_lowbyte(xs, ys, k3, steps, ArrRun{run}, leaf);
  return 0;
}
