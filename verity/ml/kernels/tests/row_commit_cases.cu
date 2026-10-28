// ../row_commit.cuh's leaf, pad, node and hm96 leaf, one case at a time, for test_row_commit.py: built by a host compiler
// (`-x c++`) it runs them on the host, and built by nvcc also on the device, a thread a case.
#include "../row_commit.cuh"

// Named: the extern "C" entry points take its Case and Out, and nvcc gives a function whose parameter types have internal
// linkage internal linkage too, which would hide rc_cases and rc_cases_cuda from ctypes.
namespace rcc {

using rc::u64;

enum Kind { LEAF, PAD, NODE, HM96, EMPTY, LEAF_AT };
// LEAF, PAD, NODE: 0 compile-time nb, 1 run-time nb; LEAF 2 compile-time a body a length (frame_leaf_nb), NODE 2 run-time
// through Compact. PAD 2 pad_of, NODE 3 node_at from node_mid, LEAF_AT 1 from its head (leaf_head, leaf_tail). HM96 0 to 4:
// the table and loop of the mask, and the frame leaf's nb; 5 and 6: mask_lanes over a strided salt in two column orders,
// then the leaf from its head.
constexpr int HM96_FORMS = 7;

struct Case {
  int32_t kind, form, sl, level;
  uint64_t pos, position;  // pos: the leaf's rank and position, the pad's rank or the node's index; LEAF_AT's position
  uint8_t did[64], schema[96], value[64], L[64], R[64], x[64], y[192];
};
struct Out {
  uint8_t bc[128], hs[64], out[64];
  int32_t status, pad_[3];
};
// the key's tables: the nibble table, and the byte table at strides 272 and 256
struct Tabs {
  const uint8_t *nib, *b272, *b256, *key;
};

RC_HD void words(const uint8_t* p, u64* w, int n) {
  for (int i = 0; i < n; ++i) w[i] = rc::be64_at(p + 8 * i);
}
RC_HD void bytes(const u64* w, uint8_t* p, int n) {
  for (int i = 0; i < n; ++i) rc::be_put(p + 8 * i, w[i], 8);
}

template <int SX>
RC_HD void leaf_ct(int form, const rc::FrameK& F, u64 pos, const u64* v, u64* out) {
  if (form == 0) {
    rc::frame_leaf<SX>(F, pos, v, out);
    return;
  }
  switch (rc::nbytes(pos)) {
    case 1: rc::frame_leaf_nb<SX, 1>(F, pos, v, out); break;
    case 2: rc::frame_leaf_nb<SX, 2>(F, pos, v, out); break;
    case 3: rc::frame_leaf_nb<SX, 3>(F, pos, v, out); break;
    default: rc::frame_leaf_nb<SX, 4>(F, pos, v, out); break;
  }
}
// the compile-time leaf (form 0, or 2 through frame_leaf_nb) for the schema lengths the vectors take: 13, 17, 18 and 22
RC_HD int leaf_by_sx(int form, const rc::FrameK& F, u64 pos, const u64* v, u64* out) {
  switch (F.sx) {
    case -5: leaf_ct<-5>(form, F, pos, v, out); return 0;
    case -1: leaf_ct<-1>(form, F, pos, v, out); return 0;
    case 0: leaf_ct<0>(form, F, pos, v, out); return 0;
    case 4: leaf_ct<4>(form, F, pos, v, out); return 0;
    default: return -1;
  }
}

template <int SX, int FORM>
RC_HD void hm96_form(const rc::FrameK& F, const rc::Hm96K& H, const Tabs& T, u64 pos, const u64* x, const u64* Y, Out& o) {
  uint32_t mk[16];
  u64 my[8], c[8], f[8];
  if constexpr (FORM == 0) rc::mask<false>(rc::NibTab<true>{T.nib}, Y, mk);
  else if constexpr (FORM == 1) rc::mask<true>(rc::NibTab<false>{T.nib}, Y, mk);
  else if constexpr (FORM == 2) rc::mask<true>(rc::ByteTab<272, false>{T.b272}, Y, mk);
  else if constexpr (FORM == 3) rc::mask<false>(rc::ByteTab<272, false>{T.b272}, Y, mk);
  else rc::mask<false>(rc::ByteTab<256, true>{T.b256}, Y, mk);
  rc::mask_words(mk, my);
  rc::salt_digest(H, Y, c);
  constexpr bool RT = FORM & 1;
  if constexpr (RT) rc::frame_first_rt(F, pos, f);
  else rc::frame_first(F, pos, f);
  rc::finish_leaf<SX, RT>(F, H, pos, x, my, c, f, o.bc, o.hs);
  bytes(f, o.out, 8);
}
template <int SX>
RC_HD int hm96_by_form(int form, const rc::FrameK& F, const rc::Hm96K& H, const Tabs& T, u64 pos, const u64* x, const u64* Y,
                       Out& o) {
  switch (form) {
    case 0: hm96_form<SX, 0>(F, H, T, pos, x, Y, o); return 0;
    case 1: hm96_form<SX, 1>(F, H, T, pos, x, Y, o); return 0;
    case 2: hm96_form<SX, 2>(F, H, T, pos, x, Y, o); return 0;
    case 3: hm96_form<SX, 3>(F, H, T, pos, x, Y, o); return 0;
    case 4: hm96_form<SX, 4>(F, H, T, pos, x, Y, o); return 0;
    default: return -1;
  }
}

template <int FORM>
struct Cols {
  RC_HD uint32_t operator()(int k) const { return FORM == 5 ? k : (7 * k + 5) % 12; }
};
// HM96 5 and 6: the salt as little-endian u32 words 3 apart, the mask in column order Cols, and the leaf from its head
template <int FORM>
RC_HD int hm96_lanes(const Case& k, const rc::Hm96K& H, const Tabs& T, const u64* x, const u64* Y, Out& o) {
  constexpr uint32_t YS = 3;
  uint32_t y[48 * YS], mk[16];
  for (int w = 0; w < 48; ++w) {
    uint32_t v = 0;
    for (int b = 3; b >= 0; --b) v = v << 8 | ((uint32_t)(Y[(4 * w + b) >> 3] >> (56 - 8 * ((4 * w + b) & 7))) & 0xFFu);
    y[w * YS] = v;
  }
  u64 my[8], B[16], h[8], mid[8], tail[16], f[8];
  rc::mask_lanes(rc::ByteTab<256, false>{T.b256}, y, YS, Cols<FORM>{}, mk);
  rc::mask_words(mk, my);
  for (int w = 0; w < 8; ++w) B[w] = x[w] ^ my[w];
  rc::salt_digest(H.mid_salt, rc::SaltU32{y, YS}, B + 8);
  rc::st_be(o.bc, B, 16);
  rc::hm96_value(H.mid_leaf, H.kw_leafpad, B, h);
  rc::st_be(o.hs, h, 8);
  const int q = rc::leaf_head(k.did, k.pos, k.pos, k.schema, k.sl, mid, tail);
  if (q < 0) return -1;
  rc::leaf_tail(mid, tail, (uint32_t)q, h, f);
  bytes(f, o.out, 8);
  return 0;
}

// the frame message of a leaf at any rank and position, hashed whole (the builders' path)
RC_HD void leaf_at(const Case& k, u64* out) {
  uint8_t m[384] = {};
  int n = rc::frame_head(m, "leaf", k.did);
  n = rc::frame_uint(m, n, k.pos);
  n = rc::frame_uint(m, n, k.position);
  n = rc::frame_part(m, n, k.schema, k.sl);
  n = rc::frame_part(m, n, k.value, 64);
  rc::digest_padded(m, n, out);
}

RC_HD int run(const Case& k, const Tabs& T, rc::FrameK& F, rc::NodeK& N, Out& o) {
  u64 v[8], d[8], L[8], R[8];
  switch (k.kind) {
    case LEAF:
      if (rc::frame_plan(k.did, k.schema, k.sl, F)) return -1;
      words(k.value, v, 8);
      if (k.form != 1) {
        if (leaf_by_sx(k.form, F, k.pos, v, d)) return -1;
      } else if (F.sx == 0) {
        rc::frame_first_rt(F, k.pos, d);
        rc::frame_second_rt<0>(F, k.pos, v, d);
      } else if (F.sx == 4) {
        rc::frame_first_rt(F, k.pos, d);
        rc::frame_second_rt<4>(F, k.pos, v, d);
      } else {
        return -1;
      }
      break;
    case PAD:
      if (k.form == 2) {
        rc::pad_of(k.did, k.pos, d);
        break;
      }
      if (rc::frame_plan(k.did, k.schema, k.sl, F)) return -1;
      if (k.form == 0) rc::pad(F, k.pos, d);
      else rc::pad_rt(F, k.pos, d);
      break;
    case NODE:
      words(k.L, L, 8);
      words(k.R, R, 8);
      if (k.form == 3) {
        u64 mid[8];
        rc::node_mid(k.did, k.level, mid);
        rc::node_at(mid, k.pos, L, R, d);
        break;
      }
      if (rc::node_plan(k.did, N)) return -1;
      if (k.form == 0) rc::node(N, k.level, k.pos, L, R, d);
      else if (k.form == 1) rc::node_rt(N, k.level, k.pos, L, R, d);
      else rc::node_rt<rc::Compact>(N, k.level, k.pos, L, R, d);
      break;
    case EMPTY:
      rc::empty_root(k.did, d);
      break;
    case LEAF_AT:
      if (k.form == 1) {
        u64 mid[8], tail[16];
        const int q = rc::leaf_head(k.did, k.pos, k.position, k.schema, k.sl, mid, tail);
        if (q < 0) return -1;
        words(k.value, v, 8);
        rc::leaf_tail(mid, tail, (uint32_t)q, v, d);
      } else {
        leaf_at(k, d);
      }
      break;
    case HM96: {
      rc::Hm96K H;
      rc::hm96_plan(T.key, H);
      u64 x[8], Y[24];
      words(k.x, x, 8);
      words(k.y, Y, 24);
      if (k.form == 5) return hm96_lanes<5>(k, H, T, x, Y, o);
      if (k.form == 6) return hm96_lanes<6>(k, H, T, x, Y, o);
      if (rc::frame_plan(k.did, k.schema, k.sl, F)) return -1;
      if (F.sx == 0) return hm96_by_form<0>(k.form, F, H, T, k.pos, x, Y, o);
      if (F.sx == 4) return hm96_by_form<4>(k.form, F, H, T, k.pos, x, Y, o);
      return -1;
    }
    default:
      return -1;
  }
  bytes(d, o.out, 8);
  return 0;
}

}  // namespace

using namespace rcc;

extern "C" int rc_case_bytes() { return (int)sizeof(Case); }
extern "C" int rc_out_bytes() { return (int)sizeof(Out); }
extern "C" int rc_hm96_forms() { return HM96_FORMS; }

// key's tables (rc::nib_entry, rc::byte_chunk): nib (TN_BYTES), b272 (256 rows 272 apart) and b256 (256 rows 256 apart)
extern "C" void rc_tables(const uint8_t* key, uint8_t* nib, uint8_t* b272, uint8_t* b256) {
  for (uint32_t t = 0; t < 512; ++t) rc::nib_entry(key, t, nib);
  for (uint32_t b = 0; b < 256; ++b)
    for (uint32_t c = 0; c < 16; ++c) {
      const rc::q4 q = rc::byte_chunk(reinterpret_cast<const rc::q4*>(nib), b, c);
      memcpy(b272 + 272 * b + 16 * c, &q, 16);
      memcpy(b256 + 256 * b + 16 * c, &q, 16);
    }
}

// key's Hm96K: mid_salt then mid_leaf
extern "C" void rc_hm96_plan(const uint8_t* key, uint64_t* out) {
  rc::Hm96K H;
  rc::hm96_plan(key, H);
  for (int i = 0; i < 8; ++i) out[i] = H.mid_salt[i], out[8 + i] = H.mid_leaf[i];
}

extern "C" void rc_cases(const Case* cases, int n, const uint8_t* key, const uint8_t* nib, const uint8_t* b272,
                         const uint8_t* b256, Out* outs) {
  static rc::FrameK F;
  static rc::NodeK N;
  const Tabs T{nib, b272, b256, key};
  for (int i = 0; i < n; ++i) {
    outs[i] = Out{};
    outs[i].status = run(cases[i], T, F, N, outs[i]);
  }
}

#ifdef __CUDACC__
namespace {

__global__ void k_cases(const Case* cases, int n, Tabs T, rc::FrameK* F, rc::NodeK* N, Out* outs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) outs[i].status = run(cases[i], T, F[i], N[i], outs[i]);
}

}  // namespace

// rc_cases on the device, a thread a case: 0, or the CUDA error
extern "C" int rc_cases_cuda(const Case* cases, int n, const uint8_t* key, const uint8_t* nib, const uint8_t* b272,
                             const uint8_t* b256, Out* outs) {
  Case* dc = nullptr;
  Out* dout = nullptr;
  rc::FrameK* dF = nullptr;
  rc::NodeK* dN = nullptr;
  uint8_t *dkey = nullptr, *dnib = nullptr, *d272 = nullptr, *d256 = nullptr;
  cudaError_t e = cudaMalloc(&dc, sizeof(Case) * n);
  if (!e) e = cudaMalloc(&dout, sizeof(Out) * n);
  if (!e) e = cudaMalloc(&dF, sizeof(rc::FrameK) * n);
  if (!e) e = cudaMalloc(&dN, sizeof(rc::NodeK) * n);
  if (!e) e = cudaMalloc(&dkey, 256);
  if (!e) e = cudaMalloc(&dnib, rc::TN_BYTES);
  if (!e) e = cudaMalloc(&d272, 256 * 272);
  if (!e) e = cudaMalloc(&d256, 256 * 256);
  if (!e) e = cudaMemcpy(dc, cases, sizeof(Case) * n, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(dkey, key, 256, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(dnib, nib, rc::TN_BYTES, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(d272, b272, 256 * 272, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemcpy(d256, b256, 256 * 256, cudaMemcpyHostToDevice);
  if (!e) e = cudaMemset(dout, 0, sizeof(Out) * n);
  if (!e) {
    k_cases<<<(n + 31) / 32, 32>>>(dc, n, Tabs{dnib, d272, d256, dkey}, dF, dN, dout);
    e = cudaGetLastError();
  }
  if (!e) e = cudaMemcpy(outs, dout, sizeof(Out) * n, cudaMemcpyDeviceToHost);
  cudaFree(dc); cudaFree(dout); cudaFree(dF); cudaFree(dN);
  cudaFree(dkey); cudaFree(dnib); cudaFree(d272); cudaFree(d256);
  return (int)e;
}
#endif
