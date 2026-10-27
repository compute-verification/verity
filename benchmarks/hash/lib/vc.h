// libvc: the served path's SHA-512 commitments on the GPU (README.md). Every launch enqueues kernels on the caller's stream
// into caller-owned buffers and returns: no allocation, no host synchronization, no stream or global state of its own, so a
// whole commit captures into a CUDA graph and replays. A launch returns 0 or a cudaError_t (a negative value: a bad argument).
//
// Byte order: every digest a caller passes or receives is the digest's 64 bytes as hashlib gives them. Positions and node
// indices are below 2^32. Rows are 16-byte aligned with a stride that is a multiple of 16, readable up to the row's bytes
// rounded up to 16. Scratch buffers are 16-byte aligned and sized by the matching *_bytes query.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VC_ABI 1

enum { VC_ROW_NONE = 0, VC_ROW_V2 = 1, VC_ROW_SEG = 2, VC_ROW_TILE = 3 };
// GIVEN: a 192-byte salt per position from the caller. PRF: y_r = SHA-512(Ks || u64be(pos) || u64be(r)), r < 3 (commit_gpu).
// CHACHA64: ChaCha20 chunks 3 pos .. 3 pos + 2, 64-bit counter and 64-bit nonce (audit.hm96_salt). CHACHA_RFC: RFC 8439's
// 32-bit counter and 12-byte nonce (verity_vllm's serving_rows.salts), the stream at 192 pos.
enum { VC_SALT_GIVEN = 0, VC_SALT_PRF = 1, VC_SALT_CHACHA64 = 2, VC_SALT_CHACHA_RFC = 3 };
// vc_commit_rows_launch's forms: AUTO picks by device and size (up to 40,000 rows SPEC on sm_90, FOLD elsewhere; above, SPEC
// on sm_120 up to 200,000 rows, else SPLIT);
// SPLIT is the salt kernel, the data kernel, then the tree's level launches; FUSED one kernel a leaf (salt, mask from the
// nibble table through L1, row, leaf), then the tree; FOLD the salt kernel then one data kernel that also folds the tree (each
// CTA its 256 leaves' subtree, the last of two siblings climbing on); SPEC one kernel (a memset of scratch's counters first)
// whose CTAs pair a row thread with a salt thread a leaf, then fold each warp's 32 leaves and climb.
enum { VC_FORM_AUTO = 0, VC_FORM_SPLIT = 1, VC_FORM_FUSED = 2, VC_FORM_FOLD = 3, VC_FORM_SPEC = 4 };

typedef struct {
  const uint8_t* domain_id;     // 64 bytes: CommitmentDomain.domain_id of a frame-v3-sha512 domain
  const char* schema;           // the frame leaf's schema, 18 or 22 bytes ("hm96-sha512/row/v2", "hm96-sha512/row-seg/v1")
  int schema_len;
  const uint8_t* salt_prefix;   // 128: hm96.SHA512.salt_prefix, or NULL for a plan without hm96 leaves
  const uint8_t* leaf_prefix;   // 128: hm96.SHA512.leaf_prefix(key)
  const uint8_t* key;           // 256: the hm96 key (its tables are built on the device by vc_tables_launch)
  int row_scheme;               // VC_ROW_*
  uint64_t row_bits;            // VC_ROW_V2 / VC_ROW_SEG: the row's length in bits
  const uint8_t* row_prefix;    // 128: P(row_bits) (row/v2; or sha512/row/v1's block), Q(row_bits) (row-seg), Q of a full
                                // 64-row tile (tiles)
  const uint8_t* row_prefix_b;  // 128: tiles only, Q of the last row tile's tile (NULL when rows % 64 == 0)
  uint32_t tile_rows, tile_cols;// VC_ROW_TILE: the C~ and U matrices' shape (rows even, cols a multiple of 64)
  int salt_source;              // VC_SALT_*
  const uint8_t* salt_key;      // PRF: 64 bytes; CHACHA*: 32
  uint64_t salt_nonce;          // CHACHA64: the nonce; CHACHA_RFC: the 12-byte nonce's low 8 bytes (the rest zero)
  const uint8_t* seed_label;    // 128: Pearl-C -h3's seed-A label block (pearl_c.S512_LABELS["seed-A"]), or NULL
} vc_spec;

int vc_plan_bytes(void);
// The plan (host memory, vc_plan_bytes()) from spec: host arithmetic only. 0, or -1 for a spec it doesn't cover.
int vc_plan_build(const vc_spec* spec, void* plan);
// The plan's device tables (the hm96 nibble table, row-seg segment headers' states), filled by vc_tables_launch once.
uint64_t vc_tables_bytes(const void* plan);
int vc_tables_launch(const void* plan, void* tables, cudaStream_t st);
// The domain's empty root, on the host.
int vc_empty_root(const void* plan, uint8_t* out64);
// Kernel attributes (the salt kernel's dynamic shared memory) on the current device: call once before capturing a graph.
int vc_init(void);

// --- primitives ----------------------------------------------------------------------------------------------------
// SHA-512 of n messages of len bytes at msgs + stride i (16-byte aligned) into out + 64 i.
int vc_sha512_launch(const uint8_t* msgs, uint64_t stride, uint64_t len, uint64_t n, uint8_t* out, cudaStream_t st);
// n compressions: out[i] = compress(states[i], blocks[i]); a state is 64 bytes (eight big-endian words), a block 128.
int vc_compress_launch(const uint8_t* states, const uint8_t* blocks, uint64_t n, uint8_t* out, cudaStream_t st);
// An H1 table's chunk leaves (SPLITK; the caller's chunks = verity.core.protocols.pouw.weights.chunks(k)): n_rows rows of
// k int8 codes at codes + codes_stride i, each with its f32 scale at scales + 4 i, as n_rows chunks leaves, leaf
// chunks i + c at out + out_stride (chunks i + c): row i's codes [c k / chunks, (c + 1) k / chunks), its scale's 4 bytes,
// zeros to out_stride. Those are row/v2 rows of 8 (k / chunks + 4) bits for vc_commit_rows_launch at out_stride. k / chunks
// a multiple of 16, codes and out 16-byte aligned, out_stride a multiple of 16 of at least k / chunks + 4.
int vc_chunk_rows_launch(const uint8_t* codes, uint64_t codes_stride, const uint8_t* scales, uint64_t n_rows, uint32_t k,
                         uint32_t chunks, uint8_t* out, uint64_t out_stride, cudaStream_t st);

// --- row digests and tile leaves -------------------------------------------------------------------------------------
// Row digests under the plan's scheme (sha512/row/v2 or sha512/row-seg/v1) of n rows at rows + stride i into out + 64 i.
uint64_t vc_rows_scratch_bytes(const void* plan, uint64_t n_rows);
int vc_rows_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, uint64_t n_rows, uint8_t* out,
                   void* scratch, cudaStream_t st);
// Pearl-C -h3's tile leaves of the plan's C~ and U (u32 words, rows x cols, row-major): tile (ti, tj) at out + 64 (ti tn + tj),
// tn = cols / 64.
uint64_t vc_tiles_count(const void* plan);
uint64_t vc_tiles_scratch_bytes(const void* plan);
int vc_tiles_launch(const void* plan, const void* tables, const uint8_t* c, const uint8_t* u, uint8_t* out, void* scratch,
                    cudaStream_t st);
// Pearl-C -h3's row seeds (pearl_c.row_seed): H_seed-A(salt || digest_i || root_B || u64le(unit << 32 | i)), 32 bytes each,
// salt (32 bytes) and root_B (64) in device memory.
int vc_row_seeds_launch(const void* plan, const uint8_t* digests, uint64_t n, const uint8_t* salt32, const uint8_t* root_b64,
                        uint32_t unit, uint8_t* out32, cudaStream_t st);

// --- hm96-sha512 leaves in a frame-v3-sha512 tree ------------------------------------------------------------------
// The tree's buffer: level after level, level 0 first (width = 2^depth nodes, the pads included), the root last.
uint64_t vc_tree_bytes(uint64_t n_leaves);
uint64_t vc_tree_width(uint64_t n_leaves);
// Salt phase, positions first .. first + m - 1 (nothing here reads the operand): pos < n_total its salt's M_key y, c and frame
// leaf's first block to scratch record i (vc_salt_scratch_bytes(m)); n_total <= pos its pad to lv0 + 64 i. salts: the GIVEN
// source's salts, row i at 192 i.
uint64_t vc_salt_scratch_bytes(uint64_t m);
int vc_salts_launch(const void* plan, const void* tables, const uint8_t* salts, uint64_t first, uint64_t m, uint64_t n_total,
                    void* scratch, uint8_t* lv0, cudaStream_t st);
// Data phase over the salt phase's records: the leaf of each position below n_total to lv0 + 64 i, from x given (64-byte
// digests at x + 64 i) or from the row (sha512/row/v2 of rows + stride i). bc (128 bytes a leaf: b || c) and hs (64: h) when
// not NULL.
int vc_leaves_x_launch(const void* plan, const uint8_t* x, uint64_t first, uint64_t m, uint64_t n_total, const void* scratch,
                       uint8_t* lv0, uint8_t* bc, uint8_t* hs, cudaStream_t st);
int vc_leaves_rows_launch(const void* plan, const uint8_t* rows, uint64_t stride, uint64_t first, uint64_t m, uint64_t n_total,
                          const void* scratch, uint8_t* lv0, uint8_t* bc, uint8_t* hs, cudaStream_t st);
// Both phases in one kernel a leaf (pads included): x from rows (row/v2) when rows is not NULL, else from x.
int vc_leaves_fused_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, const uint8_t* x,
                           const uint8_t* salts, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* lv0, uint8_t* bc,
                           uint8_t* hs, cudaStream_t st);
// frame-v3-sha512 leaves over 64-byte values (pos < n_total) and pads (n_total <= pos), positions first .. first + m - 1.
int vc_frame_leaves_launch(const void* plan, const uint8_t* values, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out,
                           cudaStream_t st);
// One level: out[i] = node(level, first_index + i, in[2 i], in[2 i + 1]).
int vc_nodes_launch(const void* plan, const uint8_t* in, uint8_t* out, uint64_t n_out, int level, uint64_t first_index,
                    cudaStream_t st);
// Every level above level 0 (already in levels), the root last; n_leaves == 0: the empty root to levels.
int vc_tree_launch(const void* plan, uint8_t* levels, uint64_t n_leaves, cudaStream_t st);
// A what-if: the tree with nodes of `arity` children (2, 4 or 8), node(level, index, c_0 .. c_(arity-1)) framed as
// frame-v3's node with arity child parts; arity 2 is vc_tree_launch's tree. Level 0 is arity^depth nodes wide.
uint64_t vc_tree_arity_bytes(uint64_t n_leaves, int arity);
uint64_t vc_tree_arity_width(uint64_t n_leaves, int arity);
int vc_tree_arity_launch(const void* plan, uint8_t* levels, uint64_t n_leaves, int arity, cudaStream_t st);

// One call: n rows' hm96 leaves (x = SHA-512(the plan's prefix chunk || row): sha512/row/v2, or row/v1's block; salts from the
// plan's source, or `salts`), pads, and the tree into levels; every leaf's b || c to bc + 128 i when bc is not NULL.
// scratch: vc_commit_scratch_bytes(n). n = 0 writes only the empty root to levels (merkle's empty commitment, whose tree has no
// levels and whose root is the domain's empty root), in every form.
uint64_t vc_commit_scratch_bytes(uint64_t n);
int vc_commit_rows_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, const uint8_t* salts,
                          uint64_t n, uint8_t* levels, uint8_t* bc, void* scratch, int form, cudaStream_t st);

// --- openings --------------------------------------------------------------------------------------------------------
// Openings of m positions (u32 in device memory) of a committed tree of n_leaves leaves whose levels the caller kept. Record i
// at out + vc_open_bytes(n_leaves) i:
//   u64le position | u64le the leaf's byte offset in levels (64 position; ~0 for a position outside the tree, whose record is
//   otherwise zero) | the 192-byte salt y | the 128-byte b || c | the 64-byte frame leaf | merkle_depth(n_leaves) siblings,
//   bottom up, 64 bytes each.
// y is read from the commit's salt source: salts + 192 position (GIVEN; the whole tree's salts), else the plan's keyed source,
// the device code the commit ran. b || c is bc + 128 position when bc (the commit's retained b || c) is given, else from x + 64
// position (the inner digests) and y, else zero. A plan without hm96 leaves (frame_spec) zeroes y and b || c. A record verifies
// with merkle.verify_opening (the value hm96.SHA512.tree_leaf(b || c)), and the paths gathered with verify_multiproof. An empty
// tree (n_leaves = 0) has no position to open, as MerkleTree.open refuses them all: every record is outside, and salts, x and
// bc may be NULL.
uint64_t vc_open_bytes(uint64_t n_leaves);
int vc_open_launch(const void* plan, const void* tables, const uint8_t* x, const uint8_t* bc, const uint8_t* salts,
                   const uint8_t* levels, uint64_t n_leaves, const uint32_t* positions, uint64_t m, uint8_t* out, cudaStream_t st);

#ifdef __cplusplus
}
#endif
