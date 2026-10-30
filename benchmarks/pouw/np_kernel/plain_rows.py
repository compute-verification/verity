"""Write the served committer's plain-leaf copy: native_rows.cu with its symbols renamed (verity_rows_p_*, namespace sha_p)
and its split form's leaf made frame-v3-sha512's row leaf over the row digest, without hm96 (L1's words, whose hiding is one
hm96 instance over their subtree): the salt kernel keeps only the frame leaf's first block, and the data kernel finishes
the frame leaf on x instead of on the hm96 leaf h. Every edit must apply exactly once, so a change to the served file
fails the build here instead of timing something else.

    python3 plain_rows.py NATIVE_ROWS_CU OUT_CU
"""

from __future__ import annotations

import re
import sys

EDITS = (
    (
        '#include "sha512.cuh"',
        '#define sha sha_p\n#include "sha512.cuh"',
    ),
    (
        """  extern __shared__ uint4 T8[];
  load_t8(Tn, T8);
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  if (i < nd) {
    u64 Y[24], c[8], my[8], f[8];
    u32 mk[16];
    if (salts) ld_be(salts + 192 * i, Y, 24);
    else salt_prf(K, pos, Y);
    mask_roll<true>(T8, Y, mk);
    salt_digest(K, Y, c);
#pragma unroll
    for (int w = 0; w < 8; ++w) my[w] = mask_word(mk, w);
    u64* s = reinterpret_cast<u64*>(sf + 192 * i);
    st_raw(s, my, 8);
    st_raw(s + 8, c, 8);
    if constexpr (H_ONLY) return;
""",
        """  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  if (i < nd) {
    u64 f[8];
    u64* s = reinterpret_cast<u64*>(sf + 192 * i);
    if constexpr (H_ONLY) return;
""",
    ),
    (
        """  const u64* s = reinterpret_cast<const u64*>(sf + 192 * i);
  ld_raw(s, B, 16);
#pragma unroll
  for (int w = 0; w < 8; ++w) B[w] ^= x[w];
  if (bc) st_be(bc + 128 * i, B, 16);
  cpy8(h, K.mid_leaf);
  sha::compress(h, B);
  sha::compress_kw(h, K.kw_leafpad);
  if (hs) st_be(hs + 64 * i, h, 8);
  if constexpr (H_ONLY) return;
  ld_raw(s + 16, f, 8);
  frame_second(K, first + i, h, f);
""",
        """  const u64* s = reinterpret_cast<const u64*>(sf + 192 * i);
  (void)B; (void)h;
  if constexpr (H_ONLY) return;
  ld_raw(s + 16, f, 8);
  frame_second(K, first + i, x, f);
""",
    ),
)


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    text = open(src).read()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"plain_rows.py: an edit's anchor occurs {n} times in {src}, not once: {old.splitlines()[0]!r}")
        text = text.replace(old, new)
    text, n = re.subn(r"\bverity_rows_", "verity_rows_p_", text)
    if not n:
        sys.exit(f"plain_rows.py: no verity_rows_ symbol in {src}")
    open(out, "w").write(text)


if __name__ == "__main__":
    main()
