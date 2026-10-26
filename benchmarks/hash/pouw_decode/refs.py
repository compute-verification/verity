"""The Python references pd's launches are held to, on Pearl-C -h3's own definitions: `pearl_c.row_digest`, A's
`audit.HM96` tree as the kernels lay it out (`fixture.tree_bytes`' levels: level 0 under each row's salt, its pads, then
each level above), `pearl_c.row_seed` with its `pearl_c.line_key`, `pearl_c.unit_seed_a` with E_A's line key line_a, E_A's
lines (`pearl_c.sample_line`: row r's line 0 under its row seed, filler row t's line t under seed_A), and the tile leaves
`pearl_c.leaf` under -h3."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from verity.ml.pouw.audit import HM96, ROW_SCHEMA, activation_binding, hm96_salt, rows_domain  # noqa: E402
from verity.ml.pouw.schemes import pearl_c as C  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402


def commit(rows: list[bytes], salt: bytes, root_b: bytes, salt_key: bytes, index: int, lines: int = 0) -> dict[str, bytes]:
    """A's commitment over `rows` (each a row's FP32 bytes) for call `index`: the buffers pd_commit writes, with E_A's
    codes over `lines` lines (the call's rows, filler rows after the real ones) when it is given."""
    m16 = len(rows)
    dom = rows_domain(activation_binding(salt, index), index, m16, HM96)
    leaves = [dom.leaf(i, i, ROW_SCHEMA, v + hm96_salt(salt_key, index, i)) for i, v in enumerate(rows)]
    levels = tree_levels(dom, leaves)
    root = levels[-1][0]
    digests = [C.row_digest(v) for v in rows]
    seeds = [C.row_seed(salt, d, root_b, index, r, "h3") for r, d in enumerate(digests)]
    seed_a = C.unit_seed_a(salt, root, root_b, index, "h3")
    out = {"rowdig_a": b"".join(digests), "tree_a": b"".join(b"".join(level) for level in levels),
           "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C.line_key(s) for s in seeds), "seed_a": bytes(seed_a),
           "line_a": C.line_key(seed_a), "dom_a": dom.domain_id}
    if lines:
        out["ea_codes"] = b"".join(bytes(C.sample_line(C.S512Seed(seeds[t]), 0, 0, 0) if t < m16 else
                                         C.sample_line(C.S512Seed(seed_a), 0, 0, t)) for t in range(lines))
    return out


def tile_leaf(c, u, ti: int, tj: int, rows: int) -> bytes:
    """Tile (ti, tj)'s leaf over C~ and U ((rows x cols) uint32 arrays), its last row block partial."""
    r = 64 * ti
    rr = min(64, rows - r)
    return C.leaf(C.Checked(c[r:r + rr, 64 * tj:64 * tj + 64].tolist(), u[r:r + rr, 64 * tj:64 * tj + 64].tolist()), "h3")
