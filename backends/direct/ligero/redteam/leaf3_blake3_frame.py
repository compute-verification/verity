"""red-team-leaf-3: adversarial check of the BLAKE3 leaf's malformed-frame domain (blake3-leaf-2/-3).  The verifier's
``leaf.blake3.leaf_bytes`` maps a well-formed digest row to the standard keyed-BLAKE3 root of the row and a MALFORMED
one (header not ``n_chunks (2..3) + 4 role``, a limb >= 2^16, a non-zero unused slot) to ``SHA-256(MALFORMED_TAG || elems)``.
The adversarial question: can a malformed frame land on a real row's BLAKE3 root (so a garbage digest opens as a
committed leaf), or can two roles / two rows collide?  numpy + the ``blake3`` package, no GPU:

    PYTHONPATH=<blake3-leaf tree> python leaf3_blake3_frame.py

Exit 0 iff (1) every malformed frame maps to the tagged-SHA-256 domain, (2) no malformed frame equals any well-formed
root, (3) the same CVs under the two role keys give different roots, and (4) native() round-trips through leaf_bytes."""
from __future__ import annotations

import numpy as np


def main() -> int:
    from backends.direct.ligero.leaf import blake3 as B3

    wb = 8                                                    # E4M3 rows
    rng = np.random.default_rng(11)
    ok = True
    roots = {}
    for role in (1, 2):
        rows = rng.integers(0, 256, size=(8, B3.K_VU), dtype=np.int64)
        elems = B3.cvs_to_elems(B3.chunk_cvs_np(rows, wb, role)[1], role)     # (8, DIGEST_ELEMS) well-formed digests
        # (4) native() round-trip: leaf_bytes(digest) == blake3(row bytes, key=role) == native path
        for r in range(rows.shape[0]):
            lb = B3.leaf_bytes(elems[r])
            import blake3 as _b3
            want = _b3.blake3(np.asarray(rows[r], dtype=np.uint8).tobytes(), key=B3.key_for(role)).digest()
            ok &= (lb == want)
        roots[role] = [B3.leaf_bytes(elems[r]) for r in range(rows.shape[0])]
    # (3) role separation: same CV elements read under the other role key -> different root (role is in the key)
    rows = rng.integers(0, 256, size=(1, B3.K_VU), dtype=np.int64)
    e1 = B3.cvs_to_elems(B3.chunk_cvs_np(rows, wb, 1)[1], 1)[0]
    # force role 2 header on role-1 CV limbs: a well-formed frame, but the fold uses role 2's key -> a different domain
    e1b = e1.copy(); nch, _ = B3.parse_header(int(e1[0])); e1b[0] = B3.header(nch, 2)
    role_sep = B3.leaf_bytes(e1) != B3.leaf_bytes(e1b)
    ok &= role_sep

    # (1)+(2) malformed frames -> tagged domain, never a real root
    good = set(x for v in roots.values() for x in v) | {B3.leaf_bytes(e1), B3.leaf_bytes(e1b)}
    malformed = []
    base = e1.copy()
    m1 = base.copy(); m1[0] = 0                               # header 0: role/n_chunks out of range
    m2 = base.copy(); m2[0] = 3 + 4 * 3                       # role 3 (only 1, 2 valid)
    m3 = base.copy(); m3[1] = 1 << 16                         # a limb >= 2^16
    m4 = base.copy(); m4[-1] = 7                              # a non-zero UNUSED slot
    for m in (m1, m2, m3, m4):
        lb = B3.leaf_bytes(m)
        import hashlib
        tagged = hashlib.sha256(B3.MALFORMED_TAG + np.asarray(m, dtype="<u4").tobytes()).digest()
        malformed.append((lb == tagged, lb in good))
    all_tagged = all(t for t, _ in malformed)
    none_collide = not any(c for _, c in malformed)
    ok &= all_tagged and none_collide

    print(f"native round-trip + role separation: role_sep={role_sep}")
    print(f"malformed frames map to tagged SHA-256: {all_tagged}; any lands on a real BLAKE3 root: {not none_collide}")
    print(f"OVERALL {'PASS (framing sound: no cross-domain collision)' if ok else 'FAIL (a domain leak)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
