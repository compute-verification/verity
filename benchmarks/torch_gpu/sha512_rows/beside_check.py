"""beside's gate: `beside gate M N K DIR`'s JSON lines and files against hashlib.

For each kernel the gate ran: y at its sampled cells against the host's Definition (the binary's `cells_wrong`, which must be
0); for its handoff arm, every row's digest against sha512/row/v2 of y's row (2 N bytes, bf16 little-endian), which the hash
warpgroup chained through the row's O tiles; for its hash arm, every CTA's 128 digests against sha512/row/v2 of the seed tile's
row repeated hash_tiles times. The seed tile is also drawn here (splitmix64 seed 5, little-endian) and must equal seed.bin.

    python3 beside_check.py GATE.jsonl DIR    # a VERDICT line a kernel and arm, then one for all; exit 1 on any mismatch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from lowreg_check import row_v2, tile_rows

TM, ROWB = 128, 512


def digest_bytes(words: bytes) -> bytes:
    """A digest as the kernel stores it (8 native little-endian u64 state words) as SHA-512's 64 bytes."""
    return b"".join(int.from_bytes(words[8 * i:8 * i + 8], "little").to_bytes(8, "big") for i in range(8))


def main(gate: str, d: str) -> int:
    lines = [json.loads(s) for s in open(gate) if s.startswith("{")]
    head, runs = lines[0], [x for x in lines[1:] if "kernel" in x]
    M, N = head["M"], head["N"]
    root = Path(d)
    seed = (root / "seed.bin").read_bytes()
    rows = tile_rows(5, TM, ROWB)
    seed_ok = seed == b"".join(rows)
    print(f"seed tile: seed.bin {'equals' if seed_ok else 'DIFFERS FROM'} splitmix64(5) drawn here")
    try:
        from verity.ml.commitments import rowleaf

        ref_ok = rowleaf.sha512_row_v2_digest(rows[0] * 3, 8 * ROWB * 3) == row_v2(rows[0] * 3)
        print(f"reference: rowleaf.sha512_row_v2_digest {'equals' if ref_ok else 'DIFFERS FROM'} hashlib's P || row")
    except ImportError as e:
        print(f"reference: rowleaf not imported ({e}); hashlib's P || row alone")
        ref_ok = True
    failed = 0
    for r in runs:
        name, arm = r["kernel"], r["arm"]
        notes = []
        good = True
        if arm in ("mma", "handoff"):
            good &= r["cells_wrong"] == 0
            notes.append(f"y {r['cells'] - r['cells_wrong']}/{r['cells']} cells equal the Definition's")
        if arm == "handoff":
            y = (root / f"{name}-handoff.y").read_bytes()
            dig = (root / f"{name}-handoff.digests").read_bytes()
            ok = sum(digest_bytes(dig[64 * i:64 * i + 64]) == row_v2(y[2 * N * i:2 * N * (i + 1)]) for i in range(M))
            good &= ok == M
            notes.append(f"{ok}/{M} row digests equal hashlib's of y's rows")
        if arm == "hash":
            dig = (root / f"{name}-hash.digests").read_bytes()
            want = [row_v2(row * r["hash_tiles"]) for row in rows]
            n = r["rows"]
            ok = sum(digest_bytes(dig[64 * i:64 * i + 64]) == want[i % TM] for i in range(n))
            good &= ok == n
            notes.append(f"{ok}/{n} digests equal hashlib's of the seed tile's rows x {r['hash_tiles']}")
        failed += not good
        print(f"VERDICT gate {name} {arm}: {'PASS' if good else 'FAIL'} {'; '.join(notes)} (regs {r['regs']}, "
              f"local {r['local_bytes']} B)")
    ok_all = not failed and seed_ok and ref_ok and runs
    print(f"VERDICT beside gate: {'PASS' if ok_all else 'FAIL'} {len(runs) - failed}/{len(runs)} kernel arms exact "
          f"(M {M}, N {N}, K {head['K']})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
