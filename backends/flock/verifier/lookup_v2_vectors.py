"""The `table/v2` generator's vectors (`lookup_v2_vectors.json`): the Lean verifier's `Lookup.buildV2` is the one
specification of a read's rows (Project doc `verified-lowering-design.md` §1.9, question 4), and Rust's `lookup.rs` and
Python's generators mirror it. Each entry is a table and a slot, `(n, lo, bits, library)`, with what `flock-verify
lookup-rows` computes: the varying width `k`, the slot's shape, and the rows digest `Lookup.rowsDigest`.

    python lookup_v2_vectors.py --lean BIN [--tables DIR] [--write]

- The small tables are generated here, deterministically, and embedded as hex: every entry can be reproduced from the file.
- The four MUFU tables are named by SHA-512 and read from `DIR/<sha512>` (an archive directory), when given.
- `--write` rewrites the file from the Lean output; otherwise the Lean output must equal it. Exit 1 on a difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "lookup_v2_vectors.json"

#: the pinned MUFU tables (`Flock.MUFU_TABLES`): name, index bits, SHA-512 of the raw u32 LE words
MUFU = [("rcp", 23, "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc"),
        ("ex2", 23, "5cb768c1725369ad905e725437463729d5b6054ca9adc498e755247e2ba848b510e3325e1c3fecad1242832aca5fa0afd463b16002a7f34050a7b377ed107877"),
        ("rsq", 24, "39f7b912bc25b913618d926725a18974af9eb246f6f36ce06ef63ffdb293d80bc222623c6d182d2c23e16f5ada2b8db9e7e6a53b3e42046763f7794ac3cef378"),
        ("sqrt", 24, "33d076106efaa955e6452200d45651a799628e76d1d1cc682168f464a03100b937bee08533761d9fa8f8a114cee2a32d322888d5f08fd320bfeb17805a5c2536")]


def small_tables() -> list[tuple[str, int, int, int, bytes]]:
    """(label, n, lo, bits, table bytes): full-width words, constant high bits (one of them 1), a constant bit below the
    varying ones, every bit constant, and one word alone setting the top bit."""
    rng = random.Random(20260928)

    def words(n, f):
        return b"".join(f(i).to_bytes(4, "little") for i in range(1 << n))

    out = [("full-8", 6, 3, 8, words(6, lambda i: rng.getrandbits(8))),
           ("low5-const-high", 6, 2, 12, words(6, lambda i: rng.getrandbits(5) | (1 << 7) | (1 << 10))),
           ("const-bit-inside", 7, 3, 10, words(7, lambda i: (rng.getrandbits(9) & ~(1 << 4)) | (1 << 2))),
           ("all-constant", 5, 2, 16, words(5, lambda i: 0x0000A5C3)),
           ("one-word-top", 7, 4, 31, words(7, lambda i: rng.getrandbits(20) | ((1 << 30) if i == 77 else 0))),
           ("random-20", 8, 5, 31, words(8, lambda i: rng.getrandbits(20)))]
    return out


def lean_rows(lean: str, path: Path, n: int, lo: int, bits: int, library: bool) -> dict:
    args = [lean, "lookup-rows", str(path), str(n), str(lo), str(bits)] + ([] if library else ["--program"])
    p = subprocess.run(args, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("LOOKUPROWS ")), None)
    if line is None:
        raise RuntimeError(p.stdout + p.stderr)
    r = json.loads(line.split(" ", 1)[1])
    return {k: r[k] for k in ("table_sha512", "n", "lo", "bits", "library", "k", "useful", "unit_log", "prod0", "lo_top",
                              "rows_sha512")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--tables", type=Path, default=None, help="a directory of tables by SHA-512 (an archive's sha512/)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    got = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, n, lo, bits, table in small_tables():
            f = Path(tmp) / f"{label}.u32"
            f.write_bytes(table)
            for library in (True, False):
                got.append({"label": label, "table_hex": table.hex(), **lean_rows(a.lean, f, n, lo, bits, library)})
    mufu = []
    if a.tables:
        for name, n, sha in MUFU:
            f = a.tables / sha
            if not f.exists():
                continue
            if hashlib.sha512(f.read_bytes()).hexdigest() != sha:
                print(f"{f}: not the table its name says", file=sys.stderr)
                return 1
            for library in (True, False):
                mufu.append({"label": name, **lean_rows(a.lean, f, n, 14, 31, library)})
    if a.write:
        doc = {"about": "table/v2 generator vectors (Flock.Lookup.buildV2, Lookup.rowsDigest); written by lookup_v2_vectors.py "
                        "--write from the compiled Lean verifier. The mirrors (Rust lookup.rs, Python) must reproduce every "
                        "rows_sha512.", "small": got, "mufu": mufu}
        VECTORS.write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"written": str(VECTORS), "small": len(got), "mufu": len(mufu)}))
        return 0
    want = json.loads(VECTORS.read_text())
    bad = [g["label"] for g, w in zip(got, want["small"]) if g != w] + ([] if len(got) == len(want["small"]) else ["count"])
    by = {(w["label"], w["library"]): w for w in want["mufu"]}
    bad += [f"{m['label']} library={m['library']}" for m in mufu if by.get((m["label"], m["library"])) != m]
    print(json.dumps({"small": len(got), "mufu": len(mufu), "disagree": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
