"""NCP's checked values as Lean itself evaluates them, over explicit A, B, E₁, F₁ and a signed permutation R, at every
16-deep checkpoint: route U's `Pouw.NCP.mmWU 128` (the checked values) and the unbiased `Pouw.mmW` over `stackX` /
`stackY` (retired, kept for comparison).  The pinned result is ``tests/vectors/ncp_lean.json``.

    python verity/core/protocols/pouw/tests/ncp_lean.py write DIR        # DIR: a copy of Security (Mathlib built)
    (cd DIR && lake build Definitions.Pouw.NCP.RouteU && lake env lean NcpCheck.lean > ncp_lean.out)
    python verity/core/protocols/pouw/tests/ncp_lean.py pin DIR/ncp_lean.out DIR

Inputs are derived here (SHAKE256 of fixed labels), so only Lean's output is pinned.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from verity.core.protocols.pouw.schemes import ncp

PINNED = Path(__file__).with_name("vectors") / "ncp_lean.json"
K, M, N = 64, 2, 2
SOURCES = ("Definitions/Pouw/NCP/Defs.lean", "Definitions/Pouw/Basic/Mat.lean", "Definitions/Pouw/NCP/RouteU.lean")


def draws(label: str, n: int, lo: int, span: int) -> list[int]:
    return [lo + b % span for b in hashlib.shake_256(f"verity/pouw/ncp-lean/v1\0{label}".encode()).digest(n)]


def perm_case(kind: str) -> tuple[list[int], list[int]]:
    if kind == "word":
        pi, sg = ncp.word_perm(K)
    elif kind == "shift24":
        pi, sg = ncp.shift_perm(K)
    else:                                                    # a random signed permutation (Fisher-Yates on SHAKE bytes)
        pi, r = list(range(K)), draws("signed/perm", 4 * K, 0, 1 << 8)
        for i in range(K - 1, 0, -1):
            j = (r[i] * 256 + r[K + i]) % (i + 1)
            pi[i], pi[j] = pi[j], pi[i]
        sg = [1 - 2 * (b & 1) for b in draws("signed/signs", K, 0, 256)]
    return list(pi), list(sg)


CASES = ("word", "shift24", "signed")


def inputs(kind: str) -> dict:
    return {"a": [draws(f"{kind}/A/{i}", K, -64, 128) for i in range(M)], "b": [draws(f"{kind}/B/{j}", K, -64, 128) for j in range(N)],
            "e1": [draws(f"{kind}/E1/{i}", K, -31, 64) for i in range(M)], "f1": [draws(f"{kind}/F1/{j}", K, -31, 64) for j in range(N)],
            "pi": perm_case(kind)[0], "sigma": perm_case(kind)[1]}


def s32(rows: list[list[list[int]]]) -> list[list[list[int]]]:
    """mod-2^32 words as signed int32 (Lean's `wrap32`)."""
    return [[[w - (1 << 32) if w >> 31 else w for w in s] for s in row] for row in rows]


def ops(kind: str) -> ncp.Ops:
    x = inputs(kind)
    return ncp.assemble(x["a"], x["b"], x["e1"], x["f1"], x["pi"], x["sigma"])


def python_checked(kind: str) -> list[list[list[int]]]:
    """The scheme's checked values (route U, β = 128)."""
    return s32(ncp.NCP(k_min=64, n_min=64).checked(None, ops(kind)))


def python_unbiased(kind: str) -> list[list[list[int]]]:
    return s32(ncp.checked_unbiased(ops(kind)))


def lean_arr(rows: list[list[int]]) -> str:
    return "#[" + ", ".join("#[" + ", ".join(f"({v} : Int)" for v in r) + "]" for r in rows) + "]"


def lean_source() -> str:
    out = ["import Definitions.Pouw.NCP.RouteU", "open Pouw Pouw.NCP", "",
           "def ofRows (xs : Array (Array Int)) : Mat := fun i j => (xs.getD i #[]).getD j 0",
           "def ofCols (xs : Array (Array Int)) : Mat := fun l j => (xs.getD j #[]).getD l 0",
           "def signedPerm (pi : Array Nat) (sg : Array Int) : Mat := fun i j => if j < pi.size ∧ i = pi.getD j 0 then sg.getD j 0 else 0",
           "def checked (k m n : Nat) (A B E1 F1 R : Mat) : List (List (List Int)) :=",
           "  let X := stackX k A E1 R",
           "  let Y := stackY k B R F1",
           "  (List.range m).map fun i => (List.range n).map fun j =>",
           "    (List.range (3 * k / 16)).map fun t => mmW (min ((t + 1) * 16) (3 * k)) X Y i j",
           "def checkedUVec (β : Int) (k m n : Nat) (A B E1 F1 R : Mat) : List (List (List Int)) :=",
           "  (List.range m).map fun i => (List.range n).map fun j =>",
           "    (List.range (3 * k / 16)).map fun t => mmWU β k A B E1 F1 R (min ((t + 1) * 16) (3 * k)) i j", ""]
    for kind in CASES:
        x = inputs(kind)
        pi = "#[" + ", ".join(str(p) for p in x["pi"]) + "]"
        sg = "#[" + ", ".join(f"({s} : Int)" for s in x["sigma"]) + "]"
        args = (f"{K} {M} {N} (ofRows {lean_arr(x['a'])}) (ofCols {lean_arr(x['b'])}) (ofRows {lean_arr(x['e1'])}) "
                f"(ofCols {lean_arr(x['f1'])}) (signedPerm {pi} {sg})")
        out.append(f"#eval checked {args}")
        out.append(f"#eval checkedUVec {ncp.BETA} {args}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        (Path(sys.argv[2]) / "NcpCheck.lean").write_text(lean_source())
    elif cmd == "pin":
        text, dec, vals, pos = Path(sys.argv[2]).read_text(), json.JSONDecoder(), [], 0
        while (pos := text.find("[", pos)) >= 0:
            value, pos = dec.raw_decode(text, pos)
            vals.append(value)
        assert len(vals) == 2 * len(CASES), len(vals)
        lean_dir = Path(sys.argv[3])
        PINNED.write_text(json.dumps({
            "schema": "verity-pouw/ncp-lean-vectors/v2", "toolchain": (lean_dir / "lean-toolchain").read_text().strip(),
            "sources_sha256": {s: hashlib.sha256((lean_dir / s).read_bytes()).hexdigest() for s in SOURCES},
            "shape": [M, K, N], "beta": ncp.BETA,
            "route_u": {kind: vals[2 * c + 1] for c, kind in enumerate(CASES)},
            "unbiased": {kind: vals[2 * c] for c, kind in enumerate(CASES)}}, indent=1) + "\n")
        print(f"wrote {PINNED}")
