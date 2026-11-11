"""CIR-242 scratch: the rows and bytes each one-writer form commits for a sparse-MLA key block, a (head, key block) and a
(row, key block), and the sum of its units' Flock blocks, from the JSON lines `cir242_key_rows.py` prints.

A wired port, or one read by reference, commits no row of its own; the carried state (``m_prev``, ``l_prev``, ``pb``,
``rescale``, ``o_prev``) is wired from the block before in every form.  The forms' keys:

* proofs' split: each scores and values unit commits its own ``kv``;
* a, by reference: the keys as one registered value, a row a column block, read by every unit;
* b, a writer: an identity `AllGather_v1{WORLD=64,N=64}` a column block, its input its own row, its output wired;
* c1, a wire a key: the scores and values units cut a key a row, each wired to its key's writer's ``o0``;
* c2, column gathers: `GatherCols{C}` a column block cut a key a row, wired to the keys' writers, its output wired.

``python cir242_key_rows.py | python cir242_forms.py``."""
from __future__ import annotations

import json
import sys

H, NC, BLOCK = 8, 9, 64 * 64 * 2
CARRIED = {"m_prev", "l_prev", "o_prev", "rescale", "pb"}


def own(u, name, drop=()):
    d = u[name]
    ins = [b for p, b in d["in_rows"] if p.split("#")[0] not in CARRIED | set(drop)]
    return len(ins) + len(d["out_rows"]), sum(ins) + sum(b for _, b in d["out_rows"])


def form(u, sc, va, keys_rows=0, keys_bytes=0, drop=(), extra_bits=0):
    out = {}
    for tag, s, v in (("first", sc + "F", va + "F"), ("later", sc + "L", va + "L")):
        (sr, sb), (vr, vb) = own(u, s, drop), own(u, v, drop)
        hr, hb = sr + 8 * vr, sb + 8 * vb
        out[tag] = (hr, hb, H * hr + keys_rows, H * hb + keys_bytes, H * (u[s]["bits"] + 8 * u[v]["bits"]) + extra_bits)
    return out


def main() -> None:
    u = {d["name"]: d for d in map(json.loads, sys.stdin)}
    forms = {
        "proofs' split": form(u, "scores64", "values64"),
        "a by reference": form(u, "scoresCols64", "values64", NC, NC * BLOCK, ("kv",)),
        "b a writer": form(u, "scoresCols64", "values64", 2 * NC, 2 * NC * BLOCK, ("kv",), NC * u["writer"]["bits"]),
        "c1 a wire a key": form(u, "scoresKeys64", "valuesWholeKeys64", drop=("kv",)),
        "c2 column gathers": form(u, "scoresCols64", "values64", NC, NC * BLOCK, ("kv",), NC * u["gatherCols"]["bits"]),
    }
    for name, f in forms.items():
        for tag, (hr, hb, rr, rb, bits) in f.items():
            print(json.dumps({"form": name, "block": tag, "head_rows": hr, "head_bytes": hb, "row_rows": rr,
                              "row_bytes": rb, "row_bits": bits}))


if __name__ == "__main__":
    main()
