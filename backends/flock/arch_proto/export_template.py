"""Export the P1 step template (``step_template.build_step``) as one 2^13-row slot of Flock's block R1CS ``A.z * B.z = z``.

Slot-local columns: 0 is the slot's constant (its row is empty here; the block wires it to the pin through Delta), then the
704 inputs (accumulator word, 16 + 16 decoded operands; rows A = {} B = {0}, sourced through Delta), then the live ANDs, then
one copy row per output bit that is not already a single committed bit. Rows past the last are empty (z = 0).

File (little-endian u32): magic 0x54504C31, rows, used, then per row ``len_a, a.., len_b, b..``, then 32 accumulator input
columns, 672 operand input columns, 32 output columns.
"""
from __future__ import annotations

import argparse
import json
import struct

import step_template as S

MAGIC = 0x54504C31
ROWS = 1 << 13


def export(path: str) -> dict:
    C = S.build_step()
    live = C.live()
    col = {}
    order = []
    for name, bits in C.inputs.items():
        for w in bits:
            (v,) = w.s
            col[v] = 1 + len(order)
            order.append(("in", v))
    for v in sorted(live):
        if C.kind[v] == "and":
            col[v] = 1 + len(order)
            order.append(("and", v))
        else:
            assert C.kind[v] == "in", C.kind[v]

    def cols(form):
        out = sorted(col[v] for v in form.s)
        return out + [0] if form.c else out

    rows = [([], [])]
    for kind, v in order:
        if kind == "in":
            rows.append(([], [0]))
        else:
            a, b = C.data[v]
            rows.append((cols(a), cols(b)))
    outs = []
    for w in C.outputs["o"]:
        if len(w.s) == 1 and w.c == 0:
            outs.append(col[next(iter(w.s))])
        else:
            outs.append(len(rows))
            rows.append((cols(w), [0]))
    used = len(rows)
    assert used <= ROWS, used
    rows += [([], [])] * (ROWS - used)
    acc = [col[next(iter(w.s))] for w in C.inputs["c"]]
    ops = [col[next(iter(w.s))] for name, bits in C.inputs.items() if name != "c" for w in bits]
    words = [MAGIC, ROWS, used]
    for a, b in rows:
        words += [len(a), *a, len(b), *b]
    words += acc + ops + outs
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(words)}I", *words))
    nnz = sum(len(a) + len(b) for a, b in rows)
    return {"rows": ROWS, "used": used, "nnz_AB": nnz, "acc_inputs": len(acc), "operand_inputs": len(ops), "outputs": len(outs),
            "output_copy_rows": sum(1 for w in C.outputs["o"] if not (len(w.s) == 1 and w.c == 0))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args(argv)
    print(json.dumps(export(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
