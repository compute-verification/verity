"""Export the P1 step template (``step_template.build_step``) as one 2^13-row slot of Flock's block R1CS ``A.z * B.z = z``.

The slot is one copy of the soundness package's ``Rows`` (``Security/Proofs/Flock/Soundness/ExecRows.lean``)
in its column order, so a block of these slots is ``Rows.stack`` placed by ``stackPlace``: the 704 inputs (accumulator word,
16 + 16 decoded operands) at 0..703, then the computed rows (the live ANDs, then one copy row per output bit that is not already
a single committed bit), then the copy's constant at ``nIn + nComp = used - 1``. A computed row reads only earlier columns or
the constant (``Rows.topo``). The constant's row is empty here: the block wires it to the pin (``placement_stack``'s
``hconstA``/``hconstB``). An input's row is wiring, not one of ``Rows``' rows: B reads the constant, and the block adds its
source on A. Rows past ``used`` are empty (z = 0).

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
    inputs = [v for bits in C.inputs.values() for (v,) in (w.s for w in bits)]
    for v in inputs:
        col[v] = len(col)
    ands = [v for v in sorted(live) if C.kind[v] == "and"]
    assert all(C.kind[v] in ("and", "in") for v in live)
    for v in ands:
        col[v] = len(col)
    copies = [w for w in C.outputs["o"] if not (len(w.s) == 1 and w.c == 0)]
    konst = len(inputs) + len(ands) + len(copies)

    def cols(form):
        out = sorted(col[v] for v in form.s)
        return out + [konst] if form.c else out

    rows = [([], [konst]) for _ in inputs] + [(cols(a), cols(b)) for a, b in (C.data[v] for v in ands)]
    outs = []
    for w in C.outputs["o"]:
        if len(w.s) == 1 and w.c == 0:
            outs.append(col[next(iter(w.s))])
        else:
            outs.append(len(rows))
            rows.append((cols(w), [konst]))
    assert len(rows) == konst
    rows.append(([], []))
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
    return {"rows": ROWS, "used": used, "n_in": len(inputs), "n_comp": len(ands) + len(copies), "const_col": konst,
            "nnz_AB": nnz, "acc_inputs": len(acc), "operand_inputs": len(ops), "outputs": len(outs), "output_copy_rows": len(copies)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args(argv)
    print(json.dumps(export(a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
