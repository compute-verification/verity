"""A ``PARTS`` row's parts as chunked members (`serving_rows`' ``part`` layout, and a ``COLS`` part's ``cols``): a part that reads
its row's committed output j (``("out", j)`` in its ``PARTS`` entry) reads the output the part returning j wrote, so the window
wires that port to the writer's output row on the same row (`wire`, `commit.wiring`): `tp_allreduce.AllReduceRmsNormMnnvl`'s
norm reads the r its residual part returned.  A wire is one writer unit per reader unit, so the writer is its part at the row's
width, one unit a row."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline.serving_templates import COLS, part, template

Row = tuple[str, Any, dict, dict]


def wire(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]]) -> list[CS.Member]:
    """``ms`` with each part member's port that reads its row's committed output j wired to the ``o0`` row of the row's unit of
    the part returning j; refused when the partition names no such part, when that part returns more than j or is in column
    blocks (more than one unit a row), or when the two read other rows."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    parts: dict[str, list[tuple[str, list[Row], tuple]]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        p = part(template(t).word, rs[0][2]["spec"]) if rs else None
        if p is not None:
            parts.setdefault(spec_base(rs[0][2]["spec"]), []).append((t, rs, p))
    base = {t: min(m.base for m in ms if m.template == t) for t in tmpls}
    writers: dict[str, tuple[int, tuple[int, ...]]] = {}
    for fam, found in parts.items():
        for t, rs, (pfam, reads, _returns) in found:
            for g, (src, j) in enumerate(reads):
                if src != "out":
                    continue
                hit = [(wt, wrs, wp) for wt, wrs, wp in found if j in wp[2]]
                if not hit:
                    raise ValueError(f"{t} reads its {fam} rows' committed output {j}, which the partition's parts of {fam} "
                                     f"({[x[0] for x in found]}) do not return: name the part that returns it")
                wt, wrs, (wfam, _wr, wret) = hit[0]
                if wret != (j,):
                    raise ValueError(f"{t} reads its {fam} rows' output {j}, and {wt} returns {list(wret)}, not that output alone")
                if [(rid, int(r["i"])) for rid, _pi, r, _d in wrs] != [(rid, int(r["i"])) for rid, _pi, r, _d in rs]:
                    raise ValueError(f"{t} and {wt}, the part whose output it reads, read other {fam} rows")
                axis = COLS.get(wfam)
                if axis is not None and any(int(d["statics"][axis]) != int(spec_statics(template(wt).word)[axis]) for *_x, d in wrs):
                    raise ValueError(f"{t} reads its {fam} rows' output {j} whole, which {wt} returns in column blocks: wire it to "
                                     f"{wfam} at the rows' {axis}, one unit a row")
                writers[t] = (g, tuple(range(base[wt], base[wt] + len(rs))))
    out = []
    for m in ms:
        if m.template in writers:
            g, ws = writers[m.template]
            lo = m.base - base[m.template]
            m = replace(m, wired=(f"in{g}", "o0", ws[lo:lo + m.n]))
        out.append(m)
    return out
