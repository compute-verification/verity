"""A fused unit's words from its operands alone (`expert_rows.fused_unit`), on `test_serving_rows_experts`' layer at E = 8: each
fused unit's words are the words `serving_rows.read_all` reads for it over the request's Program, its output is the twin of the
pair row on the unit's own inputs (its table's row at its key and its G-block's scale row), and an operand that does not fit the
template is refused."""
from __future__ import annotations

import json

import numpy as np
import pytest
from verity.definitions.numerics import fp8_moe_rows as FR

from tests.pipeline import test_serving_rows_experts as SX
from verity_vllm.pipeline import expert_rows as EX
from verity_vllm.pipeline import serving_rows as SRW


def units(monkeypatch, proj: str, n: int, nr: int):
    """Projection ``proj``'s fused window at n rows in blocks of nr: its template, the layer, its committed tables, and each fused
    unit's ``(pair, block, words read_all gives it)``."""
    tmpls, caps = SX.fused(proj, n, nr)
    ns, t = SX.build(monkeypatch, proj, n=n), tmpls[-1]
    rows = SX.rows_of(tmpls, ns)
    plan, ms = SX.cut(rows, tmpls, caps)
    tables = EX.commit_tables(plan, ns.store, None, None, SRW._slices, SX.committer(plan))
    base, out = min(m.base for m in ms if m.template == t), []
    for m, (g, _e) in zip(ms, SRW.read_all(ms, rows, ns.store, None, tables=tables)):
        if m.template == t:
            for i in range(m.n):
                p, b = plan.units[t][m.base - base + i]
                out.append((p, b, {q: a[i] for part in ("rows", "outs") for q, a in g[part].items()}))
    return t, ns, tables, out


def operands(ns, proj: str, p: int, b: int, nr: int) -> dict:
    """Pair ``p``'s block ``b``'s operands as `fused_unit` takes them, from the layer's operands and served words."""
    args, nb = ns.pairs[p][2]["args"], -(-ns.n // SX.G)
    e = ns.experts[p]
    return {"e": e, "xq": ns.ops[json.dumps(args[0])], "sx": ns.ops[json.dumps(args[1])],
            "g": int(ns.ops[json.dumps(args[3])][0]) if SX.PROJ[proj].routed else None,
            "ws": ns.ws.reshape(SX.E, nb, SX.KG)[e][b * nr // SX.G], "y": ns.out.reshape(-1, ns.n)[p][b * nr:(b + 1) * nr]}


@pytest.mark.parametrize("proj,n,nr", [("w13", SX.N, SX.NR), ("w2", SX.N, SX.NR), ("w2", 64, 16)])
def test_a_fused_units_words_from_its_operands_are_read_alls_and_its_output_the_twins(monkeypatch, proj, n, nr):
    t, ns, tables, got = units(monkeypatch, proj, n, nr)
    pr, fu = SX.PROJ[proj], SX.ports(t)
    assert len(got) == SX.P * (n // nr)
    for p, b, read in got:
        mine = EX.fused_unit(t, tables[EX.table_param(pr.w, nr * SX.K // 2, b)], **operands(ns, proj, p, b, nr))
        assert sorted(mine) == sorted(read) and all(np.array_equal(mine[q], read[q]) for q in read), (p, b)
        g = mine[fu["g"]].view("<u4")[0] if pr.routed else None
        twin, nonfinite = FR.moe_expert_fp8_block_row(mine[fu["xq"]].view(np.uint8), mine[fu["sx"]].view("<u4"),
                                                      mine[fu["row"]].view(np.uint8).reshape(nr, SX.K), mine[fu["ws"]].view("<u4"),
                                                      G=SX.G, KB=SX.KB, swap=pr.swap, g=g, dot=pr.twin)
        assert not nonfinite.any() and np.array_equal(twin, mine["o0"]) and mine["o1"].tolist() == [1]


def test_an_operand_that_does_not_fit_the_template_is_refused(monkeypatch):
    t, ns, tables, got = units(monkeypatch, "w2", SX.N, SX.NR)
    p, b, _read = got[0]
    table, ok = tables[EX.table_param(SX.PROJ["w2"].w, SX.NR * SX.K // 2, b)], operands(ns, "w2", p, b, SX.NR)
    unweighted = SX.PROJ["w13"].fused
    for bad, why in [({"g": None}, "weighted fused block given no routed weight"), ({"e": SX.E}, f"expert {SX.E} of a table"),
                     ({"e": -1}, "expert -1 of a table"), ({"xq": ok["xq"][1:]}, "its xq is"),
                     ({"sx": np.append(ok["sx"], 0)}, "its sx is"), ({"ws": ok["ws"][:1]}, "its ws is"),
                     ({"y": ok["y"][:1]}, "its y is")]:
        with pytest.raises(ValueError, match=why):
            EX.fused_unit(t, table, **{**ok, **bad})
    with pytest.raises(ValueError, match="unweighted fused block given a routed weight"):
        EX.fused_unit(unweighted, table, **ok)
    with pytest.raises(ValueError, match="its row is"):
        EX.fused_unit(SX.ids("w2", SX.N, 1).fused, table, **{**ok, "y": ok["y"][:1]})
    with pytest.raises(ValueError, match="not a fused block"):
        EX.fused_unit(SX.BLOCK, table, **ok)
