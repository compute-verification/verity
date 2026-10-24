"""A template's rows from call sites that name its output differently (`pipeline.serving_rows.read`): GLM-5.3's SiluMul_v2{I=256}
rows of one step are its routed experts', which commit their output as ``act``, and its shared experts', which commit it as ``out``.
Each row's outputs fill the member's ports in order, every row's value is read under its own member name, and its index entry keeps
that name; a row whose widths are not the member's is refused by name."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.pipeline import serving_rows as SRW

I = 4
T = f"SiluMul_v2{{I={I}}}"


def row(i: int, member: str, op: str, width: int = I) -> SRW.Row:
    r = {"i": i, "spec": T, "args": [[["n", i, 0, 2 * I]]]}
    d = {"statics": {"I": I}, "op_path": op, "members": [(member, width, "bf16")], "row": 0, "step": 16}
    return (f"r{i}", None, r, d)


@pytest.fixture
def words(monkeypatch):
    rng = np.random.default_rng(3)
    gu = {i: rng.integers(0, 1 << 16, 2 * I, dtype=np.uint16) for i in range(3)}
    out = {(f"r{i}", m): rng.integers(0, 1 << 16, I, dtype=np.uint16) for i, m in ((0, "out"), (1, "act"), (2, "act"))}
    reads = []

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            reads.append((rid, step, op, member))
            v = out.get((rid, member))
            return (None, f"no member {member} at {op}") if v is None else (v[lo:hi], None)

    monkeypatch.setattr(SRW, "_operand", lambda _pi, _store, _w, rid, _arrive, _group: gu[int(rid[1:])])
    return gu, out, reads, Store()


def test_a_template_s_rows_fill_its_ports_in_order_whatever_their_member_names(words):
    gu, out, reads, store = words
    rows = [row(0, "out", "model.layers.3.mlp.shared_experts.act_fn"), row(1, "act", "model.layers.3.mlp.experts"),
            row(2, "act", "model.layers.4.mlp.experts")]
    (m,) = SRW.members([T], [rows], [3])
    assert (m.rows, m.outs) == ((("in0", 2 * I),), (("out.out", I),))
    data, entries = SRW.read(m, rows, store, None)
    assert data["rows"]["in0"].tolist() == [gu[i].tolist() for i in range(3)]
    assert data["outs"]["out.out"].tolist() == [out[("r0", "out")].tolist(), out[("r1", "act")].tolist(), out[("r2", "act")].tolist()]
    assert [x[3] for x in reads] == ["out", "act", "act"]
    assert [e[6] for e in entries] == ["out", "act", "act"]
    assert [e[:2] for e in entries] == [[0, 1], [1, 2], [2, 3]]


def test_a_row_whose_output_widths_are_not_the_member_s_is_refused(words):
    _gu, _out, _reads, store = words
    rows = [row(0, "out", "model.layers.3.mlp.shared_experts.act_fn"), row(1, "act", "model.layers.3.mlp.experts", width=I // 2)]
    (m,) = SRW.members([T], [rows], [2])
    with pytest.raises(ValueError, match=r"row 1 \(model.layers.3.mlp.experts\): inputs \[8\] and outputs \[2\] words"):
        SRW.read(m, rows, store, None)
