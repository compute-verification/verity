"""A served window's registered reads of its weights (`pipeline.weight_rows.reads`): a slice unit's port that reads one whole
row of one weight (`Gemm_v2{N=1}`'s ``w[n]``) as ``row/v2`` bit rows of its width is a read of ``weights/<label>@<words>`` at the
weight's own leaves, in runs over consecutive units; the circuit record declares the reads beside its listed members and its
window ingress, and the window's members are cut where the reads begin and end (`weight_rows.cut`), so each covers whole
members.  The leak count holds them free until an audit opens them against the pinned roots (`interface.input_count`)."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.pipeline import serving_rows as SRW, weight_rows as WT
from verity_vllm.program.predict import window_weights as WW

from verity.core.protocols.sampled_proofs.one_stage import registered as REG, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS

ST = {"K": 32, "N": 3, "DOT": HopperBF16WgmmaDot16}
ROW, UNIT = bind(G.Gemm, **ST), bind(G.Gemm, **{**ST, "N": 1})
T = descriptor_id(UNIT)
#: the weights struct: two layers' gate weights [N, K] and a fused [2N, K] one whose second half a row reads
FIELDS = [("model.layers.0.mlp.gate.weight", 96, [3, 32], 16), ("model.layers.1.mlp.gate.weight", 96, [3, 32], 16),
          ("model.layers.2.mlp.gate_up.weight", 192, [6, 32], 16)]
#: rows 10 .. 12 of each request, each reading row 3's committed activations and its weight's run of the struct
WEIGHT = {10: (0, 96), 11: (96, 192), 12: (288, 384)}
PARTITION = {"templates": [T], "law": "subset:1", "chunk_cap": {T: 18}}


class Index:
    pnames = ["weights"]
    fields = FIELDS
    foff = [0, 96, 192]
    weight_slice, weight_bits = ProgramIndex.weight_slice, ProgramIndex.weight_bits

    def __init__(self, *_a, **_k):
        self.by_i = {i: {"i": i, "spec": descriptor_id(ROW), "args": [[["n", 3, 0, 32]], [["p", 0, lo, hi]]]}
                     for i, (lo, hi) in WEIGHT.items()}

    def describe(self, r):
        return {"statics": {"K": 32, "N": 3}, "op_path": f"model.layers.{r['i'] - 10}.mlp.gate", "row": 0, "step": 0,
                "members": [("out", 3, "bf16")]}


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(SRW, "index_of", lambda inst, _pool, **kw: Index(inst, **kw))

    def window(partition):
        w, rows = SRW.population({"r0": {}, "r1": {}}, partition, "ab" * 64)
        return w, rows, json.loads(IF.circuit_bytes(SRW.circuit_record(w, rows)))
    return window


def test_a_slice_unit_reading_a_row_of_a_weight_is_a_registered_read_of_its_rows(served):
    """Each request's three rows are 3 units each, unit n reading ``w[n]``, 32 words: layer 0's weight from leaf 0, layer 1's,
    and the fused weight's second half from leaf 96, each a read of its own parameter with step 32.  The members are cut at the
    reads' ends, the record lists them, its window ingress is ``in0`` alone, and its registered reads count free."""
    window, rows, doc = served(PARTITION)
    p0, p1, p2 = (f"weights/{f}@32" for f, *_ in FIELDS)
    assert doc["registered"]["params"] == {p0: 96, p1: 96, p2: 192}
    assert [(r["param"], r["at"], r["radix"], r["base"], r["step"]) for r in doc["registered"]["reads"]] == [
        (p0, 0, [3], 0, [32]), (p1, 3, [3], 0, [32]), (p2, 6, [3], 96, [32]),
        (p0, 9, [3], 0, [32]), (p1, 12, [3], 0, [32]), (p2, 15, [3], 96, [32])]
    assert {(r["reader"], r["port"], tuple(r.get("cut", (0, 1)))) for r in doc["registered"]["reads"]} == {(T, "in1", (0, 1))}
    assert not any(doc["registered"][k] for k in ("tables", "roots", "openings"))
    assert [(m.key, m.base, m.n) for m in window.members] == [(f"{T}/c{j}", 3 * j, 3) for j in range(6)]
    assert {g["port"] for g in doc[IF.WINDOW_INGRESS]} == {"in0"}
    count = IF.input_count(doc)
    assert count["units"] == 18 and count["registered"] == 0 and count["free"] == count["inputs"] == 18 * 2 * 32 * 16
    assert IF.window_of_record(doc, window.law, {T: UNIT}.__getitem__).members == window.members
    assert WT.reads(window.members, rows, SRW.row_units).doc() == doc["registered"]


def test_chunks_are_cut_again_at_the_reads_ends(served):
    """Chunked at 4 units, the template's chunks are cut at the reads' ends too: each read still covers whole members."""
    window, _rows, doc = served({**PARTITION, "chunk_cap": {T: 4}})
    assert [(m.base, m.n) for m in window.members] == [(0, 3), (3, 1), (4, 2), (6, 2), (8, 1), (9, 3), (12, 3), (15, 1), (16, 2)]
    assert doc["registered"] == served(PARTITION)[2]["registered"]


def values_and_rows(window, doc, key=bytes(range(32))):
    """The committer's weights committed under ``key`` (`window_weights.weight_value`, rows held), and each member's rows as
    `serving_rows.read_all` reads them: ``in0`` the activations, ``in1`` the weight row each unit reads."""
    rng, reg = np.random.default_rng(5), IF.listed_registered(doc)
    words = {f: rng.integers(0, 1 << 16, c, dtype=np.uint16) for f, c, _s, _b in FIELDS}
    values = {p: WW.weight_value("cd" * 64, p, words[p.split("/", 1)[1].rpartition("@")[0]], key, WW.WEIGHTS_NONCE + i,
                                 keep_rows=True) for i, p in enumerate(reg.params)}
    data = []
    for m in window.members:
        (r,) = [r for r in reg.reads if r.at <= m.base < r.at + r.reader_units]
        at = r.source(np.arange(m.base - r.at, m.base - r.at + m.n)) // 32
        data.append({"rows": {"in0": rng.integers(0, 1 << 16, (m.n, 32), dtype=np.uint16), "in1": CS.value_rows(values[r.param], at)},
                     "outs": {p: np.zeros((m.n, w), dtype=np.uint16) for p, w in m.outs}})
    return values, data


def test_the_committer_commits_each_read_by_reference_as_the_lean_verifier_opens_it(served, tmp_path):
    """Given the committer's weights, each read port's rows are its value's rows, its units' positions the ones the Lean
    verifier's ``--registered`` derives from the record alone (`interface.registered_reads`, under the pins of the values'
    entries); the window writes them as its members' ``registered`` and ``value_rows`` and records each value's root."""
    window, _rows, doc = served(PARTITION)
    values, data = values_and_rows(window, doc)
    got, entries, roots = WT.by_reference(window, IF.listed_registered(doc), data, values)
    assert roots == {p: v.root.hex() for p, v in values.items()}
    assert {k: e["in1"]["positions"] for k, e in entries.items()} == {f"{T}/c{j}": [3 * (j % 3 == 2) + i for i in range(3)]
                                                                       for j in range(6)}
    assert all(isinstance(d["rows"]["in1"], CS.ValueRows) and isinstance(d["rows"]["in0"], np.ndarray) for d in got)
    lean = IF.registered_reads(doc, IF.window_of_record(doc, window.law, {T: UNIT}.__getitem__),
                               {p: v.entry() for p, v in values.items()}, registered=REG)
    assert {k: (e["ports"]["in1"]["reads"], e["ports"]["in1"]["root"]) for k, e in lean.items()} == {
        k: (e["in1"]["positions"], roots[e["in1"]["value"]]) for k, e in entries.items()}
    res = CS.write(CS.commit(window, got, salt_key=bytes(32)), tmp_path, {"registered_roots": roots}, {}, registered=entries,
                   registrar=R)
    assert [m["registered"]["in1"] for m in res["members"]] == [{"value": e["in1"]["value"], "positions": e["in1"]["positions"]}
                                                                 for e in entries.values()]
    assert not any("value_rows" in m for m in res["members"]) and set(res["registered_paths"]) == set(roots)
    assert IF.registered_roots(json.loads((tmp_path / "registration.json").read_text())["window"]) == roots


def test_a_read_whose_rows_are_not_its_values_is_refused_and_no_values_commit_rows_in_full(served):
    window, _rows, doc = served(PARTITION)
    values, data = values_and_rows(window, doc)
    reg = IF.listed_registered(doc)
    assert WT.by_reference(window, reg, data, None) == (data, {}, {})
    other, _ = values_and_rows(window, doc, key=bytes(32))
    data[1]["rows"]["in1"] = data[1]["rows"]["in1"] ^ np.uint16(1)
    with pytest.raises(ValueError, match=re.escape(f"{T}/c1 in1: the rows its units read aren't 'weights/model.layers.1.mlp.gate.weight@32'")):
        WT.by_reference(window, reg, data, values)
    with pytest.raises(ValueError, match="give its committed value, or no values at all"):
        WT.by_reference(window, reg, data, {p: v for p, v in other.items() if "layers.0" in p})


def test_a_read_that_would_cut_a_wired_member_is_refused():
    m = CS.Member(T, (("in0", 32), ("in1", 32)), (("o0", 1),), 6, wired=(("in0", "o0", tuple(range(6))),))
    with pytest.raises(ValueError, match="a weight read covers part of a member a wire, a shared-row table or a row table fills"):
        WT.cut([m], IF.Registered({"weights/w@32": 96}, reads=(IF.Read(T, "in1", "weights/w@32", at=0, radix=(3,), base=0,
                                                                        step=(32,)),)))
    assert WT.cut([m], None) == [m]
