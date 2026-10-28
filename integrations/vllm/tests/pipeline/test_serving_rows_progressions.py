"""A row's argument group stored as a progression (`program.compact`'s ``["N"|"P", idx, lo, hi, dI, dL, dH, count]``, a v2
Program's form) is read as its runs, as the replay reads it (`check.replay.index.row_args`): GLM-5.3's MoeSum rows store their
TOPK expert outputs as one node progression, so their column blocks read the same words as the same row with its runs listed,
and a row template's input port counts every run of the progression."""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity_flock.class_statement import UnitStager
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.ml.definitions.registry import boolean_moe as BMOE
from verity.ml.definitions.registry import moe as M

TOPK, W, w, P0 = 3, 8, 2, 100
OP = "model.layers.3.mlp.experts"


class Index:
    """`member_at` as a rank Program's index answers it: producer node a's committed member ``out``, at op_path ``p<a>``."""

    def member_at(self, a, lo, hi):
        return {"members": [["out", W, "bf16"]], "step": 0, "op_path": f"p{a}", "row": 0}, "out", lo, hi, "bf16"


class Store:
    def __init__(self, vals: dict[str, np.ndarray]):
        self.vals = vals

    def words(self, rid, step, op, member, lo, hi):
        return self.vals[op][lo:hi], None


def experts(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {f"p{P0 + k}": (rng.standard_normal(W).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16) for k in range(n)}


def test_moe_sum_s_expert_outputs_as_one_progression_read_as_their_runs():
    vals = experts(TOPK)
    y = np.concatenate([vals[f"p{P0 + k}"] for k in range(TOPK)])
    vals[OP] = np.asarray(evaluate_batch(bind(M.MoeSum, TOPK=TOPK, H=W), y.astype(np.uint64).reshape(1, -1))).reshape(-1).astype(np.uint16)
    d = {"statics": {"TOPK": TOPK, "H": W}, "op_path": OP, "members": [("out", W, "bf16")], "row": 0, "step": 16}
    spec = descriptor_id(bind(M.MoeSum, TOPK=TOPK, H=W))
    stored = ("r0", Index(), {"i": 7, "spec": spec, "args": [[["N", P0, 0, W, 1, 0, 0, TOPK]]]}, d)
    runs = ("r0", Index(), {"i": 7, "spec": spec, "args": [[["n", P0 + k, 0, W] for k in range(TOPK)]]}, d)
    T = descriptor_id(bind(BMOE.MoeSumV3, TOPK=TOPK, H=w))
    ms = SRW.members([T], [[stored]], [W // w], {T: 3})
    got = SRW.read_all(ms, [[stored]], Store(vals), None)
    want = SRW.read_all(SRW.members([T], [[runs]], [W // w], {T: 3}), [[runs]], Store(vals), None)
    assert [g[1] for g in got] == [g[1] for g in want]
    for (g, _e), (h, _f) in zip(got, want, strict=True):
        assert all(np.array_equal(g["rows"][p], h["rows"][p]) for p in h["rows"]) and np.array_equal(g["outs"]["o0"], h["outs"]["o0"])
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(T).fn, "progressions-test", {}, CR.unit_rows(SRW.template(T))[0]
    for g, _entries in got:
        st.reference_check(g["rows"], g["outs"])


def test_a_row_template_s_port_counts_every_run_of_a_progression():
    I = 4
    vals = experts(2)
    vals["model.layers.3.mlp.act_fn"] = np.arange(I, dtype=np.uint16)
    r = {"i": 9, "spec": f"SiluMul_v2{{I={I}}}", "args": [[["N", P0, 0, I, 1, 0, 0, 2]]]}
    row = ("r0", Index(), r, {"statics": {"I": I}, "op_path": "model.layers.3.mlp.act_fn", "members": [("out", I, "bf16")],
                              "row": 0, "step": 16})
    (m,) = SRW.members([r["spec"]], [[row]], [1])
    assert (m.rows, m.outs) == ((("in0", 2 * I),), (("out.out", I),))
    data, _entries = SRW.read(m, [row], Store(vals), None)
    assert data["rows"]["in0"].reshape(-1).tolist() == [*vals[f"p{P0}"][:I].tolist(), *vals[f"p{P0 + 1}"][:I].tolist()]
