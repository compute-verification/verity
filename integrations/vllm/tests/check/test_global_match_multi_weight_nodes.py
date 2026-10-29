"""F-fp8-GM1: GM-01 / X-09 over a fold whose weights Record is NOT width-homogeneous.

`verity.ml.circuits.program._lower_input` lowers a homogeneous-width Input type to ONE `Input<w>` batch node and recurses into a
Record/Array/Tuple whose leaves have several widths -- one node per homogeneous field.  The block-scaled FP8 checkpoint
(Qwen3-4B-FP8: e4m3 weights, f32 `weight_scale_inv` 128x128 block scales, bf16 norms) therefore lowers its weights Record to
435 Input nodes, while every bf16 model lowers it to a single node = `n_input_nodes - 1`.  `batch_decomp.decompose`,
`batch_decomp.Projection` and `global_match.check` hard-coded that single node: on the FP8 B1 record GM-01 raised
`program has 436 input nodes but the header has 1 requests + 0 seeds (+1 weights expected)` on the X-09 path and, on the
global path, counted every weight read as a cross-request 'prompt-param' read (2,074,230) and gave 0 canonical addresses.

CPU reproduction on the GM-01 toy (`test_global_match`): the toy's weights Record gets one extra f32 field APPENDED
(`analytic.weights_type` monkeypatched for the fold), so the fold lowers it to 5 weight Input nodes (embed, layers, norm,
cos_sin, scale32) -- the last node is one no instance reads, exactly the FP8 shape of the defect (n_in - 1 is not the node
the GEMMs read).  Both the batched record and the B=1 components are folded under the same Record, as on the pod.
"""

from __future__ import annotations

import pytest

from verity_vllm.program import model as analytic
from verity_vllm.observe.fold import fold as fold_mod
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import global_match as gm
from tests.check import test_global_match as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401 - fixture
from verity.ml.circuits.types import Array, Record, Value

SCALE_FIELD = ("scale32", Array(4, Value(32)))


@pytest.fixture
def inhomogeneous_weights(monkeypatch):
    """The toy weights Record + one f32 field: width set {16, 32} -> one Input node per homogeneous field (5 weight nodes)."""
    base = analytic.weights_type

    def wt(C):
        return Record(*base(C).fields, SCALE_FIELD)

    monkeypatch.setattr(fold_mod.analytic, "weights_type", wt)
    return wt


def _weight_nodes(oracle_dir: str) -> tuple[C.Prog, list[int]]:
    B = C.load_oracle(oracle_dir, "oracle")
    return B, sorted(B.inputs)


def test_fixture_lowers_the_record_to_several_weight_input_nodes(memory, inhomogeneous_weights, tmp_path):
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    B, nodes = _weight_nodes(oracle)
    # 2 prompts + 5 weight fields (embed, layers, norm, cos_sin, scale32): the FP8 layout in miniature
    assert B.n_input_nodes == 7 and nodes == list(range(7))
    assert sorted(B.input_base) == [2, 3, 4, 5, 6]          # load_oracle: weight nodes = every node >= the weights parameter index
    assert B.inputs[6] == ("Input32_v1", 4)                 # the appended f32 field is the LAST node -- no instance reads it
    read = {run[1] for r in B.inst for a in r["args"] for run in a if run[0] == "n" and run[1] in B.inputs}   # runs | progressions
    assert 6 not in read and {2, 3, 4, 5} <= read           # the GEMMs/norms/rope read nodes 2..5, never n_in - 1


def test_x09_decompose_attributes_every_weight_read_as_shared_not_cross_request(memory, inhomogeneous_weights, tmp_path, monkeypatch):
    monkeypatch.setattr(bd.C, "load_derived", lambda p, label: C.load_oracle(p, label))   # the toy has no derive_step (e2e fixture)
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    comps = [g["component"]["dir"] for g in gp["requests"]]
    res = bd.decompose(oracle, comps)
    assert res["verdict"] == "PASS", res["reasons"]
    assert res["oracle"]["n_input_nodes"] == 7
    assert res["oracle"]["weights_input_nodes"] == {"n": 5, "first": 2, "last": 6, "request_owned_inputs": 2}
    assert res["attribution"]["unattributed"] == 0
    for r in res["requests"]:
        assert r.get("error") is None, r
        assert r["xreq"]["counts"] == {}, r["xreq"]         # 2,074,230 'prompt-param' reads here before the fix
        assert r["canonical_equal"], r
        assert r["totals"]["operands_different"] == 0 and r["totals"]["operands_equal"] > 0


def test_gm01_passes_on_the_multi_node_weights_record_and_names_every_weight_node(memory, inhomogeneous_weights, tmp_path):
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    g5 = T._check(res, "G5")
    assert g5["weights_input_nodes"] == {"n": 5, "first": 2, "last": 6} and g5["weights_param"] == 2
    assert g5["weight_fields"] >= 4                         # embed, ln1, qkv_w, o_w, ln2, gate_up_w, down_w, norm, cos_sin fields spanned
    g4 = T._check(res, "G4")
    assert g4["mapped"] == g4["attributed"] > 0            # every attributed instance has a canonical address (0 before the fix)
    for e in res["requests"]:
        assert e["operands"]["different"] == 0 and e["executed_steps_equal"], e


def test_gm01_multi_node_weights_keep_the_mutations_failing(memory, inhomogeneous_weights, tmp_path):
    # the widened weights-node set must not make G5 vacuous: a cross-request KV read is still a shared-state failure
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path, leak=True)
    res = gm.check(gp, oracle)
    assert res["verdict"] == "FAIL"
    assert T._check(res, "G5")["status"] == "FAIL"


def test_header_and_program_input_layout_disagreement_is_named(memory, inhomogeneous_weights, tmp_path):
    # the fold's layout is prompts, seeds, then weights: a header claiming MORE requests than the Program has prompt nodes is refused by name
    # (X-09 raises; GM-01 fails G5 by name and every weight read becomes a cross-request read -- never a guessed boundary)
    import gzip
    import json
    gp, oracle = T.assemble([T.R0, T.R1], tmp_path)
    comps = [g["component"]["dir"] for g in gp["requests"]]
    log = f"{oracle}/capture/log.jsonl.gz"
    lines = gzip.open(log, "rt").read().splitlines()
    hdr = json.loads(lines[0])
    hdr["workload"]["requests"].append({"request_id": "r9", "prompt_token_ids": [1, 2], "max_tokens": 2, "arrive_step": 0})
    lines[0] = json.dumps(hdr)
    bad = tmp_path / "bad.jsonl.gz"
    with gzip.open(bad, "wt") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="header and Program disagree on the input layout"):
        bd.decompose(oracle, comps, log_path=str(bad))
    res = gm.check(gp, oracle, log_path=str(bad))
    assert res["verdict"] == "FAIL"
    g5 = T._check(res, "G5")
    assert g5["status"] == "FAIL" and any("header and Program disagree on the input layout" in p for p in g5["problems"])
    # ... and one claiming so many that no weights node is left
    hdr["workload"]["requests"] += [dict(hdr["workload"]["requests"][-1], request_id=f"r{k}") for k in range(10, 15)]
    lines[0] = json.dumps(hdr)
    with gzip.open(bad, "wt") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=">= 1 weights node expected"):
        bd.decompose(oracle, comps, log_path=str(bad))
    res = gm.check(gp, oracle, log_path=str(bad))
    assert res["verdict"] == "FAIL"
    g5 = T._check(res, "G5")
    assert g5["status"] == "FAIL" and any(">= 1 weights node expected" in p for p in g5["problems"])
