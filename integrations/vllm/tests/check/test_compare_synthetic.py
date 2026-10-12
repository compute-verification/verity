"""compare.py on the synthetic toy program: snapshots are *derived* from a full reference evaluation of the folded
Program, so provenance (comparison 2) and expanded subcircuits (comparison 3) must be bit-exact by construction,
and a corrupted snapshot must be detected.  Exercises the snapshot store, the provenance value resolver, FunctionCall
location, evaluate_call argument order and the tokens / structure reports without any capture artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import FunctionCall, Region
from verity.ml.circuits.refs import runs

from verity_vllm.check import fold_compare as cmp
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.fold import fold

from tests.observe.test_fold_synthetic import BN, C, REQ, SAMPLED, _real_memory_available, install_stub_memory, toy_log, toy_profile

_NP = {"bfloat16": np.uint16, "int32": np.uint32, "int64": np.uint64}


def _bf16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


class ToyWeights:
    """Deterministic pseudo-random bf16 words for the weights Record (stands in for the checkpoint)."""

    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(7)
        self.w = _bf16(rng.uniform(-0.5, 0.5, size=n))

    def available(self) -> bool:
        return True

    def words(self, lo: int, hi: int) -> np.ndarray:
        return self.w[lo:hi]


def _evaluate_program(result, weights: ToyWeights) -> dict[int, int]:
    """Reference evaluation of the whole folded Program with the prompt and the toy weights prescribed."""
    prog = result.program
    circuit = prog.circuit
    t: dict[int, int] = {}
    prompt = result.requests[0]["prompt_token_ids"]
    for i, (_, _, g) in enumerate(prog.input_coll(0).refs):
        t[g] = int(prompt[i]) & 0xFFFFFFFF
    W = len(result.params) - 1
    for i, (_, _, g) in enumerate(prog.input_coll(W).refs):
        t[g] = int(weights.w[i])
    evaluate_region(circuit, Region(circuit, 0, circuit.size), t)
    return t


def _write_snapshots(result, t: dict[int, int], root: Path, corrupt_node: int | None = None) -> None:
    """Snapshot every instance's outputs (after the event) and operand views (as its own event's inputs), from the
    transcript.  Files hold raw element bytes in the row-major logical order of the view (DESIGN §3)."""
    circuit = result.program.circuit
    n_in = len(result.program.root.body.nodes) - len(result.body.nodes)
    entries = []
    (root / "values").mkdir(parents=True, exist_ok=True)

    def put(event, node, arg, view: ViewDesc, words: list[int], step: int) -> None:
        arr = np.array(words, dtype=_NP[view.dtype])
        f = root / "values" / f"{event}_{node}_{arg}.bin"
        arr.tofile(f)
        entries.append({"event": event, "arg": arg, "view": view.to_json(), "file": str(f.relative_to(root)), "step": step,
                        "layout": "logical-row-major"})

    for m in result.instances:
        call = FunctionCall(circuit, circuit.root_scope, n_in + m["node"])
        outs = [t[g] for g in call.returns]
        if corrupt_node is not None and m["node"] == corrupt_node:
            outs = [o ^ 1 for o in outs]
        for k, (clo, chi) in enumerate(cmp._component_ranges({"_spec": result.body.nodes[m["node"]].fn})):
            view = m["output_views"][k]
            if view is not None:
                put(m["event"], m["node"], f"out{k}", view, outs[clo:chi], m["step"])
        # the event's own tensor arguments "after the call" (for in-place instances these would hold the outputs,
        # which is why compare.py only consults snapshots of *earlier* events for them)
        args = [t[g] for g in call.args]
        pos = 0
        node = result.body.nodes[m["node"]]
        for k, a in enumerate(node.args):
            view = m["operand_views"][k]
            n = a.refs.count
            if view is not None and view.numel == n and not cmp._in_place(m):
                put(m["event"], m["node"], f"in{k}", view, args[pos:pos + n], m["step"])
            pos += n
    (root / "snapshots.json").write_text(json.dumps({"layout": "logical-row-major", "entries": entries}))
    (root / "tokens.json").write_text(json.dumps({REQ: [int(x) for x in SAMPLED[:len(result.sampled[REQ])]]}))


@pytest.fixture(scope="module")
def folded():
    if not _real_memory_available():
        install_stub_memory()   # module-wide; the fold tests re-patch per test anyway
    r = fold(toy_log(), toy_profile(), strict=True)
    weights = ToyWeights(r.params[-1][1].leaves)
    t = _evaluate_program(r, weights)
    return r, weights, t


def test_snapshot_store_roundtrip(tmp_path, folded):
    r, weights, t = folded
    _write_snapshots(r, t, tmp_path)
    store = cmp.SnapshotStore(tmp_path)
    assert len(store.entries) > 0
    m = next(m for m in r.instances if m["spec"].startswith("Gemm"))
    view = m["output_views"][0]
    s = store.find(view, at=m["event"])
    assert s is not None and store.covers(s, view)
    words = store.extract(s, view)
    call = FunctionCall(r.program.circuit, r.program.circuit.root_scope, len(r.program.root.body.nodes) - len(r.body.nodes) + m["node"])
    assert [int(w) for w in words] == [t[g] for g in call.returns]
    # a strided sub-view (one head of a [NH, D] row) extracts the right elements
    sub = ViewDesc(view.alloc, view.offset + C["D"], (C["D"],), (1,), view.dtype, view.device)
    assert [int(w) for w in store.extract(s, sub)] == [t[g] for g in call.returns][C["D"]:2 * C["D"]]


def test_provenance_and_subcircuits_bit_exact(tmp_path, folded):
    r, weights, t = folded
    _write_snapshots(r, t, tmp_path)
    store = cmp.SnapshotStore(tmp_path)
    b = cmp.compare_boundary(r, store, weights)
    assert b["status"] == "pass", b
    assert b["summary"].get("fail", 0) == 0 and b["summary"]["pass"] >= 4
    kinds = {c["class"] for c in b["checks"] if c["status"] == "pass"}
    assert any("attention K" in k for k in kinds) and any("decode Embedding" in k for k in kinds), kinds
    s = cmp.compare_subcircuits(r, store, weights)
    assert s["status"] == "pass", s
    specs = {e["spec"].split("{")[0] for e in s["instances"] if e["status"] == "pass"}
    assert {"Gemm_v1", "RMSNormFusedCuda_v2", "RoPE_v1", "SiluMul_v1", "TokenSelect_v1", "Attention_v3"} <= specs, s


def test_corrupted_snapshot_is_detected(tmp_path, folded):
    r, weights, t = folded
    # the RoPE instance whose output the first attention instance reads as its K rows (through the KV move)
    att = next(m for m in r.instances if m["spec"].startswith("Attention"))
    bad = next(idx for space, idx, _, _ in runs(r.body.nodes[att["node"]].args[1].refs) if space == "n")
    assert r.body.nodes[bad].fn.id.startswith("RoPE")
    _write_snapshots(r, t, tmp_path, corrupt_node=bad)
    store = cmp.SnapshotStore(tmp_path)
    s = cmp.compare_subcircuits(r, store, weights, picks=[m for m in r.instances if m["node"] == bad])
    assert s["status"] == "fail" and s["instances"][0]["n_diff"] == C["KVH"] * C["D"]   # every k leaf flipped
    b = cmp.compare_boundary(r, store, weights)
    # the corrupted RoPE k output feeds the KV rows the attention instances read: provenance must flag it
    assert b["status"] == "fail" and any(c["status"] == "fail" for c in b["checks"])


def test_tokens_and_structure_reports(tmp_path, folded):
    r, weights, t = folded
    _write_snapshots(r, t, tmp_path)
    tok = cmp.compare_tokens(r, tmp_path / "tokens.json", None)
    assert tok["status"] == "pass", tok
    assert tok["fold"] == {REQ: [int(x) for x in SAMPLED[:2]]}
    st = cmp.compare_structure(r, C=C, LP=2, STEPS=1, BN=BN)
    assert st["superset"] is True and st["n_extra"] == 1
    assert len(st["classes"]) == 1 and "final norm" in next(iter(st["classes"]))
    h = st["histogram"]
    assert all(v["delta"] == 0 for k, v in h.items() if not k.startswith("RMSNormFusedCuda"))
    assert sum(v["delta"] for v in h.values()) == 1
