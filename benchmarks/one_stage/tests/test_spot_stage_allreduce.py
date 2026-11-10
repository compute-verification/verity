"""spot_proved's staging of a step's plain allreduces (`stage.allreduce_instances`) on a fake 8-rank capture, committed as
`capture.py` commits one: symm-mem's multimem calls are staged per (token, half-row) with every rank's input and the received
row every rank must agree on; a call on NCCL's route (no recorded multimem op inside it) is counted as not staged and the
step's later calls are still staged; a multimem call no `tp_allreduce.CAPTURED` entry covers is named; the node's tree the
capture recorded beside each rank's step is the one bound, and ranks that disagree on it are refused."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from verity.definitions.registry import tp_allreduce as AR
from verity.definitions.registry import tp_allreduce_check as K

ROOT = Path(__file__).resolve().parents[3]
W, LAYER, M = 8, 3, 43


def _stage():
    spec = importlib.util.spec_from_file_location("spot_stage", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref(d: Path, r: int, a: np.ndarray) -> dict:
    raw = np.ascontiguousarray(a).tobytes()
    h = hashlib.sha512(raw).hexdigest()
    p = d / f"rank{r}" / "blobs" / h[:2] / h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return {"sha512": h, "dtype": "torch.bfloat16", "shape": list(a.shape)}


def fake_capture(d: Path, calls: list[tuple[int, str]], flip: tuple[int, int] | None = None) -> list[int]:
    """Step 0 of a capture of ``calls``, each (H, route) an M-row bf16 allreduce in layer `LAYER`: route "symm_mem" records
    torch's multimem op inside it and receives `multimem_rows` at the H200 tree, "nccl" records nothing inside it and
    receives a chain sum. ``flip``: (rank, call) whose received first word is changed. Returns each call's seq."""
    rng = np.random.default_rng(7)
    tree = [AR.GLM53_H200_TREE] * W
    events: dict[int, list[dict]] = {r: [] for r in range(W)}
    seqs = []
    for i, (H, route) in enumerate(calls):
        f = (rng.standard_normal((W, M * H)).astype(np.float32) * np.float32(0.25)).view(np.uint32)
        parts = ((f + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)
        out = K.multimem_rows(parts, tree, False) if route == "symm_mem" else K.multimem_rows(parts, [AR.chain(W)] * W, True)
        seq = len(events[0])
        seqs.append(seq)
        inside, parent = ("symm_mem::multimem_all_reduce_", seq) if route == "symm_mem" else ("aten::add", None)
        for r in range(W):
            got = out.copy()
            if flip == (r, i):
                got[0] ^= np.uint16(1)
            events[r].append({"seq": seq, "layer": LAYER, "parent": None, "op": "vllm::all_reduce",
                              "in": {"tensor": _ref(d, r, parts[r].reshape(M, H))}, "out": {"ret0": _ref(d, r, got.reshape(M, H))}})
            events[r].append({"seq": seq + 1, "layer": LAYER, "parent": parent, "op": inside, "in": {}, "out": {}})
    for r in range(W):
        events[r].append({"seq": len(events[r]), "layer": None, "parent": None, "op": "model_output", "in": {},
                          "out": {"hidden": {"shape": [M, 6144]}}})
    roots = []
    for r in range(W):
        sd = d / f"rank{r}" / "step0"
        sd.mkdir(parents=True, exist_ok=True)
        ops = "".join(json.dumps(e) + "\n" for e in events[r]).encode()
        (sd / "ops.jsonl").write_bytes(ops)
        (d / f"rank{r}" / "params.json").write_bytes(b"{}")
        ops_sha, params_sha = hashlib.sha512(ops).hexdigest(), hashlib.sha512(b"{}").hexdigest()
        root = hashlib.sha512(f"verity/spot/rank-root\0{ops_sha}\0{params_sha}".encode()).hexdigest()
        roots.append(root)
        (sd / "meta.json").write_text(json.dumps({"ops_sha512": ops_sha, "params_sha512": params_sha, "rank_root": root,
                                                  "scheduled": {"req0": M}, "check": True}))
    step_root = hashlib.sha512(("verity/spot/step-root\0" + "\0".join(roots)).encode()).hexdigest()
    for r in range(W):
        (d / f"rank{r}" / "step0" / "commit.json").write_text(json.dumps(
            {"root": step_root, "coins": "00" * 64, "rank_roots": roots, "layers": {"dense": [LAYER]}, "opened": {}}))
    return seqs


def test_an_nccl_call_is_counted_and_the_step_still_stages_its_multimem_calls(tmp_path):
    S = _stage()
    seqs = fake_capture(tmp_path, [(6144, "symm_mem"), (6144, "nccl"), (6144, "symm_mem")])
    unstaged: dict = {}
    insts = list(S.allreduce_instances(S.Capture(tmp_path, 0), unstaged))
    assert {i[0] for i in insts} == {f"AllReduceMultimem_W{W}"}
    assert sorted({i[2][0] for i in insts}) == [seqs[0], seqs[2]]
    assert len(insts) == 2 * 2 * M
    nccl = unstaged[f"AllReduceNccl_W{W}"]
    assert nccl["population"] == 2 * M and [c["seq"] for c in nccl["calls"]] == [seqs[1]]
    assert "pynccl_comm" in nccl["why"] and f"world {W}" in nccl["why"]
    assert f"AllReduceMultimem_W{W}" not in unstaged


def test_a_staged_instance_is_every_ranks_half_row_and_the_received_row(tmp_path):
    S = _stage()
    fake_capture(tmp_path, [(6144, "symm_mem")])
    insts = list(S.allreduce_instances(S.Capture(tmp_path, 0)))
    label, block, key, args_of, want_of = insts[2 * 5 + 1]
    assert key[1:] == (5, 1) and "SIZE=3072" in block and "FTZ=false" in block
    (parts,), (want,) = args_of(), want_of()
    assert parts.shape == (W, 3072) and want.shape == (3072,)
    assert np.array_equal(want, K.multimem_rows(parts, [AR.GLM53_H200_TREE] * W, False))


def test_a_rank_that_received_other_words_fails_its_instance_by_name(tmp_path):
    S = _stage()
    fake_capture(tmp_path, [(6144, "symm_mem")], flip=(5, 0))
    insts = list(S.allreduce_instances(S.Capture(tmp_path, 0)))
    with pytest.raises(LookupError, match="rank 5 received other words than rank 0"):
        insts[0][4]()
    assert insts[1][4]()[0].shape == (3072,)


def test_the_steps_recorded_tree_is_bound_and_the_ranks_files_must_agree(tmp_path):
    S = _stage()
    fake_capture(tmp_path, [(6144, "symm_mem")])
    node = [[[[[[[0, 2], 3], 1], 5], 6], 4], 7]
    for r in range(W):
        (tmp_path / f"rank{r}" / "step0" / S.TREE_FILE).write_text(json.dumps({"tree": node, "source": "art:probe"}))
    cap = S.Capture(tmp_path, 0)
    assert cap.tree == {"tree": node, "source": "art:probe"}
    assert f"TREES={S.multimem_trees(W, node)}," in next(S.allreduce_instances(cap))[1]
    (tmp_path / "rank5" / "step0" / S.TREE_FILE).write_text(json.dumps({"tree": AR.GLM53_H200_TREE, "source": "?"}))
    with pytest.raises(SystemExit, match="disagree"):
        S.Capture(tmp_path, 0)


def test_a_multimem_call_past_the_captured_tokens_is_named(tmp_path):
    S = _stage()
    seqs = fake_capture(tmp_path, [(6144, "symm_mem"), (16, "symm_mem")])
    unstaged: dict = {}
    insts = list(S.allreduce_instances(S.Capture(tmp_path, 0), unstaged))
    assert AR.covered("defaults", 6144, "symm_mem_comm", M) is not None and AR.covered("defaults", 16, "symm_mem_comm", M) is None
    assert unstaged == {f"AllReduceMultimem_W{W}": {"past_captured": [{"seq": seqs[1], "rows": M}]}}
    assert len(insts) == 2 * 2 * M
