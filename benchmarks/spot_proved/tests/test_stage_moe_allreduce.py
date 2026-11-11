"""`stage.py`'s MoE and plain all-reduce populations, and the step's op coverage, on an in-memory capture: the w13 launch is
staged beside the routed w2 one, each pair reading its own A row (the w13 launch its token's); a plain all-reduce with no
multimem op inside it is FlashInfer's MNNVL one within 0.5 MiB, at its shot, and NCCL's past it; a multimem one binds the
node's reduction tree, which capture.py's settings carry; an op inside an ungated op is listed as not gated, one inside a
gated op is not, a call its population refused is, and the MoE align is listed apart as a schedule op."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
E, H, I, TOPK, M = 2, 256, 128, 2, 3
CFG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "num_warps": 4}


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cap:
    """Ranks' events over tensors held by name; ``cells`` yields the pairs and columns it is given."""

    def __init__(self, ranks, events, tensors, cells=((3, 130),)):
        self.ranks, self.events, self.t, self._cells = ranks, events, tensors, cells
        self.whole, self.layers = True, []

    def opened(self, e, weights=False):
        return True

    only = None

    def real_rows(self, r, rows):
        return rows

    def rows(self, r, rows):
        return list(range(rows))

    def cells(self, label, call, *dims, slots=1):
        yield from self._cells

    def tensor(self, r, ref):
        return self.t[(r, ref["sha512"])]


def _ref(cap_t: dict, r: int, name: str, a: np.ndarray) -> dict:
    cap_t[(r, name)] = a
    return {"sha512": name, "shape": list(a.shape)}


def _launch(t: dict, seq: int, routed: bool) -> dict:
    rng = np.random.default_rng(seq)
    K, N = (I, H) if routed else (H, 2 * I)
    rows = M * TOPK if routed else M
    w = "w2" if routed else "w13"
    ins = {"A": _ref(t, 0, f"{w}.A", rng.integers(0, 255, (rows, K), dtype=np.uint8)),
           "A_scale": _ref(t, 0, f"{w}.As", rng.random((rows, K // 128), dtype=np.float32)),
           "B": _ref(t, 0, f"{w}.B", rng.integers(0, 255, (E, N, K), dtype=np.uint8)),
           "B_scale": _ref(t, 0, f"{w}.Bs", rng.random((E, N // 128, K // 128), dtype=np.float32)),
           "topk_weights": _ref(t, 0, f"{w}.g", rng.random((M, TOPK), dtype=np.float32)) if routed else None,
           "expert_ids": _ref(t, 0, f"{w}.ids", rng.integers(0, E, M * TOPK)),
           "mul_routed_weight": routed, "top_k": 1 if routed else TOPK, "config": CFG}
    out = {"C": _ref(t, 0, f"{w}.C", rng.integers(0, 2 ** 16, (M, TOPK, N), dtype=np.uint16))}
    return {"seq": seq, "op": "moe_launch", "parent": None, "layer": 5, "in": ins, "out": out}


def _router(t: dict, seq: int, ids: np.ndarray, layer: int = 5) -> dict:
    out = {"ret0": _ref(t, 0, "router.w", np.ones((M, TOPK), dtype=np.float32)), "ret1": _ref(t, 0, "router.ids", ids.astype(np.int32))}
    return {"seq": seq, "op": "_moe_C::grouped_topk", "parent": None, "layer": layer, "in": {}, "out": out}


def test_the_w13_launch_is_staged_beside_w2_each_pair_on_its_own_a_row_and_the_routers_expert(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    from verity_vllm import TargetProfile
    t: dict = {}
    l13, l2 = _launch(t, 1, False), _launch(t, 2, True)
    other = (E - 1) - t[(0, "w13.ids")].reshape(M, TOPK)
    cap = _Cap([0], {0: [_router(t, 0, other), l13, l2]}, t)
    insts = list(stage.moe_instances(cap, TargetProfile(compute_capability=(9, 0), batch_invariant=False)))
    assert [i[0].split("_K")[0] for i in insts] == ["MoeExpertFp8Gemm", "MoeExpertFp8GemmW"]
    assert insts[0][1].id.startswith("MoeExpertFp8Gemm_v1{") and insts[1][1].id.startswith("MoeExpertFp8GemmW_v1{")
    j, col = 3, 130
    (_, _, _, a13, y13), (_, _, _, a2, y2) = insts
    A, As, ex, B, Bs = a13()
    assert np.array_equal(A, t[(0, "w13.A")][j // TOPK]) and np.array_equal(As, t[(0, "w13.As")][j // TOPK])
    assert int(ex[0]) == int(t[(0, "router.ids")].reshape(-1)[j]) != int(t[(0, "w13.ids")][j])
    assert int(a2()[2][0]) == int(t[(0, "router.ids")].reshape(-1)[j])
    assert np.array_equal(B, t[(0, "w13.B")][:, col:col + 1]) and np.array_equal(Bs, t[(0, "w13.Bs")][:, 1:2])
    assert np.array_equal(y13()[0], t[(0, "w13.C")][j // TOPK, j % TOPK, col:col + 1])
    A, As, ex, g, B, Bs = a2()
    assert np.array_equal(A, t[(0, "w2.A")][j]) and float(g[0]) == float(t[(0, "w2.g")].reshape(-1)[j])
    assert np.array_equal(y2()[0], t[(0, "w2.C")][j // TOPK, j % TOPK, col:col + 1])
    for events in ([l13], [_router(t, 0, other, layer=4), l13]):
        with pytest.raises(LookupError, match="align's schedule"):
            list(stage.moe_instances(_Cap([0], {0: events}, t), TargetProfile(compute_capability=(9, 0), batch_invariant=False)))


def _allreduce_cap(rows_of: list[int], multimem: bool = False, W: int = 8, Hd: int = 6144) -> _Cap:
    rng = np.random.default_rng(1)
    t: dict = {}
    events: dict = {r: [] for r in range(W)}
    for i, T in enumerate(rows_of):
        out = rng.integers(0, 2 ** 15, (T, Hd), dtype=np.uint16)
        for r in range(W):
            events[r].append({"seq": 2 * i, "op": "vllm::all_reduce", "parent": None, "layer": None,
                              "in": {"tensor": _ref(t, r, f"in{i}", rng.integers(0, 2 ** 15, (T, Hd), dtype=np.uint16))},
                              "out": {"ret0": _ref(t, r, f"out{i}", out)}})
            if multimem:
                events[r].append({"seq": 2 * i + 1, "op": "symm_mem::multimem_all_reduce_", "parent": 2 * i, "in": {}, "out": {}})
    return _Cap(list(range(W)), events, t)


def test_a_plain_allreduce_without_multimem_is_mnnvl_within_half_a_mib_and_nccl_past_it(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    from verity.definitions.registry import tp_allreduce as AR
    cap = _allreduce_cap([1, 32, 43])
    assert [stage.allreduce_route(cap, cap.events[0][i]) for i in range(3)] == ["fi_ar_comm", "fi_ar_comm", "pynccl_comm"]
    assert 42 * 6144 * 2 <= AR.FI_MAX_BYTES < 43 * 6144 * 2
    unstaged: dict = {}
    insts = list(stage.allreduce_instances(cap, unstaged))
    labels = {(i[0], i[1]) for i in insts}
    assert labels == {("AllReduceMnnvl_W8_S1", "AllReduceMnnvl_v1{WORLD=8,N=3072,SHOT=1}"),
                      ("AllReduceMnnvl_W8_S2", "AllReduceMnnvl_v1{WORLD=8,N=3072,SHOT=2}")}
    assert len(insts) == 2 * (1 + 32)
    assert unstaged["AllReduceNccl_W8"]["calls"] == [{"seq": 4, "rows": 43, "cols": 6144, "bytes": 43 * 6144 * 2}]
    (parts,), (want,) = insts[1][3](), insts[1][4]()
    assert parts.shape == (8, 3072) and np.array_equal(want, cap.t[(0, "out0")][0, 3072:])
    mm = _allreduce_cap([1], multimem=True)
    assert stage.allreduce_route(mm, mm.events[0][0]) == "symm_mem_comm"


def test_a_multimem_call_binds_the_nodes_tree_and_the_registrys_only_without_one(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    from verity.definitions.registry import tp_allreduce as AR
    node = stage.parse_tree("(((((((0,2),3),1),5),6),4),7)", 8)
    assert node == AR.chain(8, (0, 2, 3, 1, 5, 6, 4, 7)) != AR.multimem_tree(AR.GLM53_H200_TREE, 8)
    cap = _allreduce_cap([1], multimem=True)
    cap.tree = None
    assert f"TREES={stage.multimem_trees(8, AR.GLM53_H200_TREE)}," in next(stage.allreduce_instances(cap))[1]
    cap.tree = {"tree": [[[[[[[0, 2], 3], 1], 5], 6], 4], 7], "source": "the probe"}
    trees = json.loads(next(stage.allreduce_instances(cap))[1].split("TREES=")[1].split(",FTZ")[0])
    assert [AR.multimem_tree(t, 8) for t in trees] == [node] * 8
    for bad in ("((0,1),1)", "((0,1,2),3)"):
        with pytest.raises(ValueError):
            stage.parse_tree(bad, 4)


def test_capture_site_writes_the_nodes_tree_into_its_settings(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("spot_capture_t", HERE / "capture.py")
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    base = ["site", str(tmp_path / "site"), "--capture", str(tmp_path / "cap"), "--steps", "3"]
    assert cap.main(base + ["--allreduce-tree", "(((0,2),3),1)", "--allreduce-tree-source", "art:probe"]) == 0
    cfg = json.loads((tmp_path / "site" / cap.CONFIG).read_text())
    assert cfg["allreduce_tree"] == {"tree": [[[0, 2], 3], 1], "source": "art:probe"}
    assert cap.main(base) == 0 and json.loads((tmp_path / "site" / cap.CONFIG).read_text())["allreduce_tree"] is None
    with pytest.raises(ValueError, match="each once"):
        cap.reduction_tree("((0,2),2)")


def test_op_coverage_lists_ops_inside_an_ungated_op_and_not_inside_a_gated_one(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    ev = [{"seq": 0, "op": "vllm::moe_forward_shared", "parent": None}, {"seq": 1, "op": "_moe_C::moe_align_block_size", "parent": 0},
          {"seq": 2, "op": "moe_launch", "parent": 0}, {"seq": 3, "op": "vllm::all_reduce", "parent": None},
          {"seq": 4, "op": "symm_mem::multimem_all_reduce_", "parent": 3}, {"seq": 5, "op": "aten::copy_", "parent": 4}]
    gated, other, schedule = stage.op_coverage(_Cap([0, 1], {0: ev, 1: ev}, {}))
    assert gated == {"moe_launch": 2, "vllm::all_reduce": 2}
    assert other == {"vllm::moe_forward_shared": 2}
    assert schedule == {"_moe_C::moe_align_block_size": 2} and set(stage.SCHEDULE_OPS) == set(schedule)
    refused = {"AllReduceNccl_W2": {"calls": [{"seq": 3}]}, "AtenAdd_other": {"calls": [{"rank": 1, "seq": 2}]}}
    gated, other, _ = stage.op_coverage(_Cap([0, 1], {0: ev, 1: ev}, {}), refused)
    assert gated == {"moe_launch": 1}
    assert other == {"vllm::moe_forward_shared": 2, "moe_launch": 1, "vllm::all_reduce": 2, "symm_mem::multimem_all_reduce_": 2,
                     "aten::copy_": 2}
