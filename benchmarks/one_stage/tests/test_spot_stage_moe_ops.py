"""spot_proved's staging of the MoE block's row ops (`stage.router_instances`, `silu_mul_instances`, `moe_sum_instances`) on an
in-memory capture, through the match gate: the router's grouped top-k on `MoeRouterSigmoidBias_v1` at the launch it takes,
``silu_and_mul`` on `SiluMul_v2` and ``moe_sum`` on `MoeSum_v1`, each per row, agree with numpy models of vLLM's kernels and
fail on a flipped word; a router off the one-group sigmoid form is counted, not staged."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cap:
    """One rank's events over tensors held by name; ``cells`` yields the one row it is given."""

    def __init__(self, events, tensors):
        self.ranks, self.events, self.t = [0], {0: events}, tensors

    def opened(self, e, weights=False):
        return True

    def real_rows(self, r, rows):
        return rows

    def cells(self, label, call, *dims):
        yield (1,)

    def tensor(self, r, ref):
        return self.t[(r, ref["sha512"])]


def _ref(t: dict, r: int, name: str, a: np.ndarray, dtype: str | None = None) -> dict:
    t[(r, name)] = a
    return {"sha512": name, "shape": list(a.shape), **({"dtype": dtype} if dtype else {})}


def _bf16(f) -> np.ndarray:
    u = np.asarray(f, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _f32(b) -> np.ndarray:
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


ROUTER = 'MoeRouterSigmoidBias_v1{E=256,TOPK=2,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}'


def _moe_ops_cap(stage, scoring_func: int = 1) -> tuple[_Cap, np.ndarray]:
    """grouped_topk, silu_and_mul and moe_sum on 2 token rows: the router's outputs its Definition's, the others' numpy models
    of vLLM's kernels. Returns the capture and numpy's top-2 experts of each row."""
    import a11
    rng = np.random.default_rng(3)
    t: dict = {}
    E, K, Hd, Id = 256, 2, 8, 8
    logits = rng.standard_normal((2, E)).astype(np.float32)
    bias = (rng.standard_normal(E) * 0.01).astype(np.float32)
    sig = (1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    top = np.argsort(-(sig + bias), axis=1, kind="stable")[:, :K]
    spec = stage.block_fn(ROUTER)
    out = np.stack([np.asarray([int(x) for x in a11.evaluate_units(spec, spec, [stage.words(lg), stage.words(bias)], 1,
                                                                   4_000_000)["call_out"]], np.uint32) for lg in logits])
    w, ids = out[:, :K].view(np.float32), out[:, K:].view(np.int32)
    gu = _bf16(rng.standard_normal((2 * K, 2 * Id)).astype(np.float32) * 3)
    g, u = _f32(gu[:, :Id]), _f32(gu[:, Id:])
    act = _bf16(_f32(_bf16(g / (np.float32(1) + np.exp(-g)))) * u)
    y = _bf16(rng.standard_normal((2, K, Hd)).astype(np.float32))
    acc = np.zeros((2, Hd), np.float32)
    for k in range(K):
        acc = (acc + _f32(y[:, k])).astype(np.float32)
    ev = [{"seq": 0, "op": "_moe_C::grouped_topk", "parent": None, "layer": 5,
           "in": {"scores": _ref(t, 0, "logits", logits, "torch.float32"), "bias": _ref(t, 0, "bias", bias, "torch.float32"),
                  "n_group": 1, "topk_group": 1, "topk": K, "renormalize": True, "routed_scaling_factor": 2.5,
                  "scoring_func": scoring_func},
           "out": {"ret0": _ref(t, 0, "w", w), "ret1": _ref(t, 0, "ids", ids)}},
          {"seq": 1, "op": "_C::silu_and_mul", "parent": None, "layer": 5,
           "in": {"input": _ref(t, 0, "gu", gu, "torch.bfloat16")}, "out": {"result": _ref(t, 0, "act", act)}},
          {"seq": 2, "op": "_moe_C::moe_sum", "parent": None, "layer": 5,
           "in": {"input": _ref(t, 0, "y", y, "torch.bfloat16"), "expert_map": None},
           "out": {"output": _ref(t, 0, "sum", _bf16(acc))}}]
    return _Cap(ev, t), top


def test_the_router_top_k_silu_and_mul_and_moe_sum_are_gated_per_row(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    cap, top = _moe_ops_cap(stage)
    insts = [*stage.router_instances(cap), *stage.silu_mul_instances(cap), *stage.moe_sum_instances(cap)]
    assert [(i[0], i[2]) for i in insts] == [("MoeRouterSigmoidBias_E256_TOPK2_block", (0, 0, 1)), ("SiluMul_I8", (0, 1, 1)),
                                             ("MoeSum_TOPK2_H8", (0, 2, 1))]
    assert insts[0][1] == ROUTER and insts[0][4]()[1].tolist() == top[1].tolist()
    for label, block, key, args_of, want_of in insts:
        g, _ = stage.gate_instance(stage.block_fn(block), key, args_of, want_of, 4_000_000)
        assert g["agree"] is True, (label, g)
    cap.t[(0, "sum")] = cap.t[(0, "sum")].copy()
    cap.t[(0, "sum")][1, 3] ^= 1
    _, block, key, args_of, want_of = insts[2]
    assert stage.gate_instance(stage.block_fn(block), key, args_of, want_of, 4_000_000)[0]["first_diff"] == 3


def test_a_router_off_the_sigmoid_one_group_form_is_counted_not_staged(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    cap, _ = _moe_ops_cap(stage, scoring_func=0)
    unstaged: dict = {}
    assert list(stage.router_instances(cap, unstaged)) == []
    assert unstaged["MoeRouter_E256"]["population"] == 2 and unstaged["MoeRouter_E256"]["calls"][0]["args"][3] == 0
