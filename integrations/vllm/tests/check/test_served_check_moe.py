"""check.replay.served_check on a served Program with a fused-MoE block (`acquire.served_moe.expand` of the op): the committed values
are what the eager per-row references compute (`rows.moe_router`, `rows.moe_expert_gemm` pair by pair with the step's Definition,
`act_rows.silu_mul_expf`, `rows.moe_sum`), over three engine steps whose launch configs (lowered to MMA v2 and to wgmma) all bind
Hopper's expert-GEMM step, and the check's vectorised twins pass them; each single perturbation, and the Ampere step's Definition,
fails by name."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire import served_moe as SM
from verity_vllm.check.replay import served_check as SC
from verity.ml.definitions.numerics import rows as R
from verity.ml.definitions.numerics.act_rows import silu_mul_expf
from verity.ml.definitions.numerics.rows import token_select_first_max
from verity.ml.definitions.registry import act, b1
from verity.ml.definitions.registry import moe as M
from verity_vllm.query.manifest.format import manifest_digest

V, H, E, TOPK, VPT, I = 64, 32, 8, 2, 4, 16
LAYER = "model.layers.0.mlp.experts"
SCHEDULE = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 3},
            {"request_id": "r1", "arrive": 1, "prompt_tokens": 40, "max_tokens": 2}]
STEPS = 3
EMB, OUT = "c00.embedding/buf0", "c00.moe_forward/buf1"
LG, RW, RI, UP, ACT, DOWN = (f"c00.moe/{n}" for n in SM.PLANES)
FACTS = {"gate": {"weight": "gate", "shape": [E, H]}, "w13": {"weight": "w13", "shape": [E, 2 * I, H]},
         "w2": {"weight": "w2", "shape": [E, H, I]}, "E": E, "top_k": TOPK, "renormalize": True, "VPT": VPT, **SM.REQUIRED}
SMALL = {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "GROUP_SIZE_M": 1, "num_stages": 3, "num_warps": 4}
WIDE = {"BLOCK_SIZE_K": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1, "num_stages": 3, "num_warps": 4}
#: the steps' M: 4 (3 prefill rows, piecewise), 41 (eager), 2 (a full graph of two decodes)
RECORD = {"layers": {LAYER: FACTS}, "pieces": {"0": LAYER}, "cc": [9, 0],
          "configs": {"4": {"False": SMALL, "True": SMALL}, "41": {"False": WIDE, "True": WIDE}, "2": {"False": WIDE, "True": SMALL}}}
UPK, DOWNK = SM.plan_key(E, H, 2 * I, False), SM.plan_key(E, I, H, True)


def _build() -> dict:
    wired = {"calls": [{"op": b1.Embedding.id, **SV.bound(b1.Embedding, V=V, H=H), "piece": 0,
                        "reads": [{"value": SV.IDS, "width": 1}, {"weight": "embed", "shape": [V, H]}], "writes": [EMB]},
                       {"op": "moe_forward", "piece": 0, "layer": {"layer_name": True}, "reads": [{"value": EMB, "width": H}],
                        "writes": [OUT]}],
             "values": {SV.IDS: {"width": 1, "piece": 0}, SV.POSITIONS: {"width": 1, "piece": 0}, EMB: {"width": H, "piece": 0},
                        OUT: {"width": H, "piece": 0}}, "attention": []}
    prog, plan = SM.expand(wired, RECORD)
    gemms = sorted({tuple(c["gemm"]) for c in prog["calls"] if c["op"] == "mm"})
    args = (SCHEDULE, STEPS, gemms, (H, V), (2, 1, 64))
    kw = dict(capture_sizes=[1, 2, 4, 8], max_num_seqs=8, split_cap=SV.SPLIT_CAP, device=SD.device_of((9, 0), 132), moe=plan)
    _plans, missing = SV.step_plans(*args, {}, **kw)
    plans, missing = SV.step_plans(*args, {f"{M_},{K},{N}": {"kernel": "nosplit"} for M_, K, N in missing}, **kw)
    assert not missing
    program = {"calls": prog["calls"], "attention": [], "values": prog["values"],
               "logits": {"name": SV.LOGITS, "x": {"value": OUT, "width": H}, "w": {"weight": "lm_head", "shape": [V, H]}, "gemm": [H, V]},
               "sampled": {"name": SV.SAMPLED, **SV.bound(b1.TokenSelect, V=V)}}
    identities = SV.identities_of(plans, program["values"], V, SCHEDULE, "test", 0)
    return {"kind": SV.KIND, "program": program, "steps_plan": plans, "program_digest": "test", "identities": identities,
            "manifest_digest": manifest_digest(identities)}


def _words(a) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {"embed": _words(rng.standard_normal((V, H))), "gate": _words(rng.standard_normal((E, H)) * H ** -0.5),
            "w13": _words(rng.standard_normal((E, 2 * I, H)) * H ** -0.5), "w2": _words(rng.standard_normal((E, H, I)) * I ** -0.5),
            "lm_head": _words(rng.standard_normal((V, H)) * H ** -0.5)}


def _expert(d: dict, ops: list, routed: bool) -> np.ndarray:
    return R.moe_expert_gemm(d["statics"], ops, routed=routed, dotted="DOT" in d["statics"])["down" if routed else "up"]


def _block(plan: dict, x: np.ndarray, W: dict) -> dict[str, np.ndarray]:
    """The block's planes on rows x, row by row and pair by pair by the eager references."""
    n = len(x)
    gate = plan["gemms"][f"{H},{E}"]
    lg = SC.GEMMS[gate["base"]](gate["statics"], x, W["gate"])
    rw, ri = np.zeros((n, TOPK), np.uint32), np.zeros((n, TOPK), np.int32)
    up, ac, down = (np.zeros((n, TOPK, w), np.uint16) for w in (2 * I, I, H))
    out = np.zeros((n, H), np.uint16)
    for t in range(n):
        r = R.moe_router({"E": E, "TOPK": TOPK, "VPT": VPT}, [lg[t]], renormalize=True)
        rw[t], ri[t] = r["router_w"], r["router_ids"].astype(np.int32)
        for k in range(TOPK):
            up[t, k] = _expert(plan["moe"][UPK], [x[t], ri[t, k], W["w13"]], False)
            ac[t, k] = silu_mul_expf({"I": I}, [up[t, k]])["out"]
            down[t, k] = _expert(plan["moe"][DOWNK], [ac[t, k], ri[t, k], rw[t, k], W["w2"]], True)
        out[t] = R.moe_sum({"TOPK": TOPK, "H": H}, [down[t].reshape(-1)])["sum"]
    return {LG: lg, RW: rw, RI: ri, UP: up.reshape(n, -1), ACT: ac.reshape(n, -1), DOWN: down.reshape(n, -1), OUT: out}


def _serve(build: dict, W: dict) -> tuple[list[SC.Step], dict[str, list[int]]]:
    rng = np.random.default_rng(5)
    prompts = {r["request_id"]: rng.integers(0, V, r["prompt_tokens"]) for r in SCHEDULE}
    last: dict[str, int] = {}
    steps = []
    for plan in build["steps_plan"]:
        rows, lo = [], 0
        for r in plan["requests"]:
            rows.append((r["request_id"], lo, lo + r["tokens"]))
            lo += r["tokens"]
        ids = np.concatenate([prompts[r] if r not in last else [last[r]] for r, _lo, _hi in rows])
        pos = np.concatenate([np.arange(q["klen"] - q["tokens"], q["klen"]) for q in plan["requests"]])
        vals = {SV.IDS: ids.astype(np.int64)[:, None], SV.POSITIONS: pos[:, None], EMB: W["embed"][ids]}
        vals.update(_block(plan, vals[EMB], W))
        d = plan["logits"]
        vals[SV.LOGITS] = SC.GEMMS[d["base"]](d["statics"], vals[OUT][[b - 1 for _r, _a, b in rows]], W["lm_head"])
        vals[SV.SAMPLED] = np.array([token_select_first_max(x) for x in vals[SV.LOGITS]], np.int64)[:, None]
        last.update({r: int(s) for (r, _a, _b), s in zip(rows, vals[SV.SAMPLED][:, 0])})
        steps.append(SC.Step(plan["step"], rows, vals))
    return steps, {r: list(map(int, p)) for r, p in prompts.items()}


@pytest.fixture(scope="module")
def served():
    build, W = _build(), _weights()
    steps, prompts = _serve(build, W)
    return build, W, steps, prompts


def _check(served, steps=None, build=None):
    b, W, st, prompts = served
    return SC.check(build or b, st if steps is None else steps, W.__getitem__, seed=None, rows=8, prompts=prompts)


def _with(steps, step: int, name: str, f) -> list[SC.Step]:
    out = list(steps)
    s = out[step]
    out[step] = SC.Step(s.step, s.rows, {**s.values, name: f(s.values[name].copy())})
    return out


def _failed(rep) -> set[tuple[int, str]]:
    return {(f["step"], f["value"]) for f in rep["failures"]}


def test_every_launch_config_binds_hoppers_expert_gemm_step(served):
    plans = served[0]["steps_plan"]
    assert [(p["gemm_m"], p["mode"]) for p in plans] == [(4, "PIECEWISE"), (41, "NONE"), (2, "FULL")]
    assert [(p["moe"][UPK]["base"], p["moe"][DOWNK]["base"]) for p in plans] == [(M.MoeExpertGemmV2.id, M.MoeExpertGemmWV2.id)] * 3


def test_the_served_moe_values_pass(served):
    rep = _check(served)
    assert rep["ok"], rep["failures"]
    assert rep["steps_checked"] == [0, 1, 2]
    kinds = {M.MoeRouterTopKNorm.id, M.MoeExpertGemmV2.id, M.MoeExpertGemmWV2.id, act.SiluMul.id, M.MoeSum.id, b1.GemmV2.id}
    assert kinds <= set(rep["tally"])
    assert all(ok == n for ok, n in rep["tally"].values())
    s1 = rep["steps"]["1"]
    picked = sum(map(len, rep["picks"]["1"].values()))
    assert (s1[UP]["rows"], s1[UP]["pairs"], s1[UP]["nonfinite"]) == (picked, TOPK * picked, 0)


def _r0(served, step: int) -> int:
    return next(lo for r, lo, _hi in served[2][step].rows if r == "r0")


@pytest.mark.parametrize("name,col", [(LG, 3), (RW, 1), (RI, 0), (UP, 2 * I + 5), (ACT, I + 1), (DOWN, H + 7), (OUT, 4)])
def test_a_flipped_word_fails_its_value(served, name, col):
    row = _r0(served, 1)

    def flip(a):
        a[row, col] ^= 1
        return a
    rep = _check(served, _with(served[2], 1, name, flip))
    assert (1, name) in _failed(rep)


def test_an_expert_id_outside_the_experts_fails_the_launches(served):
    row = _r0(served, 0)

    def far(a):
        a[row, 1] = E
        return a
    failed = _failed(_check(served, _with(served[2], 0, RI, far)))
    assert {(0, RI), (0, UP), (0, DOWN)} <= failed


def _steps_apart(served) -> tuple[dict, list[SC.Step], dict]:
    """Weights on which Ampere's k16 step and Hopper's round every pair's first `up` word apart (random rows rarely do): every
    hidden-state row 1 + 2^-8 in the first group of eight, four 2^-25 in the second, against each expert's first w13 row of ones
    there (the two-group 25-bit adder drops the 2^-25s, the one-group 26-bit adder carries them past the bf16 tie)."""
    build, W, _steps, _prompts = served
    x, w = np.zeros(H, np.float32), np.zeros(H, np.float32)
    at = [0, 1, 8, 9, 10, 11]
    x[at], w[at] = [1, 2 ** -8, *[2 ** -25] * 4], 1
    W = {**W, "embed": np.tile(_words(x), (V, 1)), "w13": W["w13"].copy()}
    W["w13"][:, 0] = _words(w)
    steps, prompts = _serve(build, W)
    return W, steps, prompts


def test_the_expert_gemm_is_the_plans_step(served):
    W, steps, prompts = _steps_apart(served)
    build = served[0]
    assert SC.check(build, steps, W.__getitem__, seed=None, rows=8, prompts=prompts)["ok"]
    other = copy.deepcopy(build)
    plan = other["steps_plan"][1]
    plan["moe"][UPK] = SV.bound(M.MoeExpertGemm, E=E, K=H, N=2 * I)
    rep = SC.check(other, steps, W.__getitem__, seed=None, rows=8, prompts=prompts)
    assert (1, UP) in _failed(rep) and rep["steps"]["1"][UP]["definition"] == plan["moe"][UPK]["definition"]
    assert rep["steps"]["1"][UP]["bad"] == rep["steps"]["1"][UP]["pairs"]


def test_a_routed_expert_weight_of_another_shape_is_refused(served):
    b, W, steps, prompts = served
    rep = SC.check(b, steps, lambda n: W[n][:-1] if n == "w2" else W[n], seed=None, rows=8, prompts=prompts)
    assert all((s, DOWN) in _failed(rep) for s in range(STEPS))
