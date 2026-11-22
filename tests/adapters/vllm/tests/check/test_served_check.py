"""check.replay.served_check on a small served Program built by hand: one layer of the Build's form (the Embedding, Inductor's
RMSNorm, AddRMSNorm, Add2RMSNorm, SiLU-and-mul and RoPE, cuBLAS GEMMs at both routes, FA3 unsplit and split, the logits and
TokenSelect) over four engine steps of three requests, whose step plans `acquire.served.step_plans` derives.  The committed
values are what the Program computes, step after step (each decode step reads the token the previous step sampled), and the
check passes them; each single perturbation fails by name."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_device as SD
from verity_vllm.check.replay import served_check as SC
from verity_vllm.definitions.numerics.rows import token_select_first_max
from verity_vllm.definitions.registry import b1, cublas_bf16
from verity_vllm.definitions.registry import bf16_gemm_check as BC
from verity_vllm.definitions.registry import fa3_check_inf as F3
from verity_vllm.definitions.registry import inductor as IR
from verity_vllm.query.manifest.format import manifest_digest

V, H, NH, KVH, D, INTER, P = 64, 64, 2, 1, 64, 32, 512
QKV = (NH + 2 * KVH) * D
NORM = dict(N=H, EPS=1e-5, XBLOCK=1, R0_BLOCK=256, NUM_WARPS=1)
SCHEDULE: list[dict[str, Any]] = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 4},
                                  {"request_id": "r1", "arrive": 1, "prompt_tokens": 2, "max_tokens": 3},
                                  {"request_id": "r2", "arrive": 2, "prompt_tokens": 200, "max_tokens": 2}]
STEPS = 4
H200 = SD.device_of((9, 0), 132)

EMB, N0, QKVV, Q, K, ATT, O_PROJ, N1, GU, ACT, DOWN, RES, OUT = (
    "c00.embedding/buf0", "c00.rms_norm/buf1", "c00.mm/buf2", "c00.rope/buf3", "c00.rope/buf4", "c01.attention/layers.0.attn",
    "c02.mm/buf0", "c02.add_rms_norm/buf1", "c02.mm/buf2", "c02.silu_mul/buf3", "c02.mm/buf4", "c02.add2_rms_norm/buf5",
    "c02.add2_rms_norm/buf6")


def _v(name: str, width: int, cols: tuple[int, int] | None = None) -> dict:
    return {"value": name, "width": width, **({"cols": list(cols)} if cols else {})}


def _w(name: str, *shape: int) -> dict:
    return {"weight": name, "shape": list(shape)}


def _call(defn, statics: dict, reads: list, writes: list) -> dict:
    return {"op": defn.id, **SV.bound(defn, **statics), "reads": reads, "writes": writes}


def _mm(x: dict, w: dict, y: str) -> dict:
    return {"op": "mm", "gemm": [w["shape"][1], w["shape"][0]], "reads": [x, w], "writes": [y]}


CS = {**_w("cos_sin", P, D), "row_of": _v(SV.POSITIONS, 1)}
PRE = [_call(b1.Embedding, dict(V=V, H=H), [_v(SV.IDS, 1), _w("embed", V, H)], [EMB]),
       _call(IR.InductorRMSNorm, NORM, [_v(EMB, H), _w("norm0", H)], [N0]),
       _mm(_v(N0, H), _w("qkv", QKV, H), QKVV),
       _call(IR.InductorRoPE, dict(NHEADS=NH, D=D), [_v(QKVV, NH * D, (0, NH * D)), CS], [Q]),
       _call(IR.InductorRoPE, dict(NHEADS=KVH, D=D), [_v(QKVV, KVH * D, (NH * D, (NH + KVH) * D)), CS], [K])]
ATTENTION: dict[str, Any] = {"piece": 1, "layer": "layers.0.attn", "name": ATT, "q": _v(Q, NH * D), "k": _v(K, KVH * D),
                             "v": _v(QKVV, KVH * D, ((NH + KVH) * D, QKV)), "NH": NH, "KVH": KVH, "D": D}
POST = [_mm(_v(ATT, NH * D), _w("o", H, NH * D), O_PROJ),
        _call(IR.InductorAddRMSNorm, NORM, [_v(O_PROJ, H), _v(EMB, H), _w("norm1", H)], [N1]),
        _mm(_v(N1, H), _w("gate_up", 2 * INTER, H), GU),
        _call(IR.InductorSiluMul, dict(I=INTER), [_v(GU, 2 * INTER)], [ACT]),
        _mm(_v(ACT, INTER), _w("down", H, INTER), DOWN),
        _call(IR.InductorAdd2RMSNorm, NORM, [_v(DOWN, H), _v(O_PROJ, H), _v(EMB, H), _w("norm2", H)], [RES, OUT])]
WIDTHS = {SV.IDS: 1, SV.POSITIONS: 1, EMB: H, N0: H, QKVV: QKV, Q: NH * D, K: KVH * D, ATT: NH * D, O_PROJ: H, N1: H, GU: 2 * INTER,
          ACT: INTER, DOWN: H, RES: H, OUT: H}


def _route(M: int) -> dict:
    return {"kernel": "splitk", "SPLIT_K": 2} if M <= 4 else {"kernel": "nosplit"}


def _build() -> dict:
    gemms = sorted({tuple(c["gemm"]) for c in PRE + POST if c["op"] == "mm"})
    args = (SCHEDULE, STEPS, gemms, (H, V), (NH, KVH, D))
    kw = dict(capture_sizes=[1, 2, 4, 8], max_num_seqs=8, split_cap=SV.SPLIT_CAP, device=H200)
    _plans, missing = SV.step_plans(*args, {}, **kw)
    plans, missing = SV.step_plans(*args, {f"{M},{K_},{N}": _route(M) for M, K_, N in missing}, **kw)
    assert not missing
    program = {"calls": PRE + POST, "attention": [ATTENTION], "values": {n: {"width": w, "piece": 0} for n, w in WIDTHS.items()},
               "logits": {"name": SV.LOGITS, "x": _v(OUT, H), "w": _w("lm_head", V, H), "gemm": [H, V]},
               "sampled": {"name": SV.SAMPLED, **SV.bound(b1.TokenSelect, V=V)}}
    identities = SV.identities_of(plans, program["values"], V, SCHEDULE, "test", 0)
    return {"kind": SV.KIND, "program": program, "steps_plan": plans, "program_digest": "test", "identities": identities,
            "manifest_digest": manifest_digest(identities)}


def _words(a) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _bf16(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> np.ndarray:
    return _words(rng.standard_normal(shape) * scale)


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    w = {"embed": _bf16(rng, V, H), "qkv": _bf16(rng, QKV, H, scale=H ** -0.5), "o": _bf16(rng, H, NH * D, scale=(NH * D) ** -0.5),
         "gate_up": _bf16(rng, 2 * INTER, H, scale=H ** -0.5), "down": _bf16(rng, H, INTER, scale=INTER ** -0.5),
         "lm_head": _bf16(rng, V, H, scale=H ** -0.5), **{f"norm{i}": _bf16(rng, H) for i in range(3)}}
    inv = 10000.0 ** (-np.arange(D // 2) / (D // 2))
    ang = np.arange(P)[:, None] * inv[None, :]
    w["cos_sin"] = _words(np.concatenate([np.cos(ang), np.sin(ang)], 1))
    return w


def _operand(ref: dict, vals: dict, W: dict) -> np.ndarray:
    if "weight" in ref:
        w = W[ref["weight"]]
        return w[_operand(ref["row_of"], vals, W).astype(np.int64)] if "row_of" in ref else w
    a = vals[ref["value"]]
    a = a[:, ref["cols"][0]:ref["cols"][1]] if "cols" in ref else a
    return a[:, 0] if ref["width"] == 1 else a


def _serve(build: dict, W: dict) -> tuple[list[SC.Step], dict[str, list[int]]]:
    """What the Program computes at each step, every value on all rows; and the prompts."""
    rng = np.random.default_rng(11)
    prompts = {r["request_id"]: rng.integers(0, V, r["prompt_tokens"]) for r in SCHEDULE}
    cache: dict[str, tuple[list, list]] = {}
    last: dict[str, int] = {}
    steps = []
    for plan in build["steps_plan"]:
        rows, lo = [], 0
        for r in plan["requests"]:
            rows.append((r["request_id"], lo, lo + r["tokens"]))
            lo += r["tokens"]
        ids = np.concatenate([prompts[r] if r not in last else [last[r]] for r, _lo, _hi in rows])
        pos = np.concatenate([np.arange(q["klen"] - q["tokens"], q["klen"]) for q in plan["requests"]])
        vals = {SV.IDS: ids.astype(np.int64)[:, None], SV.POSITIONS: pos[:, None]}

        def run(calls, vals=vals, plan=plan):
            for c in calls:
                xs = [_operand(r, vals, W) for r in c["reads"]]
                if c["op"] == "mm":
                    d = plan["gemms"][f"{c['gemm'][0]},{c['gemm'][1]}"]
                    vals[c["writes"][0]] = SC.GEMMS[d["base"]](d["statics"], *xs)
                else:
                    for name, v in zip(c["writes"], SC.TWINS[c["base"]](c["statics"], *xs)):
                        vals[name] = v

        run(PRE)
        q, k, v = (_operand(ATTENTION[x], vals, W) for x in "qkv")
        out: np.ndarray = np.zeros((lo, NH * D), np.uint16)
        for (r, a, b), req in zip(rows, plan["requests"]):
            kh, vh = cache.setdefault(r, ([], []))
            kh.append(k[a:b].reshape(-1, KVH, D))
            vh.append(v[a:b].reshape(-1, KVH, D))
            kc, vc = np.concatenate(kh), np.concatenate(vh)
            for i, d in enumerate(req["attention"]):
                S = d["statics"]
                twin, ns = SC.ATTENTION[d["base"]]
                o, _nf = twin(q[a + i].reshape(NH, D), kc[:S["T"]], vc[:S["T"]], NS=ns(S), BN=S["BN"], masked_from=S["MASKED_FROM"],
                              dot_prim=S["DOT"], inv_prim=S["INV"])
                out[a + i] = o.reshape(-1)
        vals[ATT] = out
        run(POST)
        ends = [b - 1 for _r, _a, b in rows]
        d = plan["logits"]
        vals[SV.LOGITS] = SC.GEMMS[d["base"]](d["statics"], vals[OUT][ends], W["lm_head"])
        vals[SV.SAMPLED] = np.array([token_select_first_max(x) for x in vals[SV.LOGITS]], np.int64)[:, None]
        last.update({r: int(s) for (r, _a, _b), s in zip(rows, vals[SV.SAMPLED][:, 0])})
        steps.append(SC.Step(plan["step"], rows, vals))
    return steps, {r: list(map(int, p)) for r, p in prompts.items()}


@pytest.fixture(scope="module")
def served():
    build, W = _build(), _weights()
    steps, prompts = _serve(build, W)
    return build, W, steps, prompts


def _with(steps, step: int, name: str, f) -> list[SC.Step]:
    out = list(steps)
    s = out[step]
    vals = dict(s.values)
    vals[name] = f(vals[name].copy())
    out[step] = SC.Step(s.step, s.rows, vals)
    return out


def _failed(rep) -> set[tuple[int, str]]:
    return {(f["step"], f["value"]) for f in rep["failures"]}


def test_the_plans_select_every_definition(served):
    build = served[0]
    att = {d["base"] for p in build["steps_plan"] for r in p["requests"] for d in r["attention"]}
    gemm = {d["base"] for p in build["steps_plan"] for d in [*p["gemms"].values(), p["logits"]]}
    assert att == {F3.AttentionV4.id, IR.Fa3AttentionSplit.id}
    assert gemm == {b1.GemmV2.id, cublas_bf16.CublasSplitKGemmWord.id}
    assert [p["mode"] for p in build["steps_plan"]] == ["PIECEWISE", "PIECEWISE", "NONE", "FULL"]


def _route_apart(served, step: int) -> tuple[dict, list[SC.Step]]:
    """The run with a row of the qkv GEMM at `step` that the unsplit and the split-K sums round apart (random rows rarely do):
    1 + 2^-8 in the first 16-chunk, 2^-24 in each of the third and fourth, against a weight row of ones there."""
    build, W, steps, _prompts = served
    x: np.ndarray = np.zeros(H, np.float32)
    w: np.ndarray = np.zeros(H, np.float32)
    x[[0, 1, 32, 48]], w[[0, 1, 32, 48]] = [1, 2 ** -8, 2 ** -24, 2 ** -24], 1
    W = {**W, "qkv": W["qkv"].copy()}
    W["qkv"][0] = _words(w)
    steps = _with(steps, step, N0, lambda a: np.vstack([_words(x), a[1:]]))
    s = steps[step]
    d = build["steps_plan"][step]["gemms"][f"{H},{QKV}"]
    steps[step] = SC.Step(s.step, s.rows, {**s.values, QKVV: SC.GEMMS[d["base"]](d["statics"], s.values[N0], W["qkv"])})
    return W, steps


@pytest.mark.parametrize("route", [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 2},
                                   {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 2},
                                   {"kernel": "gemv", "LANES": 16}, {"kernel": "gemv", "LANES": 32}])
def test_each_cublas_route_binds_a_definition_whose_twin_is_the_routes(route):
    """A device pin's route, the Definition `gemm_definition` binds for it and that Definition's twin here compute one function
    (`bf16_gemm_check.cublas_route_rows`) on the route probe's spread and cancelling rows."""
    d = SV.gemm_definition(route, 256, 5, H200.dot)
    for x, w in BC.route_probe_rows(3, 256, 5, np.random.default_rng(3)):
        assert np.array_equal(SC.GEMMS[d["base"]](d["statics"], x, w), BC.cublas_route_rows(route, x, w))


def test_steps_are_checked_in_order(served):
    c = SC.ServedCheck(served[0], served[1].__getitem__, seed=None)
    c.step(served[2][1])
    with pytest.raises(ValueError, match="checked in order"):
        c.step(served[2][0])


