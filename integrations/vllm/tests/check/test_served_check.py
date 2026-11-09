"""check.replay.served_check on a small served Program built by hand: one layer of the Build's form (the Embedding, Inductor's
RMSNorm, AddRMSNorm, Add2RMSNorm, SiLU-and-mul and RoPE, cuBLAS GEMMs at both routes, FA3 unsplit and split, the logits and
TokenSelect) over four engine steps of three requests, whose step plans `acquire.served.step_plans` derives.  The committed
values are what the Program computes, step after step (each decode step reads the token the previous step sampled), and the
check passes them; each single perturbation fails by name."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from verity_vllm.acquire import served as SV
from verity_vllm.acquire import served_device as SD
from verity_vllm.check import c2_rules
from verity_vllm.check.replay import served_check as SC
from verity.ml.definitions.numerics.rows import token_select_first_max
from verity.ml.definitions.registry import b1, cublas_bf16
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import inductor as IR
from verity_vllm.query.manifest.format import manifest_digest

V, H, NH, KVH, D, I, P = 64, 64, 2, 1, 64, 32, 512
QKV = (NH + 2 * KVH) * D
NORM = dict(N=H, EPS=1e-5, XBLOCK=1, R0_BLOCK=256, NUM_WARPS=1)
SCHEDULE = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 4},
            {"request_id": "r1", "arrive": 1, "prompt_tokens": 2, "max_tokens": 3},
            {"request_id": "r2", "arrive": 2, "prompt_tokens": 200, "max_tokens": 2}]
STEPS = 4
H200 = SD.device_of((9, 0), 132)

EMB, N0, QKVV, Q, K, ATT, O, N1, GU, ACT, DOWN, RES, OUT = (
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
ATTENTION = {"piece": 1, "layer": "layers.0.attn", "name": ATT, "q": _v(Q, NH * D), "k": _v(K, KVH * D),
             "v": _v(QKVV, KVH * D, ((NH + KVH) * D, QKV)), "NH": NH, "KVH": KVH, "D": D}
POST = [_mm(_v(ATT, NH * D), _w("o", H, NH * D), O),
        _call(IR.InductorAddRMSNorm, NORM, [_v(O, H), _v(EMB, H), _w("norm1", H)], [N1]),
        _mm(_v(N1, H), _w("gate_up", 2 * I, H), GU),
        _call(IR.InductorSiluMul, dict(I=I), [_v(GU, 2 * I)], [ACT]),
        _mm(_v(ACT, I), _w("down", H, I), DOWN),
        _call(IR.InductorAdd2RMSNorm, NORM, [_v(DOWN, H), _v(O, H), _v(EMB, H), _w("norm2", H)], [RES, OUT])]
WIDTHS = {SV.IDS: 1, SV.POSITIONS: 1, EMB: H, N0: H, QKVV: QKV, Q: NH * D, K: KVH * D, ATT: NH * D, O: H, N1: H, GU: 2 * I,
          ACT: I, DOWN: H, RES: H, OUT: H}


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
         "gate_up": _bf16(rng, 2 * I, H, scale=H ** -0.5), "down": _bf16(rng, H, I, scale=I ** -0.5),
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

        def run(calls):
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
        out = np.zeros((lo, NH * D), np.uint16)
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


def _check(served, steps=None, build=None, weight=None):
    b, W, st, prompts = served
    return SC.check(build or b, st if steps is None else steps, weight or W.__getitem__, seed=None, rows=8, prompts=prompts)


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


def test_the_served_values_pass(served):
    rep = _check(served)
    assert rep["ok"], rep["failures"]
    assert rep["steps_checked"] == [0, 1, 2, 3] and not rep["steps_unrun"]
    kinds = {IR.InductorRMSNorm.id, IR.InductorAddRMSNorm.id, IR.InductorAdd2RMSNorm.id, IR.InductorSiluMul.id, IR.InductorRoPE.id,
             b1.Embedding.id, b1.GemmV2.id, cublas_bf16.CublasSplitKGemmWord.id, b1.TokenSelect.id, "attention", "coverage", "batch",
             SV.IDS, SV.POSITIONS}
    assert kinds <= set(rep["tally"])
    assert all(ok == n for ok, n in rep["tally"].values())
    defs = rep["steps"]["3"][ATT]["definitions"]
    assert defs == {F3.AttentionV4.id: 2, IR.Fa3AttentionSplit.id: 1}


@pytest.mark.parametrize("name,row", [(N0, 0), (Q, 2), (ACT, 1), (RES, 2), (OUT, 0), (EMB, 1)])
def test_a_flipped_word_fails_its_value(served, name, row):
    def flip(a):
        a[row, 3] ^= 1
        return a
    rep = _check(served, _with(served[2], 1, name, flip))
    assert not rep["ok"]
    assert (1, name) in _failed(rep)


def test_a_flipped_gemm_word_on_a_sampled_row_fails(served):
    def flip(a):
        a[0, 5] ^= 1
        return a
    assert (2, DOWN) in _failed(_check(served, _with(served[2], 2, DOWN, flip)))


def _route_apart(served, step: int) -> tuple[dict, list[SC.Step]]:
    """The run with a row of the qkv GEMM at `step` that the unsplit and the split-K sums round apart (random rows rarely do):
    1 + 2^-8 in the first 16-chunk, 2^-24 in each of the third and fourth, against a weight row of ones there."""
    build, W, steps, _prompts = served
    x, w = np.zeros(H, np.float32), np.zeros(H, np.float32)
    x[[0, 1, 32, 48]], w[[0, 1, 32, 48]] = [1, 2 ** -8, 2 ** -24, 2 ** -24], 1
    W = {**W, "qkv": W["qkv"].copy()}
    W["qkv"][0] = _words(w)
    steps = _with(steps, step, N0, lambda a: np.vstack([_words(x), a[1:]]))
    s = steps[step]
    d = build["steps_plan"][step]["gemms"][f"{H},{QKV}"]
    steps[step] = SC.Step(s.step, s.rows, {**s.values, QKVV: SC.GEMMS[d["base"]](d["statics"], s.values[N0], W["qkv"])})
    return W, steps


@pytest.mark.parametrize("step", [1, 2])
def test_the_gemm_is_the_plans_cublas_route(served, step):
    W, steps = _route_apart(served, step)
    assert (step, QKVV) not in _failed(_check(served, steps, weight=W.__getitem__))
    build = copy.deepcopy(served[0])
    plan = build["steps_plan"][step]
    other = {"kernel": "nosplit"} if _route(plan["gemm_m"])["kernel"] == "splitk" else {"kernel": "splitk", "SPLIT_K": 2}
    plan["gemms"][f"{H},{QKV}"] = SV.gemm_definition(other, H, QKV, H200.dot)
    assert (step, QKVV) in _failed(_check(served, steps, build=build, weight=W.__getitem__))


@pytest.mark.parametrize("route", [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 2},
                                   {"kernel": "splitk", "SPLIT_K": 2, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 2},
                                   {"kernel": "gemv", "LANES": 16}, {"kernel": "gemv", "LANES": 32}])
def test_each_cublas_route_binds_a_definition_whose_twin_is_the_routes(route):
    """A device pin's route, the Definition `gemm_definition` binds for it and that Definition's twin here compute one function
    (`bf16_gemm_check.cublas_route_rows`) on the route probe's spread and cancelling rows."""
    d = SV.gemm_definition(route, 256, 5, H200.dot)
    for x, w in BC.route_probe_rows(3, 256, 5, np.random.default_rng(3)):
        assert np.array_equal(SC.GEMMS[d["base"]](d["statics"], x, w), BC.cublas_route_rows(route, x, w))


def test_a_skipped_step_leaves_the_cache_stale(served):
    steps = served[2]
    rep = _check(served, [steps[0], steps[2], steps[3]])
    assert rep["steps_unrun"] == [1]
    att = rep["steps"]["2"][ATT]
    assert not att["ok"] and any("cached keys" in w for w in att["why"])


def test_a_decode_step_reads_the_token_sampled_before(served):
    rid_row = next(lo for r, lo, _hi in served[2][3].rows if r == "r1")

    def other(a):
        a[rid_row, 0] = (a[rid_row, 0] + 1) % V
        return a
    rep = _check(served, _with(served[2], 3, SV.IDS, other))
    assert (3, SV.IDS) in _failed(rep)


def test_the_sampled_id_is_token_select(served):
    def other(a):
        a[0, 0] = (a[0, 0] + 1) % V
        return a
    assert (0, SV.SAMPLED) in _failed(_check(served, _with(served[2], 0, SV.SAMPLED, other)))


def test_a_missing_value_fails_coverage_and_its_readers(served):
    steps = list(served[2])
    s = steps[2]
    steps[2] = SC.Step(s.step, s.rows, {k: v for k, v in s.values.items() if k != ACT})
    rep = _check(served, steps)
    assert rep["steps"]["2"]["coverage"]["missing"] == [ACT]
    assert "not committed" in rep["steps"]["2"][DOWN]["why"]


def test_the_batch_must_be_the_plans(served):
    steps = list(served[2])
    s = steps[1]
    steps[1] = SC.Step(s.step, [(r, lo, hi) for r, lo, hi in s.rows if r != "r1"], s.values)
    rep = _check(served, steps)
    assert not rep["steps"]["1"]["batch"]["ok"]


def test_a_weight_of_another_shape_is_refused(served):
    W = served[1]
    rep = _check(served, weight=lambda n: W[n][:-1] if n == "down" else W[n])
    assert all((s, DOWN) in _failed(rep) for s in range(STEPS))


def test_a_seed_draws_the_sampled_rows(served):
    build, W, steps, prompts = served
    rep = SC.check(build, steps, W.__getitem__, rows=8, prompts=prompts, seed=5)
    assert rep["ok"], rep["failures"]
    assert rep["seed"] == 5
    drawn = rep["picks"]["2"]["r2"]
    assert drawn == SC.check(build, steps, W.__getitem__, rows=8, prompts=prompts, seed=5)["picks"]["2"]["r2"]
    assert drawn != SC.check(build, steps, W.__getitem__, rows=8, prompts=prompts, seed=6)["picks"]["2"]["r2"]
    assert len(drawn) == 8 and drawn != list(SC.sample(200, 8))
    assert rep["picks"]["2"]["r0"] == [0] and rep["steps"]["2"][ATT]["rows"] == 8 + 2
    lo = next(a for r, a, _b in steps[2].rows if r == "r2")
    undrawn = next(i for i in range(200) if i not in drawn)

    def flip(i):
        def f(a):
            a[lo + i, 5] ^= 1
            return a
        return _with(steps, 2, DOWN, f)
    assert (2, DOWN) in _failed(SC.check(build, flip(drawn[3]), W.__getitem__, rows=8, prompts=prompts, seed=5))
    assert (2, DOWN) not in _failed(SC.check(build, flip(undrawn), W.__getitem__, rows=8, prompts=prompts, seed=5))


def test_steps_are_checked_in_order(served):
    c = SC.ServedCheck(served[0], served[1].__getitem__, seed=None)
    c.step(served[2][1])
    with pytest.raises(ValueError, match="checked in order"):
        c.step(served[2][0])


def _population(served, rep=None, build=None, ranks=(0,)):
    pop = SC.population(build or served[0], rep or _check(served), ranks=list(ranks))
    return pop, c2_rules._query_population_scope({"population": pop})


def test_the_check_reaches_every_identity_of_the_build(served):
    build = served[0]
    pop, (scope, fail) = _population(served)
    total = sum(len(p["requests"]) for p in build["steps_plan"]) * (len(WIDTHS) + 2)
    assert pop["query_checks"]["ok"] and fail is None
    assert pop["query_total"]["identities"] == len(build["identities"]) == total
    assert pop["query_total"]["manifest_digest"] == build["manifest_digest"]
    assert f"reaches {total} (100.0%)" in scope and SV.KIND in scope


def test_a_step_the_check_never_reached_leaves_its_identities_without_rows(served):
    steps = served[2]
    pop, (_scope, fail) = _population(served, _check(served, [steps[0], steps[2], steps[3]]))
    c = pop["query_checks"]
    assert not c["ok"] and c["accounted"]
    # step 1's every identity, and r1's ids at step 2: no id sampled before to compare them to
    assert c["identities_without_rows"]["n"] == len(served[0]["steps_plan"][1]["requests"]) * (len(WIDTHS) + 2) + 1
    assert "identities_without_rows" in fail


def test_ids_compared_to_nothing_are_not_reached(served):
    build, W, steps, prompts = served
    rep = SC.check(build, steps, W.__getitem__, seed=None, rows=8, prompts={r: p for r, p in prompts.items() if r != "r2"})
    pop, _ = _population(served, rep)
    assert pop["query_checks"]["identities_without_rows"]["first"] == [
        {"request_id": "r2", "step": 2, "op_path": "c00.input", "member": "ids"}]


def test_an_identity_of_other_rows_disagrees(served):
    build = copy.deepcopy(served[0])
    i = next(i for i in build["identities"] if i["op_path"] == "c02.mm" and i["output_member"] == "buf4" and i["engine_step"] == 2)
    i["element_range"] = [0, i["element_range"][1] + H]
    c = _population(served, build=build)[0]["query_checks"]
    assert not c["ok"] and c["rows_disagree"]["n"] == 1


def test_a_value_the_build_does_not_require_is_outside_the_query(served):
    build = copy.deepcopy(served[0])
    build["identities"] = [i for i in build["identities"] if not (i["op_path"] == "logits_processor" and i["engine_step"] == 3)]
    c = _population(served, build=build)[0]["query_checks"]
    assert not c["ok"] and c["vus_outside_query"]["n"] == len(build["steps_plan"][3]["requests"])


def test_another_rank_disagrees(served):
    pop, (_scope, fail) = _population(served, ranks=(1,))
    assert pop["query_checks"]["ranks"] == {"replay": [1], "manifest": [0], "equal": False} and "ranks replay [1]" in fail
