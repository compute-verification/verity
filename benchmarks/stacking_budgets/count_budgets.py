"""Honest operation budgets per byte of KV written, counted from the circuits Verity builds for served decoders.

For each model and shape, the Calls of one forward step are the ones `verity_vllm.predict.build` emits (`_Specs`: the
Definitions the Build binds on the declared target), and each Call's primitives are counted by walking its body
(`batch` and `scan` multiply by their length). A provenance pass marks every collection constant, weight (fixed before
any seed: the checkpoint and its derived tables), selected (a weight read through a gather whose index is a runtime value:
MoE experts, embedding rows) or runtime, so a product is classed as runtime x runtime, runtime x selected or runtime x
static. Every Call is one row's, so a token's counts do not depend on the batch it is scheduled in.

    uv run python benchmarks/stacking_budgets/count_budgets.py --out budgets.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided

CONST, WEIGHT, SEL, RUN = 0, 1, 2, 3

LLAMA31 = {"architectures": ["LlamaForCausalLM"], "attention_bias": False, "hidden_act": "silu", "max_position_embeddings": 131072,
           "mlp_bias": False, "model_type": "llama", "rms_norm_eps": 1e-05, "num_key_value_heads": 8, "head_dim": 128,
           "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
                            "rope_type": "llama3"},
           "rope_theta": 500000.0, "tie_word_embeddings": False, "torch_dtype": "bfloat16", "vocab_size": 128256}
#: meta-llama/Llama-3.1-8B and -70B config.json (the fields the vocabulary reads)
MODELS = {
    "llama31-8b": dict(LLAMA31, hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_hidden_layers=32),
    "llama31-70b": dict(LLAMA31, hidden_size=8192, intermediate_size=28672, num_attention_heads=64, num_hidden_layers=80),
}
MOE_CONFIGS = {"qwen3-30b-a3b": "QWEN3_30B_A3B.config.json"}

_DOT_WIDTH = {"AmpereBF16TcDot16": 16, "HopperBF16WgmmaDot16": 16, "HopperE4m3QgmmaDot32": 32, "HopperE4m3MmaV2Dot32": 32,
              "BlackwellE4m3QmmaDot32": 32, "BlackwellE5m2QmmaDot32": 32, "BlackwellNvf4OmmaDot64": 64, "BlackwellMxf4OmmaDot64": 64}
_CASTS = {"F2fpBf16", "F32ToBf16Rn", "F32ToE4m3Sat", "F32ToE2m1Sat"}
_WIDEN = {"Bf16ToF32", "E4m3ToF32", "F32Fabs", "F32Neg", "F32BitsShl23"}
_ACTS = {"MufuEx2Ftz", "MufuRcpFtz", "MufuSqrtFtz", "MufuTanh", "RsqrtApprox", "DivFullRcp", "Fa2InvSum"}
_CMPS = {"F32Max", "F32GtStrict", "F32Fmaxf", "F32Fminf", "Bf16GtStrict", "F32Eq", "I32Le", "I32Eq", "F32IsFinite", "GuardNegInfZero", "F32Sat",
         "SelectBf16", "SelectF32", "SelectI32", "DivFullScaleA"}
_INT = {"I32Add", "BitAnd", "BitNot", "BitOr"}
_ADDS = {"F32Add", "F32AddFtz", "F32Sub", "F32SubFtz"}
_MULS = {"F32Mul", "F32MulFtz"}
_FMAS = {"F32Fma", "F32FmaFtz", "F32FmaSubFtz", "F32FmaRm"}


def _colls(refs) -> set:
    if isinstance(refs, (Affine, Strided)):
        return {(refs.space, refs.idx)}
    if isinstance(refs, Concat):
        out = set()
        for p in refs.parts:
            out |= _colls(p)
        return out
    if isinstance(refs, Explicit):
        return {(r[0], r[1]) for r in refs.items}
    raise TypeError(type(refs))


def _base(pid: str) -> str:
    name = pid.split("{")[0].split("[")[0]
    return name.rsplit("_v", 1)[0]


_memo: dict = {}


def count(fn, taints: tuple) -> dict:
    """{(primitive id, operand provenance): instances} of one call of `fn` with parameters of provenance `taints`."""
    return _count(fn, taints)[0]


def _count(fn, taints: tuple) -> tuple[dict, int]:
    """(census, provenance of the output) of one call of `fn`.  A gather with a runtime index over static operands outputs a
    selected weight; every other primitive outputs the max of its operands' provenance, a selected operand counting as
    runtime (a selected weight stays selected only through wiring, up to its first arithmetic use)."""
    if isinstance(fn, PrimitiveDefinition):
        if _base(fn.id).startswith("Gather") and taints and taints[0] == RUN and max(taints[1:], default=CONST) <= WEIGHT:
            return {(fn.id, taints): 1}, SEL
        t = max(taints, default=CONST)
        return {(fn.id, taints): 1}, RUN if t >= SEL else t
    key = (fn.id, taints)
    hit = _memo.get(key)
    if hit is not None:
        return hit
    body = fn.body
    out: collections.Counter = collections.Counter()
    node_t: list[int] = []

    def tof(refs) -> int:
        t = CONST
        for s, i in _colls(refs):
            t = max(t, taints[i] if s == "p" else node_t[i])
        return t

    for nd in body.nodes:
        ats = [tof(a.refs) for a in nd.args]
        if nd.form == "scan":
            ats[0] = max(ats)
        sub, ot = _count(nd.fn, tuple(ats))
        m = nd.n if nd.form in ("batch", "scan") else 1
        for k, v in sub.items():
            out[k] += m * v
        node_t.append(max(ot, ats[0]) if nd.form == "scan" else ot)
    _memo[key] = (dict(out), tof(body.ret))
    return _memo[key]


CLASSES = ("cast", "act", "act_ex2", "f32_arith", "prod_rr", "prod_rsel", "prod_rs", "tc_acc_rr", "tc_acc_rsel", "tc_acc_rs", "cmp", "gather", "gather_fanin",
           "int", "unknown")


def _prod(a: int, b: int) -> str:
    if min(a, b) >= SEL:
        return "prod_rsel" if (a, b) in ((RUN, SEL), (SEL, RUN)) else "prod_rr"
    return "prod_rs"


def classify(counts: dict) -> dict:
    """Atomic operation classes of a primitive census: casts (narrowing roundings), activations (MUFU and division-type
    special functions; `act_ex2` is the softmax exp's share), f32_arith (FP32 adds, subs and the add of an FMA on a runtime
    operand: each rounds to 24 bits), prod_rr / prod_rsel / prod_rs (scalar products of two runtime values / of a runtime
    value and a runtime-selected weight / of a runtime value and a constant or weight; tensor-core dots contribute their k
    products), tc_acc (tensor-core accumulator transitions, by the same operand classes), cmp (comparisons, max/min, selects), gather (data-dependent gathers,
    with their total fan-in), int (integer and bit ops)."""
    c = collections.Counter()
    unknown = collections.Counter()
    for (pid, ts), n in counts.items():
        b = _base(pid)
        anyrun = max(ts, default=CONST) >= SEL
        if b.startswith("Const") or not ts:
            continue
        if b in _DOT_WIDTH:
            k = _DOT_WIDTH[b]
            cls = _prod(ts[1], ts[2]) if max(ts[1], ts[2]) >= SEL else None
            if cls is not None:
                c[cls] += k * n
                c["tc_acc" + cls[4:]] += n
        elif not anyrun:
            continue
        elif b in _CASTS:
            c["cast"] += n
        elif b in _WIDEN:
            pass
        elif b in _ACTS:
            c["act"] += n
            if b == "MufuEx2Ftz":
                c["act_ex2"] += n
        elif b in _CMPS:
            c["cmp"] += n
        elif b in _INT:
            c["int"] += n
        elif b in _ADDS:
            c["f32_arith"] += n
        elif b in _MULS or b in _FMAS:
            c[_prod(ts[0], ts[1])] += n
            c["f32_arith"] += n
        elif b == "F32Div":
            # a / b: b runtime makes it a product with a runtime reciprocal
            c[_prod(ts[0], ts[1])] += n
            c["act" if ts[1] >= SEL else "f32_arith"] += n
        elif b in ("Bf16Add", "Bf16AddF2fp"):
            c["cast"] += n
        elif b in ("Bf16MulBf16", "Bf16MulF32"):
            c[_prod(ts[0], ts[1])] += n
            c["cast"] += n
        elif b == "SiluMulBf16":
            c["act"] += n
            c["cast"] += 2 * n
            c[_prod(ts[0], ts[1])] += n
        elif b in ("RopeOut", "RopeOutAdd"):
            c["prod_rr" if max(ts[2:]) >= SEL else "prod_rs"] += 2 * n
            c["f32_arith"] += n
            c["cast"] += n
        elif b.startswith("Gather"):
            c["gather"] += n
            fan = int("".join(ch for ch in pid.split("x", 1)[1].split("_")[0] if ch.isdigit()) or 0) if "x" in pid else 0
            c["gather_fanin"] += n * fan
        else:
            unknown[pid] += n
            c["unknown"] += n
    return {"classes": {k: int(c.get(k, 0)) for k in CLASSES}, "unknown": dict(unknown)}


def _add(acc: collections.Counter, d: dict, m: int = 1) -> None:
    for k, v in d.items():
        acc[k] += m * v


def target_profile(cc: tuple[int, int], sms: int):
    from verity_vllm.program.frontend.target_profile import TargetProfile
    return TargetProfile.default().with_overrides(compute_capability=cc, num_sms=sms)


class Counter:
    """One model's per-row Call censuses on one target (bf16, or FP8 decoder linears with the bf16 LM head)."""

    def __init__(self, hf: dict, tp, fp8: bool = False) -> None:
        from verity_vllm.predict.build import _Specs, model_shape
        from verity_vllm.program.frontend import targets as TG
        self.C = C = model_shape(hf)
        self.S = S = _Specs(C, tp, 1, 0, 1)
        self.tp, self.fp8, self.TG = tp, fp8, TG
        R, W = RUN, WEIGHT
        lin = {}
        for name, (K, N) in {"qkv": (C.H, C.QKV), "o": (C.NH * C.D, C.H), "gate_up": (C.H, 2 * C.I), "down": (C.I, C.H)}.items():
            if fp8:
                from verity_vllm.program.registry import fp8 as F
                from verity.ml.circuits.defs import bind
                quant = bind(F.Fp8ActQuant, K=K)
                mm = TG.scaled_mm_fp8_spec(tp, K, N, False)
                lin[name] = [(quant, (R,)), (mm, (R, R, W, W))]
            else:
                lin[name] = [(getattr(S, name), (R, W))]
        self.lin = lin
        self.parts = {}
        self.parts["embedding"] = [(S.embedding, (R, W))]
        self.parts["norm0"] = [(S.norm0, (R, W))]
        self.parts["fused_norm"] = [(S.fused, (R, R, W))]
        self.parts["rope"] = [(S.rope_q, (R, W)), (S.rope_k, (R, W))]
        self.parts["silu_mul"] = [(S.silu, (R,))]
        self.parts["logits"] = [(S.logits, (R, W))]
        self.parts["token_select"] = [(S.token_select, (R,))]
        if C.qk_norm == "head":
            self.parts["qk_norm"] = [(S.head_norm, (R, W))] * (C.NH + C.KVH)
        if C.E:
            self.parts["router"] = [(S.router, (R, W))]
            self.parts["topk"] = [(S.topk, (R,))]
            self.parts["expert_up"] = [(S.expert_up, (R, R, W))] * C.TOPK
            self.parts["expert_down"] = [(S.expert_down, (R, R, R, W))] * C.TOPK
            self.parts["moe_sum"] = [(S.moe_sum, (R,))]
            self.parts["silu_mul"] = [(S.silu, (R,))] * C.TOPK
        self._part_cache: dict = {}

    def part(self, name: str) -> dict:
        if name not in self._part_cache:
            acc = collections.Counter()
            calls = self.lin[name] if name in self.lin else self.parts[name]
            for fn, ts in calls:
                _add(acc, count(fn, ts))
            self._part_cache[name] = dict(acc)
        return self._part_cache[name]

    def attention(self, T: int, row: int = 0, own_rows: int = 1, launch_sq: int = 1) -> dict:
        return count(self.S.attention(T, row, own_rows, launch_sq), (RUN, RUN, RUN))

    def layer_static(self) -> dict:
        """One layer's per-row primitives other than attention, with two fused norms (NL of these blocks carry the 2 NL fused norms
        of a forward: every input norm but layer 0's, every post-attention norm, and the final norm)."""
        acc = collections.Counter()
        C = self.C
        names = ["fused_norm", "qkv", "rope", "o", "fused_norm"]
        if C.qk_norm == "head":
            names.append("qk_norm")
        names += ["router", "topk", "expert_up", "silu_mul", "expert_down", "moe_sum"] if C.E else ["gate_up", "silu_mul", "down"]
        for n in names:
            _add(acc, self.part(n))
        return dict(acc)

    def kv_bytes_per_token(self, kv_bytes: int) -> int:
        C = self.C
        return C.NL * 2 * C.KVH * C.D * kv_bytes


def attention_closed_form(ctr: Counter, Ts: list[int]) -> tuple[dict, int]:
    """Exact attention censuses for every T in Ts from a verified per-block increment: counts(T) = counts(T0) + j * delta for
    T = T0 + j * BN, T0 in (BN, 2 BN].  Verified exactly at sampled T against the directly counted Definition."""
    BN = ctr.S.attention(3 * 128, 0, 1, 1).statics.get("BN", 64)
    base = {}
    for T0 in range(BN + 1, 2 * BN + 1):
        base[T0] = ctr.attention(T0)
    d1, d2 = ctr.attention(2 * BN + 1), ctr.attention(3 * BN + 1)
    delta = {k: d2.get(k, 0) - d1.get(k, 0) for k in set(d1) | set(d2)}
    check = {k: d1.get(k, 0) - base[BN + 1].get(k, 0) for k in set(d1) | set(base[BN + 1])}
    assert all(delta.get(k, 0) == check.get(k, 0) for k in set(delta) | set(check)), "per-block increment is not constant"

    def at(T: int) -> dict:
        if T <= 2 * BN:
            return ctr.attention(T)
        j, r = divmod(T - BN - 1, BN)
        T0 = BN + 1 + r
        return {k: base[T0].get(k, 0) + j * delta.get(k, 0) for k in set(base[T0]) | set(delta)}

    for T in (5 * BN + 7, 9 * BN, 17 * BN + 63):
        direct = ctr.attention(T)
        cf = at(T)
        assert all(direct.get(k, 0) == cf.get(k, 0) for k in set(direct) | set(cf)), f"closed form disagrees at T={T}"
    return at, BN


def budgets(model: str, hf: dict, tp, fp8: bool, decode_T: list[int], prefill_LP: list[int], log=print) -> dict:
    t0 = time.time()
    ctr = Counter(hf, tp, fp8)
    C = ctr.C
    at, BN = attention_closed_form(ctr, decode_T)
    layer = ctr.layer_static()
    head = collections.Counter()
    _add(head, ctr.part("embedding"))
    _add(head, ctr.part("norm0"))  # layer 0's input norm; NL x layer_static's 2 NL fused norms are the other inputs, the post-attention ones and the final
    sample = collections.Counter()
    _add(sample, ctr.part("logits"))
    _add(sample, ctr.part("token_select"))
    out = {"model": model, "fp8_linears": fp8, "target": tp.to_json(), "shape": {k: getattr(C, k) for k in
                                                                                ("H", "V", "NL", "NH", "KVH", "D", "I", "E", "TOPK")},
           "attention_BN": BN, "components": {}, "decode": {}, "prefill": {}}
    comp = {"layer_static_x_NL": classify({k: v * C.NL for k, v in layer.items()})["classes"],
            "embed_and_norms": classify(dict(head))["classes"], "lm_head_and_select": classify(dict(sample))["classes"]}
    for name in ctr.parts.keys() | ctr.lin.keys():
        comp["part:" + name] = classify(ctr.part(name))["classes"]
    out["components"] = comp
    unknown = {}
    for T in decode_T:
        acc = collections.Counter()
        _add(acc, layer, C.NL)
        _add(acc, at(T), C.NL)
        _add(acc, head)
        _add(acc, sample)
        cl = classify(dict(acc))
        unknown.update(cl["unknown"])
        att = classify({k: v * C.NL for k, v in at(T).items()})["classes"]
        out["decode"][str(T)] = {"per_token": cl["classes"], "attention_per_token": att}
    for LP in prefill_LP:
        acc = collections.Counter()
        _add(acc, layer, C.NL * LP)
        _add(acc, head, LP)
        attsum = collections.Counter()
        for T in range(1, LP + 1):
            _add(attsum, at(T))
        _add(acc, attsum, C.NL)
        _add(acc, sample)  # the last row's sample
        cl = classify(dict(acc))
        unknown.update(cl["unknown"])
        att = classify({k: v * C.NL for k, v in attsum.items()})["classes"]
        out["prefill"][str(LP)] = {"per_request": cl["classes"], "attention_per_request": att, "tokens": LP}
    out["unknown_primitives"] = unknown
    out["seconds"] = round(time.time() - t0, 1)
    log(f"{model} fp8={fp8}: {out['seconds']} s, unknown {sorted(unknown)[:8]}")
    return out


def validate(hf: dict, tp, log=print) -> bool:
    """The per-row assembly `budgets` uses equals the census of the whole Program `predict.build` emits: a prefill step of M
    rows, and a request of LP prompt rows then T decode steps (each decode row's attention at its own history)."""
    from verity_vllm.predict.build import build_program
    from verity_vllm.predict.phi import Phi
    ctr = Counter(hf, tp)
    C, layer = ctr.C, ctr.layer_static()
    phi = Phi(model={"repo": "-", "revision": "-", "hf_config": hf}, target=tp.to_json())
    ok = True
    for wrapper, M, LP, T in (("step", 1, None, None), ("step", 4, None, None), ("request", None, 3, 2)):
        r = build_program(phi, wrapper, M=M or 1, LP=LP, T=T, with_desc=False)
        fn = r.program.fn
        taints = tuple(WEIGHT if name == "weights" else RUN for name, _ in fn.body.params)
        direct = classify(count(fn, taints))["classes"]
        acc = collections.Counter()
        rows = M if wrapper == "step" else LP + T
        _add(acc, layer, C.NL * rows)
        for n in ("embedding", "norm0"):
            _add(acc, ctr.part(n), rows)
        n0 = M if wrapper == "step" else LP
        for i in range(n0):
            _add(acc, count(ctr.S.attention(i + 1, i, n0, n0), (RUN,) * 3), C.NL)
        for s in range(T or 0):
            _add(acc, ctr.attention(LP + s + 1), C.NL)
        for _ in range(1 if wrapper == "step" else T + 1):
            _add(acc, ctr.part("logits"))
            _add(acc, ctr.part("token_select"))
        asm = classify(dict(acc))["classes"]
        diff = {k: (direct[k], asm[k]) for k in direct if direct[k] != asm[k]}
        log(f"validate {wrapper} M={M} LP={LP} T={T}: Program census {'==' if not diff else '!='} per-row assembly {diff or ''}")
        ok &= not diff
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama31-8b,llama31-70b,qwen3-30b-a3b")
    ap.add_argument("--decode-T", default="512,2048,8192,32768,131072")
    ap.add_argument("--prefill-LP", default="512,2048,8192,32768")
    ap.add_argument("--cc", default="12.0", help="compute capability of the target (12.0: RTX PRO 6000)")
    ap.add_argument("--sms", type=int, default=188)
    ap.add_argument("--validate", default="", help="comma-separated hf_configs files to validate the assembly on first")
    a = ap.parse_args(argv)
    cc = tuple(int(x) for x in a.cc.split("."))
    tp = target_profile(cc, a.sms)
    cfgdir = Path(__file__).resolve().parents[2] / "integrations" / "vllm" / "data" / "hf_configs"
    for f in filter(None, a.validate.split(",")):
        if not validate(json.loads((cfgdir / f).read_text()), tp):
            return 1
    dT = [int(x) for x in a.decode_T.split(",")]
    pL = [int(x) for x in a.prefill_LP.split(",")]
    results = []
    for m in a.models.split(","):
        hf = MODELS[m] if m in MODELS else json.loads((cfgdir / MOE_CONFIGS[m]).read_text())
        variants = [False, True] if m == "llama31-70b" else [False]
        for fp8 in variants:
            results.append(budgets(m, hf, tp, fp8, dT, pL))
            Path(a.out).write_text(json.dumps({"schema": "stacking-budgets/v1", "results": results}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
