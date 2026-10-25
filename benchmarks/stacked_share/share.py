"""The stacked share f of a node's live KV: the carrier stock an honest budget can hold, over the node's KV pool.

    stock(L) = 128 R_cap L / E + R_in L        bytes
    f        = stock(L) / M_dyn

This is `Security/Models/Accounting/Guarantees/Stacking.lean`'s C = 8 R_in life + 1024 (c_rho reads) life / E in bits, read in
bytes: c_rho reads, the budgeted units the audit allows a second, is the honest Program's rate R_cap; E is the units an encode of
one 1024-bit word costs (E / 128 a byte); L is a carrier's lifetime in seconds and R_in the node's run-input bytes a second. f
past 1 means the stock outgrows the pool: no KV is credited.

R_cap comes from `units.py`'s units for one model and a served trace (`benchmarks/serve_bandwidth/mix.py`'s mix.json). A step's
units are

    token T + sample X + A (r0 T + b_q Q + b_kv K + w0 n + w_start n0)

with T its token rows, X its sampled rows, K its requests' keys (mix.json's kv_tokens: each scheduled request's context after the
step), n its requests, n0 its prefills from position 0, Q its token rows' keys and A the attention calls a request makes a step
(a layer each, on every rank).
Over the trace the totals are exact. T and K are measured; X is the output tokens; Q is the sum over the trace's requests of
P (P + 1) / 2 + the sum over its decodes t = 1 .. O - 1 of P + t (a prompt token at position i reads i keys however vLLM chunks
the prompt), on the trace's own requests (`benchmarks/kv_filler/serve.py`'s `trace`); n is the decoding requests a step
(measured) plus one a request's prefill (n0: one a request), plus a bound on the prompts vLLM split over two steps (`exact`: K's excess over the keys
one prefill a prompt reads is those splits' rereads, which it holds to the bound). A step's own Q and n with prefill aren't
measured: `steps` spreads the trace's prompt-token keys and prefill calls over its prompt tokens evenly, which sums to the exact
totals.

The rate is units over the trace's busy time (the union of its steps' spans) and, as the honest peak, the largest over windows of
`width` seconds, each step's units spread over its span. A node runs gpus / tp engines, each serving the trace.

M_dyn is vLLM's KV pool: kv_blocks blocks of `block` tokens of kv_bytes_per_token bytes on each of an engine's tp GPUs.

    python share.py --model 8B=units-8b.json,mix.json --model 70B=units-70b.json,mix-70b.json --out share.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

#: E, the units an encode of one 1024-bit word costs, each over 3 rounds of 1,021 squarings: proved (8 units a squaring; counting
#: proves 8 to 24), conjectured (EncodeChainHardness at 2^-40), and the red team's cheapest construction (160 a squaring under
#: #2420, 128 while hole B is open)
E_WORD = {"proved": 8 * 3 * 1021, "conjectured": 364_497, "red team (#2420)": 160 * 3 * 1021,
          "red team, hole B open": 128 * 3 * 1021}
LIFETIMES = {"10 min": 600, "1 h": 3600}
R_INS = (100_000, 1_000_000)
BLOCK = 16
WIDTH = 1.0
CLASSES = ("roundings", "compare_select", "gathers", "folds")
COEFFICIENTS = ("r0", "b_q", "b_kv", "w0", "w_start")


def F(x: Any) -> Fraction:
    return Fraction(x)


def by_class(u: Mapping[str, Any]) -> dict[str, Fraction]:
    """Units by class (a class `CLASSES` doesn't list is refused)."""
    if set(u) - set(CLASSES):
        raise ValueError(f"units in classes {sorted(set(u) - set(CLASSES))} that property/v1 doesn't budget")
    return {c: F(u[c]) for c in CLASSES if u.get(c)}


def total(u: Mapping[str, Any]) -> Fraction:
    return sum((F(v) for v in u.values()), Fraction(0))


def scale(u: Mapping[str, Any], k: Any) -> dict[str, Fraction]:
    return {c: F(v) * F(k) for c, v in u.items()}


def add(*us: Mapping[str, Any]) -> dict[str, Fraction]:
    out: Counter = Counter()
    for u in us:
        for c, v in u.items():
            out[c] += F(v)
    return {c: v for c, v in out.items() if v}


def model_units(doc: Mapping[str, Any]) -> dict:
    """`units.py`'s document as the step model's parts: token, sample, the attention's coefficients and its calls a request-step."""
    att = doc["attention"]
    co = {c: {k: F(v) for k, v in att["coefficients"].get(c, {}).items()} for c in att["coefficients"]}
    return {"token": {k: F(v) for k, v in doc["token"].items()}, "sample": {k: F(v) for k, v in doc["sample"].items()},
            "coefficients": co, "calls": int(att["calls"])}


def step_units(m: Mapping[str, Any], *, T: Any, X: Any, Q: Any, K: Any, n: Any, n0: Any) -> dict[str, Fraction]:
    """token T + sample X + A (r0 T + b_q Q + b_kv K + w0 n + w_start n0), by key."""
    x = dict(zip(COEFFICIENTS, (F(T), F(Q), F(K), F(n), F(n0))))
    att = {c: m["calls"] * sum((co.get(name, Fraction(0)) * x[name] for name in COEFFICIENTS), Fraction(0))
           for c, co in m["coefficients"].items()}
    return add(scale(m["token"], T), scale(m["sample"], X), att)


def trace_requests(mix: Mapping[str, Any]) -> list[dict]:
    """The trace mix.json served: `serve.trace` again from its arguments (seeded, so the same requests)."""
    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here / "kv_filler"))
    from serve import trace  # noqa: E402

    a = mix["args"]
    reqs = trace(a["seed"], a["seconds"], a["burst_gap_s"], a["burst_mean"], a["prompts"], a["outputs"])
    t = mix["trace"]
    got = (len(reqs), sum(r["prompt"] for r in reqs), sum(r["output"] for r in reqs))
    if got != (t["requests"], t["prompt_tokens"], t["output_tokens"]):
        raise ValueError(f"serve.trace gives {got} (requests, prompt, output tokens), mix.json served {t}")
    return [{"prompt": r["prompt"], "output": r["output"]} for r in reqs]


MODEL = ("T", "X", "Q", "K", "n", "n0")


def exact(mix: Mapping[str, Any], requests: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """The trace's T, X, Q, K, n and n0 (see the module docstring), and K's excess over the keys the requests read with every prompt
    in one step (``split_keys``). The excess is the prompts vLLM split over steps: a later chunk's call reads the chunks before
    it again, and adds a prefill. A prompt shorter than the token budget splits only at the end of a step whose tokens filled the
    budget with a prefill to follow, at most once there (``splits``, which n counts: a bound), and its later chunk rereads fewer
    keys than the prompt has. A K the requests can't account for that way is refused."""
    steps = mix["steps"]
    T = sum(s["tokens"] for s in steps)
    K = sum(s["kv_tokens"] for s in steps)
    P = [r["prompt"] for r in requests]
    O = [r["output"] for r in requests]
    dec_keys = sum((o - 1) * p + (o - 1) * o // 2 for p, o in zip(P, O))
    Q = sum(p * (p + 1) // 2 for p in P) + dec_keys
    if T != sum(P) + sum(o - 1 for o in O):
        raise ValueError(f"the steps compute {T} token rows, the requests {sum(P)} prompt and {sum(o - 1 for o in O)} decode rows")
    budget = mix["args"]["max_batched_tokens"]
    if max(P) >= budget:
        raise ValueError(f"a prompt of {max(P)} tokens can split more than once under a budget of {budget}")
    splits = sum(1 for a, b in zip(steps, steps[1:]) if a["tokens"] == budget and a["prefill_tokens"] and b["prefill_tokens"])
    excess = K - sum(P) - dec_keys
    if not 0 <= excess <= splits * (max(P) - 1):
        raise ValueError(f"the steps' requests read {K} keys, one prefill a prompt reads {K - excess}: {splits} splits of prompts "
                         f"under {max(P)} tokens can't account for the difference")
    return {"T": T, "X": sum(O), "Q": Q, "K": K, "n": sum(s["decode_reqs"] for s in steps) + len(requests) + splits,
            "n0": len(requests), "split_keys": excess, "splits": splits}


def steps(mix: Mapping[str, Any], requests: Sequence[Mapping[str, int]], split_keys: int = 0,
          splits: int = 0) -> list[dict[str, Fraction]]:
    """Each step's T, X, Q, K, n and n0: measured, but for a step's prefill, whose keys, requests and splits are the trace's means
    over its prompt tokens."""
    P = [r["prompt"] for r in requests]
    q_pre = Fraction(sum(p * (p + 1) // 2 for p in P) - split_keys, sum(P))
    per_req = Fraction(len(P), sum(P))
    per_call = Fraction(len(P) + splits, sum(P))
    out = []
    for s in mix["steps"]:
        pre, dec = s["prefill_tokens"], s["decode_reqs"]
        out.append({"T": F(s["tokens"]), "X": dec + pre * per_req, "Q": s["kv_tokens"] - pre + pre * q_pre,
                    "K": F(s["kv_tokens"]), "n": dec + pre * per_call, "n0": pre * per_req})
    return out


def busy(mix: Mapping[str, Any]) -> float:
    """The union of the steps' spans, in seconds."""
    spans = sorted((s["t0"], s["t1"]) for s in mix["steps"])
    out, end = 0.0, float("-inf")
    for a, b in spans:
        if b > end:
            out += b - max(a, end)
            end = b
    return out


def peak(mix: Mapping[str, Any], amounts: Sequence[float], width: float = WIDTH) -> float:
    """The largest amount a second over windows of `width` seconds, each step's amount spread over its span."""
    t1 = max(s["t1"] for s in mix["steps"])
    n = max(1, int(t1 // width) + 1)
    acc = [0.0] * n
    for s, x in zip(mix["steps"], amounts):
        a, b = s["t0"], s["t1"]
        if b <= a:
            acc[min(n - 1, int(a // width))] += x
            continue
        for i in range(int(a // width), min(n - 1, int(b // width)) + 1):
            ov = min(b, (i + 1) * width) - max(a, i * width)
            if ov > 0:
                acc[i] += x * ov / (b - a)
    return max(acc) / width


def pool(mix: Mapping[str, Any], block: int = BLOCK) -> int:
    """An engine's KV pool in bytes: kv_blocks blocks of `block` tokens on each of its tp GPUs."""
    return mix["kv_blocks"] * block * mix["kv_bytes_per_token"] * mix["args"]["tp"]


def rates(m: Mapping[str, Any], mix: Mapping[str, Any], requests: Sequence[Mapping[str, int]], width: float = WIDTH) -> dict:
    """One engine's units by class over the trace, a second on average (busy) and at the peak window; tokens a second alike."""
    tot = exact(mix, requests)
    u = step_units(m, **{k: tot[k] for k in MODEL})
    per = [step_units(m, **x) for x in steps(mix, requests, tot["split_keys"], tot["splits"])]
    if add(*per) != u:
        raise AssertionError("the steps' units don't sum to the trace's")
    b = busy(mix)
    totals = [float(total(x)) for x in per]
    pk = peak(mix, totals, width)
    share = {c: float(v / total(u)) for c, v in by_class(u).items()}
    return {"trace": tot, "units": u, "busy_s": b, "tokens_per_s": tot["T"] / b,
            "tokens_per_s_peak": peak(mix, [s["tokens"] for s in mix["steps"]], width),
            "units_per_token": total(u) / tot["T"], "mean": float(total(u)) / b, "peak": pk,
            "by_class_mean": {c: s * float(total(u)) / b for c, s in share.items()},
            "by_class_peak": {c: s * pk for c, s in share.items()}, "share": share}


def f(R_cap: float, E: float, L: float, R_in: float, M: float) -> float:
    return (128 * R_cap * L / E + R_in * L) / M


def lever_models(doc: Mapping[str, Any]) -> dict[str, dict]:
    """The step model as built and under each lever, cumulatively: K' quantized once at the KV write (exact: its scale is
    static), then V' stored once with a static scale, then an int8 LM head, then static activation scales in the linears. A
    write lever takes its family's units from every key a step reads (``b_kv``) and every call (``w0``) to one write a token
    row (``r0``, `units.py`'s ``key_write``)."""
    m = model_units(doc)
    lv = doc["levers"]
    out = {"as built": m}

    def at_write(m0: dict, fam: str) -> dict:
        co = {c: dict(v) for c, v in m0["coefficients"].items()}
        zero = {name: Fraction(0) for name in COEFFICIENTS}
        for name, per in (("b_kv", lv["attention_per_key"].get(fam, {})), ("w0", lv["attention_per_call"].get(fam, {})),
                          ("r0", {c: -F(v) for c, v in lv["key_write"].items()})):
            for c, v in per.items():
                co.setdefault(c, dict(zero))[name] -= F(v)
        return {**m0, "coefficients": co}

    k = at_write(m, "Int8StaticQuant_v1")
    out["K' at write"] = k
    v = at_write(k, "Int8ColumnQuant_v1")
    out["+ V' at write"] = v
    smp = dict(v["sample"])
    gemm = [r for r in doc["callees"].values() if r["family"] == "Gemm_v2"]
    if gemm and "lm_head_int8" in lv:
        X = doc["samples"]
        smp = add(smp, scale(gemm[0]["units"], Fraction(-1, X)),
                  scale(lv["lm_head_int8"]["units"], F(gemm[0]["instances"]) / X * F(gemm[0].get("ranks", 1))))
    h = {**v, "sample": smp}
    out["+ int8 LM head"] = h
    tok = dict(h["token"])
    for cid, row in lv["linear"].items():
        rs = row["by_family"].get("Int8RowScale_v1", {})
        r = doc["callees"][cid]
        tok = add(tok, scale(rs, Fraction(-F(r["instances"]), doc["tokens"]) * F(r.get("ranks", 1))))
    out["+ static act. scales"] = {**h, "token": tok}
    return out


def primitives(doc: Mapping[str, Any], mix: Mapping[str, Any], requests: Sequence[Mapping[str, int]]) -> dict[str, Fraction]:
    """A token row's units as built over the trace by ``primitive|class``, largest first: a token row's own (the spec's extra
    under ``(TP all-reduce)``), a sampled row's times X / T, and each attention coefficient times its quantity a token."""
    pr = doc["levers"]["primitives"]
    t = exact(mix, requests)
    x = dict(zip(COEFFICIENTS, (t["T"], t["Q"], t["K"], t["n"], t["n0"])))
    out: Counter = Counter()
    for k, v in pr["token"].items():
        out[k] += F(v)
    for k, v in ((doc.get("extra") or {}).get("token") or {}).items():
        out[f"(TP all-reduce)|{k}"] += F(v)
    for k, v in pr["sample"].items():
        out[k] += F(v) * t["X"] / t["T"]
    for k, co in pr["attention"].items():
        out[k] += int(doc["attention"]["calls"]) * sum((F(co[c]) * x[c] for c in COEFFICIENTS), Fraction(0)) / t["T"]
    return {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v}


def table(models: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, int]]]], gpus: int,
          width: float = WIDTH) -> dict:
    out: dict[str, Any] = {"E_word": E_WORD, "lifetimes_s": LIFETIMES, "R_in": list(R_INS), "gpus": gpus, "width_s": width,
                           "models": {}}
    for name, (doc, mix, requests) in models.items():
        engines = gpus // mix["args"]["tp"]
        M = pool(mix) * engines
        rows = {}
        for lever, m in lever_models(doc).items():
            r = rates(m, mix, requests, width)
            node = {"mean": r["mean"] * engines, "peak": r["peak"] * engines}
            rows[lever] = {
                "units_per_token": float(r["units_per_token"]), "R_cap_node": node,
                "R_cap_node_by_class_peak": {c: x * engines for c, x in r["by_class_peak"].items()},
                "share": r["share"],
                "f": {lt: {e: {str(R_in): {"peak": f(node["peak"], E, L, R_in, M), "mean": f(node["mean"], E, L, R_in, M)}
                               for R_in in R_INS} for e, E in E_WORD.items()} for lt, L in LIFETIMES.items()}}
            if lever == "as built":
                rows[lever].update({"trace": r["trace"], "busy_s": r["busy_s"], "tokens_per_s_engine": r["tokens_per_s"],
                                    "tokens_per_s_engine_peak": r["tokens_per_s_peak"],
                                    "tokens_per_s_node": r["tokens_per_s"] * engines})
        prims = primitives(doc, mix, requests) if "primitives" in doc["levers"] else {}
        if prims and sum(prims.values()) != share_total(doc, mix, requests):
            raise AssertionError(f"{name}: the primitives' units a token don't sum to the document's")
        out["models"][name] = {"primitives_per_token": prims, "engines": engines, "M_dyn": M, "pool_engine": pool(mix), "kv_blocks": mix["kv_blocks"],
                               "kv_bytes_per_token": mix["kv_bytes_per_token"], "tp": mix["args"]["tp"],
                               "program": doc["program"], "levers": rows,
                               "input_term": {lt: {str(R_in): R_in * L / M for R_in in R_INS} for lt, L in LIFETIMES.items()}}
    return out


def share_total(doc: Mapping[str, Any], mix: Mapping[str, Any], requests: Sequence[Mapping[str, int]]) -> Fraction:
    """The units a token row as built over the trace, exactly."""
    t = exact(mix, requests)
    return total(step_units(model_units(doc), **{k: t[k] for k in MODEL})) / t["T"]


def markdown(t: Mapping[str, Any]) -> str:
    E = list(t["E_word"])
    lines = ["| model | L | " + " | ".join(f"f, E {e} ({t['E_word'][e]:,})" for e in E) + " | f, input term at 1 MB/s |",
             "|---|---|" + "---|" * (len(E) + 1)]
    for name, m in t["models"].items():
        built = m["levers"]["as built"]
        for lt in t["lifetimes_s"]:
            cells = [_pct(built["f"][lt][e][str(R_INS[0])]["peak"]) for e in E]
            lines.append(f"| {name} | {lt} | " + " | ".join(cells) + f" | {_pct(m['input_term'][lt][str(R_INS[1])])} |")
    lines.append("")
    lines.append("| model | lever (cumulative) | units/token | R_cap node, peak (units/s) | dominant class | "
                 + " | ".join(f"f, 1 h, E {e}" for e in E) + " |")
    lines.append("|---|---|---|---|---|" + "---|" * len(E))
    for name, m in t["models"].items():
        for lever, r in m["levers"].items():
            dom = max(r["share"], key=r["share"].get)
            lines.append(f"| {name} | {lever} | {r['units_per_token']:.3g} | {r['R_cap_node']['peak']:.3g} | "
                         f"{dom} {r['share'][dom]:.0%} | "
                         + " | ".join(_pct(r["f"]["1 h"][e][str(R_INS[0])]["peak"]) for e in E) + " |")
    lines += ["", "| model | primitive (class) | units/token, as built | share |", "|---|---|---|---|"]
    for name, m in t["models"].items():
        whole = sum(m["primitives_per_token"].values())
        for k, v in list(m["primitives_per_token"].items())[:8]:
            p, c = k.split("|")
            lines.append(f"| {name} | {p} ({c}) | {v:.3g} | {v / whole:.0%} |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x:.1%}" if x < 1 else f"{x:.3g}x"


def jsonable(o: Any) -> Any:
    if isinstance(o, Fraction):
        return int(o) if o.denominator == 1 else float(o)
    if isinstance(o, Mapping):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=UNITS.json,MIX.json")
    ap.add_argument("--gpus", type=int, default=8, help="a node's GPUs")
    ap.add_argument("--width", type=float, default=WIDTH, help="the peak window, in seconds")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    models = {}
    for spec in a.model:
        name, files = spec.split("=", 1)
        u, m = files.split(",")
        doc, mix = json.loads(Path(u).read_text()), json.loads(Path(m).read_text())
        models[name] = (doc, mix, trace_requests(mix))
    t = table(models, a.gpus, a.width)
    a.out.write_text(json.dumps(jsonable(t), indent=1))
    print(markdown(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
