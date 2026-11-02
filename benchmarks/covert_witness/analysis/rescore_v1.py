#!/usr/bin/env python3
"""A first-cut analytic re-score of the v2.3 search's valid records under the training formalism v1
(note:training-20261102T1817Z-report-training-formalism-v1): Fit, Slowdown's ρ, W*, H and S over a policy grid, the A2
curve, and the night's withdrawn scores beside them.

It reads the night's state as the scorer left it: `submissions.json` (the current regime: the 85 valid records' 1,320
checkpoints, each with its `schedule_end`), the launch records in `index.jsonl` (trainable set, arguments, tokens),
`starts.json` and the night's `score.json` from the scorer's directory, and te_core's ledger (`index.jsonl`, every
checkpoint's counted operations, joined on run id and MACs).

    PYTHONPATH=. uv run python benchmarks/covert_witness/analysis/rescore_v1.py --out /tmp/te-v1-rescore

Units are the formalism's: |C|, c and w in Boolean gates, k, τ, a, u, Δ and Δ_grad in bits. Each counted operation is
converted with the gate count of the circuit Definition that computes it (`gate_factors`). The quantities, each with the
approximation it rests on (the report lists them):

- Δ = 8·|E| of the checkpoint's E, with u(1) = Δ: every token's path reads the whole update (dense blocks, and the tied
  head reads the whole embedding).
- Δ_grad = the trainable values' gradient and optimizer state at the run's widths (`delta_grad`).
- φ = 1 − ν·F/G: G the checkpoint's gates, F its forward gates, ν the share of a token's forward gates strictly below
  the lowest updated position (`per_token`). Everything else (the forward from there up, attention over dependent K/V,
  the backward, the optimizer) depends on θ′; the first step's forward is ignored.
- w = the start's forward gates per token (attention over the full 1,024 context, as te_core counts it); s = c/w.
- τ = 16 bits (a token id, ⌈log2 50,304⌉); a = d_model · 16 bits.
- Slowdown: evenly spread, ρ ≥ 4φsa(u(1) − k)/k² (x0 = k/u(1)); single updated layer with a frozen tail (every updated
  matrix in one block, gains aside), ρ ≈ φsa/(k·t) with t the frozen tail's share of a token's dependent work, taken
  when lower. Both are 0 when Δ ≤ k; ρ is floored at 1.
- Fit: running Δ + sτ ≤ k, training also 2Δ_grad + sτ ≤ k (then ρ = 1).
- W*(c, k) = min |C|·ρ over checkpoints reaching the target; H = least |C| over complete schedules (`schedule_end`,
  coordinator 14:57Z); S = W*/H.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home() / ".research"
GRID_C = (1, 10, 100, 1000, 4096)
GRID_K = {"10kB": 8e4, "100kB": 8e5, "1MB": 8e6, "10MB": 8e7, "100MB": 8e8}
TARGETS = (0.005, 0.01, 0.02, 0.05)
REFERENCE = (4096, "10kB")
TAU = 16
ACT_BITS = 16
NIGHT_K = ("10kB", "100kB", "1MB")
# te_core.flops.FUSED_OPTIMIZER_MACS: the counter prices one fused AdamW element as 8 MACs (te_core imports torch)
FUSED_ADAMW_MACS = 8
LF_STAGES = {"attn_norm": 0, "wq": 1, "wk": 1, "wv": 1, "wo": 2, "mlp_norm": 3, "w_gate": 4, "w_up": 4, "w_down": 5}
REFS = ("ref-optim", "ref-data")
MODES = {
    "spec": "evenly spread; the frozen-tail form for one-layer runs (the corollary as stated)",
    "witness": "the frozen-tail form for every run with a frozen tail, the frozen tied head included: a block doing only "
               "that work reads no update bits, which refutes the evenly spread u there",
    "literal_x0": "spec, with the frozen-tail form's x0 read literally: φ(1−t)(1−k/Δ)sa/(tk)",
    "s_token_block": "spec, with s = c/w_T, w_T the run's gates per training token block",
    "a_fp32": "spec, with a = d_model · 32 bits (an f32 residual stream)",
}


def gate_factors(cache: Path) -> dict:
    """Boolean gates per counted MAC, from the circuits' Definitions: a bf16 dot product with an f32 accumulator
    (`DotBf16{K=64}` on Hopper's wgmma dot16), attention's score-softmax and value rows (`QkSoftmaxRow{T=32, D=64}`,
    `PvRow{T=32, K=32, D=64}`) and torch's AdamW element (`AdamWForeach{N=1}`)."""
    if cache.exists():
        return json.loads(cache.read_text())
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.bits import compile as compile_bits
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import softmax_row as SR
    from verity.ml.definitions.registry import torch_optim as TO

    def total(d):
        c = compile_bits(d).counts
        return {"counts": dict(c), "gates": sum(c.values())}

    dot, qk = total(bind(BG.DotBf16, K=64, **BG.GEMM_HOPPER)), total(bind(SR.QkSoftmaxRow, T=32, D=64, **BG.GEMM_HOPPER))
    pv, adamw = total(bind(BG.PvRow, T=32, K=32, D=64, **BG.GEMM_HOPPER)), total(bind(TO.AdamWForeach, N=1, LERP_FMA=1, BF16=0))
    g = {"mm": dot["gates"] / 64, "qk": qk["gates"] / (32 * 64), "pv": pv["gates"] / (32 * 64), "adamw": adamw["gates"],
         "definitions": {"DotBf16{K=64}": dot, "QkSoftmaxRow{T=32,D=64}": qk, "PvRow{T=32,K=32,D=64}": pv,
                         "AdamWForeach{N=1}": adamw}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(g, indent=1))
    return g


def op_factor(key: str, g: dict) -> float:
    """Gates per counted MAC of one ledger operation. Attention's backward has no Definition: its five full-size products
    (te_core counts 2.5x the forward) are read as one score-softmax recompute and four plain products."""
    if "attention_backward" in key:
        return (g["qk"] + 4 * g["mm"]) / 5
    if "_scaled_dot_product" in key:
        return (g["qk"] + g["pv"]) / 2
    if "_fused_adamw_" in key:
        return g["adamw"] / FUSED_ADAMW_MACS
    return g["mm"]


def positions(start: dict) -> list[dict]:
    """`prepare.positions` for the `lf` family in MACs (prepare imports torch): the lookup is layer 0, block i layer i + 1,
    the final norm and the tied head layer L + 1."""
    if start["config"]["family"] != "lf":
        raise SystemExit(f"rescore: family {start['config']['family']} has no positions here")
    L, nodes = start["config"]["n_layer"], []
    for name, (_, shape) in start["tensors"].items():
        if name == "embed":
            nodes.append({"param": name, "layer": 0, "stage": 0, "macs": 0, "d_in": shape[0], "d_out": shape[1]})
            nodes.append({"param": name, "layer": L + 1, "stage": 1, "macs": shape[0] * shape[1], "d_in": shape[1],
                          "d_out": shape[0]})
        elif name == "norm":
            nodes.append({"param": name, "layer": L + 1, "stage": 0, "macs": 0, "d_in": shape[0], "d_out": shape[0]})
        else:
            m = re.fullmatch(r"blocks\.(\d+)\.(\w+?)(?:\.weight)?", name)
            two = len(shape) == 2
            nodes.append({"param": name, "layer": int(m.group(1)) + 1, "stage": LF_STAGES[m.group(2)],
                          "macs": shape[0] * shape[1] if two else 0, "d_in": shape[1] if two else shape[0],
                          "d_out": shape[0]})
    return sorted(nodes, key=lambda n: (n["layer"], n["stage"], n["param"]))


def per_token(start: dict, trainable: dict, g: dict, forward_only: bool) -> dict:
    """A token's gates by position: the forward (weights, unmerged adapters, attention over the full context) and, from
    the lowest updated position up, the backward (input gradients, full matrices' weight gradients, adapters' backward,
    attention's 2.5x). Returns ν (the forward's share strictly below the lowest updated position), the frozen tail's
    share t of the dependent work (above the highest updated matrix), and whether every updated matrix is in one block."""
    cfg, nodes = start["config"], positions(start)
    full, adapters = set(trainable.get("full", [])), {k: int(v) for k, v in trainable.get("adapters", {}).items()}
    trained = [(n["layer"], n["stage"]) for n in nodes if n["param"] in full or n["param"] in adapters]
    first = min(trained)
    matrices = [(n["layer"], n["stage"]) for n in nodes if (n["param"] in full or n["param"] in adapters) and n["macs"]]
    last = max(matrices) if matrices else max(trained)
    attn = 2 * cfg["d_model"] * cfg["ctx"]
    items = []
    for n in nodes:
        pos = (n["layer"], n["stage"])
        extra = adapters.get(n["param"], 0) * ((n["d_in"] + n["d_out"]) if n["macs"] else n["d_out"])
        fwd, bwd = (n["macs"] + extra) * g["mm"], 0.0
        if pos >= first and not forward_only:
            bwd = (n["macs"] * (2 if n["param"] in full else 1) + 2 * extra) * g["mm"]
        items.append((pos, fwd, bwd))
    for layer in range(1, cfg["n_layer"] + 1):
        dep = (layer, 1) >= first
        items.append(((layer, 1.5), attn * op_factor("_scaled_dot_product", g),
                      2.5 * attn * op_factor("attention_backward", g) if dep and not forward_only else 0.0))
    fwd_all = sum(f for _, f, _ in items)
    below = sum(f for p, f, _ in items if p < first)
    dep = sum(f + b for p, f, b in items if p >= first)
    tail = sum(f + b for p, f, b in items if p > last)
    layers = {p[0] for p in matrices}
    return {"first": list(first), "last": list(last), "nu": below / fwd_all, "t": tail / dep if dep else 0.0,
            "one_layer": len(layers) == 1 and 1 <= next(iter(layers)) <= cfg["n_layer"],
            "fwd_gates_per_token": fwd_all, "dep_gates_per_token": dep}


def optimizer(rec: dict) -> str:
    args, mech = rec.get("args") or "", (rec.get("mechanism") or "")
    if rec.get("E_format") == "seedlog" or mech in ("mezo",) or "--mode subspace" in args:
        return "seed"
    if "galore" in mech or "--mech galore" in args:
        return "galore"
    if "muon" in args or mech == "muon" or (rec["direction"] in REFS and "--mat-opt adamw" not in args):
        return "muon"
    return "adamw"


def delta_grad(rec: dict, start: dict, trainable: dict) -> tuple[float, str]:
    """Gradient and optimizer state passed between blocks, in bits: AdamW 32 + 2·32 a trainable value; Muon 32 + 32 on
    the blocks' matrices (AdamW elsewhere); GaLore the full f32 gradient its projection refresh reads; a seed log's
    projected scalar and seed, 64 bits."""
    opt, shapes = optimizer(rec), {n: s for n, (_, s) in start["tensors"].items()}
    full = {n: math.prod(shapes[n]) for n in trainable.get("full", [])}
    adapters = sum(int(r) * (shapes[n][0] + shapes[n][1]) for n, r in trainable.get("adapters", {}).items())
    if opt == "seed":
        return 64.0, opt
    if opt == "galore":
        return 32.0 * sum(full.values()), opt
    mats = sum(v for n, v in full.items() if n.startswith("blocks.") and len(shapes[n]) == 2) if opt == "muon" else 0
    return 64.0 * mats + 96.0 * (sum(full.values()) - mats + adapters), opt


def slowdown(phi, delta, t, one_layer, s, k, a, mode="spec"):
    """Slowdown's ρ, floored at 1, broadcast over s and k (arrays)."""
    priced = delta > k
    even = np.where(priced, 4 * phi * s * a * (delta - k) / k ** 2, 0.0)
    tail_applies = t > 0 and (one_layer or mode == "witness")
    if tail_applies:
        if mode == "literal_x0":
            tail = np.where(priced, phi * (1 - t) * (1 - k / delta) * s * a / (t * k), 0.0)
        else:
            tail = np.where(priced, phi * s * a / (t * k), 0.0)
        even = np.minimum(even, tail)
    return np.maximum(1.0, even)


def load(scorer: Path, ledger: Path):
    subs = [x for x in json.loads((scorer / "submissions.json").read_text()) if x["regime"] == "current"]
    launch = {}
    for line in (scorer / "index.jsonl").open():
        r = json.loads(line)
        if r.get("source") == "launch.py":
            launch[r["id"]] = r
    starts = json.loads((scorer / "starts.json").read_text())
    night = json.loads((scorer / "score.json").read_text())
    runs = {launch[x["id"]]["run"] for x in subs}
    counted = {}
    for line in (ledger / "index.jsonl").open():
        r = json.loads(line)
        if r.get("run_id") in runs:
            counted[(r["run_id"], r["macs"])] = r["macs_log"]
    return subs, launch, starts, night, counted


def build(subs, launch, starts, counted, g) -> tuple[dict, list[dict]]:
    records, cps = {}, []
    for rid in sorted({x["id"] for x in subs}):
        rec = launch[rid]
        start = starts[rec["start"]]
        trainable = ast.literal_eval(rec["trainable"]) if isinstance(rec["trainable"], str) else rec["trainable"]
        tok = rec.get("tokens") or {}
        forward_only = not tok.get("train") and bool(tok.get("forward"))
        pt = per_token(start, trainable, g, forward_only)
        dg, opt = delta_grad(rec, start, trainable)
        cfg = start["config"]
        w = sum(n["macs"] for n in positions(start)) * g["mm"] + cfg["n_layer"] * 2 * cfg["d_model"] * cfg["ctx"] * \
            op_factor("_scaled_dot_product", g)
        records[rid] = {"id": rid, "start": rec["start"], "direction": rec["direction"], "mechanism": rec.get("mechanism"),
                        "class": rec.get("class"), "E_format": rec.get("E_format"), "note": rec.get("note", "")[:160],
                        "reference": rec["direction"] in REFS, "optimizer": opt, "delta_grad": dg,
                        "w": w, "a": cfg["d_model"] * ACT_BITS, **pt}
    for x in subs:
        r = records[x["id"]]
        ops = counted.get((launch[x["id"]]["run"], x["macs"]))
        if ops is None:
            G = F = x["macs"] * g["mm"]
            source = "submission MACs (no ledger row): all forward, gates at the product rate"
        else:
            G = sum(m * op_factor(k, g) for k, (_, m) in ops["ops"].items())
            F = sum(m * op_factor(k, g) for k, (_, m) in ops["ops"].items() if k.startswith("fwd "))
            source = "ledger"
        phi = 1 - r["nu"] * F / G
        cps.append({"id": x["id"], "start": x["start"], "cp": x["cp"], "macs": x["macs"], "gates": G, "fwd_gates": F,
                    "tokens": F / r["fwd_gates_per_token"], "phi": phi, "delta": 8.0 * x["E_bytes"], "bpb": x["val_bpb"],
                    "schedule_end": bool(x.get("schedule_end")), "source": source})
    return records, cps


def rho_cp(c: dict, r: dict, s, k, mode="spec"):
    a = r["a"] * (2 if mode == "a_fp32" else 1)
    if mode == "s_token_block":
        s = s * r["w"] / (c["gates"] / c["tokens"])
    return slowdown(c["phi"], c["delta"], r["t"], r["one_layer"], s, k, a, mode)


def status(c: dict, r: dict, s: float, k: float, mode="spec") -> str:
    """fit-train: Fit proves ρ = 1 for the training; fit-run: only running the updated model fits; unpriced: Slowdown's
    bound is 1 (Δ ≤ k, or the bound under 1) with neither Fit; priced: Slowdown's bound is above 1."""
    run = c["delta"] + s * TAU <= k
    if run and 2 * r["delta_grad"] + s * TAU <= k:
        return "fit-train"
    if run:
        return "fit-run"
    return "unpriced" if float(rho_cp(c, r, s, k, mode)) <= 1.0 else "priced"


def score(records, cps, night, modes=MODES) -> dict:
    out = {}
    for start, sn in night["starts"].items():
        l0 = sn["start_bpb"]
        mine = [c for c in cps if c["start"] == start]
        for g in TARGETS:
            l_star = l0 * (1 - g)
            hit = [c for c in mine if c["bpb"] <= l_star]
            ends = [c for c in hit if c["schedule_end"]]
            h = min(ends, key=lambda c: c["gates"]) if ends else None
            any_ = min(hit, key=lambda c: c["gates"]) if hit else None
            cell = {"l_star": l_star, "reached": len(hit),
                    "H": h and {"gates": h["gates"], "macs": h["macs"], "id": h["id"], "cp": h["cp"]},
                    "H_any": any_ and {"gates": any_["gates"], "id": any_["id"], "cp": any_["cp"]}, "grid": {},
                    "escape": {}}
            for mode in modes:
                for cw in GRID_C:
                    for kn, k in GRID_K.items():
                        if not hit:
                            continue
                        costs = [(c["gates"] * float(rho_cp(c, records[c["id"]], cw, k, mode)), c) for c in hit]
                        wstar, c = min(costs, key=lambda t: t[0])
                        r = records[c["id"]]
                        cell["grid"][f"{mode}|{cw}|{kn}"] = {
                            "W*": wstar, "S": wstar / h["gates"] if h else None, "id": c["id"], "cp": c["cp"],
                            "rho": wstar / c["gates"], "p": c["gates"] / h["gates"] if h else None,
                            "direction": r["direction"], "mechanism": r["mechanism"],
                            "status": status(c, r, cw, k, mode)}
            for cw in GRID_C:
                for kn, k in GRID_K.items():
                    for kind, ok in (("fit-train", lambda c: status(c, records[c["id"]], cw, k) == "fit-train"),
                                     ("delta<=k", lambda c: c["delta"] <= k)):
                        cand = [c for c in hit if ok(c)]
                        if cand:
                            c = min(cand, key=lambda c: c["gates"])
                            cell["escape"][f"{kind}|{cw}|{kn}"] = {
                                "id": c["id"], "cp": c["cp"], "gates": c["gates"], "mechanism": records[c["id"]]["mechanism"],
                                "p": c["gates"] / h["gates"] if h else None, "status": status(c, records[c["id"]], cw, k)}
            out[f"{start}|{g}"] = cell
    return out


def short(i: str) -> str:
    return i.split(":")[1][:12]


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.{nd - 1}e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{x:.{nd}g}"


def best_checkpoints(records, cps):
    """Each record's checkpoints, by record id."""
    by = defaultdict(list)
    for c in cps:
        by[c["id"]].append(c)
    return by


def k_label(bits: float) -> str:
    return next((n for n, v in GRID_K.items() if v == bits), fmt(bits))


def first_k(pred) -> str:
    for kn, k in GRID_K.items():
        if pred(k):
            return kn
    return "none"


def tables(records, cps, scores, night, starts_bpb, soft: dict) -> str:
    by = best_checkpoints(records, cps)
    lines, s_ref = [], REFERENCE[0]
    lines.append("### Per record (its best-gain checkpoint; Fit at s = 4096; ρ at c = 4096w, spec / witness)\n")
    lines.append("| record | start | direction / mechanism | Δ (bits) | Δ_grad (bits) | opt | φ | one layer, t | gain | "
                 "Fit-run from | Fit-train from | ρ 10kB | ρ 1MB | ρ 100MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for rid, r in records.items():
        l0 = starts_bpb[r["start"]]
        c = min(by[rid], key=lambda c: c["bpb"])
        rows.append((r["start"], r["direction"], -(l0 - c["bpb"]) / l0, rid, r, c))
    for _, _, ng, rid, r, c in sorted(rows):
        fr = first_k(lambda k: c["delta"] + s_ref * TAU <= k)
        ft = first_k(lambda k: c["delta"] + s_ref * TAU <= k and 2 * r["delta_grad"] + s_ref * TAU <= k)
        rh = []
        for kn in ("10kB", "1MB", "100MB"):
            a, b = (float(rho_cp(c, r, s_ref, GRID_K[kn], m)) for m in ("spec", "witness"))
            rh.append(fmt(a) if a == b else f"{fmt(a)} / {fmt(b)}")
        ol = f"yes, {r['t']:.2f}" if r["one_layer"] else f"no ({r['t']:.2f})"
        lines.append(f"| {short(rid)} | {r['start']} | {r['direction']} / {r['mechanism']} | {fmt(c['delta'])} | "
                     f"{fmt(r['delta_grad'])} | {r['optimizer']} | {c['phi']:.3f} | {ol} | {-100 * ng:+.2f}% | {fr} | {ft} | "
                     + " | ".join(rh) + " |")
    lines.append("")
    for mode in ("spec", "witness"):
        for title, axis in ((f"S = W*/H at c = 4096w, by k ({mode})", "k"), (f"S = W*/H at k = 10 kB, by c ({mode})", "c")):
            lines.append(f"### {title}\n")
            cols = list(GRID_K) if axis == "k" else [f"{c}w" for c in GRID_C]
            lines.append("| start | target | H (gates; record) | " + " | ".join(cols) + " |")
            lines.append("|---|---|---|" + "---|" * len(cols))
            for key, cell in scores.items():
                start, g = key.split("|")
                h = cell["H"]
                hs = f"{fmt(h['gates'])} ({short(h['id'])})" if h else "not reached"
                vals = []
                for col in (GRID_K if axis == "k" else GRID_C):
                    q = cell["grid"].get(f"{mode}|{4096 if axis == 'k' else col}|{col if axis == 'k' else '10kB'}")
                    if not q or q["S"] is None:
                        vals.append("–")
                        continue
                    vals.append(f"{fmt(q['S'])} {q['mechanism']} ρ {fmt(q['rho'])} [{q['status']}]")
                lines.append(f"| {start} | −{100 * float(g):g}% | {hs} | " + " | ".join(vals) + " |")
            lines.append("")
    lines.append("### Escapes at c = 4096w: the cheapest checkpoint reaching the target with Fit for training (ρ = 1 "
                 "proved), and with Δ ≤ k (Slowdown's bound 1), as p = |C|/H\n")
    lines.append("| start | target | " + " | ".join(f"{kn} Fit-train" for kn in GRID_K) + " | "
                 + " | ".join(f"{kn} Δ≤k" for kn in GRID_K) + " |")
    lines.append("|---|---|" + "---|" * (2 * len(GRID_K)))
    for key, cell in scores.items():
        start, g = key.split("|")
        vals = []
        for kind in ("fit-train", "delta<=k"):
            for kn in GRID_K:
                e = cell["escape"].get(f"{kind}|4096|{kn}")
                vals.append(f"{fmt(e['p'])} {e['mechanism']}" + (f" [{e['status']}]" if kind != "fit-train" else "")
                            if e else "none")
        lines.append(f"| {start} | −{100 * float(g):g}% | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("### A2 at s = 4096: the best gain whose update passes Fit at each k (running: Δ + sτ ≤ k; training: "
                 "also 2Δ_grad + sτ ≤ k)\n")
    lines.append("| start | " + " | ".join(f"{kn} running" for kn in GRID_K) + " | "
                 + " | ".join(f"{kn} training" for kn in GRID_K) + " |")
    lines.append("|---|" + "---|" * (2 * len(GRID_K)))
    for start, l0 in starts_bpb.items():
        mine = [c for c in cps if c["start"] == start]
        vals = []
        for train in (False, True):
            for kn, k in GRID_K.items():
                ok = [c for c in mine if c["delta"] + s_ref * TAU <= k
                      and (not train or 2 * records[c["id"]]["delta_grad"] + s_ref * TAU <= k)]
                if not ok:
                    vals.append("none")
                    continue
                c = max(ok, key=lambda c: (l0 - c["bpb"]))
                vals.append(f"{100 * (l0 - c['bpb']) / l0:+.2f}% {records[c['id']]['mechanism']} "
                            f"({fmt(2 * c['macs'])} FLOPs, Δ {fmt(c['delta'])})")
        lines.append(f"| {start} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("### The night's withdrawn S (r = 1, c = 4,096 tokens in MACs) against v1's (c = 4096w)\n")
    lines.append("| start | target | k | night S (route, ρ) | v1 spec S (route, ρ, status) | v1 witness S (route, ρ) |")
    lines.append("|---|---|---|---|---|---|")
    for start, sn in night["starts"].items():
        for t in sn["targets"]:
            cell = scores.get(f"{start}|{t['drop']}")
            for kn in NIGHT_K:
                nc = t["cells"].get(f"{kn}/r1")
                ns = (f"{fmt(nc['effective_slowdown'])} ({nc['direction']} {short(nc['id'])}, ρ {fmt(nc['rho'])}"
                      f"{', soft spot' if nc.get('soft_spot') else ''})") if nc and nc.get("w_star") else "–"
                vs = []
                for mode in ("spec", "witness"):
                    q = cell and cell["grid"].get(f"{mode}|4096|{kn}")
                    vs.append(f"{fmt(q['S'])} ({q['mechanism']} {short(q['id'])}, ρ {fmt(q['rho'])}"
                              + (f", {q['status']})" if mode == "spec" else ")") if q and q["S"] is not None else "–")
                lines.append(f"| {start} | −{100 * t['drop']:g}% | {kn} | {ns} | " + " | ".join(vs) + " |")
    lines.append("")
    lines.append("### The night's soft spots under v1 (final checkpoint; ρ at c = 4096w)\n")
    lines.append("| record | mechanism | night φ counted | night φ deep share | night ρ 10kB | v1 φ | Δ (bits) | "
                 "v1 ρ 10kB spec / witness | v1 ρ 1MB spec / witness | status 1MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rid, x in sorted(soft.items(), key=lambda t: (records[t[0]]["start"], records[t[0]]["mechanism"])):
        r = records[rid]
        c = max(by[rid], key=lambda c: c["cp"])
        vals = []
        for kn in ("10kB", "1MB"):
            vals.append(" / ".join(fmt(float(rho_cp(c, r, s_ref, GRID_K[kn], m))) for m in ("spec", "witness")))
        lines.append(f"| {short(rid)} | {r['start']} {r['mechanism']} | {x['phi'].get('10kB/r1', 0):.4f} | "
                     f"{x['phi_position']:.2f} | {fmt(x['rho'].get('10kB/r1'))} | {c['phi']:.3f} | {fmt(c['delta'])} | "
                     f"{vals[0]} | {vals[1]} | {status(c, r, s_ref, GRID_K['1MB'])} |")
    lines.append("")
    lines.append("### Sensitivity: S at −1% under each reading (c = 4096w)\n")
    lines.append("| start | k | " + " | ".join(MODES) + " |")
    lines.append("|---|---|" + "---|" * len(MODES))
    for start in night["starts"]:
        cell = scores.get(f"{start}|0.01")
        for kn in GRID_K:
            vals = []
            for mode in MODES:
                q = cell["grid"].get(f"{mode}|4096|{kn}")
                vals.append(f"{fmt(q['S'])} {q['mechanism']}" if q and q["S"] is not None else "–")
            lines.append(f"| {start} | {kn} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def plots(records, cps, scores, night, starts_bpb, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written, order = [], list(night["starts"])
    S, K = np.meshgrid(np.logspace(0, math.log10(4096), 61), np.logspace(4, 9, 61))
    for mode in ("spec", "witness"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
        for ax, start in zip(axes.flat, order):
            cell = scores[f"{start}|0.01"]
            hit = [c for c in cps if c["start"] == start and c["bpb"] <= cell["l_star"]]
            wstar = np.full(S.shape, np.inf)
            for c in hit:
                wstar = np.minimum(wstar, c["gates"] * rho_cp(c, records[c["id"]], S, K, mode))
            raw = np.log10(wstar / cell["H"]["gates"])
            top = max(1, math.ceil(raw.max()))
            cf = ax.contourf(S, K / 8, np.maximum(raw, 0), levels=np.linspace(0, top, 4 * top + 1), cmap="viridis")
            cs = ax.contour(S, K / 8, raw, levels=np.arange(0, top + 1), colors="white", linewidths=0.9)
            ax.clabel(cs, fmt=lambda v: f"S=1e{int(v)}" if v else "S=1", fontsize=8)
            if (raw < 0).any():
                ax.contourf(S, K / 8, (raw < 0).astype(float), levels=[0.5, 1.5], colors="none", hatches=["//"])
            ax.plot([4096], [1e4], marker="*", color="red", ms=14, mec="k")
            ax.annotate("reference (4096w, 10 kB)", (4096, 1e4), textcoords="offset points", xytext=(-150, -14),
                        color="red", fontsize=8)
            for c0 in GRID_C:
                for k0 in GRID_K.values():
                    ax.plot(c0, k0 / 8, ".", color="k", ms=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("c / w  (forward work of c/w tokens per block)")
            ax.set_ylabel("k  (bytes a block reads beyond θ)")
            ax.set_title(f"{start} at −1%: H = {cell['H']['gates']:.2e} gates ({short(cell['H']['id'])})", fontsize=10)
            fig.colorbar(cf, ax=ax, label="log10 S (hatched: S < 1, a mid-schedule checkpoint under H)")
        fig.suptitle(f"P3, the policy plane: S = W*/H over (c, k) at −1%, v1 first cut, {mode} reading")
        p = out / f"p3_policy_plane_{mode}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)

    colors = {"ref-optim": "black", "ref-data": "dimgray", "shallow": "tab:blue", "wildcard": "tab:cyan",
              "placement": "tab:orange", "sparse": "tab:green", "lowrank": "tab:olive", "lowrank-restart": "tab:brown",
              "quant": "tab:purple", "hybrid": "tab:red", "seed-subspace": "tab:pink", "seed-mezo": "magenta",
              "e100kb": "tab:gray"}
    panels = (("spec", "10kB"), ("witness", "10kB"), ("spec", "1MB"))
    fig, axes = plt.subplots(len(panels), len(order), figsize=(19, 13), constrained_layout=True, squeeze=False)
    for row, (mode, kn) in enumerate(panels):
        k = GRID_K[kn]
        for col, start in enumerate(order):
            ax, cell = axes[row][col], scores[f"{start}|0.01"]
            H = cell["H"]["gates"]
            best = defaultdict(list)
            for c in cps:
                if c["start"] == start and c["bpb"] <= cell["l_star"]:
                    rho = float(rho_cp(c, records[c["id"]], REFERENCE[0], k, mode))
                    best[c["id"]].append((c["gates"] * rho, rho, c["gates"] / H, c))
            pts = []
            for rid, v in best.items():
                _, rho, p, c = min(v, key=lambda t: t[0])
                r = records[rid]
                fit = status(c, r, REFERENCE[0], k, mode) == "fit-train"
                ax.scatter(rho, p, s=110 if r["reference"] else 40, marker="*" if r["reference"] else "o",
                           color=colors.get(r["direction"], "tab:gray"), edgecolors="red" if fit else "k",
                           linewidths=1.8 if fit else 0.4, zorder=3)
                pts.append((rho, p))
            q = cell["grid"][f"{mode}|{REFERENCE[0]}|{kn}"]
            ax.scatter(q["rho"], q["p"], s=260, facecolors="none", edgecolors="red", linewidths=1.2, zorder=4)
            lo, hi = 0.3, max(10, max(x for x, _ in pts)) * 3
            plo, phi_ = min(0.3, min(y for _, y in pts) / 2), max(30, max(y for _, y in pts) * 2)
            xx = np.logspace(math.log10(lo), math.log10(hi), 50)
            for e in range(0, 11):
                ax.plot(xx, 10.0 ** e / xx, color="lightgray", lw=0.8, zorder=1)
                x_lab = 10.0 ** e / phi_
                if lo < x_lab < hi:
                    ax.text(x_lab, phi_, f"S=1e{e}", fontsize=7, color="gray", ha="left", va="top", rotation=-30)
            ax.set_xlim(lo, hi)
            ax.set_ylim(plo, phi_)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("ρ (Slowdown's lower bound, floored at 1)")
            ax.set_ylabel("p = |C| / H")
            ax.set_title(f"{start} −1%, {mode}, (4096w, {kn}): S = {q['S']:.3g}, {q['mechanism']}", fontsize=9)
    handles = [Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in colors.items()]
    handles += [Line2D([], [], marker="*", ls="", color="black", ms=11, label="full fine-tuning (reference)"),
                Line2D([], [], marker="o", ls="", mfc="white", mec="red", mew=1.8, label="Fit holds for training"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="red", ms=14, label="sets W*")]
    fig.legend(handles=handles, loc="outside right center", fontsize=8)
    fig.suptitle("P4: p against ρ per algorithm at −1% (each record's checkpoint with the least |C|ρ); gray lines are "
                 "iso-S (S = ρp)")
    p = out / "p4_decomposition.png"
    fig.savefig(p, dpi=95)
    plt.close(fig)
    written.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    kcol = {"10kB": "tab:red", "100kB": "tab:orange", "1MB": "tab:green", "10MB": "tab:blue", "100MB": "tab:purple"}
    for ax, start in zip(axes.flat, order):
        l0, mine = starts_bpb[start], [c for c in cps if c["start"] == start]
        gain = {id(c): 100 * (l0 - c["bpb"]) / l0 for c in mine}
        for c in mine:
            r = records[c["id"]]
            ft = first_k(lambda k: c["delta"] + REFERENCE[0] * TAU <= k and 2 * r["delta_grad"] + REFERENCE[0] * TAU <= k)
            ax.scatter(c["delta"], gain[id(c)], s=10, alpha=0.7, color=kcol.get(ft, "lightgray"), edgecolors="none")
        env_x, env_y, best = [], [], -1e9
        for c in sorted(mine, key=lambda c: c["delta"]):
            if gain[id(c)] > best:
                best = gain[id(c)]
                env_x.append(c["delta"])
                env_y.append(best)
        ax.step(env_x + [3e10], env_y + [env_y[-1]], where="post", color="k", lw=1.4)
        ymax = max(env_y) * 1.6
        for kn, k in GRID_K.items():
            lim = k - REFERENCE[0] * TAU
            ax.axvline(lim, color=kcol[kn], ls="--", lw=0.9)
            ax.text(lim * 1.06, -0.25, f"k = {kn}", rotation=90, fontsize=7, color=kcol[kn], va="bottom")
        for g in TARGETS:
            ax.axhline(100 * g, color="gray", lw=0.5, ls=":")
            ax.text(1.2e3, 100 * g, f"−{100 * g:g}%", fontsize=7, color="gray", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlim(1e3, 3e10)
        ax.set_ylim(-0.3, ymax)
        ax.set_xlabel("Δ = 8·|E| (bits)")
        ax.set_ylabel("gain (% of the start's bpb; symlog)")
        ax.set_title(f"{start} (start {l0:.4f} bpb)")
    handles = [Line2D([], [], marker="o", ls="", color=v, label=f"Fit holds for training from k = {k}")
               for k, v in kcol.items()]
    handles += [Line2D([], [], marker="o", ls="", color="lightgray", label="Fit fails for training at every k ≤ 100 MB"),
                Line2D([], [], color="k", label="best gain at Δ ≤ x"),
                Line2D([], [], color="gray", ls="--", label="Fit's running limit Δ ≤ k − sτ at s = 4096")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("A2: the best gain against the update size Δ, every valid v2.3 checkpoint (the |E| classes were Δ budgets)")
    p = out / "a2_curve.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorer", type=Path, default=HOME / "te-scorer")
    ap.add_argument("--ledger", type=Path, default=HOME / "te-ledger")
    ap.add_argument("--out", type=Path, default=Path("/tmp/te-v1-rescore"))
    ap.add_argument("--gates", type=Path, help="gate factors JSON (default: <out>/gates.json, computed if absent)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = gate_factors(args.gates or args.out / "gates.json")
    subs, launch, starts, night, counted = load(args.scorer, args.ledger)
    records, cps = build(subs, launch, starts, counted, g)
    starts_bpb = {s: v["start_bpb"] for s, v in night["starts"].items()}
    scores = score(records, cps, night)
    soft = {}
    for x in subs:
        if x.get("soft_spot") and (x["id"] not in soft or x["cp"] > soft[x["id"]]["cp"]):
            soft[x["id"]] = x
    md = tables(records, cps, scores, night, starts_bpb, soft)
    (args.out / "tables.md").write_text(md)
    written = plots(records, cps, scores, night, starts_bpb, args.out)
    payload = {"formalism": "note:training-20261102T1817Z-report-training-formalism-v1", "grid_c_over_w": GRID_C,
               "grid_k_bits": GRID_K, "targets": TARGETS, "reference": REFERENCE, "tau": TAU, "act_bits": ACT_BITS,
               "modes": MODES, "gate_factors": {k: v for k, v in g.items() if k != "definitions"},
               "records": records, "checkpoints": cps, "scores": scores}
    (args.out / "rescore_v1.json").write_text(json.dumps(payload, indent=1, default=float))
    print(f"{len(records)} records, {len(cps)} checkpoints -> {args.out}: tables.md, rescore_v1.json, "
          + ", ".join(p.name for p in written))


if __name__ == "__main__":
    main()
