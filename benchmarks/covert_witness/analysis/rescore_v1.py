#!/usr/bin/env python3
"""The per-record quantities of the v2.3 search's valid records under the training formalism v2,
which te_core.rescore scores: this file holds no formula of its own (Fit, Slowdown's rho, the attack side, W*, H and S are te_core.formalism's and te_core.frontier's).

It reads the night's state as the scorer left it: `submissions.json` (the pinned versions' regimes, `load`: the valid
records' checkpoints on the starts `starts.json` has), the launch records in `index.jsonl` (trainable set, arguments, tokens) and `starts.json` from the scorer's
directory, and te_core's ledger (`index.jsonl`, every checkpoint's counted operations, joined on run id and MACs).

    PYTHONPATH=. uv run python benchmarks/covert_witness/analysis/rescore_v1.py --out /tmp/te-v1-quantities

writes them to OUT/quantities.json; `python -m te_core.rescore` (from benchmarks/covert_witness) loads this file and
scores them. Units are the formalism's: |C| and w in Boolean gates, tau, a, Delta and Delta_grad in bits. Each counted
operation is converted with the gate count of the circuit Definition that computes it (`gate_factors`). The quantities,
each with the approximation it rests on:

- Delta = 8·|E| of the checkpoint's E, with u(1) = Delta: every token's path reads the whole update (dense blocks, and
  the tied head reads the whole embedding).
- Delta_grad = the trainable values' gradient and optimizer state at the run's widths (`delta_grad`).
- Delta_t = the weights in the form the update rule reads and writes (the coordinator's 00:47Z ruling): a
  straight-through run's fp32 latents, 32 bits a value, as its record's notes count them (`latent_bits`); for every
  other run te_core takes Delta.
- token blocks: each token's forward and backward, so their gates T are the checkpoint's `fwd` and `bwd` operations
  (the optimizer's, `other`, are in none) and sigma = T/G, G the checkpoint's gates.
- affected gates in token blocks = T − nu·F: F the checkpoint's forward gates, nu the share of a token's forward
  gates strictly below the lowest updated position (`per_token`). Everything else in a token block (the forward from
  there up, attention over affected K/V, the backward) depends on theta'; the first step's forward is ignored.
- t, the frozen tail's share of a token's affected work above the highest updated matrix (`per_token`; the tied head
  is one for a run that leaves the embedding frozen), and whether every updated matrix is in one block (`one_layer`).
- u and u_in, A1's and A1''s update profiles, as points (x, bits): evenly spread, x·u(1), where t is 0; with a frozen
  tail (t > 0) u is 0 up to t and even over the changed rest, and u_in jumps to the changed part's full update u(1)
  at once (`profiles`).
- w_fwd = the start's forward gates per token (attention over the full 1,024 context, as te_core counts it); tokens =
  the checkpoint's forward gates over w_fwd.
- tau = 16 bits (a token id, ceil(log2 50,304)); a = d_model · 16 bits.
- family, P4's (`family`): full, adapter, partial, update-code, state-compressed, activation-code, seed or data.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

HOME = Path.home() / ".research"
TAU = 16
ACT_BITS = 16
# te_core.flops.FUSED_OPTIMIZER_MACS: the counter prices one fused AdamW element as 8 MACs (te_core imports torch)
FUSED_ADAMW_MACS = 8
LF_STAGES = {"attn_norm": 0, "wq": 1, "wk": 1, "wv": 1, "wo": 2, "mlp_norm": 3, "w_gate": 4, "w_up": 4, "w_down": 5}
REFS = ("ref-optim", "ref-data")
LATENT_BITS = 32


def gate_factors(cache: Path) -> dict:
    """Boolean gates per counted MAC, from the circuits' Definitions: a bf16 dot product with an f32 accumulator
    (`DotBf16{K=64}` on Hopper's wgmma dot16), attention's score-softmax and value rows (`QkSoftmaxRow{T=32, D=64}`,
    `PvRow{T=32, K=32, D=64}`) and torch's AdamW element (`AdamWForeach{N=1}`)."""
    if cache.exists():
        return json.loads(cache.read_text())
    from verity.circuits.defs import bind
    from verity.circuits.evaluation.bits import compile as compile_bits
    from verity.definitions.ops.boolean import gemm as BG
    from verity.definitions.ops.boolean import softmax_row as SR
    from verity.definitions.registry import torch_optim as TO

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
            m = re.fullmatch(r"blocks\.(\d+)\.(\w+?)(?:\.weight|\.bias)?", name)
            two = len(shape) == 2
            nodes.append({"param": name, "layer": int(m.group(1)) + 1, "stage": LF_STAGES[m.group(2)],
                          "macs": shape[0] * shape[1] if two else 0, "d_in": shape[1] if two else shape[0],
                          "d_out": shape[0]})
    return sorted(nodes, key=lambda n: (n["layer"], n["stage"], n["param"]))


def per_token(start: dict, trainable: dict, g: dict, forward_only: bool) -> dict:
    """A token's gates by position: the forward (weights, unmerged adapters, attention over the full context) and, from
    the lowest updated position up, the backward (input gradients, full matrices' weight gradients, adapters' backward,
    attention's 2.5x). Returns ν (the forward's share strictly below the lowest updated position), the frozen tail's
    share t of the affected work (above the highest updated matrix), and whether every updated matrix is in one block."""
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
    affected = sum(f + b for p, f, b in items if p >= first)
    tail = sum(f + b for p, f, b in items if p > last)
    layers = {p[0] for p in matrices}
    return {"first": list(first), "last": list(last), "nu": below / fwd_all, "t": tail / affected if affected else 0.0,
            "one_layer": len(layers) == 1 and 1 <= next(iter(layers)) <= cfg["n_layer"],
            "fwd_gates_per_token": fwd_all, "affected_gates_per_token": affected}


def profiles(u1: float, t: float) -> tuple[list[list[float]], list[list[float]]]:
    """(u, u_in) as points (x, bits) for a per-token update u(1) and a frozen tail's share t (module docstring)."""
    if 0 < t < 1:
        return [[0.0, 0.0], [t, 0.0], [1.0, u1]], [[0.0, 0.0], [0.0, u1], [1.0, u1]]
    return [[0.0, 0.0], [1.0, u1]], [[0.0, 0.0], [1.0, u1]]


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
    projected scalar and seed, 64 bits. A record whose notes state it (notes.delta_grad.bits) is priced at that
    instead (`load`, `build`)."""
    opt, shapes = optimizer(rec), {n: s for n, (_, s) in start["tensors"].items()}
    full = {n: math.prod(shapes[n]) for n in trainable.get("full", [])}
    adapters = sum(int(r) * (shapes[n][0] + shapes[n][1]) for n, r in trainable.get("adapters", {}).items())
    if opt == "seed":
        return 64.0, opt
    if opt == "galore":
        return 32.0 * sum(full.values()), opt
    mats = sum(v for n, v in full.items() if n.startswith("blocks.") and len(shapes[n]) == 2) if opt == "muon" else 0
    return 64.0 * mats + 96.0 * (sum(full.values()) - mats + adapters), opt


def latent_bits(notes: dict) -> tuple[float | None, str | None]:
    """A straight-through run's fp32 latents in bits, as its record's notes count them, and which note; (None, None)
    for a run that trains no code through its rounding (its Delta_t is Delta)."""
    dg = notes.get("delta_grad") if isinstance(notes.get("delta_grad"), dict) else {}
    acc = notes.get("accounting") if isinstance(notes.get("accounting"), dict) else {}
    if dg.get("latent_bits"):
        return float(dg["latent_bits"]), "delta_grad.latent_bits"
    if acc.get("delta_t_bits"):
        return float(acc["delta_t_bits"]), "accounting.delta_t_bits"
    if notes.get("latent_bytes_max"):
        return 8.0 * notes["latent_bytes_max"], "latent_bytes_max"
    if notes.get("coded_weights"):
        return float(LATENT_BITS * notes["coded_weights"]), "coded_weights"
    if notes.get("mode") == "qat" and notes.get("trained_parameters"):
        return float(LATENT_BITS * notes["trained_parameters"]), "trained_parameters (qat)"
    if "fp32 latents" in str(dg.get("what", "")) and dg.get("values"):
        return float(LATENT_BITS * dg["values"]), "delta_grad.values"
    return None, None


def family(r: dict) -> str:
    """P4's family of a record (the scorer's rule, v1cycle's): how its update is carried."""
    mech, fmt, opt, d = r["mechanism"] or "", r["E_format"] or "", r["optimizer"], r["direction"]
    if opt == "seed" or fmt == "seedlog":
        return "seed"
    if "act" in mech.split("-")[0]:
        return "activation-code"
    if fmt == "quant" or "qat" in mech or "4bit" in mech or "signrows" in mech:
        return "update-code"
    if opt == "galore" or d == "lowrank-restart":
        return "state-compressed"
    if fmt.startswith("lora"):
        return "adapter"
    if d == "ref-data" and "window" in mech:
        return "data"
    if d in REFS and not mech.startswith("top"):
        return "full"
    return "partial"


def pinned_regime(regime: str) -> bool:
    """A record of a pinned version (the scorer's `regime_of`): `current`, the pin's version, or `pin V`, an earlier
    one's, so the pilot's records keep their quantities once a newer version is pinned (te_core.frontier keeps the
    pin's and the pilot's); not `paired` or `earlier`."""
    return regime == "current" or regime.startswith("pin ")


def load(scorer: Path, ledger: Path):
    launch = {}
    for line in (scorer / "index.jsonl").open():
        r = json.loads(line)
        if r.get("source") == "launch.py":
            launch[r["id"]] = r
    starts = json.loads((scorer / "starts.json").read_text())
    subs = [x for x in json.loads((scorer / "submissions.json").read_text())
            if pinned_regime(x["regime"]) and launch.get(x["id"], {}).get("start") in starts]
    for rid in {x["id"] for x in subs}:
        meta = scorer / "records" / rid.split(":", 1)[-1] / "meta.json"
        notes = (json.loads(meta.read_text()).get("notes") or {}) if meta.exists() else {}
        launch[rid]["latent_bits"], launch[rid]["latent_from"] = latent_bits(notes)
        stated = (notes.get("delta_grad") or {}).get("bits") if isinstance(notes.get("delta_grad"), dict) else None
        launch[rid]["delta_grad_stated"] = float(stated) if isinstance(stated, (int, float)) and stated > 0 else None
    runs = {launch[x["id"]]["run"] for x in subs}
    counted = {}
    for line in (ledger / "index.jsonl").open():
        r = json.loads(line)
        if r.get("run_id") in runs:
            counted[(r["run_id"], r["macs"])] = r["macs_log"]
    return subs, launch, starts, counted


def build(subs, launch, starts, counted, g) -> tuple[dict, list[dict]]:
    """Each record's quantities and each checkpoint's (module docstring). A record that trains a tensor its start
    doesn't hold (a grown model: ref-data's grow-muon adds a block) is left out, with its checkpoints: the start's
    positions and shapes are all this analysis prices from."""
    records, cps = {}, []
    for rid in sorted({x["id"] for x in subs}):
        rec = launch[rid]
        start = starts[rec["start"]]
        trainable = ast.literal_eval(rec["trainable"]) if isinstance(rec["trainable"], str) else rec["trainable"]
        if not {*trainable.get("full", []), *trainable.get("adapters", {})} <= start["tensors"].keys():
            continue
        tok = rec.get("tokens") or {}
        forward_only = not tok.get("train") and bool(tok.get("forward"))
        pt = per_token(start, trainable, g, forward_only)
        dg, opt = delta_grad(rec, start, trainable)
        dg_from = "notes.delta_grad.bits" if rec.get("delta_grad_stated") else f"inferred ({opt})"
        dg = rec.get("delta_grad_stated") or dg
        cfg = start["config"]
        w_fwd = sum(n["macs"] for n in positions(start)) * g["mm"] + cfg["n_layer"] * 2 * cfg["d_model"] * cfg["ctx"] * \
            op_factor("_scaled_dot_product", g)
        records[rid] = {"id": rid, "start": rec["start"], "direction": rec["direction"], "mechanism": rec.get("mechanism"),
                        "class": rec.get("class"), "E_format": rec.get("E_format"), "note": rec.get("note", "")[:160],
                        "reference": rec["direction"] in REFS, "optimizer": opt, "delta_grad": dg,
                        "delta_grad_from": dg_from,
                        "delta_t": rec.get("latent_bits"), "delta_t_from": rec.get("latent_from"),
                        "w_fwd": w_fwd, "a": cfg["d_model"] * ACT_BITS, **pt}
        records[rid]["family"] = family(records[rid])
    for x in subs:
        if x["id"] not in records:
            continue
        r = records[x["id"]]
        ops = counted.get((launch[x["id"]]["run"], x["macs"]))
        if ops is None:
            G = F = T = x["macs"] * g["mm"]
            source = "submission MACs (no ledger row): all forward, gates at the product rate"
        else:
            G = sum(m * op_factor(k, g) for k, (_, m) in ops["ops"].items())
            F = sum(m * op_factor(k, g) for k, (_, m) in ops["ops"].items() if k.startswith("fwd "))
            T = sum(m * op_factor(k, g) for k, (_, m) in ops["ops"].items() if not k.startswith("other "))
            source = "ledger"
        delta = 8.0 * x["E_bytes"]
        u, u_in = profiles(delta, r["t"])
        cps.append({"id": x["id"], "start": x["start"], "cp": x["cp"], "macs": x["macs"], "gates": G, "fwd_gates": F,
                    "token_gates": T, "affected_gates": T - r["nu"] * F, "sigma": T / G,
                    "tokens": F / r["fwd_gates_per_token"], "delta": delta, "u": u, "u_in": u_in, "bpb": x["val_bpb"],
                    "schedule_end": bool(x.get("schedule_end")), "source": source})
    return records, cps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorer", type=Path, default=HOME / "te-scorer")
    ap.add_argument("--ledger", type=Path, default=HOME / "te-ledger")
    ap.add_argument("--out", type=Path, default=Path("/tmp/te-v1-quantities"))
    ap.add_argument("--gates", type=Path, help="gate factors JSON (default: <out>/gates.json, computed if absent)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = gate_factors(args.gates or args.out / "gates.json")
    records, cps = build(*load(args.scorer, args.ledger), g)
    payload = {"formalism": "note:training-20261102T2215Z-report-training-formalism-v2", "tau": TAU,
               "act_bits": ACT_BITS, "gate_factors": {k: v for k, v in g.items() if k != "definitions"},
               "records": records, "checkpoints": cps}
    (args.out / "quantities.json").write_text(json.dumps(payload, indent=1, default=float))
    print(f"{len(records)} records, {len(cps)} checkpoints -> {args.out / 'quantities.json'} "
          "(scored by python -m te_core.rescore)")


if __name__ == "__main__":
    main()
