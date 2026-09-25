"""Goal-1 characterization and the MAC audit.

Three things live here:

* :func:`breakdown` -- a per-matmul-class / per-shape table of an op graph.  Factories never set
  ``op.name`` (it is ``None`` everywhere), so the label is *derived*: the chain of enclosing composites
  (``Block``, ``BlockBwd/AttnSeqBwd``, ``MoeMlp/ExpertMlp``, ...), whether the ``B`` operand is a root
  parameter (and which one), and whether the op is downstream of the loss gradient.  That is enough to tell
  forward, checkpoint-recompute, activation-gradient (``dgrad``), weight-gradient (``wgrad``) and attention
  products apart, which is what a MAC audit needs.
* :func:`analytic_macs` -- two closed-form MAC-per-token estimates from the ``ModelConfig`` alone: the
  *textbook* count (forward = active weights + attention, training = 3x weights, 6 attention products) and
  the *IR-faithful* count that models exactly what the circuits in ``algorithms/transformer.py`` do
  (activation checkpointing recomputes the forward inside every block backward, the MoE factory has no
  expert backward, ...).  :func:`audit` compares both with the extracted graph.
* :func:`characterize` / :func:`characterize_all` -- the Goal-1 table (state floor ``P``, ADW, ratios,
  operand sizes, extraction time) for realistic workloads.

Run from the repo root::

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.characterize --audit          # results/audit.{json,md}
    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.characterize --table --quick  # results/characterization.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from accumulation.algorithms.registry import ALGORITHMS, BuiltProgram, build
from accumulation.bounds.adw import adw as compute_adw
from accumulation.configs import MODELS, ModelConfig, Workload
from accumulation.graph import OpGraph, extract
from accumulation.graph.opgraph import _is_terminal
from accumulation.sweeps.guard import DRIVER_LIMITS, check_limits
from accumulation.sweeps.util import (RESULTS_DIR, StepTimeout, dumps, fmt, fmt_bytes, md_table, time_limit,
                                      write_json)

REPO = Path(__file__).resolve().parents[2]

REAL_MODELS = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b", "mixtral-8x7b", "deepseek-v3", "qwen3-235b-a22b")
DENSE_ALGS = ("inference-dense", "forward-nonfixed", "pretrain-dense", "local-sgd", "lora", "hidden-adapter",
              "rl-policy-gradient", "es")
MOE_ALGS = ("inference-moe", "pretrain-moe")


# ---------------------------------------------------------------------------------------------------------
# scope labels and matmul classification
# ---------------------------------------------------------------------------------------------------------

def _short(name: str) -> str:
    return name[3:] if name.startswith("Acc") else name


def scope_labels(program) -> dict[tuple, tuple[str, ...]]:
    """``op.key -> (enclosing composite names ...)`` by re-walking the hierarchy exactly like the extractor
    (``call``/``batch`` nodes descend, terminals and scans stop)."""
    out: dict[tuple, tuple[str, ...]] = {}

    def walk(spec, path: tuple, names: tuple):
        for k, node in enumerate(spec.body.nodes):
            fn = node.fn
            if node.form == "scan" or _is_terminal(fn):
                out[(path, k)] = names
            elif node.form in ("call", "batch"):
                walk(fn, path + (k,), names + (_short(fn.name),))

    walk(program.fn, (), ())
    return out


def _after_loss(g: OpGraph) -> list[bool]:
    """``after[op] = op is a loss-gradient op or transitively consumes one`` (iterative: op ids are topological)."""
    after = [False] * len(g.ops)
    for o in g.ops:
        if o.kind == "lossgrad":
            after[o.id] = True
            continue
        for e in o.inputs:
            p = g.tensors[e.src].producer
            if p is not None and after[p]:
                after[o.id] = True
                break
    return after


def _wgrad_param_names(g: OpGraph) -> dict[int, str]:
    """op id of a weight-gradient producer -> the parameter it updates (via the ``SgdUpdate`` consumer)."""
    out: dict[int, str] = {}
    for o in g.ops:
        if o.kind not in ("sgdupdate",):
            continue
        pname = None
        gprod = None
        for e in o.inputs:
            t = g.tensors[e.src]
            if t.kind == "param" and t.role in ("accumulated", "carried", "fixed"):
                pname = pname or t.name
            elif t.kind == "op" and t.producer is not None and g.ops[t.producer].kind == "matmul":
                gprod = t.producer
        if pname is not None and gprod is not None:
            out[gprod] = pname
    return out


def classify_matmul(g: OpGraph, op, scopes: dict, after_loss: list[bool], wgrad_names: dict[int, str]) -> tuple[str, str]:
    """Return ``(class, label)`` for a matmul op.

    ``class`` is one of ``wt-fwd | wt-fwd-recompute | wt-dgrad | wt-wgrad | attn-fwd | attn-fwd-recompute |
    attn-bwd | other``; ``label`` adds the weight name where it is known (``wt-fwd[Wq]``)."""
    scope = scopes.get(op.key, ())
    a_edge = op.inputs[0] if op.inputs else None
    b_edge = op.inputs[-1] if op.inputs else None
    b_t = g.tensors[b_edge.src] if b_edge is not None else None
    in_bwd_scope = any(s.endswith("Bwd") for s in scope)
    in_attn = any(s.startswith("AttnSeq") for s in scope)
    post = after_loss[op.id]
    if op.fn_id.startswith("AccMatmulTT"):
        # weight-gradient composite: edges (dY, X) both token-major; M = weight rows, N = weight cols, K = tokens
        nm = wgrad_names.get(op.id)
        return "wt-wgrad", f"wt-wgrad[{nm}]" if nm else f"wt-wgrad[{op.statics.get('M')}x{op.statics.get('N')}]"
    if in_attn:
        if post:
            cls = "attn-bwd"
        else:
            cls = "attn-fwd-recompute" if in_bwd_scope else "attn-fwd"
        return cls, cls
    if b_t is not None and b_t.kind == "param":
        if post:
            cls = "wt-dgrad"
        else:
            cls = "wt-fwd-recompute" if in_bwd_scope else "wt-fwd"
        return cls, f"{cls}[{b_t.name}]"
    if post:
        nm = wgrad_names.get(op.id)
        return "wt-wgrad", f"wt-wgrad[{nm}]" if nm else f"wt-wgrad[{op.statics.get('M')}x{op.statics.get('N')}]"
    # an activation x activation product outside attention and before the loss (does not occur today)
    return "other", "other"


def breakdown(g: OpGraph, program=None, adw=None) -> dict:
    """Per-class / per-shape MAC breakdown of ``g``.

    Returns ``{"matmul": [row...], "by_class": {cls: macs}, "by_kind": {kind: work}, "totals": {...}}`` where a
    matmul row aggregates ops with the same ``(label, scope, M, N, K, copies)``: ``n_ops``, ``macs``
    (``M*N*K*copies*n_ops``), ``work`` (MAC-equivalents incl. the chunk/round overhead) and ``adw`` (whether
    the ops depend on accumulated/carried state).  ``program`` (a ``verity_ir.Program`` or ``BuiltProgram``)
    enables scope labels; without it the scope column is empty and recompute is not distinguished."""
    prog = getattr(program, "program", program)
    scopes = scope_labels(prog) if prog is not None else {}
    after = _after_loss(g)
    wnames = _wgrad_param_names(g)
    adw = adw or compute_adw(g)
    agg: dict[tuple, dict] = {}
    by_class: dict[str, int] = defaultdict(int)
    for o in g.ops:
        if o.kind != "matmul":
            continue
        cls, label = classify_matmul(g, o, scopes, after, wnames)
        scope = "/".join(scopes.get(o.key, ())) if scopes else ""
        s = o.statics
        key = (label, scope, s["M"], s["N"], s["K"], o.copies)
        row = agg.get(key)
        if row is None:
            row = agg[key] = {"label": label, "class": cls, "scope": scope, "M": s["M"], "N": s["N"], "K": s["K"],
                              "copies": o.copies, "n_ops": 0, "macs": 0, "work": 0, "adw": bool(adw.per_op.get(o.id))}
        row["n_ops"] += 1
        row["macs"] += o.macs()
        row["work"] += o.work
        by_class[cls] += o.macs()
    rows = sorted(agg.values(), key=lambda r: -r["macs"])
    by_kind: dict[str, int] = defaultdict(int)
    for o in g.ops:
        by_kind[o.kind] += o.work
    tot_macs = sum(r["macs"] for r in rows)
    return {"matmul": rows, "by_class": dict(by_class), "by_kind": dict(by_kind),
            "totals": {"macs": tot_macs, "work": g.total_work(), "adw_matmul_macs": adw.matmul_macs, "adw": adw.total,
                       "ops": len(g.ops), "matmul_ops": sum(1 for o in g.ops if o.kind == "matmul")}}


def format_breakdown(bd: dict, tokens: int | None = None, title: str = "") -> str:
    tot = bd["totals"]["macs"] or 1
    rows = []
    for r in bd["matmul"]:
        rr = dict(r)
        rr["share"] = r["macs"] / tot
        rr["per_token"] = r["macs"] / tokens if tokens else None
        rows.append(rr)
    cols = [("label", "label"), ("scope", "scope"), ("M", "M"), ("N", "N"), ("K", "K"), ("copies", "copies"),
            ("n_ops", "ops"), ("macs", "MACs"), ("per_token", "MACs/token"), ("share", "share"), ("adw", "ADW")]
    out = [f"### {title}" if title else "", "", md_table(rows, cols, {"share": lambda x: f"{100 * x:.1f}%" if x is not None else "-"}), ""]
    cls_rows = [{"class": k, "macs": v, "share": v / tot} for k, v in sorted(bd["by_class"].items(), key=lambda kv: -kv[1])]
    out += ["By class:", "", md_table(cls_rows, [("class", "class"), ("macs", "MACs"), ("share", "share")],
                                       {"share": lambda x: f"{100 * x:.1f}%"}), ""]
    kind_rows = [{"kind": k, "work": v} for k, v in sorted(bd["by_kind"].items(), key=lambda kv: -kv[1])]
    out += ["Work by op kind (MAC-equivalents, all ops):", "", md_table(kind_rows, [("kind", "kind"), ("work", "work")]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------
# analytic MAC model
# ---------------------------------------------------------------------------------------------------------

def analytic_macs(alg: str, cfg: ModelConfig, wl: Workload) -> dict:
    """Closed-form matmul MACs **per token** for ``alg`` on ``cfg``/``wl``.

    Returns ``{"ir_model": ..., "naive": ..., "components": {...}, "notes": [...]}``.

    ``naive`` is the textbook estimate (forward: active weights + 2 attention products per layer; training:
    3x the weight MACs and 6 attention products, no recompute).  ``ir_model`` is what the IR is *written* to
    compute (``algorithms/transformer.py``): full ``S x S`` attention (no causal mask), activation
    checkpointing (``BlockBwd`` recomputes the whole forward of the block except the last ``Wd`` product,
    ``AttnSeqBwd`` recomputes the scores once more), LoRA base products recomputed in ``LoraBlockBwd``
    (including the unused ``Wd`` base), MoE = dense-style backward with a full expert backward, ES = population x forward."""
    D, F, AO, KW, V, L, S = cfg.d, cfg.ffn, cfg.attn_out, cfg.kv_width, cfg.vocab, cfg.layers, wl.seq
    E, T, SX, Ld = cfg.experts, cfg.topk, cfg.shared_experts, cfg.dense_layers
    Lm = cfg.moe_layers
    w_attn = 2 * D * AO + 2 * D * KW            # Wq, Wo (D x AO) and Wk, Wv (D x KW)
    w_mlp = 3 * D * F                           # Wg, Wu, Wd
    a1 = S * cfg.head_dim * cfg.heads           # one attention product (QK^T or PV) per token, full S x S
    head = V * D
    comp: dict[str, float] = {"w_attn_per_layer": w_attn, "w_mlp_per_layer": w_mlp, "attn_product_per_layer": a1, "head": head}
    notes: list[str] = ["attention is full S x S per head (no causal mask): each product costs S*DH*NH MACs per token"]
    fwd_dense_layer = w_attn + w_mlp + 2 * a1
    if alg in ("inference-dense", "forward-nonfixed", "es", "inference-decode"):
        ir = naive = L * fwd_dense_layer + head
        if alg == "inference-decode":
            notes.append("decode: T teacher-forced rows per sequence attend over the S cached positions (2 products of S*DH*NH "
                         "per token, same as full-S prefill); the KV cache is a token-role root, so its products are labelled "
                         "wt-fwd[K_cache]/[V_cache] in the breakdown")
        if alg == "es":
            notes.append("ES: population x forward over tokens/population each = one forward per token; perturbations are not matmuls")
    elif alg == "inference-session":
        from accumulation.algorithms.registry import _session_PT
        Pp, Tg = _session_PT(wl)                         # prompt / generated tokens per sequence, S = Pp + Tg
        a_sess = cfg.head_dim * cfg.heads * (Pp * Pp + Tg * Pp) / S   # per token: prompt rows attend over Pp, new rows over Pp
        ir = L * (w_attn + w_mlp + 2 * a_sess) + head
        naive = L * fwd_dense_layer + head
        comp["attn_product_per_layer"] = a_sess
        notes.append("session: prefill (Pp x Pp attention) + teacher-forced decode of Tg rows attending over the internal "
                     "prompt cache only (Tg x Pp); nobody attends over the Tg new keys, so attention MACs are "
                     "(Pp^2 + Tg*Pp)/S per product instead of S")
    elif alg in ("pretrain-dense", "rl-policy-gradient", "local-sgd"):
        recompute = (w_attn + w_mlp - D * F) + 2 * a1   # BlockBwd recomputes everything up to h (not the Wd product)
        bwd = 2 * (w_attn + w_mlp) + 5 * a1            # dgrad + wgrad per weight; attention: sc recompute, dP, dQ, dK, dV
        ir = L * (fwd_dense_layer + recompute + bwd) + 3 * head
        naive = 3 * (L * (w_attn + w_mlp) + head) + L * 6 * a1
        comp.update(recompute_per_layer=recompute, bwd_per_layer=bwd)
        notes.append("activation checkpointing: BlockBwd recomputes the block forward (all products but Wd) and AttnSeqBwd "
                     "recomputes QK^T; weight MACs = 4x forward - Wd, attention = 9 products per layer instead of 6")
        if alg == "local-sgd":
            notes.append("local-sgd: local_steps chained steps over tokens/local_steps each; identical MACs per token")
        if alg == "rl-policy-gradient":
            notes.append("RL: one teacher-forced forward+backward over all tokens (gen_tokens only changes which rows carry an advantage)")
    elif alg in ("lora", "hidden-adapter"):
        r = wl.rank
        La = L if wl.adapted_layers is None else min(wl.adapted_layers, L)
        sproj = 7 * D + 2 * AO + 2 * KW + 3 * F        # sum over the 7 projections of (in + out) dims
        ad_fwd = r * sproj                             # low (Q x r x in) + up (Q x out x r) for every slot (active or not)
        fwd_ad_layer = w_attn + w_mlp + ad_fwd + 2 * a1
        bwd_ad_layer = (w_attn + w_mlp + ad_fwd + 2 * a1) + (w_attn + w_mlp) + 2 * ad_fwd + 5 * a1
        ir = (L - La) * fwd_dense_layer + La * (fwd_ad_layer + bwd_ad_layer) + 2 * head
        naive = (L - La) * fwd_dense_layer + La * (2 * (w_attn + w_mlp) + 3 * ad_fwd + 6 * a1) + 2 * head
        comp.update(adapter_fwd_per_layer=ad_fwd, adapted_layers=La)
        notes.append("LoRA: all 14 adapter slots are computed even when inactive (fixed); LoraBlockBwd recomputes the "
                     "whole block forward including the unused Wd base product; no LM-head wgrad (head is frozen)")
    elif alg in ("inference-moe", "pretrain-moe", "forward-nonfixed-moe"):
        w_moe = E * D + 3 * D * F * (T + SX)           # router + top-k experts + shared experts, per token
        ir = Ld * fwd_dense_layer + Lm * (w_attn + w_moe + 2 * a1) + head
        naive = ir
        comp.update(w_moe_per_layer=w_moe)
        notes.append("MoE: static balanced routing sends TOPK*Q/E rows to each expert -> 3*D*F*TOPK MACs per token, "
                     "exactly the active expert width")
        if alg == "pretrain-moe":
            # dense layers: as pretrain-dense.  MoE layers (MoeBlockBwd): the recompute covers the whole block
            # forward (router and expert Wd included), dgrad skips the router, the router wgrad flows only through
            # the TOPK selected gates (D*TOPK MACs per token); attention as in the dense backward (5 products).
            recompute_dense = (w_attn + w_mlp - D * F) + 2 * a1
            bwd_dense = 2 * (w_attn + w_mlp) + 5 * a1
            recompute_moe = w_attn + w_moe + 2 * a1
            bwd_moe = 2 * (w_attn + w_moe) - 2 * E * D + D * T + 5 * a1
            ir = (Ld * (fwd_dense_layer + recompute_dense + bwd_dense)
                  + Lm * ((w_attn + w_moe + 2 * a1) + recompute_moe + bwd_moe) + 3 * head)
            naive = 3 * (Ld * (w_attn + w_mlp) + Lm * (w_attn + w_moe) + head) + L * 6 * a1
            comp.update(recompute_moe_per_layer=recompute_moe, bwd_moe_per_layer=bwd_moe)
            notes.append("pretrain-moe: full expert backward (MoeBlockBwd / ExpertMlpBwd); the MoE block recompute includes "
                         "the router and the expert Wd products, the router has no dgrad and a TOPK-sparse wgrad")
    else:
        raise KeyError(alg)
    return {"ir_model": float(ir), "naive": float(naive), "components": comp, "notes": notes}


# ---------------------------------------------------------------------------------------------------------
# characterization
# ---------------------------------------------------------------------------------------------------------

def _wl_dict(wl: Workload) -> dict:
    d = asdict(wl)
    d["adapted"] = "".join(wl.adapted)
    return d


def _weight_matmul_stats(g: OpGraph) -> tuple[int, int, int]:
    """(#matmul ops whose B operand is a root param, #distinct such params, largest matmul operand bytes per copy).

    Operand bytes are ``M*K`` and ``N*K`` elements for both ``MatmulT`` (A: M x K, B: N x K) and ``MatmulTT``
    (dY: K x M, X: K x N)."""
    n_wt = 0
    params: set[int] = set()
    largest = 0
    for o in g.ops:
        if o.kind != "matmul" or not o.inputs:
            continue
        s = o.statics
        for e, rows in ((o.inputs[0], s["M"]), (o.inputs[-1], s["N"])):
            largest = max(largest, rows * s["K"] * g.tensors[e.src].width // 8)
        b = g.tensors[o.inputs[-1].src]
        if b.kind == "param":
            n_wt += 1
            params.add(b.id)
    return n_wt, len(params), largest


def characterize(alg: str, model_name: str, wl: Workload, *, bp: BuiltProgram | None = None, g: OpGraph | None = None,
                 time_limit_s: float | None = None) -> dict:
    """The Goal-1 row for one ``(algorithm, model, workload)``.  Builds and extracts unless ``bp``/``g`` are given.
    Never raises for unsupported / timed-out cases: the row carries an ``error`` field instead."""
    cfg = MODELS[model_name]
    A = ALGORITHMS[alg]
    row = _stub_row(alg, model_name, wl)
    why = A.supports(cfg, wl)
    if why:
        row["error"] = f"unsupported: {why}"
        return row
    t0 = time.perf_counter()
    try:
        with time_limit(time_limit_s):
            if bp is None:
                bp = build(alg, cfg, wl)
            t1 = time.perf_counter()
            if g is None:
                g = extract(bp)
            t2 = time.perf_counter()
    except StepTimeout as e:
        row.update(error=f"timeout: {e}", t_total=time.perf_counter() - t0)
        return row
    except Exception as e:  # noqa: BLE001 -- a broken factory must not kill the table
        row.update(error=f"{type(e).__name__}: {e}", t_total=time.perf_counter() - t0)
        return row
    a = compute_adw(g)
    P = bp.state_bytes()
    roles_bytes = {r: bp.param_bytes(r) for r in ("fixed", "accumulated", "carried", "token", "seed")}
    n_wt, n_wt_params, largest = _weight_matmul_stats(g)
    Q = wl.tokens
    an = analytic_macs(alg, cfg, wl)
    ir_per_tok = a.all_macs / Q
    row.update({
        "notes": dict(bp.notes), "t_build": t1 - t0, "t_extract": t2 - t1,
        "P": P, "bytes_by_role": roles_bytes, "token_bytes": roles_bytes["token"], "seed_bytes": roles_bytes["seed"],
        "adw_total": a.total, "adw_matmul": a.matmul, "adw_matmul_macs": a.matmul_macs,
        "all_work": a.all_work, "all_macs": a.all_macs, "adw_fraction": a.fraction,
        "adw_per_token": a.total / Q, "adw_macs_per_token": a.matmul_macs / Q, "macs_per_token": ir_per_tok,
        "P_per_token": P / Q, "adw_per_P": (a.total / P) if P else None, "adw_macs_per_P": (a.matmul_macs / P) if P else None,
        "n_ops": len(g.ops), "n_matmul_ops": sum(1 for o in g.ops if o.kind == "matmul"),
        "n_weight_matmul_ops": n_wt, "n_weight_params_in_matmuls": n_wt_params, "largest_operand_bytes": largest,
        "analytic_ir_model_per_token": an["ir_model"], "analytic_naive_per_token": an["naive"],
        "ratio_ir_over_model": ir_per_tok / an["ir_model"] if an["ir_model"] else None,
        "ratio_ir_over_naive": ir_per_tok / an["naive"] if an["naive"] else None,
    })
    return row


def characterize_guarded(alg: str, model_name: str, wl: Workload, *, time_limit_s: float = 300.0,
                         max_rss_mb: float = 3072.0, log=print) -> dict:
    """:func:`characterize` in a child process under the wall-clock + RSS watchdog (one child at a time).
    A killed child yields a stub row with ``error = "oom-guard: ..."`` / ``"timeout: ..."``; every row carries
    ``peak_rss_mb`` (sampled every 0.5 s, so short runs under-report)."""
    from accumulation.sweeps.guard import run_guarded
    spec = {"alg": alg, "model": model_name, "wl": asdict(wl), "time_limit_s": time_limit_s}
    cmd = [sys.executable, "-u", "-m", "accumulation.sweeps.characterize", "--one", json.dumps(spec)]
    res = run_guarded(cmd, timeout_s=time_limit_s + 60, max_rss_mb=max_rss_mb, cwd=REPO,
                      env={"PYTHONPATH": ".", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}, log=log)
    row = None
    if res.status == "ok":
        try:
            row = json.loads(res.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            row = _stub_row(alg, model_name, wl)
            row["error"] = f"child produced no row: {res.stderr.strip()[-300:]}"
    else:
        row = _stub_row(alg, model_name, wl)
        row["error"] = f"{res.status}: {res.detail}"
        row["t_total"] = res.elapsed_s
    row["peak_rss_mb"] = round(res.peak_rss_mb, 1)
    return row


def _wl_from_json(d: dict) -> Workload:
    d = dict(d)
    if isinstance(d.get("adapted"), list):
        d["adapted"] = tuple(d["adapted"])
    return Workload(**d)


def _char_key(r: dict) -> tuple:
    return (r["alg"], r["model"], r["tokens"], r["seq"], r.get("population", 1), r.get("rank", 16), r.get("adapted"),
            r.get("adapted_layers"), r.get("local_steps", 1), r.get("gen_tokens"))


def _stub_row(alg: str, model_name: str, wl: Workload) -> dict:
    cfg = MODELS[model_name]
    return {"alg": alg, "family": ALGORITHMS[alg].family, "model": model_name, **_wl_dict(wl), "Q": wl.tokens,
            "params_total": cfg.total_params(), "params_active": cfg.active_params(), "weight_bytes": cfg.weight_bytes(),
            "simplifications": list(ALGORITHMS[alg].simplifications), "error": None}


TASK2_TOKENS = (4096, 65536, 1 << 20, 4_194_304)
ALL_PROJ = ("q", "k", "v", "o", "g", "u", "d")


def default_workloads(alg: str, cfg: ModelConfig, *, quick: bool = False) -> list[Workload]:
    """Realistic workloads per algorithm family (Task 2): ``Q in {4096, 65536, 1M, 4M}`` at ``seq = 4096`` (plus
    ``seq = 8192`` for the 8B model) for the dense families, ``Q in {4096, 65536}`` for the MoE circuits, the ES /
    local-SGD token counts spread over population / steps, and the LoRA rank x projection variants at ``Q = 4096``.
    ``quick`` keeps only the smallest per family."""
    seqs = (4096,) if quick or cfg.name != "llama3-8b" else (4096, 8192)
    tokens = (4096,) if quick else TASK2_TOKENS
    if alg in ("inference-moe", "pretrain-moe", "forward-nonfixed-moe"):
        tokens = tuple(t for t in tokens if t <= 65536)
    out: list[Workload] = []
    if alg in ("inference-dense", "forward-nonfixed", "pretrain-dense", "inference-moe", "pretrain-moe", "forward-nonfixed-moe"):
        for s in seqs:
            for t in tokens:
                t = max(t, s)
                if t % s == 0 and t % cfg.experts == 0:
                    out.append(Workload(tokens=t, seq=s))
    elif alg in ("lora", "hidden-adapter"):
        for t in tokens:
            out.append(Workload(tokens=t, seq=4096, rank=16, adapted=ALL_PROJ))
        if not quick:
            for r in (1, 16, 64):
                for ad in (("q", "v"), ALL_PROJ):
                    out.append(Workload(tokens=4096, seq=4096, rank=r, adapted=tuple(ad)))
    elif alg == "es":
        # tokens = population x seq x (sequences per member); realistic populations 8-1024 members
        specs = ((2, 8192),) if quick else ((2, 8192), (16, 65536), (8, 1 << 20), (256, 1 << 20), (8, 4_194_304), (1024, 4_194_304))
        for p, t in specs:
            out.append(Workload(tokens=t, seq=4096, population=p))
    elif alg == "rl-policy-gradient":
        for gt in ((2048,) if quick else (256, 2048)):
            out.append(Workload(tokens=4096, seq=4096, gen_tokens=gt))
        if not quick:
            for t in tokens[1:]:
                out.append(Workload(tokens=t, seq=4096, gen_tokens=2048))
    elif alg == "local-sgd":
        specs = ((2, 8192),) if quick else ((1, 4096), (2, 8192), (4, 16384), (2, 65536), (4, 1 << 20), (4, 4_194_304))
        for h, t in specs:
            out.append(Workload(tokens=t, seq=4096, local_steps=h))
    else:
        out.append(Workload(tokens=4096, seq=4096))
    # de-duplicate, smallest first (so a timeout can back off the larger ones)
    seen: set = set()
    uniq = []
    for w in sorted(out, key=lambda w: (w.tokens, w.seq, w.population, w.rank, w.local_steps)):
        k = (w.tokens, w.seq, w.population, w.rank, w.adapted, w.adapted_layers, w.local_steps, w.gen_tokens)
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    return uniq


CHAR_COLS = [("alg", "alg"), ("model", "model"), ("tokens", "Q"), ("seq", "S"), ("extra", "extra"),
             ("params_total", "params"), ("params_active", "active"), ("P", "P (state)"), ("token_bytes", "token B"),
             ("adw_total", "ADW"), ("adw_matmul_macs", "ADW MACs"), ("adw_fraction", "ADW frac"),
             ("adw_per_token", "ADW/tok"), ("P_per_token", "P/tok"), ("adw_per_P", "ADW/P"),
             ("n_ops", "ops"), ("n_weight_matmul_ops", "wt-mm ops"), ("largest_operand_bytes", "max operand"),
             ("t_extract", "extract s"), ("error", "error")]


def _extra(row: dict) -> str:
    bits = []
    if row.get("population", 1) != 1:
        bits.append(f"pop={row['population']}")
    if row["alg"] in ("lora", "hidden-adapter"):
        bits.append(f"r={row['rank']} ad={row['adapted']}")
    if row.get("local_steps", 1) != 1:
        bits.append(f"H={row['local_steps']}")
    if row.get("gen_tokens") is not None:
        bits.append(f"gen={row['gen_tokens']}")
    return " ".join(bits)


def format_characterization(rows: list[dict]) -> str:
    rr = [dict(r, extra=_extra(r)) for r in rows]
    f = {"P": fmt_bytes, "token_bytes": fmt_bytes, "P_per_token": fmt_bytes, "largest_operand_bytes": fmt_bytes,
         "adw_fraction": lambda x: f"{100 * x:.1f}%" if x is not None else "-",
         "t_extract": lambda x: f"{x:.1f}" if x is not None else "-",
         "error": lambda x: x or ""}
    return md_table(rr, CHAR_COLS, f)


def workload_cost(alg: str, cfg: ModelConfig, wl: Workload) -> float:
    """Rough extraction cost (ops ~ active params x tokens; training-style circuits ~3x)."""
    mult = 1.0 if alg.startswith("inference") or alg in ("forward-nonfixed", "forward-nonfixed-moe", "es") else 3.0
    return cfg.active_params() * wl.tokens * mult


def characterize_all(models=REAL_MODELS, algs=None, workloads=None, *, out_dir: Path = RESULTS_DIR, quick: bool = False,
                     time_limit_s: float = 300.0, unsafe: bool = False, guarded: bool = True, max_rss_mb: float = 3072.0,
                     resume: bool = False, log=print) -> list[dict]:
    """Characterize every supported ``(alg, model, workload)``, cheapest first; write ``characterization.{json,md}``
    incrementally (merged with earlier rows, so partial re-runs keep the table complete).

    ``workloads`` may be a list (used for every algorithm) or ``None`` (``default_workloads`` per algorithm).
    With ``guarded`` (default) each row is computed in a child process under a wall clock (``time_limit_s``)
    and an RSS watchdog (``max_rss_mb``), one child at a time; killed rows carry ``error = "oom-guard: ..."``
    or ``"timeout: ..."``.  ``resume`` keeps existing rows that succeeded (or were killed by the guard) and only
    computes the missing / previously skipped ones.  Feasibility gates (rows are still written, with a
    ``skipped: ...`` error):

    * workloads outside :data:`accumulation.sweeps.guard.DRIVER_LIMITS` are skipped unless ``unsafe``;
    * once a workload of some ``(alg, model)`` hits the guard, larger ``Q`` of the same pair are skipped;
    * a workload whose extraction time, extrapolated linearly in ``Q`` from the last successful row of the same
      pair, exceeds ``3 * time_limit_s`` is skipped."""
    algs = list(algs or ALGORITHMS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "characterization.json"
    merged: dict[tuple, dict] = {}
    if json_path.exists():   # merge with earlier rows so partial re-runs (--algs / --models) keep the table complete
        for r in json.loads(json_path.read_text()):
            merged[_char_key(r)] = r
    rows: list[dict] = []

    def _flush():
        for r in rows:
            merged[_char_key(r)] = r
        allrows = sorted(merged.values(), key=lambda r: (REAL_MODELS.index(r["model"]) if r["model"] in REAL_MODELS else 99,
                                                         list(ALGORITHMS).index(r["alg"]), r["tokens"], r["seq"], r.get("population", 1),
                                                         r.get("rank", 16), r.get("adapted", ""), r.get("local_steps", 1), r.get("gen_tokens") or 0))
        write_json(json_path, allrows)
        (out_dir / "characterization.md").write_text(format_characterization(allrows) + "\n")

    todo: list[tuple[float, str, str, Workload]] = []
    for m in models:
        cfg = MODELS[m]
        for alg in algs:
            wls = list(workloads) if workloads is not None else default_workloads(alg, cfg, quick=quick)
            for wl in wls:
                if ALGORITHMS[alg].supports(cfg, wl):
                    continue
                if resume:
                    old = merged.get(_char_key(_stub_row(alg, m, wl)))
                    if old is not None and (not old.get("error") or old["error"].split(":")[0] in ("timeout", "oom-guard")):
                        continue
                todo.append((workload_cost(alg, cfg, wl), m, alg, wl))
    todo.sort(key=lambda t: (t[0], t[1], t[2], t[3].tokens))
    log(f"[characterize] {len(todo)} rows to compute (cheapest first; {len(merged)} already on file)")
    last: dict[tuple, tuple[int, float]] = {}   # (alg, model) -> (tokens, seconds) of its last successful row
    failed_at: dict[tuple, int] = {}            # (alg, model) -> tokens of a row that hit the wall clock / RSS guard
    for r in merged.values():                   # seed the back-off state from earlier rows
        k = (r["alg"], r["model"])
        if r.get("error") and r["error"].split(":")[0] in ("timeout", "oom-guard"):
            failed_at[k] = min(failed_at.get(k, r["tokens"]), r["tokens"])
    for _, m, alg, wl in todo:
        cfg = MODELS[m]
        k = (alg, m)
        t0 = time.perf_counter()
        unsafe_why = check_limits(alg, m, wl, DRIVER_LIMITS)
        if unsafe_why and not unsafe:
            row = _stub_row(alg, m, wl)
            row["error"] = f"skipped: outside extractor safe limits ({unsafe_why}); see extract_scaling"
        elif k in failed_at and wl.tokens >= failed_at[k]:
            row = _stub_row(alg, m, wl)
            row["error"] = f"skipped: Q={failed_at[k]} already hit the guard for {alg}/{m}"
        elif k in last and wl.tokens > last[k][0] and last[k][1] * wl.tokens / last[k][0] > 3 * time_limit_s:
            pred = last[k][1] * wl.tokens / last[k][0]
            row = _stub_row(alg, m, wl)
            row["error"] = f"skipped: predicted extraction {pred:.0f}s > {3 * time_limit_s:.0f}s"
        else:
            row = (characterize_guarded(alg, m, wl, time_limit_s=time_limit_s, max_rss_mb=max_rss_mb, log=log)
                   if guarded else characterize(alg, m, wl, time_limit_s=time_limit_s))
            if not row["error"]:
                last[k] = (wl.tokens, row["t_build"] + row["t_extract"])
            elif row["error"].split(":")[0] in ("timeout", "oom-guard"):
                failed_at[k] = min(failed_at.get(k, wl.tokens), wl.tokens)
        rows.append(row)
        log(f"[characterize] {alg:20s} {m:16s} Q={wl.tokens:<9d} S={wl.seq:<5d} {_extra(row):22s} "
            f"{'ERR ' + row['error'] if row['error'] else f'ops={row['n_ops']} extract={row['t_extract']:.1f}s'} "
            f"rss={row.get('peak_rss_mb', '-')}MB ({time.perf_counter() - t0:.1f}s)")
        _flush()
    return rows


# ---------------------------------------------------------------------------------------------------------
# the MAC audit
# ---------------------------------------------------------------------------------------------------------

def audit_case(alg: str, model_name: str, wl: Workload, *, out_dir: Path | None = None, time_limit_s: float | None = 300) -> dict:
    """IR MACs/token vs the two analytic estimates for one case, plus the breakdown table (written to
    ``out_dir/breakdown_<alg>_<model>.md`` when ``out_dir`` is given)."""
    cfg = MODELS[model_name]
    row = {"alg": alg, "model": model_name, **_wl_dict(wl), "error": None}
    why = ALGORITHMS[alg].supports(cfg, wl)
    if why:
        row["error"] = f"unsupported: {why}"
        return row
    t0 = time.perf_counter()
    try:
        with time_limit(time_limit_s):
            bp = build(alg, cfg, wl)
            g = extract(bp)
    except (StepTimeout, Exception) as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    a = compute_adw(g)
    bd = breakdown(g, bp, a)
    an = analytic_macs(alg, cfg, wl)
    Q = wl.tokens
    ir = a.all_macs / Q
    row.update({
        "t_extract": time.perf_counter() - t0, "n_ops": len(g.ops), "ir_macs_per_token": ir,
        "adw_macs_per_token": a.matmul_macs / Q, "analytic_ir_model": an["ir_model"], "analytic_naive": an["naive"],
        "ratio_vs_model": ir / an["ir_model"], "ratio_vs_naive": ir / an["naive"],
        "active_params": cfg.active_params(), "ir_over_active_params": ir / cfg.active_params(),
        "by_class_per_token": {k: v / Q for k, v in bd["by_class"].items()},
        "notes": an["notes"], "simplifications": list(ALGORITHMS[alg].simplifications),
    })
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"breakdown_{alg}_{model_name}.md").write_text(
            format_breakdown(bd, Q, f"{alg} @ {model_name}, tokens={Q}, seq={wl.seq} ({_extra(dict(row, alg=alg))})") + "\n")
    return row


def audit(models=("llama3-8b", "mixtral-8x7b"), algs=None, *, out_dir: Path = RESULTS_DIR, log=print) -> list[dict]:
    """Run :func:`audit_case` for every supported algorithm on ``models`` (tokens = seq = 4096, population 2, rank 16,
    local_steps 2 -- one sequence per forward everywhere); write ``audit.{json,md}`` and the per-case breakdown tables."""
    rows = []
    for m in models:
        cfg = MODELS[m]
        for alg in (algs or ALGORITHMS):
            wl = audit_workload(alg)
            if ALGORITHMS[alg].supports(cfg, wl):
                continue
            r = audit_case(alg, m, wl, out_dir=out_dir)
            rows.append(r)
            if r["error"]:
                log(f"[audit] {alg:20s} {m:16s} ERR {r['error']}")
            else:
                log(f"[audit] {alg:20s} {m:16s} IR={r['ir_macs_per_token']:.4e}/tok  model={r['analytic_ir_model']:.4e} "
                    f"({r['ratio_vs_model']:.4f})  naive={r['analytic_naive']:.4e} ({r['ratio_vs_naive']:.3f})  "
                    f"IR/active={r['ir_over_active_params']:.2f}  {r['t_extract']:.1f}s")
    # merge with the rows already on file so a partial re-run (--models / --algs) keeps the table complete
    merged: dict[tuple, dict] = {}
    p = out_dir / "audit.json"
    if p.exists():
        for r in json.loads(p.read_text()):
            merged[(r["alg"], r["model"])] = r
    for r in rows:
        merged[(r["alg"], r["model"])] = r
    allrows = sorted(merged.values(), key=lambda r: (REAL_MODELS.index(r["model"]) if r["model"] in REAL_MODELS else 99,
                                                     list(ALGORITHMS).index(r["alg"]) if r["alg"] in ALGORITHMS else 99))
    write_json(p, allrows)
    (out_dir / "audit.md").write_text(format_audit(allrows) + "\n")
    return rows


def audit_workload(alg: str) -> Workload:
    if alg == "es":
        return Workload(tokens=8 * 4096, seq=4096, population=8)
    if alg == "local-sgd":
        return Workload(tokens=2 * 4096, seq=4096, local_steps=2)
    if alg in ("lora", "hidden-adapter"):
        return Workload(tokens=4096, seq=4096, rank=16)
    return Workload(tokens=4096, seq=4096)


AUDIT_COLS = [("alg", "alg"), ("model", "model"), ("tokens", "Q"), ("ir_macs_per_token", "IR MACs/tok"),
              ("analytic_ir_model", "IR-faithful analytic"), ("ratio_vs_model", "IR/faithful"),
              ("analytic_naive", "textbook analytic"), ("ratio_vs_naive", "IR/textbook"),
              ("ir_over_active_params", "IR/active params"), ("error", "error")]


def format_audit(rows: list[dict]) -> str:
    f = {"ratio_vs_model": lambda x: f"{x:.4f}" if x is not None else "-",
         "ratio_vs_naive": lambda x: f"{x:.3f}" if x is not None else "-",
         "ir_over_active_params": lambda x: f"{x:.2f}" if x is not None else "-",
         "error": lambda x: x or ""}
    return md_table(rows, AUDIT_COLS, f)


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", action="store_true", help="MAC audit (results/audit.{json,md} + breakdown tables)")
    ap.add_argument("--table", action="store_true", help="Goal-1 characterization table (results/characterization.{json,md})")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--algs", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="smallest workload per family only")
    ap.add_argument("--time-limit", type=float, default=300.0, help="wall-clock budget per build+extract (s)")
    ap.add_argument("--max-rss-mb", type=float, default=3072.0, help="RSS ceiling for each guarded child")
    ap.add_argument("--in-process", action="store_true", help="compute rows in this process (no child / RSS watchdog)")
    ap.add_argument("--unsafe", action="store_true", help="also run workloads outside guard.DRIVER_LIMITS")
    ap.add_argument("--resume", action="store_true", help="keep rows already on file (successful or guard-killed)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--one", help="internal: characterize one (alg, model, workload) JSON spec and print the row")
    args = ap.parse_args(argv)
    if args.one:
        spec = json.loads(args.one)
        row = characterize(spec["alg"], spec["model"], _wl_from_json(spec["wl"]), time_limit_s=spec.get("time_limit_s"))
        print(dumps(row))
        return 0
    if not (args.audit or args.table):
        ap.error("choose --audit and/or --table")
    if args.audit:
        audit(models=args.models or ("llama3-8b", "mixtral-8x7b"), algs=args.algs, out_dir=args.out)
    if args.table:
        characterize_all(models=args.models or REAL_MODELS, algs=args.algs, out_dir=args.out, quick=args.quick,
                         time_limit_s=args.time_limit, unsafe=args.unsafe, guarded=not args.in_process,
                         max_rss_mb=args.max_rss_mb, resume=args.resume, log=lambda *a: print(*a, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
