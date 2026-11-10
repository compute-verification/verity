"""Committed bytes by buffer category from a served_commit run's out/layouts.jsonl (prof_arm.RootsArm): every compiled-mode buffer
of every step put in a category and a decoder layer, then per token and layer (prefill and decode), per run, and under the
fused-epilogue proof blocks a proof could take instead, each with the hashing time it would leave at the rows-sha512 floor.  Reads
JSON only; measurement only.

    python layout_bytes.py RUN/out/layouts.jsonl [--json OUT.json]   (--rate-256, --rate-4k: the floor in GB/s at 256-byte and
                                                                       4 KiB rows; --pad-ms-per-m: ms per million pad leaves)

Categories, from the compiled source's names (`c<piece>.<op>/<buf>`, served_compiled_source.py) and shapes: piece p >= 1 holds
layer p-1's o_proj, post-attention norm, gate/up, activation, down and post-MLP residual, then layer p's input norm, qkv, the
rotated q and k (k is what the KV cache stores, with v inside qkv) and, between pieces, layer p's attention output.  A norm
output is any [T, hidden] (or per-head [T, heads, head_dim]) buffer that is not a GEMM's, a residual or a rotation's.  An MoE
layer's block is one extern call (vLLM's fused-MoE custom op, `extern` or `moe_*`): its [T, hidden] output is `moe`, in down's
place, and its router and experts are inside it.  Where the norm after o_proj comes as a pair of [T, hidden] buffers (a fused
add + RMSNorm), the first is the post-attention residual, `residual_attn`.

Proof blocks (what each would take as its inputs and outputs, so what is left to commit):
  norm-folded      RMSNorm inside the next GEMM's proof block, which recomputes it from the committed residual stream: no norm output.
  residual-folded  the residual add inside o_proj's and down's proof blocks, whose outputs are the sums: down's (or the MoE block's)
                   output is not committed beside the post-MLP residual, nor o_proj's beside a post-attention residual (where
                   none is committed, o_proj's output becomes it, same bytes).
  act-folded       silu(gate) * up inside the gate/up proof block: the activation is committed, gate/up's output is not.
  all-three        the three together.
  per-layer        one proof block a decoder layer: the residual stream between layers and the KV values (k and v) only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict

CATS = ("inputs", "embedding", "norm", "qkv", "rope_q", "kv_k", "attention", "o_proj", "residual_attn", "residual", "gate_up",
        "activation", "down", "moe", "logits", "sampler", "other")
PIECE = re.compile(r"^c(\d+)\.([^/]+)/(.+)$")
LAYER = re.compile(r"layers\.(\d+)\.")


def classify(step: list) -> list[tuple[int | None, str, int]]:
    """(layer, category, bytes) for each buffer of one step's layout ([name, dtype, shape, bytes] each)."""
    hidden = next(m[2][-1] for m in step if m[0].startswith("c00.embedding/"))
    out: list[tuple[int | None, str, int]] = []
    by_piece: dict[int, list] = defaultdict(list)
    for m in step:
        name, _dt, shape, nb = m
        pm = PIECE.match(name)
        if name.startswith("logits") or "logits" in name.split("/")[0]:
            out.append((None, "logits", nb))
        elif "sampler" in name:
            out.append((None, "sampler", nb))
        elif pm and pm.group(2) == "input":
            out.append((None, "inputs", nb))
        elif pm and pm.group(2) == "attention":
            lm = LAYER.search(pm.group(3))
            out.append((int(lm.group(1)) if lm else int(pm.group(1)), "attention", nb))
        elif pm:
            by_piece[int(pm.group(1))].append((pm.group(2), shape, nb))
        else:
            out.append((None, "other", nb))
    last = max(by_piece, default=0)
    heads = max((s[-2] for bufs in by_piece.values() for _op, s, _ in bufs if len(s) == 3), default=0)
    for p, bufs in sorted(by_piece.items()):
        layer = p - 1                     # the MLP half of piece p is layer p-1's; its attention half (after down) is layer p's
        phase = "qkv_in" if p == 0 else "attn_out"
        post_mlp = 0
        n_post_mlp = 0
        for i, (op, shape, nb) in enumerate(bufs):
            is_mm = op in ("mm", "addmm", "bmm")
            flat = len(shape) == 2 and shape[-1] == hidden and not is_mm
            if _moe_block(op, shape, hidden):
                out.append((layer, "moe", nb))
                phase = "post_mlp"
                n_post_mlp = 0
            elif p == 0 and op == "embedding" and phase == "qkv_in" and not any(c == "embedding" for _l, c, _b in out):
                out.append((None, "embedding", nb))
            elif phase == "qkv_in":
                out.append((layer + 1, "qkv" if is_mm else "norm" if flat else "other", nb))
                phase = "qkv" if is_mm else phase
            elif phase == "qkv":
                if len(shape) == 3 and op == "cat":
                    out.append((layer + 1, "rope_q" if shape[-2] == heads else "kv_k", nb))
                else:
                    out.append((layer + 1, "norm" if len(shape) == 3 else "other", nb))
            elif phase == "attn_out":
                out.append((layer, "o_proj" if is_mm else "other", nb))
                phase = "post_attn" if is_mm else phase
            elif phase == "post_attn":
                if is_mm and shape[-1] < hidden // 4:
                    out.append((layer, "moe", nb))                           # the router's logits
                elif is_mm:
                    out.append((layer, "gate_up", nb))
                    phase = "mlp"
                elif flat and i + 1 < len(bufs) and _flat(bufs[i + 1], hidden):
                    out.append((layer, "residual_attn", nb))
                else:
                    out.append((layer, "norm" if flat else "other", nb))
            elif phase == "mlp":
                if is_mm and shape[-1] == hidden:
                    out.append((layer, "down", nb))
                    phase = "post_mlp"
                else:
                    out.append((layer, "activation" if not is_mm and not flat else "other", nb))
            elif phase == "post_mlp":
                if is_mm:
                    out.append((layer + 1, "qkv", nb))
                    phase = "qkv"
                elif flat:
                    post_mlp += 1
                    n_post_mlp = n_post_mlp or _flat_after_mlp(bufs, hidden)
                    if p == last:
                        out.append((layer if post_mlp < n_post_mlp else None, "residual" if post_mlp < n_post_mlp else "norm", nb))
                    else:
                        out.append((layer if post_mlp == 1 else layer + 1, "residual" if post_mlp == 1 else "norm", nb))
                else:
                    out.append((layer, "other", nb))
    return out


def _moe_block(op: str, shape: list, hidden: int) -> bool:
    return "moe" in op or (op == "extern" and len(shape) == 2 and shape[-1] == hidden)


def _flat(buf: tuple, hidden: int) -> bool:
    """A [T, hidden] buffer that is no GEMM's and no MoE block's: a norm or a residual."""
    op, shape, _ = buf
    return len(shape) == 2 and shape[-1] == hidden and op not in ("mm", "addmm", "bmm") and not _moe_block(op, shape, hidden)


def _flat_after_mlp(bufs: list, hidden: int) -> int:
    """The [T, hidden] non-GEMM buffers after a piece's down (or MoE) output: the residual, then the next norm."""
    for i in range(len(bufs) - 1, -1, -1):
        op, shape, _ = bufs[i]
        if _moe_block(op, shape, hidden) or (op in ("mm", "addmm", "bmm") and shape[-1] == hidden and i > 0):
            return sum(1 for b in bufs[i + 1:] if _flat(b, hidden))
    return 0


def tokens(step: list) -> int:
    return next(m[2][0] for m in step if m[0] == "c00.input/ids")


def unit_bytes(cls: list[tuple[int | None, str, int]], unit: str, t: int, kv_tok_layer: int, hidden_bytes: int,
               layers: int) -> int:
    drop: set[str] = set()
    if unit in ("norm-folded", "all-three"):
        drop.add("norm")
    if unit in ("act-folded", "all-three"):
        drop.add("gate_up")
    total = sum(b for _l, c, b in cls if c not in drop)
    if unit in ("residual-folded", "all-three"):
        res_layers = {lay for lay, c, _b in cls if c == "residual"}
        attn_res_layers = {lay for lay, c, _b in cls if c == "residual_attn"}
        total -= sum(b for lay, c, b in cls if c in ("down", "moe") and lay in res_layers)
        total -= sum(b for lay, c, b in cls if c == "o_proj" and lay in attn_res_layers)
    if unit == "per-layer":
        keep = sum(b for _l, c, b in cls if c in ("inputs", "embedding", "logits", "sampler"))
        total = keep + layers * t * (hidden_bytes + 2 * kv_tok_layer)
    return total


def hash_ms(nbytes: int, row: int, rate_gbs: float, pad_ms_per_m: float) -> float:
    n = max(1, -(-nbytes // row))
    pads = (1 << math.ceil(math.log2(n))) - n
    return nbytes / rate_gbs / 1e6 + pads * pad_ms_per_m / 1e6


UNITS = ("today", "norm-folded", "residual-folded", "act-folded", "all-three", "per-layer")


def summarize(rec: dict, rates: dict[int, float], pad_ms_per_m: float) -> dict:
    steps = {int(k): v for k, v in rec["steps"].items()}
    out: dict = {"arm": rec["arm"], "run": rec["run"], "sig": rec["sig"], "steps": len(steps)}
    per_token: dict[str, dict] = {}
    run_bytes = defaultdict(int)
    run_ms = defaultdict(float)
    phase_bytes = defaultdict(lambda: defaultdict(int))
    for s, step in sorted(steps.items()):
        cls = classify(step)
        t = tokens(step)
        layers = 1 + max(lay for lay, _c, _b in cls if lay is not None)
        hidden_bytes = next(b for lay, c, b in cls if c == "embedding") // t
        kv = sum(b for lay, c, b in cls if c == "kv_k") // max(1, layers) // t
        phase = "decode" if t <= 64 else "prefill"
        for unit in UNITS:
            nb = unit_bytes(cls, unit, t, kv, hidden_bytes, layers) if unit != "today" else sum(b for _l, _c, b in cls)
            run_bytes[unit] += nb
            phase_bytes[phase][unit] += nb
            for row, rate in rates.items():
                run_ms[(unit, row)] += hash_ms(nb, row, rate, pad_ms_per_m)
        if phase not in per_token or t > per_token[phase]["tokens"]:
            cat_layer = defaultdict(lambda: defaultdict(int))
            for lay, c, b in cls:
                cat_layer[c]["outside" if lay is None else lay] += b
            in_layers = {c: [v.get(lay, 0) / t for lay in range(layers)] for c, v in cat_layer.items()}
            per_token[phase] = {
                "step": s, "tokens": t, "layers": layers, "bytes_per_token": sum(b for _l, _c, b in cls) / t,
                "per_layer_per_token": {c: round(sum(v) / layers, 1) for c, v in in_layers.items() if sum(v)},
                "layer_spread": {c: [round(min(v), 1), round(max(v), 1)] for c, v in in_layers.items()
                                 if sum(v) and min(v) != max(v)},
                "outside_layers_per_token": {c: round(v.get("outside", 0) / t, 1) for c, v in cat_layer.items()
                                             if v.get("outside")},
            }
    out["per_token"] = per_token
    out["run_bytes"] = dict(run_bytes)
    out["phase_bytes"] = {p: dict(v) for p, v in phase_bytes.items()}
    out["run_hash_ms"] = {f"{u}@{row}": round(v, 2) for (u, row), v in run_ms.items()}
    return out


def render(s: dict) -> str:
    lines = [f"## {s['arm']} run {s['run']} ({s['steps']} steps, layout {s['sig']})", ""]
    for phase in ("prefill", "decode"):
        p = s["per_token"].get(phase)
        if not p:
            continue
        tot_layer = sum(p["per_layer_per_token"].values())
        outside = p["bytes_per_token"] - p["layers"] * tot_layer
        head = (f"{phase}: step {p['step']}, {p['tokens']} tokens, {p['bytes_per_token']:.0f} B a token ({p['layers']} layers x "
                f"{tot_layer:.0f} B + {outside:.0f} B outside them)")
        lines += [head, "", "| category | B / token / layer | share of a layer |", "|---|---:|---:|"]
        for c in CATS:
            v = p["per_layer_per_token"].get(c)
            if v:
                lines.append(f"| {c} | {v:.0f} | {100 * v / tot_layer:.1f}% |")
        outs = ", ".join(f"{c} {v:.0f}" for c, v in p["outside_layers_per_token"].items())
        lines.append(f"| outside layers (per token) | {outs} | |")
        if p["layer_spread"]:
            lines.append(f"| layers differ | {p['layer_spread']} | |")
        lines.append("")
    rows = sorted({int(k.split("@")[1]) for k in s["run_hash_ms"]})
    lines += ["| unit | run GB | vs today | prefill GB | decode GB | " + " | ".join(f"hash ms @ {r} B rows" for r in rows) + " |",
              "|---|---:|---:|---:|---:|" + "---:|" * len(rows)]
    base = s["run_bytes"]["today"]
    for u in UNITS:
        b = s["run_bytes"][u]
        pb = s["phase_bytes"].get("prefill", {}).get(u, 0)
        db = s["phase_bytes"].get("decode", {}).get(u, 0)
        lines.append(f"| {u} | {b / 1e9:.3f} | {b / base:.3f} | {pb / 1e9:.3f} | {db / 1e9:.3f} | "
                     + " | ".join(f"{s['run_hash_ms'][f'{u}@{r}']:.1f}" for r in rows) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layouts", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--rate-256", type=float, default=66.12, help="GB/s, rows_floor minb2 at 256-byte rows")
    ap.add_argument("--rate-4k", type=float, default=230.61, help="GB/s, rows_floor minb2 at 4 KiB rows")
    ap.add_argument("--pad-ms-per-m", type=float, default=0.392, help="ms per million pad leaves")
    a = ap.parse_args()
    seen, res = set(), []
    for path in a.layouts:
        with open(path) as f:
            recs = [json.loads(line) for line in f]
        for rec in recs:
            if rec["sig"] in seen:
                continue
            seen.add(rec["sig"])
            s = summarize(rec, {256: a.rate_256, 4096: a.rate_4k}, a.pad_ms_per_m)
            res.append(s)
            print(render(s))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
