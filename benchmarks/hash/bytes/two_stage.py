"""PRF-70 (3): the numbers for two-stage sampling (decision 43) on Qwen3-8B B32, per GPU, under proofs' condition (c): every
value another step or protocol reads (K and V across steps, the KV tree's post-RoPE K and V) stays a serve-time boundary
value. A replay unit (RU) is one decoder layer of one step, or one block (attention or MLP) of one step.

    python3 benchmarks/hash/bytes/two_stage.py            # the numbers section, markdown
    python3 benchmarks/hash/bytes/two_stage.py --json

Serve-time bytes and SHA-512 overhead come from `bytes_model.price` (the hash lane's commit model). The replay of a drawn
RU is its recompute, the layer's share of the step's stock serve time, plus committing its interior at hm96's SHA-512 rate for
that window (`commit_model.hm96_rate`, salts read ahead). All of it is model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bytes_model as bm  # noqa: E402

cm = bm.cm
WEEK_S = 604_800
#: Draws a GPU-week at B = 0.1 / 1 / 5: the int8 Llama-3.1-8B plan (art:416bfc6f…, note:proofs/20261018T0450Z-report-int8-sampling-plan)
#: and this lane's Qwen3-8B bf16 plan (attention_prologue.py, baseline, two 4 KiB leaves a key; art in the note).
DRAWS = {"int8 Llama plan": {0.1: 11_000, 1.0: 258_000, 5.0: 1_760_000}, "Qwen3-8B bf16 plan": {0.1: 3_700, 1.0: 10_612}}
#: Qwen3-8B MACs a token by block at ~1K keys (estimate): attention projections 4096 x 10,240 plus QK and PV 8,192 x 1,024;
#: the MLP 3 x 4096 x 12,288.
ATTN_SHARE = (4096 * 10_240 + 8_192 * 1_024) / (4096 * 10_240 + 8_192 * 1_024 + 3 * 4096 * 12_288)
ROWS = ("two-stage-layer", "two-stage-layer+format", "two-stage-block", "two-stage-block+format")


def interior(ru: str) -> dict[str, int]:
    """Bytes a token of a drawn RU's interior: a layer's committed bytes less its boundary; a block's split by its values."""
    if ru == "layer":
        return {"layer": bm.TODAY_BYTES - bm.RU_LAYER}
    k = bm.KEPT
    attn = k["qkv"] - 2_048 + k["rope_q"] + k["attention"]                 # V stays at the boundary
    return {"attention block": attn, "MLP block": k["activation"] + k["down"]}


def replay(gpu: str) -> dict:
    w = cm.qwen(gpu)
    tokens = {p.name: (2_560 if p.name == "prefill" else 32) for p in w.phases}
    total_tokens = sum(tokens[p.name] * p.steps for p in w.phases)
    out = {}
    for ru, share in (("layer", {"layer": 1.0}), ("block", {"attention block": ATTN_SHARE, "MLP block": 1 - ATTN_SHARE})):
        per = {}
        for part, nbytes_token in interior(ru).items():
            for p in w.phases:
                n = nbytes_token * tokens[p.name]
                recompute = p.serve_ms_per_step / bm.LAYERS * share[part]
                commit = n / (cm.hm96_rate(gpu, "SHA-512", "pre", n) * cm.GB) * 1e3
                per[(part, p.name)] = {"bytes": n, "recompute_ms": recompute, "commit_ms": commit, "ms": recompute + commit}
        by_tokens = {p.name: tokens[p.name] * p.steps / total_tokens for p in w.phases}
        parts = list(interior(ru))
        mean = sum(by_tokens[ph] * per[(part, ph)]["ms"] for part in parts for ph in by_tokens) / len(parts)
        out[ru] = {"per": {f"{a}, {b}": v for (a, b), v in per.items()}, "mean_ms_a_draw": mean,
                   "week_share": {src: {b: d * mean * 1e-3 / WEEK_S for b, d in ds.items()} for src, ds in DRAWS.items()}}
    return out


def serve() -> dict:
    rows = {r["key"]: r for r in bm.table()}
    return {k: {"run_gb": rows[k]["run_gb"], "bytes": rows[k]["bytes"], "run_pct": rows[k]["run_pct"]}
            for k in ("today", *ROWS)}


def render(s: dict, rp: dict) -> str:
    g = bm.GPUS
    lines = ["| Commit at serve time | bytes a token a layer | GB a run | SHA-512, % of serve: " + " / ".join(g) + " |",
             "|---|---|---|---|"]
    per_tl = {"today": bm.TODAY_BYTES, "two-stage-layer": bm.RU_LAYER, "two-stage-layer+format": bm.RU_LAYER,
              "two-stage-block": bm.RU_BLOCK, "two-stage-block+format": bm.RU_BLOCK}
    for k, r in s.items():
        lines.append(f"| {k} | {per_tl[k]:,} | {r['run_gb']:.2f} | " + " / ".join(f"{r['run_pct'][x]:.2f}" for x in g) + " |")
    lines += ["", "| drawn RU (one step) | GPU | interior MB, prefill / decode | recompute ms, prefill / decode | commit ms, "
              "prefill / decode | mean ms a draw |", "|---|---|---|---|---|---|"]
    for ru in ("layer", "block"):
        for x in g:
            r = rp[x][ru]
            for part in interior(ru):
                pre, dec = r["per"][f"{part}, prefill"], r["per"][f"{part}, decode"]
                lines.append(f"| {part} | {x} | {pre['bytes'] / 1e6:.1f} / {dec['bytes'] / 1e6:.2f} | "
                             f"{pre['recompute_ms']:.3f} / {dec['recompute_ms']:.3f} | {pre['commit_ms']:.3f} / "
                             f"{dec['commit_ms']:.3f} | {r['mean_ms_a_draw']:.3f} |")
    lines += ["", "| draws a GPU-week | RU | " + " | ".join(g) + " |", "|---|---|" + "---|" * len(g)]
    for src, ds in DRAWS.items():
        for b, d in ds.items():
            for ru in ("layer", "block"):
                lines.append(f"| {src}, B = {b}: {d:,} | {ru} | "
                             + " | ".join(f"{100 * rp[x][ru]['week_share'][src][b]:.4f}%" for x in g) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s, rp = serve(), {x: replay(x) for x in bm.GPUS}
    if a.json:
        print(json.dumps({"serve": s, "replay": rp, "draws": DRAWS, "attn_share": ATTN_SHARE}, indent=1, default=str))
        return 0
    print(render(s, rp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
