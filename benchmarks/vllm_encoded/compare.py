"""The run's gates and points from its arms (`serve.py`'s arm-*.json in OUT): exits non-zero unless every arm installed and
every encoded arm generated the gate prompts of the plain arm in its compilation mode bit for bit (token ids and log-probs).

Writes OUT/points.json: one point per encoded arm, batch and context, with its decode tokens/s, the stock and plain arms'
in the same compilation mode (the median over their repeats), y (encoded decode-step time over native: against stock vLLM
FP8, against the fused kernel's own plain FP8 fragments, and against stock vLLM compiled as it serves by default) and the
step's working set by kind.  The working set is an accounting from the model's
shapes, the context and the arm's store, not a counter reading: per decode step, the bytes each kind of tensor moves between
HBM and the chip if nothing is reused from L2 (the encoded weights; the KV cache read and written; the activations; the
fused kernel's chunk partial sums, written and read once; the BF16 LM head, which no arm encodes).

Writes OUT/frontier.json too: the same points in the rate audit's shared schema (`verity/codec-frontier-point/v0`, ra-codec-wide's
`benchmarks/word_codec/frontier.py`), with ``end_to_end: true`` and the context: y over stock vLLM compiled (the decode
headline's baseline), split by the run's own arms into base (the arm's kernel on plain payloads: `plain`, or the wide
kernel at k = 0), decode (over base) and tag (the tagged arm over the same arm untagged); the step runs no commit, so
y_commit is 0.  x, Δ (and τ for a tagged arm) are the codec lanes' measurements for the same field, k and packing (`CODEC`).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

TASK_COLS = 512
FRONTIER = "verity/codec-frontier-point/v0"
WORD_CODEC_POINTS = "art:615ddafdbdca40c9a79359dcd942422100b8b2d3e8d82b1e36073acdaedcb1de"  # ra-word-codec, r20261012-072836-5a84
WIDE_POINTS = "art:96b9bcd36ce70e8e5f06a2da0faddc9a6aff4242ccc7c16f6c4faa0080fd9d53"  # ra-codec-wide, r20261012-074744-2f82
#: (word bits, payload bits, k) -> the encode time of one pass's words in native passes (x), the decode's share of a pass's
#: compute (Δ) and the ro-blake3/v1 tag's (τ), as the codec lanes measured them on the same GPU
CODEC = {
    (61, 56, 2): (14.498248107145026, 0.4995289804729952, 0.7324373834472417, WORD_CODEC_POINTS),
    (61, 56, 3): (22.969029769476588, 0.7053308070724508, 0.7324373834472417, WORD_CODEC_POINTS),
    (127, 126, 1): (32.03854113091271, 0.4621287561288848, 0.8233024528242008, WIDE_POINTS),
    (127, 126, 2): (64.07708226182542, 0.8317700493707707, 0.8233024528242008, WIDE_POINTS),
    (255, 254, 1): (109.22912467435894, 1.193949892482768, 0.8223356015334597, WIDE_POINTS),
    (255, 254, 2): (218.45824934871789, 2.208046886523409, 0.8223356015334597, WIDE_POINTS),
    (521, 520, 1): (332.236924960776, 0.8769569643747882, 0.8224952428519756, WIDE_POINTS),
    (521, 520, 2): (664.473849921552, 1.6401752171760773, 0.8224952428519756, WIDE_POINTS),
}


def load(out: Path) -> list[dict]:
    arms = []
    for p in sorted(out.glob("arm-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        d = json.loads(p.read_text())
        d["tag"] = p.stem[4:]
        arms.append(d)
    return arms


def linears(ms: dict) -> list[tuple[int, int]]:
    """(N, K) of one decoder layer's linears."""
    h, i, kvh, hd, nh = ms["hidden"], ms["intermediate"], ms["kv_heads"], ms["head_dim"], ms["heads"]
    return [((nh + 2 * kvh) * hd, h), (h, nh * hd), (2 * i, h), (h, i)]


def working_set(ms: dict, b: int, ctx: float, weight_bytes: int, partials: bool, tag_bytes: int = 0) -> dict:
    layers, h, i = ms["layers"], ms["hidden"], ms["intermediate"]
    kv_token = layers * 2 * ms["kv_heads"] * ms["head_dim"] * 2
    lin = linears(ms)
    io = layers * b * sum(2 * k + 2 * k + 2 * n for n, k in lin)
    other = layers * b * (2 * 4 * 2 * h + 2 * 2 * h + 3 * 2 * i)
    part = layers * b * sum(k // TASK_COLS * n * 4 * 2 for n, k in lin) if partials else 0
    ws = {"weights": weight_bytes, "lm_head": ms["vocab"] * h * 2, "kv_read": int(b * ctx * kv_token), "kv_write": b * kv_token,
          "activations": io + other, "partials": part, "tags": tag_bytes}
    total = sum(ws.values())
    unc = total - ws["weights"]
    ws.update(total=total, uncertified=unc, uncertified_share=unc / total,
              uncertified_share_without_lm_head=(unc - ws["lm_head"]) / (total - ws["lm_head"]))
    return ws


def mode(a: dict) -> str:
    return "-c" if a.get("compiled") else ""


def group_name(a: dict) -> str:
    """Arms that repeat one configuration share a name: stock, plain, enc2-56 (k, payload bits), w521-1t (wide field bits,
    k, t when tagged), with -c when compiled."""
    if a["arm"] == "enc":
        return f"enc{a['k']}-{a['install'].get('bits')}" + mode(a)
    if a["arm"] == "wide":
        return f"w{a['install']['codec']['word_bits']}-{a['k']}{'t' if a.get('tag') else ''}" + mode(a)
    return a["arm"] + mode(a)


def encoded(a: dict) -> bool:
    return a["arm"] == "enc" or (a["arm"] == "wide" and a["k"] > 0)


def main() -> int:
    out = Path(sys.argv[1])
    arms = load(out)
    rc = out / "arms.rc"
    fails = [f"arm {line} (exit status)" for line in (rc.read_text().splitlines() if rc.exists() else ["none ran"])
             if not line.endswith(" 0")]
    for a in arms:
        if a["arm"] in ("plain", "enc", "wide") and not a.get("install", {}).get("gates"):
            fails.append(f"{a['tag']}: no install record")
    by = {}
    for a in arms:
        by.setdefault(group_name(a), []).append(a)
    gate = {}
    for a in arms:
        ref = by.get("plain" + mode(a), [])
        if not ref:
            if a["arm"] in ("enc", "wide"):
                fails.append(f"{a['tag']}: no plain arm in its compilation mode to compare its gate with")
            continue
        ref = ref[0]["gate"]
        if not a["gate"]:
            continue
        same = sum(x["token_ids"] == y["token_ids"] and x["logprobs"] == y["logprobs"] for x, y in zip(a["gate"], ref))
        tok = sum(sum(p == q for p, q in zip(x["token_ids"], y["token_ids"])) for x, y in zip(a["gate"], ref))
        total = sum(len(y["token_ids"]) for y in ref)
        gate[a["tag"]] = {"against": by["plain" + mode(a)][0]["tag"], "prompts_bit_identical": f"{same}/{len(ref)}",
                          "tokens_equal": f"{tok}/{total}"}
        if a["arm"] in ("enc", "wide") and same != len(ref):
            fails.append(f"{a['tag']}: gate generation differs from {gate[a['tag']]['against']}")
    print("gate", json.dumps(gate, indent=1))

    def tps(group: list[dict], b: int, ctx: int) -> float | None:
        v = [p["tokens_per_s"] for a in group for p in a["points"] if p["batch"] == b and p["context"] == ctx and "tokens_per_s" in p]
        return statistics.median(v) if v else None

    def ratio(x: float | None, e: float) -> float | None:
        return x / e if x else None

    points, rows, frontier = [], [], []
    run_id = os.environ.get("RESEARCH_RUN_ID")
    keys = sorted({(p["batch"], p["context"]) for a in arms for p in a["points"]})
    for name, group in by.items():
        a = group[0]
        if a["arm"] not in ("enc", "wide"):
            continue
        ms, inst, md = a["model_shape"], a["install"], mode(a)
        cod = inst.get("codec") or {"field": "p61", "word_bits": 61, "payload_bits": inst["bits"], "tag": None}
        wide = a["arm"] == "wide"
        stock, plain, compiled = by.get("stock" + md, []), by.get("plain" + md, []), by.get("stock-c", [])
        base = by.get(f"w{cod['word_bits']}-0" + md, []) if wide else plain
        untagged = by.get(f"w{cod['word_bits']}-{a['k']}" + md, []) if cod.get("tag") else []
        for b, ctx in keys:
            e, s, pl, sc = tps(group, b, ctx), tps(stock, b, ctx), tps(plain, b, ctx), tps(compiled, b, ctx)
            if e is None:
                continue
            pt = next(p for p in a["points"] if p["batch"] == b and p["context"] == ctx)
            mean_ctx = ctx + pt.get("max_tokens", 0) / 2
            ws = working_set(ms, b, mean_ctx, inst["store_bytes"], True, inst["plain_bytes"] // 8 if cod.get("tag") else 0)
            ws_stock = working_set(ms, b, mean_ctx, inst["plain_bytes"], False)
            point = {
                "lane": "ra-vllm-encoded", "end_to_end": True, "workload": "llama-3.1-8b-instruct decode",
                "engine": f"vLLM {a['vllm']}, FP8 online, full decode CUDA graphs, "
                          + ("torch.compile (vLLM's default)" if a.get("compiled") else "mode 0 (no torch.compile)"),
                "gpu": a["gpu"],
                "codec": {"field": cod["field"], "word_bits": cod["word_bits"], "rounds": a["k"],
                          "payload_bits": cod["payload_bits"], "tag": cod.get("tag")},
                "kernel": ("benchmarks/word_codec/wide.cu ww_gemm (ra-codec-wide's fused decode + tag GEMM)" if wide else
                           "benchmarks/vllm_encoded/ve.cu (word_codec wc.cu fused GEMM), form chosen per shape and batch"),
                "batch": b, "context": ctx, "tokens_per_s": e, "step_ms": b / e * 1e3,
                "native_stock_tokens_per_s": s, "native_plain_tokens_per_s": pl, "native_stock_compiled_tokens_per_s": sc,
                "y_vs_stock": ratio(s, e), "y_vs_plain": ratio(pl, e), "y_vs_stock_compiled": ratio(sc, e),
                "working_set_bytes": ws, "working_set_bytes_stock": ws_stock, "run": run_id,
            }
            points.append(point)
            rows.append((name, b, ctx, e, s, pl, point["y_vs_stock"], point["y_vs_plain"], point["y_vs_stock_compiled"],
                         ws["uncertified_share"]))
            ref = CODEC.get((cod["word_bits"], cod["payload_bits"], a["k"]))
            bs, un = tps(base, b, ctx), tps(untagged, b, ctx)
            if sc is None or bs is None or ref is None or not encoded(a):
                continue
            y_total, y_base = sc / e, sc / bs
            y_tag = sc / e - sc / un if un else 0.0
            frontier.append({
                "schema": FRONTIER, "lane": "ra-vllm-encoded", "end_to_end": True, "b": cod["word_bits"], "k": a["k"],
                "arithmetic": ("cuda-int32/" + cod["field"] + "/wide.cu" if wide else "cuda-int32/mersenne61/pb56/ve.cu")
                              + ("/tag" if cod.get("tag") else ""),
                "compiled": bool(a.get("compiled")), "batch": b, "context": ctx,
                "y_total": y_total, "y_base": y_base, "y_decode": y_total - y_base - y_tag,
                "y_tag": y_tag, "y_commit": 0.0, "x": ref[0], "delta": ref[1], "tau": ref[2] if cod.get("tag") else 0.0,
                "x_source": ref[3], "measured": True, "packing": cod["payload_bits"],
                "label": f"vLLM end to end, {name}, ctx {ctx}: whole decode step over stock vLLM FP8 compiled; base = "
                         + ("wide k = 0" if wide else "plain fragments")
                         + (", tag split by the untagged arm" if un else ", no tag split" if cod.get("tag") else "")
                         + "; no commit in the step",
                "gpu": a["gpu"], "run": run_id, "art": None,
            })
    f = lambda x, fmt: "-" if x is None else format(x, fmt)
    print(f"{'arm':10} {'B':>3} {'ctx':>5} {'enc tok/s':>10} {'stock':>8} {'plain':>8} {'y/stock':>8} {'y/plain':>8} "
          f"{'y/stk-c':>8} {'uncert':>7}")
    for r in rows:
        print(f"{r[0]:10} {r[1]:3d} {r[2]:5d} {f(r[3], '10.1f')} {f(r[4], '8.1f')} {f(r[5], '8.1f')} {f(r[6], '8.3f')} "
              f"{f(r[7], '8.3f')} {f(r[8], '8.3f')} {f(r[9], '7.1%')}")
    for name, group in by.items():
        if group[0]["arm"] not in ("enc", "wide"):
            for b, ctx in keys:
                v = tps(group, b, ctx)
                if v:
                    print(f"{name:10} {b:3d} {ctx:5d} {v:10.1f}", *(f"{p['tokens_per_s']:.1f}" for a in group for p in a["points"]
                                                                  if p["batch"] == b and p["context"] == ctx and "tokens_per_s" in p))
    summary = {"schema": "verity/ra-points/v0-provisional", "gate": gate, "fails": fails, "points": points,
               "arms": [{"tag": a["tag"], "arm": a["arm"], "compiled": bool(a.get("compiled")), "k": a["k"],
                         "install": {k: v for k, v in a.get("install", {}).items()
                                                                                     if k != "selection"},
                         "kv_tokens": a.get("kv_tokens"), "attention": a.get("attention"), "startup_s": a.get("startup_s")}
                        for a in arms]}
    (out / "points.json").write_text(json.dumps(summary, indent=1))
    (out / "frontier.json").write_text(json.dumps(frontier, indent=1))
    for p in frontier:
        if p["batch"] in (1, 32) and p["context"] == 512:
            print(f"frontier b={p['b']} k={p['k']} M={p['batch']} ctx={p['context']} {p['arithmetic']}: y_total "
                  f"{p['y_total']:.3f} = base {p['y_base']:.3f} + decode {p['y_decode']:.3f} + tag {p['y_tag']:.3f}, x {p['x']:.1f}")
    for x in fails:
        print("FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
