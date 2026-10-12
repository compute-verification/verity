"""The run's gates and points from its arms (`serve.py`'s arm-*.json in OUT): exits non-zero unless every arm installed and
every encoded arm generated the gate prompts of the plain arm in its compilation mode bit for bit (token ids and log-probs).

Writes OUT/points.json: one point per encoded arm, batch and context, with its decode tokens/s, the stock and plain arms'
in the same compilation mode (the median over their repeats), y (encoded decode-step time over native: against stock vLLM
FP8, against the fused kernel's own plain FP8 fragments, and against stock vLLM compiled as it serves by default) and the
step's working set by kind.  The working set is an accounting from the model's
shapes, the context and the arm's store, not a counter reading: per decode step, the bytes each kind of tensor moves between
HBM and the chip if nothing is reused from L2 (the encoded weights; the KV cache read and written; the activations; the
fused kernel's chunk partial sums, written and read once; the BF16 LM head, which no arm encodes).

`points.json` is in a provisional schema (`verity/ra-points/v0-provisional`) until ra-codec-wide publishes the shared one;
each point carries ``end_to_end: true``.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

TASK_COLS = 512


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


def working_set(ms: dict, b: int, ctx: float, weight_bytes: int, partials: bool) -> dict:
    layers, h, i = ms["layers"], ms["hidden"], ms["intermediate"]
    kv_token = layers * 2 * ms["kv_heads"] * ms["head_dim"] * 2
    lin = linears(ms)
    io = layers * b * sum(2 * k + 2 * k + 2 * n for n, k in lin)
    other = layers * b * (2 * 4 * 2 * h + 2 * 2 * h + 3 * 2 * i)
    part = layers * b * sum(k // TASK_COLS * n * 4 * 2 for n, k in lin) if partials else 0
    ws = {"weights": weight_bytes, "lm_head": ms["vocab"] * h * 2, "kv_read": int(b * ctx * kv_token), "kv_write": b * kv_token,
          "activations": io + other, "partials": part}
    total = sum(ws.values())
    unc = total - ws["weights"]
    ws.update(total=total, uncertified=unc, uncertified_share=unc / total,
              uncertified_share_without_lm_head=(unc - ws["lm_head"]) / (total - ws["lm_head"]))
    return ws


def mode(a: dict) -> str:
    return "-c" if a.get("compiled") else ""


def group_name(a: dict) -> str:
    """Arms that repeat one configuration share a name: stock, plain, enc2-56 (k, payload bits), with -c when compiled."""
    return (a["arm"] if a["arm"] != "enc" else f"enc{a['k']}-{a['install'].get('bits')}") + mode(a)


def main() -> int:
    out = Path(sys.argv[1])
    arms = load(out)
    rc = out / "arms.rc"
    fails = [f"arm {line} (exit status)" for line in (rc.read_text().splitlines() if rc.exists() else ["none ran"])
             if not line.endswith(" 0")]
    for a in arms:
        if a["arm"] in ("plain", "enc") and not a.get("install", {}).get("gates"):
            fails.append(f"{a['tag']}: no install record")
    by = {}
    for a in arms:
        by.setdefault(group_name(a), []).append(a)
    gate = {}
    for a in arms:
        ref = by.get("plain" + mode(a), [])
        if not ref:
            if a["arm"] == "enc":
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
        if a["arm"] == "enc" and same != len(ref):
            fails.append(f"{a['tag']}: gate generation differs from {gate[a['tag']]['against']}")
    print("gate", json.dumps(gate, indent=1))

    def tps(group: list[dict], b: int, ctx: int) -> float | None:
        v = [p["tokens_per_s"] for a in group for p in a["points"] if p["batch"] == b and p["context"] == ctx and "tokens_per_s" in p]
        return statistics.median(v) if v else None

    def ratio(x: float | None, e: float) -> float | None:
        return x / e if x else None

    points, rows = [], []
    keys = sorted({(p["batch"], p["context"]) for a in arms for p in a["points"]})
    for name, group in by.items():
        if not name.startswith("enc"):
            continue
        a = group[0]
        ms, inst, md = a["model_shape"], a["install"], mode(a)
        stock, plain, compiled = by.get("stock" + md, []), by.get("plain" + md, []), by.get("stock-c", [])
        for b, ctx in keys:
            e, s, pl, sc = tps(group, b, ctx), tps(stock, b, ctx), tps(plain, b, ctx), tps(compiled, b, ctx)
            if e is None:
                continue
            pt = next(p for p in a["points"] if p["batch"] == b and p["context"] == ctx)
            mean_ctx = ctx + pt.get("max_tokens", 0) / 2
            ws = working_set(ms, b, mean_ctx, inst["store_bytes"], True)
            ws_stock = working_set(ms, b, mean_ctx, inst["plain_bytes"], False)
            point = {
                "lane": "ra-vllm-encoded", "end_to_end": True, "workload": "llama-3.1-8b-instruct decode",
                "engine": f"vLLM {a['vllm']}, FP8 online, full decode CUDA graphs, "
                          + ("torch.compile (vLLM's default)" if a.get("compiled") else "mode 0 (no torch.compile)"),
                "gpu": a["gpu"],
                "codec": {"p": "2^61-1", "word_bits": 61, "rounds": a["k"], "payload_bits": inst["bits"], "tag": None},
                "kernel": "benchmarks/vllm_encoded/ve.cu (word_codec wc.cu fused GEMM), form chosen per shape and batch",
                "batch": b, "context": ctx, "tokens_per_s": e, "step_ms": b / e * 1e3,
                "native_stock_tokens_per_s": s, "native_plain_tokens_per_s": pl, "native_stock_compiled_tokens_per_s": sc,
                "y_vs_stock": ratio(s, e), "y_vs_plain": ratio(pl, e), "y_vs_stock_compiled": ratio(sc, e),
                "working_set_bytes": ws, "working_set_bytes_stock": ws_stock,
                "run": os.environ.get("RESEARCH_RUN_ID"),
            }
            points.append(point)
            rows.append((name, b, ctx, e, s, pl, point["y_vs_stock"], point["y_vs_plain"], point["y_vs_stock_compiled"],
                         ws["uncertified_share"]))
    f = lambda x, fmt: "-" if x is None else format(x, fmt)
    print(f"{'arm':10} {'B':>3} {'ctx':>5} {'enc tok/s':>10} {'stock':>8} {'plain':>8} {'y/stock':>8} {'y/plain':>8} "
          f"{'y/stk-c':>8} {'uncert':>7}")
    for r in rows:
        print(f"{r[0]:10} {r[1]:3d} {r[2]:5d} {f(r[3], '10.1f')} {f(r[4], '8.1f')} {f(r[5], '8.1f')} {f(r[6], '8.3f')} "
              f"{f(r[7], '8.3f')} {f(r[8], '8.3f')} {f(r[9], '7.1%')}")
    for name, group in by.items():
        if not name.startswith("enc"):
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
    for x in fails:
        print("FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
