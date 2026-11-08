"""The prefix-cache hits a pool per window loses: each served trace's hits replayed against a pool that every audit window
starts empty, so a hit keeps only the blocks an earlier request of its own window computed, and the rest is recomputed.

    pool_window_hits.py --src TREE --config GLM53_CONFIG.json --out OUT.json [--window S ...] [--phases N] LABEL=DIR ...

Per trace (`replay.read_corpus_trace`, `v1.record_of`), its steps walked in order, row by row:
- a request's served hit is its first row's start, h, in whole blocks of the engine's B tokens;
- under a pool per window it keeps its leading blocks that an earlier row of a request in its window covers, with the
  same prompt and output tokens through the block, and recomputes the rest of h;
- each row then covers its request's positions through its end, as the served engine computed them (a request covers
  its recomputed blocks from its admission on).

A request's window is its record's ingress bucket over the window's buckets, `(end - phase) // W`, at the trace's own
phase (0) and at `--phases` phases spread evenly over W. Each hit block's reuse distance d is the ingress buckets back to
the latest earlier request covering it; under a uniformly random phase a window of length W loses the block with
probability min(1, d / W), which prices any W from the distances alone. The loss is priced in GLM-5.3's FLOPs (every linear layer, MLA's
attention over min(position + 1, index_topk) keys, the indexer's over position + 1, lm_head once a sampling row), as a
share of the FLOPs the trace computed served with caching on, and of the FLOPs its hits saved. A trace sees only the
reuse inside its own span, so a window longer than the span is priced on that reuse alone.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import statistics
import sys
from array import array
from pathlib import Path

STOPS = (154820, 154827, 154829)
CONFIG_SHA256 = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69"
BUCKET_S = 0.1


def load_v1(src: Path):
    for p in (src, src / "integrations" / "vllm"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if "enc_v1" not in sys.modules:
        spec = importlib.util.spec_from_file_location("enc_v1", src / "benchmarks" / "network_e2e" / "encodings" / "v1.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["enc_v1"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["enc_v1"]


def flop_model(cfg: dict) -> dict:
    H, L, nh = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    qr, kr = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    dn, dr, dv = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    attn_proj = H * qr + qr * nh * (dn + dr) + H * (kr + dr) + kr * nh * (dn + dv) + nh * dv * H
    full = sum(1 for t in cfg["indexer_types"] if t == "full")
    ih, idim = cfg["index_n_heads"], cfg["index_head_dim"]
    idx_proj = qr * ih * idim + H * idim + H * ih
    dense = sum(1 for t in cfg["mlp_layer_types"] if t == "dense")
    I, mI = cfg["intermediate_size"], cfg["moe_intermediate_size"]
    E, k, sh = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["n_shared_experts"]
    mlp = dense * 3 * H * I + (L - dense) * (H * E + (k + sh) * 3 * H * mI)
    return {"linear": 2 * (L * attn_proj + full * idx_proj + mlp), "head": 2 * H * cfg["vocab_size"],
            "core": 2 * L * nh * (dn + dr + dv), "idx": 2 * full * ih * idim, "topk": cfg["index_topk"]}


def keys(a: int, b: int, cap: int | None = None) -> int:
    """Σ min(c, cap) over c in (a, b]: the keys the tokens at positions a … b − 1 attend to."""
    if b <= a:
        return 0
    if cap is None or b <= cap:
        return (b * (b + 1) - a * (a + 1)) // 2
    if a >= cap:
        return cap * (b - a)
    return (cap * (cap + 1) - a * (a + 1)) // 2 + cap * (b - cap)


def flops(m: dict, a: int, b: int, sample: bool) -> int:
    return m["linear"] * (b - a) + m["core"] * keys(a, b, m["topk"]) + m["idx"] * keys(a, b) + (m["head"] if sample else 0)


class Chain:
    """A request's block keys, each a digest of its tokens through the block, computed as far as asked."""

    def __init__(self, tokens, block: int):
        self.tokens, self.block, self.keys = array("I", tokens), block, []

    def upto(self, n: int) -> list[bytes]:
        B = self.block
        while len(self.keys) < n:
            u = len(self.keys)
            h = hashlib.blake2b(self.keys[-1] if self.keys else b"", digest_size=16)
            h.update(self.tokens[u * B:(u + 1) * B].tobytes())
            self.keys.append(h.digest())
        return self.keys


def replay(requests, steps, block: int, model: dict, windows: list[tuple[float, float]]) -> dict:
    """The served trace's steps against a pool per window, at each (window, phase) in buckets. `requests` is each engine
    index's (tokens, prompt length, ingress bucket); `steps` each step's rows (index, start, tokens computed).

    Each block key keeps the sorted ingress buckets of the requests covering it, and a request covers every block
    through its latest row's end, so a deeper block's holders are among a shallower one's: the blocks a window keeps
    are a prefix, found by bisection."""
    chains = [Chain(t, block) for t, _, _ in requests]
    holders: dict[bytes, list[int]] = {}
    covered = [0] * len(requests)
    admitted = [False] * len(requests)
    lost_tokens, lost_flops = [0] * len(windows), [0] * len(windows)
    lost_requests = [0] * len(windows)
    saved = saved_flops = computed_flops = hit_requests = 0
    reuse: list[tuple[float, int]] = []

    def held(key: bytes, lo: float, hi: float) -> bool:
        xs = holders.get(key)
        if not xs:
            return False
        j = bisect.bisect_left(xs, lo)
        return j < len(xs) and xs[j] < hi

    for rows in steps:
        for i, start, n in rows:
            tokens, P, end = requests[i]
            if not admitted[i]:
                admitted[i] = True
                if start % block:
                    raise ValueError(f"request {i}: admitted at {start}, not whole {block}-token blocks")
                if start:
                    hit_requests += 1
                    saved += start
                    saved_flops += flops(model, 0, start, False)
                    ks = chains[i].upto(start // block)
                    for u, key in enumerate(ks):
                        xs = holders.get(key, [])
                        j = bisect.bisect_right(xs, end) - 1
                        reuse.append((end - xs[j] if j >= 0 else 0, flops(model, u * block, (u + 1) * block, False)))
                    for c, (W, phase) in enumerate(windows):
                        lo = phase + ((end - phase) // W) * W
                        a, b = 0, len(ks)
                        while a < b:
                            m = (a + b) // 2
                            if held(ks[m], lo, lo + W):
                                a = m + 1
                            else:
                                b = m
                        if a * block < start:
                            lost_requests[c] += 1
                            lost_tokens[c] += start - a * block
                            lost_flops[c] += flops(model, a * block, start, False)
            computed_flops += flops(model, start, start + n, start + n >= P)
            whole = min(start + n, len(tokens)) // block
            if whole > covered[i]:
                for k in chains[i].upto(whole)[covered[i]:whole]:
                    bisect.insort(holders.setdefault(k, []), end)
                covered[i] = whole
    return {"saved_tokens": saved, "saved_flops": saved_flops, "computed_flops": computed_flops,
            "hit_requests": hit_requests, "lost_tokens": lost_tokens, "lost_flops": lost_flops,
            "lost_requests": lost_requests, "reuse": reuse}


#: reuse-distance bins, seconds
BINS = (1, 10, 60, 300, 600, 1200, 1800, 3600)


def distances(reuse: list[tuple[float, int]], seconds: list[float]) -> dict:
    """The hit blocks' reuse distances, each the ingress buckets back to the latest earlier request covering the block,
    weighted by the FLOPs the block saves: binned, quantiles, and the share a pool per window of each length loses
    under a uniformly random phase, min(1, d / W) a block."""
    total = sum(w for _, w in reuse)
    if not total:
        return {}
    ds = sorted(reuse)
    s = [d * BUCKET_S for d, _ in ds]

    def quantile(q: float) -> float:
        acc = 0
        for x, (_, w) in zip(s, ds):
            acc += w
            if acc >= q * total:
                return x
        return s[-1]

    edges = (0, *BINS, math.inf)
    bins = {f"{a}-{b}": sum(w for x, (_, w) in zip(s, ds) if a <= x < b) / total for a, b in zip(edges, edges[1:])}
    return {"blocks": len(ds), "mean_s": sum(x * w for x, (_, w) in zip(s, ds)) / total,
            "quantiles_s": {str(q): quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, "bins": bins,
            "lost_of_saved": {str(W): sum(min(1.0, x / W) * w for x, (_, w) in zip(s, ds)) / total
                              for W in (*seconds, 7200.0, 86400.0)}}


def read_job(job):
    src, label, d, model, seconds, phases = job
    v1 = load_v1(Path(src))
    tr = v1.R.read_corpus_trace(d)
    rec = v1.record_of(tr, STOPS, bucket_s=BUCKET_S)
    by_id = {q.id: q for q in rec.requests}
    reqs = []
    for i, rid in enumerate(tr.ids):
        q = by_id[rid]
        reqs.append((q.prompt + q.output, len(q.prompt), q.end))
    steps = [list(st.rows) for st in tr.steps]
    ends = [e for _, _, e in reqs]
    span_s = (max(ends) - min(ends)) * BUCKET_S
    windows = []
    for s in seconds:
        W = s / BUCKET_S
        windows += [(W, W * j / phases) for j in range(phases)]
    got = replay(reqs, steps, tr.engine.block_size, model, windows)
    out = {"trace": label, "requests": len(reqs), "steps": len(steps), "span_s": round(span_s, 1),
           "block_size": tr.engine.block_size, "max_model_len": tr.engine.max_model_len,
           "prefix_caching": tr.engine.enable_prefix_caching, "hit_requests": got["hit_requests"],
           "saved_tokens": got["saved_tokens"], "saved_flops": got["saved_flops"],
           "computed_flops": got["computed_flops"], "saved_share": got["saved_flops"] / got["computed_flops"],
           "reuse": distances(got["reuse"], seconds), "windows": {}}
    for k, s in enumerate(seconds):
        cs = range(k * phases, (k + 1) * phases)
        share = [got["lost_flops"][c] / got["computed_flops"] for c in cs]
        of_saved = [got["lost_flops"][c] / got["saved_flops"] if got["saved_flops"] else 0.0 for c in cs]
        c0 = k * phases
        out["windows"][str(s)] = {
            "own_phase": {"lost_requests": got["lost_requests"][c0], "lost_tokens": got["lost_tokens"][c0],
                          "lost_flops": got["lost_flops"][c0], "share_of_computed": share[0],
                          "share_of_saved": of_saved[0]},
            "phases": {"n": phases, "share_of_computed": {"mean": statistics.fmean(share), "min": min(share),
                                                          "max": max(share)},
                       "share_of_saved": {"mean": statistics.fmean(of_saved), "min": min(of_saved),
                                          "max": max(of_saved)},
                       "lost_requests_mean": statistics.fmean(got["lost_requests"][c] for c in cs)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, action="append", default=[], help="window seconds (repeat)")
    ap.add_argument("--phases", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("traces", nargs="+", help="LABEL=DIR")
    a = ap.parse_args()
    raw = a.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise SystemExit(f"{a.config}: not the pinned GLM-5.3 config.json")
    model = flop_model(json.loads(raw))
    seconds = a.window or [600.0, 1200.0, 1800.0, 3600.0]
    dirs = [t.split("=", 1) for t in a.traces]
    with mp.get_context("fork").Pool(min(a.jobs, len(dirs))) as pool:
        rows = pool.map(read_job, [(str(a.src), lb, d, model, seconds, a.phases) for lb, d in dirs])
    for r in rows:
        line = ", ".join(f"{s}: own {w['own_phase']['share_of_saved']:.4f} mean {w['phases']['share_of_saved']['mean']:.4f}"
                         f" analytic {r['reuse']['lost_of_saved'][s]:.4f}" for s, w in r["windows"].items())
        q = r["reuse"]["quantiles_s"]
        print(f"[pw] {r['trace']}: {r['requests']} requests over {r['span_s']} s, saved {r['saved_share']:.4f} of "
              f"computed FLOPs; reuse distance mean {r['reuse']['mean_s']:.1f} s, median {q['0.5']:.1f} s, p90 "
              f"{q['0.9']:.1f} s; lost of saved by window s: {line}", flush=True)
    Path(a.out).write_text(json.dumps({"seconds": seconds, "phases": a.phases, "flop_model": model, "rows": rows},
                                      indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
