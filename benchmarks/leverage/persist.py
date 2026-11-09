"""Experiment 7: one fault in a shared prefix, read by every later served token and by every request that shares it.

The fault sits in a prefix (at the attention sink's position by default, or an ordinary one), so what it changes is
cached as keys and values that every later position attends to. That is one incorrect proof block read by every
served token of a long decode, and by every request that shares the prefix (a chat session's earlier turns, a cached
system prompt). `reach.py` and `attack.py` fault the prompt's last position; this puts the fault where it persists.

Read modes (--mode):
  decode  each prompt after the faulted prefix, N served tokens (--new). Per fault, the direct tokens (those no honest
          query on the served prefix reproduces, in lockstep as in reach.py) and the tokens-only code that bounds
          what they tell a decoder: Elias gamma of D + 1, log2 C(N, D) for their positions, and each direct token
          either new (1 + log2(V - 1) bits) or a repeat of one already seen (1 + log2 of how many).
  oracle  Q receiver-chosen requests after the faulted prefix (--queries), the first --obs served tokens of each, the
          same code over all Q·obs reads of the one fault, and two decoders that never see the weights:
            pairs  each direct read (honest y0, served y1) is read as "the fault prefers y1 to y0", scored against
                   the order of the vocabulary along the fault's leave-one-out mean effect on the final norm's output
                   (u: what the logits move by, E u, is a fixed function of the weights if u is a fixed direction);
            top    the distinct served tokens of the direct reads, scored as the vocabulary's top set along u: if all
                   of them rank within u's top K, they carry log2 C(V, n) - log2 C(K, n) bits about u's order. The
                   honest tokens of the same requests, ranked the same way, are the baseline.

Faults (--values): unclamped, multiples of the site's RMS at the fault position, `zero`, and `neg` (the honest value's
negation); under --boxes, `box` (plus and minus the coordinate's box at the position's class, the most a correct
reader admits), `own` at a linear's input (one FP8 code at its token's own group scale), `zero` and `neg`. Coordinates
(--coords per site): the widest at the position's class (the massive-activation channels at the sink), then random.

Sampling (--temperature): 0 is greedy. Above 0, --seeds known fixes the Gumbel noise by (seed, request, step) and the
faulted and honest decodes read the same noise (a decoder that knows the seed couples them); --seeds hidden draws
independent noise for each of --repeats reads of a request, faulted and honest, and the decoder compares the
pluralities. A seed the server can't control is either: the decoder's own (known) or a trusted device's (hidden).

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/persist.py --model gpt2 --mode oracle \\
      --boxes boxes.pt --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, Fault, served_logprobs, sink_span, ue8m0_scale
from prompts import PROMPTS
from traffic import HELDOUT_TEXTS

PREFIX = "You are a helpful assistant. Answer every question briefly and accurately.\n\n"
SITES = ("resid", "mlph", "v", "k", "attn", "mlp")


def copy_cache(c: Cache) -> Cache:
    return Cache(dict(c.kv), c.sink)


def fault_position(lm: LM, ids: torch.Tensor, where: str) -> int:
    """The sink: the first sink host (the boxes' hosts, else position 0); ordinary: the prefix's last position."""
    if where == "ordinary":
        return ids.shape[1] - 1
    if lm.boxes is not None and lm.boxes.hosts is not None:
        return int(sink_span(ids, lm.boxes.hosts)[0, 0])
    return 0


def layers_of(lm: LM, spec: str) -> list[int]:
    L = lm.spec.layers
    return sorted({(int(x) if int(x) >= 0 else L + int(x)) for x in spec.split(",")})


def faults_at(lm: LM, taps: dict, pf: int, pc: int, sites: list[str], layers: list[int], values: list[str],
              coords: int, gen: torch.Generator) -> list[dict]:
    """The faults at position pf (class pc): per (site, layer), the `coords` widest coordinates at the class (by box, or
    unclamped by honest |value|), then random ones, each at every value of `values`."""
    out = []
    for s in sites:
        for l in layers:
            x = taps[(s, l)][0, pf].float()
            rms = float(x.pow(2).mean().sqrt())
            hi = lm.boxes.hi[(s, l)][pc].float() if lm.boxes is not None else None
            order = (hi if hi is not None else x.abs()).argsort(descending=True, stable=True).tolist()
            n_wide = (coords + 1) // 2
            rest = [c for c in torch.randperm(x.numel(), generator=gen).tolist() if c not in order[:n_wide]]
            for rank, c in enumerate(order[:n_wide] + rest[:coords - n_wide]):
                hv = float(x[c])
                cands = []
                for v in values:
                    if v == "zero":
                        cands.append((v, 0.0))
                    elif v == "neg":
                        cands.append((v, -hv))
                    elif v == "box":
                        if hi is None:
                            raise ValueError("--values box needs --boxes")
                        cands += [("box+", float(hi[c])), ("box-", -float(hi[c]))]
                    elif v == "own":
                        if s not in GROUP_SITES:
                            continue
                        g = x[c // GROUP * GROUP:(c // GROUP + 1) * GROUP]
                        own = E4M3_MAX * float(ue8m0_scale(g.abs().max()))
                        if hi is not None:
                            own = min(own, float(hi[c]))
                        cands += [("own+", own), ("own-", -own)]
                    else:
                        m = float(v)
                        cands += [(f"rms{v}", m * rms), (f"rms-{v}", -m * rms)]
                for kind, val in cands:
                    out.append({"site": s, "layer": l, "coord": c, "wide": rank < n_wide, "kind": kind,
                                "value": float(torch.tensor(val).to(torch.bfloat16)), "honest": hv, "rms": rms,
                                "box": float(hi[c]) if hi is not None else None})
    return out


def suffixes(tok, n: int, gen: torch.Generator, lo: int = 6, hi: int = 40) -> list[torch.Tensor]:
    """n receiver-chosen requests: PROMPTS and held-out texts, each cut after a random number of tokens."""
    texts = list(PROMPTS) + list(HELDOUT_TEXTS)
    out = []
    for i in range(n):
        ids = tok.encode(texts[i % len(texts)]).ids
        k = int(torch.randint(min(lo, len(ids)), min(hi, len(ids)) + 1, (1,), generator=gen))
        out.append(torch.tensor([ids[:k]]))
    return out


class Noise:
    def __init__(self, temperature: float, seed: int, vocab: int):
        self.t, self.seed, self.vocab = temperature, seed, vocab

    def gumbel(self, *key: int, rows: int = 1) -> torch.Tensor | None:
        if self.t == 0:
            return None
        h = self.seed
        for k in key:
            h = (h * 1_000_003 + k) % (1 << 62)
        g = torch.Generator().manual_seed(h)
        u = torch.rand(rows, self.vocab, generator=g, dtype=torch.float64).clamp_min(1e-300)
        return (-torch.log(-torch.log(u))).float()

    def pick(self, lp: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        return lp.argmax(-1) if g is None else (lp / self.t + g).argmax(-1)


def gamma_bits(x: int) -> float:
    return 2 * math.floor(math.log2(max(x, 1))) + 1


def code_bits(direct: np.ndarray, served: np.ndarray, vocab: int) -> float:
    """A tokens-only prefix code for one fault's reads given the honest ones: how many differ, where, and what each
    differing read served (a new token, or a repeat of one already seen)."""
    n, d = direct.size, int(direct.sum())
    bits = gamma_bits(d + 1) + (math.lgamma(n + 1) - math.lgamma(d + 1) - math.lgamma(n - d + 1)) / math.log(2)
    seen: list[int] = []
    for y in served[direct]:
        y = int(y)
        if y in seen:
            bits += 1 + math.log2(max(len(seen), 1))
        else:
            bits += 1 + math.log2(vocab - 1)
            seen.append(y)
    return bits


def log2_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def top_bits(ranks: np.ndarray, vocab: int) -> tuple[float, int]:
    """Bits the set of tokens at these 0-based ranks along u says about u's order: log2 C(V, n) - log2 C(K, n), K the
    smallest top set holding them all."""
    ranks = np.unique(ranks)
    if len(ranks) == 0:
        return 0.0, 0
    k = int(ranks.max()) + 1
    return log2_comb(vocab, len(ranks)) - log2_comb(k, len(ranks)), k


def honest_first(lm: LM, pre_h: Cache, sfx: torch.Tensor, precision: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A request's honest first position after the honest prefix: served logprobs (1, V) and final norm output (1, d)."""
    L = lm.spec.layers
    th = {("final", L): None}
    z = lm.forward(sfx, copy_cache(pre_h), taps=th, last_only=True)[:, -1]
    return served_logprobs(z, precision), th[("final", L)][:, -1].float()


def serve_oracle(lm: LM, pre_f: Cache, pre_h: Cache, sfx: torch.Tensor, hon: tuple[torch.Tensor, torch.Tensor],
                 b: int, obs: int, noise: Noise, q: int, precision: str, hidden: int) -> dict:
    """One request after the faulted prefixes (b rows) and the honest one (`hon`, its first position): served tokens,
    honest tokens on the same served prefix, and the final norm's output at the first served position (for scoring
    only)."""
    L = lm.spec.layers
    cf = copy_cache(pre_f)
    tf = {("final", L): None}
    zf = lm.forward(sfx.expand(b, -1), cf, taps=tf, last_only=True)[:, -1]
    lf, lh = served_logprobs(zf, precision), hon[0].expand(b, -1)
    out = {"delta": tf[("final", L)][:, -1].float() - hon[1]}
    if hidden:
        # Independent noise per read: the decoder compares pluralities over `hidden` reads of each.
        gf, gh = noise.gumbel(q, 1, rows=hidden), noise.gumbel(q, 2, rows=hidden)
        yf = torch.stack([noise.pick(lf, gf[r:r + 1].expand(b, -1)) for r in range(hidden)], 1)
        yh = torch.stack([noise.pick(lh, gh[r:r + 1].expand(b, -1)) for r in range(hidden)], 1)
        out["served"] = torch.mode(yf, 1).values[:, None]
        out["honest"] = torch.mode(yh, 1).values[:, None]
        tk = noise.t
        a, c = torch.log_softmax(lf / tk, -1), torch.log_softmax(lh / tk, -1)
        out["kl"] = (a.exp() * (a - c)).sum(-1).clamp_min(0)[:, None] / math.log(2)
        return out
    ys, yh_ = [], []
    ch = None
    if obs > 1:
        ch = copy_cache(pre_h).expand(b)
        lm.forward(sfx.expand(b, -1), ch, last_only=True)
    for step in range(obs):
        if step:
            lf = served_logprobs(lm.forward(ys[-1][:, None], cf)[:, -1], precision)
            lh = served_logprobs(lm.forward(ys[-1][:, None], ch)[:, -1], precision)
        g = noise.gumbel(q, step)
        ys.append(noise.pick(lf, g))
        yh_.append(noise.pick(lh, g))
    out["served"], out["honest"] = torch.stack(ys, 1), torch.stack(yh_, 1)
    return out


def decode(lm: LM, pre_f: Cache, pre_h: Cache, prompt: torch.Tensor, b: int, n: int, noise: Noise, p: int,
           precision: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The faulted decode and the honest one on the same served prefix, in lockstep: served, honest, KL (bits)."""
    cf, ch = copy_cache(pre_f), copy_cache(pre_h).expand(b)
    tok = prompt.expand(b, -1)
    ys, yh, kl = [], [], []
    tk = noise.t or 1.0
    for step in range(n):
        lf = served_logprobs(lm.forward(tok, cf, last_only=True)[:, -1], precision)
        lh = served_logprobs(lm.forward(tok, ch, last_only=True)[:, -1], precision)
        g = noise.gumbel(p, step)
        y = noise.pick(lf, g)
        ys.append(y)
        yh.append(noise.pick(lh, g))
        a, c = torch.log_softmax(lf / tk, -1), torch.log_softmax(lh / tk, -1)
        kl.append((a.exp() * (a - c)).sum(-1).clamp_min(0) / math.log(2))
        tok = y[:, None]
    return torch.stack(ys, 1), torch.stack(yh, 1), torch.stack(kl, 1)


def score_oracle(served: np.ndarray, honest: np.ndarray, deltas: np.ndarray, E: torch.Tensor, vocab: int) -> dict:
    """The decoders' bits for one fault over Q requests (first served token). deltas (Q, d): the fault's effect on the
    final norm's output at each request's first served position."""
    direct = served[:, 0] != honest[:, 0]
    Q = len(direct)
    out = {"direct": int(direct.sum()), "reads": Q}
    if not direct.any():
        return out | {"pairs": 0, "pairs_ok": 0, "pair_bits": 0.0, "top_bits": 0.0, "top_k": 0, "base_bits": 0.0,
                      "base_k": 0, "cos_mean": None}
    D = torch.from_numpy(deltas).float()
    tot = D.sum(0)
    cos, ok, seen = [], 0, set()
    n_pairs = 0
    for q in np.nonzero(direct)[0]:
        u = (tot - D[q]) / max(Q - 1, 1)
        if float(u.norm()) == 0:
            continue
        cos.append(float(torch.nn.functional.cosine_similarity(D[q], u, 0)))
        y0, y1 = int(honest[q, 0]), int(served[q, 0])
        if (y0, y1) in seen:
            continue
        seen.add((y0, y1))
        n_pairs += 1
        ok += int(float((E[y1] - E[y0]).float() @ u) > 0)
    acc = ok / n_pairs if n_pairs else 0.0
    h = 0.0 if acc in (0.0, 1.0) else -(acc * math.log2(acc) + (1 - acc) * math.log2(1 - acc))
    u_all = tot / Q
    score = (E.float() @ u_all)
    order = torch.argsort(score, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    tb, tk = top_bits(rank[torch.from_numpy(served[direct, 0])].numpy(), vocab)
    bb, bk = top_bits(rank[torch.from_numpy(honest[direct, 0])].numpy(), vocab)
    return out | {"pairs": n_pairs, "pairs_ok": ok, "pair_bits": n_pairs * max(0.0, 1 - h) if acc > 0.5 else 0.0,
                  "top_bits": tb, "top_k": tk, "top_n": int(len(np.unique(served[direct, 0]))), "base_bits": bb,
                  "base_k": bk, "cos_mean": float(np.mean(cos)) if cos else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="oracle", choices=("oracle", "decode"))
    ap.add_argument("--boxes", type=Path, help="clamp every read to clamp.py's calibrated boxes.pt")
    ap.add_argument("--where", default="sink", choices=("sink", "ordinary"), help="the fault's prefix position")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--layers", default="0,2,5,8,10,11", help="comma-separated; negative from the end")
    ap.add_argument("--values", default="zero,neg,8,64,512,4096",
                    help="zero, neg, RMS multiples (unclamped), box and own (under --boxes)")
    ap.add_argument("--coords", type=int, default=4, help="per site and layer: the widest half, then random")
    ap.add_argument("--queries", type=int, default=128, help="oracle: receiver-chosen requests sharing the prefix")
    ap.add_argument("--obs", type=int, default=1, help="oracle: served tokens a request returns")
    ap.add_argument("--prompts", type=int, default=4, help="decode: prompts after the prefix")
    ap.add_argument("--new", type=int, default=128, help="decode: served tokens")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="known", choices=("known", "hidden"))
    ap.add_argument("--repeats", type=int, default=16, help="--seeds hidden: reads of each request, faulted and honest")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-faults", type=int, default=0, help="keep at most this many faults (0: all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    if a.boxes is not None:
        lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    V = lm.spec.vocab
    gen = torch.Generator().manual_seed(a.seed)
    noise = Noise(a.temperature, a.seed, V)
    hidden = a.repeats if (a.temperature > 0 and a.seeds == "hidden") else 0
    if hidden and a.mode == "decode":
        ap.error("--seeds hidden is for --mode oracle; decode reports the KL bound")
    sites = a.sites.split(",")
    layers = layers_of(lm, a.layers)
    values = a.values.split(",")

    ids = torch.tensor([tok.encode(a.prefix).ids])
    pf = fault_position(lm, ids, a.where)
    taps = {(s, l): None for s in sites for l in layers}
    pre_h = Cache()
    lm.forward(ids, pre_h, taps=taps)
    pc = int(lm.boxes.classes(0, ids.shape[1], ids)[0][0, pf]) if lm.boxes is not None else 1
    faults = faults_at(lm, taps, pf, pc, sites, layers, values, a.coords, gen)
    if a.max_faults:
        faults = faults[:a.max_faults]
    E = lm.E.float()
    t_start = time.time()
    if a.mode == "oracle":
        reqs = suffixes(tok, a.queries, gen)
        hons = [honest_first(lm, pre_h, sfx, a.precision) for sfx in reqs]
    else:
        reqs = [torch.tensor([tok.encode(t).ids]) for t in PROMPTS[:a.prompts]]
    recs = []
    for i in range(0, len(faults), a.batch):
        chunk = faults[i:i + a.batch]
        b = len(chunk)
        pre_f = Cache()
        lm.forward(ids.expand(b, -1), pre_f, fault=Fault.of(pf, [(f["site"], f["layer"], f["coord"], f["value"])
                                                                  for f in chunk]))
        if a.mode == "oracle":
            S, H, Dl, KL = [], [], [], []
            for q, sfx in enumerate(reqs):
                o = serve_oracle(lm, pre_f, pre_h, sfx, hons[q], b, a.obs, noise, q, a.precision, hidden)
                S.append(o["served"])
                H.append(o["honest"])
                Dl.append(o["delta"])
                if "kl" in o:
                    KL.append(o["kl"])
            S, H, Dl = torch.stack(S, 1).numpy(), torch.stack(H, 1).numpy(), torch.stack(Dl, 1).numpy()
            KL = torch.stack(KL, 1).numpy() if KL else None
            for j, f in enumerate(chunk):
                direct = S[j] != H[j]
                r = dict(f) | score_oracle(S[j], H[j], Dl[j], E, V)
                r["code_bits"] = code_bits(direct.reshape(-1), S[j].reshape(-1), V)
                r["direct_reads"] = int(direct.sum())
                if KL is not None:
                    r["kl_bits"] = float(KL[j].sum())
                recs.append(r)
        else:
            per = [dict(f) | {"direct": 0, "code_bits": 0.0, "kl_bits": 0.0, "distinct": 0, "reads": 0} for f in chunk]
            for p, prompt in enumerate(reqs):
                ys, yh, kl = decode(lm, pre_f, pre_h, prompt, b, a.new, noise, p, a.precision)
                d = (ys != yh).numpy()
                for j in range(b):
                    per[j]["direct"] += int(d[j].sum())
                    per[j]["code_bits"] += code_bits(d[j], ys[j].numpy(), V)
                    per[j]["kl_bits"] += float(kl[j].sum())
                    per[j]["distinct"] += len(set(ys[j][torch.from_numpy(d[j])].tolist()))
                    per[j]["reads"] += a.new
            recs += per
        print(f"faults {i + b}/{len(faults)}, {time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
        (a.out / "persist.json").write_text(json.dumps({"partial": True, "faults": recs}))

    groups = {}
    for r in recs:
        key = (r["site"], r["layer"], r["kind"], r["wide"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (s, l, kind, wide), rs in sorted(groups.items()):
        row = {"site": s, "layer": l, "kind": kind, "wide": wide, "n": len(rs),
               "direct_mean": float(np.mean([r["direct"] for r in rs])),
               "direct_max": int(max(r["direct"] for r in rs)),
               "code_mean": float(np.mean([r["code_bits"] for r in rs])),
               "code_max": float(max(r["code_bits"] for r in rs))}
        for key in ("pair_bits", "top_bits", "base_bits", "kl_bits"):
            if key in rs[0]:
                vals = [r[key] for r in rs]
                row[key + "_mean"], row[key + "_max"] = float(np.mean(vals)), float(max(vals))
        rows.append(row)
    best = {key: max(recs, key=lambda r: r.get(key, 0.0)) for key in ("code_bits", "pair_bits", "top_bits")
            if key in recs[0]}
    summary = {"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": V, "bits_per_token": math.log2(V),
               "mode": a.mode, "where": a.where, "fault_position": pf, "fault_class": pc, "prefix_tokens": ids.shape[1],
               "boxes": str(a.boxes) if a.boxes else None, "temperature": a.temperature, "seeds": a.seeds,
               "repeats": hidden, "reads_per_fault": (a.queries * a.obs if a.mode == "oracle" else a.prompts * a.new),
               "faults": len(recs), "seconds": time.time() - t_start, "best": best, "groups": rows,
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    if lm.boxes is not None:
        summary["values_clamped"] = {f"{s}:{l}": n for (s, l), n in lm.boxes.cut.items()}
    (a.out / "persist.json").write_text(json.dumps(summary | {"faults_detail": recs}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
