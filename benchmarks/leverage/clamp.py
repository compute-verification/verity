"""Experiment 4: the range clamps' boxes, calibrated the way the structural design sets them, and checked on held-out
traffic.

Every value a proof block reads from another is clamped by its reader to a box fixed before the period (`lm.Boxes`): a
linear's input (`lm.GROUP_SITES`) per group of 128 coordinates, through its FP8 per-token-group scale, and every other
read (the residual stream, a linear's output into the residual add, the attention cache's keys and values) per
element. Each box is per position class (`lm.position_class`: position 0, where attention sinks put their massive
activations, and every other position). A box is the honest calibration traffic's largest magnitude with one power of
two of headroom: an element's is the next power of two above its maximum, doubled; a group's is 448 times its largest
UE8M0 scale, doubled (one more exponent step).

Parts (--part):

  calibrate  honest forwards over `traffic.CALIBRATION` (each text with its own greedy continuation of --new tokens)
             and every one of --first-tokens random tokens at position 0. Writes boxes.pt and calibrate.json: per site
             and position class, the boxes against the site's RMS (how far an in-range fault reaches, in the units of
             experiment 1's sweeps), and the massive activations (an element's largest value against its site's
             median RMS, with the token it sits on).
  complete   honest forwards over `traffic.HELDOUT` and other random first tokens, under the clamps: every value the
             clamps cut (an honest value outside its box is a completeness failure), and whether the clamped model's
             served greedy continuation differs from the unclamped one's.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part calibrate --out "$OUT"
  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part complete \\
      --boxes "$OUT/boxes.pt" --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, position_class, ue8m0_scale
from traffic import CALIBRATION, HELDOUT


def continued(lm: LM, tok, text: str, new: int, cap: int) -> list[int]:
    """The text's first `cap` tokens and the model's greedy continuation of `new` more, as served (under lm.boxes)."""
    ids = tok.encode(text).ids[:cap]
    c = Cache()
    z = lm.forward(torch.tensor([ids]), c, last_only=True)[:, -1]
    out = list(ids)
    for _ in range(new):
        y = int(z.argmax(-1))
        out.append(y)
        z = lm.forward(torch.tensor([[y]]), c)[:, -1]
    return out


class Stats:
    """Per (site, layer) and position class, each element's largest |value| (with where it was) and sum of squares."""

    def __init__(self, lm: LM):
        self.keys = lm.sites()
        self.amax = {k: torch.zeros(2, lm.width(k[0])) for k in self.keys}
        self.at = {k: torch.full((2, lm.width(k[0]), 2), -1, dtype=torch.long) for k in self.keys}
        self.ss = {k: torch.zeros(2, lm.width(k[0]), dtype=torch.float64) for k in self.keys}
        self.n = torch.zeros(2, dtype=torch.long)
        self.tokens: list[list[int]] = []

    def add(self, taps: dict, seq: int, positions: torch.Tensor) -> None:
        """taps[key]: (B, T, width) at `positions` (T,) of sequence `seq` (B = 1), or B sequences of one position."""
        cls = position_class(positions)
        for k in self.keys:
            x = taps[k].abs().float()
            b, t, w = x.shape
            flat = x.reshape(b * t, w)
            c = cls.repeat(b) if t > 1 else cls.expand(b)
            where = torch.stack([torch.full((b * t,), seq) if b == 1 else torch.arange(b) + seq,
                                 positions.repeat(b) if t > 1 else positions.expand(b)], 1)
            for ci in (0, 1):
                sel = c == ci
                if not sel.any():
                    continue
                v, idx = flat[sel].max(0)
                better = v > self.amax[k][ci]
                self.amax[k][ci] = torch.where(better, v, self.amax[k][ci])
                self.at[k][ci] = torch.where(better[:, None], where[sel][idx], self.at[k][ci])
                self.ss[k][ci] += flat[sel].double().pow(2).sum(0)
        self.n += torch.bincount(position_class(positions).repeat(taps[self.keys[0]].shape[0]), minlength=2)

    def boxes(self) -> dict[tuple[str, int], torch.Tensor]:
        hi = {}
        for k, a in self.amax.items():
            if k[0] in GROUP_SITES:
                g = a.view(2, -1, GROUP).amax(-1)
                hi[k] = (E4M3_MAX * 2 * ue8m0_scale(g)).repeat_interleave(GROUP, -1)
            else:
                hi[k] = torch.exp2(torch.ceil(torch.log2(a.clamp_min(2.0 ** -24))) + 1)
        return hi


def sweep_first(lm: LM, stats: Stats | None, n: int, seed: int, batch: int, seq0: int) -> None:
    """n random tokens, each alone at position 0."""
    gen = torch.Generator().manual_seed(seed)
    toks = torch.randperm(lm.spec.vocab, generator=gen)[:n]
    for i in range(0, n, batch):
        ids = toks[i:i + batch, None]
        taps = {k: None for k in lm.sites()} if stats is not None else None
        lm.forward(ids, Cache(), taps=taps)
        if stats is not None:
            stats.add(taps, seq0 + i, torch.zeros(1, dtype=torch.long))
            stats.tokens.extend([[int(t)] for t in ids[:, 0]])


def quantiles(x: torch.Tensor) -> dict:
    x = x.flatten().float()
    return {"median": float(x.median()), "p90": float(x.quantile(0.9)), "max": float(x.max())}


def part_calibrate(lm: LM, tok, a) -> dict:
    stats = Stats(lm)
    t0 = time.time()
    for i, text in enumerate(CALIBRATION):
        ids = continued(lm, tok, text, a.new, a.cap)
        taps = {k: None for k in lm.sites()}
        lm.forward(torch.tensor([ids]), Cache(), taps=taps)
        stats.tokens.append(ids)
        stats.add(taps, len(stats.tokens) - 1, torch.arange(len(ids)))
        print(f"calibrate {i}: {len(ids)} tokens, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    sweep_first(lm, stats, a.first_tokens, a.seed, a.batch, len(stats.tokens))
    hi = stats.boxes()
    Boxes(hi).save(a.out / "boxes.pt", model=a.model, revision=MODELS[a.model]["revision"],
                   amax={f"{s}:{l}": v for (s, l), v in stats.amax.items()})
    rows, massive = [], []
    for k in stats.keys:
        for ci in (0, 1):
            n = int(stats.n[ci])
            rms_el = (stats.ss[k][ci] / max(n, 1)).sqrt().float()
            site_rms = float((stats.ss[k][ci].sum() / max(n * rms_el.numel(), 1)).sqrt())
            box = hi[k][ci]
            box_u = box[::GROUP] if k[0] in GROUP_SITES else box
            rows.append({"site": k[0], "layer": k[1], "class": ci, "positions": n, "site_rms": site_rms,
                         "box_over_site_rms": quantiles(box_u / max(site_rms, 1e-30)),
                         "amax_over_site_rms": quantiles(stats.amax[k][ci] / max(site_rms, 1e-30)),
                         "box_over_element_rms": quantiles(box / rms_el.clamp_min(1e-30))})
            med = float(rms_el.median())
            top = stats.amax[k][ci].topk(3)
            for v, j in zip(top.values.tolist(), top.indices.tolist()):
                seq, pos = stats.at[k][ci, j].tolist()
                ids = stats.tokens[seq] if 0 <= seq < len(stats.tokens) else []
                massive.append({"site": k[0], "layer": k[1], "class": ci, "coord": j, "value": v,
                                "over_median_element_rms": v / max(med, 1e-30), "over_site_rms": v / max(site_rms, 1e-30),
                                "box": float(hi[k][ci, j]), "seq": seq, "pos": pos,
                                "token": tok.decode([ids[pos]]) if 0 <= pos < len(ids) else None})
    return {"part": "calibrate", "texts": len(CALIBRATION), "first_tokens": a.first_tokens, "new": a.new,
            "cap": a.cap, "positions": stats.n.tolist(), "sites": rows, "massive": massive}


def part_complete(lm: LM, tok, a) -> dict:
    boxes = Boxes.load(a.boxes)
    per_text = []
    t0 = time.time()
    seen: dict[tuple[str, int], int] = {}
    for i, text in enumerate(HELDOUT):
        lm.boxes = None
        plain = continued(lm, tok, text, a.new, a.cap)
        lm.boxes = boxes
        clamped = continued(lm, tok, text, a.new, a.cap)
        boxes.cut, boxes.worst = {}, {}
        lm.forward(torch.tensor([clamped]), Cache())
        n_prompt = min(len(tok.encode(text).ids), a.cap)
        diff = [j for j in range(n_prompt, len(plain)) if plain[j] != clamped[j]]
        per_text.append({"text": i, "tokens": len(clamped), "served_differ": len(diff),
                         "first_differ": diff[0] - n_prompt if diff else None,
                         "cut": {f"{s}:{l}": n for (s, l), n in boxes.cut.items()},
                         "worst": {f"{s}:{l}": w for (s, l), w in boxes.worst.items()}})
        for k in lm.sites():
            seen[k] = seen.get(k, 0) + len(clamped) * lm.width(k[0])
        print(f"complete {i}: {len(clamped)} tokens, {len(diff)} served tokens differ, cuts "
              f"{sum(boxes.cut.values())}, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    boxes.cut, boxes.worst = {}, {}
    lm.boxes = boxes
    sweep_first(lm, None, a.first_tokens, a.seed + 1, a.batch, 0)
    first = {"tokens": a.first_tokens, "cut": {f"{s}:{l}": n for (s, l), n in boxes.cut.items()},
             "worst": {f"{s}:{l}": w for (s, l), w in boxes.worst.items()}}
    cut_total = sum(sum(t["cut"].values()) for t in per_text)
    return {"part": "complete", "texts": len(HELDOUT), "new": a.new, "cap": a.cap, "values_seen": sum(seen.values()),
            "values_cut": cut_total, "texts_with_cuts": sum(1 for t in per_text if t["cut"]),
            "texts_served_differ": sum(1 for t in per_text if t["served_differ"]), "per_text": per_text,
            "first_tokens": first}


def context_of(tok, ids: list[int], pos: int, before: int = 6) -> str:
    return tok.decode(ids[max(0, pos - before):pos]) + "⟦" + tok.decode([ids[pos]]) + "⟧"


def part_explain(lm: LM, tok, a) -> dict:
    """Where honest traffic leaves the boxes. Unclamped, over --texts, every value outside its box (a group site: a group
    whose scale exceeds its bound), with its position's token and the tokens before it; and each text's sink-like
    positions, whose largest residual-stream element at --sink-layer is over --sink-ratio times the text's median."""
    boxes = Boxes.load(a.boxes)
    texts = HELDOUT if a.texts == "heldout" else CALIBRATION
    L = lm.spec.layers
    sink_layer = a.sink_layer if a.sink_layer >= 0 else L + a.sink_layer
    per_text, by_token = [], {}
    t0 = time.time()
    for i, text in enumerate(texts):
        lm.boxes = None
        ids = continued(lm, tok, text, a.new, a.cap)
        taps = {k: None for k in lm.sites()}
        lm.forward(torch.tensor([ids]), Cache(), taps=taps)
        cls = position_class(torch.arange(len(ids)))
        events = []
        for k in lm.sites():
            x = taps[k][0].float()
            hb = boxes.hi[k][cls].float()
            if k[0] in GROUP_SITES:
                amax = x.abs().view(len(ids), -1, GROUP).amax(-1)
                ratio = amax / hb[:, ::GROUP]
                over = ue8m0_scale(amax) > hb[:, ::GROUP] / E4M3_MAX
                coord = x.abs().view(len(ids), -1, GROUP).argmax(-1) + torch.arange(0, x.shape[1], GROUP)
            else:
                ratio = x.abs() / hb
                over = ratio > 1
                coord = torch.arange(x.shape[1]).expand_as(x)
            for t, c in over.nonzero().tolist():
                events.append((float(ratio[t, c]), k[0], k[1], t, int(coord[t, c]), float(x[t, int(coord[t, c])].abs())))
        events.sort(reverse=True)
        first: dict[int, tuple] = {}
        for e in events:
            if e[3] not in first or (e[2], e[1]) < (first[e[3]][2], first[e[3]][1]):
                first[e[3]] = e
        for t, e in first.items():
            s = by_token.setdefault(tok.decode([ids[t]]), {"positions": 0, "max_ratio": 0.0})
            s["positions"] += 1
            s["max_ratio"] = max(s["max_ratio"], max(x[0] for x in events if x[3] == t))
        r = taps[("resid", sink_layer)][0].float().abs()
        peak, at = r.max(-1)
        med = float(peak.median())
        sinks = [{"pos": t, "context": context_of(tok, ids, t), "peak": float(peak[t]), "coord": int(at[t]),
                  "over_median": float(peak[t]) / max(med, 1e-30)}
                 for t in range(len(ids)) if float(peak[t]) > a.sink_ratio * med]
        top = [{"ratio": e[0], "site": e[1], "layer": e[2], "pos": e[3], "coord": e[4], "value": e[5],
                "context": context_of(tok, ids, e[3])} for e in events[:a.top]]
        roots = sorted(({"pos": t, "site": e[1], "layer": e[2], "ratio": e[0], "context": context_of(tok, ids, t)}
                        for t, e in first.items()), key=lambda r: (r["layer"], -r["ratio"]))
        per_text.append({"text": i, "tokens": len(ids), "values_over": len(events), "positions_over": len(first),
                         "top": top, "first_over_by_position": roots[:a.top], "sinks": sinks})
        print(f"explain {i}: {len(ids)} tokens, {len(events)} values over in {len(first)} positions; worst "
              + (f"{events[0][0]:.2f}x at {events[0][1]}:{events[0][2]} {context_of(tok, ids, events[0][3])!r}"
                 if events else "none")
              + f"; sinks {[(s['pos'], s['context'][-12:], round(s['over_median'])) for s in sinks]}, {time.time() - t0:.0f} s",
              file=sys.stderr, flush=True)
    return {"part": "explain", "texts": a.texts, "sink_layer": sink_layer, "per_text": per_text,
            "by_token": dict(sorted(by_token.items(), key=lambda kv: -kv[1]["max_ratio"]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--part", required=True, choices=("calibrate", "complete", "explain"))
    ap.add_argument("--texts", default="heldout", choices=("heldout", "calibration"), help="explain: which traffic")
    ap.add_argument("--sink-layer", type=int, default=3, help="explain: the residual stream that marks sinks")
    ap.add_argument("--sink-ratio", type=float, default=16.0,
                    help="explain: a sink-like position's residual peak over the text's median peak")
    ap.add_argument("--top", type=int, default=12, help="explain: the largest overshoots and positions listed per text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, help="complete: the calibrated boxes.pt")
    ap.add_argument("--new", type=int, default=64, help="greedy continuation tokens a text adds to the traffic")
    ap.add_argument("--cap", type=int, default=768, help="a text's tokens, at most")
    ap.add_argument("--first-tokens", type=int, default=2048, help="random tokens swept at position 0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    tok = lm.tokenizer()
    t = time.time()
    res = {"calibrate": part_calibrate, "complete": part_complete, "explain": part_explain}[a.part](lm, tok, a)
    res.update({"model": a.model, "revision": MODELS[a.model]["revision"],
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "seconds": time.time() - t})
    (a.out / f"clamp-{a.part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("sites", "massive", "per_text")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
