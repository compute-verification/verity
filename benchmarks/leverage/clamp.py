"""Experiment 4: the range clamps' boxes, calibrated the way the structural design sets them, and checked on held-out
traffic.

Every value a proof block reads from another is clamped by its reader to a box fixed before the period (`lm.Boxes`): a
linear's input (`lm.GROUP_SITES`) per group of 128 coordinates, through its FP8 per-token-group scale, and every other
read (the residual stream, a linear's output into the residual add, the attention cache's keys and values) per
element. Each box is per position class (`lm.position_class`, read from positions and token ids alone): every position
before a sequence's first sink host (--hosts, from the hosts part), the run of its token from it and --window - 1
positions after the run, where attention sinks put their massive activations (position 0, unless the sequence opens
with tokens that host none, such as Qwen's digits); the tokenizer's special tokens after them (document separators and
chat delimiters, sinks too); every other position; and, with --run R, every position from the R-th copy of a host
token in a row to the end of the sequence (a long run forms the sink again and puts sinks on ordinary tokens after
it), calibrated on `traffic.run_texts`. Without --hosts the sink class is the first --window positions. A box is the
honest calibration traffic's largest magnitude with --headroom powers of two (default one; one value, or one a class):
an element's is the next power of two above its maximum, doubled; a group's is 448 times its largest UE8M0 scale,
doubled (one more exponent step).

Traffic is the hand-written sets (`traffic.CALIBRATION`, `traffic.HELDOUT`) and, with --mixed N, N texts of the pinned
datasets' (`traffic.mixed`; Pile documents need `--with pyarrow`); --shard i/n splits a part's texts across runs.
Calibration's texts end by position --cap + --new; complete and explain with --long N check N held-out Pile documents
of --cap tokens instead (`traffic.long_texts`), positions calibration never saw. Calibrate with --run adds the
calibration run texts, and complete and explain with --runs the held-out ones (`traffic.run_texts`): runs of one token,
long and short of the threshold, each followed by an ordinary text.

Parts (--part):

  hosts      the sink hosts: every token whose residual-stream peak at --sink-layer, alone at position 0, is over
             --sink-ratio times the calibration texts' median. Writes hosts.pt, which calibrate takes as --hosts.
  calibrate  honest forwards over the calibration traffic (each text with its own greedy continuation of --new
             tokens) and --first-tokens tokens at position 0, split across shards (-1: the whole vocabulary, which
             covers position 0 of any traffic, since a model's activations there are a function of the token alone). Writes
             stats.pt and, unsharded, boxes.pt and calibrate.json: per site and position class, the boxes against the
             site's RMS (how far an in-range fault reaches, in the units of experiment 1's sweeps), and the massive
             activations (an element's largest value against its site's median RMS, with the token it sits on).
  merge      joins the calibrate shards' stats.pt (--stats) into boxes.pt and calibrate.json.
  rebox      boxes.pt again from a calibrated one's (--boxes) maxima, at another --headroom.
  complete   honest forwards over the held-out traffic and, in shard 0, other random first tokens, under the clamps:
             every value the clamps cut (an honest value outside its box is a completeness failure), and whether the
             clamped model's served greedy continuation differs from the unclamped one's.
  explain    where honest traffic leaves the boxes, unclamped: each value outside its box with its token and context,
             and each text's sink-like positions with their classes.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part hosts \\
      --window 4 --out "$OUT"
  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part calibrate \\
      --window 4 --hosts "$OUT/hosts.pt" --first-tokens -1 --out "$OUT"
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
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, ue8m0_scale
from traffic import CALIBRATION, HELDOUT, long_texts, mixed, run_texts


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


CLASSES = 4


def padded(amax: torch.Tensor) -> torch.Tensor:
    """A calibration's per-class maxima with a row for every class (earlier calibrations had no run class)."""
    return torch.cat([amax, amax.new_zeros(CLASSES - amax.shape[0], amax.shape[1])]) if amax.shape[0] < CLASSES else amax


class Stats:
    """Per (site, layer) and position class, each element's largest |value| (with where it was) and sum of squares.

    A text is a sequence `seq` (its index in the traffic); the position-0 sweep's token t is sequence SWEEP + t."""

    SWEEP = 1 << 20

    def __init__(self, lm: LM, special: list[int], window: int, hosts: list[int] | None = None, run: int | None = None):
        self.keys = lm.sites()
        self.special, self.window, self.run = torch.tensor(special, dtype=torch.long), window, run
        self.hosts = None if hosts is None else torch.tensor(sorted(hosts), dtype=torch.long)
        self.rule = Boxes({}, special, window, hosts, run)
        self.amax = {k: torch.zeros(CLASSES, lm.width(k[0])) for k in self.keys}
        self.at = {k: torch.full((CLASSES, lm.width(k[0]), 2), -1, dtype=torch.long) for k in self.keys}
        self.ss = {k: torch.zeros(CLASSES, lm.width(k[0]), dtype=torch.float64) for k in self.keys}
        self.n = torch.zeros(CLASSES, dtype=torch.long)
        self.tokens: dict[int, list[int]] = {}

    def add(self, taps: dict, seqs: torch.Tensor, ids: torch.Tensor) -> None:
        """taps[key]: (B, T, width) for B sequences `seqs` (B,) of token ids (B, T) at positions 0..T-1."""
        b, t = ids.shape
        pos = torch.arange(t)[None].expand(b, t)
        c = self.rule.classes(0, t, ids)[0].expand(b, t).reshape(-1)
        where = torch.stack([seqs[:, None].expand(b, t).reshape(-1), pos.reshape(-1)], 1)
        for k in self.keys:
            flat = taps[k].abs().float().reshape(b * t, -1)
            for ci in range(CLASSES):
                sel = c == ci
                if not sel.any():
                    continue
                v, idx = flat[sel].max(0)
                better = v > self.amax[k][ci]
                self.amax[k][ci] = torch.where(better, v, self.amax[k][ci])
                self.at[k][ci] = torch.where(better[:, None], where[sel][idx], self.at[k][ci])
                self.ss[k][ci] += flat[sel].double().pow(2).sum(0)
        self.n += torch.bincount(c, minlength=CLASSES)

    def merge(self, o: "Stats") -> None:
        for k in self.keys:
            better = o.amax[k] > self.amax[k]
            self.amax[k] = torch.where(better, o.amax[k], self.amax[k])
            self.at[k] = torch.where(better[..., None], o.at[k], self.at[k])
            self.ss[k] += o.ss[k]
        self.n += o.n
        self.tokens.update(o.tokens)

    def save(self, path: Path) -> None:
        torch.save({k: getattr(self, k) for k in ("amax", "at", "ss", "n", "tokens", "window", "run")}
                   | {"special": self.special.tolist(), "hosts": None if self.hosts is None else self.hosts.tolist()}, path)

    @classmethod
    def load(cls, lm: LM, path: Path) -> "Stats":
        raw = torch.load(path)
        s = cls(lm, raw["special"], raw["window"], raw.get("hosts"), raw.get("run"))
        for k in ("amax", "at", "ss", "n", "tokens"):
            setattr(s, k, raw[k])
        if s.n.shape[0] < CLASSES:
            s.amax = {k: padded(v) for k, v in s.amax.items()}
            s.at = {k: torch.cat([v, v.new_full((CLASSES - v.shape[0], *v.shape[1:]), -1)]) for k, v in s.at.items()}
            s.ss = {k: torch.cat([v, v.new_zeros(CLASSES - v.shape[0], v.shape[1])]) for k, v in s.ss.items()}
            s.n = torch.cat([s.n, s.n.new_zeros(CLASSES - s.n.shape[0])])
        return s

    def boxes(self, headroom: list[int]) -> dict[tuple[str, int], torch.Tensor]:
        """Each class's box, with `headroom` powers of two (one value, or one a class). The early window, the special
        tokens and the runs are few positions of the calibration traffic, so their boxes also cover every value the
        ordinary class saw, and the run class's every value the special tokens' saw too."""
        h = torch.tensor(headroom * CLASSES if len(headroom) == 1 else headroom, dtype=torch.float32)[:, None]
        hi = {}
        for k, a in self.amax.items():
            a = padded(a)
            a = torch.stack([torch.maximum(a[0], a[1]), a[1], torch.maximum(a[2], a[1]),
                             torch.maximum(torch.maximum(a[3], a[2]), a[1])])
            if k[0] in GROUP_SITES:
                g = a.view(CLASSES, -1, GROUP).amax(-1)
                hi[k] = (E4M3_MAX * torch.exp2(h) * ue8m0_scale(g)).repeat_interleave(GROUP, -1)
            else:
                hi[k] = torch.exp2(torch.ceil(torch.log2(a.clamp_min(2.0 ** -24))) + h)
        return hi


def sweep_first(lm: LM, stats: Stats | None, n: int, seed: int, batch: int, part: str = "0/1") -> None:
    """n random tokens (n < 0: every token of the vocabulary), each alone at position 0, where a model's activations
    are a function of the token alone: the whole vocabulary covers every position 0 of any traffic. `part` "i/k"
    takes every k-th token from token i."""
    V = lm.spec.vocab
    toks = torch.arange(V) if n < 0 else torch.randperm(V, generator=torch.Generator().manual_seed(seed))[:n]
    i, k = (int(x) for x in part.split("/"))
    toks = toks[i::k]
    for i in range(0, len(toks), batch):
        ids = toks[i:i + batch, None]
        taps = {k: None for k in lm.sites()} if stats is not None else None
        lm.forward(ids, Cache(), taps=taps)
        if stats is not None:
            stats.add(taps, Stats.SWEEP + ids[:, 0], ids)


def traffic(split: str, n_mixed: int, n_long: int = 0, cap: int = 0) -> list[tuple[str, str]]:
    """The hand-written set of `split` and n_mixed texts of the pinned datasets' (`traffic.mixed`), as (name, text); with
    n_long, instead, that many held-out Pile documents of at least 5 characters a token of `cap` (`traffic.long_texts`)."""
    if n_long:
        return long_texts(n_long, 5 * cap)
    hand = CALIBRATION if split == "calibration" else HELDOUT
    return [(f"hand:{i}", t) for i, t in enumerate(hand)] + (mixed(split, n_mixed) if n_mixed else [])


def shard(items: list, spec: str) -> list[tuple[int, object]]:
    """spec "i/n": every n-th item from item i, with its index."""
    i, n = (int(x) for x in spec.split("/"))
    return [(j, x) for j, x in enumerate(items) if j % n == i]


def quantiles(x: torch.Tensor) -> dict:
    x = x.flatten().float()
    return {"median": float(x.median()), "p90": float(x.quantile(0.9)), "max": float(x.max())}


def part_hosts(lm: LM, tok, a) -> dict:
    """The sink hosts: every token whose largest residual-stream element at --sink-layer, alone at position 0, is over
    --sink-ratio times the median one of the hand-written calibration texts past the first --window positions. A model
    forms its sink at a sequence's first such token, where the position classes start (`lm.position_class`)."""
    key = ("resid", a.sink_layer if a.sink_layer >= 0 else lm.spec.layers + a.sink_layer)
    peaks = []
    for text in CALIBRATION:
        taps = {key: None}
        lm.forward(torch.tensor([tok.encode(text).ids[:a.cap]]), Cache(), taps=taps)
        peaks.append(taps[key][0, a.window:].abs().amax(-1).float())
    base = float(torch.cat(peaks).median())
    V = lm.spec.vocab
    ratio = torch.empty(V)
    for i in range(0, V, a.batch):
        taps = {key: None}
        lm.forward(torch.arange(i, min(V, i + a.batch))[:, None], Cache(), taps=taps)
        ratio[i:i + a.batch] = taps[key][:, 0].abs().amax(-1).float() / base
    hosts = (ratio > a.sink_ratio).nonzero()[:, 0].tolist()
    torch.save({"hosts": hosts, "base": base, "sink_layer": key[1], "sink_ratio": a.sink_ratio}, a.out / "hosts.pt")
    non = sorted((ratio <= a.sink_ratio).nonzero()[:, 0].tolist(), key=lambda t: float(ratio[t]))
    return {"part": "hosts", "sink_layer": key[1], "sink_ratio": a.sink_ratio, "base": base, "vocab": V,
            "hosts": len(hosts), "ratio_quantiles": {q: float(ratio.quantile(q)) for q in (0.001, 0.01, 0.1, 0.5)},
            "non_hosts": [[tok.decode([t]), t, float(ratio[t])] for t in non]}


def part_calibrate(lm: LM, tok, a) -> dict:
    """The shard's texts and its share of the position-0 sweep; with one shard, the boxes too (else `merge` makes
    them)."""
    stats = Stats(lm, lm.special_ids() if a.special else [], a.window,
                  torch.load(a.hosts)["hosts"] if a.hosts else None, a.run)
    t0 = time.time()
    texts = shard(traffic("calibration", a.mixed) + (run_texts("calibration") if a.run else []), a.shard)
    for j, (name, text) in texts:
        ids = continued(lm, tok, text, a.new, a.cap)
        taps = {k: None for k in lm.sites()}
        lm.forward(torch.tensor([ids]), Cache(), taps=taps)
        stats.tokens[j] = ids
        stats.add(taps, torch.tensor([j]), torch.tensor([ids]))
        print(f"calibrate {j} {name}: {len(ids)} tokens, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    sweep_first(lm, stats, a.first_tokens, a.seed, a.batch, a.shard)
    print(f"calibrate: position-0 sweep, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    stats.save(a.out / "stats.pt")
    if a.shard.split("/")[1] != "1":
        return {"part": "calibrate", "shard": a.shard, "texts": len(texts), "positions": stats.n.tolist()}
    return calibrated(lm, tok, a, stats)


def part_merge(lm: LM, tok, a) -> dict:
    stats = Stats.load(lm, a.stats[0])
    for p in a.stats[1:]:
        stats.merge(Stats.load(lm, p))
    return calibrated(lm, tok, a, stats)


def part_rebox(lm: LM, tok, a) -> dict:
    """boxes.pt again from --boxes' calibrated maxima, at --headroom."""
    raw = torch.load(a.boxes)
    stats = Stats(lm, raw["special"], raw["window"])
    stats.amax = {(k.split(":")[0], int(k.split(":")[1])): v for k, v in raw["amax"].items()}
    Boxes(stats.boxes(a.headroom), raw["special"], raw["window"], raw.get("hosts"), raw.get("run")).save(
        a.out / "boxes.pt", model=a.model, revision=MODELS[a.model]["revision"], headroom=a.headroom,
        amax=raw["amax"])
    return {"part": "rebox", "from": str(a.boxes), "window": raw["window"], "special": len(raw["special"]),
            "headroom": a.headroom}


def calibrated(lm: LM, tok, a, stats: Stats) -> dict:
    """Writes boxes.pt from the merged statistics; per site and class, the boxes against the site's RMS, and the
    largest values (massive activations) with the token each sits on."""
    hi = stats.boxes(a.headroom)
    Boxes(hi, stats.special.tolist(), stats.window, None if stats.hosts is None else stats.hosts.tolist(),
          stats.run).save(
        a.out / "boxes.pt", model=a.model, revision=MODELS[a.model]["revision"], headroom=a.headroom,
        amax={f"{s}:{l}": v for (s, l), v in stats.amax.items()})
    rows, massive = [], []
    for k in stats.keys:
        for ci in range(CLASSES):
            n = int(stats.n[ci])
            if not n:
                continue
            rms_el = (stats.ss[k][ci] / n).sqrt().float()
            site_rms = float((stats.ss[k][ci].sum() / (n * rms_el.numel())).sqrt())
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
                ids = [seq - Stats.SWEEP] if seq >= Stats.SWEEP else stats.tokens.get(seq, [])
                massive.append({"site": k[0], "layer": k[1], "class": ci, "coord": j, "value": v,
                                "over_median_element_rms": v / max(med, 1e-30), "over_site_rms": v / max(site_rms, 1e-30),
                                "box": float(hi[k][ci, j]), "seq": seq, "pos": pos,
                                "token": tok.decode([ids[pos]]) if 0 <= pos < len(ids) else None})
    return {"part": "calibrate", "texts": len(stats.tokens), "first_tokens": a.first_tokens, "new": a.new,
            "cap": a.cap, "window": stats.window, "special": len(stats.special),
            "hosts": None if stats.hosts is None else len(stats.hosts), "run": stats.run, "headroom": a.headroom,
            "positions": stats.n.tolist(), "sites": rows, "massive": massive}


def part_complete(lm: LM, tok, a) -> dict:
    boxes = Boxes.load(a.boxes)
    per_text = []
    t0 = time.time()
    seen: dict[tuple[str, int], int] = {}
    texts = traffic("heldout", a.mixed, a.long, a.cap) + (run_texts("heldout") if a.runs else [])
    for i, (name, text) in shard(texts, a.shard):
        lm.boxes = None
        plain = continued(lm, tok, text, a.new, a.cap)
        lm.boxes = boxes
        clamped = continued(lm, tok, text, a.new, a.cap)
        boxes.cut, boxes.worst = {}, {}
        lm.forward(torch.tensor([clamped]), Cache())
        n_prompt = min(len(tok.encode(text).ids), a.cap)
        diff = [j for j in range(n_prompt, len(plain)) if plain[j] != clamped[j]]
        per_text.append({"text": i, "name": name, "tokens": len(clamped), "served_differ": len(diff),
                         "first_differ": diff[0] - n_prompt if diff else None,
                         "cut": {f"{s}:{l}": n for (s, l), n in boxes.cut.items()},
                         "worst": {f"{s}:{l}": w for (s, l), w in boxes.worst.items()}})
        for k in lm.sites():
            seen[k] = seen.get(k, 0) + len(clamped) * lm.width(k[0])
        print(f"complete {i} {name}: {len(clamped)} tokens, {len(diff)} served tokens differ, cuts "
              f"{sum(boxes.cut.values())}, worst {max(boxes.worst.values(), default=0):.2f}x, {time.time() - t0:.0f} s",
              file=sys.stderr, flush=True)
    first = None
    if a.first_tokens and a.shard.split("/")[0] == "0":
        boxes.cut, boxes.worst = {}, {}
        lm.boxes = boxes
        sweep_first(lm, None, a.first_tokens, a.seed + 1, a.batch)
        first = {"tokens": a.first_tokens, "cut": {f"{s}:{l}": n for (s, l), n in boxes.cut.items()},
                 "worst": {f"{s}:{l}": w for (s, l), w in boxes.worst.items()}}
    cut_total = sum(sum(t["cut"].values()) for t in per_text)
    return {"part": "complete", "shard": a.shard, "texts": len(per_text), "new": a.new, "cap": a.cap,
            "window": boxes.window, "special": len(boxes.special), "values_seen": sum(seen.values()),
            "values_cut": cut_total, "texts_with_cuts": sum(1 for t in per_text if t["cut"]),
            "texts_served_differ": sum(1 for t in per_text if t["served_differ"]), "per_text": per_text,
            "first_tokens": first}


def context_of(tok, ids: list[int], pos: int, before: int = 6) -> str:
    return tok.decode(ids[max(0, pos - before):pos]) + "⟦" + tok.decode([ids[pos]]) + "⟧"


def part_explain(lm: LM, tok, a) -> dict:
    """Where honest traffic leaves the boxes. Unclamped, over --texts, every value outside its box (a group site: a group
    whose scale exceeds its bound), with its position's class and token and the tokens before it; each text's largest
    value against its box in each class (a box 2^h times wider cuts nothing in a text whose peak is under 2^h); and each
    text's sink-like positions, whose largest residual-stream element at --sink-layer is over --sink-ratio times the
    text's median."""
    boxes = Boxes.load(a.boxes)
    L = lm.spec.layers
    sink_layer = a.sink_layer if a.sink_layer >= 0 else L + a.sink_layer
    per_text, by_token = [], {}
    t0 = time.time()
    texts = traffic(a.texts, a.mixed, a.long, a.cap) + (run_texts(a.texts) if a.runs else [])
    for i, (name, text) in shard(texts, a.shard):
        lm.boxes = None
        ids = continued(lm, tok, text, a.new, a.cap)
        taps = {k: None for k in lm.sites()}
        lm.forward(torch.tensor([ids]), Cache(), taps=taps)
        cls = boxes.classes(0, len(ids), torch.tensor([ids]))[0][0]
        events = []
        peak_ratio = torch.zeros(len(ids))
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
            peak_ratio = torch.maximum(peak_ratio, ratio.amax(-1))
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
                  "over_median": float(peak[t]) / max(med, 1e-30), "class": int(cls[t])}
                 for t in range(len(ids)) if float(peak[t]) > a.sink_ratio * med]
        top = [{"ratio": e[0], "site": e[1], "layer": e[2], "pos": e[3], "class": int(cls[e[3]]), "coord": e[4],
                "value": e[5], "context": context_of(tok, ids, e[3])} for e in events[:a.top]]
        roots = sorted(({"pos": t, "site": e[1], "layer": e[2], "ratio": e[0], "context": context_of(tok, ids, t)}
                        for t, e in first.items()), key=lambda r: (r["layer"], -r["ratio"]))
        per_text.append({"text": i, "name": name, "tokens": len(ids), "values_over": len(events),
                         "positions_over": len(first), "top": top, "first_over_by_position": roots[:a.top],
                         "sinks": sinks, "sink_class_end": int((cls == 0).long().cumprod(0).sum()),
                         "peak_ratio_by_class": [float(peak_ratio[cls == c].max()) if (cls == c).any() else None
                                                 for c in range(CLASSES)]})
        print(f"explain {i} {name}: {len(ids)} tokens, {len(events)} values over in {len(first)} positions; worst "
              + (f"{events[0][0]:.2f}x at {events[0][1]}:{events[0][2]} {context_of(tok, ids, events[0][3])!r}"
                 if events else "none")
              + f"; sinks {[(s['pos'], s['context'][-12:], round(s['over_median'])) for s in sinks]}, {time.time() - t0:.0f} s",
              file=sys.stderr, flush=True)
    return {"part": "explain", "texts": a.texts, "shard": a.shard, "sink_layer": sink_layer, "per_text": per_text,
            "by_token": dict(sorted(by_token.items(), key=lambda kv: -kv[1]["max_ratio"]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--part", required=True, choices=("hosts", "calibrate", "merge", "rebox", "complete", "explain"))
    ap.add_argument("--texts", default="heldout", choices=("heldout", "calibration"), help="explain: which traffic")
    ap.add_argument("--mixed", type=int, default=0,
                    help="texts of the pinned datasets (traffic.mixed) added to the hand-written set of the part's split")
    ap.add_argument("--long", type=int, default=0,
                    help="complete, explain: only this many held-out Pile documents long enough for --cap tokens")
    ap.add_argument("--runs", action="store_true",
                    help="complete, explain: add the split's run texts (traffic.run_texts), runs of one token and after")
    ap.add_argument("--shard", default="0/1", help="i/n: this run's share of the texts (merge joins calibrate's)")
    ap.add_argument("--stats", type=Path, nargs="*", default=[], help="merge: the shards' stats.pt")
    ap.add_argument("--window", type=int, default=1,
                    help="calibrate: the sink class's positions from the first sink host on, past any run of it")
    ap.add_argument("--hosts", type=Path, help="calibrate: the hosts part's hosts.pt, where each sequence's classes start")
    ap.add_argument("--run", type=int,
                    help="calibrate: with --hosts, every position past the sink class's from the RUN-th copy of a host "
                         "token in a row to the end of the sequence is the run class, calibrated with traffic.run_texts")
    ap.add_argument("--special", action=argparse.BooleanOptionalAction, default=True,
                    help="calibrate: give the tokenizer's special tokens their own class")
    ap.add_argument("--headroom", type=int, nargs="+", default=[1],
                    help="calibrate/merge/rebox: powers of two of headroom on each box, one value or one a class")
    ap.add_argument("--sink-layer", type=int, default=3, help="hosts, explain: the residual stream that marks sinks")
    ap.add_argument("--sink-ratio", type=float, default=16.0,
                    help="hosts, explain: a sink-like position's residual peak over the honest median peak")
    ap.add_argument("--top", type=int, default=12, help="explain: the largest overshoots and positions listed per text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, help="complete, explain: the calibrated boxes.pt; rebox: the one to redo")
    ap.add_argument("--new", type=int, default=64, help="greedy continuation tokens a text adds to the traffic")
    ap.add_argument("--cap", type=int, default=768, help="a text's tokens, at most")
    ap.add_argument("--first-tokens", type=int, default=2048,
                    help="tokens swept at position 0 (calibrate: -1 for the whole vocabulary)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if len(a.headroom) not in (1, CLASSES):
        ap.error(f"--headroom takes one value or {CLASSES}, one a position class")
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    tok = lm.tokenizer()
    t = time.time()
    parts = {"hosts": part_hosts, "calibrate": part_calibrate, "merge": part_merge, "rebox": part_rebox,
             "complete": part_complete, "explain": part_explain}
    res = parts[a.part](lm, tok, a)
    res.update({"model": a.model, "revision": MODELS[a.model]["revision"],
                "args": {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                         for k, v in vars(a).items()},
                "seconds": time.time() - t})
    (a.out / f"clamp-{a.part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("sites", "massive", "per_text", "non_hosts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
