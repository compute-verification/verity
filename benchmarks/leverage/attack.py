"""Experiment 5: a tokens-only attack with chosen in-range interior faults, in exact weight bits a fault.

The decoder knows every weight but one, w = W_down[l][i, j] (the down projection of an interior layer l, as its bf16
code), and its prior is the empirical distribution of the other codes of that matrix. It sees served greedy tokens
only. Its faults are in range: hidden unit j of layer l (`mlph`), at a prompt's last position, set to a value inside
that unit's box (clamp.py's boxes.pt, whose clamps every forward applies), a 16-bit fault in an interior layer.

First honest use: the decoder reads each prompt's honest greedy continuation (--obs tokens), which narrows the prior
where an honest output already depends on w. Then each round:
  - plan: for each prompt and each value on a grid across the box, the outcome (the first --obs served tokens) of
    `--reps` posterior quantiles of w, simulated exactly (`lm.Edit` sets the weight; the clamps are public); pick the
    fault whose outcome splits the posterior's mass most evenly (the largest outcome entropy), then refine its value;
  - fault: the server serves the faulted request, and the decoder keeps the candidates whose simulated outcome is the
    one served.
It stops when one code remains (w read exactly), when no fault on the grid splits the posterior, or after --max-faults.
Bits are the posterior's entropy drop, so a fault's bits are what its outcome told the decoder beyond honest use.

The decoder here is generous: it knows the rest of the model exactly and simulates the serving stack bit for bit, so
the measured bits a fault bound from above what an attack of this form gets without that knowledge, and from below
what chosen in-range faults can give.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/attack.py --model qwen2.5-0.5b \\
      --boxes "$OUT/clamp/boxes.pt" --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import LM, MODELS, Boxes, Cache, Edit, Fault, position_class, served_logprobs
from prompts import PROMPTS


class Prompt:
    """One prompt's honest prefix: its ids, a batch-1 cache of every layer and the residual stream into each layer."""

    def __init__(self, lm: LM, tok, text: str, l: int):
        self.ids = torch.tensor([tok.encode(text).ids])
        self.t0 = self.ids.shape[1] - 1
        self.cache = Cache()
        key = ("resid", l - 1) if l > 0 else ("emb", -1)
        taps = {key: None, ("mlph", l): None}
        lm.forward(self.ids, self.cache, taps=taps)
        self.x_in = taps[key]
        self.a_j = taps[("mlph", l)][0, self.t0]


def outcomes(lm: LM, p: Prompt, l: int, i: int, j: int, cands: torch.Tensor, w_true: float, v: float | None,
             n_obs: int, batch: int) -> torch.Tensor:
    """(len(cands), n_obs): the first n_obs greedy tokens served for p with W_down[l][i, j] = cands[k], and, unless v
    is None, the fault mlph[l][j] := v at p's last position."""
    out = []
    for s in range(0, len(cands), batch):
        cs = cands[s:s + batch]
        k = len(cs)
        edit = Edit(l, i, j, (cs - w_true).float())
        ck = Cache({m: p.cache.kv[m] for m in range(l)})
        fault = Fault.of(p.t0, [("mlph", l, j, v)] * k) if v is not None else None
        z = lm.forward(None, ck, fault=fault, edit=edit, x_in=p.x_in.expand(k, -1, -1), first_layer=l, pos0=0,
                       last_only=True)[:, -1]
        toks = [served_logprobs(z, "bf16").argmax(-1)]
        for _ in range(1, n_obs):
            z = lm.forward(toks[-1][:, None], ck, edit=edit)[:, -1]
            toks.append(served_logprobs(z, "bf16").argmax(-1))
        out.append(torch.stack(toks, 1))
    return torch.cat(out)


def split(out: torch.Tensor, mass: torch.Tensor) -> tuple[float, torch.Tensor]:
    """The entropy of the outcome under the candidates' masses, and each candidate's outcome class."""
    _, cls = torch.unique(out, dim=0, return_inverse=True)
    pm = torch.zeros(int(cls.max()) + 1, dtype=torch.float64).index_add_(0, cls, mass.double())
    pm = pm[pm > 0] / pm.sum()
    return float(-(pm * pm.log2()).sum()), cls


def entropy(mass: torch.Tensor) -> float:
    q = mass.double() / mass.sum()
    q = q[q > 0]
    return float(-(q * q.log2()).sum())


def quantile_reps(vals: torch.Tensor, mass: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n candidates at evenly spaced quantiles of the posterior (vals sorted), each carrying its share of the mass."""
    if len(vals) <= n:
        return vals, mass
    cdf = mass.double().cumsum(0) / mass.sum()
    idx = torch.searchsorted(cdf, (torch.arange(n, dtype=torch.float64) + 0.5) / n).clamp(max=len(vals) - 1)
    idx = idx.unique()
    return vals[idx], torch.ones(len(idx), dtype=torch.float64)


def attack(lm: LM, tok, l: int, i: int, j: int, a, prompts: list[Prompt]) -> dict:
    """Bits are surprisals: a round's is -log2 of the posterior mass the served outcome keeps, so a read ending on the
    true code has taken -log2 of its prior probability in all, H(prior) on average over targets."""
    W = lm.down_proj(l)
    w_true = float(W[i, j])
    vals, counts = torch.unique(W.to(torch.bfloat16).float(), return_counts=True)
    mass = counts.double() / counts.sum()
    if not bool((vals == w_true).any()):
        raise ValueError("the target's code is not in its matrix's codes")
    hi = lm.boxes.hi[("mlph", l)][int(position_class(torch.tensor(prompts[0].t0)))].float()
    box = float(hi[j])
    rec = {"layer": l, "i": i, "j": j, "w": w_true, "box": box, "candidates": len(vals), "prior_bits": entropy(mass),
           "target_bits": -math.log2(float(mass[vals == w_true])), "a_j": [float(p.a_j[j]) for p in prompts],
           "rounds": []}
    t_start = time.time()
    honest_bits, honest = 0.0, []
    for p in prompts:
        o = outcomes(lm, p, l, i, j, vals, w_true, None, a.obs, a.batch)
        truth = o[(vals == w_true).nonzero()[0, 0]]
        honest.append(truth)
        ok = (o == truth).all(1)
        honest_bits += -math.log2(float(mass[ok].sum() / mass.sum()))
        vals, mass = vals[ok], mass[ok]
    rec["honest_bits"] = honest_bits
    grid = torch.linspace(-box, box, a.grid)
    faults, fault_bits = 0, 0.0
    while len(vals) > 1 and faults < a.max_faults:
        rv, rm = quantile_reps(vals, mass, a.reps)
        best = (0.0, None, None)
        for pi, p in enumerate(prompts):
            for v in grid.tolist():
                h, _ = split(outcomes(lm, p, l, i, j, rv, w_true, v, a.obs, a.batch), rm)
                if h > best[0] + 1e-9:
                    best = (h, pi, v)
        if best[1] is None or best[0] < a.min_bits:
            rec["stopped"] = f"no fault on the grid splits off more than {a.min_bits} bits"
            break
        h, pi, v = best
        step = 2 * box / (a.grid - 1)
        for v2 in torch.linspace(v - step / 2, v + step / 2, a.refine).tolist():
            if abs(v2) <= box:
                h2, _ = split(outcomes(lm, prompts[pi], l, i, j, rv, w_true, v2, a.obs, a.batch), rm)
                if h2 > h + 1e-9:
                    h, v = h2, v2
        o = outcomes(lm, prompts[pi], l, i, j, vals, w_true, v, a.obs, a.batch)
        truth = o[(vals == w_true).nonzero()[0, 0]]
        ok = (o == truth).all(1)
        got = -math.log2(float(mass[ok].sum() / mass.sum()))
        vals, mass = vals[ok], mass[ok]
        faults += 1
        fault_bits += got
        rec["rounds"].append({"prompt": pi, "value": v, "value_over_box": v / box, "planned_bits": h, "bits": got,
                              "candidates": len(vals), "direct": bool((truth != honest[pi]).any()),
                              "seconds": time.time() - t_start})
        print(f"target l={l} i={i} j={j}: fault {faults} prompt {pi} v={v:.4g} ({v / box:+.3f} box) planned "
              f"{h:.2f} got {got:.2f} bits, {len(vals)} left, {time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
    rec.update({"faults": faults, "exact": len(vals) == 1, "found": bool(float(vals[0]) == w_true) if len(vals) == 1
                else bool((vals == w_true).any()), "fault_bits": fault_bits,
                "bits_per_fault": fault_bits / faults if faults else 0.0, "posterior_candidates": len(vals),
                "seconds": time.time() - t_start})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--boxes", type=Path, required=True, help="clamp.py's calibrated boxes.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[-12, -2], help="interior layers (negative from the end)")
    ap.add_argument("--targets", default="widest,random", help="per layer: the widest-box unit's largest entry, "
                    "and a random unit's random entry")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--obs", type=int, default=1, help="served tokens a request returns that the decoder reads")
    ap.add_argument("--grid", type=int, default=13, help="fault values across the box, planned per round")
    ap.add_argument("--refine", type=int, default=9)
    ap.add_argument("--reps", type=int, default=64, help="posterior quantiles planned per round")
    ap.add_argument("--min-bits", type=float, default=0.05, help="stop when no planned fault splits off more")
    ap.add_argument("--max-faults", type=int, default=24)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    lm.boxes = Boxes.load(a.boxes)
    amax = torch.load(a.boxes)["amax"]
    tok = lm.tokenizer()
    gen = torch.Generator().manual_seed(a.seed)
    L = lm.spec.layers
    res = {"model": a.model, "revision": MODELS[a.model]["revision"], "targets": [],
           "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    for l in [x if x >= 0 else L + x for x in a.layers]:
        if not 0 <= l < L - 1:
            raise ValueError(f"layer {l} is not interior")
        prompts = [Prompt(lm, tok, t, l) for t in PROMPTS[:a.prompts]]
        hi = lm.boxes.hi[("mlph", l)][1].float()
        W = lm.down_proj(l)
        for kind in a.targets.split(","):
            if kind == "widest":
                by_amax = amax[f"mlph:{l}"][1].argsort(descending=True, stable=True)
                j = int(by_amax[hi[by_amax].argsort(descending=True, stable=True)][0])
                i = int(W[:, j].abs().argmax())
            else:
                j = int(torch.randint(0, W.shape[1], (1,), generator=gen))
                i = int(torch.randint(0, W.shape[0], (1,), generator=gen))
            r = attack(lm, tok, l, i, j, a, prompts)
            r["target"] = kind
            res["targets"].append(r)
            (a.out / "attack.json").write_text(json.dumps(res, indent=1))
    done = [t for t in res["targets"] if t["faults"]]
    res["bits_per_fault"] = (sum(t["fault_bits"] for t in done) / sum(t["faults"] for t in done)) if done else 0.0
    (a.out / "attack.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=1))
    for t in res["targets"]:
        print(json.dumps({k: v for k, v in t.items() if k not in ("rounds", "a_j")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
