"""Experiment 5: a tokens-only attack with chosen in-range interior faults, in exact weight bits a fault.

The decoder knows every weight but one unknown U of the down projection W_down[l] of an interior layer l, and holds a
prior over U taken from the same matrix (--unknowns):
  entry   U = W_down[l][i, j], as a bf16 code; the prior is the empirical distribution of the matrix's bf16 codes;
  column  U = W_down[l][:, j], unit j's whole column; the prior is uniform over the matrix's columns, so an exact read
          is log2(ff) bits, and a fault's bits say how much of the column the served token is a function of.
It sees served greedy tokens only. Its faults are in range: hidden unit j of layer l (`mlph`), at a prompt's last
position (clamp.py's boxes.pt, whose clamps every forward applies), set to (--range)
  own     one FP8 code at the token's own group scale, at most 448 times it: what a block writing that one value can do,
          a 16-bit fault;
  box     any value inside the unit's box: beyond the own range it rewrites the group's scale, and with it every code
          of the group, so it is a whole-group writer's fault.

Prompts (--prompts): the first half from PROMPTS, the rest prefixes of held-out traffic (traffic.HELDOUT_TEXTS), one a
text, ending where the honest greedy token's margin over the runner-up is smallest, where a fault moves it most easily.

First honest use: the decoder reads each prompt's honest greedy continuation (--obs tokens), which narrows the prior
where an honest output already depends on U. Then each round:
  - plan: for each prompt and each value on a grid across the box, the outcome (the first --obs served tokens) of
    `--reps` representatives of the posterior (an entry's quantiles, a random sample of a column's candidates),
    simulated exactly (`lm.Edit` sets U; the clamps are public); pick the fault whose outcome splits their mass most
    evenly (the largest outcome entropy), then refine its value;
  - fault: the server serves the faulted request, and the decoder keeps the candidates whose simulated outcome is the
    one served.
It stops when one candidate remains (U read exactly), when no fault on the grid splits the posterior, or after
--max-faults. Bits are surprisals: a round's is -log2 of the posterior mass its served outcome keeps, so a read ending on
the true candidate has taken -log2 of its prior probability in all.

The decoder here is generous: it knows the rest of the model exactly and simulates the serving stack bit for bit, so
the measured bits a fault bound from above what an attack of this form gets without that knowledge, and from below
what chosen in-range faults can give a decoder with this prior.

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
from lm import E4M3_MAX, GROUP, LM, MODELS, Boxes, Cache, Edit, Fault, served_logprobs, ue8m0_scale
from prompts import PROMPTS
from traffic import HELDOUT_TEXTS, lead_run


class Prompt:
    """One prompt's honest prefix: its ids, a batch-1 cache of every layer and the residual stream into each layer,
    and its last position's class under lm.boxes."""

    def __init__(self, lm: LM, ids: torch.Tensor, l: int):
        self.ids = ids
        self.t0 = ids.shape[1] - 1
        self.cls = int(lm.boxes.classes(0, ids.shape[1], ids)[0][0, -1])
        self.cache = Cache()
        key = ("resid", l - 1) if l > 0 else ("emb", -1)
        taps = {key: None, ("mlph", l): None}
        lm.forward(ids, self.cache, taps=taps)
        self.x_in = taps[key]
        self.a_j = taps[("mlph", l)][0, self.t0]


def pick_prompts(lm: LM, tok, n: int, l: int, cap: int = 48, lead: str = "") -> list[Prompt]:
    """The first (n + 1) // 2 of PROMPTS, then held-out prefixes at their smallest honest margin, smallest first, each
    after `lead`, which the prefixes are cut past. The prefixes are short because every candidate recomputes every
    position from layer l up."""
    k = len(tok.encode(lead).ids) if lead else 0
    out = [Prompt(lm, torch.tensor([tok.encode(lead + t).ids]), l) for t in PROMPTS[:(n + 1) // 2]]
    cands = []
    for text in HELDOUT_TEXTS:
        ids = torch.tensor([tok.encode(lead + text).ids[:k + cap]])
        top = served_logprobs(lm.forward(ids, Cache())[0], "bf16").topk(2, -1).values
        margin = top[:, 0] - top[:, 1]
        margin[:k + 8] = math.inf
        p = int(margin.argmin())
        cands.append((float(margin[p]), ids[:, :p + 1]))
    cands.sort(key=lambda c: c[0])
    return out + [Prompt(lm, ids, l) for _, ids in cands[:n - len(out)]]


def outcomes(lm: LM, p: Prompt, l: int, j: int, edit_of, cands: torch.Tensor, v: float | None, n_obs: int,
             batch: int) -> torch.Tensor:
    """(len(cands), n_obs): the first n_obs greedy tokens served for p with the unknown set to each candidate
    (`edit_of`), and, unless v is None, the fault mlph[l][j] := v at p's last position."""
    out = []
    for s in range(0, len(cands), batch):
        cs = cands[s:s + batch]
        k = len(cs)
        edit = edit_of(cs)
        ck = Cache({m: p.cache.kv[m] for m in range(l)})
        fault = Fault.of(p.t0, [("mlph", l, j, v)] * k) if v is not None else None
        z = lm.forward(p.ids.expand(k, -1), ck, fault=fault, edit=edit, x_in=p.x_in.expand(k, -1, -1), first_layer=l,
                       pos0=0, last_only=True)[:, -1]
        toks = [served_logprobs(z, "bf16").argmax(-1)]
        for _ in range(1, n_obs):
            z = lm.forward(toks[-1][:, None], ck, edit=edit)[:, -1]
            toks.append(served_logprobs(z, "bf16").argmax(-1))
        out.append(torch.stack(toks, 1))
    return torch.cat(out)


def own_scale(p: Prompt, j: int) -> float:
    """Unit j's group scale at p's last position: one FP8 code there reaches at most 448 times it."""
    return float(ue8m0_scale(p.a_j[j // GROUP * GROUP:(j // GROUP + 1) * GROUP].abs().max()))


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


def sample_reps(vals: torch.Tensor, mass: torch.Tensor, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n candidates drawn without replacement in proportion to their mass, each then carrying an equal share."""
    if len(vals) <= n:
        return vals, mass
    idx = torch.multinomial(mass.double(), n, replacement=False, generator=gen)
    return vals[idx], torch.ones(n, dtype=torch.float64)


def attack(lm: LM, l: int, i: int, j: int, unknown: str, a, prompts: list[Prompt], pc: int,
           gen: torch.Generator) -> dict:
    W = lm.down_proj(l)
    if unknown == "entry":
        stored, truth = float(W[i, j]), float(W[i, j].to(torch.bfloat16))
        vals, counts = torch.unique(W.to(torch.bfloat16).float(), return_counts=True)
        mass = counts.double() / counts.sum()

        def edit_of(cs):
            return Edit(l, i, j, (cs - stored).float())

        def reps_of(vs, ms):
            return quantile_reps(vs, ms, a.reps)
    else:
        truth = j
        vals = torch.arange(W.shape[1])
        mass = torch.ones(len(vals), dtype=torch.float64) / len(vals)
        col = W[:, j].float()

        def edit_of(cs):
            return Edit(l, None, j, W[:, cs].T.float() - col)

        def reps_of(vs, ms):
            return sample_reps(vs, ms, a.reps, gen)
    box = float(lm.boxes.hi[("mlph", l)][pc][j])
    scales = [own_scale(p, j) for p in prompts]
    own = [min(E4M3_MAX * s, box) for s in scales]
    lims = own if a.range == "own" else [box] * len(prompts)

    def value(v: float, pi: int) -> float:
        """v as the fault writes it: under --range own, the nearest FP8 code at the prompt's group scale."""
        if a.range == "box":
            return v
        return float(torch.tensor(v / scales[pi]).to(torch.float8_e4m3fn).float()) * scales[pi]

    rec = {"layer": l, "unknown": unknown, "i": i if unknown == "entry" else None, "j": j,
           "w": truth if unknown == "entry" else None, "box": box, "range": a.range, "own": own,
           "candidates": len(vals), "prior_bits": entropy(mass), "target_bits": -math.log2(float(mass[vals == truth])),
           "a_j": [float(p.a_j[j]) for p in prompts], "rounds": []}
    t_start = time.time()
    honest_bits, honest = 0.0, []
    for p in prompts:
        o = outcomes(lm, p, l, j, edit_of, vals, None, a.obs, a.batch)
        served = o[(vals == truth).nonzero()[0, 0]]
        honest.append(served)
        ok = (o == served).all(1)
        honest_bits += -math.log2(float(mass[ok].sum() / mass.sum()))
        vals, mass = vals[ok], mass[ok]
    rec["honest_bits"] = honest_bits
    faults, fault_bits = 0, 0.0
    while len(vals) > 1 and faults < a.max_faults:
        rv, rm = reps_of(vals, mass)
        best = (0.0, None, None)
        for pi, p in enumerate(prompts):
            for v in sorted({value(x, pi) for x in torch.linspace(-lims[pi], lims[pi], a.grid).tolist()}):
                h, _ = split(outcomes(lm, p, l, j, edit_of, rv, v, a.obs, a.batch), rm)
                if h > best[0] + 1e-9:
                    best = (h, pi, v)
        if best[1] is None or best[0] < a.min_bits:
            rec["stopped"] = f"no fault on the grid splits off more than {a.min_bits} bits"
            break
        h, pi, v = best
        step = 2 * lims[pi] / (a.grid - 1)
        for v2 in torch.linspace(v - step / 2, v + step / 2, a.refine).tolist():
            if abs(v2) <= lims[pi]:
                v2 = value(v2, pi)
                h2, _ = split(outcomes(lm, prompts[pi], l, j, edit_of, rv, v2, a.obs, a.batch), rm)
                if h2 > h + 1e-9:
                    h, v = h2, v2
        o = outcomes(lm, prompts[pi], l, j, edit_of, vals, v, a.obs, a.batch)
        served = o[(vals == truth).nonzero()[0, 0]]
        ok = (o == served).all(1)
        got = -math.log2(float(mass[ok].sum() / mass.sum()))
        classes = int(torch.unique(o, dim=0).shape[0])
        vals, mass = vals[ok], mass[ok]
        faults += 1
        fault_bits += got
        rec["rounds"].append({"prompt": pi, "value": v, "value_over_box": v / box, "within_own": abs(v) <= own[pi],
                              "planned_bits": h, "bits": got, "outcome_classes": classes, "candidates": len(vals),
                              "direct": bool((served != honest[pi]).any()), "seconds": time.time() - t_start})
        print(f"target {unknown} l={l} i={i} j={j}: fault {faults} prompt {pi} v={v:.4g} ({v / box:+.3f} box) "
              f"planned {h:.2f} got {got:.2f} bits of {classes} outcomes, {len(vals)} left, "
              f"{time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
    rec.update({"faults": faults, "exact": len(vals) == 1, "found": bool((vals == truth).any()),
                "fault_bits": fault_bits, "bits_per_fault": fault_bits / faults if faults else 0.0,
                "max_fault_bits": max((r["bits"] for r in rec["rounds"]), default=0.0),
                "posterior_candidates": len(vals), "seconds": time.time() - t_start})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--boxes", type=Path, required=True, help="clamp.py's calibrated boxes.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[-12, -2], help="interior layers (negative from the end)")
    ap.add_argument("--unknowns", default="entry,column", help="what the decoder doesn't know: one weight, or the "
                    "faulted unit's whole column")
    ap.add_argument("--targets", default="widest,random", help="per layer: the widest-box unit (and for an entry its "
                    "largest), and a random unit (and a random entry)")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--obs", type=int, default=1, help="served tokens a request returns that the decoder reads")
    ap.add_argument("--lead-run", nargs=2, metavar=("UNIT", "N"),
                    help="every prompt follows a run of N copies of UNIT (traffic.lead_run), so under boxes with a run "
                         "class its faults sit in it")
    ap.add_argument("--range", default="own", choices=("own", "box"),
                    help="fault values: one FP8 code at the token's own group scale (a 16-bit fault), or anywhere in "
                         "the unit's box (beyond the own range, a whole-group writer's)")
    ap.add_argument("--grid", type=int, default=13, help="fault values across the box, planned per round")
    ap.add_argument("--refine", type=int, default=9)
    ap.add_argument("--reps", type=int, default=64, help="posterior representatives planned per round")
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
        lead = lead_run(a.lead_run[0], int(a.lead_run[1])) if a.lead_run else ""
        prompts = pick_prompts(lm, tok, a.prompts, l, lead=lead)
        classes = sorted({p.cls for p in prompts})
        if len(classes) != 1:
            raise ValueError(f"layer {l}: the prompts' last positions are in classes {classes}, and one box needs one "
                             "class: change --prompts or --lead-run")
        pc = classes[0]
        hi = lm.boxes.hi[("mlph", l)][pc].float()
        W = lm.down_proj(l)
        for kind in a.targets.split(","):
            if kind == "widest":
                by_amax = amax[f"mlph:{l}"][pc].argsort(descending=True, stable=True)
                j = int(by_amax[hi[by_amax].argsort(descending=True, stable=True)][0])
                i = int(W[:, j].abs().argmax())
            else:
                j = int(torch.randint(0, W.shape[1], (1,), generator=gen))
                i = int(torch.randint(0, W.shape[0], (1,), generator=gen))
            for unknown in a.unknowns.split(","):
                r = attack(lm, l, i, j, unknown, a, prompts, pc, gen)
                r["target"], r["fault_class"] = kind, pc
                res["targets"].append(r)
                (a.out / "attack.json").write_text(json.dumps(res, indent=1))
    for unknown in a.unknowns.split(","):
        done = [t for t in res["targets"] if t["faults"] and t["unknown"] == unknown]
        res[f"bits_per_fault_{unknown}"] = (sum(t["fault_bits"] for t in done) / sum(t["faults"] for t in done)
                                            if done else 0.0)
    (a.out / "attack.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "targets"}, indent=1))
    for t in res["targets"]:
        print(json.dumps({k: v for k, v in t.items() if k not in ("rounds", "a_j")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
