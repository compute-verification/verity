"""HT-DEFENCE's tables, from rotated.py's records: for design U as built and for each defence against the exact split,
the least over the leaf over every credited matrix of Llama-3.1-8B and -70B at each chip@price, the matrix that sets
it, and the share of the model's block-linear MACs credited.

    python benchmarks/hardness_defence/tables.py [--runs ~/.research/runs] [--matrices]

The sources are the runs in SOURCES, fetched first with `research fetch RUN --all --on NODE`. Every record is re-scored
with rotated.py's current cost formulas (`rotated.score`), except K-chunked ones, which carry their own. Where several
records price one matrix the same way (more ranks, other orders), the least per chip@price is kept. A matrix credited at
a margin is credited at its least over the leaf / 1.05 of its MACs, the least over chips and prices.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
import rotated  # noqa: E402

LINE = 1.05
PRICES = ("0.84", "1.48", "1.78")
KEYS = [f"{chip}@{p}" for chip in ("sm120", "h200") for p in PRICES]
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
QKV = KINDS[:3]
# (hidden, kv width, intermediate, layers)
MODELS = {"8b": (4096, 1024, 14336, 32), "70b": (8192, 1024, 28672, 80)}

# run, its output directory under out/ (a glob), model, and what an older record doesn't say itself
SOURCES = [
    # HT-PUBLIC v2 (note:compute-research/20261101T1122Z-report-hardness-public-v2)
    ("r20261101-072723-fc78", "llama3.1-8b-rot-0.02", "8b", {"target": 0.02}),
    ("r20261101-075205-a6a1", "llama3.1-8b-qk-rot-0.02", "8b", {"target": 0.02}),
    ("r20261101-074723-0271", "8b-t0.02", "8b", {"target": 0.02}),
    ("r20261101-074723-0271", "8b-t0.05", "8b", {"target": 0.05}),
    ("r20261101-074723-0271", "8b-t0.10", "8b", {"target": 0.10}),
    ("r20261101-074723-0271", "8b-t0.20", "8b", {"target": 0.20}),
    ("r20261101-074723-0271", "8b-defl32", "8b", {"target": 0.02, "deflate": 32}),
    ("r20261101-074723-0271", "8b-defl64", "8b", {"target": 0.02, "deflate": 64}),
    ("r20261101-074723-0271", "8b-defl128", "8b", {"target": 0.02, "deflate": 128}),
    ("r20261101-080825-b2c1", "8b-l0-t0.06", "8b", {"target": 0.06}),
    ("r20261101-080825-b2c1", "8b-l0-t0.08", "8b", {"target": 0.08}),
    ("r20261101-080825-b2c1", "8b-l0-t0.12", "8b", {"target": 0.12}),
    ("r20261101-080825-b2c1", "8b-l0-t0.15", "8b", {"target": 0.15}),
    ("r20261101-080825-b2c1", "70b-t0.02", "70b", {"target": 0.02}),
    ("r20261101-082608-3981", "70b-kv-rot-0.02", "70b", {"target": 0.02}),
    ("r20261101-082610-5b3a", "70b-q-rot-0.02", "70b", {"target": 0.02}),
    ("r20261101-100522-128b", "70b-qs-rot-0.02", "70b", {"target": 0.02}),
    ("r20261101-083947-3763", "70b-kv-t0.05", "70b", {"target": 0.05}),
    ("r20261101-083947-3763", "70b-kv-t0.10", "70b", {"target": 0.10}),
    ("r20261101-083947-3763", "70b-kv-t0.20", "70b", {"target": 0.20}),
    ("r20261101-092230-3105", "70b-q0-t0.05", "70b", {"target": 0.05}),
    ("r20261101-092230-3105", "70b-q0-t0.10", "70b", {"target": 0.10}),
    ("r20261101-092230-3105", "70b-q0-t0.20", "70b", {"target": 0.20}),
    ("r20261101-094903-7967", "70b-q1-t0.05", "70b", {"target": 0.05}),
    ("r20261101-094903-7967", "70b-q1-t0.10", "70b", {"target": 0.10}),
    ("r20261101-094903-7967", "70b-q1-t0.20", "70b", {"target": 0.20}),
    # this round
    ("r20261104-083102-e258", "8b-vo-rest", "8b", {}),
    ("r20261104-083102-e258", "8b-mlp-rest", "8b", {}),
    ("r20261104-083102-e258", "8b-rot-*", "8b", {}),
    ("r20261104-083102-e258", "8b-noise*", "8b", {}),
    ("r20261104-083102-e258", "70b-rot-*", "70b", {}),
    ("r20261104-083102-e258", "70b-noise*", "70b", {}),
    ("r20261104-083102-64dc", "70b-q-rest", "70b", {}),
    ("r20261104-083102-64dc", "70b-q-clip/t*", "70b", {}),
    ("r20261104-083102-64dc", "70b-down-k8192", "70b", {}),
    ("r20261104-083215-0ae4", "70b-ogu", "70b", {}),
    ("r20261104-092045-ef7b", "70b-down-full", "70b", {}),
    ("r20261104-085532-5d0f", "70b-gu-split", "70b", {}),
    ("r20261104-083537-aba6", "70b-gu-clip20", "70b", {}),
    ("r20261104-083537-aba6", "70b-defl32-*", "70b", {"deflate": 32}),
    ("r20261104-083537-aba6", "70b-defl8-*", "70b", {"deflate": 8}),
    ("r20261104-092045-32ee", "8b-orders-*", "8b", {"orders": True}),
    ("r20261104-092045-32ee", "70b-orders-*", "70b", {"orders": True}),
    ("r20261104-093237-c694", "8b-*", "8b", {"orders": True}),
    ("r20261104-093237-c694", "70b-*", "70b", {"orders": True}),
    ("r20261104-093237-b5f7", "70b-*", "70b", {"orders": True}),
    ("r20261104-104721-53db", "70b-*", "70b", {"orders": True}),
]


def kind(tensor: str) -> str:
    return tensor.split(".")[-2]


def layer(tensor: str) -> int:
    return int(tensor.split(".")[2])


def tensor_name(L: int, k: str) -> str:
    return f"model.layers.{L}.{'self_attn' if k in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'}.{k}.weight"


def shape(model: str, k: str) -> tuple[int, int]:
    h, kv, inter, _ = MODELS[model]
    return {"q_proj": (h, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, h), "gate_proj": (inter, h),
            "up_proj": (inter, h), "down_proj": (h, inter)}[k]


def macs(model: str, k: str) -> int:
    n, K = shape(model, k)
    return n * K


def total_macs(model: str) -> int:
    return MODELS[model][3] * sum(macs(model, k) for k in KINDS)


class Book:
    """model -> tensor -> variant -> {chip@price: (over the leaf, route)}; a variant is (target, rot, deflate, live
    noise f, K chunks). split and orders record whether any record of that variant searched the split, and the
    attacker's row and column orders (`price.permute`)."""

    def __init__(self):
        self.v = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.split = set()
        self.orders = set()
        self.runs = collections.defaultdict(set)

    def add(self, model, tensor, var, scores, split, run, orders=False):
        cur = self.v[model][tensor].setdefault(var, {})
        for k, (ol, route) in scores.items():
            if k not in cur or ol < cur[k][0]:
                cur[k] = (ol, route)
        if split:
            self.split.add((model, tensor, var))
        if orders:
            self.orders.add((model, tensor, var))
        self.runs[(model, tensor, var)].add(run)

    def get(self, model, tensor, target=0.02, rot="hadamard", deflate=0, noise=0, chunks=1):
        return self.v[model][tensor].get((target, rot, deflate, noise, chunks))


def load(runs: Path) -> Book:
    book = Book()
    for run, pattern, model, tags in SOURCES:
        dirs = sorted((runs / run / "out").glob(pattern))
        for d in dirs:
            files = sorted(d.glob("matrices.jsonl")) + sorted(d.glob("t*/matrices.jsonl"))
            for f in files:
                for line in f.read_text().splitlines():
                    rec = json.loads(line)
                    target = rec.get("target") if rec.get("target") is not None else tags.get("target")
                    target = round(float(target), 4)
                    rot = rec.get("rot") or "hadamard"
                    deflate = tags.get("deflate", 0)
                    orders = tags.get("orders", False)
                    if rec.get("chunked"):
                        var = (target, rot, deflate, 0, rec["chunks"])
                        scores = {k: (v["over_leaf"], v["route"]) for k, v in rec["best"].items()}
                        book.add(model, rec["tensor"], var, scores, True, run, orders)
                        continue
                    split = bool(rec.get("isvd")) or "energy_top" in rec
                    scores = {}
                    for k in KEYS:
                        s = rotated.score(rec, k)
                        scores[k] = (s["over_leaf"], s["route"])
                    book.add(model, rec["tensor"], (target, rot, deflate, 0, 1), scores, split, run, orders)
                    if "noise" in rec:
                        ns = {k: (s["over_leaf"], s["route"]) for k in KEYS for s in [rotated.noise_score(rec, k)]}
                        book.add(model, rec["tensor"], (target, rot, deflate, rec["noise"]["f"], 1), ns, True, run,
                                 orders)
    return book


def least_of(scores: dict) -> float:
    return min(v[0] for v in scores.values())


def all_tensors(model: str):
    return [tensor_name(L, k) for L in range(MODELS[model][3]) for k in KINDS]


# A defence chooses, per matrix, the variant served and its credit (a fraction of its MACs): None is uncredited.
def base(book, model, t):
    return book.get(model, t), 1.0


def clip_choice(book, model, t, cap=1.0):
    """The least clip target whose variant clears the line at every chip and price; else the one that comes closest."""
    vs = {var: s for var, s in book.v[model][t].items() if var[1:] == ("hadamard", 0, 0, 1) and var[0] <= cap}
    if not vs:
        return None, False
    ok = sorted(var for var, s in vs.items() if least_of(s) >= LINE)
    var = ok[0] if ok else max(vs, key=lambda v: least_of(vs[v]))
    return var, bool(ok)


def clip(book, model, t):
    var, _ = clip_choice(book, model, t)
    return (book.v[model][t][var] if var else None), 1.0


def margin(book, model, t):
    s = book.get(model, t)
    if s is None:
        return None, 1.0
    return s, min(1.0, least_of(s) / LINE)


def qkv_out(book, model, t):
    return (None, 0.0) if kind(t) in QKV else base(book, model, t)


def broken_out(book, model, t):
    s = book.get(model, t)
    return (None, 0.0) if s is not None and least_of(s) < LINE else (s, 1.0)


def operating(down="margin"):
    """Clip where a clip clears the line, uncredited where none does; 70B's down_proj at its whole K credited at a
    margin, or in K chunks."""
    def choose(book, model, t):
        if model == "70b" and kind(t) == "down_proj":
            if down == "chunks":
                s = book.get(model, t, chunks=4)
                return (s, 1.0) if s else (None, 1.0)
            return margin(book, model, t)
        var, ok = clip_choice(book, model, t)
        if var is None:
            return None, 1.0
        return (book.v[model][t][var], 1.0) if ok else (None, 0.0)
    return choose


def rotation(rot):
    def choose(book, model, t):
        s = book.get(model, t, rot=rot)
        return (s, 1.0) if s else base(book, model, t)
    return choose


def noise(f):
    def choose(book, model, t):
        s = book.get(model, t, noise=f)
        return (s, 1.0) if s else base(book, model, t)
    return choose


DEFENCES = {
    "design U (2%, no defence)": base,
    "1: second Hadamard (where priced)": rotation("hadamard2"),
    "1: Haar rotation (where priced)": rotation("haar"),
    "1: output-side Hadamard too (where priced)": rotation("both"),
    "3: live weight noise f=1 (where priced)": noise(1),
    "3: live weight noise f=4 (where priced)": noise(4),
    "2: least clip that clears, up to 20%": clip,
    "4: broken matrices at a margin": margin,
    "4: broken matrices uncredited": broken_out,
    "4: q, k, v uncredited": qkv_out,
    "operating point (down_proj at a margin)": operating("margin"),
    "operating point (down_proj in K chunks)": operating("chunks"),
}


def evaluate(book, model, choose):
    """Per chip@price the least cost over credit of a credited matrix (its over the leaf / the share of its MACs
    credited), the matrix setting it, the credited share of the model's MACs, and the credited matrices unpriced."""
    worst = {k: (math.inf, None, None) for k in KEYS}
    credited = 0.0
    unpriced = []
    for t in all_tensors(model):
        s, credit = choose(book, model, t)
        if credit == 0:
            continue
        credited += macs(model, kind(t)) * credit
        if s is None:
            unpriced.append(t)
            continue
        for k in KEYS:
            ol = s[k][0] / credit
            if ol < worst[k][0]:
                worst[k] = (ol, t, s[k][1])
    return worst, credited / total_macs(model), unpriced


def short(t: str) -> str:
    return f"L{layer(t)} {kind(t).removesuffix('_proj')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--matrices", action="store_true", help="also every matrix the split or a defence moves")
    args = ap.parse_args()
    book = load(args.runs)
    print("| defence | model | sm_120 0.84 / 1.48 / 1.78 | H200 0.84 / 1.48 / 1.78 | least set by | MACs credited | unpriced |")
    print("|---|---|---|---|---|---|---|")
    for name, choose in DEFENCES.items():
        for model in MODELS:
            worst, share, unpriced = evaluate(book, model, choose)
            cells = [" / ".join(f"{worst[f'{c}@{p}'][0]:.3f}" for p in PRICES) for c in ("sm120", "h200")]
            k0 = min(KEYS, key=lambda k: worst[k][0])
            setter = f"{short(worst[k0][1])} ({worst[k0][2]}, {k0})" if worst[k0][1] else "-"
            print(f"| {name} | {model} | {cells[0]} | {cells[1]} | {setter} | {100 * share:.2f}% | {len(unpriced)} |")
    if args.matrices:
        print()
        print("| model | matrix | 2% least (sm_120 / H200 at 0.84) | route | clip chosen | its least | split, orders |")
        print("|---|---|---|---|---|---|---|")
        for model in MODELS:
            for t in all_tensors(model):
                s = book.get(model, t)
                var, ok = clip_choice(book, model, t)
                if s is None or least_of(s) >= LINE + 0.04:
                    continue
                c = book.v[model][t][var]
                k0 = min(KEYS, key=lambda k: s[k][0])
                searched = [w for w, seen in (("split", book.split), ("orders", book.orders)) if (model, t, var) in seen]
                print(f"| {model} | {short(t)} | {s['sm120@0.84'][0]:.3f} / {s['h200@0.84'][0]:.3f} | {s[k0][1]} | "
                      f"{var[0]:.0%}{'' if ok else ' (none clears)'} | {least_of(c):.3f} | "
                      f"{', '.join(searched) or 'neither'} |")


if __name__ == "__main__":
    main()
