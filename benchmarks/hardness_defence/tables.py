"""HT-DEFENCE's tables, from rotated.py's records: for design U as built and for each defence against the exact split,
the least over the leaf over every credited matrix of Llama-3.1-8B and -70B at each chip@price, the matrix that sets
it, and the share of the model's block-linear MACs credited.

    python benchmarks/hardness_defence/tables.py [--runs ~/.research/runs] [--matrices]

The sources are the runs in SOURCES, fetched first with `research fetch RUN --all --on NODE`. Every record is re-scored
with rotated.py's current cost formulas (`rotated.score`), except K-chunked ones, which carry their own. Where several
records price one matrix the same way (more ranks, other orders), the least per chip@price is kept. A matrix credited at
a margin is credited at its least over the leaf / 1.05 of its MACs, the least over chips and prices.

A rotation or live-noise variant is priced on a few matrices in the natural order only, beside design U's codes of the
same run: its scores here are design U's least (every order priced) plus its change against that same-run record
(estimated), so that it isn't credited with what the attacker's orders take off design U.
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
    ("r20261104-105910-69ec", "8b-*", "8b", {"orders": True}),
    ("r20261104-105910-69ec", "70b-*", "70b", {"orders": True}),
    ("r20261104-111402-81ab", "70b-*", "70b", {"orders": True}),
    ("r20261104-112755-5cbd", "70b-*", "70b", {"deflate": 32, "orders": True}),
    ("r20261104-114043-136c", "70b-*", "70b", {"orders": True}),
    ("r20261104-121750-fe3b", "70b-gu-rest", "70b", {"orders": True}),
    ("r20261104-121750-fa83", "70b-down-rest", "70b", {"orders": True}),
    ("r20261104-121751-43c7", "70b-o-rest", "70b", {"orders": True}),
    ("r20261104-122606-1140", "70b-*", "70b", {"orders": True}),
    ("r20261104-122626-1f88", "8b-*", "8b", {"orders": True}),
    ("r20261104-122626-2c76", "70b-*", "70b", {"orders": True}),
    ("r20261104-122626-cd66", "70b-*", "70b", {"orders": True}),
    ("r20261104-133126-ac09", "70b-*", "70b", {"orders": True}),
    ("r20261104-133157-6b84", "70b-*", "70b", {"orders": True}),
    ("r20261104-133157-6b84", "8b-*", "8b", {"orders": True}),
    ("r20261104-144714-7768", "70b-*", "70b", {"deflate": 32, "orders": True}),
    ("r20261104-163431-e8a3", "70b-*", "70b", {"orders": True}),
    ("r20261104-163633-f732", "70b-*", "70b", {"orders": True}),
    ("r20261104-163633-f732", "8b-*", "8b", {"orders": True}),
    ("r20261104-172736-7a11", "70b-*", "70b", {"orders": True}),
    ("r20261104-172736-7a11", "8b-*", "8b", {"orders": True}),
    # HT-GENEROUS: svd and rpca in every order on every credited q and k (generous.py)
    ("r20261105-193551-8b78", "70b-*-trial", "70b", {"orders": True}),
    ("r20261105-193551-8b78", "8b-*-trial", "8b", {"orders": True}),
    ("r20261105-194937-f99c", "70b-q-gen-lo", "70b", {"orders": True}),
    ("r20261105-194955-3c0d", "70b-q-gen-hi", "70b", {"orders": True}),
    ("r20261105-194501-d467", "70b-k-gen", "70b", {"orders": True}),
    ("r20261105-194501-d467", "8b-*-gen", "8b", {"orders": True}),
    ("r20261105-203857-5e10", "70b-k-refine", "70b", {"orders": True}),
    ("r20261105-203857-5e10", "8b-*-refine", "8b", {"orders": True}),
    ("r20261105-211648-629d", "70b-q-r*", "70b", {"orders": True}),
    ("r20261105-215916-4b22", "70b-q-l53-r*", "70b", {"orders": True}),
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
    noise f, K chunks). split and orders record whether any record of that variant searched the split, and the split
    in the attacker's row and column orders too (`price.permute`). Per variant, ordered_ranks are the split's ranks
    priced in all orders and natural_ranks each rank's least split priced in the natural order."""

    def __init__(self):
        self.v = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.split = set()
        self.orders = set()
        self.split_least = {}
        self.ordered_ranks = collections.defaultdict(set)
        self.natural_ranks = collections.defaultdict(dict)
        self.runs = collections.defaultdict(set)

    def add(self, model, tensor, var, scores, split, run, orders=False, splits=None):
        cur = self.v[model][tensor].setdefault(var, {})
        for k, (ol, route) in scores.items():
            if k not in cur or ol < cur[k][0]:
                cur[k] = (ol, route)
        if split:
            self.split.add((model, tensor, var))
        if orders and split:
            self.orders.add((model, tensor, var))
        key = (model, tensor, var)
        splits = splits or {}
        self.split_least[key] = min(self.split_least.get(key, math.inf), min(splits.values(), default=math.inf))
        for r, sl in splits.items():
            if orders:
                self.ordered_ranks[key].add(r)
            else:
                self.natural_ranks[key][r] = min(self.natural_ranks[key].get(r, math.inf), sl)
        self.runs[key].add(run)

    def natural_only(self, key) -> float:
        """The least split over the leaf at the ranks of a variant priced in the natural order only."""
        return min((sl for r, sl in self.natural_ranks[key].items() if r not in self.ordered_ranks[key]),
                   default=math.inf)

    def get(self, model, tensor, target=0.02, rot="hadamard", deflate=0, noise=0, chunks=1):
        return self.v[model][tensor].get((target, rot, deflate, noise, chunks))


def splits(rec: dict) -> dict:
    """Per rank, the least over the leaf of a record's exact split alone, over chips and prices."""
    n, K = rec["shape"]
    return {int(r): min((rotated.lrx(int(r), n, K, v["d"]) + v["routes"][k]["cost"]) / rec["leaf"] for k in KEYS)
            for r, v in rec.get("isvd", {}).items()}


def chunk_splits(rec: dict) -> dict:
    """The same for a record in K chunks, each chunk on its own least split, under the one rank "chunks"."""
    n, kc, m = rec["shape"][0], rec["K_chunk"], rec["chunks"]

    def at(k):
        return sum(min((rotated.lrx(int(r), n, kc, v["d"]) + v["routes"][k]["cost"] for r, v in part["isvd"].items()),
                       default=math.inf) for part in rec["per_chunk"]) / m / rec["leaf"]
    return {"chunks": min(at(k) for k in KEYS)}


def load(runs: Path, sources=SOURCES, generous: bool = False) -> Book:
    """The book of sources' records; with generous, each scored at its least of the exact and generous routes."""
    book = Book()
    refs, variants = {}, []
    over, route = ("generous_over_leaf", "generous_route") if generous else ("over_leaf", "route")
    for run, pattern, model, tags in sources:
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
                        book.add(model, rec["tensor"], var, scores, True, run, orders, chunk_splits(rec))
                        continue
                    split = bool(rec.get("isvd")) or "energy_top" in rec
                    scores = {}
                    for k in KEYS:
                        s = rotated.score(rec, k)
                        scores[k] = (s[over], s[route])
                    var = (target, rot, deflate, 0, 1)
                    book.add(model, rec["tensor"], var, scores, split, run, orders, splits(rec))
                    if not orders and var == U:
                        refs[(run, model, rec["tensor"])] = scores
                    elif not orders and rot != U[1]:
                        variants.append((run, model, rec["tensor"], var, scores, None))
                    if "noise" in rec:
                        ns = {k: (s["over_leaf"], s["route"]) for k in KEYS for s in [rotated.noise_score(rec, k)]}
                        nvar = (target, rot, deflate, rec["noise"]["f"], 1)
                        book.add(model, rec["tensor"], nvar, ns, True, run, orders)
                        variants.append((run, model, rec["tensor"], nvar, ns, scores))
    for run, model, t, var, scores, ref in variants:
        ref, base = ref or refs.get((run, model, t)), book.get(model, t)
        if ref and base and var[0] == U[0]:
            book.v[model][t][var] = {k: (base[k][0] + scores[k][0] - ref[k][0], scores[k][1]) for k in KEYS}
    return book


def least_of(scores: dict) -> float:
    return min(v[0] for v in scores.values())


def all_tensors(model: str):
    return [tensor_name(L, k) for L in range(MODELS[model][3]) for k in KINDS]


# A defence chooses, per matrix, the variant served (a key of Book.v[model][tensor]; None: not priced) and its credit,
# the share of its MACs credited (0: uncredited).
U = (0.02, "hadamard", 0, 0, 1)


def present(book, model, t, var):
    return var if var in book.v[model][t] else None


def base(book, model, t):
    return present(book, model, t, U), 1.0


def clip_choice(book, model, t, cap=1.0):
    """The least clip target whose variant clears the line at every chip and price; else the one that comes closest."""
    vs = {var: s for var, s in book.v[model][t].items() if var[1:] == U[1:] and var[0] <= cap}
    if not vs:
        return None, False
    ok = sorted(var for var, s in vs.items() if least_of(s) >= LINE)
    var = ok[0] if ok else max(vs, key=lambda v: least_of(vs[v]))
    return var, bool(ok)


def clip(book, model, t):
    return clip_choice(book, model, t)[0], 1.0


def margin(book, model, t):
    var = present(book, model, t, U)
    return var, min(1.0, least_of(book.v[model][t][var]) / LINE) if var else 1.0


def qkv_margin(book, model, t):
    return margin(book, model, t) if kind(t) in QKV else base(book, model, t)


def qkv_out(book, model, t):
    return (None, 0.0) if kind(t) in QKV else base(book, model, t)


def broken_out(book, model, t):
    var = present(book, model, t, U)
    return (None, 0.0) if var and least_of(book.v[model][t][var]) < LINE else (var, 1.0)


def deflated(book, model, t):
    """The least deflation of design U's 2% codes that clears the line, if any was priced."""
    ok = sorted(var for var, s in book.v[model][t].items()
                if var[0] == U[0] and var[1] == U[1] and var[2] and var[3:] == U[3:] and least_of(s) >= LINE)
    return ok[0] if ok else None


# The four matrices whose design-U service carries 70B's quality loss (quality.py: served exact, they take WikiText-2
# from +11.2% to +4.3% over bf16), and that no clip tried brings over the line
EXACT = {"70b": {tensor_name(0, "q_proj"), tensor_name(0, "k_proj"), tensor_name(0, "v_proj"),
                 tensor_name(1, "v_proj")}}


def operating(cap=1.0, rest="out", down="margin", exact=False, deflate=False):
    """The least clip of at most cap that clears the line; where none does, with deflate the least deflation that
    clears, else `rest` (uncredited, or credited at a margin as design U); 70B's down_proj at its whole K at a margin,
    or in K chunks; with exact, EXACT uncredited (served exact)."""
    def choose(book, model, t):
        if exact and t in EXACT.get(model, ()):
            return None, 0.0
        if model == "70b" and kind(t) == "down_proj":
            if down == "chunks":
                return present(book, model, t, U[:4] + (4,)), 1.0
            return margin(book, model, t)
        var, ok = clip_choice(book, model, t, cap)
        if var is None or ok:
            return var, 1.0
        d = deflated(book, model, t) if deflate else None
        if d:
            return d, 1.0
        return margin(book, model, t) if rest == "margin" else (None, 0.0)
    return choose


def rotation(rot):
    def choose(book, model, t):
        var = present(book, model, t, (U[0], rot) + U[2:])
        return (var, 1.0) if var else base(book, model, t)
    return choose


def noise(f):
    def choose(book, model, t):
        var = present(book, model, t, U[:3] + (f, 1))
        return (var, 1.0) if var else base(book, model, t)
    return choose


DEFENCES = {
    "design U (2%, no defence)": base,
    "1: second Hadamard (where priced, estimated)": rotation("hadamard2"),
    "1: Haar rotation (where priced, estimated)": rotation("haar"),
    "1: output-side Hadamard too (where priced, estimated)": rotation("both"),
    "2: least clip that clears, up to 30%": clip,
    "3: live weight noise f=1 (where priced, estimated)": noise(1),
    "3: live weight noise f=4 (where priced, estimated)": noise(4),
    "4: q, k, v at a margin": qkv_margin,
    "4: every broken matrix at a margin": margin,
    "4: q, k, v uncredited": qkv_out,
    "4: every broken matrix uncredited": broken_out,
    "2 and 4: clip up to 30% where it clears, else uncredited; down_proj at a margin": operating(),
    "2 and 4: the same, down_proj in K chunks": operating(down="chunks"),
    "4, 70B's four exact: every broken matrix at a margin; L0 q, k, v and L1 v uncredited": operating(
        0.02, "margin", exact=True),
    "4, 70B's four exact, gate/up deflated by 32 where that clears": operating(0.02, "margin", exact=True,
                                                                               deflate=True),
    "4, 70B's four exact, down_proj in K chunks of 7,168": operating(0.02, "margin", down="chunks", exact=True),
    "2 and 4, 70B's four exact: clip up to 10% where it clears, else at a margin": operating(0.1, "margin",
                                                                                           exact=True),
    "2 and 4, 70B's four exact: clip up to 5% where it clears, else at a margin": operating(0.05, "margin",
                                                                                          exact=True),
}
RECOMMENDED = "4, 70B's four exact, gate/up deflated by 32 where that clears"

NEAR = 0.085  # the most a reorder took off a split, over the leaf (70B L62 q at 2%, r20261104-093237-b5f7 against natural)


def evaluate(book, model, choose):
    """Per chip@price the least cost over credit of a credited matrix (its over the leaf / the share of its MACs
    credited), the matrix setting it, the credited share of the model's MACs, the credited matrices unpriced, and
    those a reorder could take under the line, or further under it: a split at a rank priced in the natural order
    only whose cost less NEAR, over the credit, is under both the line and the matrix's least cost over credit
    (`natural`), or a variant within NEAR of the line never priced with the split (`unsplit`). A deflated matrix's
    part served apart, r·(n + K) MACs, is uncredited and counts in the model's MACs."""
    worst = {k: (math.inf, None, None) for k in KEYS}
    credited, apart = 0.0, 0
    unpriced, natural, unsplit = [], [], []
    for t in all_tensors(model):
        var, credit = choose(book, model, t)
        if credit == 0:
            continue
        if var and var[2]:
            apart += var[2] * sum(shape(model, kind(t)))
        credited += macs(model, kind(t)) * credit
        if var is None:
            unpriced.append(t)
            continue
        s, key = book.v[model][t][var], (model, t, var)
        if key not in book.split:
            if least_of(s) / credit < LINE + NEAR:
                unsplit.append(t)
        elif (book.natural_only(key) - NEAR) / credit < min(LINE, least_of(s) / credit):
            natural.append(t)
        for k in KEYS:
            ol = s[k][0] / credit
            if ol < worst[k][0]:
                worst[k] = (ol, t, s[k][1])
    return worst, credited / (total_macs(model) + apart), unpriced, natural, unsplit


def served_map(book, model, choose) -> dict:
    """What a defence serves off design U's 2% codes: a clip target, deflR for design U's codes of W·R less its top-R
    SVD part (`quant.DeflatedQLinear`), or `out` for EXACT's matrices; a matrix credited at a margin or uncredited
    otherwise is served as design U."""
    out = {}
    for t in all_tensors(model):
        var, credit = choose(book, model, t)
        if credit == 0 and t in EXACT.get(model, ()):
            out[t] = "out"
        elif credit and var and var != U:
            assert var[1] == U[1] and var[3:] == U[3:] and (var[0] == U[0] or not var[2]), \
                f"{t}: {var} isn't a clip or a deflation qat.py serves"
            out[t] = f"defl{var[2]}" if var[2] else var[0]
    return out


def short(t: str) -> str:
    return f"L{layer(t)} {kind(t).removesuffix('_proj')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--matrices", action="store_true", help="also every matrix the split or a defence moves")
    ap.add_argument("--map", choices=MODELS, help="print --defence's targets map for MODEL (qat.py --targets: a clip "
                    "target, or `out` for a matrix served exact; arm70b-out.sh) and stop")
    ap.add_argument("--defence", default=RECOMMENDED, choices=DEFENCES)
    args = ap.parse_args()
    book = load(args.runs)
    if args.map:
        print(json.dumps(served_map(book, args.map, DEFENCES[args.defence]), indent=1))
        return
    print("| defence | model | sm_120 0.84 / 1.48 / 1.78 | H200 0.84 / 1.48 / 1.78 | least set by | MACs credited | "
          f"unpriced | within {NEAR} of the line: a rank priced in the natural order only / unsplit |")
    print("|---|---|---|---|---|---|---|---|")
    for name, choose in DEFENCES.items():
        for model in MODELS:
            worst, share, unpriced, natural, unsplit = evaluate(book, model, choose)
            cells = [" / ".join(f"{worst[f'{c}@{p}'][0]:.3f}" for p in PRICES) for c in ("sm120", "h200")]
            k0 = min(KEYS, key=lambda k: worst[k][0])
            setter = f"{short(worst[k0][1])} ({worst[k0][2]}, {k0})" if worst[k0][1] else "-"
            print(f"| {name} | {model} | {cells[0]} | {cells[1]} | {setter} | {100 * share:.3f}% | {len(unpriced)} | "
                  f"{len(natural)} / {len(unsplit)} |")
    if args.matrices:
        print()
        print(f"{RECOMMENDED}, on every matrix the split breaks at 2% or that it serves otherwise:")
        print()
        print("| model | matrix | 2% least (sm_120 / H200 at 0.84) | its route | served | credited | its least | "
              "priced in |")
        print("|---|---|---|---|---|---|---|---|")
        choose = DEFENCES[RECOMMENDED]
        for model in MODELS:
            for t in all_tensors(model):
                var, credit = choose(book, model, t)
                s = book.get(model, t)
                if s is None or (least_of(s) >= LINE and var == U):
                    continue
                k0 = min(KEYS, key=lambda k: s[k][0])
                if credit == 0:
                    served, its = "exact" if t in EXACT.get(model, ()) else "2% clip", "–"
                elif var is None:
                    served, its = "not priced yet", "–"
                else:
                    served = (f"{var[4]} K chunks" if var[4] > 1 else f"deflated by {var[2]}" if var[2]
                              else "design U" if var == U else f"{var[0]:.0%} clip")
                    its = f"{least_of(book.v[model][t][var]):.3f}"
                served += f" | {credit:.1%}"
                key = (model, t, var)
                nat = sorted(r for r in book.natural_ranks[key] if r not in book.ordered_ranks[key])
                searched = ("natural order" if key not in book.orders else
                            f"all orders, natural at {', '.join(map(str, nat))}" if nat else "all orders")
                searched = searched if key in book.split else "no split"
                print(f"| {model} | {short(t)} | {s['sm120@0.84'][0]:.3f} / {s['h200@0.84'][0]:.3f} | {s[k0][1]} | "
                      f"{served} | {its} | {searched if var else '–'} |")


if __name__ == "__main__":
    main()
