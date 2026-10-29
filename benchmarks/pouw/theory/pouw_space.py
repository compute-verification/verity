#!/usr/bin/env python3
"""The PoUW design space's calculator (NIGHT.md §2): a design is one module on each axis, N noise, D de-noise, F format,
C commitment (density, operands), H hashing (rows, hiding), P premises (final word, forming, derivation), at a W
workload point. Each module contributes fields with their sources, and the calculator enumerates every compatible
composition, prices it under CORE.md §2a on the H200 for Llama-3.2-1B int8 at T 512 (64 and 512 rows) and T 8,192, a
V4-like dense layer, V4-Pro's forward and its routed experts, and decode, and prints each point's ranking and its
Pareto frontier at each rank.

T1 is the additive gap cost(H) − min cost(adv) as a share of stock's m·n·K, in W1 units (an int8 MAC one unit, a register
written `reg` = 32 under H32, measured on the H200). The honest program is the best possible, so every register it writes
costs `reg`. Under DistinctWrites each committed, pairwise-distinct, noise-dependent word costs the adversary at least
`reg`. The gap is honest's work that no premise forces, by term:

- final word: each dot product's last word is A·Bᵀ, free advice (32/K at L 32), unless the de-noise leaves noise in it
  (`final_target`: the last word is a target). T1 quantifies over the data the prover chooses (M0), so it can know the
  total, and N-SAKU's middle-segment skip (proved premise-free) then frees a whole segment: an exact design pays
  max(word, L)/K whatever the premise, and the final-word lemma, stated on the grid uniform in the call's input,
  closes the term only under a claim's `data_model = "grid"` (its origin's model);
- uncommitted steps: r·(1 − u/L) between words committed every L, where a word is charged u = reg (MH100w's
  anyFunctionCeiling) or a premise's `word_units` (BamFloor, 128/3 at L 64 and 48 beyond, N-DENSE's correction of
  SW-SEM; PureLanes or CellPure, L to 128), plus what the best-known fast scheme (Strassen-Winograd, `price.fast_matmul`)
  saves under a premise that credits the k32 counting itself; and the share of words the data repeats (`dup`);
- coin branch: a prover that branches on the coins copies each row's words whose noise vector repeats an earlier row's
  at this salt, and a copy is free in MH100w (N-RED A1): `branch_rows` expected repeated rows a call, by the noise's
  code (`rep_sum` for ssn-1l's code values, `rep_bits` for a b-bit hash code, `rep_lanes` for independent lanes,
  `distinct` for a draw from N values), each n words at the word's charge;
- forming: the formed operand entries at reg/4 each, unless committed under `commit-forced` or charged by the counting
  lemma, and the format's own work (`format_forming`: a rotation's adds at reg each);
- lane fix: F's route's second write to each register whose signed bytes cross zero under the noise (SW-SMELL S9);
- noise derivation: the noise entries derived from XOF bytes (one LOP3 a packed register, CORE-SMELL C28), the same way,
  and the derivation's intermediates no commitment holds, always;
- de-noise: the de-noise's correction work, unless forced;
- pre-noise hashing: bytes hashed before the noise (A's commitment feeds the call's seed), credited once the call is
  atomic; so are rows holding only final words (`final_rows`, C28);
- row repeat: an hm96 row digest is a function of the row alone, so a prover that branches on equal compression inputs
  hashes a repeated row once, and in M0 padding, chosen repeated tokens and a replayed request repeat every row of A
  and of the final words: that credited hashing is gap (N-COMPOSE), unless each row digest's prefix block binds
  (salt, forward, position) (`row-bound`), which costs one compression a row, credited;
- hiding: each salt's work (the PRF and c, and M_key·y, which salt-forced credits; `price.salt`, N-HIDE's W1 count at
  the floor), which a prover with one fixed salt skips with nothing checked changing (SW-HASH §7), unless forced. The
  rows module's `layout` says how many salts: l0 one a leaf, l1 one for A's call and `forward_salts` a forward, l2
  only `forward_salts` (N-HIDE's subtree layouts, private roots). A card's `salt_basis = "leaf-time"` prices a salt as
  SW-HASH's claims did, the salt kernel's share of the leaf's hash time. Hiding is required (NIGHT.md §1 default 2), so
  no ruling waives it;
- host: the prover's host hashes f_host of the rows (its best SHA-512 rate against the card's and its own) and computes
  their words off the card (N-FLOOR's route H), f_host × (credited hashing + words), unless PeriodInventory counts the
  host at θ's rows (HostsCounted) under the premise that θ's row is at least the host's best rate (host-row);
- hashing slack: credited hashing under ShaSpeed, δ times its work (δ = 0 as §2a states it). δ* is the precision
  ShaSpeed needs for T1 to hold (CORE-SMELL C27). The coins' expansion is credited hashing too; T1 with it uncredited
  is reported beside T1 (NIGHT §1 default 4).

The network term is reported beside T1 and outside it: DitherInstance's egress slack n_max · W_mac · E over stock, with
n_max the most MACs a coin bit over the point's matmuls, W_mac the honest work a MAC (the stacked product's and the
credited hashing's) and E the egress cap a served token (`price.egress_cap`, and at its `alt`). A coin bit is an XOF
byte's eight or a streamed coin's own (`stream_bits`, N-EGRESS's coin lanes). Streaming bounds a design whose coins stream
from the challenger a call (`certifier-coins`, under which the prover hashes no XOF for them, so their bytes aren't
credited); a design seeded from the window's salt is outsourced by one message, so its coins don't (`network_streamed`
is what streaming would give). The certifier's fresh salt a leaf (`certifier-salt`, R1b) bounds n_max at a leaf's MACs
over 512 whatever the coins (N-RED's argument); under R1 an outsourced result comes back as words, a word's MACs over
32. `network` is the smallest bound a design takes, null if it takes none. A point that is one matmul has no forward's
egress to divide, so the term isn't priced there.

R1 (`offload`, N-ALT's commit through the certifier): the GPU streams every word, operand and A_j to the network
certifier, which commits them, so the prover hashes nothing (the module drops the base's hashing premises) and T2 pays
the bytes over `price.link` in hashing's place. Its check (b) (`distinct_loss`, N-RED) credits only an entry's
distinct values: it closes the attacks that list it in `closed_by` and costs the honest its chance revisits.

T1's copy model is model P with N-FLOOR's measured floors (`select-floor`, `copy-floor`, in the base; NIGHT §7's
default): a coin lane's step repeats a word with its coin's chance p*, and masked cross-row copies cost more than they
save. `t1_bare_p` is T1 without the floors, adding a noise's `masked_p` share (N-ADAPT's masked channel) where it is
priced; `t1_model_c` is T1 under CopyGrain(4), where a copy needs four entries' coins to agree, so a coin lane's steps
leave nothing (`branch_rows_c`, the rows whose branch survives under C). The ranking reads the default.

Hashing is priced at the best known SHA-512 compression's register count per form (N-SHALB): a chained compression
(`price.hash_work`, 3,532), the bulk of every hm96 row, and a row digest's first, from the constant chaining value
(`price.hash_work_iv`, 3,510), which the row binding's prefix block is. W1 counts registers and the honest program is
the best, so every term that prices hashing reads them, and the served committer's excess (3,632) is printed as its
own gap.

T2 is t_H / t_S, a serial sum: the stock step, the stacked product's extra over the plain GEMM, the noise's coins a call
and NCP's formed copies, A's root before each call's noise (an estimate), exporting each committed word, and hashing at
the layout's measured rate, judged on the faster (async) stock step. T3 is the served format's measured perplexity rise:
the noise's served format when served, else the format's own; an unmeasured T3 fails.

Ranks (NIGHT.md §1): 0, 1, 2, 3, 4a (ShaSpeed's kind: a standardized public primitive), 4bU (SAK on literally uniform
operands), 4b (SAK on our noised operands), 5. Each premise serves the matmul part or the hashing part, and a design's
overall rank is the larger; an open premise ranks 5 until settled (the effective rank, beside the nominal one, its
premises' own). A design an open attack targets can't rank first: an attack targets modules or premises, and one with
`with_any` only designs that also take one of those. An attack that `needs` chosen weights stands while M0 lets the
prover choose the model (the open model-choice question), and each point also reports its best with every such attack
refuted (the model fixed).

A design whose modules leave a field unpriced is refused and never ranked; so is one outside its bound's domain (below
64 rows a call, C15; past its noise's `n_max` columns) or taking a refuted premise. Cards (the `card` tables) name
designs by their modules, with their origin's claimed numbers, which are checked here.

The ranking, at each point (NIGHT.md §5): designs that meet T1 first (T1 is hard; those that don't by T1), then free
of open attacks, then the effective overall rank, then the nominal, then T1, T2 and T3 (an unmeasured T3 last), then
the fewest open premises.
Per-period figures are on the card's one basis (`scale.basis`): stock work held fixed.

  uv run python benchmarks/pouw/theory/pouw_space.py [--card PATH] [--add PATH ...] [--out DIR] [--delta D] [--mutations]

`--add` merges a lane's card file (its cards and any modules, premises, prices or attacks they need) into the card.
"""
from __future__ import annotations

import copy
import csv
import functools
import itertools
import json
import math
import sys
import tomllib
from pathlib import Path

from exact_kit.checks import Checks

CARD = Path(__file__).with_name("pouw_space.toml")
TERMS = ("final_word", "uncommitted_steps", "coin_branch", "forming", "lane_fix", "noise_derivation", "denoise",
         "distinct_loss", "pre_noise_hashing", "final_rows", "row_repeat", "hiding", "host", "hashing_slack")
GAP_TERMS = ("final_word", "uncommitted_steps", "forming", "lane_fix", "noise_derivation", "denoise")
PRICED = ("r", "formed_x", "formed_y", "noise_x", "noise_y", "derive_extra", "denoise", "pre_noise_bytes", "operand_max",
          "lane_fix", "wide_measured", "final_target", "dup", "L", "word_bits", "commit_formed_x", "commit_formed_y",
          "commit_noise_x", "commit_noise_y", "rows_mix_final", "a_row", "row")
STATUSES = ("proved", "measured", "ruled", "assumed", "scratch", "open", "refuted")
UNSETTLED = ("open",)
RANKS = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, "4a": 4.0, "4bU": 4.25, "4b": 4.5, 5: 5.0}
LABEL = {v: str(k) for k, v in RANKS.items()}
PARTS = ("matmul", "hashing")
CLASSES = {"formed_x": "forming", "formed_y": "forming", "noise_x": "noise_derivation", "noise_y": "noise_derivation"}
HASH_CREDIT = ("ShaSpeed", "sha-writes", "carried-words", "hash-compose")
HIDING_PREMISES = ("salt-forced", "mask-optimal", "hiding-ruled")
TARGET = {"t1": 0.01, "t2": 100.0, "t3": 0.05}
MODEL_FIXED = "every attack needing chosen weights (model-public: the model fixed)"
WORD_BYTES = 4
SALT_BITS = 512
BLOCK = 128
MIN_ROWS = 64
PERIODS = {"window": None, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
SHOWN = 25
FRONTIER_SHOWN = 12
def distinct(m: float, N: float) -> float:
    """The expected number of distinct values among m uniform draws from N."""
    return N * -math.expm1(m * math.log1p(-1 / N)) if N > 1 else min(m, 1.0)


@functools.lru_cache(maxsize=None)
def rep_bits(m: float, b: int, S: int) -> float:
    """Expected row-steps repeating an earlier row's vector over m identical rows and S steps, when step t's vector is
    fixed by b + t - 1 uniform coin bits (ssn-1l-hw)."""
    return sum(m - distinct(m, 2.0 ** (b + t - 1)) for t in range(1, int(S) + 1) if b + t - 1 < 1000)


@functools.lru_cache(maxsize=None)
def rep_at(m: float, b: int, t: int) -> float:
    """Row-steps repeating an earlier row's vector at step t alone, over m identical rows, when its vector is fixed by b
    uniform code bits and the sum of t - 1 step bits: 2^b x t classes, the sum binomial."""
    t = int(t)
    d = sum(-math.expm1(m * math.log1p(-math.comb(t - 1, c) / 2.0 ** (t - 1 + b))) for c in range(t))
    return m - 2.0 ** b * d


@functools.lru_cache(maxsize=None)
def rep_sum(m: float, b: int, S: int) -> float:
    """The same summed over steps 1..S (ssn-1l)."""
    return sum(rep_at(m, b, t) for t in range(1, int(S) + 1))


@functools.lru_cache(maxsize=None)
def rep_lanes(m: float, lanes: int, S: int) -> float:
    """The same when step t's vector is fixed by `lanes` sums of t step bits each (ssn-a's row lanes against one W0),
    counted as uniform over the (t + 1)^lanes values the sums take, which bounds it from below if that is the state."""
    return sum(m - distinct(m, (t + 1.0) ** lanes) for t in range(1, int(S) + 1))


FNS = {"__builtins__": {}, "ceil": math.ceil, "floor": math.floor, "min": min, "max": max, "log2": math.log2,
       "distinct": distinct, "rep_bits": rep_bits, "rep_sum": rep_sum, "rep_at": rep_at, "rep_lanes": rep_lanes}


class Refused(Exception):
    """A design or point the card doesn't price: its reasons, each naming the field."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class CardError(Exception):
    """The card itself is malformed (a field defined twice, a cycle, a missing source)."""


@functools.lru_cache(maxsize=None)
def _code(s: str):
    return compile(s, "<card>", "eval")


def ev(s, **v) -> float:
    if s is None or s == "?":
        raise Refused(["unpriced"])
    if isinstance(s, (int, float)):
        return float(s)
    return float(eval(_code(str(s)), FNS, v))  # noqa: S307 - the card's arithmetic


def rank_value(r) -> float:
    if r not in RANKS:
        raise CardError(f"rank {r!r} is not one of {list(RANKS)}")
    return RANKS[r]


def interp(xs: list[float], ys: list[float], x: float) -> float:
    """Log-log between the table's points, its end values outside them."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = next(j for j in range(len(xs) - 1) if xs[j] <= x <= xs[j + 1])
    t = math.log(x / xs[i]) / math.log(xs[i + 1] / xs[i])
    if ys[i] <= 0 or ys[i + 1] <= 0:
        return ys[i] + t * (ys[i + 1] - ys[i])
    return math.exp(math.log(ys[i]) + t * math.log(ys[i + 1] / ys[i]))


class Scope(dict):
    """A composition's fields at one shape, each evaluated on first read from the module that defines it."""

    def __init__(self, fields: "Fields", env: dict):
        super().__init__(env)
        self.fields = fields
        self.stack: list[str] = []

    def __missing__(self, key: str) -> float:
        if key in FNS:
            raise KeyError(key)
        f = self.fields
        if key not in f.exprs:
            if f.lenient:
                return 0.0
            raise Refused([f"field {key!r} unpriced: no module of {f.label} provides it"])
        expr = f.exprs[key]
        if expr in (None, "?"):
            if f.lenient:
                return 0.0
            raise Refused([f"{f.origin[key]}.{key} unpriced"])
        if key in self.stack:
            raise CardError(f"fields {' -> '.join(self.stack + [key])} form a cycle")
        self.stack.append(key)
        try:
            v = float(expr) if isinstance(expr, (int, float)) else float(eval(_code(str(expr)), FNS, self))  # noqa: S307
        finally:
            self.stack.pop()
        self[key] = v
        return v


class Fields:
    """A composition's merged field expressions, with the module each came from; scopes are cached per shape."""

    def __init__(self, exprs: dict, origin: dict, label: str, consts: dict, lenient: bool = False):
        self.exprs, self.origin, self.label, self.consts, self.lenient = exprs, origin, label, consts, lenient
        self._at: dict = {}

    def at(self, m: float, n: float, K: float) -> Scope:
        k = (m, n, K)
        if k not in self._at:
            self._at[k] = Scope(self, {"m": m, "n": n, "K": K} | self.consts)
        return self._at[k]


def merge(into: dict, add: dict, where: str) -> None:
    """`add`'s tables into `into`; a key both define is a card error, so an added file never overrides the card."""
    for k, v in add.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v, f"{where}.{k}")
        elif k in into:
            raise CardError(f"{where}.{k} is already defined: an added file adds, never overrides")
        else:
            into[k] = v


def load(path: Path = CARD, adds: list[Path] = ()) -> dict:
    """The card, with each added file's modules, premises, prices, attacks and cards merged in (a lane's card file)."""
    card = tomllib.loads(path.read_text())
    for p in adds:
        merge(card, tomllib.loads(p.read_text()), p.name)
    for (axis, slot) in card["space"]["slots"]:
        mods = modules(card, axis, slot)
        for mid, mod in list(mods.items()):
            if "like" in mod:
                base = mods.get(mod["like"])
                if base is None or "like" in base:
                    raise CardError(f"{axis}.{mid}: like = {mod['like']!r} names no plain module of its slot")
                new = copy.deepcopy(base)
                for k, v in mod.items():
                    if k == "fields":
                        new["fields"] = new.get("fields", {}) | v
                    elif k == "D":
                        for dm, tab in v.items():
                            t = new.setdefault("D", {}).setdefault(dm, {})
                            t.update({x: (t.get("fields", {}) | y if x == "fields" else y) for x, y in tab.items()})
                    elif k == "premises":
                        new["premises"] = list(dict.fromkeys(new.get("premises", []) + v))
                    elif k != "like":
                        new[k] = v
                mods[mid] = new | {"like_of": mod["like"]}
        for mid, mod in list(mods.items()):
            if "each" in mod:
                del mods[mid]
                for v in mod["each"]:
                    name = mod["name"].format(L=v)
                    mods[name] = {k: w for k, w in mod.items() if k not in ("each", "name")} | {
                        "fields": {k: (w.format(L=v) if isinstance(w, str) else w) for k, w in mod["fields"].items()},
                        "name": name, "each_of": mid}
    return card


def modules(card: dict, axis: str, slot: str) -> dict:
    tab = card.get(axis, {})
    single = sum(1 for a, _ in card["space"]["slots"] if a == axis) == 1
    return tab if single else tab.setdefault(slot, {})


def consts(card: dict) -> dict:
    return {"reg": card["price"]["reg"]["value"], "epr": card["price"]["entries_per_reg"]["value"]}


def choice_of(card: dict, named: dict) -> dict:
    """A card's modules by slot; an axis with one slot may be named by its letter (N, D, F)."""
    slots = card["space"]["slots"]
    single = {a: s for a, s in slots if sum(1 for b, _ in slots if b == a) == 1}
    return {s: named.get(s, named.get(next((a for a, t in single.items() if t == s), s))) for _, s in slots}


def compose(card: dict, choice: dict, extra: list[str] | None = None, lenient: bool = False,
            title: str | None = None, source: str | None = None) -> dict:
    """The design `choice` names (a module id a slot), or Refused. An incompatible choice (a module's `needs` or `when`
    unmet, or a noise without a table for the de-noise) raises Refused with `incompatible` first in its reasons.
    `lenient` (a mutation only) reads every unpriced field as 0 and an unpriced width as stock's."""
    sp = card["space"]
    choice = choice_of(card, choice)
    exprs, origin, prem, mods, frag = dict(sp.get("fields", {})), dict.fromkeys(sp.get("fields", {}), "space"), [], {}, []
    reasons = []
    for axis, slot in sp["slots"]:
        mid = choice.get(slot)
        mod = modules(card, axis, slot).get(mid)
        if mod is None:
            if lenient:
                mods[slot] = {}
                frag.append(str(mid))
                continue
            what = {"N": "its forming, noise derivation, de-noise and pre-noise hashing are unpriced"}.get(axis, "unpriced")
            reasons.append(f"no {axis}.{slot} module {mid!r} in the card: {what}")
            continue
        mods[slot] = mod
        frag.append(mod.get("name", mid))
        layers = [(f"{axis}.{mid}", mod.get("fields", {}), False)]
        if axis == "D":
            ntab = mods.get("noise", {}).get("D", {})
            if choice.get("noise") in modules(card, "N", "noise") and mid not in ntab:
                raise Refused(["incompatible", f"N.{choice['noise']} has no table for D.{mid}"])
            if mid in ntab:
                layers.append((f"N.{choice['noise']}.D.{mid}", ntab[mid].get("fields", {}), True))
                prem += ntab[mid].get("premises", [])
        for where, fl, override in layers:
            for k, v in fl.items():
                if k in exprs and origin[k] != "space" and not override:
                    raise CardError(f"field {k!r} defined by both {origin[k]} and {where}")
                exprs[k], origin[k] = v, where
        prem += mod.get("premises", [])
    if reasons:
        raise Refused(reasons)
    if lenient:
        exprs.setdefault("r", 1)
        origin.setdefault("r", "lenient")
    label = "/".join(f for f in frag[:7]) + ("/" + "+".join(f for f in frag[7:] if f) if any(frag[7:]) else "")
    fields = Fields(exprs, origin, label, consts(card), lenient)
    probe = fields.at(**card["space"]["probe"])
    for slot, mod in mods.items():
        for need in mod.get("needs", []):
            if need not in exprs and not lenient:
                raise Refused(["incompatible", f"{slot} {choice.get(slot)} needs {need!r}"])
        if "when" in mod:
            try:
                ok = bool(eval(_code(mod["when"]), FNS, probe))  # noqa: S307 - the card's condition
            except Refused as e:
                raise Refused(["incompatible", f"{slot} {choice.get(slot)}: {mod['when']} reads "
                                               f"{'; '.join(e.reasons)}"]) from e
            if not ok:
                raise Refused(["incompatible", f"{slot} {choice.get(slot)}: {mod['when']}"])
        for cond, ps in mod.get("premises_when", []):
            if eval(_code(cond), FNS, probe):  # noqa: S307 - the card's condition
                prem += ps
    prem += extra or []
    drops = {p for m in mods.values() for p in m.get("drops", [])}
    prem = [p for p in sp["base"] if p not in drops] + prem
    noise = mods.get("noise", {})
    dmod = mods.get("denoise", {})
    ntab = noise.get("D", {}).get(choice.get("denoise"), {})
    t3_rule = ntab.get("t3", dmod.get("t3", "F"))
    fmt = choice.get("format")
    if isinstance(t3_rule, dict):
        t3 = t3_rule.get(fmt)
    elif t3_rule == "N":
        t3 = noise.get("t3", {}).get(fmt)
    else:
        t3 = mods.get("format", {}).get("t3")
    sources = {slot: m.get("source", "") for slot, m in mods.items()} | (
        {"noise_denoise": ntab.get("source", "")} if ntab else {})
    rows = mods.get("rows", {})
    return {"name": label, "choice": dict(choice), "fields": fields, "premises": list(dict.fromkeys(prem)),
            "coins": noise.get("coins", {}), "t3": t3, "n_max": exprs.get("n_max"), "sources": sources,
            "layout": rows.get("layout", "l0"), "forward_salts": rows.get("forward_salts", 0),
            "title": title or " · ".join(m.get("title", "") for m in mods.values() if m.get("title")),
            "source": source or "; ".join(v for v in sources.values() if v), "variant": noise.get("variant", ""),
            "salt": mods.get("hiding", {}).get("salt")}


def space(card: dict, lenient: bool = False) -> tuple[dict, int, dict]:
    """Every compatible composition (modules with `enumerate = false` only through cards); the count of incompatible
    choices; and the modules left out of the product because a premise they take is refuted."""
    slots = card["space"]["slots"]
    prem = card["premise"]
    options, dropped = [], {}
    for axis, slot in slots:
        ids = []
        for mid, mod in modules(card, axis, slot).items():
            if mod.get("enumerate", True) is False:
                continue
            bad = [p for p in mod.get("premises", []) if prem.get(p, {}).get("status") == "refuted"]
            if bad:
                dropped[f"{axis}.{slot}.{mid}"] = [f"{p}: {prem[p]['source']}" for p in bad]
                continue
            ids.append(mid)
        options.append((slot, ids))
    designs, incompatible = {}, 0
    for combo in itertools.product(*(ids for _, ids in options)):
        choice = {slot: mid for (slot, _), mid in zip(options, combo)}
        try:
            d = compose(card, choice, lenient=lenient)
        except Refused as e:
            if e.reasons[0] == "incompatible":
                incompatible += 1
                continue
            raise
        designs[d["name"]] = d
    return designs, incompatible, dropped


def card_designs(card: dict) -> dict:
    """Each card's design, or its refusal."""
    out = {}
    for cid, c in card.get("card", {}).items():
        try:
            d = compose(card, c["modules"], c.get("premises", []), title=c.get("title"), source=c.get("source"))
            out[cid] = d | {"card": cid}
        except Refused as e:
            out[cid] = e
    return out


def validate(card: dict, d: dict) -> list[str]:
    """Why the design can't be priced, beyond its fields (premises and their requirements)."""
    out: list[str] = []
    prem = card["premise"]
    ids = d["premises"]
    for p in ids:
        if p not in prem:
            out.append(f"premise {p!r} is not in the card")
        elif prem[p].get("status") not in STATUSES:
            out.append(f"premise {p!r} has status {prem[p].get('status')!r}, not one of {STATUSES}")
        elif prem[p].get("status") == "refuted":
            out.append(f"premise {p!r} is refuted ({prem[p].get('source', '')})")
        else:
            for q in prem[p].get("requires", []):
                if q not in ids:
                    out.append(f"premise {p!r} requires {q!r}, which the design doesn't take")
    if not d.get("source"):
        out.append("design source missing")
    return out


def effects(card: dict, ids: list[str], L: int) -> tuple[set[str], bool, float, bool]:
    """The terms the premises close, whether committed registers are forced, the W1 units a committed word every L
    steps is charged (`reg` under DistinctWrites alone, MH100w's anyFunctionCeiling; more under a premise's
    `word_units`), and whether the best-known fast scheme's saving is charged against that credit (`fast_saving`: a
    premise that credits the k32 counting itself, not a floor over every program)."""
    prem = card["premise"]
    reg = card["price"]["reg"]["value"]
    closes = set().union(*(set(prem[p].get("closes", [])) for p in ids if p in prem))
    forced = any(prem[p].get("forces_committed", False) for p in ids if p in prem)
    u, fast = max([(reg, False)] + [(ev(prem[p]["word_units"], L=L, reg=reg), bool(prem[p].get("fast_saving")))
                                    for p in ids if p in prem and "word_units" in prem[p]])
    return closes, forced, u, fast


def calls(card: dict, point: str) -> list[dict]:
    pt = card["point"][point]
    model = card["model"][pt["model"]]
    return [{"name": e[0], "n": e[1], "K": e[2], "count": e[3], "m": pt["tokens"] * e[4],
             "serial": e[5] if len(e) > 5 else e[3]} for e in model["matmuls"]]


def fast_saving(card: dict, L: int, m: float, n: float, operand_max: float) -> tuple[float, int]:
    """The best-known scheme's saving against the k32 counting, as a share of a segment's mnL tensor units, and its levels.

    A level of a <2,2,2; t> scheme runs t products of half the depth. A product's summed side is split into two s8 parts
    when two operands' range passes the tensor's, so it costs c = Σ_p split^(summed sides of p) / 8 of the tensor units
    (P1's floor). ℓ levels pay c^ℓ (a lower bound: sums of sums only split more), plus each level's output-block
    additions (registers at `reg`) and operand-block additions (entries at reg/4):
    saving_ℓ = 1 − c^ℓ − (out·reg/(4L) + (op/2)·(reg/epr)/4·(1/m + 1/n)) · Σ_{i<ℓ} (t/4)^i, with `op` the
    operand-block additions of both sides, each of a quarter of its operand."""
    f = card["price"]["fast_matmul"]
    reg, epr = card["price"]["reg"]["value"], card["price"]["entries_per_reg"]["value"]
    split = 1 if 2 * operand_max <= f["tensor_max"] else 2
    t, best = len(f["products"]), (0.0, 0)
    c = sum(split ** (a + b) for a, b in f["products"]) / 8
    per_level = f["output_adds"] * reg / (4 * L) + f["operand_adds"] / 2 * (reg / epr) / 4 * (1 / m + 1 / n)
    for lv in range(1, f["max_levels"] + 1):
        if L / 2 ** lv < 32 or min(m, n) / 2 ** lv < 1:
            break
        s = 1 - c ** lv - per_level * sum((t / 4) ** i for i in range(lv))
        best = max(best, (s, lv))
    return best


def leaf_time_ratio(card: dict, leaf: float) -> float:
    """hm96's hash time a byte at this leaf size against 16 KiB rows."""
    h = card["price"]["hash_rows"]
    r0 = h["rows"][0]
    if leaf < r0:
        c = h["fixed_compressions"]
        return (leaf / BLOCK + c) / leaf / ((r0 / BLOCK + c) / r0)
    return interp(h["rows"], h["time_ratio"], leaf)


def leaf_hiding(card: dict, leaf: float) -> tuple[float, float]:
    """A leaf's salt side in compressions' time: (PRF and c, M_key·y)."""
    h = card["price"]["hash_rows"]
    hide = interp(h["rows"], [n * s for n, s in zip(h["ns_leaf"], h["hide_share"])], leaf)
    mask = interp(h["rows"], [n * s for n, s in zip(h["ns_leaf"], h["mask_share"])], leaf)
    return (hide - mask) / h["comp_ns"], mask / h["comp_ns"]


def salt_units(card: dict, basis: str, leaves: list[float], own: dict | None = None) -> tuple[list[float], list[float]]:
    """W1 units a salt for each object's leaves: (the PRF and c, M_key·y). "w1" is N-HIDE's register count, one price
    whatever the leaf; "leaf-time" is SW-HASH's, the salt kernel's share of the leaf's hash time at the floor's price a
    compression, which its claims assumed. A hiding module's own `salt` (`own`; a design that replaces hm96's salt
    side, such as N-SALT's pad) is its price on either basis, its `forced` the part salt-forced's analogue leaves."""
    if own is not None:
        return [own["unforced"] - own["forced"]] * len(leaves), [float(own["forced"])] * len(leaves)
    if basis == "w1":
        s = card["price"]["salt"]
        return [s["unforced"] - s["forced"]] * len(leaves), [float(s["forced"])] * len(leaves)
    if basis == "leaf-time":
        unit = BLOCK * card["price"]["hash_work_floor"]["value"]
        both = [leaf_hiding(card, leaf) for leaf in leaves]
        return [a * unit for a, _ in both], [b * unit for _, b in both]
    raise CardError(f"salt basis {basis!r} is not w1 or leaf-time")


def coinsncp_ms(card: dict, entries: float, table: str = "coinsncp") -> float:
    c = card["price"][table]
    if entries >= c["entries"][-1]:
        return c["ms"][-1] * entries / c["entries"][-1]
    return interp(c["entries"], c["ms"], entries)


def call_units(card: dict, d: dict, m: float, n: float, K: float, g: float = 1.0, rate: float = 0.0) -> dict:
    """One call's T1 terms in W1 units, the credited hashing's bytes, and its share of T2's per-call parts (ms) when `g`
    calls run side by side in one serial step and hashing runs at `rate` B/s on 16 KiB rows."""
    f = d["fields"].at(m, n, K)
    price = card["price"]
    reg, epr, hw = price["reg"]["value"], price["entries_per_reg"]["value"], price["hash_work"]["value"]
    hw_floor = price["hash_work_floor"]["value"]
    dup = f["dup"]
    r = f["r"]
    L = f["L"]
    depth = round(r * K, 6)
    if depth % L:
        raise Refused([f"{r:g}K = {depth:g} is not a whole number of words at L {L:g}"])
    if L >= depth:
        raise Refused([f"a word every {L:g} steps commits only the final word of a {depth:g}-deep dot product, and the "
                       f"final word isn't noise-dependent"])
    if f["word_bits"] != 32:
        raise Refused([f"a {f['word_bits']:g}-bit word is unpriced: DistinctWrites counts 32-bit registers, and no "
                       f"premise prices a narrower committed word"])
    closes, forced, u_word, fast = effects(card, d["premises"], L)
    stock = m * n * K
    words = depth / L
    skip = not f["final_target"] and d.get("data_model", "m0") == "m0"
    final_free = ("final_word" not in closes or skip) and not f["final_target"]
    honest_tensor = r * m * n * K
    t = dict.fromkeys(TERMS, 0.0)
    t["final_word"] = m * n * (max(u_word, L) if skip else u_word) if final_free else 0.0
    t["uncommitted_steps"] = honest_tensor - m * n * words * u_word
    if abs(t["uncommitted_steps"]) < 1e-9 * honest_tensor:
        t["uncommitted_steps"] = 0.0
    if fast and L > 32:
        t["uncommitted_steps"] += fast_saving(card, L, m, n, f["operand_max"])[0] * honest_tensor
    t["uncommitted_steps"] += dup * honest_tensor
    t["coin_branch"] = f["branch_rows"] * n * u_word * 32 / L
    branch_c = f["branch_rows_c"] * n * u_word * 32 / L
    committed = {k for k in CLASSES if f[f"commit_{k}"]}
    ent = {k: f[k] for k in CLASSES}
    for k, term in CLASSES.items():
        if not ((forced and k in committed) or term in closes):
            t[term] += ent[k] * reg / epr
    t["forming"] += f["format_forming"]
    if "noise_derivation" not in closes:
        t["noise_derivation"] += f["derive_extra"] * reg / epr
    t["lane_fix"] = f["lane_fix"] * reg / epr
    if "denoise" not in closes:
        t["denoise"] = f["denoise"]
    off = bool(f["offload"])
    t["distinct_loss"] = f["distinct_loss"] * m * n * words * u_word
    pre_b = f["pre_noise_bytes"]
    word_b = m * n * words * WORD_BYTES
    op_b = sum(ent[k] for k in committed)
    if "pre_noise_hashing" not in closes:
        t["pre_noise_hashing"] = pre_b * hw
    if not f["rows_mix_final"] and "final_rows" not in closes:
        t["final_rows"] = m * n * WORD_BYTES * hw
    if d.get("data_model", "m0") == "m0" and "row_repeat" not in closes:
        t["row_repeat"] = (pre_b * hw if "pre_noise_hashing" in closes else 0.0) + (
            m * n * WORD_BYTES * hw if not f["rows_mix_final"] and "final_rows" in closes else 0.0)

    a_leaf = min(f["a_row"], pre_b)
    row = f["row"]
    objects = [(pre_b, a_leaf), (word_b, min(row, word_b)), (op_b, min(row, op_b) if op_b else 1.0)]
    bound_b = BLOCK * sum(math.ceil(b / leaf) for b, leaf in objects if b) if "row_repeat" in closes and not off else 0.0
    layout = d.get("layout", "l0")
    if off:
        salts = [0, 0, 0]
    elif layout == "l0":
        salts = [math.ceil(b / leaf) if b else 0 for b, leaf in objects]
    else:
        salts = [0, 1 if layout == "l1" else 0, 0]
    salt_w, mask_w = salt_units(card, d.get("salt_basis", "w1"), [leaf for _, leaf in objects], d.get("salt"))
    salt_c = sum(s * w for s, w in zip(salts, salt_w))
    mask_c = sum(s * w for s, w in zip(salts, mask_w))
    hid = [s * (w + k) for s, w, k in zip(salts, salt_w, mask_w)]
    if "hiding" not in closes:
        t["hiding"] = (0.0 if "hiding_salt" in closes else salt_c) + (0.0 if "hiding_mask" in closes else mask_c)
    hiding_closed = salt_c + mask_c - t["hiding"]
    salt_b = salt_c / hw_floor if ("hiding_salt" in closes and "hiding" not in closes) else 0.0
    xof = {table: [float(s) if isinstance(s, (int, float)) else float(eval(_code(s), FNS, f)) for s in exprs]  # noqa: S307
           for table, exprs in d["coins"].items()}
    if not xof and not d["fields"].lenient:
        raise Refused([f"N.{d['choice'].get('noise')}.coins unpriced"])
    xof_b = sum(map(sum, xof.values()))
    coin_b = 0.0 if "coin_hashing" in closes else xof_b * price["coin_bytes"]["value"]
    coin_bits = 8 * xof_b + f["stream_bits"]
    credited_b = 0.0 if off else word_b + op_b + coin_b + salt_b + bound_b + (pre_b if "pre_noise_hashing" in closes else 0.0)
    link_s = (word_b + op_b + pre_b) / price["link"]["value"] if off else 0.0

    lat = price["latency"]
    a_leaves = math.ceil(pre_b / a_leaf) if a_leaf else 0
    tree = math.ceil(math.log2(a_leaves)) if a_leaves > 1 else 0
    chain = math.ceil(a_leaf / BLOCK) + 1 + 2 + 3 * tree + 1
    a_chain_s = 0.0 if off else chain * lat["compression_s"] + (2 + tree) * lat["launch_s"]
    a_bulk_s = g * pre_b * leaf_time_ratio(card, a_leaf) / rate if rate and a_leaf and not off else 0.0
    ex = price["extract"]
    sts = ex["sts_units_per_word"] / (ex["units_per_clock"] * ex["sts_mhz"] * 1e6)
    stg = ex["stg_units_per_word"] / (ex["units_per_clock"] * ex["stg_mhz"] * 1e6)
    nwords = m * n * words
    ratio = (lambda leaf: f["hash_time_ratio"]) if "hash_time_ratio" in d["fields"].exprs else (
        lambda leaf: leaf_time_ratio(card, leaf))
    bulk = ((word_b * ratio(objects[1][1]) + (op_b * ratio(objects[2][1]) if op_b else 0.0) + bound_b) / rate
            if rate and not off else 0.0)
    return {"stock": stock, "terms": t, "credited_bytes": credited_b, "bound_bytes": bound_b,
            "hashed_bytes": word_b + op_b + pre_b,
            "word_bytes": word_b, "operand_bytes": op_b, "pre_noise_bytes": pre_b, "coin_bytes": coin_b, "r": r,
            "x_entries": ent["formed_x"] + ent["noise_x"], "y_entries": ent["formed_y"] + ent["noise_y"],
            "noise_y": ent["noise_y"], "mK": m * K, "nK": n * K, "words": nwords, "word_units": nwords * u_word,
            "extra_macs": (r - 1) * stock,
            "macs_per_coin_bit": m * n * K / coin_bits if coin_bits else math.inf,
            "macs_per_salt_bit": min(row, word_b) / WORD_BYTES * stock / nwords / SALT_BITS, "coin_branch_c": branch_c,
            "macs_per_word_bit": stock / nwords / 32 if off else math.inf, "link_ms": link_s * 1e3,
            "masked": f["masked_p"] * stock if f["masked_p"] >= 0 else math.nan,
            "coinform_ms": sum(coinsncp_ms(card, b * g, table) for table, bs in xof.items() for b in bs) / g,
            "a_ms": max(a_chain_s, a_bulk_s) / g * 1e3, "a_chain": chain / g, "bulk_hash_ms": bulk * 1e3,
            "sts_ms": nwords * sts * 1e3, "stg_ms": nwords * stg * 1e3, "hiding_closed": hiding_closed,
            "hiding_a": hid[0], "hiding_words": hid[1], "hiding_operands": hid[2], "salts": sum(salts),
            "wide_measured": f["wide_measured"]}


def t3_of(card: dict, served: str | None, model: str) -> dict:
    e = card.get("t3", {}).get(served or "")
    if not e or ("models" in e and model not in e["models"]) or e.get("kind") == "unmeasured":
        why = f": {e['source']}" if e and e.get("kind") == "unmeasured" else ""
        return {"ctx2k": None, "ctx8k": None, "kind": "unmeasured", "key": served,
                "source": f"no T3 for {served!r} on {model}{why}"}
    return {"ctx2k": e["ctx2k"], "ctx8k": e["ctx8k"], "kind": e["kind"], "key": served, "source": e["source"]}


def ranks(card: dict, ids: list[str]) -> dict:
    """Each part's rank, nominal (its premises' own) and effective (5 while one is open), and the overall (the larger)."""
    prem = card["premise"]
    out = {}
    for part in PARTS:
        ps = [p for p in ids if prem[p].get("part", "matmul") == part]
        nom = max((rank_value(prem[p]["rank"]) for p in ps), default=0.0)
        out[part] = nom
        out[f"{part}_effective"] = 5.0 if any(prem[p]["status"] in UNSETTLED for p in ps) else nom
    out["overall"] = max(out[p] for p in PARTS)
    out["overall_effective"] = max(out[f"{p}_effective"] for p in PARTS)
    return out


def attacked(card: dict, d: dict) -> list[str]:
    """The open attacks on the design: a target names a premise, a module as slot.id or axis.id, or a noise by its id;
    a premise or module in the attack's `closed_by` refuses it for the design."""
    axis = {s: a for a, s in card["space"]["slots"]}
    targets = (set(d["premises"]) | {f"{s}.{m}" for s, m in d["choice"].items()}
               | {f"{axis[s]}.{m}" for s, m in d["choice"].items() if s in axis} | {d["choice"].get("noise", "")})
    return [a for a, v in card.get("attack", {}).items()
            if isinstance(v, dict) and v.get("status") == "open" and targets & set(v.get("targets", []))
            and ("with_any" not in v or targets & set(v["with_any"])) and not targets & set(v.get("closed_by", []))]


SUM_KEYS = ("stock", "credited_bytes", "bound_bytes", "hashed_bytes", "word_bytes", "operand_bytes", "pre_noise_bytes", "coin_bytes",
            "x_entries", "y_entries", "noise_y", "mK", "nK", "words", "word_units", "coinform_ms", "a_ms", "a_chain",
            "bulk_hash_ms",
            "sts_ms", "stg_ms", "extra_macs", "hiding_closed", "hiding_a", "hiding_words", "hiding_operands", "salts",
            "coin_branch_c", "link_ms", "masked")


def evaluate(card: dict, d: dict, point: str, delta: float = 0.0) -> dict:
    """The design at the point: T1's terms, T2, T3, ranks and what's open. Raises Refused."""
    reasons = validate(card, d)
    if reasons:
        raise Refused(reasons)
    pt = card["point"][point]
    cs = calls(card, point)
    small = [f"{x['name']} ({x['m']:g} x {x['n']})" for x in cs if min(x["m"], x["n"]) < MIN_ROWS]
    if small:
        raise Refused([f"outside the bound's domain: {', '.join(small)} below {MIN_ROWS} rows a call, where MH100's floor "
                       f"is shape-blind (CORE-SMELL C15)"])
    n_max = ev(d["n_max"]) if d.get("n_max") is not None else math.inf
    wide = [f"{x['name']} (n {x['n']:,.0f})" for x in cs if x["n"] > n_max]
    if wide:
        raise Refused([f"outside the noise's domain: {', '.join(wide)} past its column code's {n_max:,.0f} columns"])
    rate = pt.get("hash_rate", card["price"]["hash_rate"]["value"])
    agg: dict = dict.fromkeys(SUM_KEYS, 0.0) | {"terms": dict.fromkeys(TERMS, 0.0)}
    per_matmul = {}
    wide_measured = r = None
    bit_macs = salt_macs = word_macs = 0.0
    for x in cs:
        u = call_units(card, d, x["m"], x["n"], x["K"], x["count"] / x["serial"], rate)
        r, wide_measured = u["r"], u["wide_measured"]
        bit_macs = max(bit_macs, u["macs_per_coin_bit"])
        salt_macs = max(salt_macs, u["macs_per_salt_bit"])
        word_macs = max(word_macs, u["macs_per_word_bit"])
        w = x["count"]
        for k in SUM_KEYS:
            agg[k] += w * u[k]
        for tk in TERMS:
            agg["terms"][tk] += w * u["terms"][tk]
        per_matmul[x["name"]] = {"m": x["m"], "n": x["n"], "K": x["K"], "calls": w,
                                 "t1": sum(u["terms"].values()) / u["stock"],
                                 "terms": {k: v / u["stock"] for k, v in u["terms"].items()}}
    stock = agg["stock"]
    hw = card["price"]["hash_work"]["value"]
    fs = d.get("forward_salts", 0) if d.get("layout", "l0") != "l0" else 0
    if fs:
        closes = effects(card, d["premises"], 32)[0]
        (sw,), (mw,) = salt_units(card, d.get("salt_basis", "w1"), [1.0], d.get("salt"))
        charged = 0.0 if "hiding" in closes else ((0.0 if "hiding_salt" in closes else sw)
                                                  + (0.0 if "hiding_mask" in closes else mw))
        agg["terms"]["hiding"] += fs * charged
        agg["hiding_closed"] += fs * (sw + mw - charged)
        agg["hiding_words"] += fs * (sw + mw)
        agg["salts"] += fs
        if "hiding_salt" in closes and "hiding" not in closes:
            agg["credited_bytes"] += fs * sw / card["price"]["hash_work_floor"]["value"]
    hash_share = ((agg["credited_bytes"] - agg["bound_bytes"]) * hw
                  + agg["bound_bytes"] * card["price"]["hash_work_iv"]["value"]) / stock
    host = card["price"]["host_sha"]["value"]
    f_host = host / (rate + host)
    host_route = f_host * (hash_share + agg["word_units"] / stock)
    terms = {k: v / stock for k, v in agg["terms"].items()}
    if "host" not in effects(card, d["premises"], 32)[0]:
        terms["host"] = host_route
    w_mac = (stock + agg["extra_macs"]) / stock + hash_share
    egress = card["price"]["egress_cap"]
    per_bit = None if pt.get("matmul_only") else w_mac * egress["value"] * pt["tokens"] / stock
    network_streamed = None if per_bit is None else bit_macs * per_bit
    streamed = "certifier-coins" in d["premises"]
    salted = "certifier-salt" in d["premises"]
    bounds = {"streamed": bit_macs if streamed else math.inf, "salt": salt_macs if salted else math.inf,
              "words": word_macs}
    bound_by = min(bounds, key=bounds.get)
    n_bound = bounds[bound_by]
    network = None if per_bit is None or n_bound == math.inf else n_bound * per_bit
    alt = egress["alt"] / egress["value"]
    terms["hashing_slack"] = delta * hash_share
    rest = sum(v for k, v in terms.items() if k != "hashing_slack")
    t1 = rest + terms["hashing_slack"]
    t1_model_c = t1 - terms["coin_branch"] + agg["coin_branch_c"] / stock
    t1_bare_p = None if math.isnan(agg["masked"]) else t1 + agg["masked"] / stock
    delta_star = max(0.0, TARGET["t1"] - rest) / hash_share if hash_share else math.inf
    committer_gap = agg["credited_bytes"] * (card["price"]["hash_work_floor"]["value"] - hw) / stock
    t2 = t2_of(card, point, agg, r, bool(wide_measured))
    t3 = t3_of(card, d["t3"], pt["model"])
    prem = card["premise"]
    ids = d["premises"]
    rk = ranks(card, ids)
    open_items = [{"premise": p, "owner": prem[p].get("owner", ""), "status": prem[p]["status"],
                   "rank": prem[p]["rank"], "part": prem[p].get("part", "matmul"), "source": prem[p]["source"]}
                  for p in ids if prem[p]["status"] in UNSETTLED]
    scratch = [p for p in ids if prem[p]["status"] == "scratch"]
    assumed = [{"premise": p, "rank": prem[p]["rank"], "source": prem[p]["source"]}
               for p in ids if prem[p]["status"] == "assumed" and p not in card["space"]["base"]]
    t3w = max(v for v in (t3["ctx2k"], t3["ctx8k"]) if v is not None) if t3["ctx2k"] is not None else None
    att = attacked(card, d)
    return {"design": d["name"], "title": d["title"], "point": point, "t1": t1, "terms": terms,
            "gap": sum(terms[k] for k in GAP_TERMS), "t1_pass": t1 <= TARGET["t1"],
            "t1_coins_uncredited": t1 + agg["coin_bytes"] * hw / stock, "hiding_closed": agg["hiding_closed"] / stock,
            "hiding_words": agg["hiding_words"] / stock, "hiding_a": agg["hiding_a"] / stock,
            "delta": delta, "delta_star": delta_star, "hash_units_per_mac": hash_share,
            "hash_bytes_per_mac": agg["credited_bytes"] / stock, "coin_bytes_per_mac": agg["coin_bytes"] / stock,
            "pre_noise_units": agg["pre_noise_bytes"] * hw / stock,
            "committer_gap": committer_gap, "sha_regs": hw * BLOCK / card["price"]["reg"]["value"],
            "f_host": f_host, "host_route": host_route, "network": network, "network_streamed": network_streamed,
            "coins_streamed": streamed, "macs_per_coin_bit": bit_macs, "w_mac": w_mac, "egress_cap": egress["value"],
            "network_streamed_alt": None if network_streamed is None else network_streamed * alt,
            "network_alt": None if network is None else network * alt, "egress_cap_alt": egress["alt"],
            "macs_per_salt_bit": salt_macs, "certifier_salted": salted,
            "network_bound": None if network is None else bound_by, "macs_per_word_bit": word_macs,
            "t1_model_c": t1_model_c, "t1_model_c_pass": t1_model_c <= TARGET["t1"], "t1_bare_p": t1_bare_p,
            "row_bound_share": agg["bound_bytes"] / agg["credited_bytes"] if agg["credited_bytes"] else 0.0,
            "t2": t2, "t2_pass": t2["judged"] <= TARGET["t2"],
            "t3": t3, "t3_worst": t3w, "t3_pass": t3w is not None and t3w <= TARGET["t3"], "ranks": rk,
            "rank": rk["overall"], "rank_effective": rk["overall_effective"], "premises": ids, "choice": d["choice"],
            "open": open_items, "scratch": scratch, "assumed": assumed, "attacks": att, "per_matmul": per_matmul,
            "stock_macs_forward": stock, "a_chain_compressions": agg["a_chain"], "salts_forward": agg["salts"],
            "card": d.get("card")}


def t2_of(card: dict, point: str, agg: dict, r: float, wide_measured: bool) -> dict:
    """t_H / t_S a forward: stock, plus the stacked product's extra (measured: GD-DENOISE's 3K-wide call; else the plain
    GEMM's time per MAC on the extra width, an estimate), the coins and formed copies a call, A's root before each
    call's noise, the words' export, and hashing; and CORE-NCP's reading (GD-DENOISE's stand-ins for forming, NCP's
    only)."""
    pt = card["point"][point]
    price = card["price"]
    src = card["point"][pt.get("gemm_from", point)]
    parts = src["gemm_parts"]
    rate = pt.get("hash_rate", price["hash_rate"]["value"])
    if pt.get("matmul_only"):
        src_macs = sum(x["count"] * x["m"] * x["n"] * x["K"] for x in calls(card, pt["gemm_from"]))
        plain = pt.get("plain_gemm_ms", agg["stock"] / (src_macs / src["plain_gemm_ms"]))
        stock = {"sync": plain, "async": plain}
    else:
        plain = src["plain_gemm_ms"]
        stock = dict(pt["stock_ms"])
    if wide_measured:
        gemm = plain * (parts["wide"] * (r - 1) / 2 + parts["noised"])
    else:
        gemm = plain * agg["extra_macs"] / agg["stock"]
    extra_parts = {"wide_gemm": gemm, "coins_forming": agg["coinform_ms"], "a_root": agg["a_ms"],
                   "export": agg["sts_ms"]} | ({"link": agg["link_ms"]} if agg["link_ms"] else {})
    extra = sum(extra_parts.values())
    hash_ms = agg["bulk_hash_ms"]
    eng = {e: {"stock_ms": s, "t_H_ms": s + extra + hash_ms, "ratio": (s + extra + hash_ms) / s,
               "hash_share": hash_ms / (s + extra + hash_ms)} for e, s in stock.items()}
    unfused = max((s + extra - agg["sts_ms"] + agg["stg_ms"] + hash_ms) / s for s in stock.values())
    rx = agg["x_entries"] / (4 * agg["mK"])
    ry = agg["y_entries"] / (4 * agg["nK"])
    old = (plain * parts["wide"] * (r - 1) / 2 + plain * parts["a_side"] * rx + plain * parts["noised"]
           + plain * parts["w_side"] * ry + agg["noise_y"] / 64 / price["coin_rate"]["value"] * 1e3
           + agg["hashed_bytes"] / rate * 1e3)
    return {"engines": eng, "judged": max(v["ratio"] for v in eng.values()), "unfused": unfused,
            "extra_ms": extra, "extra_parts_ms": extra_parts, "hash_ms": hash_ms, "plain_gemm_ms": plain,
            "core_ncp": {e: (s + old) / s for e, s in stock.items()},
            "without_hashing": max((s + extra) / s for s in stock.values()), "bytes_forward": agg["hashed_bytes"],
            "kind": "estimate on measured parts: " + (pt.get("gemm_source", "") if pt.get("matmul_only") else
                                                      f"{pt['stock_kind']} stock; 3K product {src['gemm_kind']}") +
                    "; coins and forming measured (coinsncp); A's root chain estimated; export measured on resident "
                    "tiles; no end-to-end honest program exists"}


def rank_key(row: dict) -> tuple:
    return (not row["t1_pass"], 0.0 if row["t1_pass"] else row["t1"], bool(row["attacks"]), row["rank_effective"],
            row["rank"], round(row["t1"], 12), row["t2"]["judged"],
            row["t3_worst"] if row["t3_worst"] is not None else math.inf, len(row["open"]))


def ranking(card: dict, designs: dict, point: str, delta: float = 0.0) -> tuple[list[dict], dict[str, list[str]]]:
    rows, refused = [], {}
    for name, d in designs.items():
        try:
            rows.append(evaluate(card, d, point, delta))
        except Refused as e:
            refused[name] = e.reasons
    return sorted(rows, key=rank_key), refused


def objectives(row: dict) -> tuple[float, float, float]:
    return row["t1"], row["t2"]["judged"], row["t3_worst"] if row["t3_worst"] is not None else math.inf


def frontier(rows: list[dict]) -> list[dict]:
    """The rows no other row beats on T1, T2 and T3 at once (ties kept once, by name)."""
    pts = sorted(rows, key=lambda r: (objectives(r), r["design"]))
    out: list[dict] = []
    for r in pts:
        o = objectives(r)
        if not any(all(a <= b for a, b in zip(objectives(q), o)) for q in out):
            out.append(r)
    return out


def frontiers(rows: list[dict], key: str) -> list[tuple[float, list[dict]]]:
    """At each rank level (by `key`: rank_effective or rank), the Pareto frontier of the T1-passing designs at or below it,
    where it changes."""
    passing = [r for r in rows if r["t1_pass"] and not r["attacks"]]
    out, last = [], None
    for lv in sorted({r[key] for r in passing}):
        fr = frontier([r for r in passing if r[key] <= lv])
        names = [r["design"] for r in fr]
        if names != last:
            out.append((lv, fr))
            last = names
    return out


def opened(row: dict) -> str:
    return ", ".join(o["premise"] for o in row["open"] if o["premise"] not in HASH_CREDIT) or "-"


def rk(row: dict, eff: bool = False) -> str:
    r = row["ranks"]
    s = "_effective" if eff else ""
    return f"{LABEL[r['matmul' + s]]}/{LABEL[r['hashing' + s]]} ({LABEL[r['overall' + s]]})"


def needs(row: dict) -> list[str]:
    out = [f"{o['premise']} ({o['owner']}, {o['part']} part, rank {o['rank']} once settled; open): {o['source']}"
           for o in row["open"]]
    out += [f"{p} (proved in scratch, in no lock: lock it)" for p in row["scratch"]]
    out += [f"{a['premise']} (assumed, rank {a['rank']}): {a['source']}" for a in row["assumed"]]
    if row["hiding_closed"]:
        out.append(f"the salt side's hiding work, {row['hiding_closed']:.2%} of stock, is closed by "
                   f"{' and '.join(p for p in row['premises'] if p in HIDING_PREMISES)}")
    if not row["terms"]["host"]:
        out.append(f"the host's work is counted by period-inventory (HostsCounted at θ's rows) and host-row: without "
                   f"them the host hashes f_host = {row['f_host']:.2%} of the rows and computes their words off the card "
                   f"(N-FLOOR's route H), {row['host_route']:.2f}x stock, {row['host_route'] / TARGET['t1']:.0f}x T1's "
                   f"allowance")
    if row["network_streamed"] is not None:
        out.append(f"the network term, reported beside T1 and outside it pending Daniel (N-ALT): DitherInstance's slack "
                   f"n_max · W_mac · E over stock, at E = {row['egress_cap']:g} bits a token, n_max "
                   f"{row['macs_per_coin_bit']:,.0f} MACs a coin bit and W_mac {row['w_mac']:.1f} W1 a MAC, is "
                   f"{row['network_streamed']:.2%} of stock ({row['network_streamed_alt']:.2%} at "
                   f"{row['egress_cap_alt']:g} bits) if the call's coins stream from the challenger"
                   + (" (certifier-coins, taken)" if row["coins_streamed"] else
                      "; this design's coins are seeded from the window's salt, so one message outsources the window "
                      "and OutsourceWithinEgress gives no bound by the coins")
                   + (f"; the certifier's per-leaf salt (certifier-salt, taken) bounds n_max at a leaf's MACs over "
                      f"{SALT_BITS}, {row['macs_per_salt_bit']:,.0f}, so the term is at most {row['network']:.2%} "
                      f"({row['network_alt']:.2%} at {row['egress_cap_alt']:g} bits; N-RED final §3, an argument, "
                      f"its arithmetic checked: a fresh salt a leaf and a leaf count the protocol fixes)"
                      if row["certifier_salted"] else "")
                   + (f"; under R1 an outsourced result comes back through the port as words, "
                      f"{row['macs_per_word_bit']:.3f} MACs a bit, so the term is {row['network']:.1e} (N-ALT final; "
                      f"the word stream ends at the certifier and is excluded from E)"
                      if row["network_bound"] == "words" else ""))
    if row["t1_bare_p"] is not None:
        out.append(f"under bare model P, without select-floor and copy-floor, masked cross-row copies are credited and "
                   f"unpaid: T1 {row['t1_bare_p']:.2%} (N-ADAPT final, checked at toy shapes); T1 rests on P with the "
                   f"two measured floors (NIGHT §7, 11:30Z), under which that channel is 0 by argument")
    if row["t1_model_c"] != row["t1"]:
        out.append(f"under model C (CopyGrain(4)) T1 would be {row['t1_model_c']:.3%} against P's {row['t1']:.3%}, the "
                   f"coin branch {row['terms']['coin_branch']:.3%} under P; CopyGrain(4) is false on sm_90 (N-FLOOR), so "
                   f"on the H200 C reduces to P with copy-floor")
    if not row["t1_pass"]:
        big = max(((k, v) for k, v in row["terms"].items()), key=lambda kv: kv[1])
        out.append(f"T1 {row['t1']:.2%} > 1%: its largest term is {big[0]} {big[1]:.2%}")
    t2 = row["t2"]
    big = max(t2["extra_parts_ms"].items(), key=lambda kv: kv[1])
    if not row["t2_pass"]:
        out.append(f"T2 {t2['judged']:.1f}x > 100x (hashing {t2['hash_ms']:.1f} ms a forward, largest other part {big[0]} "
                   f"{big[1]:.1f} ms); {row['hash_bytes_per_mac']:.3f} B of credited hashing a useful MAC")
    elif t2["unfused"] > TARGET["t2"]:
        out.append(f"a committer fused with the GEMM: exporting the words to HBM instead of shared memory makes T2 "
                   f"{t2['unfused']:.1f}x")
    out.append(f"T2's parts measured where stated: A's root before each call's noise ({row['a_chain_compressions']:.0f} "
               f"compressions' chain a forward, {t2['extra_parts_ms']['a_root']:.2f} ms) is an estimate nobody has timed")
    if row["t3_worst"] is None:
        out.append(f"T3 unmeasured: {row['t3']['source']}")
    elif not row["t3_pass"]:
        out.append(f"T3 {row['t3_worst']:+.2%} > 5%")
    regs = row["sha_regs"] * min(row["delta_star"], 1.0)
    out.append(f"ShaWrites as ShaLB({row['sha_regs']:,.0f}) (rank 4a, N-SHALB) exact to {regs:.2f} registers a compression "
               f"(delta* = {row['delta_star']:.2e} of its {row['hash_units_per_mac']:.0f} W1 units a useful MAC; "
               f"ShaSpeed's form and the composition with DistinctWrites, hash-compose, are its theorems in scratch, "
               f"N-COMPOSE), {row['row_bound_share']:.2%} of it the row binding's prefix blocks; if the honest hasher "
               f"is our committer (3,632 "
               f"registers), not the best, its excess alone is {row['committer_gap']:.2f}x stock; with the coins' "
               f"expansion uncredited T1 is {row['t1_coins_uncredited']:.3%}")
    return out


def periods(card: dict, row: dict) -> dict:
    """Over each period, on the card's one basis ([scale] basis): stock work held fixed, for one H200 and the datacenter
    at the int8 tensor peak times mfu_serve, and for the world at the census's inference compute. Each scope gives its
    stock work, the gap (T1 of it), the allowance (1% of it), ShaSpeed's credited hashing and the slack delta* leaves
    it, in W1 at stock's AND rate (W1 units x and_per_mac), and the credited hashing in real AND gates too (`sha_and` a
    compression); the hardware that serves it under the design is T2 times stock's, beside it, never folded in."""
    sc = card["scale"]
    and_mac = card["price"]["and_per_mac"]["value"]
    sha_and_mac = row["hash_bytes_per_mac"] * card["price"]["sha_and"]["value"] / BLOCK
    tH = min(e["t_H_ms"] for e in row["t2"]["engines"].values())
    tS = min(e["stock_ms"] for e in row["t2"]["engines"].values())
    gpu = card["price"]["mac_rate_peak"]["value"] * sc["mfu_serve"]
    scopes = {"h200": gpu, "datacenter": gpu * sc["datacenter_gpus"], "world": sc["world_mac_per_year"] / PERIODS["year"]}
    out: dict = {"stock_mac_per_s": scopes, "hardware_multiple": tH / tS,
                 "proved": "T1's terms are the card's priced counts under the row's premises",
                 "scaled": "the datacenter is one H200 at mfu_serve times datacenter_gpus; the world is "
                           "world_mac_per_year; the hardware multiple is t_H's estimate on measured parts"}
    for p, s in PERIODS.items():
        s = sc["window_s"] if s is None else s
        out[p] = {"seconds": s}
        for scope, macs_s in scopes.items():
            work = macs_s * s * and_mac
            out[p][scope] = {"stock_ands": work, "gap_ands": row["t1"] * work, "allowance_ands": 0.01 * work,
                             "shaspeed_ands": row["hash_units_per_mac"] * work,
                             "shaspeed_real_ands": sha_and_mac * macs_s * s,
                             "host_route_ands": row["host_route"] * work,
                             "shaspeed_slack_ands": min(row["delta_star"], 1e9) * row["hash_units_per_mac"] * work}
    return out


def period_table(card: dict, row: dict) -> list[str]:
    per = periods(card, row)
    sc = card["scale"]
    lines = [f"Per monitoring period on the card's one basis: {sc['basis']} AND gates at "
             f"{card['price']['and_per_mac']['value']:g} an int8 MAC (estimate). The design serves that work on "
             f"{per['hardware_multiple']:.1f}x stock's hardware (T2). The gap is T1 x the stock work and the allowance 1% of "
             f"it, in W1 at stock's AND rate, since T1 is judged in W1; ShaSpeed's slack is delta* x its credited "
             f"hashing, the most it may err before T1 exceeds 1%. The credited hashing is also given in real AND gates, "
             f"{card['price']['sha_and']['value']:,} a compression. Route H is what the host takes off the card if "
             f"period-inventory or host-row fails (in T1 then), in W1 at stock's AND rate.", "",
             "| period | scope | stock work | gap (T1) | allowance (1%) | ShaSpeed-credited hashing, W1 at stock's AND "
             "rate | the same in real AND gates | ShaSpeed slack at delta* | route H if period-inventory fails |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in PERIODS:
        for scope in ("h200", "datacenter", "world"):
            v = per[p][scope]
            lines.append(f"| {p} ({per[p]['seconds']:,} s) | {scope} | {v['stock_ands']:.3e} | {v['gap_ands']:.3e} "
                         f"({row['t1']:.3%}) | {v['allowance_ands']:.3e} | {v['shaspeed_ands']:.3e} "
                         f"({row['hash_units_per_mac']:.0f}x) | {v['shaspeed_real_ands']:.3e} "
                         f"({v['shaspeed_real_ands'] / v['stock_ands']:.2f}x) | {v['shaspeed_slack_ands']:.3e} | "
                         f"{v['host_route_ands']:.3e} ({row['host_route']:.2f}x) |")
    return lines


def unsettled_premises(card: dict, rows: list[dict]) -> list[str]:
    taken = {p for r in rows for p in r["premises"]}
    return [p for p, v in card["premise"].items() if p in taken and (v.get("status") in UNSETTLED + ("scratch",)
            or (v.get("status") == "assumed" and p not in card["space"]["base"]))]


def if_refuted(card: dict, rows: list[dict]) -> list[tuple[str, dict | None]]:
    """The best design if each unsettled premise (open, scratch, or assumed beyond the base) is refuted, and if all of
    them are but the hashing credit and PeriodInventory. Refuting a premise refuses exactly the designs that take it
    and changes no other row, so this filters the ranking."""
    uns = unsettled_premises(card, rows)
    out = []
    keep = set(HASH_CREDIT) | {"period-inventory", "host-row"}
    for drop in [[p] for p in uns] + [[p for p in uns if p not in keep]]:
        best = next((r for r in rows if not set(drop) & set(r["premises"])), None)
        out.append((drop[0] if len(drop) == 1 else "every unsettled premise but the hashing credit and PeriodInventory "
                    "with its host row", best))
    return out


def if_attack_refuted(card: dict, rows: list[dict]) -> list[tuple[str, dict | None]]:
    """The best T1-passing design if each open attack on a passing design is refuted alone, if every attack that needs
    the prover to choose the weights is (the model fixed, the open model-choice question), and if all of them are: each
    ranked again with the refuted attacks gone, so an attacked design can overtake an unattacked one."""
    hit = sorted({a for r in rows if r["t1_pass"] for a in r["attacks"]})

    def best(gone: set[str]) -> dict | None:
        pool = [r for r in rows if r["t1_pass"] and set(r["attacks"]) <= gone]
        return min(pool, key=lambda r: rank_key(r | {"attacks": [a for a in r["attacks"] if a not in gone]}), default=None)

    out = [(a, best({a})) for a in hit]
    cw = {a for a in hit if card["attack"][a].get("needs") == "chosen weights"}
    if len(cw) > 1 or cw and len(hit) > 1:
        out.append((MODEL_FIXED, best(cw)))
    if len(hit) > 1:
        out.append(("every open attack", best(set(hit))))
    return out


def leaner(card: dict, rows: list[dict], best: dict) -> list[tuple[str, dict]]:
    """For each unsettled premise the best row takes beyond the base, the first T1-passing row whose premises are the
    best's without it (or fewer), under no attack the best isn't under: what the headline's T1 costs without it."""
    base, prem = set(card["space"]["base"]), card["premise"]
    out = []
    for p in best["premises"]:
        if p in base or prem[p]["status"] not in UNSETTLED + ("scratch",):
            continue
        b = next((r for r in rows if r["t1_pass"] and p not in r["premises"]
                  and set(r["premises"]) <= set(best["premises"]) and set(r["attacks"]) <= set(best["attacks"])), None)
        if b is not None:
            out.append((p, b))
    return out


def alternates(rows: list[dict]) -> dict[str, dict | None]:
    """The best row free of open attacks that meets T1 and T2 at any rank, and the best if every unmeasured T3 came in
    under its target."""
    hopeful = sorted(rows, key=lambda r: rank_key(r | {"t3_pass": r["t3_pass"] or r["t3_worst"] is None}))
    return {"t1_t2_any_rank": next((r for r in rows if r["t1_pass"] and r["t2_pass"] and not r["attacks"]), None),
            "if_unmeasured_t3_passes": hopeful[0] if hopeful else None}


def joint(results: dict, points: tuple[str, ...]) -> list[dict]:
    """Designs priced at every one of `points`, keyed by their worst: T1 passing at all of them, then rank, T2, T3."""
    by = [{r["design"]: r for r in results[p]["rows"]} for p in points if p in results]
    if len(by) != len(points):
        return []
    out = []
    for name in sorted(set.intersection(*(set(b) for b in by))):
        rs = [b[name] for b in by]
        t3s = [r["t3_worst"] for r in rs]
        out.append({"design": name, "rows": rs, "t1_pass": all(r["t1_pass"] for r in rs),
                    "t1": max(r["t1"] for r in rs), "t2": max(r["t2"]["judged"] for r in rs),
                    "t3": None if any(t is None for t in t3s) else max(t3s),
                    "rank_effective": rs[0]["rank_effective"], "rank": rs[0]["rank"], "attacks": rs[0]["attacks"]})
    return sorted(out, key=lambda j: (not j["t1_pass"], bool(j["attacks"]), j["rank_effective"], j["rank"],
                                      j["t2"] > TARGET["t2"], j["t3"] is None or j["t3"] > TARGET["t3"], j["t1"], j["t2"]))


def pct(x: float | None) -> str:
    return "?" if x is None else f"{x:.2%}"


def t2s(row: dict) -> str:
    e = row["t2"]["engines"]
    return f"{e['sync']['ratio']:.1f}x / {e['async']['ratio']:.1f}x"


def row_line(i, row: dict) -> str:
    t = row["terms"]
    return (f"| {i} | {row['design']} | {rk(row)} | {rk(row, True)} | **{pct(row['t1'])}** "
            f"{'pass' if row['t1_pass'] else 'fail'} | {copy_models(row)} | {pct(row['t1_coins_uncredited'])} | {pct(t['final_word'])} | "
            f"{pct(t['uncommitted_steps'])} | {pct(t['coin_branch'])} | {pct(t['forming'])} | {pct(t['lane_fix'])} | "
            f"{pct(t['noise_derivation'])} | "
            f"{pct(t['denoise'])} | {pct(t['hiding'])} ({pct(row['hiding_closed'])}) | {pct(t['host'])} "
            f"({row['host_route']:.2f}x) | {t2s(row)} "
            f"({row['t2']['unfused']:.1f}x) {'pass' if row['t2_pass'] else 'fail'} | {pct(row['t3_worst'])} | "
            f"{row['delta_star']:.1e} | {net(row)} | {len(row['open'])}: {opened(row)} | {', '.join(row['attacks']) or '-'} |")


def net(row: dict) -> str:
    """The network term (DitherInstance's egress slack over stock) at both caps, outside the core T1: streamed coins or
    the certifier's leaf salt bound it, and nothing bounds a seeded design whose salt the prover holds."""
    if row["network_streamed"] is None:
        return "n/a (one matmul; the term is a forward's)"
    def two(a, b):
        return " / ".join(f"{v:.0%}" if v >= 0.1 else f"{v:.2%}" for v in (a, b))
    if row["network"] is not None:
        return f"{two(row['network'], row['network_alt'])} ({row['network_bound']})"
    return f"unbounded, seeded ({two(row['network_streamed'], row['network_streamed_alt'])} if streamed)"


def copy_models(row: dict) -> str:
    """T1 under bare model P (no measured floors: N-ADAPT's masked cross-row copies, where priced) and under model C."""
    return f"{pct(row['t1_bare_p'])} / {pct(row['t1_model_c'])}"


HEADER = ["| # | design | rank m/h (overall) | effective | T1 (P + floors) | bare P / model C | +coins | final word | uncommitted | "
          "coin branch | forming | lane fix | derivation | de-noise | hiding (closed) | host (route H) | "
          "T2 sync / async (unfused) | T3 | delta* | network, cap 32 / 320 (not in T1) | open (matmul) | open attacks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def render(card: dict, results: dict, space_info: dict) -> str:
    lines = ["# PoUW design space ranking (pouw_space.py)", "",
             "T1: additive gap over stock in W1 units (H32 floor 32 a register, measured on the H200, run "
             "r20261029-021534-a5a3); terms are shares of stock; '+coins' is T1 with the coins' expansion uncredited. "
             "T2: t_H/t_S judged on the faster stock step (unfused: the words exported to HBM). T3: worst of "
             "WikiText-2 2K and 8K over stock int8 (unmeasured fails). Rank: matmul part / hashing part (overall), "
             "nominal (the premises' own ranks) and effective (an open premise ranks 5). Hiding is required (NIGHT §1 "
             "default 2): hiding-ruled is out of the space. Designs are named "
             "N/D/F/density/operands/rows/hiding[/P: fw final-word lemma, count the counting lemma, min "
             "OneInstructionMinimal].", "",
             f"The space: {space_info['designs']:,} compatible compositions of {space_info['modules']} modules "
             f"({space_info['incompatible']:,} incompatible choices skipped); modules left out because a premise they "
             f"take is refuted: " + ("; ".join(f"{k} ({'; '.join(v)[:120]})" for k, v in space_info["dropped"].items())
                                     or "none") + ".", ""]
    for point, res in results.items():
        if point.startswith("_"):
            continue
        pt = card["point"][point]
        lines += [f"## {pt['title']}", "",
                  f"{len(res['rows']):,} designs priced, {len(res['refused']):,} refused"
                  + (f" (every one: {next(iter(res['refused'].values()))[0]})" if not res["rows"] and res["refused"] else "")
                  + ".", ""]
        if point in card.get("cover", {}):
            cv = card["cover"][point]
            lines += ["The whole step, not only the linears this card prices (T1 over every component; estimates; "
                      + cv["source"] + "): " + "; ".join(
                          f"{what} " + ", ".join(f"{k} {pct(v)}" for k, v in cv[key].items())
                          for key, what in (("step_t1", "step T1"), ("uncovered", "W1 no target covers"),
                                            ("t3_added", "T3 added")) if key in cv) + ".", ""]
        if not res["rows"]:
            continue
        for key, what in (("rank_effective", "effective rank (open premises at 5)"), ("rank", "nominal rank (open premises settled at their own rank)")):
            lines += [f"Pareto frontier of the T1-passing designs over (T1, T2, T3), at each {what}:", "",
                      "| up to rank | design | rank m/h (overall) | T1 (P + floors) | bare P / model C | +coins | "
                      "network, cap 32 / 320 (not in T1) | T2 sync / async | T3 | open |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            frs = res["frontier_effective" if key == "rank_effective" else "frontier_nominal"]
            for lv, fr in frs:
                for r in sorted(fr, key=lambda r: r["t2"]["judged"])[:FRONTIER_SHOWN]:
                    lines.append(f"| {LABEL[lv]} | {r['design']} | {rk(r, key == 'rank_effective')} | {pct(r['t1'])} | "
                                 f"{copy_models(r)} | {pct(r['t1_coins_uncredited'])} | {net(r)} | {t2s(r)} | "
                                 f"{pct(r['t3_worst'])} | {opened(r)} |")
                if len(fr) > FRONTIER_SHOWN:
                    lines.append(f"| {LABEL[lv]} | ({len(fr) - FRONTIER_SHOWN} more on the frontier; the JSON has them) | | | | | | | | |")
            if not frs:
                lines.append("| - | no design passes T1 free of open attacks | | | | | | | | |")
            lines.append("")
        lines += [f"The top {SHOWN} by the ranking (the CSV has every row):", ""] + HEADER
        for i, row in enumerate(res["rows"][:SHOWN], 1):
            lines.append(row_line(i, row))
        reasons: dict[str, int] = {}
        for v in res["refused"].values():
            k = v[0].split(":")[0][:90]
            reasons[k] = reasons.get(k, 0) + 1
        lines += ["", "Refused, by first reason:", ""] + [f"- {k}: {c:,}" for k, c in sorted(reasons.items(), key=lambda kv: -kv[1])]
        best = res["rows"][0]
        p = best["t2"]["extra_parts_ms"]
        if best["attacks"]:
            lines += ["", f"**No design ranks first** (NIGHT §5: a design with an open finding can't): every design "
                          f"{'passing T1' if best['t1_pass'] else 'priced'} here carries an open attack. First by the "
                          f"ranking otherwise, under {', '.join(best['attacks'])}:"]
        lines += ["", f"**{'Best' if not best['attacks'] else 'First, attacked'}: {best['design']}** ({best['title']}). "
                      f"T2's parts a forward: stock "
                      f"{best['t2']['engines']['async']['stock_ms']:.2f} ms (async), "
                      + ", ".join(f"{k} {v:.2f} ms" for k, v in p.items())
                      + f", hashing {best['t2']['hash_ms']:.2f} ms. It still needs:", ""]
        lines += [f"- {n}" for n in needs(best)]
        lean = leaner(card, res["rows"], best)
        if lean:
            lines += ["", "Without each premise it takes beyond the base (the same attacks or fewer):", "",
                      "| without | design | rank (overall) | T1 | T2 sync / async | T3 |", "|---|---|---|---|---|---|"]
            lines += [f"| {p} | {b['design']} | {rk(b)} | {pct(b['t1'])} | {t2s(b)} | {pct(b['t3_worst'])} |"
                      for p, b in lean]
        if res["if_attack_refuted"]:
            lines += ["", "If an open attack is refuted:", "",
                      "| refuted | best then | rank (overall) | T1 | T2 sync / async | T3 | still open |",
                      "|---|---|---|---|---|---|---|"]
            for a, b in res["if_attack_refuted"]:
                lines.append(f"| {a} | - | - | - | - | - | - |" if b is None else
                             f"| {a} | {b['design']} | {rk(b)} | {pct(b['t1'])} | {t2s(b)} | {pct(b['t3_worst'])} | "
                             f"{opened(b)} |")
        lines += ["", "If a premise fails:", "", "| refuted | best then | rank (overall) | T1 | T2 sync / async | T3 | still open |",
                  "|---|---|---|---|---|---|---|"]
        for drop, b in res["if_refuted"]:
            lines.append(f"| {drop} | - | - | - | - | - | - |" if b is None else
                         f"| {drop} | {b['design']} | {rk(b)} | {pct(b['t1'])} {'pass' if b['t1_pass'] else 'fail'} | "
                         f"{t2s(b)} | {pct(b['t3_worst'])} | {opened(b)} |")
        alt = alternates(res["rows"])
        if alt["t1_t2_any_rank"] is None:
            lines += ["", "No design meets T1 and T2 together at any rank."]
        for what, b in alt.items():
            if b is not None and b is not best:
                lines += ["", f"Best {'meeting T1 and T2 at any rank' if what == 't1_t2_any_rank' else 'if every unmeasured T3 passes'}"
                              f": {b['design']}, rank {rk(b)}, T1 {pct(b['t1'])}, T2 {t2s(b)}, T3 "
                              f"{pct(b['t3_worst'])}, open: {opened(b)}."]
        lines += [""] + period_table(card, best) + [""]
    for title, js in results.get("_joint", {}).items():
        lines += [f"## Joint: {title}", "", "| # | design | rank (overall, nominal / effective) | worst T1 | worst T2 | worst T3 |",
                  "|---|---|---|---|---|---|"]
        for i, j in enumerate(js[:10], 1):
            lines.append(f"| {i} | {j['design']} | {LABEL[j['rank']]} / {LABEL[j['rank_effective']]} | {pct(j['t1'])} "
                         f"{'pass' if j['t1_pass'] else 'fail'} | {j['t2']:.1f}x | {pct(j['t3'])} |")
        lines.append("")
    lines += ["## Cards", "", "| card | origin | point | design | T1 | gap | claims |", "|---|---|---|---|---|---|---|"]
    for cid, per in results.get("_cards", {}).items():
        for point, v in per.items():
            if isinstance(v, list):
                lines.append(f"| {cid} | {card['card'][cid].get('origin', '')} | {point} | refused | - | - | {'; '.join(v)[:160]} |")
            else:
                lines.append(f"| {cid} | {card['card'][cid].get('origin', '')} | {point} | {v['design']} | {pct(v['t1'])} | "
                             f"{pct(v['gap'])} | {v.get('claim_text', '')} |")
    return "\n".join(lines)


CLAIM_BASES = ("gap", "t1", "hiding_words", *TERMS)


def claim_value(row: dict, basis: str, terms: list[str] | None = None) -> float:
    if basis == "gap":
        return sum(row["terms"][k] for k in (terms or GAP_TERMS))
    if basis == "t1":
        return row["t1"]
    if basis == "hiding_words":
        return row["hiding_words"]
    if basis in TERMS:
        return row["terms"][basis]
    raise CardError(f"claim basis {basis!r} is not one of {CLAIM_BASES}")


def main() -> int:
    checks = Checks()
    args = checks.args
    card_path = Path(args[args.index("--card") + 1]) if "--card" in args else CARD
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(".")
    delta = float(args[args.index("--delta") + 1]) if "--delta" in args else 0.0
    adds = [Path(args[i + 1]) for i, a in enumerate(args) if a == "--add"]
    card = load(card_path, adds)
    designs, incompatible, dropped = space(card)
    cds = card_designs(card)
    n_modules = sum(len(modules(card, a, s)) for a, s in card["space"]["slots"])
    space_info = {"designs": len(designs), "incompatible": incompatible, "dropped": dropped, "modules": n_modules}
    results: dict = {}
    for point in card["point"]:
        rows, refused = ranking(card, designs, point, delta)
        results[point] = {"rows": rows, "refused": refused, "if_refuted": if_refuted(card, rows),
                          "if_attack_refuted": if_attack_refuted(card, rows),
                          "frontier_effective": frontiers(rows, "rank_effective"),
                          "frontier_nominal": frontiers(rows, "rank")}
    results["_joint"] = {"Llama-3.2-1B at T 512 (64 and 512 rows) and T 8,192": joint(results, ("t512", "t512r512", "t8192")),
                         "V4 dense, V4-Pro and its experts": joint(results, ("v4like", "v4pro", "v4experts"))}
    card_rows: dict = {}
    for cid, d in cds.items():
        card_rows[cid] = {}
        claims = card["card"][cid].get("claims", {})
        for point in (claims or {"t8192": None}):
            if isinstance(d, Refused):
                card_rows[cid][point] = d.reasons
                continue
            cl = claims.get(point) or {}
            try:
                card_rows[cid][point] = evaluate(card, d | {"salt_basis": cl.get("salt_basis", "w1"),
                                                             "data_model": cl.get("data_model", "m0")}, point, delta)
            except Refused as e:
                card_rows[cid][point] = e.reasons
    results["_cards"] = card_rows
    points = [p for p in card["point"]]
    for point in points:
        res = results[point]
        print(f"== {card['point'][point]['title']} ({len(res['rows']):,} priced, {len(res['refused']):,} refused)")
        for i, row in enumerate(res["rows"][:8], 1):
            print(f"  {i:2d}. {row['design']:<52} {rk(row)} eff {LABEL[row['rank_effective']]}  T1 {row['t1']:7.3%}  "
                  f"T2 {t2s(row)}  T3 {pct(row['t3_worst'])}  open {opened(row)}"
                  + (f"  ATTACKED {', '.join(row['attacks'])}" if row["attacks"] else ""))
        for lv, fr in res["frontier_nominal"]:
            print(f"   frontier at nominal rank <= {LABEL[lv]}: " + ", ".join(
                f"{r['design']} ({pct(r['t1'])}, {r['t2']['judged']:.1f}x, {pct(r['t3_worst'])})"
                for r in sorted(fr, key=lambda r: r["t2"]["judged"])[:4]) + (f" +{len(fr) - 4}" if len(fr) > 4 else ""))

    def row(point: str, name: str) -> dict:
        return next(r for r in results[point]["rows"] if r["design"] == name)

    def cardrow(cid: str, point: str = "t8192") -> dict:
        v = card_rows[cid].get(point)
        if v is None or isinstance(v, list):
            d = cds[cid]
            v = evaluate(card, d, point, delta)
            card_rows[cid][point] = v
        return v

    def nm(*parts: str) -> str:
        return "/".join(parts[:7]) + ("/" + "+".join(parts[7:]) if parts[7:] else "")

    # (1) The cards' claims: tonight's calculators reproduced on their own designs.
    for cid, c in card.get("card", {}).items():
        for point, cl in c.get("claims", {}).items():
            v = card_rows[cid][point]
            if isinstance(v, list):
                checks.check(f"card {cid} at {point} is priced", False, "; ".join(v)[:120])
                continue
            for basis in CLAIM_BASES:
                if basis in cl:
                    got = claim_value(v, basis, cl.get("terms"))
                    ok = abs(got - cl[basis]) <= cl["tol"]
                    v["claim_text"] = (v.get("claim_text", "") + f" {basis} {cl[basis]:.4g} ({c.get('origin')}) vs "
                                       f"{got:.4g}{' ok' if ok else ' MISMATCH'};")
                    checks.check(f"card {cid} ({c.get('origin')}): {basis} at {point} reproduces its {cl[basis]:.4g} "
                                 f"within {cl['tol']:g} ({cl['source']})", ok, f"{got:.5g}")

    # (2) The figures tonight's calculator reproduced, on the same designs.
    for point, want in (("t8192", 0.0210), ("t512", 0.393), ("t512r512", 0.0649)):
        r = cardrow("ncp-k32", point)
        got = r["terms"]["final_word"] + r["terms"]["forming"]
        checks.check(f"ncp-k32 at {point}: final word + forming reproduce CORE-NCP §7's fused figure {want:.2%}",
                     abs(got - want) < 0.0006, f"{got:.3%}")
    r = cardrow("ncp-k32")
    checks.check("the final-word term over Llama-3.2-1B's linears is 32·Σn/ΣnK = 1.239% (CORE-TIME final)",
                 abs(r["terms"]["final_word"] - 32 * 23552 / 60817408) < 1e-12, f"{r['terms']['final_word']:.4%}")
    for point, want in (("t8192", (154.8, 186.4)), ("t512", (16.8, 33.4)), ("t512r512", (92.5, 162.4))):
        e = cardrow("ncp-k32", point)["t2"]["core_ncp"]
        got = (e["sync"], e["async"])
        checks.check(f"ncp-k32's T2 on CORE-NCP's parts at {point} reproduces CORE-NCP §6's {want[0]}x / {want[1]}x",
                     all(abs(g / w - 1) < 0.01 for g, w in zip(got, want)), f"{got[0]:.1f}x / {got[1]:.1f}x")
    r = cardrow("ncp-k32")
    checks.check("ncp-k32 hashes 0.375 B of k32 words a useful MAC (12/32, CORE-NCP §2)",
                 abs(r["t2"]["bytes_forward"] / r["stock_macs_forward"] - (0.375 + 14336 / 60817408)) < 1e-9,
                 f"{r['t2']['bytes_forward'] / r['stock_macs_forward']:.4f} B (words 0.375 and A's commitment)")
    bar = card["price"]["hash_work_bar"]["value"] / card["price"]["hash_work"]["value"]
    checks.check("hashing A's commitment before the noise is 0.21x stock at T 8,192 at ShaLB's count, CORE-SMELL C28's "
                 "0.42x at the bar's issue-time price, credited under the atomic call (SW-SMELL P3), not left as gap",
                 0.40 < r["pre_noise_units"] * bar < 0.44 and 0.20 < r["pre_noise_units"] < 0.22
                 and r["terms"]["pre_noise_hashing"] == 0, f"{r['pre_noise_units']:.3f}x ({r['pre_noise_units'] * bar:.3f}x at the bar)")
    checks.check("binding (salt, forward, position) into each row digest's prefix block costs one compression a row: at "
                 "16 KiB rows 1/128 of the credited hashing (N-COMPOSE: 1/130 of a row's hash, framing included), and "
                 "it closes the repeated-row branch", 1 / 132 < r["row_bound_share"] < 1 / 124
                 and r["terms"]["row_repeat"] == 0, f"{r['row_bound_share']:.3%} ({1 / r['row_bound_share']:.0f} to one)")
    checks.check("deriving E1 and F1 from XOF bytes is 8/n + 8/m: 0.29% fused at T 8,192 (CORE-SMELL C28)",
                 abs(r["terms"]["noise_derivation"] - 0.0029) < 0.0002, f"{r['terms']['noise_derivation']:.3%}")
    grid = {"data_model": "grid"}
    sh = evaluate(card, designs[nm("ncp", "cancel", "int7-U", "k32", "xyef", "r16k", "mask", "fw")] | grid, "t8192")
    rest = sum(v for k, v in sh["terms"].items() if k != "hashing_slack")
    checks.check("with every honest write forced and the salt side closed, credited hashing is about 346x stock at "
                 "T 8,192 at ShaLB's count, CORE-SMELL C27's 670x at the bar's issue-time price, and on the grid "
                 "delta* spends what route U's lane fix leaves of 1%",
                 660 < sh["hash_units_per_mac"] * bar < 700 and abs(sh["delta_star"] * sh["hash_units_per_mac"] + rest - 0.01)
                 < 1e-12, f"delta* {sh['delta_star']:.2e}, {sh['hash_units_per_mac']:.0f} units a MAC, other terms {rest:.3%}")
    lead = row("t8192", nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "mask", "min"))
    served = lead["f_host"] * (lead["hash_bytes_per_mac"] * card["price"]["hash_work_floor"]["value"]
                               + lead["host_route"] / lead["f_host"] - lead["hash_units_per_mac"])
    checks.check("route H on ssn_1l at T 8,192 is N-FLOOR final's: the host hashes 4.04% of the rows and computes their "
                 "words, 4.83x stock at the served committer's 908 W1 a byte and 4.71x at the best leaf (here at the "
                 "chain form's 883.0); period-inventory with host-row closes it, so it is no T1 term",
                 abs(lead["f_host"] - 0.0404) < 0.0001 and abs(served / 4.83 - 1) < 0.01
                 and abs(lead["host_route"] / 4.71 - 1) < 0.01 and lead["terms"]["host"] == 0,
                 f"f_host {lead['f_host']:.3%}, {served:.2f}x at 908, {lead['host_route']:.2f}x at 3,532")
    s1l = row("t8192", nm("ssn-1l", "served", "s8", "k32", "w", "r256k", "mask", "min"))
    over = 2 / 3532 * s1l["hash_units_per_mac"]
    checks.check("crediting 3,534 to a chained compression (N-SHALB interim 1's count) would credit 2 registers a "
                 "compression no program writes: on ssn_1l at T 8,192, 6.5% of stock (N-SHALB interim 2), so the card "
                 "credits the chain form's 3,532 and the prefix block's constant-chaining-value form at 3,510",
                 abs(over / 0.065 - 1) < 0.08 and card["price"]["hash_work"]["value"] == 3532 * 32 / 128
                 and card["price"]["hash_work_iv"]["value"] == 3510 * 32 / 128,
                 f"{s1l['hash_units_per_mac']:.1f} W1 a MAC of hashing, {over:.2%} of stock")
    alt = cardrow("np-alt-r1b")
    at_alt = lead["network_streamed"] * 119.7 / lead["w_mac"]
    checks.check("the network term is N-ALT final's: on the tagged lead at T 8,192 gate_up's 349,525 MACs a coin bit, and "
                 "N-ALT's 137.6% of stock at its W_mac of 119.7 (908 W1 a byte) is this card's at the card's W_mac; a "
                 "design whose coins are seeded from the window's salt has no bound, and N-ALT's R1b, its coins streamed "
                 "under certifier-coins, has one; a point that is one matmul (V4-like) has no forward's egress to divide",
                 abs(lead["macs_per_coin_bit"] - 349525.33) < 1 and abs(at_alt / 1.376 - 1) < 0.005
                 and lead["network"] is None and alt["coins_streamed"] and alt["network"] is not None
                 and all(r["network_streamed"] is None for r in results["v4like"]["rows"]),
                 f"{lead['macs_per_coin_bit']:,.0f} MACs a coin bit, W_mac {lead['w_mac']:.1f}, "
                 f"{lead['network_streamed']:.1%} ({at_alt:.1%} at 119.7); R1b {alt['network']:.1%}")
    epi = {p: row(p, nm("ssn-1l-tag", "epilogue", "s8", "k32", "w", "r256k", "cert", "min"))
           for p in ("t512", "t512r512", "t8192")}
    fw = 32 * 23552 / 60817408
    checks.check("N-EXACT's epilogue fold on the tagged lead (code and tags cancelled in one more k32 step, the separable "
                 "row term removed by the IADD3 and FFMA) serves stock's int8 product, T3 0, but its final word is then "
                 "fixed by r_i alone on chosen identical rows, so the coin branch copies its repeats: under 1% at 64 "
                 "rows only, and at T 8,192 nearly the whole final word",
                 all(r["t3_worst"] == 0 and r["terms"]["coin_branch"] > 0.99 * r["gap"] for r in epi.values())
                 and epi["t512"]["t1_pass"] and not epi["t512r512"]["t1_pass"] and not epi["t8192"]["t1_pass"]
                 and 0.99 * fw < epi["t8192"]["terms"]["coin_branch"] < fw,
                 ", ".join(f"{p} T1 {r['t1']:.3%} (branch {r['terms']['coin_branch']:.3%})" for p, r in epi.items()))
    for point, want in (("t8192", 0.024), ("t512", 0.520)):
        t = cardrow("ncp-k32", point)["terms"]
        got = t["final_word"] + t["forming"] + t["noise_derivation"]
        checks.check(f"on settled premises ncp-k32's final word, forming and derivation are SW-SMELL P3's {want:.1%} at "
                     f"{point}", abs(got - want) < 0.001, f"{got:.2%}")
    lu = row("t512", nm("ncp", "cancel", "int7-U", "k32", "w", "r16k", "open"))
    lm = row("t8192", nm("ncp", "cancel", "int7-mix", "k32", "w", "r16k", "open"))
    checks.check("route U's Y lane fix is 24/m (SW-SMELL S9): 37.5% of stock at 64 rows; the mixed route stays under route "
                 "U's 0.293% at T 8,192", abs(lu["terms"]["lane_fix"] - 0.375) < 1e-12
                 and lm["terms"]["lane_fix"] <= 24 / 8192 + 1e-12, f"{lu['terms']['lane_fix']:.2%}, mix {lm['terms']['lane_fix']:.3%}")
    hb = card["price"]["extract"]["units_per_clock"] * 1980e6 / card["price"]["hash_rate"]["value"]
    checks.check("T2's hash rate at 16 KiB rows is SW-GPU final's measured 2,043-2,159 units a byte at 1,980 MHz alone "
                 "(r20261029-042705-c421, -045936-22a0, -050641-3a03, -051426-adb8)", 2043 <= hb <= 2159, f"{hb:.0f}")
    checks.check("coinsncp's interpolation returns the measured times at measured sizes, and the smallest call's time below",
                 abs(coinsncp_ms(card, 16777216) - 0.061657) < 1e-12 and coinsncp_ms(card, 1000) == 0.011485, None)
    a = call_units(card, designs[nm("ncp", "cancel", "int7-U", "k32", "w", "r256k", "open")], 64, 3072, 2048, 1, 510.48e9)
    checks.check("A's root at 64 rows by activation rows: 16 + 1 + 2 + 3·6 + 1 = 38 compressions before the noise",
                 a["a_chain"] == 38, f"{a['a_chain']:g}")

    # (3) The space itself.
    ncp = [n for n, d in designs.items() if d["choice"]["noise"] == "ncp"]
    hmods = modules(card, "H", "hiding")
    hidings = [h for h, m in hmods.items() if f"H.hiding.{h}" not in dropped and m.get("enumerate", True)]
    pinned = [h for h in hidings if "when" in hmods[h]]
    checks.check("the space holds, for NCP, every density (k32, and 6 L's under each of 2 credits) x 3 formats (int7 "
                 "route U, the cheaper route, rotated) x 6 row layouts (3 leaf sizes salted every leaf, N-HIDE's l1, l2f "
                 "and l2w) x each hiding module the card and its added files hold (R1's at one layout, the certifier's "
                 "rows) x 2 final-word charges x the 9 (operands, forming, derivation) choices a charge doesn't "
                 "duplicate: 13 x 3 x (6 x H + R) x 2 x 9; hiding-ruled is out (default 2)",
                 len(ncp) == 13 * 3 * (6 * (len(hidings) - len(pinned)) + len(pinned)) * 2 * 9 and "H.hiding.hr" in dropped,
                 f"{len(ncp):,} NCP designs, {len(hidings)} hiding modules ({', '.join(hidings)}); dropped {sorted(dropped)}")
    w = [call_units(card, designs[nm("ssn-1l", "served", "s8", "k32", "w", "r16k", "open")], 64, 2048, K) for K in (2048, 8192)]
    got = [(u["r"], u["coin_bytes"] / card["price"]["coin_bytes"]["value"] * 8 / 64) for u in w]
    checks.check("ssn-1l's stacked width and coin bits a row a call are SW-NOISE's: 2,144/2,048 and 3 whole coin words "
                 "(96 bits) at K 2,048, 8,480/8,192 and 9 words (288 bits) at K 8,192 (ssn_1l_layout)",
                 all(abs(g[0] - x[0]) < 1e-12 and abs(g[1] - x[1]) < 1e-9
                     for g, x in zip(got, ((2144 / 2048, 96), (8480 / 8192, 288)))), f"{got}")
    pw, px = (evaluate(card, designs[nm("per2", "cancel", "act124", "k32", o, "r16k", "mask", "fw")] | grid, "t8192")
              for o in ("w", "xyef"))
    checks.check("per2 under the final-word lemma on the grid with the salt side closed is SW-NOISE's 0.190% plus its "
                 "words' leftover hiding, and 0 gap with X and V committed", abs(pw["gap"] - 0.0019) < 0.00003 and px["gap"] < 1e-12,
                 f"gap {pw['gap']:.4%}, with C1 {px['gap']:.4%}")
    b64 = row("t8192", nm("ncp", "cancel", "int7-U", "k64-bam", "xyef", "r16k", "open", "fw"))
    checks.check("a word every 64 steps at rank <= 3 is credited BamFloor's 128/3 units, not 64 (N-DENSE): "
                 "r(1 - (128/3)/64) = 100% of stock uncommitted, so it fails T1",
                 abs(b64["terms"]["uncommitted_steps"] - 1.0) < 1e-9
                 and not b64["t1_pass"] and b64["ranks"]["matmul"] == 3, f"{b64['terms']['uncommitted_steps']:.2%}")
    shares = {L: row("t8192", nm("ncp", "cancel", "int7-U", f"k{L}-bam", "xyef", "r16k", "open", "fw"))["terms"]["uncommitted_steps"] / 3
              for L in (64, 128, 256)}
    checks.check("the uncredited share a word every L under BamFloor is N-DENSE's corrected floor (SW-SEM's 64 - 1024/L "
                 "counted pre-images and merge inputs twice): 1/3, 0.625 and 0.8125 at L 64, 128 and 256",
                 all(abs(shares[L] - w) < 1e-9 for L, w in ((64, 1 / 3), (128, 0.625), (256, 0.8125))), f"{shares}")
    checks.check("the 05:05Z best (on the k64 merge charge) is refused: SW-OLD final refutes it",
                 isinstance(cds["ncp-k64-merge-0505"], Refused) or any(
                     "refuted" in x for v in card_rows["ncp-k64-merge-0505"].values() if isinstance(v, list) for x in v),
                 str(card_rows["ncp-k64-merge-0505"].get("t8192"))[:90])
    s256, s63, s126 = (fast_saving(card, L, 8192, 8192, om) for L, om in ((256, 63), (512, 63), (512, 126)))
    checks.check("Strassen-Winograd on operands within +-63 saves nothing against the k32 counting at L 256 "
                 "(1/8 - 56/256 < 0) and 1/8 - 56/512 - 8(2/8192) = 1.37% at L 512, m = n = 8,192; on NCP's +-126 "
                 "operands its sums leave s8 and it spends 18/8 of the steps, saving nothing (P1's floor)",
                 s256[0] <= 0 and abs(s63[0] - (1 / 8 - 56 / 512 - 16 / 8192)) < 1e-12 and s63[1] == 1 and s126[0] == 0,
                 f"+-63: L 256 {s256[0]:.2%}, L 512 {s63[0]:.3%}; +-126: L 512 {s126[0]:.2%}")
    r64, r128, r256 = (row("t8192", nm("ncp", "cancel", "int7-U", f"k{L}-pl", "xyef", "r16k", "open", "fw")) for L in (64, 128, 256))
    checks.check("crediting a word every L > 32 with L units rests on PureLanes or CellPure, rank 4b (SW-SEM final), and "
                 "only to L 128: at L 256 the word is credited BamFloor's 48 units and 3(1 - 48/256) = 243.75% stays",
                 r64["ranks"]["matmul"] == 4.5 and r128["terms"]["uncommitted_steps"] == 0
                 and r64["terms"]["uncommitted_steps"] == 0
                 and abs(r256["terms"]["uncommitted_steps"] - 3 * (1 - 48 / 256)) < 1e-9,
                 f"k64 rank {rk(r64)}, k256 uncommitted {r256['terms']['uncommitted_steps']:.1%}")
    t = row("t8192", nm("ncp", "cancel", "int7-U", "k32", "w", "r16k", "mask", "fw"))
    checks.check("ranks split by part: ShaWrites (ShaSpeed's form its theorem) and mask-optimal are 4a on the hashing "
                 "part, W1 (card-within-accel-w1) and PeriodInventory 3 on the matmul part, and the overall rank is the "
                 "larger; PeriodInventory "
                 "is open, so the effective rank is 5", t["ranks"]["hashing"] == 4.0 and t["ranks"]["matmul"] == 3.0
                 and t["rank"] == 4.0 and card["premise"]["ShaSpeed"]["rank"] == 0
                 and t["rank_effective"] == 5.0 and RANKS["4a"] < RANKS["4bU"] < RANKS["4b"] < RANKS[5], rk(t))
    llama = ("t512", "t512r512", "t8192")
    exact = {p: [r for r in results[p]["rows"] if r["choice"]["denoise"] == "cancel"] for p in llama}
    checks.check("in M0 every exact design (the noise cancelled in the GEMM) fails T1 at every Llama point: N-SAKU's "
                 "skip frees a segment a dot product whatever the premise, at L 32 the 32·Σn/ΣnK = 1.239% the final-word "
                 "lemma closes only on the grid", all(exact.values()) and all(
                     not r["t1_pass"] and r["terms"]["final_word"] >= 32 * 23552 / 60817408 - 1e-12
                     for v in exact.values() for r in v),
                 ", ".join(f"{p} {len(v):,} exact, least T1 {min(r['t1'] for r in v):.2%}" for p, v in exact.items()))
    gu = evaluate(card, designs[nm("ncp", "cancel", "int7-U", "k32", "w", "r16k", "open")] | grid, "t512")
    gs = evaluate(card, designs[nm("ncp", "cancel", "int7-U", "k32", "w", "r256k", "salt", "fw")] | grid, "t8192")
    checks.check("on the grid NCP at 64 rows still needs route U' (route U's Y lane fix is 37.5%, SW-SMELL S9), and at "
                 "T 8,192 with a salt every 256 KiB leaf the salt side alone is over 1% (N-HIDE §2)",
                 gu["terms"]["lane_fix"] > 0.01 and gs["terms"]["hiding"] > 0.01,
                 f"lane fix {gu['terms']['lane_fix']:.1%}, salt side {gs['terms']['hiding']:.1%}")
    ok = True
    for p, res in results.items():
        if p.startswith("_"):
            continue
        for key, rkey in (("frontier_effective", "rank_effective"), ("frontier_nominal", "rank")):
            for lv, fr in res[key]:
                pool = [r for r in res["rows"] if r["t1_pass"] and not r["attacks"] and r[rkey] <= lv]
                fo = [objectives(r) for r in fr]
                ok &= all(not any(all(a <= b for a, b in zip(objectives(q), o)) and objectives(q) != o for q in pool) for o in fo)
                ok &= all(any(all(a <= b for a, b in zip(f, objectives(q))) for f in fo) for q in pool)
    checks.check("each frontier is exactly the non-dominated set: no frontier design is beaten on T1, T2 and T3 at once "
                 "by a design at its rank, and every passing design at that rank is matched or beaten by one", ok, None)
    checks.check("decode (16 rows a call) is refused for every design as outside the bound's domain (C15), not priced",
                 not results["decode"]["rows"] and all("domain" in v[0] for v in results["decode"]["refused"].values()),
                 f"{len(results['decode']['refused']):,} refused")

    try:
        merge(copy.deepcopy(card), {"price": {"reg": {"value": 16}}}, "added.toml")
        merged = "merged"
    except CardError as e:
        merged = str(e)
    checks.check("an added card file adds and never overrides: one that redefines price.reg.value is a card error",
                 "already defined" in merged, merged[:80])

    # (4) Refusals: what an unpriced term does.
    for name in ("dither-c", "fresh-u", "accel-d5-d6"):
        checks.check(f"card {name} is refused, never ranked: no N module prices it",
                     isinstance(cds[name], Refused), str(cds[name])[:90])
    checks.check("every ranked row has all of T1's terms priced, and a source for each design",
                 all(set(r["terms"]) == set(TERMS) for p, res in results.items() if not p.startswith("_")
                     for r in res["rows"]) and all(d["source"] for d in designs.values()), f"{len(designs):,} designs")
    ssn = [n for n, d in designs.items() if d["choice"]["noise"] in ("ssn-1l", "ssn-a")]
    checks.check("SSN is refused on V4-Pro, whose wq_b (n 65,536) is past its column code's 28,559 columns "
                 "(SW-LEAN's ssnCode_inj), and priced on V4-like (n <= 16,384) where its other terms allow",
                 all("noise's domain" in results["v4pro"]["refused"][n][0] for n in ssn if n in results["v4pro"]["refused"])
                 and not any(r["choice"]["noise"] in ("ssn-1l", "ssn-a") for r in results["v4pro"]["rows"])
                 and any(r["choice"]["noise"] == "ssn-1l" for r in results["v4like"]["rows"]), None)
    checks.check("V4-Pro's routed experts (128 rows a call at 8,192 tokens) are inside the domain",
                 bool(results["v4experts"]["rows"]), f"{len(results['v4experts']['rows']):,} rows")
    s1 = {p: [r for r in results[p]["rows"] if r["choice"]["noise"] == "ssn-1l"] for p in ("t512", "t512r512", "t8192")}
    checks.check("with the coin branch priced (N-RED A1: a prover that branches on the coins copies each repeated row's "
                 "words), ssn-1l fails T1 at every Llama point and chosen-rows attacks it",
                 all(s1.values()) and all(not r["t1_pass"] and "chosen-rows" in r["attacks"] for v in s1.values() for r in v),
                 ", ".join(f"{p} min branch {min(r['terms']['coin_branch'] for r in v):.1%}" for p, v in s1.items() if v))
    a1 = {("nn-prmt-sv4", "t512"): 0.188824, ("nn-prmt-sv4", "t512r512"): 0.659851, ("nn-prmt-sv4", "t8192"): 1.00252,
          ("nn-prmt-sv4-aorb", "t8192"): 0.954505}
    got = {k: cardrow(*k)["terms"]["coin_branch"] for k in a1}
    sv28 = [cardrow("nn-rot2-sv28-aorb", p)["terms"]["coin_branch"] for p in ("t512", "t512r512", "t8192")]
    checks.check("N-NOISE's untagged sv4 codes collapse on chosen identical owners as ssn-1l's code does: the card's "
                 "coin branch reproduces its a1_chosen at 64, 512 and 8,192 rows, side B on identical weight columns, "
                 "and sv28's 2^28 states leave under 1e-6; its attack entries, naming modules as N.<id>, attack them",
                 all(abs(got[k] - w) < 1e-5 for k, w in a1.items()) and max(sv28) < 1e-6
                 and "nn-a1-a2-chosen" in cardrow("nn-prmt-sv4-aorb")["attacks"]
                 and "nn-sv28-tile" in cardrow("nn-rot2-sv28-aorb")["attacks"],
                 ", ".join(f"{k[0]} {k[1]} {v:.4%}" for k, v in got.items()) + f", sv28 {max(sv28):.1e}")
    tag = designs[nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "cert", "min")]
    xs = calls(card, "t8192")
    a2c = sum(x["count"] * x["m"] * x["n"] * (tag["fields"].at(x["m"], x["n"], x["K"])["S"] - 1) * 16
              for x in xs) / sum(x["count"] * x["m"] * x["n"] * x["K"] for x in xs)
    checks.check("N-RED's A2c on the tagged lead, a row repeating its previous step's vector at each later step whose "
                 "coin bit is 0, is (S - 1)·16/K of stock on the card's S: its 51.57% at m 8,192 (checked word for word "
                 "at m 384); it needs chosen weights, so it is an open attack, not a T1 term",
                 abs(a2c - 0.5157) < 0.0005 and card["attack"]["a2c-cancel-lane"]["needs"] == "chosen weights",
                 f"{a2c:.2%}")
    cw = {"fix2-quadrant", "a2c-cancel-lane", "walk"}
    refuted = {p: dict(results[p]["if_attack_refuted"]) for p in llama}
    blc = lambda r: r["choice"]["noise"].startswith("tag-blc")  # noqa: E731
    lane = {(nz, p): row(p, nm(nz, "served", "s8", "k32", "w", "r256k", "cert", "min"))
            for nz in ("tag-blc1a-t128", "tag-blc1a-t64") for p in ("t512", "t8192")}
    r1 = lambda r: r["choice"]["hiding"] in ("r1", "r1-db")  # noqa: E731
    first_off_r1 = {p: next(r for r in results[p]["rows"] if not r1(r)) for p in llama}
    checks.check("on Llama R1 with the certifier's check (b) ranks first (NIGHT §5), and N-EGRESS's coin lanes first off "
                 "R1: (b) closes A2c, A8 and the walk within an entry at the honest's revisit cost, and a later-step "
                 "weight the certifier streams after A_j leaves them a residual the coin branch prices (p* a step, model "
                 "P), neither with an open attack; every tagged design with neither carries fix2-quadrant, A2c and the "
                 "walk, each needing chosen weights, and with the model fixed (model-public) a tagged design with "
                 "neither ranks first again; blc1a-t128 fails at 64 rows a call, where half its tile's later steps carry "
                 "the public lane, and blc1a-t64 passes there; on V4-like an unattacked per2 ranks first",
                 all(results[p]["rows"][0]["choice"]["hiding"] == "r1-db" and results[p]["rows"][0]["t1_pass"]
                     and not results[p]["rows"][0]["attacks"] and blc(first_off_r1[p]) and first_off_r1[p]["t1_pass"]
                     and not first_off_r1[p]["attacks"] for p in llama)
                 and all(cw <= set(r["attacks"]) for p in llama for r in results[p]["rows"]
                         if r["t1_pass"] and r["choice"]["noise"] == "ssn-1l-tag" and r["choice"]["hiding"] != "r1-db")
                 and all(v.get(MODEL_FIXED) is not None and not blc(v[MODEL_FIXED])
                         and "tag" in v[MODEL_FIXED]["choice"]["noise"] for v in refuted.values())
                 and not lane[("tag-blc1a-t128", "t512")]["t1_pass"] and lane[("tag-blc1a-t128", "t8192")]["t1_pass"]
                 and lane[("tag-blc1a-t64", "t512")]["t1_pass"]
                 and not results["v4like"]["rows"][0]["attacks"] and results["v4like"]["rows"][0]["choice"]["noise"] == "per2",
                 ", ".join(f"{p} {results[p]['rows'][0]['design']}, off R1 {first_off_r1[p]['design']}, fixed "
                           f"{v[MODEL_FIXED]['design'] if v.get(MODEL_FIXED) else 'none'}" for p, v in refuted.items())
                 + f"; blc1a-t128 at 64 rows {lane[('tag-blc1a-t128', 't512')]['t1']:.1%}"
                 + f"; v4like {results['v4like']['rows'][0]['design']}")
    lead = results["t512"]["rows"][0]
    lean = dict(leaner(card, results["t512"]["rows"], lead))
    oim = lean.get("one-instruction-minimal")
    tag0 = row("t512", nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "cert"))
    checks.check("the T 512 lead takes one-instruction-minimal, which it doesn't need to pass: without it the tagged "
                 "lead's noise derivation is 0.048% (N-SMELL interim 2 item 7's 0.0483%), and a coin lane's own loads add "
                 "to it where it has them",
                 "one-instruction-minimal" in lead["premises"] and oim is not None and oim["t1_pass"]
                 and 0.00045 < tag0["t1"] < 0.00052,
                 f"{oim['design']} {oim['t1']:.4%}; tag lead {tag0['t1']:.4%}" if oim else "none")
    xs = calls(card, "t8192")
    b1, b4 = (row("t8192", nm(nz, "served", "s8", "k32", "w", "r256k", "cert", "min")) for nz in
              ("tag-blc1a-t128", "tag-blc4a-t64"))
    b1n = row("t8192", nm("tag-blc1a-t128", "served", "s8", "k32", "w", "r256k", "cert"))
    tagn = row("t8192", nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "cert"))

    def residual(nz: str) -> float:
        f = designs[nm(nz, "served", "s8", "k32", "w", "r256k", "cert", "min")]["fields"]
        return sum(x["count"] * x["m"] * x["n"] * (f.at(x["m"], x["n"], x["K"])["S"] - 1)
                   * f.at(x["m"], x["n"], x["K"])["blc_p"] * 32 for x in xs) / sum(
            x["count"] * x["m"] * x["n"] * x["K"] for x in xs)

    loads = b1n["terms"]["noise_derivation"] - tagn["terms"]["noise_derivation"]
    checks.check("N-EGRESS's coin lanes reproduce its arithmetic at T 8,192: blc1a-t128's 550 MACs a coin bit, network "
                 "term 0.217% at 3,532 (2.17% at 320 bits) and lane loads 0.209% uncredited; blc4a-t64's 0.36% at 320 "
                 "bits (0.37% at the floor); each A2c/A8 residual is (S - 1)·p*·32/K on its own S (0.415% and 0.334%, "
                 "where N-EGRESS's 0.40% and 0.30% scale the lead's S)",
                 abs(b1["macs_per_coin_bit"] / 550 - 1) < 0.005 and abs(b1["network_streamed"] - 0.00217) < 0.00005
                 and abs(b1["network_streamed_alt"] - 0.0217) < 0.0005 and abs(loads - 0.00209) < 0.00005
                 and 0.0035 < b4["network_streamed_alt"] < 0.0038
                 and abs(b1["terms"]["coin_branch"] - residual("tag-blc1a-t128")) < 1e-9
                 and abs(b4["terms"]["coin_branch"] - residual("tag-blc4a-t64")) < 1e-9,
                 f"{b1['macs_per_coin_bit']:,.1f} MACs a bit, {b1['network_streamed']:.4%} / {b1['network_streamed_alt']:.3%}, "
                 f"loads {loads:.4%}; blc4a {b4['network_streamed_alt']:.3%}; residuals {b1['terms']['coin_branch']:.4%} / "
                 f"{b4['terms']['coin_branch']:.4%}")
    salted = {lf: row("t8192", nm("ssn-1l-tag", "served", "s8", "k32", "w", lf, "cert", "min")) for lf in ("r256k", "r16k")}
    masked = row("t8192", nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "mask", "min"))
    red = {"r256k": (3886, 0.0153), "r16k": (245, 0.00097)}
    checks.check("N-RED final §3's leaf bound: the certifier's fresh salt a leaf (certifier-salt) bounds n_max at a leaf's "
                 "MACs over 512 whatever the coins, 3,886 and 1.53% at cap 32 and W_mac 119.7 for a 256 KiB leaf, 245 and "
                 "0.097% for 16 KiB, within 2% (this card takes the point's largest MACs a word, down's); a prover-held "
                 "salt (mask) bounds nothing, and the coin lanes' streamed 85 MACs a bit is the smaller bound on blc4a",
                 all(abs(salted[lf]["macs_per_salt_bit"] / n - 1) < 0.02 and salted[lf]["network_bound"] == "salt"
                     and abs(salted[lf]["network"] * 119.7 / salted[lf]["w_mac"] / v - 1) < 0.02
                     for lf, (n, v) in red.items())
                 and masked["network"] is None and b4["network_bound"] == "streamed",
                 ", ".join(f"{lf} {r['macs_per_salt_bit']:,.0f} MACs a salt bit, {r['network']:.3%} / {r['network_alt']:.2%}"
                           f" ({r['network'] * 119.7 / r['w_mac']:.3%} at 119.7)" for lf, r in salted.items()))
    r1n = {p: row(p, nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "r1")) for p in llama}
    r1hw = row("t8192", nm("ssn-1l-hw", "served", "s8", "k32", "w", "r256k", "r1"))
    r1b = {p: row(p, nm("ssn-1l-tag", "served", "s8", "k32", "w", "r256k", "r1-db")) for p in llama}
    alt_t2 = {"t512": 97.5, "t512r512": 568.9, "t8192": 665.3}
    link = {p: r["t2"]["extra_parts_ms"]["link"] / r["t2"]["engines"]["async"]["stock_ms"] for p, r in r1n.items()}
    dl = {k: ev(card["H"]["hiding"]["r1-db"]["fields"]["distinct_loss"], K=k) for k in (2048, 8192)}
    checks.check("R1 is N-ALT final's: the prover hashes nothing, its link time over the async stock step is 97.5 / 568.9 / "
                 "665.3 at 400 Gb/s within 0.5%, T1 on the tagged lead is SW-NOISE's derivation share (0.048%) and on "
                 "hw 0.197% at T 8,192 within 2%, its network term the words' way back, 3.3e-8 of stock within 5%; "
                 "the check (b) variant adds N-RED's 0.086% / 0.196% at K 2,048 / 8,192 and takes no open attack, "
                 "where R1 alone keeps fix2-quadrant, A2c and the walk",
                 all(r["hash_units_per_mac"] == 0 and r["terms"]["hiding"] == 0 and abs(link[p] / alt_t2[p] - 1) < 0.005
                     and abs(r["t1"] - 0.000483) < 0.000002 and cw <= set(r["attacks"]) for p, r in r1n.items())
                 and abs(r1hw["t1"] / 0.00197 - 1) < 0.02 and abs(r1n["t8192"]["network"] / 3.29e-8 - 1) < 0.05
                 and abs(dl[2048] - 0.00086) < 1e-9 and abs(dl[8192] / 0.00196 - 1) < 0.005
                 and all(not r["attacks"] and r["terms"]["distinct_loss"] > 0 for r in r1b.values()),
                 ", ".join(f"{p} link {link[p]:.1f}, T2 {r['t2']['judged']:.1f}" for p, r in r1n.items())
                 + f"; T1 {r1n['t512']['t1']:.4%}, hw {r1hw['t1']:.3%}; network {r1n['t8192']['network']:.2e}; "
                   f"(b) {r1b['t512']['terms']['distinct_loss']:.4%} ({dl[2048]:.4%} / {dl[8192]:.4%})")
    b4p = row("t512", nm("tag-blc4a-t64", "served", "s8", "k32", "w", "r256k", "cert", "min"))
    b1p = row("t512", nm("tag-blc1a-t128", "served", "s8", "k32", "w", "r256k", "cert", "min"))
    half = sum(x["count"] * x["m"] * x["n"] * 32 * (lambda g: (g["S"] - 1 - g["blc_steps"]) / 2)(
        designs[b1p["design"]]["fields"].at(x["m"], x["n"], x["K"])) for x in calls(card, "t512")) / b1p["stock_macs_forward"]
    checks.check("model C against model P (N-RED final §4): under CopyGrain(4) the coin lanes' steps leave nothing, so "
                 "blc4a-t64's T1 drops by its whole coin branch; a 128-row tile on 64 rows keeps A2c's ½ on its "
                 "public-lane steps; a design without coin lanes prices the same under both",
                 abs(b4p["t1_model_c"] - (b4p["t1"] - b4p["terms"]["coin_branch"])) < 1e-12 and b4p["terms"]["coin_branch"] > 0
                 and abs(b1p["t1_model_c"] - (b1p["t1"] - b1p["terms"]["coin_branch"] + half)) < 1e-9 and half > 0.25
                 and tagn["t1_model_c"] == tagn["t1"] and masked["t1_model_c"] == masked["t1"],
                 f"blc4a-t64 {b4p['t1']:.3%} P, {b4p['t1_model_c']:.3%} C; blc1a-t128 at 64 rows {b1p['t1']:.2%} P, "
                 f"{b1p['t1_model_c']:.2%} C")
    noisy = [n for n, d in designs.items() if d["choice"]["noise"] in ("nsf", "perm")]
    checks.check("a noise whose words repeat on data (nsf, perm) is refused while its share is unpriced",
                 noisy and all(n in results["t8192"]["refused"] and "dup" in results["t8192"]["refused"][n][0] for n in noisy),
                 results["t8192"]["refused"].get(noisy[0], ["priced"])[0][:80] if noisy else "none")

    if checks.mutations:
        def mutated(fn) -> dict:
            c = copy.deepcopy(card)
            fn(c)
            return c

        def one(c: dict, choice: dict, point: str = "t8192", model: str = "m0"):
            try:
                return evaluate(c, compose(c, choice) | {"data_model": model}, point)
            except Refused as e:
                return e.reasons

        base = dict(cds["ncp-k32"]["choice"])
        c = mutated(lambda c: c["N"]["ncp"]["D"]["cancel"]["fields"].pop("formed_y"))
        v = one(c, base)
        checks.mutation("an NCP card without its weight-side forming count is refused, not priced at 0",
                        isinstance(v, list) and "formed_y" in v[0], str(v)[:80])
        c = mutated(lambda c: c["N"]["ncp"]["D"]["cancel"]["fields"].pop("fix_U"))
        checks.mutation("an NCP card without route U's lane fix leaves NCP out of the int7 formats, not priced at 0",
                        not any(d["choice"]["noise"] == "ncp" for d in space(c)[0].values()), None)
        c = mutated(lambda c: c["space"]["fields"].__setitem__("word_bits", 16))
        v = one(c, base)
        checks.mutation("a 16-bit committed word is refused: no premise prices it", isinstance(v, list) and "16-bit" in v[0], str(v)[:60])
        c = mutated(lambda c: c["premise"]["commit-forced"].__setitem__("status", "refuted"))
        checks.mutation("a refuted commit-forced (SW-LEAN's counterexample) leaves every committed-operand module out of "
                        "the space", all(d["choice"]["operands"] == "w" for d in space(c)[0].values()), None)
        lc = copy.deepcopy(card)
        lenient = {cid: compose(lc, lc["card"][cid]["modules"] | {"rows": "r256k", "hiding": "mask"},
                                lc["card"][cid].get("premises", []), lenient=True, source="lenient")
                   for cid in ("dither-c", "fresh-u", "accel-d5-d6")}
        top = [cid for cid, d in lenient.items() if evaluate(lc, d, "t8192")["t1_pass"]]
        checks.mutation("read every unpriced field as 0 and CORE-RANK's unpriced designs (hidden at the cheapest "
                        "layout) enter T1's passing tier: the refusal is what keeps them out", bool(top),
                        f"passing tier {top}")
        nosem = cardrow("ncp-k64-nosem")
        checks.mutation("committing every 64 steps without the k32 semantics leaves r(1 - 32/64) = 150% of stock "
                        "uncommitted: the density needs candidate 2", nosem["terms"]["uncommitted_steps"] > 1.4,
                        f"{nosem['terms']['uncommitted_steps']:.0%}")
        c = mutated(lambda c: c["N"]["ncp"]["D"]["cancel"]["fields"].update(formed_x="m*K", formed_y="n*K"))
        ch = cds["ncp-k32"]["choice"]
        a1, b1 = one(c, ch)["terms"]["forming"], cardrow("ncp-k32")["terms"]["forming"]
        checks.mutation("forming one copy a side would be a third of NCP's 24/n + 24/m: the stack's 3 copies decide the term",
                        abs(3 * a1 - b1) < 1e-12 and b1 > 0.0085, f"{a1:.3%} vs {b1:.3%}")
        d1 = evaluate(card, designs[nm("ncp", "cancel", "int7-U", "k32", "xyef", "r16k", "mask", "fw")], "t8192", delta=1e-4)
        checks.mutation("a ShaSpeed slack of 0.01% breaks T1 for the all-forced design: hashing's work is 346x stock",
                        not d1["t1_pass"], f"T1 {d1['t1']:.2%}")
        c = mutated(lambda c: c["premise"]["pure-lanes"].__setitem__("word_units", "reg*L/32"))
        k1024 = {"noise": "ncp", "denoise": "cancel", "format": "int7-mix", "density": "k1024-pl", "operands": "xyef",
                 "rows": "r256k", "hiding": "mask", "final": "fw", "forming": "none", "derivation": "none"}
        e, e0 = one(c, k1024, "v4like", "grid"), one(card, k1024, "v4like", "grid")
        checks.mutation("credit PureLanes' full L at every L and ncp/k1024-pl/xyef passes T1 on V4-like on the grid: the "
                        "cap at L 128 from SW-SEM's margins keeps it out", e["t1_pass"] and not e0["t1_pass"],
                        f"T1 {e['t1']:.2%} uncapped, {e0['t1']:.0%} capped")
        ssn_e = {"noise": "ssn-1l", "denoise": "served", "format": "s8", "density": "k32", "operands": "ef",
                 "rows": "r16k", "hiding": "mask", "final": "none", "forming": "none", "derivation": "none"}
        c = mutated(lambda c: c["N"]["ssn-1l"]["fields"].__setitem__("derive_extra", "0"))
        e, e0 = one(c, ssn_e), one(card, ssn_e)
        checks.mutation("drop SSN's rotates and committing its noise registers would close the derivation to 0: the "
                        "rotate no commitment holds is what stays", e["terms"]["noise_derivation"] == 0
                        and e0["terms"]["noise_derivation"] > 0, f"{e0['terms']['noise_derivation']:.4%} with the rotates")
        c = mutated(lambda c: c["price"]["salt"].update(unforced=0, forced=0))
        sal = dict(ssn_e, noise="ssn-1l-tag", operands="w", hiding="salt", rows="r256k")
        e, e0 = one(c, sal), one(card, sal)
        checks.mutation("a card that prices the salt side at nothing lets ssn-1l-tag salted every 256 KiB leaf without "
                        "mask-optimal pass T1 at T 8,192: the hiding term is what keeps it out", e["t1_pass"] and not e0["t1_pass"],
                        f"T1 {e['t1']:.2%} vs {e0['t1']:.2%}")
        c = mutated(lambda c: c["N"]["ssn-1l"]["fields"].__setitem__("branch_rows", "0"))
        br = dict(ssn_e, operands="w", rows="r256k")
        e, e0 = one(c, br), one(card, br)
        checks.mutation("price ssn-1l's coin branch at nothing and it passes T1 at T 8,192 again: N-RED A1's copies are "
                        "what keep it out", e["t1_pass"] and not e0["t1_pass"] and e0["terms"]["coin_branch"] > 0.99,
                        f"T1 {e['t1']:.3%} vs {e0['t1']:.1%}")
        c = mutated(lambda c: c["N"]["ssn-1l-hw"]["fields"].__setitem__("branch_rows", "rep_bits(m, 4, S)"))
        hw = dict(ssn_e, noise="ssn-1l-hw", operands="w", rows="r256k")
        e, e0 = one(c, hw), one(card, hw)
        checks.mutation("give ssn-1l-hw's rows a 4-bit code where N-PREM's has 16 and its coin branch is over 1% at "
                        "T 8,192: the 16 bits a row are what close the branch", e["terms"]["coin_branch"] > 0.01
                        and e0["terms"]["coin_branch"] < 0.002,
                        f"{e['terms']['coin_branch']:.2%} vs {e0['terms']['coin_branch']:.3%}")
        c = mutated(lambda c: c["N"]["ssn-1l-tag"]["D"]["epilogue"]["fields"].__setitem__("branch_rows", "0"))
        ep = dict(ssn_e, noise="ssn-1l-tag", denoise="epilogue", operands="w", rows="r256k", hiding="cert",
                  derivation="min")
        e, e0 = one(c, ep), one(card, ep)
        checks.mutation("price the tag epilogue's final-word coin branch at nothing and it passes T1 at T 8,192 with T3 0: "
                        "the final word's repeats on r_i are what keep it out", e["t1_pass"] and not e0["t1_pass"],
                        f"T1 {e['t1']:.3%} vs {e0['t1']:.3%}")
        c = mutated(lambda c: c["N"]["tag-blc1a-t128"]["fields"].__setitem__("branch_rows", "m*(S - 1)*blc_p"))
        bl = dict(ssn_e, noise="tag-blc1a-t128", operands="w", rows="r256k", hiding="cert", derivation="min")
        e, e0 = one(c, bl, "t512"), one(card, bl, "t512")
        checks.mutation("price blc1a-t128's coin-free later steps at a 64-row call as if each carried a coin and it passes "
                        "T1 at T 512: the public lane in half its steps is what A2c takes there",
                        e["t1_pass"] and not e0["t1_pass"], f"T1 {e['t1']:.3%} vs {e0['t1']:.2%}")
        c = mutated(lambda c: c["N"]["ssn-1l"]["fields"].pop("n_max"))
        e = one(c, dict(ssn_e, rows="r256k"), "v4pro")
        checks.mutation("drop SSN's column-code limit and ssn-1l is priced on V4-Pro's 65,536-column wq_b, a distinctness "
                        "claim on a matmul its lemma doesn't cover", isinstance(e, dict), str(e)[:60])
        c = mutated(lambda c: c["premise"]["hiding-ruled"].__setitem__("status", "open"))
        hr = dict(ssn_e, noise="ssn-1l-tag", operands="w", hiding="hr", rows="r256k")
        e = one(c, hr)
        checks.mutation("restore the hiding ruling and ssn-1l-tag passes T1 at T 8,192 with the salt side waived: default 2 "
                        "is what keeps it out", isinstance(e, dict) and e["t1_pass"] and e["hiding_closed"] > TARGET["t1"],
                        f"T1 {e['t1']:.3%}, waived {e['hiding_closed']:.0%}" if isinstance(e, dict) else str(e)[:60])
        p2 = designs[nm("per2", "cancel", "act124", "k32", "w", "r256k", "mask", "fw", "count", "min")]["choice"]
        e, e0 = one(card, p2, "t8192", "grid"), one(card, p2)
        checks.mutation("price on the grid, where the final-word lemma holds, and per2/cancel under it passes T1 at "
                        "T 8,192: N-SAKU's skip in M0 is what keeps every exact design out", e["t1_pass"] and not e0["t1_pass"],
                        f"T1 {e['t1']:.3%} on the grid, {e0['t1']:.2%} in M0")
        c = mutated(lambda c: c["premise"]["period-inventory"].pop("closes"))
        tag = dict(ssn_e, noise="ssn-1l-tag", operands="w", rows="r256k", derivation="min")
        e, e0 = one(c, tag), one(card, tag)
        checks.mutation("drop HostsCounted from PeriodInventory and the lead fails T1 by route H, the host's hashing "
                        "and words off the card: the host term is what period-inventory and host-row close",
                        not e["t1_pass"] and e["terms"]["host"] > 4 and e0["t1_pass"],
                        f"T1 {e['t1']:.0%} vs {e0['t1']:.3%}")
        c = mutated(lambda c: c["premise"]["row-bound"].pop("closes"))
        e = one(c, tag)
        checks.mutation("leave the row digests unbound and the lead fails T1 in M0: padding, chosen repeated tokens and a "
                        "replayed request let a branching prover skip A's credited hashing (N-COMPOSE), about 0.21x stock "
                        "at T 8,192", not e["t1_pass"] and abs(e["terms"]["row_repeat"] - e["pre_noise_units"]) < 1e-9
                        and 0.20 < e["terms"]["row_repeat"] < 0.22 and e0["t1_pass"],
                        f"row repeat {e['terms']['row_repeat']:.3f}x, T1 {e['t1']:.1%}")
        pool = {k: v for k, v in designs.items() if v["choice"]["noise"] in ("ssn-1l-tag", "ssn-1l-hw")}
        clean = mutated(lambda c: c.__setitem__("attack", {}))
        c = mutated(lambda c: c.__setitem__("attack", {"x": {"targets": ["ssn-tag-distinct"], "status": "open", "source": "test"}}))
        rows_c, _ = ranking(c, pool, "t8192")
        rows_0, _ = ranking(clean, pool, "t8192")
        checks.mutation("an open attack on ssn-tag-distinct keeps the ssn-1l-tag designs, first without it, from ranking first",
                        rows_c and rows_0[0]["choice"]["noise"] == "ssn-1l-tag" and rows_c[0]["choice"]["noise"] != "ssn-1l-tag"
                        and any(r["attacks"] for r in rows_c), rows_c[0]["design"] if rows_c else "none")

    out.mkdir(parents=True, exist_ok=True)
    md = render(card, results, space_info)
    (out / "ranking.md").write_text(md + "\n")

    def slim(r: dict) -> dict:
        return {k: v for k, v in r.items() if k not in ("per_matmul",)}

    blob = {"schema": "np-card/result/v1", "card": {k: v for k, v in card.items() if k not in ("card",)},
            "space": space_info, "delta": delta,
            "points": {p: {"rows": [slim(r) for r in res["rows"][:60]], "rows_total": len(res["rows"]),
                           "refused_total": len(res["refused"]),
                           "frontier_effective": [{"rank": LABEL[lv], "designs": [slim(r) for r in fr]} for lv, fr in res["frontier_effective"]],
                           "frontier_nominal": [{"rank": LABEL[lv], "designs": [slim(r) for r in fr]} for lv, fr in res["frontier_nominal"]],
                           "best": res["rows"][0]["design"] if res["rows"] else None,
                           "best_needs": needs(res["rows"][0]) if res["rows"] else [],
                           "best_per_matmul": res["rows"][0]["per_matmul"] if res["rows"] else None,
                           "alternates": {k: b and {"design": b["design"], "rank": b["rank"], "t1": b["t1"],
                                                    "t2": b["t2"]["judged"], "t3": b["t3_worst"], "needs": needs(b)}
                                          for k, b in alternates(res["rows"]).items()},
                           "best_periods": periods(card, res["rows"][0]) if res["rows"] else None,
                           "ranks_first": bool(res["rows"]) and not res["rows"][0]["attacks"],
                           "if_refuted": [{"refuted": drop, "best": b and b["design"], "t1": b and b["t1"],
                                           "t2": b and b["t2"]["judged"], "open": b and [o["premise"] for o in b["open"]]}
                                          for drop, b in res["if_refuted"]],
                           "if_attack_refuted": [{"refuted": a, "best": b and b["design"], "t1": b and b["t1"],
                                                  "t2": b and b["t2"]["judged"], "rank": b and LABEL[b["rank"]]}
                                                 for a, b in res["if_attack_refuted"]],
                           "without": [{"without": p, "design": b["design"], "t1": b["t1"], "rank": LABEL[b["rank"]]}
                                       for p, b in (leaner(card, res["rows"], res["rows"][0]) if res["rows"] else [])]}
                       for p, res in results.items() if not p.startswith("_")},
            "joint": {k: [{x: y for x, y in j.items() if x != "rows"} for j in v[:20]] for k, v in results["_joint"].items()},
            "cards": {cid: {p: (v if isinstance(v, list) else slim(v)) for p, v in per.items()} for cid, per in card_rows.items()}}
    (out / "pouw_space.json").write_text(json.dumps(blob, indent=1, default=str) + "\n")
    with (out / "ranking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point", "position", "design", *(f"{a}.{s}" for a, s in card["space"]["slots"]), "rank_matmul",
                    "rank_hashing", "rank", "rank_effective", "t1", "t1_bare_p", "t1_model_c", "t1_coins_uncredited", *TERMS, "hiding_closed", "t2_sync", "t2_async", "t2_unfused", "t3_worst", "delta_star",
                    "hash_bytes_per_mac", "network", "network_alt", "network_bound", "network_streamed",
                    "network_streamed_alt", "coins_streamed", "open", "attacks"])
        for p, res in results.items():
            if p.startswith("_"):
                continue
            for i, r in enumerate(res["rows"], 1):
                e = r["t2"]["engines"]
                ch = r["choice"]
                w.writerow([p, i, r["design"], *(ch[s] for _, s in card["space"]["slots"]),
                            LABEL[r["ranks"]["matmul"]], LABEL[r["ranks"]["hashing"]], LABEL[r["rank"]],
                            LABEL[r["rank_effective"]], f"{r['t1']:.6g}",
                            "" if r["t1_bare_p"] is None else f"{r['t1_bare_p']:.6g}", f"{r['t1_model_c']:.6g}",
                            f"{r['t1_coins_uncredited']:.6g}",
                            *(f"{r['terms'][k]:.6g}" for k in TERMS), f"{r['hiding_closed']:.6g}",
                            f"{e['sync']['ratio']:.4g}", f"{e['async']['ratio']:.4g}", f"{r['t2']['unfused']:.4g}",
                            "" if r["t3_worst"] is None else f"{r['t3_worst']:.4g}",
                            f"{r['delta_star']:.4g}", f"{r['hash_bytes_per_mac']:.4g}",
                            "" if r["network"] is None else f"{r['network']:.4g}",
                            "" if r["network_alt"] is None else f"{r['network_alt']:.4g}", r["network_bound"] or "",
                            "" if r["network_streamed"] is None else f"{r['network_streamed']:.4g}",
                            "" if r["network_streamed_alt"] is None else f"{r['network_streamed_alt']:.4g}",
                            int(r["coins_streamed"]),
                            " ".join(o["premise"] for o in r["open"]), " ".join(r["attacks"])])
    print(f"wrote {out / 'ranking.md'}, {out / 'pouw_space.json'}, {out / 'ranking.csv'}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
