"""The credit difftest's cases: `credit.credited`'s, `credit.drawn`'s, `credit.outcome`'s, `credit.window`'s,
`credit.period_credit`'s and `credit.a_link`'s arguments.

A `credited` case is an honest served window read through its challenger's session (`pouw_rounds_cases.Session`): its
manifest rows (the serving process's call rows, each with its credit bit, a credited one with its z table's root), the
round log (each recorded round root and its coin), and the draw: `credit.draw_root` over them, recorded by the
challenger, then its draw coin opened after it; the auditor's ledger opens and ends the session for the window. Then one
mutation (`CREDIT_NAMES`) aimed at one refusal: a credit bit flipped after the draw coin with the draw root recomputed
(``changed``, RT4-INT-32's withdrawn verdict), a row changed likewise, a missing or malformed bit, a credited row with no
z root, the draw root left stale, no draw coin, a coin of another family, and the forms.

A `drawn` case is a draw coin, the draw's stream, registration and receipt, and a window's layout
(`circuit.h1window.program`'s arguments, `layout`): linears at one or two K, called step by step in a shuffled order, so
each K's stratum has ranges apart. Its mutations break each rule `credit.strata` mirrors from `h1window.program`, the
coin's and the arguments' forms, and ``budget`` draws at K = `credit.RECORD_K` from a population larger than it (each
stratum's k_s rounded up, so a unit or two past K).

An `outcome` case is a `drawn` case's arguments and one accepted proof outcome for each unit the coin draws; its
mutations drop one (``missing``), reject one (``failed``), repeat one, add an undrawn unit, prove the units of another
coin the prover chose (``chosen``), and break the forms.

A `window` case is a `credited` case with its layout, one call per manifest row, and an accepted proof of each unit its
draw coin draws under the served window's arguments; its mutations add a call the manifest lacks, prove units the prover
chose under its own stream, fail or drop a drawn unit, and pass a `credited` refusal through.

A `period` case is one to three such windows in one period of the auditor's ledger (`period-open`, each window's session
opened and ended, then the `period-close` listing them), presented in its close's order, a window of no period beside it
in the ledger. Its mutations (RT4-INT-42) leave the period unclosed or a window incomplete, drop, reorder or add a window,
present the window of no period among the period's, name a period never opened or no period, forge the ledger, refuse
one window, and break the forms.

An `a_link` case is a window's registered entries, among them one or two A values (``…/a/<K>``) whose trees are frame-v3
SHA-512 trees (`verity.ml.commitments.merkle.tree_levels`) of `steps` equal blocks, and its round log: each round's
`credit.cover` segments of every A value under `credit.round_root` of a random inner root. Its mutations register an A
tree other than the one the rounds committed (RT4-INT-34), rewrite a segment with or without its round root, leave a gap,
add a segment, misalign one, list an unregistered value, register none or one twice, and break the forms.

`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_credit.py` holds each
mutation to the refusal it aims at.
"""
from __future__ import annotations

import hashlib
import itertools
import random

import pouw_rounds_cases as rc
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import window as Wd
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

MODULES = ["q_proj", "k_proj", "o_proj", "down_proj", "attn·out"]


def _hex(rng: random.Random) -> str:
    return rng.randbytes(64).hex()


def _row(rng: random.Random, i: int) -> dict:
    """A manifest row as the serving process writes it (`Call.summary()` with its tables and bound), its credit bit
    among it."""
    bit = rng.random() < 0.7
    tables = {"a": _hex(rng), **({"z": _hex(rng)} if bit or rng.random() < 0.5 else {})}
    return {"index": i, "step": rng.randrange(4), "layer": rng.randrange(32), "module": rng.choice(MODULES),
            "served": rng.random() < 0.8, "rows": rng.randint(1, 64), "k": rng.choice([64, 4096]), "n": 4096,
            "base": rng.randrange(1 << 20), "units": rng.randint(1, 8), "leaves": [rng.randrange(-3, 1 << 40), None],
            "tables": tables, "bound": {"m": rng.randint(1, 1 << 16), "ok": True}, Cr.CREDIT: bit}


def _draw_root(d: dict) -> str:
    salt, served = bytes.fromhex(d["salt"]), d["served"]
    calls = Cr.calls_root(salt, d["calls"], None if served is None else bytes.fromhex(served))
    return Cr.draw_root(calls, bytes.fromhex(d["draw"]["receipts"]), d["rounds"]).hex()


def honest_credit(rng: random.Random, name: str = "honest", rows: int | None = None, sid: str = "s-1",
                  leaf: int = rc.RECORD_LEAF) -> tuple[dict, rc.Session]:
    """An honest served window of `rows` manifest rows (one to five when None) whose draw coin opened after its draw root
    (none for ``no-draw``), its session `sid` signing its record at `leaf`."""
    rounds = rng.choice([1, 2, 3])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    d = {"salt": _hex(rng), "calls": [_row(rng, i) for i in range(rng.randint(1, 5) if rows is None else rows)],
         "served": rng.choice([None, _hex(rng)]), "draw": {"receipts": _hex(rng)}}

    def draw_after(rec) -> bytes:
        d["rounds"] = [{"root": r.hex(), "coin": o.to_json()} for r, o in zip(rec.roots, rec.openings)]
        d["draw"]["root"] = _draw_root(d)
        return bytes.fromhex(d["draw"]["root"])

    s = rc.Session(rng, ch, [rng.randbytes(64) for _ in range(rounds)], salt=None, sid=sid, leaf=leaf,
                   draw=None if name == "no-draw" else draw_after)
    if name == "no-draw":
        d["draw"]["coin"] = ch.open(Wd.DRAW, 0, draw_after(s.rounds)).to_json()
    else:
        d["draw"]["coin"] = s.draw[0].to_json()
    return {"key": s.key(), "window": rc._window(rng), "session": s.files(), **d}, s


def _flip_bit(d: dict, rng: random.Random, to: bool | None = None) -> None:
    rows = [r for r in d["calls"] if to is None or r[Cr.CREDIT] != to] or d["calls"]
    row = rng.choice(rows)
    row[Cr.CREDIT] = not row[Cr.CREDIT] if to is None else to
    if row[Cr.CREDIT]:
        row["tables"].setdefault("z", _hex(rng))


def _mutate_credit(d: dict, s: rc.Session, name: str, rng: random.Random) -> None:
    calls, j = d["calls"], rng.randrange(len(d["calls"]))
    match name:
        case "changed":
            _flip_bit(d, rng)
            d["draw"]["root"] = _draw_root(d)
        case "withdrawn":
            _flip_bit(d, rng, to=False)
            d["draw"]["root"] = _draw_root(d)
        case "row-changed":
            calls[j]["tables"]["a"] = _hex(rng)
            d["draw"]["root"] = _draw_root(d)
        case "stale-root":
            _flip_bit(d, rng)
        case "no-bit":
            del calls[j][Cr.CREDIT]
        case "bit-form":
            calls[j][Cr.CREDIT] = rng.choice([1, 0, "true", None, [True]])
        case "no-z":
            calls[j][Cr.CREDIT] = True
            calls[j]["tables"] = rng.choice([{"a": _hex(rng)}, {"a": _hex(rng), "z": _hex(rng)[:-2]}, None,
                                             {"a": _hex(rng), "z": _hex(rng).upper()}])
        case "index":
            calls[j]["index"] = rng.choice([j + 1, str(j), None, True if j == 1 else -1])
        case "row-form":
            deep = 0
            for _ in range(Cr.DEPTH):
                deep = [deep]
            calls[j] = rng.choice([[1], "row", None, {**calls[j], "bound": deep}])
        case "calls-form":
            d["calls"] = rng.choice([None, {"0": calls}, "x"])
        case "salt-form":
            d["salt"] = rng.choice([None, d["salt"][:-2], d["salt"].upper(), 5])
        case "served-form":
            d["served"] = rng.choice(["", "ab", 5, _hex(rng).upper()])
        case "rounds-form":
            k = rng.randrange(len(d["rounds"]))
            d["rounds"] = rng.choice([None, "x", d["rounds"][:k] + [5] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "root": "zz"}] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "coin": None}] + d["rounds"][k + 1:]])
        case "draw-form":
            d["draw"] = rng.choice([None, {**d["draw"], "receipts": "zz"}, {k: v for k, v in d["draw"].items()
                                                                            if k != "receipts"}])
        case "no-draw":
            pass
        case "draw-coin":
            coin = d["draw"]["coin"]
            other = Wd.live_challenger(s.ch.commitment.epoch, len(d["rounds"]), rng.randbytes)
            d["draw"]["coin"] = rng.choice([{**coin, "salt": rc._flip(coin["salt"])}, None, {**coin, "index": 1},
                                            {**coin, "after": rc._flip(coin["after"])},
                                            other.open(Wd.DRAW, 0, bytes.fromhex(d["draw"]["root"])).to_json()])
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "H": 0}])
        case "forged-ledger":
            d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(1))
        case _:
            raise ValueError(name)


CREDIT_NAMES = ["changed", "withdrawn", "row-changed", "stale-root", "no-bit", "bit-form", "no-z", "index", "row-form",
                "calls-form", "salt-form", "served-form", "rounds-form", "draw-form", "no-draw", "draw-coin", "key-form",
                "forged-ledger"]
CREDIT_KEYS = ("key", "ledger", "window", "session", "salt", "calls", "served", "rounds", "draw")


def credit_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `credited` case: an honest window, then mutation `name`."""
    d, s = honest_credit(rng, name)
    if name != "honest":
        _mutate_credit(d, s, name, rng)
    if "ledger" not in d:
        d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(0))
    return {k: d[k] for k in CREDIT_KEYS}


def credit_args(d: dict) -> tuple:
    """`credit.credited`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in CREDIT_KEYS), rc.lms_verify)


LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def layout(rng: random.Random, big: bool = False) -> dict:
    """A served window's layout, `h1window.program`'s arguments: one to four linears at one or two K, each n whole blocks
    of the tile's C, called once a step over one to three steps (A block s at step s) in a shuffled order; ``big``, a
    population past `credit.RECORD_K`."""
    r, c = rng.choice([(1, 1), (2, 4), (4, 8), (8, 8), (16, 16)])
    ks = rng.sample([32, 64, 4096, 14336], rng.randint(1, 2))
    names = rng.sample(LINEARS, rng.randint(len(ks), 4))
    linears = [[name, ks[i] if i < len(ks) else rng.choice(ks), c * (rng.randint(1, 4) if not big else 64)]
               for i, name in enumerate(names)]
    calls = [[name, step] for step in range(rng.randint(1, 3)) for name in names]
    rng.shuffle(calls)
    m = r << (rng.randint(0, 2) if not big else 9)
    return {"calls": calls, "linears": linears, "m": m, "tile": [r, c]}


def _draw_args(rng: random.Random) -> dict:
    stream = rng.choice([_hex(rng), "h1/w-3", ""])
    receipt = rng.choice([{"receipts": _hex(rng)}, {}, {"a": "b", "pass": "x y~"}])
    return {"coin": _hex(rng), "stream": stream, "registration": _hex(rng), "receipt": receipt}


def _break_layout(lay: dict, rng: random.Random, name: str) -> dict:
    lin, (r, c) = lay["linears"], lay["tile"]
    match name:
        case "layout-form":
            return rng.choice([None, [], {**lay, "extra": 1}, {k: v for k, v in lay.items() if k != "m"},
                               {**lay, "tile": [r]}, {**lay, "tile": [0, c]}, {**lay, "m": -1}, {**lay, "m": True},
                               {**lay, "calls": []}, {**lay, "linears": []}, {**lay, "linears": lin + [["x", 32]]},
                               {**lay, "calls": lay["calls"] + [[lin[0][0], -1]]}, {**lay, "calls": [[5, 0]]}])
        case "named-twice":
            return {**lay, "linears": lin + [[lin[0][0], lin[0][1], lin[0][2]]]}
        case "unknown-linear":
            return {**lay, "calls": lay["calls"] + [["lm_head", 0]]}
        case "uncalled":
            return {**lay, "linears": lin + [["lm_head", lin[0][1], c]]}
        case "slot":
            return {**lay, "m": rng.choice([r * 3, r // 2 if r > 1 else 0, r * 2 + 1 if r > 1 else 3])}
        case "kn":
            bad = rng.choice([[lin[0][0], 0, lin[0][2]], [lin[0][0], lin[0][1], lin[0][2] + 1 if c > 1 else 0],
                              [lin[0][0], lin[0][1], 0]])
            return {**lay, "linears": [bad] + lin[1:]}
        case "blocks":
            return {**lay, "calls": lay["calls"] + [[lin[0][0], 1 + max(b for _l, b in lay["calls"]) + 1]]}
    raise ValueError(name)


def drawn_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `drawn` case: a draw coin's arguments and a window's layout, then mutation `name`."""
    d = {**_draw_args(rng), "layout": layout(rng, big=name == "budget")}
    match name:
        case "honest" | "budget":
            pass
        case "coin-form":
            d["coin"] = rng.choice([None, d["coin"][:-2], d["coin"].upper(), 5])
        case "args-form":
            k, bad = rng.choice([("stream", None), ("stream", 5), ("registration", "zz"),
                                 ("registration", d["registration"].upper()), ("receipt", None), ("receipt", ["a"]),
                                 ("receipt", {"a": 1}), ("receipt", {"a": 'quote"d'}), ("receipt", {"\\": "x"}),
                                 ("receipt", {"a": "é"})])
            d[k] = bad
        case _:
            d["layout"] = _break_layout(d["layout"], rng, name)
    return d


DRAWN_NAMES = ["budget", "coin-form", "args-form", "layout-form", "named-twice", "unknown-linear", "uncalled", "slot", "kn",
               "blocks"]
DRAWN_KEYS = ("coin", "stream", "registration", "receipt", "layout")


def _proofs(rng: random.Random, units: list[int]) -> list[dict]:
    proofs = [{"unit": u, "verdict": "accept"} for u in units]
    rng.shuffle(proofs)
    return proofs


def _mutate_proofs(proofs: list[dict], rng: random.Random, name: str, units: list[int]) -> list[dict]:
    j = rng.randrange(len(proofs))
    match name:
        case "missing":
            del proofs[j]
        case "failed":
            proofs[j]["verdict"] = "reject"
        case "twice":
            proofs.insert(rng.randrange(len(proofs) + 1), dict(proofs[j]))
        case "undrawn":
            picked = set(units)
            u = next(u for u in itertools.count(rng.randrange(1 << 20)) if u not in picked)
            proofs.insert(rng.randrange(len(proofs) + 1), {"unit": u, "verdict": "accept"})
        case "proofs-form":
            proofs = rng.choice([None, "x", proofs + [5], proofs[:j] + [{**proofs[j], "verdict": "maybe"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "why": "x"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "unit": -1}] + proofs[j + 1:]])
        case _:
            raise ValueError(name)
    return proofs


def _partial(rng: random.Random) -> tuple[dict, list[int]]:
    """A draw whose coin leaves some of the window's units undrawn, and its units."""
    while True:
        d = {**_draw_args(rng), "layout": layout(rng)}
        units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
        if len(units) < sum(hi - lo for s in Cr.strata(d["layout"])[0] for lo, hi in s["units"]):
            return d, units


def outcome_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `outcome` case: a draw's arguments and an accepted proof of each unit its coin draws, then mutation `name`."""
    if name == "chosen":
        d, units = _partial(rng)
        other = {**d, "coin": _hex(rng)}
        chosen, _ = Cr.drawn(*(other[k] for k in DRAWN_KEYS))
        return {**d, "proofs": _proofs(rng, chosen if chosen != units else units[1:] + [max(units) + 1])}
    d = {**_draw_args(rng), "layout": layout(rng)}
    units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
    proofs = _proofs(rng, units)
    return {**d, "proofs": proofs if name == "honest" else _mutate_proofs(proofs, rng, name, units)}


OUTCOME_NAMES = ["missing", "failed", "twice", "undrawn", "chosen", "proofs-form"]
OUTCOME_KEYS = (*DRAWN_KEYS, "proofs")


def _served_units(d: dict, stream: str | None = None) -> list[int]:
    """The units `d`'s draw coin draws under the served window's arguments (`credit.window`'s), or under `stream`."""
    coin = d["draw"]["coin"]["coin"]
    got, why = Cr.drawn(coin, d["window"]["root"] if stream is None else stream, d["draw"]["root"],
                        {"receipts": d["draw"]["receipts"]}, d["layout"])
    assert why is None, why
    return got


def honest_window(rng: random.Random, name: str = "honest", sid: str = "s-1", leaf: int = rc.RECORD_LEAF
                  ) -> tuple[dict, rc.Session]:
    """An honest served window (`honest_credit`) with its layout, one call per manifest row, and an accepted proof of
    each unit its draw coin draws."""
    lay = layout(rng)
    d, s = honest_credit(rng, name if name in CREDIT_NAMES else "honest", rows=len(lay["calls"]), sid=sid, leaf=leaf)
    d["layout"] = lay
    d["proofs"] = _proofs(rng, _served_units(d))
    return d, s


def window_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `window` case: an honest window, then mutation `name`."""
    credit = name.removeprefix("credit-")
    d, s = honest_window(rng, credit)
    match name:
        case "honest":
            pass
        case "manifest-count":
            lin = d["layout"]["linears"][0][0]
            d["layout"]["calls"].insert(rng.randrange(len(d["layout"]["calls"]) + 1), [lin, 0])
        case "chosen":
            d, s = honest_window(rng)
            while True:
                chosen = _served_units(d, stream=_hex(rng))
                if chosen != sorted(p["unit"] for p in d["proofs"]):
                    break
                d, s = honest_window(rng)
            d["proofs"] = _proofs(rng, chosen)
        case "failed" | "missing":
            d["proofs"] = _mutate_proofs(d["proofs"], rng, name, [])
        case "layout-form":
            d["layout"] = _break_layout(d["layout"], rng, "layout-form")
        case _ if credit in CREDIT_NAMES:
            _mutate_credit(d, s, credit, rng)
        case _:
            raise ValueError(name)
    if "ledger" not in d:
        d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(0))
    return {k: d[k] for k in WINDOW_KEYS}


WINDOW_NAMES = ["manifest-count", "chosen", "failed", "missing", "layout-form", "credit-changed", "credit-no-draw",
                "credit-forged-ledger"]
WINDOW_KEYS = ("key", "ledger", *Cr.WINDOW_KEYS)
#: `credit.window`'s arguments, in order
WINDOW_ARGS = ("key", "ledger", "window", "session", "salt", "calls", "served", "rounds", "draw", "layout", "proofs")


def window_args(d: dict) -> tuple:
    """`credit.window`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in WINDOW_ARGS), rc.lms_verify)


def _evidence(d: dict) -> dict:
    return {k: d[k] for k in Cr.WINDOW_KEYS}


def period_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `period` case: one to three honest windows in one period of the auditor's ledger, presented in its close's
    order, and a window of no period opened after the close; then mutation `name`."""
    n = rng.randint(2 if name == "order" else 1, 3)
    pname = rng.choice(rc.PERIODS)
    wins = [honest_window(rng, sid=f"s-{i + 1}", leaf=4 * i + 2) for i in range(n)]
    foreign = honest_window(rng, sid="o-1", leaf=4 * n + 3)
    pub = wins[0][1].pub
    ends = [{} for _ in wins]
    if name == "incomplete":
        ends[rng.randrange(n)] = {"complete": False}
    events = [("period-open", None, {"period": pname, "lane": rc.LANE, "by": rc.LANE, "cap": 4,
                                     "circuit": rng.choice(rc.CIRCUITS), "pinned": "ab" * 20,
                                     "key_sha512": hashlib.sha512(pub).hexdigest()})]
    for (d, s), end in zip(wins, ends):
        events += [("open", s.sid, rc._open(rng, d["window"], pub)), ("end", s.sid, rc._end(s, **end))]
    outcomes = [{"root": d["window"]["root"], "session": s.sid, "complete": end.get("complete", True)}
                for (d, s), end in zip(wins, ends)]
    events.append(("period-close", None, {"period": pname, "lane": rc.LANE, "by": rc.LANE, "windows": outcomes,
                                          "complete": all(w["complete"] for w in outcomes)}))
    fd, fs = foreign
    events += [("open", fs.sid, rc._open(rng, fd["window"], pub)), ("end", fs.sid, rc._end(fs))]
    leaves = [0] + [x for i in range(n) for x in (4 * i + 1, 4 * i + 4)] + [4 * n + 1, 4 * n + 2, 4 * n + 5]
    key = rc._key(1) if name == "forged-ledger" else rc._key(0)
    ledger = rc.ledger_lines(events if name != "no-close" else [e for e in events if e[0] != "period-close"], key,
                             leaves if name != "no-close" else [q for q, e in zip(leaves, events) if e[0] != "period-close"])
    windows = [_evidence(d) for d, _s in wins]
    out = {"key": wins[0][0]["key"], "ledger": ledger, "name": pname}
    i = rng.randrange(n)
    match name:
        case "honest" | "incomplete" | "no-close" | "forged-ledger":
            pass
        case "missing":
            del windows[i]
        case "order":
            i, j = rng.sample(range(n), 2)
            windows[i], windows[j] = windows[j], windows[i]
        case "extra":
            windows.append(_evidence(fd) if rng.random() < 0.5 else dict(windows[i]))
        case "foreign":
            windows.insert(rng.randrange(n), _evidence(fd))
        case "unopened":
            out["name"] = next(p for p in rc.PERIODS if p != pname)
        case "name":
            out["name"] = rng.choice([None, 5, [pname]])
        case "window-refused":
            windows[i]["proofs"] = _mutate_proofs(windows[i]["proofs"], rng, rng.choice(["failed", "missing"]), [])
        case "windows-form":
            windows = rng.choice([None, {"0": windows[0]}, windows[:i] + [5] + windows[i + 1:],
                                  windows[:i] + [{k: v for k, v in windows[i].items() if k != "proofs"}]
                                  + windows[i + 1:]])
        case "key-form":
            out["key"] = rng.choice([None, {**out["key"], "public": "00"}])
        case _:
            raise ValueError(name)
    return {**out, "windows": windows}


PERIOD_NAMES = ["no-close", "incomplete", "missing", "order", "extra", "foreign", "unopened", "name", "window-refused",
                "windows-form", "forged-ledger", "key-form"]
PERIOD_KEYS = ("key", "ledger", "name", "windows")


def _tree(rng: random.Random, count: int) -> tuple[str, list[tuple[bytes, ...]]]:
    """A frame-v3 SHA-512 tree over `count` leaf digests: its domain id and levels."""
    dom = CommitmentDomain(rng.randbytes(64), rng.randrange(-2, 4), RangeIndexedDomain(0, count), "sha512")
    return dom.domain_id.hex(), tree_levels(dom, [rng.randbytes(64) for _ in range(count)])


def _log(steps: int, values: dict, inners: list[str]) -> list[dict]:
    """The round log: each round's root over its inner root and its `cover` segments of every A value."""
    out = []
    for r in range(steps):
        segs = {}
        for v, (count, levels) in values.items():
            per = count // steps
            segs[v] = [[lv, i, levels[lv][i].hex()] for lv, i in Cr.cover(r * per, (r + 1) * per)]
        out.append({"inner": inners[r], "a": segs})
    return _reroot(out)


def _reroot(log: list[dict]) -> list[dict]:
    for r, x in enumerate(log):
        x["root"] = Cr.round_root(r, bytes.fromhex(x["inner"]), {v: [(lv, i, bytes.fromhex(h)) for lv, i, h in s]
                                                                  for v, s in x["a"].items()}).hex()
    return log


def a_link_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `a_link` case: a window's registered entries and round log, then mutation `name`."""
    steps = rng.choice([2, 3, 4] if name in ("blocks", "swapped") else [1, 2, 3, 4])
    window = f"h1/w-{rng.randrange(100)}"
    values, entries = {}, []
    for k in rng.sample([64, 4096, 14336], rng.randint(1, 2)):
        per = rng.choice([2, 3, 4, 5] if name == "swapped" else [1, 2, 3, 4, 5, 8])
        domain, levels = _tree(rng, steps * per)
        v = f"{window}/a/{k}"
        values[v] = (steps * per, levels)
        entries.append({"value": v, "root": levels[-1][0].hex(), "leaves": steps * per, "words": 96, "domain": domain})
    entries += [{"value": f"{window}/z/{rng.choice([64, 4096])}", "root": _hex(rng), "leaves": 3, "domain": _hex(rng)},
                {"value": f"{window}/a/x", "root": _hex(rng)}][:rng.randint(0, 2)]
    rng.shuffle(entries)
    log = _log(steps, values, [_hex(rng) for _ in range(steps)])
    a = [e for e in entries if Cr.is_a(e["value"])]
    e = rng.choice(a)
    v = e["value"]
    r = rng.randrange(steps)
    segs = log[r]["a"][v]
    match name:
        case "honest":
            pass
        case "untied":
            e["root"] = _tree(rng, e["leaves"])[1][-1][0].hex()
        case "late-rows":
            count, levels = values[v]
            domain, other = _tree(rng, count)
            e["root"], e["domain"] = other[-1][0].hex(), domain
        case "rewritten":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
            _reroot(log)
        case "round-root":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
        case "gap":
            del segs[rng.randrange(len(segs))]
            _reroot(log)
        case "extra":
            segs.append([0, values[v][0], _hex(rng)])
            _reroot(log)
        case "swapped":
            if len(segs) > 1:
                segs[0], segs[1] = segs[1], segs[0]
            else:
                segs[0] = [segs[0][0], segs[0][1] + 1, segs[0][2]]
            _reroot(log)
        case "misaligned":
            lv, i, h = segs[0]
            segs[0] = [lv + 1, i // 2, h] if lv + 1 < Cr.LEVELS else [lv, i + 1, h]
            _reroot(log)
        case "unknown":
            log[r]["a"][f"{window}/a/{rng.choice([1, 2, 3])}"] = [[0, 0, _hex(rng)]]
            _reroot(log)
        case "no-a":
            entries = [x for x in entries if not Cr.is_a(x["value"])]
        case "twice":
            entries.append(dict(e))
        case "entry-form":
            k, bad = rng.choice([("root", None), ("root", "zz"), ("domain", None), ("domain", e["domain"].upper()),
                                 ("leaves", 0), ("leaves", True), ("leaves", Cr.MAX_ROWS + 1), ("leaves", "4")])
            e[k] = bad
        case "blocks":
            e["leaves"] += 1 if (e["leaves"] + 1) % steps else 2 if (e["leaves"] + 2) % steps else 3
        case "log-form":
            log = rng.choice([[], None, "x"])
        case "round-form":
            log[r] = rng.choice([5, {**log[r], "inner": "zz"}, {k: x for k, x in log[r].items() if k != "root"}])
        case "segments-form":
            bad = rng.choice([None, [[Cr.LEVELS, 0, _hex(rng)]], [[0, -1, _hex(rng)]], [[0, 0, "zz"]], [[0, 0]],
                              [[True, 0, _hex(rng)]]])
            log[r]["a"] = rng.choice([{**log[r]["a"], v: bad}, [segs]])
        case "entries-form":
            entries = rng.choice([None, entries + [5], {"0": entries}])
        case _:
            raise ValueError(name)
    return {"rounds": log, "entries": entries}


A_LINK_NAMES = ["untied", "late-rows", "rewritten", "round-root", "gap", "extra", "swapped", "misaligned", "unknown",
                "no-a", "twice", "entry-form", "blocks", "log-form", "round-form", "segments-form", "entries-form"]
