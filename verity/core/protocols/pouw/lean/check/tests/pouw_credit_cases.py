"""The credit difftest's cases: `credit.credited`'s, `credit.outcome`'s and `credit.a_link`'s arguments.

A `credited` case is an honest served window read through its challenger's session (`pouw_rounds_cases.Session`): its
manifest rows (the serving process's call rows, each with its credit bit, a credited one with its z table's root), the
round log (each recorded round root and its coin), and the draw: `credit.draw_root` over them, recorded by the
challenger, then its draw coin opened after it; the auditor's ledger opens and ends the session for the window. Then one
mutation (`CREDIT_NAMES`) aimed at one refusal: a credit bit flipped after the draw coin with the draw root recomputed
(``changed``, RT4-INT-32's withdrawn verdict), a row changed likewise, a missing or malformed bit, a credited row with no
z root, the draw root left stale, no draw coin, a coin of another family, a host pinned outside the verifier's copy of
main or repinned mid-session (`pouw_rounds_cases.PIN_NAMES`), and the forms.

An `outcome` case is a draw's units and one proof outcome each, all accepts; its mutations drop one (``missing``),
reject one (``failed``), repeat one, add an undrawn unit, and break the forms.

An `a_link` case is a window's registered entries, among them one or two A values (``…/a/<K>``) whose trees are frame-v3
SHA-512 trees (`verity.ml.commitments.merkle.tree_levels`) of `steps` equal blocks, and its round log: each round's
`credit.cover` segments of every A value under `credit.round_root` of a random inner root. Its mutations register an A
tree other than the one the rounds committed (RT4-INT-34), rewrite a segment with or without its round root, leave a gap,
add a segment, misalign one, list an unregistered value, register none or one twice, and break the forms.

`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_credit.py` holds each
mutation to the refusal it aims at.
"""
from __future__ import annotations

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


def honest_credit(rng: random.Random, name: str = "honest") -> tuple[dict, rc.Session]:
    """An honest served window whose draw coin opened after its draw root (none for ``no-draw``)."""
    rounds = rng.choice([1, 2, 3])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    d = {"salt": _hex(rng), "calls": [_row(rng, i) for i in range(rng.randint(1, 5))],
         "served": rng.choice([None, _hex(rng)]), "draw": {"receipts": _hex(rng)}}

    def draw_after(rec) -> bytes:
        d["rounds"] = [{"root": r.hex(), "coin": o.to_json()} for r, o in zip(rec.roots, rec.openings)]
        d["draw"]["root"] = _draw_root(d)
        return bytes.fromhex(d["draw"]["root"])

    s = rc.Session(rng, ch, [rng.randbytes(64) for _ in range(rounds)], salt=None,
                   draw=None if name == "no-draw" else draw_after)
    if name == "no-draw":
        d["draw"]["coin"] = ch.open(Wd.DRAW, 0, draw_after(s.rounds)).to_json()
    else:
        d["draw"]["coin"] = s.draw[0].to_json()
    return {"key": s.key(), "main": list(rc.MAIN), "window": rc._window(rng), "session": s.files(), **d}, s


def _flip_bit(d: dict, rng: random.Random, to: bool | None = None) -> None:
    rows = [r for r in d["calls"] if to is None or r[Cr.CREDIT] != to] or d["calls"]
    row = rng.choice(rows)
    row[Cr.CREDIT] = not row[Cr.CREDIT] if to is None else to
    if row[Cr.CREDIT]:
        row["tables"].setdefault("z", _hex(rng))


def _mutate_credit(d: dict, s: rc.Session, name: str, rng: random.Random) -> None:
    if rc._pinned(d, name, rng, lambda **kw: rc.ledger_lines(rc.ledger_events(rng, d["window"], s, **kw), rc._key(0))):
        return
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
                "forged-ledger", *rc.PIN_NAMES]
CREDIT_KEYS = ("key", "main", "ledger", "window", "session", "salt", "calls", "served", "rounds", "draw")


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


def outcome_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `outcome` case: a draw's units, each with one accepted proof, then mutation `name`."""
    drawn = rng.sample(range(64), rng.randint(1, 6))
    proofs = [{"unit": u, "verdict": "accept"} for u in drawn]
    rng.shuffle(proofs)
    j = rng.randrange(len(proofs))
    match name:
        case "honest":
            pass
        case "missing":
            del proofs[j]
        case "failed":
            proofs[j]["verdict"] = "reject"
        case "twice":
            proofs.insert(rng.randrange(len(proofs) + 1), dict(proofs[j]))
        case "undrawn":
            proofs.insert(rng.randrange(len(proofs) + 1), {"unit": 64 + rng.randrange(8), "verdict": "accept"})
        case "drawn-form":
            drawn = rng.choice([[], None, drawn + drawn[:1], drawn + [-1], drawn + [True], drawn + ["1"], {"0": 1}])
        case "proofs-form":
            proofs = rng.choice([None, "x", proofs + [5], proofs[:j] + [{**proofs[j], "verdict": "maybe"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "why": "x"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "unit": -1}] + proofs[j + 1:]])
        case _:
            raise ValueError(name)
    return {"drawn": drawn, "proofs": proofs}


OUTCOME_NAMES = ["missing", "failed", "twice", "undrawn", "drawn-form", "proofs-form"]


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
