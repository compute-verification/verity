"""The fill's difftest cases (`fill`, ``pouw-period``'s ``fills``): an honest served period (`pouw_credit_cases
.period_instance`) and, after its close, its fill period ``<name>:fill`` in the same auditor's ledger: its open commits
the filler circuit (J10-R units over ``PouwH1Tile_v5`` at one small K), its draw K and its weights' root and model, then
one or two filler windows, each its session opened and ended, then its close. Each filler window's challenger session
holds a salt coin and a draw coin (`fill.FILL_FAMILIES`), its salt coin opened after its open root (`fill.open_root` of
the weights it registers and its session), its Program salt that coin's (`h1tile.live_salt`); it registers each K's pad
table (`fill.pad_entry`), its weights and each call's z/y and receipt tables under its open root, then its draw coin opens
after `fill.draw_root` over the salt and those entries, and each unit that coin draws has an accepted proof outcome
(`pouw_credit_cases._window_proofs`). Each mutation (`NAMES`) aims at one of `fill._period_outcomes`' refusals, made where
a prover would make it: before the draw for the pad and the salt, after it for a stale entry."""
from __future__ import annotations

import hashlib
import itertools
import random

import pouw_credit_cases as cc
import pouw_rounds_cases as rc
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

NAMES = ["fill-pad-root", "fill-pad-missing", "fill-extra-value", "fill-salt-other", "fill-salt-unreceived",
         "fill-stale-entry", "fill-draw-root", "fill-coin-layout", "fill-unit-failed", "fill-unit-missing",
         "fill-window-missing", "fill-no-close", "fill-weights-other", "fill-no-k", "fill-form", "fill-v6"]
#: the refusal code each mutation aims at, as `fill._period_outcomes` names it (``fill`` a refused filler window's)
AIMS = {"fill-pad-root": "fill", "fill-pad-missing": "fill", "fill-extra-value": "fill", "fill-salt-other": "fill",
        "fill-salt-unreceived": "fill", "fill-stale-entry": "fill", "fill-draw-root": "fill",
        "fill-coin-layout": "fill", "fill-unit-failed": "fill", "fill-unit-missing": "fill", "fill-window-missing":
        "missing", "fill-no-close": "period", "fill-weights-other": "fill", "fill-no-k": "budget",
        "fill-form": "fill", "fill-v6": "form"}
#: the refused window's own code, inside a ``fill`` refusal
INNER = {"fill-pad-root": "pad", "fill-pad-missing": "pad", "fill-extra-value": "pad", "fill-salt-other": "salt",
         "fill-salt-unreceived": "salt", "fill-stale-entry": "draw", "fill-draw-root": "draw",
         "fill-coin-layout": "coin", "fill-unit-failed": "failed", "fill-unit-missing": "missing",
         "fill-weights-other": "weights"}


def fill_layout(rng: random.Random) -> dict:
    """A small filler window's layout at one K of 32 or 64, so the verifier's pad stays a few thousand codes."""
    r, c = rng.choice([(1, 1), (2, 4), (4, 8)])
    k = rng.choice([32, 64])
    names = rng.sample(cc.LINEARS, rng.randint(1, 3))
    linears = [[name, k, c * rng.randint(1, 2)] for name in names]
    calls = [[name, 0] for name in names]
    rng.shuffle(calls)
    return {"calls": calls, "linears": linears, "m": r << rng.randint(0, 1), "tile": [r, c]}


def _entry(program: str, window: str, v: str, leaves: int, words: int, rng: random.Random) -> dict:
    return {"value": v, "root": cc._hex(rng), "leaves": leaves, "words": words,
            "domain": G.domain(program, v, words, leaves, G.WORD_BITS, window).domain_id.hex()}


def honest_fill(rng: random.Random, lay: dict, desc: str, prefix: str, weights: list[dict], sid: str, leaf: int,
                big_k: int, name: str = "honest") -> tuple[dict, rc.Session]:
    """One filler window's evidence (`fill.FILL_KEYS`) and its challenger's session, under mutation `name` where it is
    made before the window's draw coin or changes its coins."""
    program = hashlib.sha512(bytes.fromhex(desc)).hexdigest()
    families = Wd.live_families(1) if name == "fill-coin-layout" else F.FILL_FAMILIES
    ch = Ch.LiveChallenger(Wd.LIVE_SCHEME, rng.randrange(4), families, rng.randbytes)
    r = F.open_root(bytes.fromhex(cc.fold(weights)), sid)
    window = {"root": r.hex()}
    s = rc.Session(rng, ch, [], salt=rng.randbytes(64) if name == "fill-salt-unreceived" else r, sid=sid, leaf=leaf)
    opening = s.salt[0]
    salt = HT.live_salt(ch.commitment, opening.coin, 0, opening.after)
    if name == "fill-salt-other":
        salt = rng.randbytes(64)
    entries = [dict(e) for e in weights]
    pads = [F.pad_entry(program, prefix, r.hex(), k, rows) for k, rows in F.pad_rows(lay).items()]
    if name != "fill-pad-missing":
        entries += pads
    sizes = [size for size, _w in Cr.call_work(lay)]
    for c, size in enumerate(sizes):
        entries.append(_entry(program, r.hex(), f"{prefix}zy@{c}", size, 4, rng))
        if rng.random() < 0.5:
            entries.append(_entry(program, r.hex(), f"{prefix}receipt@{c}", size, 4, rng))
    if name == "fill-pad-root":
        e = next(e for e in entries if e["value"].endswith("@0") and "/a/" in e["value"])
        e["root"] = cc._hex(rng)
    if name == "fill-extra-value":
        entries.append(_entry(program, r.hex(), f"{prefix}ids@0", 1, 2, rng))
    rng.shuffle(entries)
    receipts = F.entries_root(entries)
    root = F.draw_root(salt, receipts)
    record = Ch.LiveRecord(ch, Wd.DRAW, itertools.count(10 ** 13).__next__)
    s.draw = [record.record(root)]
    s.lines = s.lines + [canonical_json_text(e) for e in record.log]
    s.sealed = s.lines
    s.close = s.signed()
    d = {"key": s.key(), "window": window, "session": s.files(), "salt": salt.hex(), "opening": opening.to_json(),
         "entries": entries, "descriptor": desc, "layout": lay, "rounds": [{}],
         "draw": {"receipts": receipts.hex(), "root": root.hex(), "coin": s.draw[0].to_json()}}
    if name == "fill-coin-layout":
        d["proofs"] = []
        return d, s
    d["proofs"] = cc._window_proofs(rng, {**d, "entries": entries + pads if name == "fill-pad-missing" else entries},
                                    cc._served_units(d, k=big_k))
    return d, s


def _extend(lines: list[str], events: list[tuple[str, str | None, dict]], key: lms.PrivateKey,
            leaves: list[int]) -> list[str]:
    """`lines`, the auditor's ledger, with `events` chained after them and signed at `leaves`."""
    out = list(lines)
    for (kind, sid, body), leaf in zip(events, leaves):
        entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest(),
                 "schema": Ch.LEDGER_SCHEMA, "seq": len(out), "session": sid}
        m = Ch.ledger_message(lms.public_key(key)[:16], entry)
        out.append(Ch.ledger_text({**entry, "sig": lms.sign_at(key, leaf, m).encode().hex()}))
    return out


def with_fills(rng: random.Random, d: dict, name: str = "fill") -> dict:
    """Honest period `d` (`period_instance`'s, before it is served as a case) with its fill period after its close and
    ``fills``, each filler window's evidence in its close's order; then mutation `name`."""
    lay = fill_layout(rng)
    prefix = f"fill/w-{rng.randrange(100)}/"
    desc = cc.descriptor(rng, lay, prefix, j10r=True, tile=6 if name == "fill-v6" else 5)
    program = hashlib.sha512(bytes.fromhex(desc)).hexdigest()
    weights = cc.weights_of(rng, lay, program, prefix)
    big_k = rng.randint(1, cc.units_of(lay))
    n = rng.randint(2 if name == "fill-window-missing" and rng.random() < 0.5 else 1, 2)
    base = 4 * len(d["windows"]) + 6
    bad = rng.randrange(n)
    wins = [honest_fill(rng, lay, desc, prefix, weights, f"f-{j + 1}", base + 2 + 4 * j, big_k,
                        name if j == bad else "honest") for j in range(n)]
    pub = wins[0][1].pub
    signed = cc.signs(weights)
    if name == "fill-weights-other":
        signed[Cr.WEIGHTS_FIELD] = cc._hex(rng)
    opened = {"period": F.period_name(d["name"]), "lane": rc.LANE, "by": rc.LANE, "cap": 4, "circuit": program,
              Cr.K_FIELD: big_k, **signed, Cr.PERIOD_PIN_FIELD: rc.MAIN[-1],
              "key_sha512": hashlib.sha512(pub).hexdigest()}
    if name == "fill-no-k":
        del opened[Cr.K_FIELD]
    events = [("period-open", None, opened)]
    for w, s in wins:
        events += [("open", s.sid, rc._open(rng, w["window"], pub)), ("end", s.sid, rc._end(s))]
    leaves = [base] + [x for j in range(n) for x in (base + 1 + 4 * j, base + 4 + 4 * j)]
    if name != "fill-no-close":
        events.append(("period-close", None, {
            "period": opened["period"], "lane": rc.LANE, "by": rc.LANE, "complete": True,
            "windows": [{"root": w["window"]["root"], "session": s.sid, "complete": True} for w, s in wins]}))
        leaves.append(base + 1 + 4 * n)
    ledger = _extend(d["ledger"], events, rc._key(0), leaves)
    fills = [{k: w[k] for k in F.FILL_KEYS} for w, _s in wins]
    i = rng.randrange(n)
    match name:
        case "fill-stale-entry":
            e = next(e for e in fills[i]["entries"] if "/zy@" in e["value"])
            e["root"] = cc._hex(rng)
        case "fill-draw-root":
            fills[i]["draw"] = {**fills[i]["draw"], "root": cc._hex(rng)}
        case "fill-unit-failed":
            fills[bad]["proofs"] = cc._mutate_proofs(fills[bad]["proofs"], rng, "failed", [])
        case "fill-unit-missing":
            fills[bad]["proofs"] = cc._mutate_proofs(fills[bad]["proofs"], rng, "missing", [])
        case "fill-window-missing":
            del fills[i]
        case "fill-form":
            fills = rng.choice([None, {"0": fills[0]}, [{k: v for k, v in fills[0].items() if k != "opening"}]])
    return {**d, "ledger": ledger, "fills": fills}
