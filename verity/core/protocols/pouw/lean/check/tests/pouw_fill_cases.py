"""The fill's difftest cases (`fill`, ``pouw-period``'s ``fills``): an honest served period (`pouw_credit_cases
.period_instance`) and, after its close, its fill period ``<name>:fill`` in the same auditor's ledger: its open commits
the filler circuit (`fill.descriptor` of a small Dither filler shape) and its draw K, then one or two filler windows, each
its session opened and ended, then its close. Each filler window's root is `fill.open_root` of the circuit and its
session, and its challenger session holds a salt coin, a call coin a call and a draw coin (`fill.families`): the salt
coin and every call coin opened after the root, the window's salt that coin's (`h1tile.live_salt`), then the draw coin
after `fill.draw_root` over the salt and each call's value root, and each block that coin draws has an accepted proof
outcome on its own statement (`fill.statement`, `fill.registered`). Each mutation (`NAMES`) aims at one of
`fill.period_fill`'s refusals, made where a prover would make it: before the draw for the coins, after it for a stale
root."""
from __future__ import annotations

import hashlib
import itertools
import json
import random

import pouw_credit_cases as cc
import pouw_rounds_cases as rc
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import fill as F
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

NAMES = ["fill-form", "fill-window-missing", "fill-no-close", "fill-no-k", "fill-descriptor", "fill-shape",
         "fill-open-other", "fill-coin-layout", "fill-salt-unreceived", "fill-salt-other", "fill-calls-form",
         "fill-call-unreceived", "fill-roots-form", "fill-stale-root", "fill-draw-root", "fill-unit-failed",
         "fill-unit-missing", "fill-statement-other", "fill-registered-other"]
#: the refusal code each mutation aims at, as `fill.period_fill` names it (``fill`` a refused filler window's)
AIMS = {**{name: "fill" for name in NAMES}, "fill-window-missing": "missing", "fill-no-close": "period",
        "fill-no-k": "budget"}
#: the refused window's own code, inside a ``fill`` refusal
INNER = {"fill-descriptor": "descriptor", "fill-shape": "descriptor", "fill-open-other": "open",
         "fill-coin-layout": "coin", "fill-salt-unreceived": "salt", "fill-salt-other": "salt",
         "fill-calls-form": "calls", "fill-call-unreceived": "calls", "fill-roots-form": "roots",
         "fill-stale-root": "draw", "fill-draw-root": "draw", "fill-unit-failed": "failed",
         "fill-unit-missing": "missing", "fill-statement-other": "statement", "fill-registered-other": "registered"}


def fill_shape(rng: random.Random) -> tuple[D.Shape, int]:
    """A small Dither filler shape and its call count: one or two tiles a side, K a few L′, L dividing it."""
    lp = rng.choice([32, 64, 128])
    q = rng.choice([1, 2, 4])
    t = rng.choice([x for x in (1, 2, 4) if q % x == 0])
    sh = D.Shape(128 * rng.randint(1, 2), 128 * rng.randint(1, 2), lp * q, lp, lp * t, rng.choice([4096, 16384, 32768]),
                 o=False)
    return sh, rng.randint(1, 3)


def _digest(rng: random.Random) -> bytes:
    return bytes.fromhex(cc._hex(rng))


def _descriptor(rng: random.Random, sh: D.Shape, calls: int, name: str) -> bytes:
    raw = F.descriptor(sh, calls, _digest(rng))
    if name == "fill-descriptor":
        j = json.loads(raw)
        match rng.randrange(3):
            case 0:
                j["unit"] = "PouwDitherLast_v1"
            case 1:
                j["calls"] = 0
            case _:
                del j["row"]
        raw = Cr._canonical(j).encode()
    if name == "fill-shape":
        j = json.loads(raw)
        x = rng.choice(["m", "lp", "L", "t"])
        if x == "t":
            j.update(lp=32, L=32 * (F.T_MAX + 1), k=32 * (F.T_MAX + 1))
        else:
            j[x] = {"m": 100, "lp": 48, "L": 3 * sh.k}[x]
        raw = Cr._canonical(j).encode()
    return raw


def honest_fill(rng: random.Random, raw: bytes, sid: str, leaf: int, big_k: int, name: str = "honest"
                ) -> tuple[dict, rc.Session]:
    """One filler window's evidence (`fill.FILL_KEYS`) and its challenger's session, under mutation `name` where it is
    made before the window's draw coin or changes its coins."""
    got, _ = F.descriptor_of(raw)
    sh, calls = got if got is not None else (D.Shape(128, 128, 32, 32, 32, o=False), 1)
    circuit = hashlib.sha512(raw).digest()
    fams = F.families(calls + 1 if name == "fill-coin-layout" else calls)
    ch = Ch.LiveChallenger(Wd.LIVE_SCHEME, rng.randrange(4), fams, rng.randbytes)
    r = _digest(rng) if name == "fill-open-other" else F.open_root(circuit, sid)
    s = rc.Session(rng, ch, [], salt=_digest(rng) if name == "fill-salt-unreceived" else r, sid=sid, leaf=leaf)
    opening = s.salt[0]
    salt = HT.live_salt(ch.commitment, opening.coin, 0, opening.after)
    if name == "fill-salt-other":
        salt = _digest(rng)
    clock = itertools.count(10 ** 13).__next__
    record = Ch.LiveRecord(ch, F.CALL, clock)
    odd = rng.randrange(calls)
    coins = [record.record(_digest(rng) if name == "fill-call-unreceived" and c == odd else r)
             for c in range(fams[1][1])]
    roots = [_digest(rng) for _ in range(calls)]
    receipts = F.receipts(roots)
    root = F.draw_root(salt, receipts)
    draw = Ch.LiveRecord(ch, Wd.DRAW, itertools.count(10 ** 14).__next__)
    s.draw = [draw.record(root)]
    s.lines = s.lines + [canonical_json_text(e) for e in (*record.log, *draw.log)]
    s.sealed = s.lines
    s.close = s.signed()
    d = {"window": {"root": r.hex()}, "session": s.files(), "salt": salt.hex(), "opening": opening.to_json(),
         "calls": [o.to_json() for o in coins[:calls]], "roots": [x.hex() for x in roots], "descriptor": raw.hex(),
         "draw": {"receipts": receipts.hex(), "root": root.hex(), "coin": s.draw[0].to_json()}}
    units, why = Cr._window_drawn(d["window"], d["draw"], F.stratum(sh, calls), big_k)
    d["proofs"] = [] if why is not None else [
        {"unit": u, "verdict": "accept", "program": F.statement(circuit, salt, coins[b[0]].coin, b).hex(),
         "registered": F.registered(b[0], roots[b[0]])} for u in units for b in [F.block_of(sh, u)]]
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
    sh, calls = fill_shape(rng)
    n = rng.randint(2 if name == "fill-window-missing" and rng.random() < 0.5 else 1, 2)
    bad = rng.randrange(n)
    raw = _descriptor(rng, sh, calls, name)
    big_k = rng.randint(1, 48)
    base = 4 * len(d["windows"]) + 6
    wins = [honest_fill(rng, raw, f"f-{j + 1}", base + 2 + 4 * j, big_k, name if j == bad else "honest")
            for j in range(n)]
    pub = wins[0][1].pub
    opened = {"period": F.period_name(d["name"]), "lane": rc.LANE, "by": rc.LANE, "cap": 4,
              "circuit": hashlib.sha512(raw).hexdigest(), Cr.K_FIELD: big_k, Cr.PERIOD_PIN_FIELD: rc.MAIN[-1],
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
    f = fills[bad]
    match name:
        case "fill-calls-form":
            f["calls"] = f["calls"][:-1] if rng.random() < 0.5 else f["calls"] + f["calls"][:1]
        case "fill-roots-form":
            f["roots"] = rng.choice([f["roots"][:-1], f["roots"] + [cc._hex(rng)], [x.upper() for x in f["roots"]]])
        case "fill-stale-root":
            f["roots"][rng.randrange(len(f["roots"]))] = cc._hex(rng)
        case "fill-draw-root":
            f["draw"] = {**f["draw"], "root": cc._hex(rng)}
        case "fill-unit-failed":
            f["proofs"] = cc._mutate_proofs(f["proofs"], rng, "failed", [])
        case "fill-unit-missing":
            f["proofs"] = cc._mutate_proofs(f["proofs"], rng, "missing", [])
        case "fill-statement-other":
            p = rng.choice(f["proofs"])
            p["program"] = cc._hex(rng)
        case "fill-registered-other":
            p = rng.choice(f["proofs"])
            t = p["registered"]["t"]
            p["registered"] = rng.choice([{"t": {**t, "root": cc._hex(rng)}}, {"t": {**t, "value": "fill/T@9"}},
                                          {**p["registered"], "a": t}])
        case "fill-window-missing":
            del fills[i]
        case "fill-form":
            fills = rng.choice([None, {"0": fills[0]}, [{k: v for k, v in fills[0].items() if k != "opening"}]])
    return {**d, "ledger": ledger, "fills": fills}
