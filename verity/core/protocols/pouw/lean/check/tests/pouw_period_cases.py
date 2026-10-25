"""The Pearl-C period difftest's cases: `credit.pearl_c_period_credit`'s arguments, ``{"key", "ledger", "pins", "main",
"name", "passes", "earlier"}``.

An honest case is one period of lane `pouw_rounds_cases.LANE` in the auditor's signed ledger holding one to three Pearl-C
salt epochs, each a salt-only challenger session the gate reserved for the salt coin alone, opened on the epoch's root
R = `salt.epoch_open_root(W, k)` and ended complete, over one or two weights' roots W, each W's epochs numbered on from
the lane's earlier credited ones (``earlier``); one to three passes of each epoch, their call indices running across the
epoch (each pass's a block right after the one before), presented with their session's files, their context and the
descriptor whose SHA-512 the period's open committed. Then one mutation (`NAMES`, each aimed at one clause's refusal,
but ``stale-pin``, a window pinned behind its period's open, which the rule credits). The leaves: the period's open at
0, epoch j's reserve, open, session record, session close and end at 5j + 1 … 5j + 5, the close after.

`tests/test_difftest.py` draws them for the harness.
"""
from __future__ import annotations

import hashlib
import json
import random

import pouw_pass_cases as PC
import pouw_rounds_cases as RC
from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import EPOCH_SALT_DOMAIN
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.randomness import derive

#: `pearl_c_period_credit`'s arguments, in order (then the signature check)
KEYS = ("key", "ledger", "pins", "main", "name", "passes", "earlier")

#: the mutations, by the clause each is aimed at: the inputs' form; exact match; salt-only sessions; credited sessions;
#: accepted passes; consecutive epochs; distinct calls
NAMES = ["passes-form", "manifest-hex", "manifest-json", "no-root", "earlier-form",
         "missing", "abandoned", "order", "foreign-root", "incomplete", "unclosed", "other-name", "forged",
         "reserve-layout", "h1-session", "unreserved",
         "descriptor-hex", "other-circuit", "stale-pin", *RC.PIN_NAMES,
         "resalted", "context", "files", "call-range", "seeded", "named-epoch", "misnamed",
         "spent", "repeat", "inner-gap",
         "repeated-call", "twice"]
#: the mutations that need two epochs, and those that need two epochs on one W
_TWO = {"order", "files", "inner-gap"}
_ONE_W = {"inner-gap"}


def _dims(rng: random.Random) -> dict[int, tuple[int, int]]:
    return {w: (rng.randint(1, 9), rng.randint(1, 9)) for w in rng.sample(range(1 << 20), rng.randint(1, 3))}


def _matmuls(rng: random.Random, dims: dict[int, tuple[int, int]], start: int, n: int) -> list[dict]:
    """`n` calls (at least one per weight of `dims`) indexed `start`, `start` + 1, …, in a shuffled order."""
    order = [*dims, *rng.choices(list(dims), k=n - len(dims))]
    rng.shuffle(order)
    us = [{"index": start + i, "m": rng.randint(1, 9), "k": dims[w][1], "n": dims[w][0], "weight": w, "real_rows": 1}
          for i, w in enumerate(order)]
    rng.shuffle(us)
    return us


def _salt_of(s: RC.Session, w: bytes) -> dict:
    """The manifest fields session `s`'s salt coin gives a pass over weights' root `w`."""
    c, o = s.ch.commitment, s.salt[0]
    salt = derive(o.coin, EPOCH_SALT_DOMAIN, {"epoch": c.epoch, "coins": c.root, "weights": w}).stream("salt", 32)
    return {"coins": c.to_json(), "salt_opening": o.to_json(), "salt": salt.hex()}


def _session(rng: random.Random, r: bytes, sid: str, leaf: int, h1: bool = False) -> RC.Session:
    """A salt-only session `sid` (H1's live layout, every coin released, when `h1`) whose salt coin opened after `r`,
    its record signed at `leaf`."""
    if h1:
        rounds = rng.choice([1, 2])
        ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
        return RC.Session(rng, ch, [rng.randbytes(64) for _ in range(rounds)], salt=r, sid=sid, leaf=leaf,
                          draw=lambda _: rng.randbytes(64))
    ch = Ch.LiveChallenger(Wd.LIVE_SCHEME, rng.randrange(4), Wd.SALT_FAMILIES, rng.randbytes)
    return RC.Session(rng, ch, [], salt=r, sid=sid, leaf=leaf)


def _epochs(rng: random.Random, name: str, n: int | None) -> tuple[list[dict], list[list]]:
    """The period's `n` epochs (drawn when None) in the close's order, each its weights group, k and root, and the
    lane's earlier credited epochs."""
    n = rng.randint(2 if name in _TWO else 1, 3) if n is None else n
    groups = []
    for _ in range(1 if name in _ONE_W else rng.randint(1, 2)):
        dims = _dims(rng)
        weights = {str(w): {"n": nn, "k": kk, "root": rng.randbytes(64).hex()} for w, (nn, kk) in dims.items()}
        w = CC.weights_root(Co.SCHEME, {int(x): bytes.fromhex(v["root"]) for x, v in weights.items()})
        groups.append({"dims": dims, "weights": weights, "w": w,
                       "next": rng.choice([0, 0, 1, 2, rng.randrange(1 << 20), (1 << 64) - 4])})
    earlier = []
    for g in groups:
        if g["next"]:
            ks = sorted({g["next"] - 1, *(rng.randrange(g["next"]) for _ in range(rng.randint(0, 2)))})
            earlier += [[g["w"].hex(), k] for k in ks]
    earlier += [[rng.randbytes(64).hex(), rng.randrange(1 << 64)] for _ in range(rng.randint(0, 1))]
    rng.shuffle(earlier)
    chosen = [rng.choice(groups) for _ in range(n)]
    if name == "spent":
        # epoch n_W of the first epoch's W was opened outside a credited period, so its numbering broke
        chosen[0]["next"] += 1
    epochs = []
    for j, g in enumerate(chosen):
        if name == "inner-gap" and j == n - 1:
            g["next"] += 1
        epochs.append({"g": g, "k": g["next"], "r": PS.epoch_open_root(g["w"], g["next"])})
        g["next"] += 1
    if name == "repeat":
        e = rng.choice(epochs)
        earlier.append([e["g"]["w"].hex(), e["k"]])
    return epochs, earlier


def _passes(rng: random.Random, epochs: list[dict], name: str, count: int | None) -> None:
    """Each epoch's `count` passes' matmuls (drawn when None), their call indices one block across the epoch, below
    2^32."""
    for j, e in enumerate(epochs):
        least = 2 if name == "repeated-call" and j == 0 else 1
        sizes = [rng.randint(len(e["g"]["dims"]), len(e["g"]["dims"]) + 3)
                 for _ in range(rng.randint(least, 3) if count is None else max(count, least))]
        start = rng.choice([0, 0, rng.randrange(1 << 12), CC.CALL_BOUND - sum(sizes)])
        e["passes"] = []
        for size in sizes:
            e["passes"].append(_matmuls(rng, e["g"]["dims"], start, size))
            start += size
    if name == "repeated-call":
        first, second = epochs[0]["passes"][:2]
        u = rng.choice(second)
        u["index"] = rng.choice(first)["index"]


def _order(rng: random.Random, epochs: list[dict]) -> list[tuple[int, int]]:
    """The passes as presented, (epoch, pass): any interleaving whose epochs first appear in the close's order."""
    queues = [[(j, i) for i in range(len(e["passes"]))] for j, e in enumerate(epochs)]
    out, shown = [], 0
    while any(queues):
        ready = [q for q in queues[:shown + 1] if q]
        q = rng.choice(ready) if rng.random() < 0.5 else ready[0]
        out.append(q.pop(0))
        shown = max(shown, out[-1][0] + 1)
    return out


def instance(rng: random.Random, name: str = "honest", epochs: int | None = None,
             passes: int | None = None) -> dict:
    """One `pearl_c_period_credit` case: an honest Pearl-C period (of `epochs` epochs and `passes` passes each, drawn
    when None), then mutation `name`."""
    pname = rng.choice(RC.PERIODS)
    desc = rng.randbytes(rng.randint(1, 48))
    eps, earlier = _epochs(rng, name, epochs)
    _passes(rng, eps, name, passes)
    h1 = rng.randrange(len(eps)) if name == "h1-session" else None
    for j, e in enumerate(eps):
        e["s"] = _session(rng, e["r"], RC.session_id(rng), 5 * j + 3, h1=j == h1)
    extra = None
    if name == "abandoned":
        # a complete salt-only session on the epoch after the last, in the period, that no pass presents
        g = eps[-1]["g"]
        extra = {"g": g, "k": g["next"], "r": PS.epoch_open_root(g["w"], g["next"]), "passes": []}
        extra["s"] = _session(rng, extra["r"], RC.session_id(rng), 5 * len(eps) + 3)
    ledgered = eps + ([extra] if extra else [])
    pub = eps[0]["s"].pub
    pins = list(RC.PINS)
    stale = rng.randrange(len(eps)) if name == "stale-pin" else None
    opened: dict[int, dict] = {}
    j = rng.randrange(len(eps))
    match name:
        case "unpinned":
            opened[j] = {"pinned": RC.UNLOGGED}
        case "branch-pin":
            pins.append(Ch.pin_line(RC.BRANCH, RC.MAIN[0], "2026-10-23T03:05:00Z"))
            opened[j] = {"pinned": RC.BRANCH, "pins": Ch.pins_seal(pins)}
        case "unsealed-pins":
            opened[j] = {"pins": rng.choice([None, {"lines": 0, "sha512": Ch.log_digest([])}])}
    open_body = {"period": pname, "lane": RC.LANE, "by": RC.LANE, "cap": len(ledgered) + rng.randint(0, 2),
                 "circuit": hashlib.sha512(desc).hexdigest(), Cr.PERIOD_PIN_FIELD: RC.MAIN[2 if stale is not None else -1],
                 "key_sha512": hashlib.sha512(pub).hexdigest()}
    events, leaves = [("period-open", None, open_body)], [0]
    for j, e in enumerate(ledgered):
        s, c = e["s"], e["s"].ch.commitment
        o = RC._open(rng, {"root": e["r"].hex()}, s.pub)
        o.update(scheme=c.scheme, epoch=c.epoch, families=[list(f) for f in c.families])
        if j == stale:
            o.update(pinned=RC.MAIN[4])
        o = {k: v for k, v in {**o, **opened.get(j, {})}.items() if v is not None}
        end = {**PC.gate_end(s), **({"complete": False} if name == "incomplete" and j == 0 else {})}
        if name == "unreserved" and j == 0:
            events += [("open", s.sid, o), ("end", s.sid, RC._end(s, **end))]
            leaves += [5 * j + 2, 5 * j + 5]
        else:
            if name == "reserve-layout" and j == 0:
                # the open carries its reserve's request (the ledger's rule), so both name the other layout
                o["families"] = rng.choice([[list(f) for f in Wd.live_families(1)], [[Wd.SALT, 2]],
                                            [[Wd.SALT, 1], [Wd.DRAW, 1]], [[Wd.SALT, True]], None])
                if o["families"] is None:
                    del o["families"]
            reserve, opening = RC._reserved(rng, s.sid, o)
            events += [reserve, opening, ("end", s.sid, RC._end(s, **end))]
            leaves += [5 * j + 1, 5 * j + 2, 5 * j + 5]
    windows = [{"root": e["r"].hex(), "session": e["s"].sid, "complete": not (name == "incomplete" and j == 0)}
               for j, e in enumerate(ledgered)]
    if name != "unclosed":
        events.append(("period-close", None, {"period": pname, "lane": RC.LANE, "by": RC.LANE, "windows": windows,
                                              "complete": all(w["complete"] for w in windows)}))
        leaves.append(5 * len(ledgered) + 1)
    ledger = RC.ledger_lines(RC._bind(events), RC._key(1 if name == "forged" else 0), leaves)
    entries = [json.loads(line) for line in ledger]
    order = _order(rng, eps)
    if name == "order":
        perm = list(range(len(eps)))
        a, b = rng.sample(perm, 2)
        perm[a], perm[b] = perm[b], perm[a]
        order = [(j, i) for j in perm for i in range(len(eps[j]["passes"]))]
    if name == "missing":
        gone = rng.randrange(len(eps))
        order = [p for p in order if p[0] != gone]
    out = []
    for j, i in order:
        e = eps[j]
        opened_e = next(x for x in entries if x["kind"] == "open" and x["session"] == e["s"].sid)
        man = {"schema": CC.MANIFEST_SCHEMA, "matmuls": e["passes"][i], "weights": e["g"]["weights"],
               "run": f"r-{rng.randrange(1 << 30):x}", "pass_index": e["k"], **_salt_of(e["s"], e["g"]["w"]),
               "session": PC.carried(opened_e, e["s"].ch.commitment)}
        out.append({"man": man, "j": j, "s": e["s"], "desc": desc, "e": e})
    d = {"key": eps[0]["s"].key(), "ledger": ledger, "pins": pins, "main": list(RC.MAIN), "name": pname,
         "earlier": earlier}
    _mutate(rng, name, d, out, eps)
    if "passes" not in d:
        d["passes"] = [_serialized(rng, p) for p in out]
    if name == "twice":
        p = rng.choice(d["passes"])
        d["passes"].insert(rng.randint(d["passes"].index(p) + 1, len(d["passes"])), p)
    _mutate_after(rng, name, d)
    return d


def _serialized(rng: random.Random, p: dict) -> dict:
    """Pass `p` as the period presents it, its context naming its manifest's bytes by their SHA-512."""
    raw = p.get("raw") or PC._raw(p["man"], rng)
    digest = p.get("digest") or hashlib.sha512(raw).hexdigest()
    return {"context": f"pearl-c/{p['man'].get('run', 'r')}/{digest}", "descriptor": p.get("descriptor", p["desc"].hex()),
            "manifest": p.get("hex", raw.hex()), "session": p.get("files") or p["s"].files()}


def _mutate(rng: random.Random, name: str, d: dict, out: list[dict], eps: list[dict]) -> None:
    """Mutation `name` of the period's passes `out` before they are serialized, or of the case `d`."""
    if not out:
        return
    p = rng.choice(out)
    man = p["man"]
    match name:
        case "manifest-hex":
            p["hex"] = rng.choice([PC._raw(man, rng).hex().upper(), "", PC._raw(man, rng).hex()[:-1], "zz", None, 5])
        case "manifest-json":
            p["raw"] = rng.choice([b"[1]", b"not json", b"\xff\xfe", b'"x"', b"NaN", b"null", b"{"])
        case "no-root":
            if rng.random() < 0.3:
                man.pop("session") if rng.random() < 0.5 else man.update(session=rng.choice([None, "s", [1]]))
            else:
                man["session"] = {**man["session"], "root": rng.choice([None, 5, [man["session"]["root"]], True])}
        case "earlier-form":
            w = eps[0]["g"]["w"].hex()
            d["earlier"] = rng.choice([None, {}, "x", [[w]], [[w, -1]], [[w, 1 << 64]], [[w, True]], [[w.upper(), 0]],
                                       [[w[:-2], 0]], [[w, 0, 1]], [[w, "0"]], [[w, [0]]], [w, 0], [[None, 0]]])
        case "foreign-root":
            man["session"] = {**man["session"], "root": rng.choice([rng.randbytes(64).hex(), "zz",
                                                                     rng.randbytes(64).hex().upper()])}
        case "other-name":
            d["name"] = rng.choice([n for n in RC.PERIODS if n != d["name"]] + [None, 5, ["p-0"]])
        case "descriptor-hex":
            p["descriptor"] = rng.choice(["A" + p["desc"].hex()[1:], "", "zz", p["desc"].hex()[:-1], None, 5])
        case "other-circuit":
            p["descriptor"] = rng.randbytes(len(p["desc"]) + 1).hex()
        case "no-pins":
            d["pins"] = None
        case "no-main":
            d["main"] = None
        case "main-form":
            d["main"] = rng.choice(["\n".join(RC.MAIN), [c.upper() for c in RC.MAIN], [c[:10] for c in RC.MAIN],
                                    RC.MAIN + [None]])
        case "resalted":
            man["salt"] = rng.choice([PC._flip(man["salt"]), None, man["salt"].upper()])
        case "context":
            p["digest"] = PC._flip(hashlib.sha512(PC._raw(man, rng)).hexdigest())
        case "files":
            other = next(e for e in eps if e["s"] is not p["s"])
            p["files"] = other["s"].files()
        case "call-range":
            rng.choice(man["matmuls"])["index"] = rng.choice([CC.CALL_BOUND, CC.CALL_BOUND + rng.randrange(1, 1 << 20),
                                                               (1 << 64) - 1])
        case "seeded":
            ch = PC.Coins(rng, rng.randrange(4))
            root = CC.weights_root(Co.SCHEME, PC._roots(man))
            man.update(coins=ch.commitment, salt_opening=ch.open("salt", root))
        case "named-epoch":
            man["pass_index"] = p["e"]["k"] + 1 if p["e"]["k"] < (1 << 64) - 1 else p["e"]["k"] - 1
        case "misnamed":
            g = man["session"]
            k = rng.choice(["session", "seq", "commitment", "secret"])
            man["session"] = {**g, "secret": rng.randbytes(32).hex()} if k == "secret" else {
                **g, k: {"session": RC.session_id(rng), "seq": g["seq"] + 1, "commitment": PC._flip(g["commitment"])}[k]}


def _mutate_after(rng: random.Random, name: str, d: dict) -> None:
    """Mutation `name` of the serialized case `d`."""
    if name == "passes-form":
        p = rng.choice(d["passes"])
        d["passes"] = rng.choice([None, {}, "x", [], [*d["passes"], 5], [{k: v for k, v in p.items() if k != "session"}],
                                  [{**p, "extra": 1}]])


def args(d: dict) -> tuple:
    """`pearl_c_period_credit`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in KEYS), RC.lms_verify)
