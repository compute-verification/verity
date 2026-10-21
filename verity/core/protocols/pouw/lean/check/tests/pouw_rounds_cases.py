"""The rounds and salt difftests' cases: `core_check.rounds`'s arguments, ``{"key", "main", "ledger", "window",
"session", "roots", "openings"}``, and `core_check.salt`'s, ``{"key", "main", "ledger", "window", "session", "weights",
"requests", "opening"}``, every entry of their ledgers written by the host's tree at `PIN`, a commit of `MAIN`, the
verifier's copy of main's first-parent history.

A rounds case is an honest live window read through its challenger's session: a live epoch's coins
(`window.live_challenger`), the challenger's record of them signed at leaf `RECORD_LEAF` of a small LMS key (the
window's pinned key), its release log (the salt coin opened after a root, sometimes, then each round root recorded in
order and its coin opened after it, the challenger's `LiveRecord`, in canonical JSON on an integer clock), the session's
close signed at the next leaf over the log's chain, no refused hellos, and the auditor's v2 ledger opening that session
for the window (``{root}``) under the key's hash, among others and sometimes inside a period with its cap and circuit,
and ending it closed, sealing the log and hellos the session's challenger left and naming its record's leaf and
commitment; its entries up to the session's open sign at leaves 0, 1, …, and those after at `RECORD_LEAF` + 2 on.  Then
one mutation (`ROUNDS_NAMES`) aimed at one refusal of `rounds`: the session's (an in-process challenger's coins, an
unsigned or tampered log, a ledger signed under another key, under v1, cut before the session's end or at a reused leaf,
a second session for the window's root, an end naming another record leaf or no hellos, hellos it didn't seal, a host
pinned outside main's first-parent history or repinned mid-session (`PIN_NAMES`), …) or the coin rule's.

A salt case is an honest live window's salt coin read through its challenger's session the same way: the session S a
reserve entry of the auditor's ledger names (every session of that ledger reserved first, the gate's two-step open),
opened on the window's root R, `salt.window_open_root` of its weights' root, its round-0 request table's root and S, and
its challenger's salt coin opened after R.  Then one mutation (`SALT_NAMES`) aimed at one refusal of `salt`: the
session's, a session opened without a reserve, R under another session's id, another window's weights or requests, H1's
salt opened after the weights' root, a log showing another root, or the coin rule's.

`tests/test_difftest.py` draws them for the harness, and
`verity/core/protocols/pouw/tests/test_pouw_core_check.py` holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import random
from collections.abc import Callable

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

#: the mutations that need at least one recorded round, and those that need one round left unrecorded
_RECORDED = {"twice", "early-root", "never", "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "label",
             "openings-form", "swapped", "rewritten", "tampered-log"}
_LEFT = {"early"}
#: a small LMS parameter set: the ledger's entries up to the session's open from leaf 0, the record at `RECORD_LEAF`, the
#: close at the next leaf, the ledger's later entries from the one after
PARAMS = lms.Params(64, 4, 5)
RECORD_LEAF = 8
HOST = "vy-challenger-test"
LANE = "pouw"
#: the commit of the host's tree that writes every entry of a case's ledger, one of `MAIN`, the verifier's copy of
#: main's first-parent history
PIN = "ab" * 20
MAIN = ["0f" * 20, PIN, "c3" * 20]
#: the mutations of the host's pin, for `rounds` and `salt` alike: the session pinned to a pushed commit main's
#: first-parent history doesn't hold, its end repinned to another of main's commits, an end naming no pin or a malformed
#: one, and the verifier holding no copy of main's history or a malformed one
PIN_NAMES = ["pin-not-main", "repinned", "no-pin", "pin-form", "no-main", "main-form"]


@functools.cache
def _key(seed: int) -> lms.PrivateKey:
    return lms.keygen(PARAMS, bytes([seed]) * 64)


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


def _flip(h: str) -> str:
    b = bytearray(bytes.fromhex(h))
    b[0] ^= 1
    return b.hex()


def _record(ch: Ch.LiveChallenger, roots: list[bytes]) -> list[dict]:
    rec = Ch.LiveRecord(ch, Wd.ROUND)
    return [rec.record(r).to_json() for r in roots]


class Session:
    """A challenger's session over `ch`'s coins as its host leaves it, its record and close signed under key `seed`: its
    salt coin opened after root `salt` (none when None), then round coins after `roots`, then its draw coin after the
    root `draw` makes of the round coins' record (none when None)."""

    def __init__(self, rng: random.Random, ch: Ch.LiveChallenger, roots: list[bytes], salt: bytes | None, seed: int = 0,
                 sid: str = "s-1", draw: Callable[[Ch.LiveRecord | None], bytes] | None = None) -> None:
        self.ch, self.k, self.sid = ch, _key(seed), sid
        self.pub = lms.public_key(self.k)
        ch.sign_record(HOST, self.pub, lambda m: lms.sign_at(self.k, RECORD_LEAF, m).encode())
        clock = itertools.count(rng.choice([0, 1, 10 ** 12])).__next__
        self.lines = [canonical_json_text({"event": "commitment", "root": ch.commitment.root.hex()})]
        records = [] if salt is None else [Ch.LiveRecord(ch, Wd.SALT, clock)]
        self.salt = [rec.record(salt) for rec in records]
        self.rounds = Ch.LiveRecord(ch, Wd.ROUND, clock) if ch.commitment.family(Wd.ROUND) else None
        for r in roots:
            self.rounds.record(r)
        records += [] if self.rounds is None else [self.rounds]
        draws = [] if draw is None else [Ch.LiveRecord(ch, Wd.DRAW, clock)]
        self.draw = [rec.record(draw(self.rounds)) for rec in draws]
        records += draws
        self.lines += [canonical_json_text(e) for rec in records for e in rec.log]
        self.sealed, self.hellos, self.sealed_hellos = self.lines, [], []
        self.close = self.signed()

    def signed(self, lines: list[str] | None = None, sid: str | None = None, ended: str | None = None) -> dict:
        lines = self.lines if lines is None else lines
        sid = self.sid if sid is None else sid
        chain = Ch.log_chain(lines)
        m = Ch.close_message(self.pub[:16], HOST, self.ch.commitment, sid, len(lines), chain, ended)
        return Ch.ChallengerClose(sid, len(lines), chain, ended, lms.sign_at(self.k, RECORD_LEAF + 1, m).encode()).to_json()

    def files(self) -> dict:
        return {"close": self.close, "hellos": list(self.hellos), "log": list(self.lines), "record": self.ch.record.to_json()}

    def key(self) -> dict:
        return Ch.ChallengerKey(PARAMS.W, PARAMS.H, self.pub).to_json()


PERIODS = ["p-0", "smoke-20261020", "2026-10-20T09:00Z", "a"]
CIRCUITS = ["ab" * 64, "0f" * 64]


def _window(rng: random.Random) -> dict:
    return {"root": rng.randbytes(64).hex()}


def _open(rng: random.Random, window: dict, public: bytes, pin: str = PIN) -> dict:
    return {"window": window, "host": HOST, "pinned": pin,
            "key_sha512": hashlib.sha512(public).hexdigest(), "port": 7811, "by": LANE,
            **({"for": LANE} if rng.random() < 0.3 else {})}


def leaves_of(events: list[tuple[str, str | None, dict]], sid: str = "s-1") -> list[int]:
    """The leaves `ledger_lines` signs `events` at: 0, 1, … up to session `sid`'s open, then `RECORD_LEAF` + 2 on, so the
    session's record and close (at `RECORD_LEAF` and the next) fall between its open and its end."""
    j = next((i for i, e in enumerate(events) if e[:2] == ("open", sid)), len(events))
    return [i if i <= j else RECORD_LEAF + 2 + i - j - 1 for i in range(len(events))]


def ledger_lines(events: list[tuple[str, str | None, dict]], k: lms.PrivateKey, leaves: list[int] | None = None,
                 schema: str = Ch.LEDGER_SCHEMA) -> list[str]:
    """The auditor's ledger of `events` (kind, session, body), chained and signed under `k` at `leaves` (`leaves_of`'s by
    default), in `schema`."""
    leaves = leaves_of(events) if leaves is None else leaves
    out: list[str] = []
    for i, (kind, sid, body) in enumerate(events):
        entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest() if out else "0" * 128,
                 "schema": schema, "seq": i, "session": sid}
        m = Ch.ledger_message(lms.public_key(k)[:16], entry)
        out.append(Ch.ledger_text({**entry, "sig": lms.sign_at(k, leaves[i], m).encode().hex()}))
    return out


def _end(s: Session, **changes) -> dict:
    files = {Ch.LOG_FILE: Ch.log_digest(s.sealed), Ch.HELLOS_FILE: Ch.log_digest(s.sealed_hellos),
             Ch.RECORD_FILE: hashlib.sha512(canonical_json_text(s.ch.record.to_json()).encode()).hexdigest(),
             Ch.CLOSE_FILE: hashlib.sha512(canonical_json_text(s.close).encode()).hexdigest()}
    body = {"closed": True, "files": files, "rc": 0, "record_leaf": RECORD_LEAF, "commitment": s.ch.commitment.root.hex(),
            "refused_hellos": len(s.sealed_hellos), "complete": True, "pinned": PIN}
    return {k: v for k, v in {**body, **changes}.items() if v is not None}


def _reserved(rng: random.Random, sid: str, body: dict) -> list[tuple[str, str | None, dict]]:
    """Session `sid`'s reserve, then its open with `body` carrying the reserve's request, by its prover, inside its
    minutes; `_bind` names the reserve's seq in the open once the ledger is laid out."""
    minutes = rng.choice([1, 30, 720])
    start = f"2026-10-{rng.randint(10, 28)}T{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}Z"
    asked = {**{k: body[k] for k in ("by", "for") if k in body}, "minutes": minutes}
    opened = Ch.later(start, rng.randint(0, minutes))
    return [("reserve", sid, {**asked, "started_utc": start, "expires_utc": Ch.later(start, minutes),
                              "pinned": body["pinned"]}),
            ("open", sid, {**body, **asked, "opened_by": LANE, "started_utc": opened})]


def _bind(events: list) -> list:
    """`events` with each open of a reserved session naming its reserve's seq."""
    at = {sid: i for i, (kind, sid, _) in enumerate(events) if kind == "reserve"}
    return [(k, sid, {**b, "reserve": at[sid]}) if k == "open" and sid in at else (k, sid, b) for k, sid, b in events]


def ledger_events(rng: random.Random, window: dict, s: Session, *, reserved: bool = False, pin: str = PIN, **end
                  ) -> list[tuple[str, str | None, dict]]:
    """The window's session `s` opened and ended closed (its end's body changed by `end`), sealing the log and hellos its
    challenger left, its entries written by the host's tree at `pin`, among up to two others, sometimes inside a period
    of lane `LANE` that a close ends.  `reserved` reserves every session first, as the gate's two-step open does
    (`_reserved`), among up to one other: the period counts its windows at their reserves."""
    o = _open(rng, window, s.pub, pin)
    named = ((_reserved(rng, s.sid, o) if reserved else [("open", s.sid, o)])
             + [("end", s.sid, _end(s, **{"pinned": pin, **end}))])
    others = []
    for i in range(rng.randint(0, 1 if reserved else 2)):
        sid = session_id(rng) if reserved else f"o-{i}"
        o = _open(rng, _window(rng), s.pub)
        others.append((_reserved(rng, sid, o) if reserved else [("open", sid, o)])
                      + [("end", sid, {"closed": True, "files": {}, "rc": 0, "complete": True})])
    queues = [named] + others
    events = []
    while any(queues):
        events.append(rng.choice([q for q in queues if q]).pop(0))
    if rng.random() < 0.5:
        name = rng.choice(PERIODS)
        events.insert(0, ("period-open", None, {"period": name, "lane": LANE, "by": LANE, "cap": 4,
                                                "circuit": rng.choice(CIRCUITS), "pinned": PIN,
                                                "key_sha512": hashlib.sha512(s.pub).hexdigest()}))
        if rng.random() < 0.5:
            roots = {sid: b["window"]["root"] for kind, sid, b in events if kind == "open"}
            windows = [{"root": roots[sid], "session": sid, "complete": True}
                       for kind, sid, b in events if kind == ("reserve" if reserved else "open")]
            events.append(("period-close", None, {"period": name, "lane": LANE, "by": LANE, "windows": windows,
                                                  "complete": True, "pinned": PIN}))
    return _bind(events)


def honest_rounds(rng: random.Random, name: str = "honest") -> tuple[dict, Session]:
    """An honest live window, read through its challenger's session: the key, the window, the session's files, and the
    round coins it opened, after each recorded round root (the auditor's ledger is `with_ledger`'s, once the case is
    mutated)."""
    rounds = rng.choice([1, 2, 3, 4, 8]) if name != "swapped" else rng.choice([2, 3, 4, 8])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    lo = 2 if name == "swapped" else 1 if name in _RECORDED else 0
    k = rng.randint(lo, rounds - 1 if name in _LEFT else rounds)
    salt = rng.randbytes(64) if name != "label" and rng.random() < 0.5 else None
    s = Session(rng, ch, [rng.randbytes(64) for _ in range(k)], salt=salt)
    return {"key": s.key(), "main": list(MAIN), "window": _window(rng), "session": s.files(),
            "roots": [r.hex() for r in s.rounds.roots], "openings": [o.to_json() for o in s.rounds.openings]}, s


def with_ledger(d: dict, s: Session, rng: random.Random) -> dict:
    """Case `d` with the auditor's ledger for its window, unless its mutation wrote its own."""
    if "ledger" not in d:
        d["ledger"] = ledger_lines(ledger_events(rng, d["window"], s), _key(0))
    return {k: d[k] for k in ("key", "main", "ledger", "window", "session", "roots", "openings")}


def _pinned(d: dict, name: str, rng: random.Random, ledger) -> bool:
    """Pin mutation `name` (`PIN_NAMES`) of case `d`, whose ledger `ledger(**keywords)` writes from `ledger_events`'s
    keywords; whether `name` is one."""
    match name:
        case "pin-not-main":
            d["ledger"] = ledger(pin=rng.randbytes(20).hex())
        case "repinned":
            d["ledger"] = ledger(pinned=rng.choice([c for c in MAIN if c != PIN]))
        case "no-pin":
            d["ledger"] = ledger(pinned=None)
        case "pin-form":
            d["ledger"] = ledger(pinned=rng.choice([PIN.upper(), PIN[:-2], PIN + "00", "", 5, [PIN]]))
        case "no-main":
            d["main"] = None
        case "main-form":
            d["main"] = rng.choice(["x", {"main": MAIN}, "\n".join(MAIN), MAIN + [PIN.upper()], MAIN + [None],
                                    [c + "\n" for c in MAIN]])
        case _:
            return False
    return True


def _set(lines: list[str], i: int, **changes) -> list[str]:
    doc = json.loads(lines[i])
    doc.update(changes)
    return lines[:i] + [canonical_json_text(doc)] + lines[i + 1:]


def _with_open(events: list, at: int, sid: str, body: dict) -> list:
    """`events` with an open of session `sid` inserted at `at`, and any period-close after it listing it too."""
    out = events[:at] + [("open", sid, body)] + events[at:]
    return [(k, s, {**b, "windows": [{"root": o["window"]["root"], "session": t, "complete": t != sid}
                                     for kk, t, o in out if kk == "open"], "complete": False})
            if k == "period-close" else (k, s, b) for k, s, b in out]


def _mutate(d: dict, s: Session, name: str, rng: random.Random) -> None:
    ch, roots, ops, files = s.ch, d["roots"], d["openings"], d["session"]
    c = ch.commitment
    j = rng.randrange(len(ops)) if ops else 0
    if _pinned(d, name, rng, lambda **kw: ledger_lines(ledger_events(rng, d["window"], s, **kw), _key(0))):
        return
    match name:
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "W": 3}, {**d["key"], "H": 0},
                                   {k: v for k, v in d["key"].items() if k != "W"}])
        case "seeded-coins":
            files["record"] = {**files["record"], "commitment": Co.Challenger(c.epoch, rng.randbytes).commitment.to_json()}
        case "in-process":
            other = Session(rng, Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes),
                            [bytes.fromhex(r) for r in roots], salt=None, seed=1)
            d["session"], s.sealed, s.ch = other.files(), other.lines, other.ch
            d["openings"] = [o.to_json() for o in other.rounds.openings]
        case "not-live":
            scheme, layout = rng.choice([
                ("verity/vllm/commit-coins/v1", Wd.live_families(len(roots) + 1)),
                (Wd.LIVE_SCHEME, ((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.DRAW, 1), ("pouw/rounds", len(roots) + 1)))])
            other = Session(rng, Ch.LiveChallenger(scheme, c.epoch, layout, rng.randbytes), [], salt=None)
            d["session"], s.sealed, s.ch = other.files(), other.lines, other.ch
            d["roots"], d["openings"] = [], []
        case "unsigned":
            files["close"] = None
        case "close-signature":
            sig = bytearray(bytes.fromhex(files["close"]["signature"]))
            sig[9] ^= 1
            files["close"] = {**files["close"], "signature": sig.hex()}
        case "tampered-log":
            i = len(files["log"]) - 1 - 2 * rng.randrange(len(ops))
            files["log"] = _set(files["log"], i - 1, root=_flip(json.loads(files["log"][i - 1])["root"]))
        case "unclosed":
            files["close"] = s.signed(ended=rng.choice(["the stream closed", "a client's frames are hello"]))
        case "no-ledger":
            d["ledger"] = None
        case "no-session":
            d["ledger"] = ledger_lines(ledger_events(rng, d["window"], s), _key(0))
            d["window"] = _window(rng)
        case "second-session":
            events = ledger_events(rng, d["window"], s)
            first = next(i for i, e in enumerate(events) if e[0] == "open")
            d["ledger"] = ledger_lines(_with_open(events, rng.randint(first + 1, len(events)), "s-2",
                                                  _open(rng, d["window"], s.pub)), _key(0))
        case "v1-ledger":
            events = [(k, sid, {**b, "window": {"commit": b["window"]["root"], "period": rng.choice(PERIODS)}}
                       if k == "open" else b) for k, sid, b in ledger_events(rng, d["window"], s) if k in ("open", "end")]
            d["ledger"] = ledger_lines(events, _key(0), schema=Ch.LEDGER_SCHEMA_V1)
        case "end-form":
            drop = rng.choice(["record_leaf", "commitment", "refused_hellos", "hellos"])
            files = {k: v for k, v in _end(s)["files"].items() if k != Ch.HELLOS_FILE}
            d["ledger"] = ledger_lines(ledger_events(rng, d["window"], s, **(
                {"files": files} if drop == "hellos" else {drop: None})), _key(0))
        case "record-leaf":
            events = ledger_events(rng, d["window"], s, record_leaf=RECORD_LEAF + 1)
            d["ledger"] = ledger_lines(events, _key(0))
        case "hellos":
            files["hellos"] = rng.choice([[canonical_json_text({"n": 0, "reason": "no hello"})], None, "x"])
        case "reused-leaf":
            events = ledger_events(rng, d["window"], s)
            leaves = leaves_of(events)
            i = rng.randrange(1, len(events))
            leaves[i] = rng.randint(0, leaves[i - 1])
            d["ledger"] = ledger_lines(events, _key(0), leaves)
        case "other-session":
            files["close"] = s.signed(sid="s-2")
        case "other-key":
            other = hashlib.sha512(lms.public_key(_key(1))).hexdigest()
            d["ledger"] = ledger_lines([(k, sid, {**b, "key_sha512": other} if k in ("open", "period-open") else b)
                                        for k, sid, b in ledger_events(rng, d["window"], s)], _key(1))
        case "forged-ledger":
            d["ledger"] = ledger_lines(ledger_events(rng, d["window"], s), _key(1))
        case "truncated":
            events = ledger_events(rng, d["window"], s)
            d["ledger"] = ledger_lines(events[:[e[:2] for e in events].index(("end", s.sid))], _key(0))
        case "unsealed":
            s.sealed = s.lines + s.lines[-1:]
        case "rewritten":
            r = rng.randbytes(64).hex()
            d["roots"] = roots[:j] + [r] + roots[j + 1:]
            d["openings"] = ops[:j] + [{**ops[j], "after": r}] + ops[j + 1:]
        case "roots-form":
            d["roots"] = rng.choice([None, "x", roots + ["zz"], roots + [rng.randbytes(32).hex()], [5], {"0": roots}])
        case "openings-list":
            d["openings"] = rng.choice([None, {"0": ops}, "x"])
        case "openings-form":
            k = rng.choice(["coin", "path", "label", "index", "epoch"])
            bad = {"coin": ["zz", None], "path": [None, 4, [rng.randbytes(33).hex()]], "label": [5, None],
                   "index": ["0", None, True, 1.5], "epoch": [str(c.epoch), None]}[k]
            d["openings"] = rng.choice([ops[:j] + [{**ops[j], k: rng.choice(bad)}] + ops[j + 1:], ops + [5]])
        case "twice":
            d["openings"] = ops[:j + 1] + [ops[j]] + ops[j + 1:]
        case "early":
            after = bytes.fromhex(roots[-1]) if roots else rng.randbytes(64)
            d["openings"] = ops + [ch.open(Wd.ROUND, len(roots), after).to_json()]
        case "early-root":
            d["roots"] = roots[:-1]
        case "never":
            d["openings"] = ops[:-1]
        case "swapped":
            d["openings"] = [ops[1], ops[0]] + ops[2:]
        case "after":
            d["openings"] = ops[:j] + [{**ops[j], "after": _flip(ops[j]["after"])}] + ops[j + 1:]
        case "root":
            d["roots"] = roots[:j] + [_flip(roots[j])] + roots[j + 1:]
        case "foreign-epoch":
            d["openings"] = _record(Wd.live_challenger(c.epoch + 1, c.family(Wd.ROUND), rng.randbytes),
                                    [bytes.fromhex(r) for r in roots])
        case "foreign":
            d["openings"] = _record(Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes),
                                    [bytes.fromhex(r) for r in roots])
        case "tampered":
            d["openings"] = ops[:j] + [{**ops[j], "salt": _flip(ops[j]["salt"])}] + ops[j + 1:]
        case "malformed":
            d["openings"] = ops[:j] + [{**ops[j], "coin": ops[j]["coin"][:-2]}] + ops[j + 1:]
        case "too-many":
            d["roots"] = roots + [rng.randbytes(64).hex() for _ in range(c.family(Wd.ROUND) + 1 - len(roots))]
        case "label":
            d["openings"] = [ch.open(Wd.SALT, 0, bytes.fromhex(roots[0])).to_json()] + ops[1:]
        case _:
            raise ValueError(name)


ROUNDS_NAMES = ["key-form", "seeded-coins", "in-process", "not-live", "unsigned", "close-signature", "tampered-log",
                "unclosed", "no-ledger", "no-session", "second-session", "v1-ledger", "end-form", "record-leaf", "hellos",
                "reused-leaf", "other-session",
                "other-key", "forged-ledger", "truncated", "unsealed", "rewritten",
                "roots-form", "openings-list", "openings-form", "twice", "early", "early-root", "never", "swapped",
                "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "too-many", "label", *PIN_NAMES]


def rounds_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `rounds` case: an honest live window, then mutation `name`."""
    d, s = honest_rounds(rng, name)
    if name != "honest":
        _mutate(d, s, name, rng)
    return with_ledger(d, s, rng)


def rounds_args(d: dict) -> tuple:
    """`core_check.rounds`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return d["key"], d["main"], d["ledger"], d["window"], d["session"], d["roots"], d["openings"], lms_verify


def session_id(rng: random.Random) -> str:
    """A session id of the auditor's gate's form, ``s<date>T<time>Z-<4 hex>``."""
    return f"s{rng.randrange(10 ** 8):08d}T{rng.randrange(10 ** 6):06d}Z-{rng.randbytes(2).hex()}"


def _salted(rng: random.Random, c: Ch.LiveCommitment, sid: str, salt: bytes | None, epoch: int | None = None
            ) -> Session:
    """Session `sid` over a fresh challenger of `c`'s layout (of epoch `epoch`, `c`'s by default), signed under key 0:
    its salt coin opened after root `salt` (none when None), then some of its round coins."""
    rounds = c.family(Wd.ROUND)
    ch = Wd.live_challenger(c.epoch if epoch is None else epoch, rounds, rng.randbytes)
    return Session(rng, ch, [rng.randbytes(64) for _ in range(rng.randint(0, rounds))], salt=salt, sid=sid)


def honest_salt(rng: random.Random) -> tuple[dict, Session]:
    """An honest live window's salt coin, read through its challenger's session: the session S the auditor's gate
    reserved, its open root R (`salt.window_open_root` of the window's weights' root, its round-0 request table's root
    and S), and the challenger's salt coin opened after R, then some round coins; the case holds the key, the window
    (``{R}``), the session's files, the weights' and requests' roots and the salt coin (the auditor's ledger is
    `with_salt_ledger`'s, once the case is mutated)."""
    ch = Wd.live_challenger(rng.randrange(4), rng.choice([1, 2, 4]), rng.randbytes)
    weights, requests, sid = rng.randbytes(64), rng.randbytes(64), session_id(rng)
    root = PS.window_open_root(weights, requests, sid)
    s = _salted(rng, ch.commitment, sid, root)
    return {"key": s.key(), "main": list(MAIN), "window": {"root": root.hex()}, "session": s.files(),
            "weights": weights.hex(), "requests": requests.hex(), "opening": s.salt[0].to_json()}, s


def salt_ledger(rng: random.Random, window: dict, s: Session, reserved: bool = True, k: int = 0, **events
                ) -> list[str]:
    """The auditor's ledger opening session `s` for `window`, every session reserved first unless not `reserved`,
    signed under key `k` (`events`: `ledger_events`'s other keywords)."""
    events = ledger_events(rng, window, s, reserved=reserved, **events)
    return ledger_lines(events, _key(k), leaves_of(events, s.sid))


def with_salt_ledger(d: dict, s: Session, rng: random.Random) -> dict:
    """Case `d` with the auditor's ledger reserving and opening session `s` for its window, unless its mutation wrote
    its own."""
    if "ledger" not in d:
        d["ledger"] = salt_ledger(rng, d["window"], s)
    return {k: d[k] for k in ("key", "main", "ledger", "window", "session", "weights", "requests", "opening")}


def _resalted(d: dict, s: Session, rng: random.Random, root: bytes) -> Session:
    """Case `d` as a session S of the same id opened on root `root`, its salt coin opened after it."""
    n = _salted(rng, s.ch.commitment, s.sid, root)
    d.update(window={"root": root.hex()}, session=n.files(), opening=n.salt[0].to_json())
    return n


def _mutate_salt(d: dict, s: Session, name: str, rng: random.Random) -> Session:
    """Mutation `name` of salt case `d` over session `s`, and the session the auditor's ledger opens."""
    c, op, files = s.ch.commitment, d["opening"], d["session"]
    root, w, t = (bytes.fromhex(x) for x in (d["window"]["root"], d["weights"], d["requests"]))
    if _pinned(d, name, rng, lambda **kw: salt_ledger(rng, d["window"], s, **kw)):
        return s
    match name:
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "W": 3}, {**d["key"], "H": 0}])
        case "no-ledger":
            d["ledger"] = None
        case "no-session":
            d["ledger"] = salt_ledger(rng, d["window"], s)
            d["window"] = _window(rng)
        case "forged-ledger":
            d["ledger"] = salt_ledger(rng, d["window"], s, k=1)
        case "in-process":
            other = Session(rng, Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes), [], salt=root, seed=1,
                            sid=s.sid)
            d["session"], d["opening"], s.sealed, s.ch = other.files(), other.salt[0].to_json(), other.lines, other.ch
        case "unsealed":
            s.sealed = s.lines + s.lines[-1:]
        case "other-session":
            files["close"] = s.signed(sid=session_id(rng))
        case "not-live":
            scheme, layout = rng.choice([("verity/vllm/commit-coins/v1", Wd.live_families(1)),
                                         (Wd.LIVE_SCHEME, ((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, 1)))])
            other = Session(rng, Ch.LiveChallenger(scheme, c.epoch, layout, rng.randbytes), [], salt=None, sid=s.sid)
            d["session"], s.sealed, s.ch = other.files(), other.lines, other.ch
        case "no-reserve":
            d["ledger"] = salt_ledger(rng, d["window"], s, reserved=False)
        case "form":
            k = rng.choice(["weights", "requests"])
            d[k] = rng.choice([None, d[k].upper(), d[k][:-2], 5, [d[k]]])
        case "weights":
            d["weights"] = rng.randbytes(64).hex()
        case "requests":
            d["requests"] = rng.randbytes(64).hex()
        case "session":
            # the window's open root under another session's id than the one the ledger reserved
            return _resalted(d, s, rng, PS.window_open_root(w, t, session_id(rng)))
        case "weights-root":
            # H1's salt before the window-open root: the session opened on the weights' root, its salt coin after it
            return _resalted(d, s, rng, w)
        case "received":
            other = _salted(rng, c, s.sid, rng.choice([None, rng.randbytes(64), w]))
            d["session"] = other.files()
            if other.salt:
                d["opening"] = other.salt[0].to_json()
            return other
        case "opening-form":
            k = rng.choice(["coin", "path", "label", "index", "epoch", "after"])
            bad = {"coin": ["zz", None], "path": [None, 4, [rng.randbytes(33).hex()]], "label": [5, None],
                   "index": ["0", None, True, 1.5], "epoch": [str(c.epoch), None], "after": ["zz", 3]}[k]
            d["opening"] = rng.choice([{**op, k: rng.choice(bad)}, None, [op]])
        case "after":
            d["opening"] = {**op, "after": _flip(op["after"])}
        case "label":
            d["opening"] = s.ch.open(Wd.DRAW, 0, root).to_json()
        case "foreign":
            d["opening"] = _salted(rng, c, s.sid, root).salt[0].to_json()
        case "foreign-epoch":
            d["opening"] = _salted(rng, c, s.sid, root, epoch=c.epoch + 1).salt[0].to_json()
        case "tampered":
            d["opening"] = {**op, "salt": _flip(op["salt"])}
        case "malformed":
            d["opening"] = {**op, "coin": op["coin"][:-2]}
        case _:
            raise ValueError(name)
    return s


SALT_NAMES = ["key-form", "no-ledger", "no-session", "forged-ledger", "in-process", "unsealed", "other-session",
              "not-live", "no-reserve", "form", "weights", "requests", "session", "weights-root", "received",
              "opening-form", "after", "label", "foreign", "foreign-epoch", "tampered", "malformed", *PIN_NAMES]


def salt_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `salt` case: an honest live window's salt coin, then mutation `name`."""
    d, s = honest_salt(rng)
    if name != "honest":
        s = _mutate_salt(d, s, name, rng)
    return with_salt_ledger(d, s, rng)


def salt_args(d: dict) -> tuple:
    """`core_check.salt`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (d["key"], d["main"], d["ledger"], d["window"], d["session"], d["weights"], d["requests"], d["opening"],
            lms_verify)
