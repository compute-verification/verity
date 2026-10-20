"""The rounds difftest's cases: `core_check.rounds`'s arguments, ``{"key", "ledger", "window", "session", "roots",
"openings"}``.

A case is an honest live window read through its challenger's session: a live epoch's coins (`window.live_challenger`),
the challenger's record of them signed at leaf 0 of a small LMS key (the window's pinned key), its release log (the
salt coin's events, sometimes, then each round root recorded in order and its coin opened after it, the challenger's
`LiveRecord`, in canonical JSON on an integer clock), the session's close signed at leaf 1 over the log's chain, and the
auditor's signed ledger opening that session for the window under the key's hash, among others, and ending it closed,
sealing the log the session's challenger left.  Then one mutation (`ROUNDS_NAMES`) aimed at one refusal of `rounds`: the
session's (an in-process challenger's coins, an unsigned or tampered log, a ledger signed under another key or cut before
the session's end, a second session for the window, …) or the coin rule's.  `tests/test_difftest.py` draws them for the harness, and
`verity/core/protocols/pouw/tests/test_pouw_core_check.py` holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import random

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

#: the mutations that need at least one recorded round, and those that need one round left unrecorded
_RECORDED = {"twice", "early-root", "never", "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "label",
             "openings-form", "swapped", "rewritten", "tampered-log"}
_LEFT = {"early"}
#: a small LMS parameter set: the record at leaf 0, the close at leaf 1, the ledger's entries at leaves 2 and 3
PARAMS = lms.Params(64, 4, 2)
HOST = "vy-challenger-test"


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
    """A challenger's session over `ch`'s coins as its host leaves it, its record and close signed under key `seed`."""

    def __init__(self, rng: random.Random, ch: Ch.LiveChallenger, roots: list[bytes], salt: bool, seed: int = 0,
                 sid: str = "s-1") -> None:
        self.ch, self.k, self.sid = ch, _key(seed), sid
        self.pub = lms.public_key(self.k)
        ch.sign_record(HOST, self.pub, lambda m: lms.sign_at(self.k, 0, m).encode())
        clock = itertools.count(rng.choice([0, 1, 10 ** 12])).__next__
        self.lines = [canonical_json_text({"event": "commitment", "root": ch.commitment.root.hex()})]
        records = [Ch.LiveRecord(ch, Wd.SALT, clock)] if salt else []
        for rec in records:
            rec.record(rng.randbytes(64))
        self.rounds = Ch.LiveRecord(ch, Wd.ROUND, clock) if ch.commitment.family(Wd.ROUND) else None
        for r in roots:
            self.rounds.record(r)
        records += [] if self.rounds is None else [self.rounds]
        self.lines += [canonical_json_text(e) for rec in records for e in rec.log]
        self.sealed = self.lines
        self.close = self.signed()

    def signed(self, lines: list[str] | None = None, sid: str | None = None, ended: str | None = None) -> dict:
        lines = self.lines if lines is None else lines
        sid = self.sid if sid is None else sid
        chain = Ch.log_chain(lines)
        m = Ch.close_message(self.pub[:16], HOST, self.ch.commitment, sid, len(lines), chain, ended)
        return Ch.ChallengerClose(sid, len(lines), chain, ended, lms.sign_at(self.k, 1, m).encode()).to_json()

    def files(self) -> dict:
        return {"close": self.close, "log": list(self.lines), "record": self.ch.record.to_json()}

    def key(self) -> dict:
        return Ch.ChallengerKey(PARAMS.W, PARAMS.H, self.pub).to_json()


PERIODS = ["p-0", "smoke-20261020", "2026-10-20T09:00Z", "a"]


def _window(rng: random.Random) -> dict:
    return {"commit": rng.randbytes(64).hex(), "period": rng.choice(PERIODS)}


def _open(rng: random.Random, window: dict, public: bytes) -> dict:
    return {"window": window, "host": HOST, "pinned": rng.randbytes(20).hex(),
            "key_sha512": hashlib.sha512(public).hexdigest(), "port": 7811}


def ledger_lines(events: list[tuple[str, str, dict]], k: lms.PrivateKey) -> list[str]:
    """The auditor's ledger of `events` (kind, session, body), chained and signed under `k` at leaves 2 and 3."""
    out: list[str] = []
    for i, (kind, sid, body) in enumerate(events):
        entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest() if out else "0" * 128,
                 "schema": Ch.LEDGER_SCHEMA, "seq": i, "session": sid}
        m = Ch.ledger_message(lms.public_key(k)[:16], entry)
        out.append(canonical_json_text({**entry, "sig": lms.sign_at(k, 2 + i % 2, m).encode().hex()}))
    return out


def ledger_events(rng: random.Random, window: dict, s: Session) -> list[tuple[str, str, dict]]:
    """The window's session `s` opened and ended closed, sealing the log its challenger left, among up to two others."""
    named = [("open", s.sid, _open(rng, window, s.pub)),
             ("end", s.sid, {"closed": True, "files": {Ch.LOG_FILE: Ch.log_digest(s.sealed)}, "rc": 0})]
    others = [[("open", f"o-{i}", _open(rng, _window(rng), s.pub)), ("end", f"o-{i}", {"closed": True, "files": {}, "rc": 0})]
              for i in range(rng.randint(0, 2))]
    queues = [named] + others
    events = []
    while any(queues):
        events.append(rng.choice([q for q in queues if q]).pop(0))
    return events


def honest_rounds(rng: random.Random, name: str = "honest") -> tuple[dict, Session]:
    """An honest live window, read through its challenger's session: the key, the window, the session's files, and the
    round coins it opened, after each recorded round root (the auditor's ledger is `with_ledger`'s, once the case is
    mutated)."""
    rounds = rng.choice([1, 2, 3, 4, 8]) if name != "swapped" else rng.choice([2, 3, 4, 8])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    lo = 2 if name == "swapped" else 1 if name in _RECORDED else 0
    k = rng.randint(lo, rounds - 1 if name in _LEFT else rounds)
    s = Session(rng, ch, [rng.randbytes(64) for _ in range(k)], salt=name != "label" and rng.random() < 0.5)
    return {"key": s.key(), "window": _window(rng), "session": s.files(),
            "roots": [r.hex() for r in s.rounds.roots], "openings": [o.to_json() for o in s.rounds.openings]}, s


def with_ledger(d: dict, s: Session, rng: random.Random) -> dict:
    """Case `d` with the auditor's ledger for its window, unless its mutation wrote its own."""
    if "ledger" not in d:
        d["ledger"] = ledger_lines(ledger_events(rng, d["window"], s), _key(0))
    return {k: d[k] for k in ("key", "ledger", "window", "session", "roots", "openings")}


def _set(lines: list[str], i: int, **changes) -> list[str]:
    doc = json.loads(lines[i])
    doc.update(changes)
    return lines[:i] + [canonical_json_text(doc)] + lines[i + 1:]


def _mutate(d: dict, s: Session, name: str, rng: random.Random) -> None:
    ch, roots, ops, files = s.ch, d["roots"], d["openings"], d["session"]
    c = ch.commitment
    j = rng.randrange(len(ops)) if ops else 0
    match name:
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "W": 3}, {**d["key"], "H": 0},
                                   {k: v for k, v in d["key"].items() if k != "W"}])
        case "seeded-coins":
            files["record"] = {**files["record"], "commitment": Co.Challenger(c.epoch, rng.randbytes).commitment.to_json()}
        case "in-process":
            other = Session(rng, Wd.live_challenger(c.epoch, c.family(Wd.ROUND), rng.randbytes),
                            [bytes.fromhex(r) for r in roots], salt=False, seed=1)
            d["session"], s.sealed = other.files(), other.lines
            d["openings"] = [o.to_json() for o in other.rounds.openings]
        case "not-live":
            scheme, layout = rng.choice([
                ("verity/vllm/commit-coins/v1", Wd.live_families(len(roots) + 1)),
                (Wd.LIVE_SCHEME, ((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.DRAW, 1), ("pouw/rounds", len(roots) + 1)))])
            other = Session(rng, Ch.LiveChallenger(scheme, c.epoch, layout, rng.randbytes), [], salt=False)
            d["session"], s.sealed = other.files(), other.lines
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
            d["window"] = {**d["window"], "period": "p-other"}
        case "second-session":
            events = ledger_events(rng, d["window"], s)
            events.insert(rng.randint(0, len(events)), ("open", "s-2", _open(rng, d["window"], s.pub)))
            d["ledger"] = ledger_lines(events, _key(0))
        case "other-session":
            files["close"] = s.signed(sid="s-2")
        case "other-key":
            other = hashlib.sha512(lms.public_key(_key(1))).hexdigest()
            d["ledger"] = ledger_lines([(k, sid, {**b, "key_sha512": other} if k == "open" else b)
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
                "unclosed", "no-ledger", "no-session", "second-session", "other-session", "other-key", "forged-ledger",
                "truncated", "unsealed", "rewritten",
                "roots-form", "openings-list", "openings-form", "twice", "early", "early-root", "never", "swapped",
                "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "too-many", "label"]


def rounds_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `rounds` case: an honest live window, then mutation `name`."""
    d, s = honest_rounds(rng, name)
    if name != "honest":
        _mutate(d, s, name, rng)
    return with_ledger(d, s, rng)


def rounds_args(d: dict) -> tuple:
    """`core_check.rounds`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return d["key"], d["ledger"], d["window"], d["session"], d["roots"], d["openings"], lms_verify
