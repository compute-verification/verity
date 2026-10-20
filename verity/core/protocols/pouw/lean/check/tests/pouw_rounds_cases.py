"""The rounds difftest's cases: `core_check.rounds`'s arguments, ``{"key", "auditor", "window", "session", "roots",
"openings"}``.

A case is an honest live window read through its challenger's session: a live epoch's coins (`window.live_challenger`),
the challenger's record of them signed at leaf 0 of a small LMS key (the window's pinned key), its release log (the
salt coin's events, sometimes, then each round root recorded in order and its coin opened after it, the challenger's
`LiveRecord`, in canonical JSON on an integer clock), the session's close signed at leaf 1 over the log's chain, and the
auditor's record naming that session for the window under the key's hash.  Then one mutation (`ROUNDS_NAMES`) aimed at
one refusal of `rounds`: the session's (an in-process challenger's coins, an unsigned or tampered log, a second session for
the window, …) or the coin rule's.  `tests/test_difftest.py` draws them for the harness, and
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
#: a small LMS parameter set: the record at leaf 0, the close at leaf 1
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


def _window(rng: random.Random) -> dict:
    return {"commit": rng.randbytes(64).hex(), "period": rng.choice([0, 1, 7, 1 << 40]),
            "pouw_salt": rng.randbytes(64).hex(), "stream": rng.randbytes(64).hex()}


def _entry(rng: random.Random, window: dict, sid: str, public: bytes) -> dict:
    return {**window, "code": rng.randbytes(20).hex(), "key": hashlib.sha512(public).hexdigest(), "session": sid}


def honest_rounds(rng: random.Random, name: str = "honest") -> tuple[dict, Session]:
    """An honest live window, read through its challenger's session: the key, the auditor's record, the window, the
    session's files, and the round coins it opened, after each recorded round root."""
    rounds = rng.choice([1, 2, 3, 4, 8]) if name != "swapped" else rng.choice([2, 3, 4, 8])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    lo = 2 if name == "swapped" else 1 if name in _RECORDED else 0
    k = rng.randint(lo, rounds - 1 if name in _LEFT else rounds)
    s = Session(rng, ch, [rng.randbytes(64) for _ in range(k)], salt=name != "label" and rng.random() < 0.5)
    window = _window(rng)
    auditor = [_entry(rng, _window(rng), f"o-{i}", s.pub) for i in range(rng.randint(0, 2))]
    auditor.insert(rng.randint(0, len(auditor)), _entry(rng, window, s.sid, s.pub))
    return {"key": s.key(), "auditor": auditor, "window": window, "session": s.files(),
            "roots": [r.hex() for r in s.rounds.roots], "openings": [o.to_json() for o in s.rounds.openings]}, s


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
            d["session"] = other.files()
            d["openings"] = [o.to_json() for o in other.rounds.openings]
        case "not-live":
            scheme, layout = rng.choice([
                ("verity/vllm/commit-coins/v1", Wd.live_families(len(roots) + 1)),
                (Wd.LIVE_SCHEME, ((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.ROUND, len(roots) + 1))),
                (Wd.LIVE_SCHEME, ((Wd.SALT, 1), (Wd.DRAW, 1), ("pouw/rounds", len(roots) + 1)))])
            d["session"] = Session(rng, Ch.LiveChallenger(scheme, c.epoch, layout, rng.randbytes), [], salt=False).files()
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
        case "no-auditor":
            d["auditor"] = None
        case "no-session":
            d["window"] = {**d["window"], rng.choice(["commit", "pouw_salt", "stream"]): rng.randbytes(64).hex()}
        case "second-session":
            d["auditor"] = d["auditor"] + [_entry(rng, d["window"], "s-2", s.pub)]
        case "other-session":
            files["close"] = s.signed(sid="s-2")
        case "other-key":
            d["auditor"] = [{**e, "key": hashlib.sha512(lms.public_key(_key(1))).hexdigest()}
                            if e["session"] == s.sid else e for e in d["auditor"]]
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
                "unclosed", "no-auditor", "no-session", "second-session", "other-session", "other-key", "rewritten",
                "roots-form", "openings-list", "openings-form", "twice", "early", "early-root", "never", "swapped",
                "after", "root", "foreign-epoch", "foreign", "tampered", "malformed", "too-many", "label"]


def rounds_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `rounds` case: an honest live window, then mutation `name`."""
    d, s = honest_rounds(rng, name)
    if name != "honest":
        _mutate(d, s, name, rng)
    return d


def rounds_args(d: dict) -> tuple:
    """`core_check.rounds`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return d["key"], d["auditor"], d["window"], d["session"], d["roots"], d["openings"], lms_verify
