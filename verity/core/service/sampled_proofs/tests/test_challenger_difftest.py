"""The proof service's challenger in Lean (`SampledProofs.Challenger`, the verifier of record) against its Python mirror
(`verity.core.protocols.sampled_proofs.challenger`), through `sampled-proofs-difftest`'s challenger pieces: a coin
commitment and an opening as `from_json` reads them, `live_reason` and `order_reason`; and the challenger's signed record
(`SampledProofs.ChallengerRecord`): the verifier's key and a record as `from_json` reads them, with `record_message`, and
`trusted_reason`.

The cases are a challenger's honest coins over drawn layouts (labels of several clients: PoUW's, vLLM's, memory's), then
one mutation each (`COMMITMENT`, `OPENING`, `REASON`, `ORDER`): a coin opened twice, one opened before its root was
recorded, a foreign epoch's or challenger's, a tampered salt or path, a coin the layout doesn't hold, each malformed
form. The record's (`KEY`, `RECORD`, `TRUSTED`) are a record the challenger signed with a small LMS key, then a missing
record, a prover's commitment or a prover's key, a tampered root, epoch, host or signature, another parameter set, each
malformed form. The session's (`CLOSE`, `RELEASED`, `LEDGER`, `SESSION`) are a session a small key signed and the
auditor's ledger opening and ending it among others, then a close, log or ledger entry edited, dropped, re-chained or
signed under another key, a ledger signed under its own key, one cut before the session's end, a second session for the
window, a record on another host, a log the ledger didn't seal, each malformed form. Each piece answers
``{"refused": why}`` where the challenger refuses, so the reasons are compared too.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments.identity import canonical_json_text
from verity.ml.crypto import lms

PACKAGE = Path(__file__).resolve().parents[1]
PIECES = "sampled-proofs-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LABELS = ["pouw/salt", "pouw/draw", "pouw/round", "vllm/positions", "vllm/identities", "memory/audit", "a", "x-1"]
SCHEMES = ["verity/pouw/epoch-coins/live/v1", "verity/vllm/commit-coins/v1", "verity/memory/audit-coins/v1", "m"]


def _hex(s: Any) -> bytes:
    if type(s) is not str or Ch._HEX.fullmatch(s) is None:
        raise D.Reject("not lowercase hex")
    return bytes.fromhex(s)


def _request(j: Any, keys: list[str]) -> None:
    if not isinstance(j, dict) or sorted(j) != keys:
        raise D.Reject(f"a request's keys are {', '.join(keys)}")


def _refusing(f) -> Any:
    try:
        return f()
    except Ch.CoinError as e:
        return {"refused": str(e)}


def commitment(j: Any) -> Any:
    def go():
        c = Ch.LiveCommitment.from_json(j)
        return {**c.to_json(), "count": c.count, "domain": Ch.live_domain(c.scheme, c.epoch, c.families).domain_id.hex()}
    return _refusing(go)


def opening(j: Any) -> Any:
    return _refusing(lambda: Ch.LiveOpening.from_json(j).to_json())


def reason(j: Any) -> Any:
    _request(j, ["after", "commitment", "index", "label", "opening"])
    if type(j["label"]) is not str:
        raise D.Reject("label is not a string")
    if type(j["index"]) is not int:
        raise D.Reject("index is not an integer")
    after = _hex(j["after"])

    def go():
        c, o = Ch.LiveCommitment.from_json(j["commitment"]), Ch.LiveOpening.from_json(j["opening"])
        return {"reason": Ch.live_reason(c, o, j["label"], j["index"], after)}
    return _refusing(go)


def order(j: Any) -> Any:
    _request(j, ["commitment", "label", "openings", "roots"])
    if type(j["label"]) is not str:
        raise D.Reject("label is not a string")
    if type(j["roots"]) is not list:
        raise D.Reject("roots is not a list")
    roots = [_hex(r) for r in j["roots"]]
    if type(j["openings"]) is not list:
        raise D.Reject("openings is not a list")

    def go():
        c = Ch.LiveCommitment.from_json(j["commitment"])
        return {"reason": Ch.order_reason(c, j["label"], roots, [Ch.LiveOpening.from_json(o) for o in j["openings"]])}
    return _refusing(go)


# ---- the cases ----------------------------------------------------------------------------------------------------------

def _flip(h: str, at: int = 0) -> str:
    b = bytearray(bytes.fromhex(h))
    b[at % len(b)] ^= 1
    return b.hex()


def _challenger(rng: random.Random, families=None, epoch=None, scheme=None) -> Ch.LiveChallenger:
    if families is None:
        families = [(label, rng.choice([1, 1, 2, 3, 4, 5, 8, 9, 17])) for label in rng.sample(LABELS, rng.randint(1, 4))]
    return Ch.LiveChallenger(rng.choice(SCHEMES) if scheme is None else scheme,
                             rng.choice([0, 1, 7, 1 << 40]) if epoch is None else epoch, families, rng.randbytes)


def _coin(rng: random.Random, c: Ch.LiveCommitment) -> tuple[str, int]:
    label, n = rng.choice(c.families)
    return label, rng.randrange(n)


COMMITMENT_NAMES = ["honest", "keys", "extra", "families-type", "family-type", "root-upper", "root-odd", "root-short",
                    "scheme", "scheme-type", "epoch", "epoch-type", "count", "count-type", "label", "family-length",
                    "empty", "distinct", "too-many"]


def commitment_case(rng: random.Random, name: str) -> Any:
    d = _challenger(rng).commitment.to_json()
    fam = d["families"]
    match name:
        case "honest":
            pass
        case "keys":
            del d[rng.choice(sorted(d))]
        case "extra":
            d["count"] = 3
        case "families-type":
            d["families"] = rng.choice([None, "a", {"a": 1}])
        case "family-type":
            d["families"] = fam[:-1] + [rng.choice(["a", None, 1, {"a": 1}])]
        case "root-upper":
            d["root"] = d["root"].upper()
        case "root-odd":
            d["root"] = d["root"][:-1]
        case "root-short":
            d["root"] = d["root"][:rng.choice([0, 2, 126])]
        case "scheme":
            d["scheme"] = rng.choice(["", "Verity/x", "a b", "a_b", "é"])
        case "scheme-type":
            d["scheme"] = rng.choice([None, 1, ["m"]])
        case "epoch":
            d["epoch"] = rng.choice([-1, -(1 << 40)])
        case "epoch-type":
            d["epoch"] = rng.choice([True, "1", None])
        case "count":
            fam[-1][1] = rng.choice([0, -1])
        case "count-type":
            fam[-1][1] = rng.choice([True, "1", None])
        case "label":
            fam[-1][0] = rng.choice(["", "A", "a b", 5, None])
        case "family-length":
            fam[-1] = rng.choice([fam[-1][:1], fam[-1] + [1], []])
        case "empty":
            d["families"] = []
        case "distinct":
            d["families"] = fam + [[fam[0][0], 1]]
        case "too-many":
            d["families"] = [["a", Ch.MAX_COINS], ["b", rng.randint(1, 3)]]
    return d


OPENING_NAMES = ["honest", "keys", "extra", "epoch", "index", "label", "after", "coin", "salt", "path-type", "path-item",
                 "path-short-item"]


def opening_case(rng: random.Random, name: str) -> Any:
    ch = _challenger(rng)
    o = ch.open(*_coin(rng, ch.commitment), rng.randbytes(64)).to_json()
    match name:
        case "honest":
            pass
        case "keys":
            del o[rng.choice(sorted(o))]
        case "extra":
            o["root"] = "00"
        case "epoch" | "index":
            o[name] = rng.choice([True, "1", None, [1]])
        case "label":
            o["label"] = rng.choice([1, None, ["a"]])
        case "after" | "coin" | "salt":
            o[name] = rng.choice([o[name].upper() + "AB", o[name] + "0", None, 5])
        case "path-type":
            o["path"] = rng.choice([None, "00", {"0": "00"}])
        case "path-item":
            o["path"] = o["path"] + [rng.choice([None, 5, "AB" * 64])]
        case "path-short-item":
            o["path"] = o["path"] + [rng.randbytes(rng.choice([0, 32, 63, 65])).hex()]
    return o


REASON_NAMES = ["honest", "epoch", "other-coin", "coin-short", "salt-short", "salt", "coin", "path", "path-length",
                "after", "no-coin", "index-range", "foreign", "root", "scheme", "commitment-form", "opening-form",
                "request"]


def reason_case(rng: random.Random, name: str) -> Any:
    ch = _challenger(rng)
    c = ch.commitment
    label, index = _coin(rng, c)
    after = rng.randbytes(rng.choice([0, 32, 64]))
    o = ch.open(label, index, after).to_json()
    d = {"after": after.hex(), "commitment": c.to_json(), "index": index, "label": label, "opening": o}
    match name:
        case "honest":
            pass
        case "epoch":
            d["opening"] = {**o, "epoch": o["epoch"] + rng.choice([1, -1, 1 << 50])}
        case "other-coin":
            others = [(lb, i) for lb, n in c.families for i in range(n) if (lb, i) != (label, index)]
            if others:
                d["label"], d["index"] = rng.choice(others)
            else:
                d["opening"] = {**o, "index": index + 1}
        case "coin-short":
            d["opening"] = {**o, "coin": o["coin"][:rng.choice([-2, 64])]}
        case "salt-short":
            d["opening"] = {**o, "salt": o["salt"] + "00"}
        case "salt" | "coin":
            d["opening"] = {**o, name: _flip(o[name], rng.randrange(64))}
        case "path":
            if o["path"]:
                j = rng.randrange(len(o["path"]))
                d["opening"] = {**o, "path": o["path"][:j] + [_flip(o["path"][j])] + o["path"][j + 1:]}
            else:
                d["opening"] = {**o, "path": [rng.randbytes(64).hex()]}
        case "path-length":
            longer = o["path"] + [rng.randbytes(64).hex()]
            d["opening"] = {**o, "path": rng.choice([o["path"][:-1], longer]) if o["path"] else longer}
        case "after":
            d["after"] = rng.randbytes(64).hex() if not after or rng.random() < 0.5 else _flip(after.hex())
        case "no-coin":
            d["label"] = rng.choice([lb for lb in LABELS + ["nope"] if c.family(lb) is None])
        case "index-range":
            d["index"] = rng.choice([-1, c.family(label), c.family(label) + 5])
        case "foreign":
            d["opening"] = _challenger(rng, c.families, c.epoch, c.scheme).open(label, index, after).to_json()
        case "root":
            d["commitment"] = {**d["commitment"], "root": _flip(d["commitment"]["root"])}
        case "scheme":
            d["commitment"] = {**d["commitment"], "scheme": rng.choice([s for s in SCHEMES if s != c.scheme])}
        case "commitment-form":
            d["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "opening-form":
            d["opening"] = opening_case(rng, rng.choice(OPENING_NAMES[1:]))
        case "request":
            k = rng.choice(["after", "index", "label", None])
            if k is None:
                del d[rng.choice(sorted(d))]
            else:
                d[k] = rng.choice({"after": [None, "AB", True, 1], "index": [None, "AB", True, "1"],
                                   "label": [None, True, 1, ["a"]]}[k])
    return d


ORDER_NAMES = ["honest", "twice", "early", "early-root", "never", "swapped", "after", "root", "foreign-epoch", "foreign",
               "tampered", "malformed", "too-many", "label", "no-family", "commitment-form", "opening-form", "request"]


def order_case(rng: random.Random, name: str) -> Any:
    need = 2 if name == "swapped" else 1
    families = [(label, rng.choice([1, 2, 3, 4, 5, 8])) for label in rng.sample(LABELS, rng.randint(1, 3))]
    families[0] = (families[0][0], max(families[0][1], need + 1))
    ch = _challenger(rng, families)
    c, label = ch.commitment, families[0][0]
    n = c.family(label)
    rec = Ch.LiveRecord(ch, label)
    k = rng.randint(need, n - 1 if name == "early" else n)
    for _ in range(k):
        rec.record(rng.randbytes(64))
    roots, ops = [r.hex() for r in rec.roots], [o.to_json() for o in rec.openings]
    d = {"commitment": c.to_json(), "label": label, "openings": ops, "roots": roots}
    j = rng.randrange(len(ops))

    def again(epoch=c.epoch):
        other = Ch.LiveRecord(_challenger(rng, families, epoch, c.scheme), label)
        return [other.record(bytes.fromhex(r)).to_json() for r in roots]
    match name:
        case "honest":
            pass
        case "twice":
            d["openings"] = ops[:j + 1] + [ops[j]] + ops[j + 1:]
        case "early":
            d["openings"] = ops + [ch.open(label, k, bytes.fromhex(roots[-1])).to_json()]
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
            d["openings"] = again(c.epoch + 1)
        case "foreign":
            d["openings"] = again()
        case "tampered":
            d["openings"] = ops[:j] + [{**ops[j], "salt": _flip(ops[j]["salt"])}] + ops[j + 1:]
        case "malformed":
            d["openings"] = ops[:j] + [{**ops[j], "coin": ops[j]["coin"][:-2]}] + ops[j + 1:]
        case "too-many":
            d["roots"] = roots + [rng.randbytes(64).hex() for _ in range(n + 1 - k)]
        case "label":
            others = [lb for lb, _ in families[1:]]
            other = ch.open(others[0], 0, bytes.fromhex(roots[0])) if others else None
            d["openings"] = [other.to_json() if other else {**ops[0], "label": "nope"}] + ops[1:]
        case "no-family":
            d["label"] = rng.choice([lb for lb in LABELS + ["nope"] if c.family(lb) is None])
        case "commitment-form":
            d["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "opening-form":
            d["openings"] = ops[:j] + [opening_case(rng, rng.choice(OPENING_NAMES[1:]))] + ops[j + 1:]
        case "request":
            k_ = rng.choice(["label", "roots", "openings", None])
            if k_ is None:
                del d[rng.choice(sorted(d))]
            elif k_ == "roots":
                d["roots"] = rng.choice([None, roots + ["AB"], roots + [5], "00"])
            else:
                d[k_] = rng.choice([None, 5, "x" if k_ == "openings" else ["x"]])
    return d


def _cases(make, names):
    return st.builds(make, st.randoms(use_true_random=False), st.sampled_from(["honest"] * (len(names) // 3) + names))


# ---- the challenger's signed record ---------------------------------------------------------------------------------

#: small LMS parameter sets (W, H), each key made in milliseconds
PARAMS = [(1, 2), (2, 3), (4, 2), (8, 1)]
HOSTS = ["vy-nebius-cpu-1", "auditor.example", "h", "A_1"]


@functools.cache
def _key(w: int, h: int, seed: int) -> lms.PrivateKey:
    return lms.keygen(lms.Params(64, w, h), bytes([seed]) * 64)


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


def _signed(rng: random.Random) -> tuple[Ch.ChallengerKey, Ch.LiveChallenger, Ch.ChallengerRecord]:
    """A challenger's record, signed at a drawn leaf of a small key, and the verifier's key for it."""
    w, h = rng.choice(PARAMS)
    k = _key(w, h, 0)
    ch = _challenger(rng)
    pub = lms.public_key(k)
    q = rng.randrange(2 ** h)
    ch.sign_record(rng.choice(HOSTS), pub, lambda m: lms.sign_at(k, q, m).encode())
    return Ch.ChallengerKey(w, h, pub), ch, ch.record


def key(j: Any) -> Any:
    return _refusing(lambda: Ch.ChallengerKey.from_json(j).to_json())


def record(j: Any) -> Any:
    _request(j, ["I", "record"])
    I = _hex(j["I"]) if type(j["I"]) is str and Ch._HEX.fullmatch(j["I"]) else None
    if I is None:
        raise D.Reject("I is not lowercase hex")

    def go():
        r = Ch.ChallengerRecord.from_json(j["record"])
        return {"message": Ch.record_message(I, r.host, r.commitment).hex(), "record": r.to_json()}
    return _refusing(go)


def trusted(j: Any) -> Any:
    _request(j, ["commitment", "key", "record"])

    def go():
        k, c = Ch.ChallengerKey.from_json(j["key"]), Ch.LiveCommitment.from_json(j["commitment"])
        return {"reason": Ch.trusted_reason(k, j["record"], c, lms_verify)}
    return _refusing(go)


KEY_NAMES = ["honest", "keys", "extra", "W", "H", "H-type", "public-short", "public-upper", "public-type"]


def key_case(rng: random.Random, name: str) -> Any:
    w, h = rng.choice(PARAMS)
    d = {"H": h, "W": w, "public": lms.public_key(_key(w, h, 0)).hex()}
    match name:
        case "honest":
            pass
        case "keys":
            del d[rng.choice(sorted(d))]
        case "extra":
            d["N"] = 64
        case "W":
            d["W"] = rng.choice([0, 3, 16, -1])
        case "H":
            d["H"] = rng.choice([0, 31, -2])
        case "H-type":
            d[rng.choice(["H", "W"])] = rng.choice([True, "2", None, [2]])
        case "public-short":
            d["public"] = d["public"][:rng.choice([0, 32, 158])] + rng.choice(["", "00" * 2])
        case "public-upper":
            d["public"] = d["public"].upper()
        case "public-type":
            d["public"] = rng.choice([None, 5, ["00"]])
    return d


RECORD_NAMES = ["honest", "keys", "extra", "host-empty", "host-chars", "host-type", "signature", "commitment-form", "I-short",
                "request"]


def record_case(rng: random.Random, name: str) -> Any:
    _, _, r = _signed(rng)
    d = r.to_json()
    j = {"I": rng.randbytes(16).hex(), "record": d}
    match name:
        case "honest":
            pass
        case "keys":
            del d[rng.choice(sorted(d))]
        case "extra":
            d["public"] = "00" * 80
        case "host-empty":
            d["host"] = ""
        case "host-chars":
            d["host"] = rng.choice(["a b", "a/b", "é", "a:1", "x\n"])
        case "host-type":
            d["host"] = rng.choice([None, 1, ["h"]])
        case "signature":
            d["signature"] = rng.choice([d["signature"].upper(), d["signature"] + "0", None, 7])
        case "commitment-form":
            d["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "I-short":
            j["I"] = rng.randbytes(rng.choice([0, 15, 17, 64])).hex()
        case "request":
            k = rng.choice(["I", None])
            if k is None:
                del j[rng.choice(sorted(j))]
            else:
                j["I"] = rng.choice([None, "AB", 5, "0"])
    return j


TRUSTED_NAMES = ["honest", "missing", "prover-commitment", "prover-key", "root", "epoch", "host", "signature", "signature-short",
                 "leaf", "params", "record-form", "key-form", "commitment-form", "request"]


def _flip_root(c: dict) -> dict:
    return {**c, "root": _flip(c["root"], 3)}


def trusted_case(rng: random.Random, name: str) -> Any:
    k, ch, r = _signed(rng)
    d = {"commitment": ch.commitment.to_json(), "key": k.to_json(), "record": r.to_json()}
    rec = d["record"]
    match name:
        case "honest":
            pass
        case "missing":
            d["record"] = None
        case "prover-commitment":
            d["commitment"] = _challenger(rng, ch.commitment.families, ch.commitment.epoch, ch.commitment.scheme).commitment.to_json()
        case "prover-key":
            other = _key(k.W, k.H, 1)
            if rng.random() < 0.5:
                d["key"] = {**d["key"], "public": lms.public_key(other).hex()}
            else:
                m = Ch.record_message(other.I, r.host, ch.commitment)
                d["record"] = {**rec, "signature": lms.sign_at(other, rng.randrange(2 ** k.H), m).encode().hex()}
        case "root":
            d["record"] = {**rec, "commitment": _flip_root(rec["commitment"])}
            d["commitment"] = rng.choice([d["commitment"], d["record"]["commitment"]])
        case "epoch":
            d["record"] = {**rec, "commitment": {**rec["commitment"], "epoch": rec["commitment"]["epoch"] + 1}}
        case "host":
            d["record"] = {**rec, "host": rng.choice([h for h in HOSTS if h != rec["host"]])}
        case "signature":
            d["record"] = {**rec, "signature": _flip(rec["signature"], rng.randrange(len(rec["signature"]) // 2))}
        case "signature-short":
            d["record"] = {**rec, "signature": rec["signature"][:rng.choice([0, 8, len(rec["signature"]) - 2])]}
        case "leaf":
            d["record"] = {**rec, "signature": (2 ** k.H + rng.randrange(4)).to_bytes(4, "big").hex() + rec["signature"][8:]}
        case "params":
            others = [p for p in PARAMS if p != (k.W, k.H)]
            w, h = rng.choice(others)
            d["key"] = {**d["key"], "W": w, "H": h}
        case "record-form":
            d["record"] = record_case(rng, rng.choice(RECORD_NAMES[1:-3]))["record"]
        case "key-form":
            d["key"] = key_case(rng, rng.choice(KEY_NAMES[1:]))
        case "commitment-form":
            d["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "request":
            del d[rng.choice(sorted(d))]
    return d


# ---- the challenger's session -------------------------------------------------------------------------------------------

SESSIONS = ["s-1", "s-20261020-1", "A.b_c-9", "x" * Ch.MAX_SESSION_ID]


def _roots(roots: dict[str, tuple[bytes, ...]]) -> dict[str, list[str]]:
    return {label: [r.hex() for r in rs] for label, rs in roots.items()}


def close(j: Any) -> Any:
    _request(j, ["I", "close", "commitment", "host"])
    I = _hex(j["I"])
    if type(j["host"]) is not str:
        raise D.Reject("host is not a string")

    def go():
        c = Ch.ChallengerClose.from_json(j["close"])
        m = Ch.close_message(I, j["host"], Ch.LiveCommitment.from_json(j["commitment"]), c.session, c.lines, c.chain,
                             c.ended)
        return {"close": c.to_json(), "message": m.hex()}
    return _refusing(go)


def released(j: Any) -> Any:
    _request(j, ["commitment", "log"])
    if type(j["log"]) is not list or not all(type(line) is str for line in j["log"]):
        raise D.Reject("log is not a list of strings")

    def go():
        roots = Ch.released_roots_of(Ch.LiveCommitment.from_json(j["commitment"]), j["log"])
        return {"chain": Ch.log_chain(j["log"]).hex(), "roots": _roots(roots)}
    return _refusing(go)


def ledger(j: Any) -> Any:
    _request(j, ["key", "ledger", "window"])

    def go():
        a = Ch.audited_session(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), lms_verify)
        return {"files": a.files, "host": a.host, "session": a.session}
    return _refusing(go)


def session(j: Any) -> Any:
    _request(j, ["key", "ledger", "session", "window"])

    def go():
        c, roots = Ch.trusted_session_of(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), j["session"],
                                         lms_verify)
        return {"commitment": c.to_json(), "roots": _roots(roots)}
    return _refusing(go)


class Session:
    """A challenger's session as its host leaves it: a record signed at leaf q of a small key, a release log over drawn
    roots in canonical JSON on an integer clock, and the close signed at leaf q + 1 (None: its client closed it)."""

    def __init__(self, rng: random.Random, ended: str | None = None) -> None:
        self.w, self.h = rng.choice(PARAMS)
        self.k = _key(self.w, self.h, 0)
        self.pub = lms.public_key(self.k)
        self.q = rng.randrange(2 ** self.h - 1)
        self.ch = _challenger(rng)
        self.host = rng.choice(HOSTS)
        self.ch.sign_record(self.host, self.pub, lambda m: lms.sign_at(self.k, self.q, m).encode())
        self.key = Ch.ChallengerKey(self.w, self.h, self.pub)
        c = self.ch.commitment
        self.lines = [canonical_json_text({"event": "commitment", "root": c.root.hex()})]
        clock = itertools.count(rng.choice([0, 1, 10 ** 12])).__next__
        records = {label: Ch.LiveRecord(self.ch, label, clock) for label, _ in c.families}
        picks = [label for label, n in c.families for _ in range(rng.randint(0, min(n, 4)))] or [c.families[0][0]]
        rng.shuffle(picks)
        for label in picks:
            n = len(records[label].log)
            records[label].record(rng.choice([rng.randbytes(64), bytes(64)]))
            self.lines += [canonical_json_text(e) for e in records[label].log[n:]]
        self.id = rng.choice(SESSIONS)
        self.close = self.signed(ended)

    def signed(self, ended: str | None, lines: list[str] | None = None, session: str | None = None) -> dict:
        lines = self.lines if lines is None else lines
        sid = self.id if session is None else session
        chain = Ch.log_chain(lines)
        m = Ch.close_message(self.pub[:16], self.host, self.ch.commitment, sid, len(lines), chain, ended)
        return Ch.ChallengerClose(sid, len(lines), chain, ended, lms.sign_at(self.k, self.q + 1, m).encode()).to_json()

    def files(self) -> dict:
        return {"close": self.close, "log": list(self.lines), "record": self.ch.record.to_json()}


PERIODS = ["p-0", "smoke-20261020", "2026-10-20T09:00Z", "a", "x" * 64, "0.1_b"]


def _window(rng: random.Random) -> dict:
    return {"commit": rng.randbytes(64).hex(), "period": rng.choice(PERIODS)}


class Ledger:
    """The auditor's ledger of session `s`'s window and up to two others' sessions, each opened, then ended, interleaved:
    `events` its entries' (kind, session, body), `lines` them chained and signed, each at a drawn leaf of `s`'s key (or
    `signer`'s, an `edit(i, entry)` applied to each entry before it is signed)."""

    def __init__(self, rng: random.Random, s: Session, closed: bool = True) -> None:
        self.rng, self.s, self.window = rng, s, _window(rng)
        files = {Ch.LOG_FILE: Ch.log_digest(s.lines), "record.json": rng.randbytes(64).hex()}
        if rng.random() < 0.5:
            files["serve.log"] = rng.randbytes(64).hex()
        named = [("open", s.id, self.open_body(self.window, s.host)),
                 ("end", s.id, {"closed": closed, "files": files, "rc": 0 if closed else 1})]
        others = [[("open", sid, self.open_body(_window(rng), f"vy-challenger-1.{sid}")),
                   ("end", sid, {"closed": rng.random() < 0.8, "files": {}, "rc": 0})]
                  for sid in ("o-1", "o-2")[:rng.randint(0, 2)]]
        queues = [named] + others
        self.events = []
        while any(queues):
            q = rng.choice([q for q in queues if q])
            self.events.append(q.pop(0))

    def open_body(self, window: dict, host: str, public: bytes | None = None) -> dict:
        body = {"window": window, "host": host, "pinned": self.rng.randbytes(20).hex(),
                "key_sha512": hashlib.sha512(self.s.pub if public is None else public).hexdigest(),
                "port": self.rng.choice([7811, 7812])}
        if self.rng.random() < 0.3:
            body["started_utc"] = "2026-10-20T09:45:48Z"
        return body

    def index(self, kind: str, session: str) -> int:
        return next(i for i, (k, sid, _) in enumerate(self.events) if (k, sid) == (kind, session))

    def lines(self, events=None, edit=None, signer: lms.PrivateKey | None = None) -> list[str]:
        k = self.s.k if signer is None else signer
        out: list[str] = []
        for i, (kind, sid, body) in enumerate(self.events if events is None else events):
            entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest() if out else "0" * 128,
                     "schema": Ch.LEDGER_SCHEMA, "seq": i, "session": sid}
            if edit is not None:
                edit(i, entry)
            m = Ch.ledger_message(lms.public_key(k)[:16], entry)
            out.append(canonical_json_text({**entry, "sig": lms.sign_at(k, self.rng.randrange(2 ** k.params.H), m)
                                            .encode().hex()}))
        return out


CLOSE_NAMES = ["honest", "keys", "extra", "chain-short", "chain-upper", "signature-type", "lines-zero", "lines-big",
               "lines-type", "ended-type", "session-chars", "session-type", "commitment-form", "I-short", "request"]


def close_case(rng: random.Random, name: str) -> Any:
    s = Session(rng, rng.choice([None, None, "the stream closed"]))
    d = dict(s.close)
    j = {"I": s.pub[:16].hex() if rng.random() < 0.7 else rng.randbytes(16).hex(), "close": d,
         "commitment": s.ch.commitment.to_json(), "host": s.host}
    match name:
        case "honest":
            pass
        case "keys":
            del d[rng.choice(sorted(d))]
        case "extra":
            d["host"] = s.host
        case "chain-short":
            d["chain"] = d["chain"][:rng.choice([0, 2, 126])]
        case "chain-upper":
            d["chain"] = d["chain"].upper()
        case "signature-type":
            d["signature"] = rng.choice([None, 5, d["signature"].upper(), d["signature"] + "0"])
        case "lines-zero":
            d["lines"] = rng.choice([0, -1])
        case "lines-big":
            d["lines"] = Ch.MAX_LOG_LINES + rng.choice([1, 1 << 70])
        case "lines-type":
            d["lines"] = rng.choice([True, "3", None, [3]])
        case "ended-type":
            d["ended"] = rng.choice([1, False, ["x"], {"a": "b"}])
        case "session-chars":
            d["session"] = rng.choice(["", "a b", "a/b", "é", "x" * (Ch.MAX_SESSION_ID + 1), "s\n"])
        case "session-type":
            d["session"] = rng.choice([None, 1, ["s-1"]])
        case "commitment-form":
            j["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "I-short":
            j["I"] = rng.randbytes(rng.choice([0, 15, 17, 64])).hex()
        case "request":
            k = rng.choice(["I", "host", None])
            if k is None:
                del j[rng.choice(sorted(j))]
            else:
                j[k] = rng.choice([None, 5, ["x"]] + (["AB", "0"] if k == "I" else []))
    return j


def _set(lines: list[str], i: int, **changes) -> list[str]:
    doc = json.loads(lines[i])
    doc.update(changes)
    return lines[:i] + [canonical_json_text(doc)] + lines[i + 1:]


RELEASED_NAMES = ["honest", "head", "head-form", "empty", "odd", "root", "after", "index", "label", "label-release",
                  "event", "time", "time-type", "extra", "non-canonical", "torn", "long", "float", "commitment-form",
                  "request"]


def released_case(rng: random.Random, name: str) -> Any:
    s = Session(rng)
    lines = list(s.lines)
    i = 1 + 2 * rng.randrange((len(lines) - 1) // 2)
    j = {"commitment": s.ch.commitment.to_json(), "log": lines}
    match name:
        case "honest":
            pass
        case "head":
            j["log"] = [canonical_json_text({"event": "commitment", "root": _flip(s.ch.commitment.root.hex())})] + lines[1:]
        case "head-form":
            j["log"] = [json.dumps(json.loads(lines[0]))] + lines[1:]
        case "empty":
            j["log"] = []
        case "odd":
            j["log"] = lines[:-1]
        case "root":
            j["log"] = _set(lines, i, root=_flip(json.loads(lines[i])["root"]))
        case "after":
            j["log"] = _set(lines, i + 1, after=rng.choice([_flip(json.loads(lines[i + 1])["after"]), None, 5]))
        case "index":
            j["log"] = _set(lines, rng.choice([i, i + 1]), index=rng.choice([-1, 1 << 40, True, "0", None]))
        case "label":
            j["log"] = _set(lines, i, label=rng.choice(["nope", 5, None]))
        case "label-release":
            j["log"] = _set(lines, i + 1, label=rng.choice(["nope", 5, None]))
        case "event":
            j["log"] = _set(lines, rng.choice([i, i + 1]), event=rng.choice(["commitment", "rooted", 1, None]))
        case "time":
            j["log"] = _set(lines, rng.choice([i, i + 1]), **{rng.choice(["received_at", "released_at"]): -1})
        case "time-type":
            k = "received_at" if rng.random() < 0.5 else "released_at"
            j["log"] = _set(lines, i if k == "received_at" else i + 1, **{k: rng.choice(["1", None, True, [1]])})
        case "extra":
            j["log"] = _set(lines, rng.choice([i, i + 1]), retry=True)
        case "non-canonical":
            j["log"] = lines[:i] + [json.dumps(json.loads(lines[i]))] + lines[i + 1:]
        case "torn":
            j["log"] = lines[:i] + [lines[i][:rng.randrange(1, len(lines[i]))]] + lines[i + 1:]
        case "long":
            j["log"] = _set(lines, i, received_at=10 ** rng.choice([4096, 4100]))
        case "float":
            n = json.loads(lines[i])["index"]
            written = rng.choice([f"{n}.0", f"{n}e0", f"{n}E0", f"{n * 10}e-1"])
            j["log"] = lines[:i] + [lines[i].replace(f'"index":{n},', f'"index":{written},', 1)] + lines[i + 1:]
        case "commitment-form":
            j["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "request":
            if rng.random() < 0.5:
                del j[rng.choice(sorted(j))]
            else:
                j["log"] = rng.choice([None, "x", lines + [5]])
    return j


LEDGER_NAMES = ["honest", "no-ledger", "ledger-type", "line-type", "non-canonical", "torn", "long", "keys", "kind",
                "schema", "seq", "prev", "dropped", "swapped", "signature", "signature-form", "other-signer",
                "self-keyed", "other-key", "open-form", "end-form", "second-open", "reopen", "end-unopened",
                "end-twice", "no-open", "truncated", "unclosed", "window-form", "key-form", "params", "request"]


def _entry_edit(target: int, f):
    def edit(i, entry):
        if i == target:
            f(entry)
    return edit


def ledger_case(rng: random.Random, name: str) -> Any:
    s = Session(rng)
    lg = Ledger(rng, s, closed=name != "unclosed")
    j = {"key": s.key.to_json(), "ledger": None, "window": lg.window}
    i = rng.randrange(len(lg.events))
    other = _key(s.w, s.h, 1)
    match name:
        case "no-ledger":
            pass
        case "ledger-type":
            j["ledger"] = rng.choice(["x", 5, {"lines": []}])
        case "line-type":
            lines = lg.lines()
            j["ledger"] = lines[:i] + [rng.choice([None, 5, ["x"], {"a": 1}])] + lines[i + 1:]
        case "non-canonical":
            lines = lg.lines()
            j["ledger"] = lines[:i] + [json.dumps(json.loads(lines[i]))] + lines[i + 1:]
        case "torn":
            lines = lg.lines()
            j["ledger"] = lines[:i] + [lines[i][:rng.randrange(1, len(lines[i]))]] + lines[i + 1:]
        case "long":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e["body"].update(pad="x" * Ch.MAX_LEDGER_LINE)))
        case "keys":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(rng.choice([{"extra": 1}, {"seq": None}]))))
        case "kind":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(kind=rng.choice(["close", "", 1, None]))))
        case "schema":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(
                schema=rng.choice(["verity/challenger-auditor/v2", ""]))))
        case "seq":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(seq=e["seq"] + rng.choice([1, 2, 1 << 40]))))
        case "prev":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(prev=_flip(e["prev"], rng.randrange(64)))))
        case "dropped":
            lines = lg.lines()
            k = rng.randrange(len(lines) - 1)
            j["ledger"] = lines[:k] + lines[k + 1:]
        case "swapped":
            lines = lg.lines()
            j["ledger"] = [lines[1], lines[0]] + lines[2:]
        case "signature":
            lines = lg.lines()
            e = json.loads(lines[i])
            j["ledger"] = lines[:i] + [canonical_json_text({**e, "sig": _flip(e["sig"], rng.randrange(len(e["sig"]) // 2))})] \
                + lines[i + 1:]
        case "signature-form":
            lines = lg.lines()
            e = json.loads(lines[i])
            sig = rng.choice([e["sig"].upper(), e["sig"] + "0", e["sig"][:8], None, 5])
            j["ledger"] = lines[:i] + [canonical_json_text({**e, "sig": sig})] + lines[i + 1:]
        case "other-signer":
            j["ledger"] = lg.lines(signer=other)
        case "self-keyed":
            events = [(k, sid, {**b, "key_sha512": hashlib.sha512(lms.public_key(other)).hexdigest()} if k == "open" else b)
                      for k, sid, b in lg.events]
            j["ledger"] = lg.lines(events, signer=other)
        case "other-key":
            j["ledger"] = lg.lines()
            j["key"] = {**j["key"], "public": lms.public_key(other).hex()}
        case "open-form":
            o = lg.index("open", s.id)
            bad = rng.choice([{"window": {**lg.window, "period": rng.choice(["", ".p", "a b", 1, "é"])}},
                              {"window": {**lg.window, "commit": lg.window["commit"][:-2]}},
                              {"window": {**lg.window, "stream": "00"}}, {"host": rng.choice(["", "a b", None])},
                              {"pinned": rng.choice(["ab" * 19, "AB" * 20, None])},
                              {"key_sha512": rng.choice([None, "00"])}])
            j["ledger"] = lg.lines(edit=_entry_edit(o, lambda e: e.update(body={**e["body"], **bad})))
        case "end-form":
            o = lg.index("end", s.id)
            bad = rng.choice([{"closed": rng.choice([None, 1, "true"])}, {"files": rng.choice([None, [], "x"])},
                              {"files": {Ch.LOG_FILE: rng.choice(["00", None, "AB" * 64])}}])
            j["ledger"] = lg.lines(edit=_entry_edit(o, lambda e: e.update(body={**e["body"], **bad})))
        case "second-open":
            events = list(lg.events)
            events.insert(rng.randint(0, len(events)), ("open", "s-other", lg.open_body(lg.window, "vy-challenger-1.x")))
            j["ledger"] = lg.lines(events)
        case "reopen":
            events = list(lg.events)
            events.append(("open", s.id, lg.open_body(_window(rng), s.host)))
            j["ledger"] = lg.lines(events)
        case "end-unopened":
            events = list(lg.events)
            events.insert(rng.randint(0, len(events)), ("end", "never", {"closed": True, "files": {}, "rc": 0}))
            j["ledger"] = lg.lines(events)
        case "end-twice":
            events = list(lg.events)
            events.append(lg.events[lg.index("end", s.id)])
            j["ledger"] = lg.lines(events)
        case "no-open":
            j["ledger"] = lg.lines()
            j["window"] = rng.choice([{**lg.window, "period": "p-other"}, {**lg.window, "commit": rng.randbytes(64).hex()}])
        case "truncated":
            j["ledger"] = lg.lines()[:lg.index(rng.choice(["open", "end"]), s.id)]
        case "window-form":
            j["ledger"] = lg.lines()
            j["window"] = rng.choice([{**lg.window, "period": rng.choice([0, "", "-p", "x" * 65])},
                                      {**lg.window, "stream": "00"}, {"commit": lg.window["commit"]}, None])
        case "key-form":
            j["ledger"] = lg.lines()
            j["key"] = key_case(rng, rng.choice(KEY_NAMES[1:]))
        case "params":
            j["ledger"] = lg.lines()
            w, h = rng.choice([p for p in PARAMS if p != (s.w, s.h)])
            j["key"] = {**j["key"], "W": w, "H": h}
        case "request":
            j["ledger"] = lg.lines()
            del j[rng.choice(sorted(j))]
        case _:
            j["ledger"] = lg.lines()
    return j


SESSION_NAMES = ["honest", "no-ledger", "no-open", "unclosed-ledger", "other-host", "unsealed-log", "unlisted-log",
                 "wrong-session", "session-form", "no-record", "record-signature", "no-close", "close-form",
                 "close-signature", "close-renamed", "rechained", "log-tampered", "log-dropped", "log-form", "unclosed",
                 "key-form", "request"]


def session_case(rng: random.Random, name: str) -> Any:
    s = Session(rng)
    lg = Ledger(rng, s, closed=name != "unclosed-ledger")
    files = s.files()
    j = {"key": s.key.to_json(), "ledger": None, "session": files, "window": lg.window}
    lines = files["log"]
    end = lg.index("end", s.id)
    match name:
        case "no-ledger":
            pass
        case "no-open":
            j["ledger"] = lg.lines()
            j["window"] = {**lg.window, "period": "p-other"}
        case "other-host":
            o = lg.index("open", s.id)
            host = rng.choice([h for h in HOSTS if h != s.host] + [f"vy-challenger-1.{s.id}"])
            j["ledger"] = lg.lines(edit=_entry_edit(o, lambda e: e["body"].update(host=host)))
        case "unsealed-log":
            j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"]["files"].update(
                {Ch.LOG_FILE: _flip(e["body"]["files"][Ch.LOG_FILE], 5)})))
        case "unlisted-log":
            j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"]["files"].pop(Ch.LOG_FILE)))
        case "wrong-session":
            sid = rng.choice([x for x in SESSIONS + ["o-3"] if x != s.id])
            j["ledger"] = lg.lines([(k, sid if x == s.id else x, b) for k, x, b in lg.events])
        case "session-form":
            j["session"] = rng.choice([{k: v for k, v in files.items() if k != "log"}, {**files, "extra": 1}, None])
        case "no-record":
            files["record"] = None
        case "record-signature":
            files["record"] = {**files["record"], "signature": _flip(files["record"]["signature"], 9)}
        case "no-close":
            files["close"] = None
        case "close-form":
            files["close"] = rng.choice([{**files["close"], "lines": 0}, {**files["close"], "chain": "00"}, "x"])
        case "close-signature":
            files["close"] = {**files["close"], "signature": _flip(files["close"]["signature"], 9)}
        case "close-renamed":
            files["close"] = {**files["close"], "session": "renamed"}
            j["ledger"] = lg.lines([(k, "renamed" if x == s.id else x, b) for k, x, b in lg.events])
        case "rechained":
            files["log"] = _set(lines, 1, root=_flip(json.loads(lines[1])["root"]))
            files["close"] = {**files["close"], "chain": Ch.log_chain(files["log"]).hex()}
        case "log-tampered":
            files["log"] = _set(lines, rng.randrange(1, len(lines)), **{"x": 1})
        case "log-dropped":
            files["log"] = lines[:-1]
        case "log-form":
            files["log"] = rng.choice([None, "x", lines[:-1] + [5]])
        case "unclosed":
            files["close"] = s.signed(rng.choice(["the stream closed", "a client's frames are hello", ""]))
        case "key-form":
            j["key"] = key_case(rng, rng.choice(KEY_NAMES[1:]))
        case "request":
            del j[rng.choice(sorted(j))]
    if j.get("ledger", 0) is None and name != "no-ledger":
        j["ledger"] = lg.lines()
    return j


COMMITMENT = D.Port(PIECES, "challenger_commitment", commitment, _cases(commitment_case, COMMITMENT_NAMES),
                    package=PACKAGE, need_rejects=False, of_record="lean")
OPENING = D.Port(PIECES, "challenger_opening", opening, _cases(opening_case, OPENING_NAMES), package=PACKAGE,
                 need_rejects=False, of_record="lean")
REASON = D.Port(PIECES, "challenger_reason", reason, _cases(reason_case, REASON_NAMES), package=PACKAGE, of_record="lean")
ORDER = D.Port(PIECES, "challenger_order", order, _cases(order_case, ORDER_NAMES), package=PACKAGE, of_record="lean")
KEY = D.Port(PIECES, "challenger_key", key, _cases(key_case, KEY_NAMES), package=PACKAGE, need_rejects=False,
             of_record="lean")
RECORD = D.Port(PIECES, "challenger_record", record, _cases(record_case, RECORD_NAMES), package=PACKAGE, of_record="lean")
TRUSTED = D.Port(PIECES, "challenger_trusted", trusted, _cases(trusted_case, TRUSTED_NAMES), package=PACKAGE,
                 of_record="lean")
CLOSE = D.Port(PIECES, "challenger_close", close, _cases(close_case, CLOSE_NAMES), package=PACKAGE, need_rejects=False,
               of_record="lean")
RELEASED = D.Port(PIECES, "challenger_released", released, _cases(released_case, RELEASED_NAMES), package=PACKAGE,
                  of_record="lean")
LEDGER = D.Port(PIECES, "challenger_ledger", ledger, _cases(ledger_case, LEDGER_NAMES), package=PACKAGE,
                need_rejects=False, of_record="lean")
SESSION = D.Port(PIECES, "challenger_session", session, _cases(session_case, SESSION_NAMES), package=PACKAGE,
                 need_rejects=False, of_record="lean")

pieces = D.server_fixture(COMMITMENT)

RUNS = [(COMMITMENT, commitment_case, COMMITMENT_NAMES, 300), (OPENING, opening_case, OPENING_NAMES, 300),
        (REASON, reason_case, REASON_NAMES, 400), (ORDER, order_case, ORDER_NAMES, 400), (KEY, key_case, KEY_NAMES, 200),
        (RECORD, record_case, RECORD_NAMES, 200), (TRUSTED, trusted_case, TRUSTED_NAMES, 300),
        (CLOSE, close_case, CLOSE_NAMES, 200), (RELEASED, released_case, RELEASED_NAMES, 300),
        (LEDGER, ledger_case, LEDGER_NAMES, 300), (SESSION, session_case, SESSION_NAMES, 300)]


@pytest.mark.parametrize("port, make, names, examples", RUNS, ids=[r[0].fn for r in RUNS])
def test_a_challenger_piece_agrees(pieces, port, make, names, examples):
    """Each piece agrees on drawn cases and on every mutation over a few seeds, the honest ones accepted (no reason, no
    refusal) and each mutation refused."""
    stats = D.run(port, examples=examples, server=pieces)
    outcomes = Counter()
    for name in names:
        for seed in range(4):
            kind, text = D.replay(port, make(random.Random(f"{name}/{seed}"), name), pieces)
            clean = kind == "output" and '"refused"' not in text and '"reason":"' not in text
            outcomes[name, clean] += 1
            assert clean == (name == "honest"), (port.fn, name, seed, kind, text)
    print(f"{port.fn}: {stats}; swept {sum(outcomes.values())}")
