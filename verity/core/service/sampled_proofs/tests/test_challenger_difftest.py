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
malformed form. The session's (`CLOSE`, `RELEASED`, `LEDGER`, `HEAD`, `SESSION`) are a session a small key signed and
the auditor's v2 ledger opening and ending it among others, its record's leaf between the two, sometimes after an older
v1 session and inside a period of its lane, then a close, log or ledger entry edited, dropped, re-chained or signed under
another key, a ledger signed under its own key, one cut before the session's end, a second session for the window, a v1
window, a v1 entry after a v2, a record leaf outside its session, an entry's or used twice, a period named twice, past
its cap, committing no cap or circuit or closing with other windows, a record on another host or at another leaf, a log
or refused hellos the ledger didn't seal, infra's published heads forked, each malformed form. The pins log's
(`PINNED`) are a session whose open seals the host's pins log and pins its commit, then a commit off main or pinned
after the open, no pins log, a malformed one or line, one cut or with a sealed line edited, and an open sealing
none. A draw's coins (`READ`, `DRAW_COINS`: `read_reason` and `coins_reason`, `SampledProofs.ChallengerCoins`) are
a draw's coins from one family of a ledger session's challenger, opened after the prover's commitment between other
roots, the bytes it read and the session's evidence, then a coin opened after the commitment and skipped, a wrong
opening, coins opened before the commitment (their `after` rewritten or not), a coin opened and not read, the bytes
tampered or longer, another commitment, a release log edited, the window of another session the ledger opens (a prover
picking the window), a session whose pin is off the verifier's copy of main, no pins log, each malformed form. Each
piece answers
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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.sampled_proofs import challenger as Ch, coins as T
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
#: those for a key that signs a ledger too: 32 leaves, room for a ledger's entries at increasing leaves on either side of
#: its session's record
LEDGER_PARAMS = [(1, 6), (2, 6), (4, 6)]
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
        return Ch.audited_session(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), lms_verify)._asdict()
    return _refusing(go)


def head(j: Any) -> Any:
    _request(j, ["heads", "key"])

    def go():
        return {"ledger": Ch.ledger_head(j["heads"], Ch.ChallengerKey.from_json(j["key"]), lms_verify)}
    return _refusing(go)


def session(j: Any) -> Any:
    _request(j, ["key", "ledger", "main", "pins", "session", "window"])

    def go():
        c, roots = Ch.trusted_session_of(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), j["pins"],
                                         j["main"], j["session"], lms_verify)
        return {"commitment": c.to_json(), "roots": _roots(roots)}
    return _refusing(go)


def credited(j: Any) -> Any:
    _request(j, ["descriptor", "key", "ledger", "main", "pins", "window"])
    d = _hex(j["descriptor"])

    def go():
        return Ch.credited_session(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), j["pins"],
                                   j["main"], d, lms_verify)._asdict()
    return _refusing(go)


def pinned(j: Any) -> Any:
    _request(j, ["key", "ledger", "main", "pins", "window"])

    def go():
        return Ch.pinned_session(j["ledger"], j["window"], Ch.ChallengerKey.from_json(j["key"]), j["pins"],
                                 j["main"], lms_verify)._asdict()
    return _refusing(go)


HELLOS = ['{"error":"a client\'s frames are hello","peer":"10.0.0.7"}', "x", "", "\u00e9\x7f"]


class Session:
    """A challenger's session as its host leaves it: a record signed at leaf q of a small key (drawn, or `q`), a release
    log over drawn roots in canonical JSON on an integer clock, the close signed at leaf q + 1 (None: its client closed
    it), under a key of one of `params`, and the lines of the hellos it refused."""

    def __init__(self, rng: random.Random, ended: str | None = None, params=PARAMS, q: int | None = None) -> None:
        self.w, self.h = rng.choice(params)
        self.k = _key(self.w, self.h, 0)
        self.pub = lms.public_key(self.k)
        self.q = rng.randrange(2 ** self.h - 1) if q is None else q
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
        self.hellos = [rng.choice(HELLOS) for _ in range(rng.choice([0, 0, 1, 3]))]

    def signed(self, ended: str | None, lines: list[str] | None = None, session: str | None = None) -> dict:
        lines = self.lines if lines is None else lines
        sid = self.id if session is None else session
        chain = Ch.log_chain(lines)
        m = Ch.close_message(self.pub[:16], self.host, self.ch.commitment, sid, len(lines), chain, ended)
        return Ch.ChallengerClose(sid, len(lines), chain, ended, lms.sign_at(self.k, self.q + 1, m).encode()).to_json()

    def files(self) -> dict:
        return {"close": self.close, "hellos": list(self.hellos), "log": list(self.lines),
                "record": self.ch.record.to_json()}


PERIODS = ["p-0", "smoke-20261020", "2026-10-20T09:00Z", "a", "x" * 64, "0.1_b"]
LANES = ["pouw", "vllm", "memory.audit"]
CIRCUITS = ["ab" * 64, "0" * 128, "c" * 128]
#: a reserve's start: across a day, a leap day, a century's non-leap February, a year, and at either end of `datetime`'s
STARTS = ["2026-10-20T12:00:00Z", "2024-02-28T23:59:59Z", "2100-02-28T12:00:00Z", "2026-12-31T23:30:00Z",
          "0001-01-01T00:00:00Z", "9999-12-31T11:00:00Z"]
PEERS = ["10.0.0.7", "vy-nebius-1", "::1"]
#: a body's note, each written as `ledger_text` escapes it: ASCII without DEL, as the entry's signed digest (canonical
#: JSON, which refuses any other) holds
NOTES = ["~", "a\"b\\c\n", "\x01", "\x1f "]
#: a key `_body` removes
DROP = object()
#: an open's `reserve` that `Ledger.lines` resolves to its session's reserve's seq
RESOLVE = object()


def _window(rng: random.Random) -> dict:
    return {"root": rng.randbytes(64).hex()}


def _body(b: dict, **changes) -> dict:
    return {k: v for k, v in {**b, **changes}.items() if v is not DROP}


def _sign_ledger(k: lms.PrivateKey, q: int, entry: dict) -> str:
    """`entry` signed at leaf q of `k`, under its schema's tag or, for a schema the ledger has none for, v2's."""
    known = type(entry["schema"]) is str and entry["schema"] in Ch.LEDGER_TAGS
    m = Ch.ledger_message(lms.public_key(k)[:16], entry if known else {**entry, "schema": Ch.LEDGER_SCHEMA})
    return lms.sign_at(k, q, m).encode().hex()


class Ledger:
    """The auditor's ledger of session `s`'s window and up to two others' sessions, each opened, then ended, interleaved,
    sometimes after an older v1 session's, and sometimes (always when `period` is set: "open" or "closed") inside a
    period of `s`'s lane, its close when it has one listing its windows: `events` its entries' (kind, session, body),
    `lines` them chained and signed at drawn increasing leaves of `s`'s key, `s`'s record's leaf between its open's and
    its end's (or at `leaves`, or under `signer`, an `edit(i, entry)` applied to each entry before it is signed).  When
    `two_step` (drawn when `s`'s id is a reserved one, unless set), every v2 session is reserved before it is opened, its
    open binding the reserve, and sometimes another reserve is never opened."""

    def __init__(self, rng: random.Random, s: Session, closed: bool = True, period: str | None = None,
                 two_step: bool | None = None) -> None:
        self.rng, self.s, self.window = rng, s, _window(rng)
        self.two_step = (Ch._RESERVED_SESSION.fullmatch(s.id) is not None
                         and (rng.random() < 0.8 if two_step is None else two_step))
        self._ids = itertools.count()
        self.v1: list[dict] = []
        files = {Ch.LOG_FILE: Ch.log_digest(s.lines), Ch.HELLOS_FILE: Ch.log_digest(s.hellos),
                 Ch.RECORD_FILE: rng.randbytes(64).hex(), Ch.CLOSE_FILE: rng.randbytes(64).hex()}
        if rng.random() < 0.5:
            files["serve.log"] = rng.randbytes(64).hex()
        self.lane = rng.choice(LANES)
        named, self.opened = self.session_events(s.id, self.window, s.host, self.lane)
        self.ended = {"closed": closed, "commitment": s.ch.commitment.root.hex(), "complete": closed, "files": files,
                      "rc": 0 if closed else 1, "record_leaf": s.q, "refused_hellos": len(s.hellos)}
        named.append(("end", s.id, self.ended))
        queues = [named]
        for tag in ("o-1", "o-2")[:rng.randint(0, 2)]:
            sid, done = self.sid(tag), rng.random() < 0.8
            events, _ = self.session_events(sid, _window(rng), f"vy-challenger-1.{sid}", rng.choice(LANES))
            queues.append(events + [("end", sid, {"closed": done, "complete": done and rng.random() < 0.8,
                                                  "files": {}, "rc": 0})])
        if self.two_step and rng.random() < 0.5:
            queues.append([("reserve", self.sid("dead"), self.reserve_body(rng.choice(LANES)))])
        self.events: list[tuple[str, str | None, dict]] = []
        while any(queues):
            q = rng.choice([q for q in queues if q])
            self.events.append(q.pop(0))
        self.period_open = None
        if period is not None or rng.random() < 0.5:
            self.period_open = {"by": "vy-admin", "cap": rng.randint(4, 8), "circuit": rng.choice(CIRCUITS),
                                "key_sha512": hashlib.sha512(s.pub).hexdigest(), "lane": self.lane,
                                "period": rng.choice(PERIODS)}
            first = self.index("reserve" if self.two_step else "open", s.id)
            self.events.insert(rng.randint(0, first), ("period-open", None, self.period_open))
            if period == "closed" or period is None and rng.random() < 0.6:
                opens = [k for k, (kind, _, _) in enumerate(self.events) if kind == "open"]
                at = rng.randint(max([self.index("end", s.id)] + opens) + 1, len(self.events))
                self.events.insert(at, ("period-close", None, self.close_body(self.events[:at])))
        if rng.random() < 0.3:
            old = {"commit": rng.randbytes(64).hex(), "period": rng.choice(PERIODS)}
            self.v1 = [{"host": "vy-challenger-1", "key_sha512": hashlib.sha512(s.pub).hexdigest(),
                        "pinned": rng.randbytes(20).hex(), "port": 7811, "window": old},
                       {"closed": True, "files": {}, "rc": 0}]
            self.events = [("open", "v1-0", self.v1[0]), ("end", "v1-0", self.v1[1])] + self.events

    def open_body(self, window: dict, host: str, lane: str, public: bytes | None = None) -> dict:
        body = {"window": window, "host": host, "pinned": self.rng.randbytes(20).hex(),
                "key_sha512": hashlib.sha512(self.s.pub if public is None else public).hexdigest(),
                "port": self.rng.choice([7811, 7812])}
        body.update(self.rng.choice([{"by": lane}, {"by": "vy-admin", "for": lane}, {"by": lane, "for": ""}]))
        if self.rng.random() < 0.3:
            body["started_utc"] = "2026-10-20T09:45:48Z"
        if self.rng.random() < 0.2:
            body["note"] = self.rng.choice(NOTES)
        return body

    def sid(self, tag: str) -> str:
        """A session id: `tag`, or a fresh reserved one when the ledger is two-step."""
        return f"s20261020T{next(self._ids):06d}Z-{self.rng.randbytes(2).hex()}" if self.two_step else tag

    def reserve_body(self, lane: str) -> dict:
        rng = self.rng
        start, minutes = rng.choice(STARTS), rng.choice([1, 30, 720, rng.randint(1, 720)])
        body = rng.choice([{"by": lane}, {"by": "vy-admin", "for": lane}])
        body.update(peer=rng.choice(PEERS), scheme=rng.choice(SCHEMES), epoch=rng.choice([0, 1, 7]),
                    families=[[rng.choice(LABELS), rng.randint(1, 4)]], minutes=minutes, started_utc=start,
                    expires_utc=Ch.later(start, minutes))
        if rng.random() < 0.2:
            body["note"] = rng.choice(NOTES)
        return body

    def session_events(self, sid: str, window: dict, host: str, lane: str) -> tuple[list, dict]:
        """Session `sid`'s entries up to its open, and its open's body: the open alone, or when the ledger is two-step its
        reserve and the open binding it, by the reserve's prover and inside its minutes."""
        if not self.two_step:
            body = self.open_body(window, host, lane)
            return [("open", sid, body)], body
        rng, reserve = self.rng, self.reserve_body(lane)
        body = _body(self.open_body(window, host, lane), by=DROP, **{"for": DROP})
        body.update({k: reserve[k] for k in Ch.RESERVED if k in reserve}, reserve=RESOLVE,
                    opened_by=rng.choice([reserve["by"], reserve.get("for", reserve["by"])]),
                    started_utc=Ch.later(reserve["started_utc"], rng.randint(0, reserve["minutes"])))
        return [("reserve", sid, reserve), ("open", sid, body)], body

    def close_body(self, before: list) -> dict:
        """The period-close after entries `before`: the windows of the period's lane since its open, each counted at its
        reserve or one-step open, its root null when it isn't opened among `before`, complete when its end is among
        `before` and complete."""
        p = self.period_open
        start = next(i for i, (_, _, b) in enumerate(before) if b is p)
        outcomes = []
        for kind, sid, b in before[start:]:
            counts = kind == "reserve" or kind == "open" and "reserve" not in b and not any(b is v for v in self.v1)
            if counts and (b.get("for") or b["by"]) == p["lane"]:
                opened = next((o for k, x, o in before if (k, x) == ("open", sid)), None)
                end = next((e for k, x, e in before if (k, x) == ("end", sid)), None)
                outcomes.append({"complete": opened is not None and end is not None and end.get("complete") is True,
                                 "root": None if opened is None else opened["window"]["root"], "session": sid})
        return {"complete": bool(outcomes) and all(w["complete"] for w in outcomes), "lane": p["lane"],
                "period": p["period"], "windows": outcomes}

    def index(self, kind: str, session: str) -> int:
        return next(i for i, (k, sid, _) in enumerate(self.events) if (k, sid) == (kind, session))

    def leaves(self, events=None) -> list[int]:
        """Increasing leaves of `s`'s key for `events`, `s`'s record's leaf q between its open's and its end's and none
        of theirs."""
        events = self.events if events is None else events
        n, q = len(events), self.s.q
        o = next((i for i, (_, _, b) in enumerate(events) if b is self.opened), 0)
        e = next((i for i, (_, _, b) in enumerate(events) if b is self.ended), n)
        lo, hi = sorted((min(o + 1, n), min(e, n)))
        r = self.rng.randint(lo, hi)
        return (sorted(self.rng.sample(range(q), r)) + sorted(self.rng.sample(range(q + 1, 2 ** self.s.h), n - r)))

    def lines(self, events=None, edit=None, signer: lms.PrivateKey | None = None, leaves: list[int] | None = None
              ) -> list[str]:
        k = self.s.k if signer is None else signer
        events = self.events if events is None else events
        leaves = self.leaves(events) if leaves is None else leaves
        out: list[str] = []
        for i, (kind, sid, body) in enumerate(events):
            schema = Ch.LEDGER_SCHEMA_V1 if any(body is b for b in self.v1) else Ch.LEDGER_SCHEMA
            if body.get("reserve") is RESOLVE:
                body = {**body, "reserve": next((k for k, (x, y, _) in enumerate(events) if (x, y) == ("reserve", sid)), 0)}
            entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest() if out else "0" * 128,
                     "schema": schema, "seq": i, "session": sid}
            if edit is not None:
                edit(i, entry)
            out.append(Ch.ledger_text({**entry, "sig": _sign_ledger(k, leaves[i], entry)}))
        return out


def _ledger_session(rng: random.Random, reserved: bool | None = None) -> Session:
    """A session whose record's leaf leaves room on either side for a ledger's entries, its id a reserved one (drawn,
    unless `reserved` says)."""
    s = Session(rng, params=LEDGER_PARAMS, q=rng.randrange(20, 40))
    if reserved or reserved is None and rng.random() < 0.5:
        s.id = f"s20261019T{rng.randrange(10 ** 6):06d}Z-{rng.randbytes(2).hex()}"
        s.close = s.signed(None)
    return s


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


LEDGER_NAMES = ["honest", "no-ledger", "ledger-type", "line-type", "non-canonical", "unicode", "torn", "long", "keys",
                "kind", "schema", "seq", "prev", "dropped", "swapped", "signature", "signature-form", "other-signer",
                "self-keyed", "other-key", "open-form", "end-form", "end-fields", "second-open", "reopen",
                "end-unopened", "end-twice", "no-open", "truncated", "unclosed", "window-form", "key-form", "params",
                "request", "reused-leaf", "v1-ledger", "v1-after-v2", "record-outside", "record-entry", "record-twice",
                "period-form", "period-v1", "period-twice", "period-close", "past-cap", "cap", "circuit", "no-cap",
                "reserve-form", "reserve-again", "open-unbound", "open-request", "opened-by", "open-late",
                "open-after-close", "one-step-after-reserve"]
#: the cases whose ledger is two-step
RESERVE_NAMES = {"reserve-form", "reserve-again", "open-unbound", "open-request", "opened-by", "open-late",
                 "open-after-close", "one-step-after-reserve"}


def _entry_edit(target: int, f):
    def edit(i, entry):
        if i == target:
            f(entry)
    return edit


def _body_edit(target: int, **changes):
    return _entry_edit(target, lambda e: e.update(body=_body(e["body"], **changes)))


def ledger_case(rng: random.Random, name: str) -> Any:
    two = name in RESERVE_NAMES
    s = _ledger_session(rng, reserved=True if two else None)
    period = {"period-form": "closed", "period-twice": "open", "period-close": "closed", "past-cap": "open",
              "cap": "open", "circuit": "open", "no-cap": "open", "open-after-close": "open"}.get(name)
    lg = Ledger(rng, s, closed=name != "unclosed", period=period, two_step=True if two else None)
    j = {"key": s.key.to_json(), "ledger": None, "window": lg.window}
    i = rng.randrange(len(lg.events))
    o, end = lg.index("open", s.id), lg.index("end", s.id)
    r = lg.index("reserve", s.id) if lg.two_step else None
    reserve = lg.events[r][2] if r is not None else None
    p = None if lg.period_open is None else next(k for k, (_, _, b) in enumerate(lg.events) if b is lg.period_open)
    other = _key(s.w, s.h, 1)
    events = list(lg.events)
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
        case "unicode":
            note = rng.choice(["\U0001f600", "\\ud800", "\\udfff", "\u00e9", "\u00e9"])
            if note.isascii():
                j["ledger"] = lg.lines(edit=_body_edit(o, note=note))
            else:
                # no entry's signed digest (canonical JSON) holds a string past ASCII, so the note goes into the signed
                # line's text: the emoji as `ledger_text` escapes it (a surrogate pair), é as itself (not canonical)
                lines = lg.lines(edit=_body_edit(o, note="?"))
                text = json.dumps(note, ensure_ascii=note != "\u00e9")
                j["ledger"] = lines[:o] + [lines[o].replace('"note":"?"', '"note":' + text, 1)] + lines[o + 1:]
        case "torn":
            lines = lg.lines()
            j["ledger"] = lines[:i] + [lines[i][:rng.randrange(1, len(lines[i]))]] + lines[i + 1:]
        case "long":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e["body"].update(pad="x" * Ch.MAX_LEDGER_LINE)))
        case "keys":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(rng.choice([{"extra": 1}, {"seq": None}]))))
        case "kind":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(kind=rng.choice(["close", "", 1, None,
                                                                                            "period"]))))
        case "schema":
            j["ledger"] = lg.lines(edit=_entry_edit(i, lambda e: e.update(
                schema=rng.choice(["verity/challenger-auditor/v3", "", ["x"], {"v": 2}, 2]))))
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
            j["ledger"] = lines[:i] + [Ch.ledger_text({**e, "sig": _flip(e["sig"], rng.randrange(len(e["sig"]) // 2))})] \
                + lines[i + 1:]
        case "signature-form":
            lines = lg.lines()
            e = json.loads(lines[i])
            sig = rng.choice([e["sig"].upper(), e["sig"] + "0", e["sig"][:8], e["sig"][:6], None, 5])
            j["ledger"] = lines[:i] + [Ch.ledger_text({**e, "sig": sig})] + lines[i + 1:]
        case "other-signer":
            j["ledger"] = lg.lines(signer=other)
        case "self-keyed":
            mine = hashlib.sha512(lms.public_key(other)).hexdigest()
            j["ledger"] = lg.lines(signer=other, edit=lambda _, e: e["kind"] in ("open", "period-open") and e.update(
                body={**e["body"], "key_sha512": mine}))
        case "other-key":
            j["ledger"] = lg.lines()
            j["key"] = {**j["key"], "public": lms.public_key(other).hex()}
        case "open-form":
            w = lg.window["root"]
            bad = rng.choice([{"window": {"root": w, "period": "p-0"}}, {"window": {"root": w[:-2]}},
                              {"window": {"root": w + rng.choice(["\n", " "])}}, {"window": {"root": w.upper()}},
                              {"window": {"commit": w, "period": "p-0"}}, {"host": rng.choice(["", "a b", None])},
                              {"pinned": rng.choice(["ab" * 19, "AB" * 20, None])},
                              {"key_sha512": rng.choice([None, "00"])}, {"by": rng.choice([None, 5, DROP])},
                              {"for": rng.choice([None, 5, ["pouw"]])}])
            j["ledger"] = lg.lines(edit=_body_edit(o, **bad))
        case "end-form":
            bad = rng.choice([{"closed": rng.choice([None, 1, "true"])}, {"files": rng.choice([None, [], "x"])},
                              {"files": {Ch.LOG_FILE: rng.choice(["00", None, "AB" * 64])}},
                              {"record_leaf": rng.choice([-1, "3", True, [s.q]])}])
            j["ledger"] = lg.lines(edit=_body_edit(end, **bad))
        case "end-fields":
            files = lg.ended["files"]
            bad = rng.choice([{"record_leaf": DROP}, {"record_leaf": None}, {"commitment": rng.choice([DROP, "00", 5])},
                              {"refused_hellos": rng.choice([DROP, -1, True, "0"])},
                              {"files": _body(files, **{rng.choice(Ch.SEALED_FILES): DROP})}])
            j["ledger"] = lg.lines(edit=_body_edit(end, **bad))
        case "second-open":
            events.insert(rng.randint(0, len(events)), ("open", "s-other", lg.open_body(lg.window, "vy-challenger-1.x",
                                                                                       "other")))
            j["ledger"] = lg.lines(events)
        case "reused-leaf":
            leaves = lg.leaves()
            k = rng.randrange(1, len(leaves))
            leaves[k] = rng.randint(0, leaves[k - 1])
            j["ledger"] = lg.lines(leaves=leaves)
        case "reopen":
            events.append(("open", s.id, lg.open_body(_window(rng), s.host, "other")))
            j["ledger"] = lg.lines(events)
        case "end-unopened":
            events.insert(rng.randint(0, len(events)), ("end", "never", {"closed": True, "files": {}, "rc": 0}))
            j["ledger"] = lg.lines(events)
        case "end-twice":
            events.append(lg.events[end])
            j["ledger"] = lg.lines(events)
        case "no-open":
            j["ledger"] = lg.lines()
            j["window"] = _window(rng)
        case "truncated":
            j["ledger"] = lg.lines()[:rng.choice([o, end])]
        case "window-form":
            j["ledger"] = lg.lines()
            w = lg.window["root"]
            j["window"] = rng.choice([{"root": w, "period": "p-0"}, {"root": w + "\n"}, {"root": w.upper()},
                                      {"commit": w, "period": "p-0"}, {"root": w[:-2]}, {"root": None}, None, [w]])
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
        case "v1-ledger":
            events = [(k, sid, _body(b, window={"commit": b["window"]["root"], "period": rng.choice(PERIODS)})
                       if k == "open" and "root" in b["window"] else b) for k, sid, b in lg.events if not k.startswith("p")]
            lg.v1 = [b for _, _, b in events]
            j["ledger"] = lg.lines(events)
        case "v1-after-v2":
            lg.v1 = lg.v1 + [lg.events[-1][2]]
            j["ledger"] = lg.lines()
        case "record-outside":
            leaves = lg.leaves()
            j["ledger"] = lg.lines(leaves=leaves, edit=_body_edit(end, record_leaf=rng.choice([leaves[o], leaves[end]])))
        case "record-entry":
            sid = lg.sid("o-9")
            events.insert(o + 1, ("reserve", sid, lg.reserve_body("other")) if lg.two_step
                          else ("open", sid, lg.open_body(_window(rng), "h", "other")))
            leaves = lg.leaves(events)
            j["ledger"] = lg.lines(events, leaves=leaves, edit=_body_edit(end + 1, record_leaf=leaves[o + 1]))
        case "record-twice":
            sid = lg.sid("o-8")
            events[o:o] = lg.session_events(sid, _window(rng), "h", "other")[0]
            events.append(("end", sid, {"closed": True, "files": {}, "record_leaf": s.q}))
            j["ledger"] = lg.lines(events)
        case "period-form":
            k = rng.choice([k for k, (kind, _, _) in enumerate(events) if kind.startswith("period")])
            kind, _, b = events[k]
            bad = rng.choice([{"period": rng.choice(["", ".p", "x" * 65, 5])}, {"lane": rng.choice([DROP, "a b"])}]
                             + ([{"key_sha512": rng.choice([DROP, "00"])}] if kind == "period-open" else
                                [{"windows": rng.choice([None, [{"root": "00"}]])}, {"complete": rng.choice([None, 1])}]))
            if rng.random() < 0.3:
                events[k] = (kind, rng.choice(["s-1", ""]), b)
            else:
                events[k] = (kind, None, _body(b, **bad))
            j["ledger"] = lg.lines(events)
        case "period-v1":
            b = {**lg.period_open} if lg.period_open else {"key_sha512": hashlib.sha512(s.pub).hexdigest(), "lane": "a",
                                                            "period": "p-0"}
            lg.v1 = lg.v1 + [b]
            events.insert(0, ("period-open", None, b))
            j["ledger"] = lg.lines(events)
        case "period-twice":
            again = rng.choice([{"period": "p-again"}, {"lane": "other"}])
            events.insert(p + 1, ("period-open", None, _body(lg.period_open, **again)))
            j["ledger"] = lg.lines(events)
        case "period-close":
            k = next(k for k, (kind, _, _) in enumerate(events) if kind == "period-close")
            b = events[k][2]
            ws = b["windows"]
            bad = rng.choice([{"windows": ws[:-1]}, {"windows": [{**w, "complete": not w["complete"]} for w in ws]},
                              {"windows": ws + [{"complete": False, "root": "ab" * 64, "session": "o-7"}]},
                              {"complete": not b["complete"]}, {"period": "p-never"}, {"lane": "other"}])
            events[k] = ("period-close", None, _body(b, **bad))
            j["ledger"] = lg.lines(events)
        case "past-cap":
            lg.period_open["cap"] = 1
            sid = lg.sid("o-7")
            events.insert(p + 1, ("reserve", sid, lg.reserve_body(lg.lane)) if lg.two_step
                          else ("open", sid, lg.open_body(_window(rng), "h", lg.lane)))
            j["ledger"] = lg.lines(events)
        case "cap":
            j["ledger"] = lg.lines(edit=_body_edit(p, cap=rng.choice([0, -1, Ch.CAP[1] + 1, "4", True, None, [4]])))
        case "circuit":
            j["ledger"] = lg.lines(edit=_body_edit(p, circuit=rng.choice(["AB" * 64, "aB" * 64, "", "-x", "a b", 5,
                                                                           "x" * 257, None, "~"])))
        case "no-cap":
            j["ledger"] = lg.lines(edit=_body_edit(p, **rng.choice([{"cap": DROP, "circuit": DROP},
                                                                     {"cap": None, "circuit": None}])))
        case "reserve-form":
            start, minutes = reserve["started_utc"], reserve["minutes"]
            bad = rng.choice([
                {"minutes": rng.choice([0, 721, True, "30", None, DROP])},
                {"expires_utc": rng.choice([Ch.later(start, minutes - 1), None, DROP, start + " "])},
                {"started_utc": rng.choice(["2026-02-29T12:00:00Z", "2100-02-29T12:00:00Z", "2026-10-20T24:00:00Z",
                                            "2026-10-20 12:00:00Z", "0000-01-01T00:00:00Z", "2026-10-20T12:00:00z",
                                            "2026-1-20T12:00:00Z", 5, DROP])},
                {"by": rng.choice([None, "", ".x", 5, DROP])}, {"for": rng.choice([None, "", "a b", 5])}])
            j["ledger"] = lg.lines(edit=_body_edit(r, **bad))
        case "reserve-again":
            match rng.randrange(3):
                case 0:
                    events.insert(rng.randint(r + 1, len(events)), ("reserve", s.id, lg.reserve_body(lg.lane)))
                case 1:
                    events[r] = ("reserve", rng.choice(SESSIONS + ["s20261019T000000Z-ABCD", "s2026101T000000Z-abcd"]),
                                 reserve)
                case _:
                    lg.v1 = lg.v1 + [reserve]
            j["ledger"] = lg.lines(events)
        case "open-unbound":
            if rng.random() < 0.3:
                del events[r]
                j["ledger"] = lg.lines(events)
            else:
                j["ledger"] = lg.lines(edit=_body_edit(o, reserve=rng.choice(
                    [True, False, None, str(r), r + 1, r - 1, [r], DROP if rng.random() < 0.2 else r + 2])))
        case "open-request":
            k = rng.choice(Ch.RESERVED)
            v = reserve.get(k)
            match k:
                case "by" | "peer" | "scheme":
                    other = "other"
                case "for":
                    other = DROP if "for" in reserve else "other"
                case "families":
                    other = [[f[0], True if f[1] == 1 else f[1] + 1] for f in v]
                case _:
                    other = {0: False, 1: True}.get(v, v + 1)
            j["ledger"] = lg.lines(edit=_body_edit(o, **{k: other}))
        case "opened-by":
            j["ledger"] = lg.lines(edit=_body_edit(o, opened_by=rng.choice(["other", "", "vy-Admin", None, 5, ["x"],
                                                                            DROP])))
        case "open-late":
            start, expiry = reserve["started_utc"], reserve["expires_utc"]
            late = Ch.later(expiry, rng.randint(1, 100)) or "x"
            early = "0001-01-01T00:00:00Z" if start != "0001-01-01T00:00:00Z" else late
            j["ledger"] = lg.lines(edit=_body_edit(o, started_utc=rng.choice(
                [late, early, start[:-3] + "60Z", "2026-02-30T12:00:00Z", start.lower(), 5, None, DROP])))
        case "open-after-close":
            at = rng.randint(r + 1, o)
            events.insert(at, ("period-close", None, lg.close_body(events[:at])))
            j["ledger"] = lg.lines(events)
        case "one-step-after-reserve":
            j["ledger"] = lg.lines(edit=_body_edit(o, reserve=DROP))
        case _:
            j["ledger"] = lg.lines()
    return j


HEAD_NAMES = ["honest", "no-heads", "empty", "heads-type", "head-refused", "fork", "key-form", "request"]


def head_case(rng: random.Random, name: str) -> Any:
    s = _ledger_session(rng)
    lg = Ledger(rng, s)
    leaves = lg.leaves()
    lines = lg.lines(leaves=leaves)
    heads = [lines[:rng.randint(0, len(lines))] for _ in range(rng.randint(0, 2))] + [lines]
    rng.shuffle(heads)
    j = {"heads": heads, "key": s.key.to_json()}
    match name:
        case "no-heads":
            j["heads"] = None
        case "empty":
            j["heads"] = []
        case "heads-type":
            j["heads"] = rng.choice(["x", 5, {"a": []}])
        case "head-refused":
            torn = lines[:-1] + [rng.choice([lines[-1][:-3], None, 5])]
            j["heads"] = rng.choice([[torn], heads + [torn], [None], [5, lines]])
        case "fork":
            k = rng.randrange(len(lines))
            forked = lg.lines(leaves=leaves, edit=_entry_edit(k, lambda e: e.update(body={**e["body"], "fork": 1})))
            j["heads"] = [*heads, forked[:rng.randint(k + 1, len(forked))]]
            rng.shuffle(j["heads"])
        case "key-form":
            j["key"] = key_case(rng, rng.choice(KEY_NAMES[1:]))
        case "request":
            del j[rng.choice(sorted(j))]
    return j


def _pin_lines(rng: random.Random, n: int) -> list[str]:
    return [Ch.pin_line(rng.randbytes(20).hex(), rng.randbytes(20).hex(), rng.choice(STARTS)) for _ in range(n)]


def _first_parent(rng: random.Random, pin: str) -> list[str]:
    """The auditor's first-parent list of main, `pin` on it."""
    main = [rng.randbytes(20).hex() for _ in range(rng.randint(0, 4))]
    main.insert(rng.randint(0, len(main)), pin)
    return main


def _pinned(rng: random.Random, lg: Ledger, name: str) -> tuple[list[str], list[str]]:
    """The host's published pins log and the auditor's first-parent list of main for `lg`'s window: its open's pin on
    main's first-parent line and recorded among the first lines of the log, which the open seals, the log sometimes
    longer; then, for `name`, the open pinned to a commit the log doesn't record ("unpinned"), or to one the log records
    that isn't on main's first-parent line, as a branch commit merged into main isn't ("branch-pin")."""
    n = rng.randint(1, 4)
    pins = _pin_lines(rng, n + rng.choice([0, 0, 1]))
    pin = lg.opened["pinned"]
    pins[rng.randrange(n)] = Ch.pin_line(pin, rng.randbytes(20).hex(), rng.choice(STARTS))
    lg.opened["pins"] = Ch.pins_seal(pins[:n])
    main = _first_parent(rng, pin)
    if name == "unpinned":
        lg.opened["pinned"] = rng.choice([c for c in main if c != pin] + [rng.randbytes(20).hex()])
    elif name == "branch-pin":
        main.remove(pin)
    return pins, main


SESSION_NAMES = ["honest", "no-ledger", "no-open", "unclosed-ledger", "other-host", "unsealed-log", "unlisted-log",
                 "wrong-session", "session-form", "no-record", "record-signature", "no-close", "close-form",
                 "close-signature", "close-renamed", "rechained", "log-tampered", "log-dropped", "log-form", "unclosed",
                 "key-form", "request", "record-leaf", "commitment", "hellos-form", "hellos-unsealed", "hellos-count",
                 "unpinned", "branch-pin"]


def session_case(rng: random.Random, name: str) -> Any:
    s = _ledger_session(rng)
    lg = Ledger(rng, s, closed=name != "unclosed-ledger")
    pins, main = _pinned(rng, lg, name)
    files = s.files()
    j = {"key": s.key.to_json(), "ledger": None, "main": main, "pins": pins, "session": files, "window": lg.window}
    lines = files["log"]
    end = lg.index("end", s.id)
    match name:
        case "no-ledger":
            pass
        case "no-open":
            j["ledger"] = lg.lines()
            j["window"] = _window(rng)
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
            j["session"] = rng.choice([_body(files, log=DROP), _body(files, hellos=DROP), {**files, "extra": 1}, None])
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
        case "record-leaf":
            q = rng.choice([x for x in range(2 ** s.h) if x != s.q])
            m = Ch.record_message(s.pub[:16], s.host, s.ch.commitment)
            files["record"] = Ch.ChallengerRecord(s.ch.commitment, s.host, lms.sign_at(s.k, q, m).encode()).to_json()
        case "commitment":
            j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"].update(commitment=rng.randbytes(64).hex())))
        case "hellos-form":
            files["hellos"] = rng.choice([None, "x", files["hellos"] + [5], {"lines": []}])
        case "hellos-unsealed":
            if rng.random() < 0.5:
                files["hellos"] = files["hellos"] + ["x"]
                j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"].update(
                    refused_hellos=len(files["hellos"]))))
            else:
                j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"]["files"].update(
                    {Ch.HELLOS_FILE: _flip(e["body"]["files"][Ch.HELLOS_FILE], 3)})))
        case "hellos-count":
            j["ledger"] = lg.lines(edit=_entry_edit(end, lambda e: e["body"].update(
                refused_hellos=e["body"]["refused_hellos"] + rng.choice([1, 2]))))
    if j.get("ledger", 0) is None and name != "no-ledger":
        j["ledger"] = lg.lines()
    return j


CREDITED_NAMES = ["honest", "other-circuit", "descriptor-edited", "no-period", "ledger", "descriptor-form", "unpinned",
                  "branch-pin", "no-main"]


def _descriptor(rng: random.Random) -> bytes:
    """A declared window Program's descriptor as the verifier holds it (canonical JSON), or bytes of any kind."""
    if rng.random() < 0.3:
        return rng.randbytes(rng.choice([1, 64, 1000]))
    calls = [{"module": f"model.layers.{i}.mlp", "k": rng.choice([4096, 14336])} for i in range(rng.randint(0, 6))]
    return canonical_json_text({"calls": calls, "root": "RemoteWindow", "schema": "verity-program/v1"}).encode()


def credited_case(rng: random.Random, name: str) -> Any:
    """A window in its lane's period, pinned to a logged commit on main's first-parent line (`_pinned`), whose circuit is
    the SHA-512 of the descriptor the verifier holds; then a period committing another circuit, the descriptor edited,
    the window in no period, a refused ledger, a descriptor not hex, a pin the log doesn't record, a logged pin off
    main's first-parent line, no copy of main."""
    if name == "ledger":
        j = ledger_case(rng, rng.choice(LEDGER_NAMES[1:]))
        if isinstance(j, dict):
            j["descriptor"] = _descriptor(rng).hex()
            j["pins"], j["main"] = _pin_lines(rng, rng.randint(1, 3)), _first_parent(rng, rng.randbytes(20).hex())
        return j
    s = _ledger_session(rng)
    lg = Ledger(rng, s, period=rng.choice(["open", "closed"]))
    pins, main = _pinned(rng, lg, name)
    d = _descriptor(rng)
    lg.period_open["circuit"] = rng.choice(CIRCUITS) if name == "other-circuit" else hashlib.sha512(d).hexdigest()
    j = {"descriptor": d.hex(), "key": s.key.to_json(), "ledger": lg.lines(), "main": main, "pins": pins,
         "window": lg.window}
    match name:
        case "descriptor-edited":
            j["descriptor"] = rng.choice([_flip(d.hex(), rng.randrange(len(d))), (d + b"\n").hex(), d[:-1].hex()])
        case "no-period":
            j["ledger"] = lg.lines([e for e in lg.events if not e[0].startswith("period")])
        case "descriptor-form":
            j["descriptor"] = rng.choice([None, 5, d.hex().upper() + "AB", d.hex() + "0", ["00"]])
        case "no-main":
            j["main"] = None
    return j


PINNED_NAMES = ["honest", "off-main", "pinned-later", "no-pins", "pins-form", "line-form", "cut", "edited", "no-seal",
                "ledger", "request", "branch-commit", "no-main", "main-form"]


def pinned_case(rng: random.Random, name: str) -> Any:
    """A window whose open names its host's pin and seals the first lines of the host's pins log, one of which pins it,
    the published log those lines and sometimes later pins, the pin on the auditor's first-parent list of main; then a
    commit off main or pinned only after the open, no pins log, one of another form or with a malformed line, one cut
    short of the seal or with a sealed line edited, dropped or moved, an open sealing none or a malformed seal, a refused
    ledger, a request without a key, a logged pin off main's first-parent line (a branch commit merged into main), no
    copy of main or one of another form."""
    if name == "ledger":
        j = ledger_case(rng, rng.choice(LEDGER_NAMES[1:]))
        if isinstance(j, dict):
            j["pins"], j["main"] = _pin_lines(rng, rng.randint(1, 3)), _first_parent(rng, rng.randbytes(20).hex())
        return j
    s = _ledger_session(rng)
    lg = Ledger(rng, s)
    n = rng.randint(1, 4)
    pins = _pin_lines(rng, n + rng.choice([0, 0, 1, 2]))
    at = rng.randrange(n)
    pins[at] = Ch.pin_line(lg.opened["pinned"], rng.randbytes(20).hex(), rng.choice(STARTS))
    lg.opened["pins"] = Ch.pins_seal(pins[:n])
    match name:
        case "off-main":
            pins[at] = Ch.pin_line(rng.randbytes(20).hex(), rng.randbytes(20).hex(), rng.choice(STARTS))
            lg.opened["pins"] = Ch.pins_seal(pins[:n])
        case "pinned-later":
            pins[at] = Ch.pin_line(rng.randbytes(20).hex(), rng.randbytes(20).hex(), rng.choice(STARTS))
            lg.opened["pins"] = Ch.pins_seal(pins[:n])
            pins.append(Ch.pin_line(lg.opened["pinned"], rng.randbytes(20).hex(), rng.choice(STARTS)))
        case "no-seal":
            d = Ch.log_digest(pins[:n])
            lg.opened["pins"] = rng.choice([
                DROP, None, {"lines": 0, "sha512": Ch.log_digest([])}, {"lines": True, "sha512": d},
                {"lines": str(n), "sha512": d}, {**Ch.pins_seal(pins[:n]), "at": 1}, {"lines": n, "sha512": d.upper()},
                {"lines": n}, [n, d]])
            if lg.opened["pins"] is DROP:
                del lg.opened["pins"]
    main = _first_parent(rng, lg.opened["pinned"])
    j = {"key": s.key.to_json(), "ledger": lg.lines(), "main": main, "pins": pins, "window": lg.window}
    match name:
        case "no-pins":
            j["pins"] = None
        case "branch-commit":
            j["main"] = [c for c in main if c != lg.opened["pinned"]]
        case "no-main":
            j["main"] = None
        case "main-form":
            j["main"] = rng.choice(["\n".join(main), {"main": main}, 5, [c.upper() for c in main],
                                    [c[:12] for c in main], main + [None], main + [rng.randbytes(32).hex()],
                                    [c + "\n" for c in main]])
        case "pins-form":
            j["pins"] = rng.choice(["\n".join(pins), {"lines": pins}, 5, True])
        case "line-form":
            line = pins[rng.randrange(len(pins))]
            bad = rng.choice([line.replace('"main"', '"Main"'), line.replace(":", ": ", 1), line[:-1] + ',"x":1}',
                              line.upper(), line[:-2] + '"}', line + " ", 5, None, json.loads(line), "",
                              line.replace("Z\"", "+00:00\""), line[:12] + line[13:]])
            j["pins"] = [bad if p == line else p for p in pins]
        case "cut":
            j["pins"] = pins[:rng.randrange(n)]
        case "edited":
            k = rng.randrange(n)
            moved = Ch.pin_line(json.loads(pins[k])["commit"], rng.randbytes(20).hex(), rng.choice(STARTS))
            edits = [pins[:k] + [moved] + pins[k + 1:]]
            if n > 1:
                edits.append(pins[:k] + pins[k + 1:])
                edits.append([pins[(k + 1) % n] if i == k else pins[k] if i == (k + 1) % n else p
                              for i, p in enumerate(pins)])
            j["pins"] = rng.choice(edits)
        case "request":
            del j[rng.choice(sorted(j))]
    return j


# ---- a draw's coins -----------------------------------------------------------------------------------------------------

def read(j: Any) -> Any:
    _request(j, ["after", "commitment", "read", "roots", "segment"])
    after, bytes_read = _hex(j["after"]), _hex(j["read"])
    if j["roots"] is None:
        roots = None
    elif type(j["roots"]) is list:
        roots = [_hex(r) for r in j["roots"]]
    else:
        raise D.Reject("roots is not a list")
    return _refusing(lambda: {"reason": T.read_reason(Ch.LiveCommitment.from_json(j["commitment"]), j["segment"], after,
                                                      bytes_read, roots)})


def draw_coins(j: Any) -> Any:
    _request(j, ["after", "coins", "read", "window"])
    after, bytes_read = _hex(j["after"]), _hex(j["read"])
    if type(j["window"]) is not str:
        raise D.Reject("window is not a string")
    return {"reason": T.coins_reason(j["coins"], j["window"], after, bytes_read, lms_verify)}


def _pairs(log: Sequence[dict]) -> list[str]:
    """A record's new log entries as the challenger's release log writes them, each root followed by its coin's release
    (`record_many` records a batch's roots first)."""
    releases = {e["index"]: e for e in log if e["event"] == "release"}
    return [canonical_json_text(x) for e in log if e["event"] == "root" for x in (e, releases[e["index"]])]


class Drawn:
    """A draw's coins from family `draw` of a ledger session's challenger, as `ChallengerCoins` opens them after `after`
    (one read, or a read and a rerun's extension), between other roots: the other family's, `draw` coins opened before
    the commitment (at least one when `before`), and later ones. `ledger` opens the session for its window. `after` is
    the commitment (drawn, by default), and `reads` reads the coins (`reads(coins)`, the bytes read), by default in one
    or two reads of drawn lengths."""

    def __init__(self, rng: random.Random, before: bool = False, after: bytes | None = None,
                 reads: Callable[[Callable[[int], bytes]], bytes] | None = None) -> None:
        s = self.s = _ledger_session(rng)
        s.ch = _challenger(rng, [("other", 2), ("draw", 12)])
        s.ch.sign_record(s.host, s.pub, lambda m: lms.sign_at(s.k, s.q, m).encode())
        c = s.ch.commitment
        s.lines = [canonical_json_text({"event": "commitment", "root": c.root.hex()})]
        clock = itertools.count(rng.choice([0, 1, 10 ** 12])).__next__
        self.records = {label: Ch.LiveRecord(s.ch, label, clock) for label, _ in c.families}
        for _ in range(rng.randint(0, 2)):
            self._step("other", lambda: self.records["other"].record(rng.randbytes(64)))
        draw = self.records["draw"]
        self.early = [self._step("draw", lambda: draw.record(rng.choice([rng.randbytes(64), bytes(64)])))
                      for _ in range(rng.randint(1 if before else 0, 2))]
        self.after = rng.randbytes(64) if after is None else after
        coins = T.ChallengerCoins(self.records["draw"])
        self.segment = coins.commit(self.after)
        if reads is None:
            self.read = b"".join(self._step("draw", lambda: coins(rng.randint(1, 3 * 64)))
                                 for _ in range(rng.choice([1, 1, 2])))
        else:
            self.read = self._step("draw", lambda: reads(coins))
        for _ in range(rng.randint(0, 2)):
            self._step("draw", lambda: self.records["draw"].record(rng.randbytes(64)))
        s.close = s.signed(None)
        self.ledger = Ledger(rng, s)
        self.pins, self.main = _pinned(rng, self.ledger, "honest")

    def _step(self, label: str, f):
        n = len(self.records[label].log)
        out = f()
        self.s.lines += _pairs(self.records[label].log[n:])
        return out

    def roots(self) -> list[str]:
        return [r.hex() for r in self.records["draw"].roots]

    def evidence(self, segment: dict) -> dict:
        return {"key": self.s.key.to_json(), "ledger": self.ledger.lines(), "main": self.main, "pins": self.pins,
                "segment": segment, "session": self.s.files()}

    def other_window(self, rng: random.Random) -> str:
        """The root of a window the ledger opens for another session, when it has one, else a window it doesn't open."""
        others = [b["window"]["root"] for kind, sid, b in self.ledger.events
                  if kind == "open" and sid != self.s.id and "root" in b["window"]]
        return rng.choice(others) if others else _window(rng)["root"]


#: each a mutation of the draw's segment or bytes, which both pieces refuse
DRAW_MUTATIONS = ["skipped", "wrong-opening", "before", "unread", "read-tampered", "other-after", "segment-form",
                  "opening-form"]


def _mutated(rng: random.Random, d: Drawn, name: str) -> tuple[dict, bytes, bytes]:
    """The segment, the bytes read and the commitment for mutation `name` (or the honest ones)."""
    seg, read_, after = d.segment.to_json(), d.read, d.after
    ops = seg["openings"]
    match name:
        case "skipped":
            if rng.random() < 0.5:
                seg = {**seg, "openings": ops[1:], "start": seg["start"] + 1}
                read_ = b"".join(o.coin for o in d.segment.openings[1:])
            else:
                seg = {**seg, "openings": ops[:-1]}
                read_ = b"".join(o.coin for o in d.segment.openings[:-1])
        case "wrong-opening":
            k = rng.randrange(len(ops))
            o = dict(ops[k])
            what = rng.choice(["coin", "salt", "path", "index", "epoch", "label"])
            if what in ("coin", "salt"):
                o[what] = _flip(o[what], rng.randrange(64))
            elif what == "path":
                o["path"] = [_flip(p) if i == 0 else p for i, p in enumerate(o["path"])] if o["path"] else ["00" * 64]
            elif what == "label":
                seg = {**seg, "label": "other"}
            else:
                o[what] += rng.choice([1, -1])
            seg = {**seg, "openings": ops[:k] + [o] + ops[k + 1:]}
        case "before":
            early = [o.to_json() for o in d.early]
            if rng.random() < 0.6:
                early = [{**o, "after": after.hex()} for o in early]
            seg = {**seg, "openings": early + ops, "start": seg["start"] - len(early)}
            read_ = b"".join(o.coin for o in d.early) + d.read
        case "unread":
            read_ = read_[:Ch.COIN_BYTES * (len(ops) - 1)]
        case "read-tampered":
            if rng.random() < 0.5:
                b = bytearray(read_)
                b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
                read_ = bytes(b)
            else:
                read_ = b"".join(o.coin for o in d.segment.openings) + rng.randbytes(rng.randint(1, 64))
        case "other-after":
            after = rng.randbytes(64)
        case "segment-form":
            seg = rng.choice([_body(seg, start=DROP), {**seg, "extra": 1},
                              {**seg, "start": rng.choice([-1, True, "0", None])},
                              {**seg, "label": rng.choice([None, 5, ["draw"]])},
                              {**seg, "openings": rng.choice([None, "x", {}])}, [seg], None])
        case "opening-form":
            k = rng.randrange(len(ops))
            bad = rng.choice([_body(ops[k], salt=DROP), {**ops[k], "coin": ops[k]["coin"].upper()},
                              {**ops[k], "index": str(ops[k]["index"])}, {**ops[k], "path": ["00"]}, None, 5])
            seg = {**seg, "openings": ops[:k] + [bad] + ops[k + 1:]}
    return seg, read_, after


READ_NAMES = ["honest"] + DRAW_MUTATIONS + ["commitment-form", "request"]


def _seeded(rng: random.Random) -> random.Random:
    """A plain generator seeded from `rng`: a `Drawn` session draws more bytes than Hypothesis's smallest example may
    hold when its `randoms` record each one."""
    return random.Random(rng.getrandbits(64))


def read_case(rng: random.Random, name: str) -> Any:
    rng = _seeded(rng)
    d = Drawn(rng, before=name == "before")
    seg, read_, after = _mutated(rng, d, name)
    roots = d.roots() if name != "honest" or rng.random() < 0.7 else None
    j = {"after": after.hex(), "commitment": d.s.ch.commitment.to_json(), "read": read_.hex(), "roots": roots,
         "segment": seg}
    match name:
        case "commitment-form":
            j["commitment"] = commitment_case(rng, rng.choice(COMMITMENT_NAMES[1:]))
        case "request":
            k = rng.choice(["after", "read", "roots", None])
            if k is None:
                del j[rng.choice(sorted(j))]
            else:
                j[k] = rng.choice(["AB", "0", 5]) if k != "roots" else rng.choice(["00", 5, {"draw": []}])
    return j


DRAW_COINS_NAMES = ["honest"] + DRAW_MUTATIONS + ["log-tampered", "other-window", "window-form", "key-form",
                                                   "off-main", "no-pins", "coins-form", "request"]


def draw_coins_case(rng: random.Random, name: str) -> Any:
    rng = _seeded(rng)
    d = Drawn(rng, before=name == "before")
    seg, read_, after = _mutated(rng, d, name)
    coins = d.evidence(seg)
    j = {"after": after.hex(), "coins": coins, "read": read_.hex(), "window": d.ledger.window["root"]}
    match name:
        case "log-tampered":
            log = coins["session"]["log"]
            coins["session"]["log"] = _set(log, rng.randrange(1, len(log)), **{"x": 1})
        case "other-window":
            j["window"] = d.other_window(rng)
        case "window-form":
            j["window"] = rng.choice(["", "AB" * 64, j["window"][:-2]])
        case "key-form":
            coins["key"] = key_case(rng, rng.choice(KEY_NAMES[1:]))
        case "off-main":
            coins["main"] = [c for c in coins["main"] if c != d.ledger.opened["pinned"]]
        case "no-pins":
            coins["pins"] = None
        case "coins-form":
            j["coins"] = rng.choice([_body(coins, segment=DROP), {**coins, "window": d.ledger.window}, None,
                                     [coins]])
        case "request":
            if rng.random() < 0.5:
                del j[rng.choice(sorted(j))]
            else:
                k = rng.choice(["after", "read", "window"])
                j[k] = rng.choice(["AB", "0", 5]) if k != "window" else rng.choice([5, None, {"root": j["window"]}])
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
HEAD = D.Port(PIECES, "challenger_head", head, _cases(head_case, HEAD_NAMES), package=PACKAGE, need_rejects=False,
              of_record="lean")
SESSION = D.Port(PIECES, "challenger_session", session, _cases(session_case, SESSION_NAMES), package=PACKAGE,
                 need_rejects=False, of_record="lean")
CREDITED = D.Port(PIECES, "challenger_credited", credited, _cases(credited_case, CREDITED_NAMES), package=PACKAGE,
                  need_rejects=False, of_record="lean")
PINNED = D.Port(PIECES, "challenger_pinned", pinned, _cases(pinned_case, PINNED_NAMES), package=PACKAGE,
                need_rejects=False, of_record="lean")
READ = D.Port(PIECES, "challenger_read", read, _cases(read_case, READ_NAMES), package=PACKAGE, of_record="lean")
DRAW_COINS = D.Port(PIECES, "challenger_draw_coins", draw_coins, _cases(draw_coins_case, DRAW_COINS_NAMES),
                    package=PACKAGE, of_record="lean")

pieces = D.server_fixture(COMMITMENT)

RUNS = [(COMMITMENT, commitment_case, COMMITMENT_NAMES, 300), (OPENING, opening_case, OPENING_NAMES, 300),
        (REASON, reason_case, REASON_NAMES, 400), (ORDER, order_case, ORDER_NAMES, 400), (KEY, key_case, KEY_NAMES, 200),
        (RECORD, record_case, RECORD_NAMES, 200), (TRUSTED, trusted_case, TRUSTED_NAMES, 300),
        (CLOSE, close_case, CLOSE_NAMES, 200), (RELEASED, released_case, RELEASED_NAMES, 300),
        (LEDGER, ledger_case, LEDGER_NAMES, 300), (HEAD, head_case, HEAD_NAMES, 100),
        (SESSION, session_case, SESSION_NAMES, 300), (CREDITED, credited_case, CREDITED_NAMES, 200),
        (PINNED, pinned_case, PINNED_NAMES, 200), (READ, read_case, READ_NAMES, 200),
        (DRAW_COINS, draw_coins_case, DRAW_COINS_NAMES, 200)]


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
