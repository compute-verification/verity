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
malformed form. Each piece answers ``{"refused": why}`` where the challenger refuses, so the reasons are compared too.
"""

from __future__ import annotations

import functools
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.sampled_proofs import challenger as Ch
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

pieces = D.server_fixture(COMMITMENT)

RUNS = [(COMMITMENT, commitment_case, COMMITMENT_NAMES, 300), (OPENING, opening_case, OPENING_NAMES, 300),
        (REASON, reason_case, REASON_NAMES, 400), (ORDER, order_case, ORDER_NAMES, 400), (KEY, key_case, KEY_NAMES, 200),
        (RECORD, record_case, RECORD_NAMES, 200), (TRUSTED, trusted_case, TRUSTED_NAMES, 300)]


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
