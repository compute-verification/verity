"""Catches the Lean check of a load's registrations (`NetworkVerifier.Load`, the ``load`` and ``transfer-registration``
functions of ``network-records verdicts``) deciding a load otherwise than the Python reference
(`load_reference.check_load`, `transfer_rows.check_registration`), which neither one's own tests can: the Lean check
is the verdict `relay_check.check_load` gives, and the reference the spec it was written from.

The cases are generated from one seed: loads at the tiny shape and one at the H200's, each value's head signed under a
test key, and one registration read for one expected transfer; each bad case is a good one with one thing changed.
On every case the two accept or refuse alike, and name the same refusal of the same values in the same order
(`NAMES`: the reference's words to the Lean's names), every refusal the reference has seen at least once
(`REFUSALS`). What the Lean check refuses and the reference doesn't is asserted as such, case by case (`STRICTER`):
the Lean check is the stricter one, never the looser."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import shutil
from typing import Any

import pytest

from tests.verity.protocols.accounting.network.verifier import load_reference as REF
from verity.commitments.crypto import lms
from verity.protocols.accounting.network.verifier import lean_audit as LA, load_check as LK
from verity.protocols.accounting.network.verifier import record_files as RF, transfer_rows as TR
from verity.protocols.accounting.network.verifier.certificate import (Certificate, Direction, LmsKey,
                                                                               certificate_message)

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]

SEED = 20
PARAMS = lms.Params(64, 1, 6)
PRIVATE = lms.keygen(PARAMS, hashlib.sha512(b"the load's attester").digest())
OTHER = lms.keygen(PARAMS, hashlib.sha512(b"another attester").digest())
KEY, SESSION = LmsKey(PARAMS, lms.public_key(PRIVATE)), bytes(range(32))
PROGRAM = hashlib.sha512(b"the reading program").hexdigest()
FIELDS = ("format", "link", "direction", "window", "kind", "value", "program", "words", "rows", "word_bits",
          "row_schema", "domain")

#: the reference's words for a registration's refusal, each with the Lean check's name of it, in the reference's order
NAMES = (
    (r"^a transfer is a ", "kind"),
    (r"^a transfer value's leaves are ", "leaf-width"),
    (r"^a transfer value's rows are ", "row-schema"),
    (r"^a program digest is lowercase hex of 64 bytes", "program"),
    (r"^a transfer registration is a JSON object", "not-object"),
    (r"^the transfer registration's (\w+) is .*, and the verifier's ", r"field:\1"),
    (r"^a transfer registration holds the keys ", "keys"),
    (r"doesn't read: its root is lowercase hex", "root"),
    (r"doesn't read: ", "certificate"),
    (r"^the transfer registration's certificate doesn't sign its head", "head"),
    (r"^the transfer registration's certificate doesn't verify under ", "signature"),
)
#: and for a load's, read before any registration
LOAD_NAMES = ((r"^no shape ", "shape"), (r"^not a load's registrations ", "format"),
              (r"^not one registration of each of ", "values"))
#: every refusal the reference has, by the Lean check's name: each is seen on a generated case
REFUSALS = ({n for _, n in NAMES if "\\" not in n} | {n for _, n in LOAD_NAMES} | {f"field:{k}" for k in FIELDS}
            | {"encode"})
#: what only the Lean check refuses, by its name
STRICTER = {"not-object", "format", "load-keys", "key-session", "certificate"}
#: of those, the document's own problems, given beside its registrations'
OWN = ((None, "load-keys"), (None, "key-session"))


def named(words: str, names=NAMES) -> str:
    """The Lean check's name of the reference's refusal `words`; ``encode`` for a head that doesn't encode, which the
    reference words by the field."""
    for pattern, name in names:
        m = re.search(pattern, words)
        if m:
            return m.expand(name)
    assert re.search(r"is an unsigned \d+-bit integer|is at most 65,535 UTF-8 bytes", words), words
    return "encode"


def sign(digest: bytes, link: str = LK.LINK, direction: Direction = Direction.INGRESS, window: int = 0, *, q: int = 0,
         session: bytes = SESSION, private: lms.PrivateKey = PRIVATE) -> Certificate:
    key_id = lms.public_key(private)[:16]
    message = certificate_message(key_id, session, link, direction, window, digest)
    return Certificate(key_id, link, direction, window, digest, lms.sign_at(private, q, message).encode())


def expected(v: LK.Value, link: str = LK.LINK, program: str = PROGRAM) -> dict[str, Any]:
    """The transfer the verifier expects for a load's value, as a ``transfer-registration`` case states it."""
    return {"link": link, "direction": "ingress", "window": LK.WINDOW, "kind": LK.KIND, "value": v.name,
            "program": program, "words": v.reg_words, "rows": v.leaves, "word_bits": v.word_bits, "row_schema": v.schema}


def registration(e: dict[str, Any], rng: random.Random, **signed: Any) -> dict[str, Any]:
    """The registration of the transfer `e` at a root of `rng`'s, its head signed (`sign`'s ``signed``)."""
    dom = TR.domain(e["program"], e["value"], e["words"], e["rows"], e["word_bits"]).domain_id
    head = TR.TransferHead(e["link"], Direction(e["direction"]), e["window"], e["kind"], e["value"],
                           bytes.fromhex(e["program"]), e["words"], e["rows"], dom, rng.randbytes(64), e["word_bits"],
                           e["row_schema"])
    return TR.registration(head, sign(head.digest(), e["link"], Direction(e["direction"]), e["window"],
                                      **{"q": rng.randrange(64)} | signed))


def load(rng: random.Random, shape: LK.Shape = LK.TINY, link: str = LK.LINK, program: str = PROGRAM,
         **signed: Any) -> dict[str, Any]:
    """An honest ``load.json`` (`load_relay.load_doc`'s keys) under the test key and session."""
    return {"format": LK.FORMAT, "shape": shape.name, "link": link, "window": LK.WINDOW, "program": program,
            "link_format": "plain", "key": RF.public_json(KEY), "session": SESSION.hex(),
            "values": [registration(expected(v, link, program), rng, **signed) for v in LK.manifest(shape)]}


# --- the two checks, each as (accepted values, [(value or None, the refusal's name)]) --------------------------------

def python_load(doc: Any) -> tuple[int, int, list[tuple[str | None, str]]]:
    summary, problems = REF.check_load(doc, KEY, SESSION)
    out = []
    for p in problems:
        m = re.fullmatch(r"load(?: (\S+))?: (.*)", p, re.S)
        out.append((m[1], named(m[2], NAMES if m[1] else LOAD_NAMES)))
    return summary["values"], summary["registered"], out


def lean_load(doc: Any) -> tuple[int, int, list[tuple[str | None, str]]]:
    got = LA.SERVER.call("load", {"key": RF.public_json(KEY), "session": SESSION.hex(), "load": doc})
    assert set(got) == {"values", "registered", "problems"}, got
    return got["values"], got["registered"], [(p.get("value"), p["refusal"]) for p in got["problems"]]


def reference_has(problems: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    """The Lean check's problems but the document's own that the reference doesn't read (`OWN`)."""
    return [p for p in problems if p not in OWN]


def python_registration(e: dict[str, Any], reg: Any) -> str | None:
    try:
        TR.check_registration(reg, KEY, SESSION, link=e["link"], direction=Direction(e["direction"]), window=e["window"],
                              kind=e["kind"], value=e["value"], program=e["program"], words=e["words"], rows=e["rows"],
                              word_bits=e["word_bits"], schema=e["row_schema"])
    except ValueError as ex:
        return named(str(ex))
    return None


def lean_registration(e: dict[str, Any], reg: Any) -> str | None:
    got = LA.SERVER.call("transfer-registration", {"key": RF.public_json(KEY), "session": SESSION.hex(), "expected": e,
                                                    "registration": reg})
    assert set(got) <= {"refusal", "why"} and "refusal" in got, got
    return got["refusal"]


# --- one registration -------------------------------------------------------------------------------------------------

def other(v: Any, rng: random.Random) -> list[Any]:
    """Values a field doesn't hold: of its type, and of every other JSON type, the number's equals among them."""
    if isinstance(v, str):
        same = [v + "x", v[:-1], v.upper() if v.upper() != v else v.lower(), ""]
    else:
        same = [v + 1, v + rng.randrange(2, 1 << 20), 1 << 70]
    return same + [float(v) if isinstance(v, int) else 0, v == 1 if isinstance(v, int) else 1, None, [v], {"v": v},
                   str(v) if isinstance(v, int) else 0.5, 1e22, -1]


def flipped(h: str, rng: random.Random) -> str:
    at = rng.randrange(len(h))
    return h[:at] + ("0" if h[at] != "0" else "1") + h[at + 1:]


def registration_cases(rng: random.Random, pick=None) -> list[tuple[str, dict[str, Any], Any]]:
    """`(what, expected, registration)`: good registrations, and each with one thing wrong. The good ones are `pick`'s
    where it is given, else of any link, program, execution phase and row format."""
    values = LK.manifest(LK.TINY)
    out: list[tuple[str, dict[str, Any], Any]] = []

    def good() -> tuple[dict[str, Any], dict[str, Any]]:
        if pick is not None:
            return pick()
        e = expected(rng.choice(values), rng.choice([LK.LINK, "s1/uplink", "", "é"]),
                     hashlib.sha512(rng.randbytes(8)).hexdigest())
        if rng.random() < 0.3:
            e |= {"word_bits": 32}
        if rng.random() < 0.3:
            e |= {"row_schema": TR.SCHEMAS[1]}
        if rng.random() < 0.3:
            e |= {"window": rng.choice([1, 7, (1 << 40) + 3])}
        return e, registration(e, rng)

    for _ in range(12):
        out.append(("good", *good()))
    # the expected transfer isn't one: the registration is the one signed for the transfer it was changed from
    for change in ({"kind": "store"}, {"kind": ""}, {"direction": "egress"}, {"word_bits": 8}, {"word_bits": 0},
                   {"word_bits": 64}, {"row_schema": "hm96-sha512/row/v1"}, {"row_schema": ""}, {"program": "00"},
                   {"program": PROGRAM.upper()}, {"program": PROGRAM + "0"}, {"program": ""},
                   {"program": "g" + PROGRAM[1:]}):
        e, reg = good()
        out.append((f"expected {change}", e | change, reg | change))
        out.append((f"expected {change}, the registration the good one", e | change, reg))
    for reg in (None, [], "x", 0, 1.5, True, [good()[1]]):
        out.append((f"registration {reg!r:.20}", good()[0], reg))
    for k in FIELDS:
        e, reg = good()
        for v in other(reg[k], rng):
            out.append((f"{k}: {v!r:.30}", e, reg | {k: v}))
        out.append((f"no {k}", e, {x: y for x, y in reg.items() if x != k}))
        # the head the attester signed is another transfer's: every field its own, one not the verifier's
        if k in ("link", "window", "value", "program", "words", "rows", "word_bits", "row_schema"):
            theirs = e | {k: {"link": e["link"] + "/b", "window": e["window"] + 1, "value": e["value"] + ".b",
                                "program": hashlib.sha512(b"another").hexdigest(), "words": e["words"] * 2,
                                "rows": e["rows"] + 1, "word_bits": 48 - e["word_bits"],
                                "row_schema": TR.SCHEMAS[TR.SCHEMAS[0] == e["row_schema"]]}[k]}
            out.append((f"signed for another {k}", e, registration(theirs, rng)))
    for extra in ({"salt": "00"}, {"": 0}, {"Root": "00" * 64}):
        e, reg = good()
        out.append((f"another key {extra}", e, reg | extra))
    for k in ("root", "certificate"):
        e, reg = good()
        out.append((f"no {k}", e, {x: y for x, y in reg.items() if x != k}))
    for root in ("", "00" * 63, "00" * 65, None, 5, ["00" * 64], "0g" * 64, "AB" * 64):
        e, reg = good()
        out.append((f"root {root!r:.20}", e, reg | {"root": root}))
    e, reg = good()
    out.append(("root another, lowercase hex", e, reg | {"root": flipped(reg["root"], rng)}))
    for c in (None, [], "x", {}, {"key": "00"}, lambda c: c | {"salt": "00"},
              lambda c: {k: v for k, v in c.items() if k != "signature"}, lambda c: c | {"key": 5},
              lambda c: c | {"digest": None}, lambda c: c | {"signature": ["00"]}, lambda c: c | {"key": "0"},
              lambda c: c | {"digest": "zz"}, lambda c: c | {"signature": "0x00"}):
        e, reg = good()
        out.append(("certificate not one", e, reg | {"certificate": c(reg["certificate"]) if callable(c) else c}))
    for k in ("key", "digest", "signature"):
        e, reg = good()
        c = reg["certificate"]
        out.append((f"certificate {k} flipped", e, reg | {"certificate": c | {k: flipped(c[k], rng)}}))
        out.append((f"certificate {k} cut", e, reg | {"certificate": c | {k: c[k][:-2]}}))
        out.append((f"certificate {k} empty", e, reg | {"certificate": c | {k: ""}}))
    e, reg = good()
    theirs = registration(e, rng)
    out.append(("another registration's certificate", e, reg | {"certificate": theirs["certificate"]}))
    out.append(("another registration's digest, this signature", e,
                reg | {"certificate": reg["certificate"] | {"digest": theirs["certificate"]["digest"]}}))
    for signed in ({"session": bytes(range(1, 33))}, {"private": OTHER}, {"link": "s9/uplink"},
                   {"direction": Direction.EGRESS}, {"window": 1}):
        e = good()[0]
        dom = TR.domain(e["program"], e["value"], e["words"], e["rows"], e["word_bits"]).domain_id
        head = TR.TransferHead(e["link"], Direction.INGRESS, e["window"], e["kind"], e["value"],
                               bytes.fromhex(e["program"]), e["words"], e["rows"], dom, rng.randbytes(64),
                               e["word_bits"], e["row_schema"])
        args = {"link": e["link"], "direction": Direction.INGRESS, "window": e["window"]} | signed
        cert = sign(head.digest(), args.pop("link"), args.pop("direction"), args.pop("window"), **args)
        out.append((f"signed under {sorted(signed)}", e, TR.registration(head, cert)))
    # a head that doesn't encode: the registration states the verifier's own fields, and nothing signs them
    for change in ({"words": 1 << 32}, {"rows": 1 << 32}, {"words": (1 << 40) + 1}, {"link": "l" * 65536},
                   {"value": "v" * 65536}, {"link": "é" * 40000}):
        e, reg = good()
        e = e | change
        dom = TR.domain(e["program"], e["value"], e["words"], e["rows"], e["word_bits"]).domain_id
        out.append((f"expected {sorted(change)} past its width", e, reg | change | {"domain": dom.hex()}))
    for change in ({"rows": 0}, {"words": 0}, {"value": ""}, {"link": "l" * 65535}, {"value": "v" * 65535},
                   {"words": (1 << 32) - 1, "rows": (1 << 32) - 1}, {"window": (1 << 49) + 1}):
        e = good()[0] | change
        out.append((f"good at {sorted(change)}'s edge", e, registration(e, rng)))
    return out


def test_one_registration_is_read_alike():
    rng = random.Random(SEED)
    cases, seen = registration_cases(rng), {}
    for what, e, reg in cases:
        e, reg = json.loads(json.dumps(e)), json.loads(json.dumps(reg))
        py, ln = python_registration(e, reg), lean_registration(e, reg)
        assert py == ln, (what, py, ln, e, reg)
        seen[py] = seen.get(py, 0) + 1
    assert seen.get(None, 0) >= 12, seen
    assert REFUSALS - {n for _, n in LOAD_NAMES} <= set(seen), sorted(REFUSALS - set(seen))
    assert len(cases) >= 250, len(cases)


# --- a load -----------------------------------------------------------------------------------------------------------

def load_cases(rng: random.Random) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for link, program in ((LK.LINK, PROGRAM), ("s1/uplink", hashlib.sha512(b"another program").hexdigest()), ("", PROGRAM)):
        out.append(("good", load(rng, link=link, program=program)))
    out.append(("good, the H200's", load(rng, LK.H200)))
    good = load(rng)
    regs = good["values"]
    for shape in ("", "tiny", None, 5, ["tiny-h32-l1"], LK.H200.name, "TINY-H32-L1"):
        out.append((f"shape {shape!r}", good | {"shape": shape}))
    out.append(("no shape", {k: v for k, v in good.items() if k != "shape"}))
    for change in ({"format": "x"}, {"format": None}, {"format": LK.FORMAT + "/"}, {"window": 1}, {"window": "0"},
                   {"window": None}, {"window": -1}, {"window": [0]}, {"link": None}, {"link": 0}, {"link": ["s0/uplink"]}):
        out.append((f"{change}", good | change))
    for k in ("format", "window", "link"):
        out.append((f"no {k}", {x: y for x, y in good.items() if x != k}))
    for values in (None, {}, "x", 19, [], regs[:-1], regs + [regs[0]], regs[1:], {str(i): r for i, r in enumerate(regs)}):
        out.append((f"values {values!r:.20}", good | {"values": values}))
    out.append(("no values", {k: v for k, v in good.items() if k != "values"}))
    out.append(("the H200's shape over the tiny's values", good | {"shape": LK.H200.name}))
    for program in (None, 5, "00", PROGRAM.upper(), hashlib.sha512(b"another program").hexdigest()):
        out.append((f"program {program!r:.20}", good | {"program": program}))
    out.append(("no program", {k: v for k, v in good.items() if k != "program"}))
    out.append(("another link", good | {"link": "s1/uplink"}))
    out.append(("two values swapped", good | {"values": [regs[1], regs[0], *regs[2:]]}))
    out.append(("the values reversed", good | {"values": regs[::-1]}))
    out.append(("one value twice", good | {"values": [regs[0], regs[0], *regs[2:]]}))
    out.append(("signed in another session", load(rng, session=bytes(range(1, 33))) | {"session": SESSION.hex()}))
    out.append(("signed under another key", load(rng, private=OTHER)))
    # one registration of the load changed, as one registration's cases change it: every problem is that value's
    wanted = [expected(v) for v in LK.manifest(LK.TINY)]

    def pick() -> tuple[dict[str, Any], dict[str, Any]]:
        e = rng.choice(wanted)
        return e, registration(e, rng)

    for what, e, reg in registration_cases(rng, pick):
        if e in wanted:
            at = wanted.index(e)
            out.append((f"value {at} {what}", good | {"values": regs[:at] + [reg] + regs[at + 1:]}))
    for _ in range(40):
        doc = copy.deepcopy(good)
        for at in rng.sample(range(len(regs)), rng.randrange(1, 5)):
            k = rng.choice([*FIELDS, "root", "key", "digest", "signature", "drop"])
            reg = doc["values"][at]
            if k == "drop":
                del reg[rng.choice(sorted(reg))]
            elif k in ("key", "digest", "signature"):
                reg["certificate"][k] = flipped(reg["certificate"][k], rng)
            elif k == "root":
                reg[k] = flipped(reg[k], rng)
            else:
                reg[k] = rng.choice(other(reg[k], rng))
        out.append(("several values changed", doc))
    return out


def test_a_load_is_read_alike():
    """The H200's load, 733 registrations, is among the cases: a load at the tiny shape can't hold the Lean manifest's
    layers to the reference's, having one."""
    rng = random.Random(SEED + 1)
    cases, seen, accepted = load_cases(rng), set(), 0
    for what, doc in cases:
        doc = json.loads(json.dumps(doc))
        py, ln = python_load(doc), lean_load(doc)
        assert bool(py[2]) == bool(ln[2]), (what, py, ln)
        assert py == (ln[0], ln[1], reference_has(ln[2])), (what, py, ln)
        accepted += not py[2]
        seen |= {name for _, name in py[2]}
        assert (py[1] == py[0] and py[0] > 0) == (not py[2]), (what, py)
    assert accepted >= 16, accepted
    assert python_load(cases[3][1]) == (733, 733, []), cases[3][0]
    assert {n for _, n in LOAD_NAMES} <= seen, seen
    assert REFUSALS - {"kind", "leaf-width", "row-schema", "encode"} <= seen, sorted(REFUSALS - seen)
    assert len(cases) >= 200, len(cases)


def test_what_only_the_lean_check_refuses():
    """Each case the reference accepts, or can't read, and the Lean check refuses by the name given."""
    rng = random.Random(SEED + 2)
    good = load(rng)
    regs = good["values"]
    assert python_load(good)[2] == lean_load(good)[2] == []

    def certificate(change) -> dict[str, Any]:
        return good | {"values": [regs[0] | {"certificate": change(regs[0]["certificate"])}, *regs[1:]]}

    accepted = {
        "load-keys": [good | {"note": "x"}, good | {"link_format": 5},
                      {k: v for k, v in good.items() if k != "link_format"}],
        "key-session": [good | {"session": "00" * 32}, good | {"session": SESSION.hex().upper()},
                        good | {"key": good["key"] | {"params": [64, 1, 7]}},
                        good | {"key": good["key"] | {"key_id": "00" * 16}},
                        good | {"key": RF.public_json(LmsKey(PARAMS, lms.public_key(OTHER)))}, good | {"key": None}],
        "format": [good | {"window": False}, good | {"window": 0.0}],
    }
    for name, docs in accepted.items():
        for doc in docs:
            doc = json.loads(json.dumps(doc))
            assert python_load(doc)[2] == [], (name, doc)
            n = len(regs) if name != "format" else 0
            assert lean_load(doc) == (n, n, [(None, name)]), (name, doc)
    for k in ("key", "session"):
        doc = {x: y for x, y in good.items() if x != k}
        assert python_load(doc)[2] == [] and lean_load(doc) == (len(regs), len(regs), list(OWN)), k
    for k in ("key", "digest", "signature"):
        for change in (str.upper, lambda h: h[:2] + " " + h[2:]):
            doc = certificate(lambda c: c | {k: change(c[k])})
            assert python_load(doc)[2] == [], k
            assert lean_load(doc) == (len(regs), len(regs) - 1, [("tok", "certificate")]), k
    for doc in ([], None, "x", 5):
        with pytest.raises(AttributeError):
            REF.check_load(doc, KEY, SESSION)
        assert lean_load(doc) == (0, 0, [(None, "not-object")])
    assert STRICTER == set(accepted) | {"certificate", "not-object"}


def test_a_case_without_the_deployments_key_and_session_is_no_verdict():
    """`relay_check.check_load` refuses a load the Lean check gives no verdict on."""
    good = load(random.Random(SEED + 3))
    key, session = RF.public_json(KEY), SESSION.hex()
    for case in ({"key": key, "session": session}, {"key": key, "session": session, "load": good, "accept": True},
                 {"key": key | {"params": [64, 1, 6.5]}, "session": session, "load": good},
                 {"key": key | {"params": [32, 1, 6]}, "session": session, "load": good},
                 {"key": key | {"public": key["public"].upper()}, "session": session, "load": good},
                 {"key": key | {"public": key["public"][:-2]}, "session": session, "load": good},
                 {"key": key | {"key_id": "00" * 16}, "session": session, "load": good},
                 {"key": {k: v for k, v in key.items() if k != "key_id"}, "session": session, "load": good},
                 {"key": key, "session": session.upper(), "load": good}, {"key": key, "session": 5, "load": good}):
        with pytest.raises(ValueError, match="outside the Lean audit's domain"):
            LA.SERVER.call("load", case)
