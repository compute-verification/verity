"""The proof's identifier v1 (`verity.core.protocols.pouw.identifier`, PROTOCOL.md "The proof's identifier (v1)"): every pinned case
recomputed from its inputs byte for byte, the vectors cover every named scheme, one field changed changes the bytes, the
bytes are RFC 8785 (restated here and checked against it, and written out literally once), and bad input is refused by
name."""

from __future__ import annotations

import hashlib
import json

import pytest

import identifier_vectors as V
from verity.core.protocols.pouw import identifier as I
from verity.core.protocols.pouw.audit import Epoch, Workload
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.ml.protocols.pouw.schemes import SCHEMES, get

PINNED = json.loads(V.OUT.read_text())
BY_CASE = {c["case"]: c for c in PINNED["cases"]}
SALT = bytes(range(32))
#: three matmuls on three weights, so that any one field of any one of them can change alone
BASE = ((3, (64, 1024, 64), 2), (4, (32, 1024, 128), 5), (5, (64, 2048, 64), 7))
FIELDS = ("index", "m", "k", "n", "weight")


def jcs(v) -> str:
    """RFC 8785 on the identifier's domain, restated: members in UTF-16 code-unit order of their names, integers in decimal,
    strings with RFC 8785's escapes, no whitespace."""
    if isinstance(v, bool) or v is None or isinstance(v, float):
        raise TypeError(f"{v!r} is outside the identifier's domain")
    if isinstance(v, int):
        assert abs(v) < 1 << 53
        return str(v)
    if isinstance(v, str):
        short = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
        return '"' + "".join(short.get(c) or (f"\\u{ord(c):04x}" if ord(c) < 0x20 else c) for c in v) + '"'
    if isinstance(v, list):
        return "[" + ",".join(map(jcs, v)) + "]"
    return "{" + ",".join(f"{jcs(k)}:{jcs(v[k])}" for k in sorted(v, key=lambda s: s.encode("utf-16-be"))) + "}"


def epoch_of(case: dict) -> Epoch:
    work = Workload(tuple(Matmul(i, Shape(m, k, n), w) for i, m, k, n, w in case["matmuls"]))
    return Epoch(work, {w: bytes.fromhex(r) for w, r in case["weight_roots"]}, bytes.fromhex(case["salt"]))


def build(matmuls=BASE, salt: bytes = SALT, roots: dict[int, bytes] | None = None) -> Epoch:
    work = Workload(tuple(Matmul(i, Shape(*shape), w) for i, shape, w in matmuls))
    return Epoch(work, roots or {w: hashlib.sha256(b"root %d" % w).digest() for w in work.weights()}, salt)


def test_every_case_is_generated():
    assert [c[0] for c in V.CASES] == list(BY_CASE)
    assert PINNED["format"] == I.FORMAT == "verity/pouw/identifier/v1"


@pytest.mark.parametrize("case", [c[0] for c in V.CASES])
def test_vectors_are_pinned(case):
    spec = next(c for c in V.CASES if c[0] == case)
    assert V.case_vector(*spec) == BY_CASE[case]


@pytest.mark.parametrize("case", list(BY_CASE))
def test_every_case_recomputes_from_its_inputs_byte_for_byte(case):
    c = BY_CASE[case]
    epoch = epoch_of(c)
    wb = I.workload_bytes(epoch)
    assert I.workload_json(epoch) == c["workload"] == json.loads(wb)
    assert wb == bytes.fromhex(c["workload_bytes"])
    assert I.workload_digest(epoch) == hashlib.sha512(wb).digest() == bytes.fromhex(c["workload_sha512"])
    ib = I.identifier_bytes(c["scheme"], epoch)
    assert I.identifier_v1(c["scheme"], epoch) == c["identifier"] == json.loads(ib)
    assert ib == bytes.fromhex(c["identifier_bytes"])
    assert hashlib.sha512(ib).digest() == bytes.fromhex(c["identifier_sha512"])
    assert c["identifier"] == {"format": I.FORMAT, "scheme": c["scheme"], "salt": c["salt"], "workload": c["workload_sha512"]}
    assert c["workload"]["weights_root"] == epoch.weights_root.hex()


def test_the_vectors_cover_every_named_scheme():
    assert {c["scheme"] for c in PINNED["cases"]} == set(SCHEMES)


def test_the_identifier_names_exactly_the_registered_schemes():
    assert I.SCHEMES == tuple(SCHEMES)


def test_the_64_bit_fields_are_exact_decimal_strings():
    """As JSON numbers they would be IEEE 754 doubles, and 2^53 + 1 would read back as 2^53."""
    got = [(u["index"], u["weight"]) for u in BY_CASE["pearl-c-nvfp4-v0-u64"]["workload"]["matmuls"]]
    assert got == [(str(V.U64_MAX), str((1 << 53) + 1)), (str(1 << 53), str((1 << 53) + 1)), ("0", str(V.U64_MAX - 1))]
    assert float((1 << 53) + 1) == float(1 << 53)


def test_the_matmuls_are_in_declaration_order():
    c = BY_CASE["pearl-c-sm120-v1-declared-order"]
    assert [u["index"] for u in c["workload"]["matmuls"]] == ["5", "2", "9"]


def variants():
    """(what changed, scheme, epoch): the base, and the base with exactly one thing changed."""
    yield "base", "pearl-c-h100-v1", build()
    for name in SCHEMES:
        if name != "pearl-c-h100-v1":
            yield f"scheme {name}", name, build()
    for byte in (0, 17, 31):
        for bit in (0, 7):
            salt = bytearray(SALT)
            salt[byte] ^= 1 << bit
            yield f"salt byte {byte} bit {bit}", "pearl-c-h100-v1", build(salt=bytes(salt))
    for pos in range(len(BASE)):
        for field in FIELDS:
            i, (m, k, n), w = BASE[pos]
            row = {"index": i, "m": m, "k": k, "n": n, "weight": w}
            row[field] += {"index": 100, "m": 16, "k": 128, "n": 64, "weight": 100}[field]
            changed = (row["index"], (row["m"], row["k"], row["n"]), row["weight"])
            yield f"matmul {pos} {field}", "pearl-c-h100-v1", build(BASE[:pos] + (changed,) + BASE[pos + 1:])
    for w in (2, 5, 7):
        roots = {v: hashlib.sha256(b"root %d" % v).digest() for v in (2, 5, 7)}
        roots[w] = bytes([roots[w][0] ^ 1]) + roots[w][1:]
        yield f"weight {w}'s root", "pearl-c-h100-v1", build(roots=roots)
    yield "matmuls 0 and 1 swapped", "pearl-c-h100-v1", build((BASE[1], BASE[0], BASE[2]))
    yield "the last matmul dropped", "pearl-c-h100-v1", build(BASE[:-1])
    yield "a matmul added", "pearl-c-h100-v1", build(BASE + ((6, (16, 1024, 64), 2),))


def test_changing_any_one_field_changes_the_bytes():
    seen: dict[bytes, str] = {}
    for what, scheme, epoch in variants():
        b = I.identifier_bytes(scheme, epoch)
        assert b not in seen, f"{what} gives the same identifier as {seen.get(b)}"
        seen[b] = what
    assert len(seen) == 1 + (len(SCHEMES) - 1) + 6 + len(BASE) * len(FIELDS) + 3 + 3


@pytest.mark.parametrize("case", list(BY_CASE))
def test_the_bytes_are_rfc_8785(case):
    c = BY_CASE[case]
    wb, ib = bytes.fromhex(c["workload_bytes"]), bytes.fromhex(c["identifier_bytes"])
    assert jcs(c["workload"]).encode() == wb and jcs(c["identifier"]).encode() == ib
    for b in (wb, ib):
        assert b.isascii() and not set(b) & set(b" \t\n\r\\")  # no whitespace, and no string here needs an escape


def test_the_bytes_are_canonical_written_out():
    epoch = build(((1, (16, 1024, 64), 2),), roots={2: b"\x22" * 32})
    root = epoch.weights_root.hex()
    wb = ('{"matmuls":[{"index":"1","k":1024,"m":16,"n":64,"weight":"2"}],"weights_root":"' + root + '"}').encode()
    assert I.workload_bytes(epoch) == wb
    assert I.identifier_bytes("ncp-v1", epoch) == (
        '{"format":"verity/pouw/identifier/v1","salt":"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",'
        '"scheme":"ncp-v1","workload":"' + hashlib.sha512(wb).hexdigest() + '"}').encode()


def test_a_scheme_is_named_by_its_registered_name():
    epoch = build()
    for name in SCHEMES:
        assert I.identifier_bytes(get(name), epoch) == I.identifier_bytes(name, epoch)


@pytest.mark.parametrize("salt", [b"", b"\0" * 31, b"\0" * 33, b"\0" * 63, b"\0" * 65, "00" * 32, bytearray(32)])
def test_a_salt_that_is_not_32_or_64_bytes_is_refused(salt):
    with pytest.raises(DomainError, match="the salt is 32 bytes, or a certified epoch's 64"):
        I.identifier_v1("pearl-c-h100-v1", build(salt=salt))


def test_a_certified_epochs_64_byte_salt_is_named_whole():
    salt = bytes(range(64))
    assert I.identifier_v1("pearl-c-h100-v1", build(salt=salt))["salt"] == salt.hex()


@pytest.mark.parametrize("scheme", ["ncp-v9", "", "NCP-V1", "pearl-c-h100-v1 ", None, object()])
def test_an_unregistered_scheme_is_refused(scheme):
    with pytest.raises(DomainError, match="no PoUW scheme"):
        I.identifier_bytes(scheme, build())


def test_a_dimension_past_2_53_is_refused():
    epoch = build(((0, (1 << 53, 1024, 64), 2),))
    with pytest.raises(DomainError, match=r"outside RFC 8785's domain: \$\.matmuls\[0\]\.m: integer"):
        I.identifier_bytes("ncp-v1", epoch)
    assert I.workload_json(build(((0, ((1 << 53) - 1, 1024, 64), 2),)))["matmuls"][0]["m"] == (1 << 53) - 1
