"""The anchored inputs' anchors (`verity.core.protocols.pouw.circuit.anchors`): the verifier builds them, refuses a prover's that differ,
and refuses a repeated call index in a window (the red team's X-SPC-72); a tile check's are public input regions, their
values unpacked into whole words."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

import circuit_fixtures as F
import pouw_regions_vectors as RV
from verity.ml.crypto.blake3 import blake3, keyed
from verity.core.protocols.pouw.circuit import ncp2, pc4, pc8, traced_as
from verity.core.protocols.pouw.circuit import reference as R
from verity.core.protocols.pouw.circuit.anchors import ANCHORED, Anchors, Ledger, public_ports, region_bytes, region_words
from verity.core.protocols.pouw.protocol import DomainError

SHAPE = ncp2.Shape(32, 64, 32)


def _anchors(call: R.Call) -> Anchors:
    return Anchors(call.salt, call.forward, call.layer, call.site, call.weight, tuple(call.rowids), tuple(call.colids))


def test_the_verifier_builds_the_anchored_inputs():
    """From its own anchors, the verifier's inputs are exactly the reference call's."""
    call = F.call(32, 64, 32)
    t = traced_as("ncp-v2-shift24", 32, 64, 32)
    args = call.args()
    assert t.call_args(_anchors(call), *args[:6]) == args
    assert t.weight_args(_anchors(call), call.weight_args()[0]) == call.weight_args()


def test_a_prover_s_anchored_inputs_must_be_the_verifier_s():
    call = F.call(32, 64, 32)
    an, fn = _anchors(call), ncp2.linear(SHAPE)
    an.check_call(fn, call.args())
    for field, value in (("rowids", list(range(32))), ("forward", [4, 0]), ("weight", [12, 0])):
        other = list(call.args())
        other[[p for p, _ in fn.signature()[0]].index(field)] = value
        with pytest.raises(DomainError):
            an.check_call(fn, other)
    an.check_weight(ncp2.weight(SHAPE), call.weight_args())


def test_anchors_that_cannot_be_a_call_s_are_refused():
    call = F.call(32, 64, 32)
    with pytest.raises(DomainError):
        Anchors(call.salt, 3, 5, 2, 11, (7,) * 32, tuple(call.colids)).check(SHAPE)       # repeated row ids
    with pytest.raises(DomainError):
        Anchors(call.salt, 3, 5, 2, 11, tuple(call.rowids), tuple(call.colids)[:16]).check(SHAPE)
    with pytest.raises(DomainError):
        Anchors(bytes(31), 3, 5, 2, 11, tuple(call.rowids), tuple(call.colids)).check(SHAPE)


def test_a_region_is_its_values_unpacked_into_whole_words():
    """`region_words`: a value of at most 16 bits one u16 word, a 32-bit one two, low half first, little-endian bytes; values
    wider than the region's, or a region short of a whole 128-bit word, are refused."""
    assert region_words(list(range(8)), 4) == list(range(8))
    assert region_words([0x12345678, 0x9ABCDEF0, 0, 1], 32) == [0x5678, 0x1234, 0xDEF0, 0x9ABC, 0, 0, 1, 0]
    assert region_bytes([0x5678, 0x1234]) == bytes([0x78, 0x56, 0x34, 0x12])
    for values, bits in (([16] + [0] * 7, 4), ([0] * 4, 4), ([0] * 3, 32), ([-1] + [0] * 7, 8)):
        with pytest.raises(DomainError):
            region_words(values, bits)


def test_pinned_region_vectors():
    """The regions vectors (`pouw_regions_vectors`) are what `TileAnchors.regions` gives; their seeds are the public values
    under the scheme's labels, Pearl-C4's the first 32 bytes of SHA-512 after the label's zero-padded 128-byte block and
    Pearl-C's (h0, Pearl's) keyed BLAKE3, and Pearl-C4's E line is its quarter of SHA-512 after the line label's block
    (`pearl_c4.line`), its nibbles each one u16 word."""
    pinned = json.loads(RV.OUT.read_text())
    assert RV.vectors() == pinned
    for c in pinned["cases"]:
        if c["circuit"] == "pc4":
            def h(role, msg):
                return hashlib.sha512(f"pearl-c4/v1/{role}\0".encode().ljust(128, b"\0") + msg).digest()[:32]
        else:
            def h(role, msg):
                return keyed(blake3(f"pearl-c/v0/{role}".encode()), msg)
        salt, ra, rb = (bytes.fromhex(c[p]) for p in ("salt", "root_a", "root_b"))
        seed_a = h("seed-A", salt + ra + rb + c["matmul"].to_bytes(8, "little"))
        assert (seed_a.hex(), h("seed-B", salt + rb).hex()) == (c["seed_a"], c["seed_b"])
        if c["circuit"] == "pc4":
            i = c["rows"][1]
            block = hashlib.sha512(b"pearl-c4/v2/line\0".ljust(128, b"\0") + seed_a + bytes([0])
                                   + (i // 4).to_bytes(4, "little")).digest()
            raw = block[16 * (i % 4):16 * (i % 4) + 16]
            e = next(r for r in c["units"][0]["regions"] if r["port"] == "e")
            assert e["bytes"] == bytes(b for x in raw for n in (x & 0xF, x >> 4) for b in (n, 0)).hex()


#: each anchor's rule id, as META's ``derivation`` names it: part of C-Flock's circuit pin, so a changed rule is a new version
RULES = {"pc4": {a: f"verity/pouw/anchor/pearl-c4/{a}/v2" for a in ANCHORED},
         "pc8": {a: f"verity/pouw/anchor/pearl-c/{a}/v1" for a in ANCHORED}}


def test_an_anchor_names_its_rule_not_its_function():
    """A region names its derivation by the rule's versioned id (`Derivation.rule`), one per scheme and anchor, so moving or
    renaming the function that computes it moves no circuit pin; a rule that is not such an id is refused."""
    for name, shape in (("pc4", pc4.Shape), ("pc8", pc8.Shape)):
        assert {a: d.rule for a, d in shape.derivations.items()} == RULES[name]
        moved = {a: dataclasses.replace(d, fn=lambda *args, f=d.fn: f(*args)) for a, d in shape.derivations.items()}
        alias = type("Moved", (), {"ports": shape.ports, "derivations": moved, "seeds": shape.seeds})
        for role in shape.ports:
            assert public_ports(alias, role) == public_ports(shape, role)
    d = pc4.DERIVATIONS["ea"]
    for bad in (f"{d.fn.__module__}.{d.fn.__qualname__}", "verity/pouw/anchor/pearl-c4/ea", "verity/pouw/anchor/ea/v1",
                "verity/pouw/anchor/pearl-c4/ea/v01"):
        with pytest.raises(ValueError, match="versioned id"):
            dataclasses.replace(d, rule=bad)


@pytest.mark.parametrize("case", RV.CASES, ids=[c[0] for c in RV.CASES])
def test_the_public_inputs_file_is_c_flocks(case):
    """`TileAnchors.public_inputs` is C-Flock's ``verity/flock-public-inputs/v1``: a key-sorted JSON header line naming each
    port's words, seed (the scheme's label) and derivation function, as META does, then each instance's regions, u16 LE,
    instance-major, the ports in the unit's parameter order; a tile, which has no region, or no instance is refused."""
    t = RV.anchors(*case)
    label = "pearl-c4/v1" if case[1] == "pc4" else "pearl-c/v0"
    for role, indexes in RV.FILES:
        head, body = t.public_inputs(role, indexes).split(b"\n", 1)
        h = json.loads(head)
        assert head == json.dumps(h, sort_keys=True).encode()
        assert h["format"] == "verity/flock-public-inputs/v1" and h["instances"] == len(indexes)
        names = list(t.shape.ports[role]["regions"])
        assert [p["port"] for p in h["ports"]] == names
        for p in h["ports"]:
            anchor = t.shape.ports[role]["regions"][p["port"]]
            d = t.shape.derivations[anchor]
            assert p["seed"] == f"{label}/{'seed-A' if d.seed == 'seed_a' else 'seed-B'}"
            assert p["derivation"] == RULES[case[1]][anchor]
            assert p["words"] == len(t.regions(role, indexes[0])[p["port"]])
        assert body == b"".join(region_bytes(t.regions(role, i)[p]) for i in indexes for p in names)
    for role, indexes in (("tile", (0,)), ("row_a", ())):
        with pytest.raises(DomainError):
            t.public_inputs(role, indexes)


def test_the_public_inputs_file_is_in_parameter_leaf_order():
    """The verifier's file holds each region in the unit's parameter leaf order, and its check refuses any other order:
    F_B in read order (transposed, code p of line t at word p·k + t, as C-Flock's staging first read it) or two instances
    swapped, every word otherwise right, is refused at its first differing word, as C-Flock's Lean verifier refuses it."""
    t = RV.anchors("pc4-64", "pc4", 64, None)
    own = t.public_inputs("row_a", (0, 1))
    t.check_public_inputs("row_a", (0, 1), own)
    head, body = own.split(b"\n", 1)
    words = [int.from_bytes(body[j:j + 2], "little") for j in range(0, len(body), 2)]
    sizes = [(p["port"], p["words"]) for p in json.loads(head)["ports"]]
    assert sizes == [("e", 32), ("fa", 2048), ("fb", 2048)]
    per = sum(n for _p, n in sizes)
    fb = slice(32 + 2048, per)
    read_order = []
    for i in range(2):
        w = words[i * per:(i + 1) * per]
        f = w[fb]
        w[fb] = [f[32 * line + p] for p in range(32) for line in range(64)]
        read_order += w
    misordered = head + b"\n" + region_bytes(read_order)
    assert misordered != own and sorted(read_order) == sorted(words)
    with pytest.raises(DomainError, match=r"^public input fb: word \d+ of instance 0 is not the verifier's own recomputation"):
        t.check_public_inputs("row_a", (0, 1), misordered)
    with pytest.raises(DomainError, match=r"^public input e: word \d+ of instance 0 "):
        t.check_public_inputs("row_a", (0, 1), t.public_inputs("row_a", (1, 0)))
    with pytest.raises(DomainError, match="header"):
        t.check_public_inputs("row_a", (0, 1), t.public_inputs("row_a", (0,)))


def test_a_window_refuses_a_repeated_call_index():
    """(forward, position) names one call (A11, `UniqueCallIndices`): a second call with it is refused."""
    call = F.call(32, 64, 32)
    ledger = Ledger()
    ledger.admit(_anchors(call), SHAPE)
    ledger.admit(Anchors(call.salt, call.forward + 1, call.layer, call.site, call.weight, tuple(call.rowids), tuple(call.colids)), SHAPE)
    with pytest.raises(DomainError):
        ledger.admit(_anchors(call), SHAPE)
