"""Pearl-C4's anchored inputs' public input regions, on `test_circuit_anchors`' helpers: the regions vectors
(`pearl_c4_regions_vectors`) are what `TileAnchors.regions` gives, its anchors name their rules, and its `--public-inputs`
files are Flock's, in the unit's parameter leaf order."""
from __future__ import annotations

import hashlib
import json

import pearl_c4_regions_vectors as RV4
import test_circuit_anchors as T
from verity.protocols.experimental.pearl_c4.prover import pc4
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.anchors import ANCHORED

#: each anchor's rule id, as META's ``derivation`` names it: part of Flock's circuit pin, so a changed rule is a new version
RULES = {a: f"verity/pouw/anchor/pearl-c4/{a}/v2" for a in ANCHORED}


def test_pinned_region_vectors():
    """Pearl-C4's seeds are the first 32 bytes of SHA-512 after the label's zero-padded 128-byte block, and its E line is
    its quarter of SHA-512 after the line label's block (`pearl_c4.line`), its nibbles each one u16 word."""
    pinned = json.loads(RV4.OUT.read_text())
    assert RV4.vectors() == pinned
    for c in pinned["cases"]:
        def h(role, msg):
            return hashlib.sha512(f"pearl-c4/v1/{role}\0".encode().ljust(128, b"\0") + msg).digest()[:32]
        seed_a = T.seeds_are_the_labels(c, h)
        i = c["rows"][1]
        block = hashlib.sha512(b"pearl-c4/v2/line\0".ljust(128, b"\0") + seed_a + bytes([0])
                               + (i // 4).to_bytes(4, "little")).digest()
        raw = block[16 * (i % 4):16 * (i % 4) + 16]
        e = next(r for r in c["units"][0]["regions"] if r["port"] == "e")
        assert e["bytes"] == bytes(b for x in raw for n in (x & 0xF, x >> 4) for b in (n, 0)).hex()


def test_an_anchor_names_its_rule_not_its_function():
    T.anchor_names_its_rule(pc4.Shape, RULES, pc4.DERIVATIONS["ea"], "pearl-c4")


def test_the_public_inputs_file_is_c_flocks():
    T.public_inputs_file_is_c_flocks(RV4.anchors(*RV4.CASES[0]), "pearl-c4/v1", RULES)


def test_the_public_inputs_file_is_in_parameter_leaf_order():
    T.file_is_in_parameter_leaf_order(RV4.anchors("pc4-64", "pc4", 64, None), 64)
