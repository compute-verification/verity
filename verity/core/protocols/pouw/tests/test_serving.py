"""`verity.core.protocols.pouw.serving`: every Pearl-C name in use parses and prints back unchanged, the hashing format is a switch whose
tree hash is the scheme's, and a schedule is refused where its format needs A's tree on the critical path."""
from __future__ import annotations

import pytest
from verity.core.protocols.pouw.serving import (
    DEFERRED,
    HASH_FORMATS,
    SCHEDULES,
    SERIAL,
    GateRecord,
    KernelVariant,
    PassCommitment,
    Schedule,
    SchemeId,
    ShapeRule,
    StepIO,
    hash_format,
    schedule_refusals,
    variant_refusals,
)

#: every Pearl-C name in use: `schemes.SCHEMES`, `pouw_pearl_c_device.SCHEMES`, Pearl-C4 and its -h3, and PearlC's
#: price-override spelling
NAMES_IN_USE = ("pearl-c-h100-v0", "pearl-c-h100-v1", "pearl-c-h100-v1-h3", "pearl-c-sm120-v1-h3", "pearl-c-h100-v1@sm_120",
                "pearl-c-sm120-v1", "pearl-c-sm120-unpromoted-v1-h3", "pearl-c-nvfp4-v0", "pearl-c-nvfp4-v0-h3",
                "pearl-c-mxfp4-v0")


@pytest.mark.parametrize("name", NAMES_IN_USE)
def test_every_name_in_use_parses_and_prints_back(name):
    assert SchemeId.parse(name).name == name


def test_the_parts_are_the_names_parts():
    s = SchemeId.parse("pearl-c-sm120-unpromoted-v1-h3")
    assert (s.family, s.device, s.forming, s.precision, s.hashing, s.prices) == ("pearl-c", "sm120-unpromoted", "v1", "e4m3", "h3", None)
    c4 = SchemeId.parse("pearl-c-nvfp4-v0")
    assert (c4.family, c4.device, c4.precision, c4.hashing) == ("pearl-c4", "sm120", "nvfp4", "h0")
    assert SchemeId.parse("pearl-c-h100-v1@sm_120").prices == "sm_120"


def test_the_hashing_format_is_a_switch():
    h3 = SchemeId.parse("pearl-c-sm120-v1-h3")
    assert h3.with_hashing("h0").name == "pearl-c-sm120-v1"
    assert SchemeId.parse("pearl-c-sm120-v1").with_hashing("h3") == h3
    assert SchemeId.parse("pearl-c-nvfp4-v0").with_hashing("h3").name == "pearl-c-nvfp4-v0-h3"


@pytest.mark.parametrize("name, why", [
    ("pearl-c-a100-v1-h3", "device 'a100'"),
    ("pearl-c-sm120-hot-v1-h3", "device 'sm120-hot'"),
    ("pearl-c-sm120-v9-h3", "forming 'v9'"),
    ("pearl-c-sm120-v1-h7", "hashing 'h7'"),
    ("pearl-c-sm120-v1-h1", "hashing 'h1'"),
    ("pearl-c-sm120-v1-h2", "hashing 'h2'"),
    ("pearl-c-sm120-v1-h3s", "not a Pearl-C scheme name"),
    ("pearl-fp8-v4", "not a Pearl-C scheme name"),
    ("ncp-v2", "not a Pearl-C scheme name"),
])
def test_a_name_outside_the_grammar_is_refused_by_what_is_wrong(name, why):
    with pytest.raises(ValueError, match=why):
        SchemeId.parse(name)


def test_the_formats_are_h0_and_sha512_h3():
    """One hash, SHA-512 (Daniel, 2026-10-06): h3's trees are hm96-sha512 and its rows seed on their SHA-512 digests; h0 is
    the unhashed default."""
    assert list(HASH_FORMATS) == ["h0", "h3"]
    h0, h3 = HASH_FORMATS["h0"], HASH_FORMATS["h3"]
    assert (h0.tree_hash, h0.seed_a, h0.row_cap, h0.row_digest) == ("sha256", "unit-root", False, None)
    assert (h3.tree_hash, h3.seed_a, h3.row_cap, h3.row_digest) == ("hm96-sha512", "row", True, "sha512/row-seg/v1")


def test_hash_format_reads_a_name_an_id_or_a_scheme_object():
    class Scheme:
        hashing = "h3"
    assert hash_format("pearl-c-sm120-v1-h3") is HASH_FORMATS["h3"]
    assert hash_format(SchemeId.parse("pearl-c-h100-v1")) is HASH_FORMATS["h0"]
    assert hash_format(Scheme()) is HASH_FORMATS["h3"]
    with pytest.raises(ValueError, match="no hashing format"):
        hash_format(object())


def test_the_serial_and_deferred_schedules_run_under_every_format():
    assert set(SCHEDULES) == {"serial", "deferred", "deferred-caller", "deferred-a"}
    for f in HASH_FORMATS.values():
        for s in SCHEDULES.values():
            if s.name == "deferred-a":
                continue
            assert schedule_refusals(f, s) == []
    assert schedule_refusals(HASH_FORMATS["h3"], SCHEDULES["deferred-a"]) == []
    (why,) = schedule_refusals(HASH_FORMATS["h0"], SCHEDULES["deferred-a"])
    assert "a_tree can't leave the main lane" in why


def test_the_deferred_schedule_is_the_measured_one():
    """The chain at the greatest priority and the side lane at the least, as panel attempt 57's chains ran; the caller's own
    priority is a separate schedule, so a row says which one it timed."""
    assert (DEFERRED.side, DEFERRED.main_priority, DEFERRED.side_priority, DEFERRED.ring) == ({"tile_hash"}, "greatest", "least", 2)
    assert SCHEDULES["deferred-caller"].main_priority == "caller" and SCHEDULES["deferred-caller"].side == DEFERRED.side


def test_a_trees_leave_the_main_lane_only_where_forming_seeds_per_row():
    early = Schedule("h3o", frozenset({"a_tree", "tile_hash"}), ring=2)
    assert schedule_refusals(HASH_FORMATS["h3"], early) == []
    (why,) = schedule_refusals(HASH_FORMATS["h0"], early)
    assert "a_tree can't leave the main lane" in why and "seed_A from A's root" in why


def test_row_seeds_never_leave_the_main_lane():
    """Forming reads each row's digest and seed (h3's per-row keying), so `row_seeds` is pinned under every format."""
    early = Schedule("rs", frozenset({"row_seeds", "tile_hash"}), ring=2)
    for f in HASH_FORMATS.values():
        (why,) = schedule_refusals(f, early)
        assert "row_seeds can't leave the main lane" in why and "each row's digest and seed" in why


def test_a_schedule_is_refused_for_each_thing_wrong_with_it():
    bad = Schedule("bad", frozenset({"gemm", "tile_hash", "hash_everything"}), ring=1, main_priority="high")
    got = schedule_refusals(HASH_FORMATS["h0"], bad)
    assert any("unknown step groups ['hash_everything']" in r for r in got)
    assert any("gemm can't leave the main lane" in r for r in got)
    assert any("needs ring >= 2" in r for r in got)
    assert any("main_priority 'high'" in r for r in got)


LLAMA_8B = ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))


def variant(**kw):
    base = dict(family="pearl-c", device="sm120", schemes=frozenset({"pearl-c-sm120-v1-h3"}),
                schedules=frozenset({"serial", "deferred"}), shapes=ShapeRule(k_multiple=1024, n_multiple=128, m_align=16),
                source_sha256="3f0a91c2d4e5" + "0" * 52, binary_sha256="ab" * 32, loader="cubin-ctypes", prices="cast-8.72",
                steps={"hash_leaf": StepIO("tile_hash", frozenset({"digests", "k_leaf"}), frozenset({"leaves"}))})
    return KernelVariant(**{**base, **kw})


def gate(v, **kw):
    base = dict(variant=v.id, run="r20260930-124211-a304", gates="harness@f5e584af", shapes=LLAMA_8B, exact=True,
                poisoned_accepted=True, controls_rejected=True, schedules_equal=True, sass_ok=True, toolkit="nvcc 12.9 -O3",
                prices="cast-8.72")
    return GateRecord(**{**base, **kw})


def test_a_variant_is_named_by_its_sources_digest():
    v = variant()
    assert v.id == "pearl-c-sm120/3f0a91c2d4e5" and v == variant(steps={}) and hash(v) == hash(variant(steps={}))
    assert variant(source_sha256="9b1e07aa41c0" + "0" * 52).id != v.id


def test_a_gated_variant_serves_its_scheme_and_schedule_at_every_gated_shape():
    v = variant()
    assert variant_refusals(v, "pearl-c-sm120-v1-h3", "deferred", LLAMA_8B, gate(v)) == []


def test_a_variant_is_refused_for_each_thing_it_lacks():
    v = variant()
    got = variant_refusals(v, "pearl-c-sm120-v1", "h3o", [(18944, 3584)], None)
    assert got == ["pearl-c-sm120/3f0a91c2d4e5 computes pearl-c-sm120-v1-h3, not pearl-c-sm120-v1",
                   "pearl-c-sm120/3f0a91c2d4e5 runs deferred, serial, not h3o",
                   "pearl-c-sm120/3f0a91c2d4e5: (n, k) = (18944, 3584): the kernel takes k a multiple of 1024 in [1, 16777216] "
                   "and n a multiple of 128",
                   "pearl-c-sm120/3f0a91c2d4e5 has no device gate"]


def test_a_gate_admits_only_its_own_variant_at_its_own_shapes_and_prices():
    v, other = variant(), variant(source_sha256="9b1e07aa41c0" + "0" * 52)
    got = variant_refusals(v, "pearl-c-sm120-v1-h3", "serial", [*LLAMA_8B, (8192, 8192)],
                           gate(other, controls_rejected=False, prices="cast-32.06"))
    assert got == [f"{v.id}: the gate is {other.id}'s",
                   f"{v.id}: its gate (r20260930-124211-a304) failed controls_rejected",
                   f"{v.id}: its gate didn't cover [(8192, 8192)]",
                   f"{v.id}: gated on priced path cast-32.06, declares cast-8.72"]


def test_a_pass_commitment_is_json_with_hex_bytes():
    c = PassCommitment("pearl-c-sm120-v1-h3", 0, 1, 3, b"\x01" * 32, b"\x02" * 32, b"\x03" * 32, b"\x04" * 32, 128, 4096,
                       schedule="deferred", kernel="pearl-c-sm120/000000000000", controls=("poison",))
    j = c.to_json()
    assert j["salt"] == "01" * 32 and j["records_root"] is None and j["controls"] == ["poison"] and j["tiles"] == 4096
