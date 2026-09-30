"""`verity_pouw.serving`: every Pearl-C name in use parses and prints back unchanged, the hashing format is a switch whose
keys are the scheme's, and a schedule is refused where its format needs A's tree on the critical path."""
from __future__ import annotations

import pytest
from verity.commitments.blake3 import blake3
from verity_pouw.serving import (
    DEFERRED,
    HASH_FORMATS,
    SCHEDULES,
    SERIAL,
    PassCommitment,
    Schedule,
    SchemeId,
    hash_format,
    schedule_refusals,
)

#: every Pearl-C name registered or recorded on the open PRs (#449, #540, #548, #549, #555) and the ones their panel lines name
NAMES_IN_USE = ("pearl-c-h100-v0", "pearl-c-h100-v1", "pearl-c-h100-v1-h1", "pearl-c-h100-v1-h2", "pearl-c-h100-v1-h3",
                "pearl-c-h100-v1@sm_120", "pearl-c-sm120-v1", "pearl-c-sm120-v1-h1", "pearl-c-sm120-v2",
                "pearl-c-sm120-v2-hot", "pearl-c-sm120-v2-hot-h3", "pearl-c-nvfp4-v0", "pearl-c-mxfp4-v0")


@pytest.mark.parametrize("name", NAMES_IN_USE)
def test_every_name_in_use_parses_and_prints_back(name):
    assert SchemeId.parse(name).name == name


def test_the_parts_are_the_names_parts():
    s = SchemeId.parse("pearl-c-sm120-v2-hot-h3")
    assert (s.family, s.device, s.forming, s.precision, s.hashing, s.prices) == ("pearl-c", "sm120", "v2-hot", "e4m3", "h3", None)
    c4 = SchemeId.parse("pearl-c-nvfp4-v0")
    assert (c4.family, c4.device, c4.precision, c4.hashing) == ("pearl-c4", "sm120", "nvfp4", "h0")
    assert SchemeId.parse("pearl-c-h100-v1@sm_120").prices == "sm_120"


def test_the_hashing_format_is_a_switch():
    h1 = SchemeId.parse("pearl-c-sm120-v1-h1")
    assert h1.with_hashing("h2").name == "pearl-c-sm120-v1-h2"
    assert h1.with_hashing("h0").name == "pearl-c-sm120-v1"
    assert SchemeId.parse("pearl-c-nvfp4-v0").with_hashing("h1").name == "pearl-c-nvfp4-v0-h1"


@pytest.mark.parametrize("name, why", [
    ("pearl-c-a100-v1-h1", "device 'a100'"),
    ("pearl-c-sm120-v9-h1", "forming 'v9'"),
    ("pearl-c-sm120-v1-h7", "hashing 'h7'"),
    ("pearl-fp8-v4", "not a Pearl-C scheme name"),
    ("ncp-v2", "not a Pearl-C scheme name"),
])
def test_a_name_outside_the_grammar_is_refused_by_what_is_wrong(name, why):
    with pytest.raises(ValueError, match=why):
        SchemeId.parse(name)


def test_the_formats_keys_are_blake3_of_their_labels_and_h0_has_none():
    """`pearl_c.H1_KEYS` is ``blake3(f"pearl-c/h1/{role}")``; h2 and h3 keep h1's digests and leaf (#532, #549)."""
    for h in ("h1", "h2", "h3"):
        f = HASH_FORMATS[h]
        assert f.message_key() == blake3(b"pearl-c/h1/message") and f.leaf_key() == blake3(b"pearl-c/h1/leaf")
    assert HASH_FORMATS["h0"].message_key() is None and HASH_FORMATS["h0"].leaf_key() is None
    assert [HASH_FORMATS[h].tree_hash for h in ("h0", "h1", "h2", "h3")] == ["sha256", "blake3", "blake3-s256", "blake3-s256"]


def test_hash_format_reads_a_name_an_id_or_a_scheme_object():
    class Scheme:
        hashing = "h3"
    assert hash_format("pearl-c-sm120-v1-h1") is HASH_FORMATS["h1"]
    assert hash_format(SchemeId.parse("pearl-c-h100-v1")) is HASH_FORMATS["h0"]
    assert hash_format(Scheme()) is HASH_FORMATS["h3"]
    with pytest.raises(ValueError, match="no hashing format"):
        hash_format(object())


def test_the_serial_and_deferred_schedules_run_under_every_format():
    assert SCHEDULES == {"serial": SERIAL, "deferred": DEFERRED}
    for f in HASH_FORMATS.values():
        assert schedule_refusals(f, SERIAL) == [] and schedule_refusals(f, DEFERRED) == []


def test_a_trees_leave_the_main_lane_only_where_forming_seeds_per_row():
    early = Schedule("h3o", frozenset({"a_tree", "tile_hash"}), ring=2)
    assert schedule_refusals(HASH_FORMATS["h3"], early) == []
    for h in ("h1", "h2"):
        (why,) = schedule_refusals(HASH_FORMATS[h], early)
        assert "a_tree can't leave the main lane" in why and "seed_A from A's root" in why


def test_a_schedule_is_refused_for_each_thing_wrong_with_it():
    bad = Schedule("bad", frozenset({"gemm", "tile_hash", "hash_everything"}), ring=1, main_priority="high")
    got = schedule_refusals(HASH_FORMATS["h1"], bad)
    assert any("unknown step groups ['hash_everything']" in r for r in got)
    assert any("gemm can't leave the main lane" in r for r in got)
    assert any("needs ring >= 2" in r for r in got)
    assert any("main_priority 'high'" in r for r in got)


def test_a_pass_commitment_is_json_with_hex_bytes():
    c = PassCommitment("pearl-c-sm120-v1-h1", 0, 1, 3, b"\x01" * 32, b"\x02" * 32, b"\x03" * 32, b"\x04" * 32, 128, 4096,
                       schedule="deferred", kernel="pearl-c-sm120/000000000000", controls=("poison",))
    j = c.to_json()
    assert j["salt"] == "01" * 32 and j["records_root"] is None and j["controls"] == ["poison"] and j["tiles"] == 4096
