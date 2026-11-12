"""The Flock backend's pinned lowering and its support declaration."""
from __future__ import annotations

import pytest
from verity.definitions.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120

from verity_flock import lowering
from verity_flock.backend import FLOCK_FRAME_V3_BLAKE3 as FLOCK, GemmSub, Supported, Unavailable, Unsupported


@pytest.mark.parametrize("relation", sorted(lowering.PIPES))
def test_lowering_is_the_pinned_one(relation):
    assert lowering.digest(relation) == lowering.PINS[relation]


@pytest.mark.parametrize("relation", ["bf16-hopper", "fp8-ada"])
@pytest.mark.slow
def test_lowered_unit_matches_the_reference(relation):
    assert lowering.self_check(relation, n=40) == 0


def test_non_finite_accumulator_leaves_a_row_unsatisfied():
    pipe = lowering.PIPES["bf16-hopper"]
    x = [0x3F80] * pipe.k
    _, _, ok = lowering.evaluate("bf16-hopper", x, x, 0x7F800000)
    assert not ok
    _, _, ok = lowering.evaluate("bf16-hopper", x, x, 0x3F800000)
    assert ok


def test_io_words_are_aligned_and_zero_padded():
    head = lowering.netlist("bf16-hopper").split("\n", 1)[0].split()
    assert head[0] == lowering.FORMAT and head[6:11] == ["0", "1", "2", "3", "4"]
    ins, outs = int(head[5]), [int(c) for c in head[12:12 + int(head[11])]]
    assert ins == 5 and len(outs) == 2 and int(head[2]) <= 1 << lowering.K_LOG


def test_supports():
    for t in (BF16_HOPPER, FP8_ADA_MMA, FP8_HOPPER_WGMMA):
        assert isinstance(FLOCK.supports(GemmSub(t)), (Supported, Unavailable))
    assert isinstance(FLOCK.supports(GemmSub(FIRST)), (Supported, Unavailable))
    assert isinstance(FLOCK.supports(GemmSub(NVFP4_SM120)), Unsupported)
    assert isinstance(FLOCK.supports(GemmSub(BF16_HOPPER, 1000)), Unsupported)
    assert isinstance(FLOCK.supports(GemmSub(FP8_ADA_MMA, 512)), Unsupported)


def test_lower_refuses_an_unpinned_lowering(monkeypatch):
    if isinstance(FLOCK.supports(GemmSub(BF16_HOPPER)), Unavailable):
        monkeypatch.setattr("shutil.which", lambda _: "/bin/true")
    cs = FLOCK.lower(GemmSub(BF16_HOPPER))
    assert cs["sha256"] == lowering.PINS["bf16-hopper"] and cs["units"] == 96 and cs["row_bytes"] == 3072
    monkeypatch.setitem(lowering.PINS, "bf16-hopper", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        FLOCK.lower(GemmSub(BF16_HOPPER))


def test_record_names_only_standard_hashes():
    rec = FLOCK.record()
    assert {h["class"] for h in rec["hashes"]} == {"standard"}
    assert rec["zk_class"] == "NON_ZK_PROOF"


@pytest.mark.parametrize("relation", ["bf16-ampere", "fp8-ada"])
def test_a_spine_input_set_is_written_on_its_chain_and_refused_when_its_y_differs(tmp_path, relation):
    """`instance_file.write_set` re-chains each row under the relation's model and holds the set's recorded y to it: the BF16 word
    of an epilogue relation, else the FP32 accumulator, whose committed word drops its zero low bits (`instance_file.Y_WORD`)."""
    import json

    import numpy as np
    from verity.definitions.silicon.cast import f32_to_bf16_word

    from verity_flock import instance_file

    pipe, rng = lowering.PIPES[relation], np.random.default_rng(0)
    K, n = 2 * pipe.k, 4
    if pipe.word_bits == 16:
        def words():
            return (rng.integers(0, 2, (n, K)) << 15 | rng.integers(112, 136, (n, K)) << 7 | rng.integers(0, 128, (n, K))).astype("<u2")
        rows, y_file = ("u16", "x.u16", "w.u16"), "y.u16"
    else:
        def words():
            return (rng.integers(0, 2, (n, K)) << 7 | rng.integers(0x20, 0x50, (n, K))).astype("u1")
        rows, y_file = ("u8", "x.u8", "w.u8"), "y.u32"
    x, w = words(), words()
    accs = instance_file._chain_rows((relation, x, w))
    y = np.asarray([f32_to_bf16_word(int(a[-1])) if pipe.epilogue else int(a[-1]) for a in accs],
                   dtype="<u2" if pipe.epilogue else "<u4")
    d = tmp_path / "set"
    d.mkdir()
    x.tofile(d / rows[1])
    w.tofile(d / rows[2])
    y.tofile(d / y_file)
    ports = {"x": {"file": rows[1], "dtype": rows[0], "words": K}, "w": {"file": rows[2], "dtype": rows[0]},
             "y": {"file": y_file, "dtype": y_file.split(".")[1]}}
    (d / "manifest.json").write_text(json.dumps({"set": "t", "content_digest": "00" * 32, "ports": ports}))
    h = instance_file.write_set(tmp_path / "i.bin", relation, d, 0, n)
    assert (h["format"], h["vus"], h["k"], h["units"], h["epilogue"]) == (instance_file.FORMAT, n, K, 2, pipe.epilogue)
    assert h["schemas"]["y"] == instance_file.word_schema(16 if pipe.epilogue else instance_file.Y_WORD[relation][0])[0]
    y[0] ^= 1 << (0 if pipe.epilogue else 10)
    y.tofile(d / y_file)
    with pytest.raises(ValueError, match=f"1 recorded outputs differ from the {relation} model chain"):
        instance_file.write_set(tmp_path / "j.bin", relation, d, 0, n)
