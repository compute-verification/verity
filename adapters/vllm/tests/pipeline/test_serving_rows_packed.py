"""A packed row template (`pipeline.serving_rows.Template.packed`): Fp8GroupQuantUe8m0_v1{K=256,G=128}, a bf16 row in and its
w8 codes and two f32 scales out, read off one Program row through a committed store.  Its member counts each port's u16 words
by the template's element widths, and its rows are the elements' little-endian bytes, two to a word: an 8-bit element is half a
word (even element low), a 32-bit one two words (low half first), which is the bit order of M0's rows.  An element read at
another width than the template's port is refused, as is a row whose elements are not the ports' whole words, except a signed
integer member wider than its port: `TokenSelect_v1`'s i32 token, which the sampler commits as torch's int64, is read by value."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.pipeline import serving_rows as SRW

T = "Fp8GroupQuantUe8m0_v1{K=256,G=128}"
OP = "model.layers.2.mlp.experts.quant"


class Index:
    """The ProgramIndex calls `_slices` makes: the row's one argument is member ``x`` of the producing row."""

    def member_at(self, a, lo, hi):
        return {"members": [["x", 256, "bf16"]], "step": 3, "op_path": "model.layers.2.mlp.experts.act", "row": 0}, "x", lo, hi, "bf16"


class Store:
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values

    def words(self, rid, step, op_path, member, elo, ehi):
        return self.values[member][elo:ehi], str(self.values[member].dtype)


def _row(members=(("q", 256, "w8"), ("s", 2, "f32"))):
    r = {"i": 7, "spec": T, "args": [[["n", 0, 0, 256]]]}
    d = {"members": [list(m) for m in members], "step": 3, "op_path": OP, "row": 0}
    return ("r0", Index(), r, d)


def test_a_packed_row_templates_ports_are_its_elements_little_endian_bytes():
    assert SRW.template(T).packed == (16, 8, 32)
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 16, 256, dtype=np.uint16)
    q = rng.integers(0, 1 << 8, 256, dtype=np.uint8)
    s = rng.integers(0, 1 << 32, 2, dtype=np.uint32)
    (m,) = SRW.members([T], [[_row()]])
    assert (m.rows, m.outs, m.n) == ((("in0", 256),), (("out.q", 128), ("out.s", 4)), 1)

    data, entries = SRW.read(m, [_row()], Store({"x": x, "q": q, "s": s}), lambda *a: None)
    assert np.array_equal(data["rows"]["in0"], x.reshape(1, -1))
    want_q = q[0::2].astype(np.uint16) | (q[1::2].astype(np.uint16) << 8)
    want_s = np.array([s[0] & 0xFFFF, s[0] >> 16, s[1] & 0xFFFF, s[1] >> 16], dtype=np.uint16)
    assert np.array_equal(data["outs"]["out.q"], want_q.reshape(1, -1))
    assert np.array_equal(data["outs"]["out.s"], want_s.reshape(1, -1))
    words = data["outs"]["out.q"].reshape(-1).astype("<u2")
    assert np.array_equal(np.unpackbits(words.view(np.uint8), bitorder="little"), np.unpackbits(q, bitorder="little"))
    assert entries == [[0, 1, "r0", 3, 2, OP, "q,s", 0, 7]]


def test_a_packed_port_read_at_another_width_or_not_whole_words_is_refused():
    x = np.zeros(256, dtype=np.uint16)
    (m,) = SRW.members([T], [[_row()]])
    with pytest.raises(ValueError, match="output q: uint16 elements, the template's port is 8-bit elements"):
        SRW.read(m, [_row()], Store({"x": x, "q": np.zeros(256, np.uint16), "s": np.zeros(2, np.uint32)}), lambda *a: None)
    with pytest.raises(ValueError, match=r"committed members of \[3, 2\] elements are not its ports' whole u16 words"):
        SRW.members([T], [[_row((("q", 3, "w8"), ("s", 2, "f32")))]])


def _sampled_row(dtype: str = "i64"):
    r = {"i": 9, "spec": "TokenSelect_v1{V=16}", "args": [[["n", 0, 0, 16]]]}
    d = {"members": [["sampled_token_ids", 1, dtype]], "step": 3, "op_path": "sampler", "row": 0}
    return ("r0", Index(), r, d)


def test_the_sampled_i64_token_is_read_by_value_into_token_selects_i32_port():
    """`TokenSelect_v1` returns an i32 and the sampler commits torch's int64 token, which the store reads as its u64 word: the
    port's two words are the token's value, low half first, as the replay links tokens (`check.replay.linkage`)."""
    T16 = "TokenSelect_v1{V=16}"
    assert SRW.template(T16).packed == (16, 32)
    x = np.arange(16, dtype=np.uint16)
    (m,) = SRW.members([T16], [[_sampled_row()]])
    assert (m.rows, m.outs) == ((("in0", 16),), (("out.sampled_token_ids", 2),))
    data, _ = SRW.read(m, [_sampled_row()], Store({"x": x, "sampled_token_ids": np.array([0x2_5DC3], np.uint64)}), lambda *a: None)
    assert data["outs"]["out.sampled_token_ids"].tolist() == [[0x5DC3, 0x2]]


def test_a_token_outside_the_port_or_an_unsigned_member_wider_than_it_is_refused():
    x = np.zeros(16, dtype=np.uint16)
    (m,) = SRW.members(["TokenSelect_v1{V=16}"], [[_sampled_row()]])
    with pytest.raises(ValueError, match=r"output sampled_token_ids: element 0 of a uint64 member is -1, not in a 32-bit port's"):
        SRW.read(m, [_sampled_row()], Store({"x": x, "sampled_token_ids": np.array([2**64 - 1], np.uint64)}), lambda *a: None)
    with pytest.raises(ValueError, match="output sampled_token_ids: uint64 elements, the template's port is 32-bit elements"):
        SRW.read(m, [_sampled_row("u64")], Store({"x": x, "sampled_token_ids": np.array([7], np.uint64)}), lambda *a: None)
