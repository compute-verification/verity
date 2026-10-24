"""A packed row template (`pipeline.serving_rows.Template.packed`): Fp8GroupQuantUe8m0_v1{K=256,G=128}, a bf16 row in and its
w8 codes and two f32 scales out, read off one Program row through a committed store.  Its member counts each port's u16 words
by the template's element widths, and its rows are the elements' little-endian bytes, two to a word: an 8-bit element is half a
word (even element low), a 32-bit one two words (low half first), which is the bit order of M0's rows.  An element read at
another width than the template's port is refused, as is a row whose elements are not the ports' whole words."""
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
