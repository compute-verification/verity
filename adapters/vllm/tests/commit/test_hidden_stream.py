"""The M1 stream's layout (`hidden_stream.StreamLayout`): a position's word, and the position of a word."""

from __future__ import annotations

import pytest

from verity_vllm.commit import hidden_stream as hs


def _layout() -> hs.StreamLayout:
    return hs.StreamLayout(HB=2, M=64, NB=2)


def test_layout_word_index_roundtrip():
    L = _layout()
    for cls, kw in [("S", dict(col=200)), ("P", dict(col=5)), ("RP", dict(col=130)), ("ROW", dict(block=1, lane=3)),
                    ("ROW", dict(block=0, field=1)), ("O", dict(block=1, d=63)), ("FIN", dict(field=3))]:
        w = L.word_index(cls, 1, 17, **kw)
        u = L.unrank(w)
        assert u["cls"] == cls and u["slab"] == 1 and u["row"] == 17
    assert L.total_words % 64 == 0
    assert L.unrank(L.total_words - 1)["cls"] == "FIN"


def test_word_index_is_each_classes_arithmetic():
    """Each class's word written out: the class's offset, the words a (slab, row) takes in it, the position's word among
    them; on a tile that is not the default one, with the seventh class."""
    L = hs.StreamLayout(HB=2, M=64, NB=2, BN=64, D=128, ms=True)
    o, sr = L.offsets, 1 * 64 + 17
    assert L.N == 128
    want = [("S", dict(col=100), o["S"] + sr * 128 + 100, dict(col=100)),
            ("P", dict(col=5), o["P"] + sr * 128 + 5, dict(col=5)),
            ("RP", dict(col=101), o["RP"] + sr * 64 + 50, dict(col=100)),
            ("ROW", dict(block=1, lane=3), o["ROW"] + (sr * 2 + 1) * 8 + 7, dict(block=1, field=7)),
            ("ROW", dict(block=0, field=hs.ROW_GUARD_FIELD), o["ROW"] + sr * 2 * 8 + 3, dict(block=0, field=3)),
            ("O", dict(block=1, d=127), o["O"] + (sr * 2 + 1) * 128 + 127, dict(block=1, d=127)),
            ("FIN", dict(field=2), o["FIN"] + sr * 4 + 2, dict(field=2)),
            ("MS", dict(block=1), o["MS"] + sr * 2 + 1, dict(block=1))]
    for cls, kw, word, back in want:
        assert L.word_index(cls, 1, 17, **kw) == word, cls
        assert L.unrank(word) == {"cls": cls, "slab": 1, "row": 17, **back}, cls
    assert L.word_index("MS", 1, 63, block=1) == L.total_words - 1


def test_word_index_refuses_what_the_layout_does_not_hold():
    L = _layout()
    for cls in ("MS", "total", "Q", ""):                 # MS: this layout was built without it
        with pytest.raises(ValueError) as e:
            L.word_index(cls, 0, 0, col=0, block=0, d=0, field=0)
        assert e.value.args == (cls,)
    for slab, row in ((2, 0), (0, 64), (-1, 0), (0, -1)):
        with pytest.raises(ValueError, match="position outside the layout"):
            L.word_index("S", slab, row, col=0)
    for cls, kw in [("S", dict(col=256)), ("P", dict()), ("RP", dict(col=-1)), ("ROW", dict(block=2, lane=0)),
                    ("ROW", dict(block=0)), ("O", dict(block=0)), ("O", dict(block=0, d=64)), ("FIN", dict())]:
        with pytest.raises(AssertionError):
            L.word_index(cls, 0, 0, **kw)
