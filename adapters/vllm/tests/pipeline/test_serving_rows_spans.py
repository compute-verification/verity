"""A part's ports may take a span of the row's argument group or committed member (`pipeline.serving_templates`' ``PARTS``:
a read ``("arg", g, lo, hi)`` or ``("out", j, lo, hi)``, a return ``(j, lo, hi)``, each bound an element index or a row
static, ``hi`` None the end), and a ``COLS`` part's blocks are each port's own width, of any element width.  The BF16 norms'
cut written with spans over whole groups and members is the cut as it is: the same members, rows, index entries, roots and
leaf layer, byte for byte.  The parts of a row's cut return each element of each of its outputs once: a cut whose spans
leave one unreturned or return one twice is refused by name, as is a port a ``COLS`` part's blocks would cut inside an
element or a u16 word."""
from __future__ import annotations

import numpy as np
import pytest

from tests.pipeline.test_serving_rows_norm_split import CHUNK_APPLY, CHUNK_SUMSQ, N, served, templates, wired  # noqa: F401
from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import part_rows as PR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST

#: the norms' cuts as `serving_templates.PARTS` has them, every port of a group or member spanned whole: by a row static, by
#: the end (None) or by its width
SPANNED = {
    "DsaRMSNorm_v1": (("DsaRMSNormSumSq_v1", (("arg", 0, 0, "N"),), ("r",)),
                      ("DsaRMSNormApply_v1", (("arg", 0, 0, "N"), ("arg", 1, 0, None), ("mid", "r")), ((0, 0, "N"),))),
    "RMSNormFusedCudaVector8_v3": (
        ("RMSNormFusedSumSqCudaVector8_v1", (("arg", 0, 0, "N"), ("arg", 1)), ("scale",)),
        ("RMSNormFusedApplyCudaVector8_v1", (("arg", 0), ("arg", 1, 0, N), ("arg", 2, 0, None), ("mid", "scale")),
         ((0, 0, "N"), (1, 0, None)))),
}


def cut(monkeypatch, family: str, entry: tuple) -> None:
    """Serve ``family``'s rows by the cut ``entry`` in place of its ``PARTS`` entry."""
    monkeypatch.setitem(ST.PARTS, family, entry)
    for p in entry:
        monkeypatch.setitem(ST.PART_OF[p[0]], family, p)


def commitment(x: dict, tmpls: list[str]) -> tuple:
    """The window `serving_rows.population` derives over ``x``'s rows, read and committed under a fixed salt key: its members,
    each member's rows, outputs and index entries, the roots and the leaf layer."""
    caps = {tmpls[0]: CHUNK_SUMSQ, tmpls[1]: CHUNK_APPLY}
    w, rows = SRW.population({"r0": {}}, {"templates": tmpls, "law": "subset:2", "chunk_cap": caps}, "ab" * 64)
    got = SRW.read_all(w.members, rows, x["store"], None)
    c = CS.commit(w, [g for g, _e in got], salt_key=bytes(32))
    data = [({p: a.tobytes() for p, a in g["rows"].items()}, {p: a.tobytes() for p, a in g["outs"].items()}, e) for g, e in got]
    return w.members, data, {p: r.hex() for p, r in c.roots.items()}, c.leaf_layer


@pytest.mark.parametrize("family", list(SPANNED))
def test_a_cut_written_with_whole_spans_commits_what_the_cut_commits(served, monkeypatch, family):
    """The control: the norm's cut and the same cut with every port spanned whole give one window, byte for byte."""
    x = served(family, ops=(None, None))
    tmpls = list(templates(family))
    today = commitment(x, tmpls)
    cut(monkeypatch, family, SPANNED[family])
    assert SRW.part(SRW.template(tmpls[1]).word, x["spec"]) == SPANNED[family][1]
    spanned = commitment(x, tmpls)
    assert spanned[0] == today[0]
    assert spanned[1] == today[1]
    assert spanned[2] == today[2] and spanned[3] == today[3]


def test_a_cut_whose_spans_leave_an_output_s_elements_unreturned_is_refused_by_name(served, monkeypatch):
    x = served("DsaRMSNorm_v1")
    sumsq, (fam, reads, _ret) = SPANNED["DsaRMSNorm_v1"]
    cut(monkeypatch, "DsaRMSNorm_v1", (sumsq, (fam, reads, ((0, 0, N // 2),))))
    TS, TA_ = templates("DsaRMSNorm_v1")
    with pytest.raises(ValueError, match=rf"row 7 \(.*\): DsaRMSNorm_v1's output out: elements \[{N // 2}, {N}\) are returned "
                                         r"by no part of its cut"):
        ms, rows = wired([TS, TA_], {TS: CHUNK_SUMSQ, TA_: CHUNK_APPLY})
        SRW.read_all(ms, rows, x["store"], None)


def test_a_cut_whose_spans_return_an_output_s_elements_twice_is_refused_by_name(served, monkeypatch):
    x = served("RMSNormFusedCudaVector8_v3")
    sumsq, (fam, reads, _ret) = SPANNED["RMSNormFusedCudaVector8_v3"]
    cut(monkeypatch, "RMSNormFusedCudaVector8_v3", (sumsq, (fam, reads, ((0, 0, "N"), (0, 8, 16)))))
    TS, TA_ = templates("RMSNormFusedCudaVector8_v3")
    with pytest.raises(ValueError, match=r"row 7 \(.*\): RMSNormFusedCudaVector8_v3's output out: elements \[8, 16\) are returned "
                                         r"by more than one part of its cut \(RMSNormFusedApplyCudaVector8_v1 and "
                                         r"RMSNormFusedApplyCudaVector8_v1\)"):
        ms, rows = wired([TS, TA_], {TS: CHUNK_SUMSQ, TA_: CHUNK_APPLY})
        SRW.read_all(ms, rows, x["store"], None)


def test_a_cols_port_its_blocks_would_cut_inside_an_element_is_refused_by_name(served, monkeypatch):
    x = served("DsaRMSNorm_v1")
    sumsq, (fam, _reads, ret) = SPANNED["DsaRMSNorm_v1"]
    cut(monkeypatch, "DsaRMSNorm_v1", (sumsq, (fam, (("arg", 0, 0, "N"), ("arg", 1, 1, None), ("mid", "r")), ret)))
    TS, TA_ = templates("DsaRMSNorm_v1")
    ms, rows = wired([TS, TA_], {TS: CHUNK_SUMSQ, TA_: CHUNK_APPLY})
    with pytest.raises(ValueError, match=rf"row 7 \(.*\): its arg 1 is {N - 1} 16-bit elements, which its {N // 64} blocks would "
                                         r"cut inside an element"):
        SRW.read_all(ms, rows, x["store"], None)


def test_a_cols_port_is_cut_into_its_own_width_of_any_element_width():
    a = np.arange(16, dtype=np.uint16)
    assert PR.blocks(a, 16, 4, "x").tolist() == a.reshape(4, 4).tolist()
    assert PR.blocks(a, 32, 8, "x").tolist() == a.reshape(8, 2).tolist()
    assert PR.blocks(a, 8, 16, "x").tolist() == a.reshape(16, 1).tolist()
    with pytest.raises(ValueError, match=r"x is 16 32-bit elements, which its 32 blocks would cut inside an element"):
        PR.blocks(np.arange(32, dtype=np.uint16), 32, 32, "x")
    with pytest.raises(ValueError, match=r"x is 32 8-bit elements, which its 32 blocks would cut inside a u16 word"):
        PR.blocks(a, 8, 32, "x")


def test_a_span_is_read_off_the_row_s_statics():
    assert ST.bounds(None, {}, 8, "x") == (0, 8)
    assert ST.bounds((0, "N"), {"N": 6}, 8, "x") == (0, 6)
    assert ST.bounds(("N", None), {"N": 6}, 8, "x") == (6, 8)
    assert ST.bounds((2, 3), {}, 8, "x") == (2, 3)
    for span in ((0, 9), (4, 4), ("N", 2)):
        with pytest.raises(ValueError, match=r"x: span .* is \[.*\), not a run inside its 8 elements"):
            ST.bounds(span, {"N": 6}, 8, "x")


def test_a_spanned_port_reads_those_elements_of_the_row_s_group_or_member(served):
    x = served("DsaRMSNorm_v1")
    rows, _defs, _counts = SRW.member_rows({"r0": {}}, list(templates("DsaRMSNorm_v1")))
    rid, pi, r, d = rows[0][0]
    arg = PR.operand(pi, r, d, "arg", 1, 16, 1, x["store"], None, rid, 0, SRW._slices, SRW._packed, {}, (3, 5))
    assert arg.tolist() == x["args"][71][3:5].tolist()
    out = PR.operand(pi, r, d, "out", 0, 16, 0, x["store"], None, rid, 0, SRW._slices, SRW._packed, {}, (N - 4, None))
    assert out.tolist() == x["words"]["model.layers.6.self_attn.kv_a_layernorm"]["out"][N - 4:].tolist()
