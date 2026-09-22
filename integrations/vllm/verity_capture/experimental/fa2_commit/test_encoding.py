"""Canonical encoding: positions are a bijection with identities, arithmetic in (T, D, BN); ids fit the frozen grammar."""

from __future__ import annotations

import pytest

from verity_capture.experimental.fa2_commit.encoding import COMMITTED_STAGES, Geometry, RowLayout, ValueId, row_count


@pytest.mark.parametrize("T", [1, 2, 7, 8, 63, 64, 65, 127, 128, 129, 155, 160, 255, 256, 257, 383, 384, 385])
def test_rank_unrank_bijection(T: int) -> None:
    geo = Geometry()
    rl = RowLayout(geo, layer=3, head=5, qrow=T - 1)
    ids = list(rl.ids())
    assert len(ids) == rl.count == row_count(T, geo)
    assert len(set(ids)) == len(ids)
    for i, vid in enumerate(ids):
        assert rl.rank(vid) == i
    stages = {v.stage for v in ids}
    assert stages == set(COMMITTED_STAGES) - ({"ALPHA"} if rl.nb == 1 else set())
    # every visible key column has exactly one S and one P32; every block has M, 4 lanes, D accumulators
    assert sum(1 for v in ids if v.stage == "S") == T == sum(1 for v in ids if v.stage == "P32")
    assert sum(1 for v in ids if v.stage == "O") == rl.nb * geo.D
    assert sum(1 for v in ids if v.stage == "ALPHA") == rl.nb - 1
    # visit order: version 0 is the LAST block
    first = ids[0]
    assert first.stage == "S" and first.block == rl.nb - 1 and first.version == 0 and first.index == (rl.nb - 1) * geo.BN


def test_rank_rejects_foreign_and_malformed_ids() -> None:
    rl = RowLayout(Geometry(), 0, 0, 200)
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 1, 200, "S", 0, 0, 1))          # other head
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 0, 200, "S", 0, 0, 0))          # wrong version for block 0 (nb = 2 -> version 1)
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 0, 200, "S", 1, 60, 0))         # column 60 is not in block 1
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 0, 200, "ALPHA", 1, 0, 0))      # first visited block has no rescale
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 0, 200, "P16", 0, 3, 1))        # derived stage: no committed position
    with pytest.raises(KeyError):
        rl.rank(ValueId(0, 0, 200, "OUT", 0, 3, 1))        # not a stage


def test_canonical_id_fits_production_grammar() -> None:
    from verity_vllm import identity

    vid = ValueId(12, 7, 154, "P32", 1, 130, 0)
    cid = vid.canonical_id("a594a58ec69b", "B0-cov_b0_b1_c256", "prefill")
    parsed = identity.parse(cid)
    assert parsed.kind == "value" and parsed.t == 154 and parsed.layer == "12"
    assert parsed.coordinate == "7,1,130,0" and parsed.op == "self_attn.attn.fa2.P32"
