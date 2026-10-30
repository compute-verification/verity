"""FA3's forward tile (`fa3_model.kblock_n_for`, `kblock_m_for`): each branch of `tile_size_fwd_sm90` under `use_one_mma_wg`
(one MMA warpgroup only at head dims 64 and 128, while seqlen_q * qheads_per_kvhead <= 64), and a head dim with no rule refused;
`modelled_tile`, the tile the split plan reads, only at the head dims the model covers."""
from __future__ import annotations

import pytest

from verity.ml.definitions.numerics import fa3_model as FA3

#: (D, seqlen_q, qheads_per_kvhead) -> (kBlockM, kBlockN)
TILES = {
    (64, 16, 4): (64, 192),
    (64, 17, 4): (192, 128),
    (32, 1, 1): (192, 128),
    (96, 1, 1): (192, 128),
    (128, 64, 1): (64, 128),
    (128, 65, 1): (128, 128),
    (256, 1, 1): (128, 80),
    (256, 4096, 4): (128, 80),
}


@pytest.mark.parametrize(("shape", "tile"), sorted(TILES.items()))
def test_each_branch_of_the_tile_rule(shape, tile):
    assert (FA3.kblock_m_for(*shape), FA3.kblock_n_for(*shape)) == tile


@pytest.mark.parametrize("D", [192, 512])
def test_a_head_dim_with_no_rule_is_refused(D):
    with pytest.raises(ValueError, match=f"head dim {D}: no kBlockN rule"):
        FA3.kblock_n_for(D, 512, 1)
    with pytest.raises(ValueError, match=f"head dim {D}: no kBlockM rule"):
        FA3.kblock_m_for(D, 512, 1)


@pytest.mark.parametrize(("shape", "tile"), sorted(TILES.items()))
def test_the_modelled_tile_is_the_rule_s_at_a_modelled_head_dim_only(shape, tile):
    if shape[0] in FA3.SUPPORTED_HEAD_DIMS:
        assert FA3.modelled_tile(*shape) == tile
    else:
        with pytest.raises(ValueError, match=f"head dim {shape[0]}: not modelled"):
            FA3.modelled_tile(*shape)


def test_a_split_plan_at_a_head_dim_the_model_does_not_cover_is_refused():
    with pytest.raises(ValueError, match="head dim 32: not modelled"):
        FA3.split_plan([2], 8, 8, 32, num_sms=132)
