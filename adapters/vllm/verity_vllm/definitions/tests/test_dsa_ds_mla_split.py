"""`DsaLatentRowDsMla_v1` cut at its scale, the one f32 word each of its products reads: the scale part `DsaRMSNormSumSq_v1{KVL,
EPS}` reading kv's first KVL words, tile part j `DsaLatentRowDsMlaTile_v1{TILE}` reading those words [j·TILE, (j+1)·TILE), their
weights and the scale, and the rotary part `RoPEHeadGptJ_v2{R}` reading kv's last R words and cs return the row's words on
`test_rmsnorm_fused_add`'s rows (Gaussian rows at three scales, uniformly random words with NaN, ±inf, ±0 and subnormals among
them, and rows whose squares or variance overflow, underflow or cancel). Output 0 is each tile's kv_c then the rotary words,
output 1 each tile's bytes, output 2 each tile's scale. Their gates are the row's, the tile part's Boolean version returns its
words, and at GLM-5.3's widths a tile part fits a Flock proof block."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation import evaluate
from verity.circuits.ops.boolean.versions import boolean_version
from verity_vllm.definitions.registry import dsa_ds_mla as M
from verity_vllm.definitions.registry import dsa_norm as DN
from verity_vllm.definitions.registry.boolean_rope import RoPEHeadGptJV2
from verity.commitments.periods import interface as IF

_spec = importlib.util.spec_from_file_location("rows_of_test_rmsnorm_fused_add", Path(__file__).with_name("test_rmsnorm_fused_add.py"))
_rows_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rows_test)
_rows, _flat, _gates, _gauss = _rows_test._rows, _rows_test._flat, _rows_test._gates, _rows_test._gauss

#: (KVL, R, TILE): the least row (256 latent words in four tiles of 64, 8 rotary values) and GLM-5.3's kv_a row
CASES = [(256, 8, 64), (512, 64, 128)]


def _parts(KVL: int, R: int, TILE: int) -> tuple[object, object, object, object]:
    return (bind(M.DsaLatentRowDsMla, KVL=KVL, R=R, TILE=TILE, EPS=1e-5), bind(DN.DsaRMSNormSumSq, N=KVL, EPS=1e-5),
            bind(M.DsaLatentRowDsMlaTile, TILE=TILE), bind(RoPEHeadGptJV2, D=R))


def _words(a) -> list[int]:
    return [int(v) for v in a]


def _split(sumsq, tile, rope, KVL: int, TILE: int, kv: np.ndarray, w: np.ndarray, cs: np.ndarray) -> list[int]:
    """The scale part's scale, then the tile parts' and the rotary part's outputs joined in the row's order."""
    (r,) = _flat(evaluate(sumsq, _words(kv[:KVL])))
    tiles = [_flat(evaluate(tile, _words(kv[j * TILE:(j + 1) * TILE]), _words(w[j * TILE:(j + 1) * TILE]), r))
             for j in range(KVL // TILE)]
    kpe = _flat(evaluate(rope, _words(kv[KVL:]), _words(cs)))
    return [*(v for t in tiles for v in t[:TILE]), *kpe, *(v for t in tiles for v in t[TILE:2 * TILE]), *(t[2 * TILE] for t in tiles)]


@pytest.mark.parametrize("KVL,R,TILE", CASES)
def test_the_scale_tile_and_rotary_parts_are_the_row(KVL, R, TILE):
    row, sumsq, tile, rope = _parts(KVL, R, TILE)
    rng = np.random.default_rng(KVL + R + TILE)
    for kind, x, _, w in _rows(rng, KVL):
        kv, cs = np.concatenate([x, _gauss(rng, R, 1.0)]), _gauss(rng, R, 1.0)
        want = _flat(evaluate(row, _words(kv), _words(w), _words(cs)))
        got = _split(sumsq, tile, rope, KVL, TILE, kv, w, cs)
        assert len(got) == len(want) and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])


@pytest.mark.parametrize("KVL,R,TILE", CASES)
def test_their_gates_are_the_rows(KVL, R, TILE):
    """The row Calls the scale and the rotary part's Definitions once and takes each tile's products, roundings and quantisation
    as a tile part does; nothing is recomputed."""
    row, sumsq, tile, rope = _parts(KVL, R, TILE)
    split = Counter({sumsq.id: 1, rope.id: 1}) + Counter({gate: n * (KVL // TILE) for gate, n in _gates(tile).items()})
    assert split == _gates(row)


def test_on_bits_the_tile_part_is_its_boolean_version_and_returns_its_words():
    """The switch takes the word tile part to `boolean_dsa_ds_mla`'s `_v2`, which returns its words on every row kind, the scale
    each row's own."""
    TILE = 32
    tile = bind(M.DsaLatentRowDsMlaTile, TILE=TILE)
    bits = boolean_version(tile)
    assert BL.is_boolean(bits) and TR.word_of(bits).id == tile.id
    rows = _rows(np.random.default_rng(11), TILE)
    x, w = (np.stack([r[k] for r in rows]) for k in (1, 3))
    sumsq = bind(DN.DsaRMSNormSumSq, N=TILE, EPS=1e-5)
    r = np.array([_flat(evaluate(sumsq, _words(x[i])))[0] for i in range(len(rows))], np.uint32)
    got = [np.asarray(a).reshape(len(rows), -1) for a in BS.evaluate_words(bits, x, w, r)]
    for i in range(len(rows)):
        want = _flat(evaluate(tile, _words(x[i]), _words(w[i]), int(r[i])))
        assert [int(v) for a in got for v in a[i]] == want, rows[i][0]


def test_at_glm_widths_a_tile_part_fits_a_proof_block():
    IF.check_proof_block(bind(M.DsaLatentRowDsMlaTile, TILE=128))
