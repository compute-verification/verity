"""The fallback commit call on the CPU: its roots are `verity.ml.commitments.row_table`'s from the rows, salts and domains the
module states, its openings verify against them, and shape H's accounting is pinned.  The device committer is held to the
same reference on the H200 by ``check_roots.py``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import commit_fb as F  # noqa: E402
from verity.ml.commitments import row_table as RT  # noqa: E402

SALT = F.SaltSource(key=bytes(range(32)), segment=7 * 53 + 3, context=bytes(range(64, 128)))


def small_layer(rng: np.random.Generator, batch: int, layers: int = 2) -> dict:
    """Shape H's tables at d = 64 (4 heads of 16, MLP 256, context 4) for ``layers`` layers and the head, random values."""
    spec = F.shape_h(d=64, heads=4, head_dim=16, kv_heads=1, mlp=256, ctx=4, vocab=16)
    out = {}
    for prefix, part in [(f"model.layers.{l}.", spec["layer"]) for l in range(layers)] + [("model.head.", spec["head"])]:
        for name, (kind, n) in part.items():
            if kind == "s8":
                out[prefix + name] = (rng.integers(-128, 128, (batch, n), dtype=np.int8),
                                      rng.standard_normal(batch).astype(np.float32))
            elif kind == "bf16":
                out[prefix + name] = rng.integers(0, 1 << 16, (batch, n), dtype=np.uint16)
            else:
                out[prefix + name] = rng.integers(0, 16, (batch,), dtype=np.int32)
    return out


def shuffled_rows(rng: np.random.Generator, batch: int) -> list[tuple[str, int]]:
    ids = [(f"req-{k}", int(t)) for k, t in zip(rng.permutation(batch), rng.integers(0, 31, batch), strict=True)]
    return ids


def test_root_is_the_row_table_reference():
    rng = np.random.default_rng(1)
    for batch in (1, 5, 8):
        outs, rows = small_layer(rng, batch), shuffled_rows(rng, batch)
        got = F.commit_step(outs, SALT, rows=rows)
        ref = F.reference(outs, SALT, rows=rows)
        assert got.root == ref.root and len(got.root) == 64
        assert [r for _op, r in got.tables] == [t.root for t in ref.tables]
        assert [op for op, _r in got.tables] == sorted(outs, key=lambda op: F.RTB.table_key(op))


def test_slot_rows_rank_in_slot_order():
    rng = np.random.default_rng(2)
    outs = small_layer(rng, 12, layers=1)
    c = F.commit_step(outs, SALT)
    assert c.ranks == tuple(range(12)) and c.rows == tuple(F.slot_rows(12))
    assert c.root == F.reference(outs, SALT).root


def test_root_binds_values_salts_and_segment():
    rng = np.random.default_rng(3)
    outs = small_layer(rng, 4, layers=1)
    base = F.commit_step(outs, SALT).root
    op = "model.layers.0.mlp.down_proj"
    flipped = dict(outs, **{op: outs[op].copy()})
    flipped[op][2, 5] ^= 1
    assert F.commit_step(flipped, SALT).root != base
    assert F.commit_step(outs, F.SaltSource(bytes(32), SALT.segment, SALT.context)).root != base
    assert F.commit_step(outs, F.SaltSource(SALT.key, SALT.segment + 1, SALT.context)).root != base


def test_open_rows_verifies_against_the_root():
    rng = np.random.default_rng(4)
    batch = 6
    outs, rows = small_layer(rng, batch, layers=1), shuffled_rows(rng, batch)
    c = F.commit_step(outs, SALT, rows=rows)
    for op in ("model.layers.0.self_attn.o_proj.quant", "model.layers.0.self_attn.softmax", "model.head.argmax"):
        opening, witness = F.open_rows(c, op, outs[op], SALT, [0, 3, 5])
        assert opening.index == c.index(op) and sorted(witness) == sorted({c.ranks[b] for b in (0, 3, 5)})
        assert RT.verify_step(c.root, SALT.context, SALT.segment, len(c.tables), batch, opening, witness)
        bad = {r: (bytes([w[0][0] ^ 1]) + w[0][1:], w[1], w[2]) for r, w in witness.items()}
        assert not RT.verify_step(c.root, SALT.context, SALT.segment, len(c.tables), batch, opening, bad)
    codes, scales = outs["model.layers.0.self_attn.o_proj.quant"]
    row, n_bits, _salt = F.open_rows(c, "model.layers.0.self_attn.o_proj.quant", (codes, scales), SALT, [2])[1][c.ranks[2]]
    assert row == codes[2].tobytes() + scales[2:3].tobytes() and n_bits == 8 * len(row)


def test_open_rows_refuses_other_values():
    rng = np.random.default_rng(5)
    outs = small_layer(rng, 3, layers=1)
    c = F.commit_step(outs, SALT)
    op = "model.layers.0.residual_mlp"
    other = outs[op].copy()
    other[0, 0] ^= 1
    with pytest.raises(ValueError, match="not the committed table"):
        F.open_rows(c, op, other, SALT, [0])


def test_refusals():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="even number"):
        F.commit_step({"model.layers.0.x": rng.integers(0, 255, (3, 3), dtype=np.uint8)}, SALT)
    with pytest.raises(ValueError, match="batch's 3 rows"):
        F.commit_step({"model.layers.0.a": np.zeros((3, 4), np.uint16), "model.layers.0.b": np.zeros((2, 4), np.uint16)}, SALT)
    with pytest.raises(ValueError, match="row ids for a batch"):
        F.commit_step({"model.layers.0.a": np.zeros((3, 4), np.uint16)}, SALT, rows=F.slot_rows(2))
    with pytest.raises(ValueError, match="twice"):
        F.commit_step({"model.layers.0.a": np.zeros((2, 4), np.uint16)}, SALT, rows=[("r", 0), ("r", 0)])
    with pytest.raises(ValueError, match="quantizer output"):
        F.commit_step({"model.layers.0.q": (np.zeros((2, 4), np.int16), np.zeros(2, np.float32))}, SALT)
    with pytest.raises(ValueError, match="32-byte row key"):
        F.SaltSource(bytes(31), 0, bytes(64))


def test_segments_number_a_steps_parts():
    assert [F.segment_of(5, p, 53) for p in (0, 52)] == [265, 317]
    assert F.segment_of(6, 0, 53) == F.segment_of(5, 52, 53) + 1
    with pytest.raises(ValueError):
        F.segment_of(5, 53, 53)


def test_shape_h_bytes():
    """Shape H's committed bytes per token: a layer 672,064 every op, 459,056 with compute-accounting's fused cuts left out,
    and the head 49,684.  At 20,480 sequences and 52 layers a step is 717 GB every op, 490 GB fused."""
    spec = F.shape_h()
    layer = sum(F.row_bytes(k, n) for k, n in spec["layer"].values())
    fused = sum(F.row_bytes(k, n) for name, (k, n) in spec["layer"].items() if name not in F.FUSED_OUT)
    head = sum(F.row_bytes(k, n) for k, n in spec["head"].values())
    assert (layer, fused, head) == (672_064, 459_056, 49_684)
    assert len(spec["layer"]) == 16 and spec["layer"]["self_attn.qkv_proj"] == ("bf16", 16_512)
    assert (round(20_480 * (52 * layer + head) / 1e9), round(20_480 * (52 * fused + head) / 1e9)) == (717, 490)
