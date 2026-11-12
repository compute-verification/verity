"""The fp8_ds_mla read-back capture (`fp8_read_back_capture_gpu`) on the CPU: its task map against the kernel's, walked page by
page as `cp_gather_and_upconvert_fp8_kv_cache_page` walks it; its planted rows, which carry each hand vector and every code;
its in-flight comparison and its compare phase, which find one flipped bit of one word; and its SASS counts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.properties import fp8_read_back_capture_gpu as X
from verity.definitions.registry import fp8_ds_mla_upconvert as U

DN, DR = U.FP8_DS_MLA_GLM53["DN"], U.FP8_DS_MLA_GLM53["DR"]


def pages(table, ws, starts, batch: int, total: int, bs: int) -> dict[int, tuple[int, int, int, int]]:
    """Output row -> (request, position, block, slot), task by task as cache_kernels.cu:1139-1176 and :1387-1425 run them:
    ceil(total / bs) + 2 batch tasks, each a page of one request."""
    out = {}
    for task in range(-(-total // bs) + 2 * batch):
        rel = task
        for r in range(batch):
            ob = min(int(ws[r]), total)
            oe = min(int(ws[r + 1]) if r + 1 < batch else total, total)
            n = max(oe - ob, 0)
            sb = 0 if starts is None else int(starts[r])
            first = sb // bs
            npages = -(-(sb + n) // bs) - first
            if rel < npages:
                lb = first + rel
                pb = lb * bs
                cb, ce = max(sb, pb), min(sb + n, pb + bs)
                for t in range(cb - pb, ce - pb):
                    out[ob + cb - sb + t - (cb - pb)] = (r, pb + t, int(table[r, lb]), t)
                break
            rel -= npages
    return out


@pytest.mark.parametrize("seed", range(40))
def test_sources_is_the_kernels_task_map(seed):
    rng = np.random.default_rng(seed)
    batch, bs = int(rng.integers(1, 5)), int(rng.choice([16, 64]))
    lens = rng.integers(0, 200, batch)
    ws = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32)
    total = int(lens.sum())
    starts = rng.integers(0, 150, batch).astype(np.int32) if seed % 2 else None
    table = rng.integers(0, 1000, (batch, 24)).astype(np.int32)
    got = X.sources(table, ws, starts, batch, total, bs)
    want = pages(table, ws, starts, batch, total, bs)
    assert len(want) == total
    assert all(tuple(got[o]) == want[o] for o in range(total))


def test_planted_rows_carry_the_vectors_and_every_code():
    rows = X.planted_rows(np.random.default_rng(0))
    assert rows.shape[1] == U.row_bytes(DN, DR) == 656
    nope, scale, _ = U.split_rows(rows, DN, DR)
    for i, (b, s, want, _why) in enumerate(U.HAND_VECTORS):
        assert (nope[i] == b).all() and (scale[i] == s).all()
        words = U.upconvert_rows(nope[i:i + 1], scale[i:i + 1], np.zeros((1, DR), np.uint16))[0]
        assert (words[0, :DN] == want).all()
    assert set(np.unique(nope).tolist()) == set(range(256))
    assert set(U.EDGE_SCALES) <= set(np.unique(scale).tolist())


def test_held_names_the_first_differing_word():
    rows = X.planted_rows(np.random.default_rng(1))[:40]
    words = U.upconvert_rows(*U.split_rows(rows, DN, DR))[0]
    assert X.held(rows, words) == {"rows": 40, "rows_differing": 0, "words_differing": 0}
    words[7, 3] ^= 1
    got = X.held(rows, words)
    assert (got["rows_differing"], got["words_differing"], got["first"]["row"], got["first"]["word"]) == (1, 1, 7, 3)


def test_compare_holds_saved_rows_to_the_definition(tmp_path):
    rows = X.planted_rows(np.random.default_rng(2))[:24]
    words = U.upconvert_rows(*U.split_rows(rows, DN, DR))[0]
    for d in ("planted", "calls"):
        (tmp_path / d).mkdir()
    np.savez_compressed(tmp_path / "planted" / "00.npz", rows=rows[:12], words=words[:12])
    np.savez_compressed(tmp_path / "calls" / "00000.npz", rows=rows[12:], words=words[12:])
    assert X.compare(tmp_path, library=False) == 0
    doc = json.loads((tmp_path / "compare.json").read_text())
    assert [(s["rows"], s["rows_differing"]) for s in doc["sets"].values()] == [(12, 0), (12, 0)]
    bad = words[12:].copy()
    bad[5, DN + 1] ^= 0x8000
    np.savez_compressed(tmp_path / "calls" / "00000.npz", rows=rows[12:], words=bad)
    assert X.compare(tmp_path, library=False) == 1
    doc = json.loads((tmp_path / "compare.json").read_text())
    assert doc["sets"]["served"]["rows_differing"] == 1 and doc["sets"]["served"]["words_differing"] == 1


def test_compare_refuses_an_empty_set(tmp_path):
    rows = X.planted_rows(np.random.default_rng(3))[:4]
    (tmp_path / "planted").mkdir()
    np.savez_compressed(tmp_path / "planted" / "00.npz", rows=rows, words=U.upconvert_rows(*U.split_rows(rows, DN, DR))[0])
    assert X.compare(tmp_path, library=False) == 1


SASS = """
        /*0290*/                   F2I.FTZ.U32.TRUNC.NTZ R5, R4 ;                    /* 0x0000000400057305 */
        /*0e00*/                   F2FP.F16.E4M3.UNPACK_B R11, R21 ;                 /* 0x00000015ff0b723e */
        /*0ec0*/                   HADD2.F32 R9, -RZ, R11.H0_H0 ;                    /* 0x2000000bff097230 */
        /*0fa0*/                   FMUL R8, R14.reuse, R19 ;                         /* 0x000000130e087220 */
        /*0fb0*/              @!P0 FMUL.FTZ R30, R14, R11 ;                          /* 0x0000000b0e1e8220 */
        /*0fc0*/                   F2FP.BF16.F32.PACK_AB R8, R9, R8 ;                /* 0x00000009ff08723e */
        /*0fd0*/                   FFMA.RZ R1, R2, R3, R4 ;                          /* 0x0000000302017223 */
"""


def test_sass_counts_by_opcode():
    c = X.sass_counts(SASS)
    assert (c["F2FP.F16.E4M3.UNPACK_B"], c["HADD2.F32"], c["FMUL"], c["FMUL.FTZ"], c["F2FP.BF16.F32.PACK_AB"]) == (1, 1, 1, 1, 1)
    assert (c["FFMA"], c["FFMA*"], c[".FTZ outside F2I"], c["instructions"]) == (0, 1, 1, 7)
