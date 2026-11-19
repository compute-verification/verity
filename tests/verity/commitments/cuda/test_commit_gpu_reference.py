"""commit_gpu's reference (``commit_gpu/reference.py``): a prepared case's b || c, hm96 leaves and frame leaves are
``verity.commitments``' for the salts in its ``salts.bin`` (192 bytes a leaf in leaf order), and its parameter file holds
no salt key."""
from __future__ import annotations

from pathlib import Path

import pytest

from verity.commitments import hm96
from verity.commitments.cuda.commit_gpu import reference as R
from verity.commitments.rowleaf import (SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2, row_leaf,
                                        sha512_row_seg_digest, sha512_row_v2_digest)

INST = hm96.SHA512


@pytest.mark.parametrize("mode, n, n_bits, stride", [("rows", 37, 256, 32), ("rows", 5, 13, 16), ("seg", 3, 8193, 1040),
                                                     ("e2e", 6, R.E2E_BITS, 0)])
def test_a_cases_leaves_are_hm96s_for_its_salts(tmp_path: Path, mode: str, n: int, n_bits: int, stride: int):
    R.prepare_case(str(tmp_path), mode, n, n_bits, stride, R.KEY, seed=7)
    read = lambda name: (tmp_path / name).read_bytes()  # noqa: E731
    rows, salts, ref_bc, ref_h, levels = (read(f) for f in ("rows.bin", "salts.bin", "ref_bc.bin", "ref_h.bin", "ref_levels.bin"))
    assert len(salts) == 192 * n
    digest, schema = ((sha512_row_seg_digest, SCHEMA_HM96_SHA512_ROW_SEG) if mode == "seg"
                      else (sha512_row_v2_digest, SCHEMA_HM96_SHA512_ROW_V2))
    for pos in range(n):
        row = R.e2e_row(rows, pos) if mode == "e2e" else rows[pos * stride:pos * stride + (n_bits + 7) // 8]
        bc = INST.commit_string(R.KEY, digest(row, n_bits), salts[192 * pos:192 * pos + 192])
        h = INST.tree_leaf(R.KEY, bc)
        assert ref_bc[128 * pos:128 * pos + 128] == bc
        assert ref_h[64 * pos:64 * pos + 64] == h
        assert levels[64 * pos:64 * pos + 64] == row_leaf(R.domain(n), pos, h, schema)


def test_the_parameter_file_holds_no_salt_key():
    p = R.params(9, 256)
    assert len(p) == 4 + 128 * 3 + 64 + 65536 and p[:4] == b"CGP2"
    assert p[132:260] == INST.salt_prefix and p[260:388] == INST.leaf_prefix(R.KEY)
    assert p[388:452] == R.domain(9).domain_id and p[452:] == R.t_table(R.KEY)
