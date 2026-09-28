"""sha256/row-nvfp4/v1 and blake3-keyed/row-nvfp4/v1: the NVFP4 row bytes (codes packed two per byte, then the scale bytes),
their digests and frame-v3 leaves, against flock-backend's reference vectors (computed with hashlib and the blake3 package)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.ml.commitments import rowleaf as R
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

VECTORS = json.loads((Path(R.__file__).parent / "frame_v3" / "nvfp4_row_v1_vectors.json").read_text())
K, SCALES = VECTORS["K"], VECTORS["scales"]


def _cases() -> dict[str, tuple[list[int], list[int]]]:
    rng = random.Random(20260926)
    return {"zeros": ([0] * K, [0] * SCALES),
            "ramp": ([i % 16 for i in range(K)], [0x38] * SCALES),
            "random": ([rng.randrange(16) for _ in range(K)], [rng.randrange(128) for _ in range(SCALES)])}


@pytest.mark.parametrize("vec", VECTORS["vectors"], ids=lambda v: f"{v['case']}-role{v['role']}")
def test_nvfp4_row_digests_match_the_reference_vectors(vec: dict) -> None:
    codes, scales = _cases()[vec["case"]]
    rb = R.nvfp4_row_bytes(codes, scales)
    assert len(rb) == vec["row_bytes_len"] == VECTORS["row_bytes"] == 864
    assert rb[:40].hex() == vec["row_bytes_head_hex"] and rb[K // 2:K // 2 + 8].hex() == vec["scale_bytes_hex"]
    assert R.sha256_row_nvfp4_digest(codes, scales, vec["role"]).hex() == vec["sha256/row-nvfp4/v1"]
    assert R.blake3_row_nvfp4_digest(codes, scales, vec["role"]).hex() == vec["blake3-keyed/row-nvfp4/v1"]


def test_the_layout_is_the_code_buffer_then_the_scale_buffer() -> None:
    codes = [1, 2] + [0] * (K - 2)
    scales = [7] + [0] * (SCALES - 1)
    rb = R.nvfp4_row_bytes(codes, scales)
    assert rb[0] == 0x21 and rb[K // 2] == 7 and len(rb) == K // 2 + K // 16
    assert len(R.sha256_row_nvfp4_prefix(R.ROLE_X, K)) == 64
    assert R.sha256_row_nvfp4_layout(R.ROLE_W, K).value_len == R.blake3_row_nvfp4_layout(R.ROLE_W, K).value_len == 864


def test_the_schemas_separate_from_each_other_and_from_the_word_rows() -> None:
    codes, scales = _cases()["random"]
    x = R.sha256_row_nvfp4_digest(codes, scales, R.ROLE_X)
    assert x != R.sha256_row_nvfp4_digest(codes, scales, R.ROLE_W)
    assert R.blake3_row_nvfp4_digest(codes, scales, R.ROLE_X) != R.blake3_row_nvfp4_digest(codes, scales, R.ROLE_W)
    dom = CommitmentDomain(bytes(32), 0, RangeIndexedDomain(4))
    assert R.row_leaf(dom, 0, x, R.SCHEMA_SHA256_ROW_NVFP4) != R.row_leaf(dom, 0, x, R.SCHEMA_SHA256_ROW)
    assert R.row_leaf(dom, 0, x, R.SCHEMA_BLAKE3_ROW_NVFP4) != R.row_leaf(dom, 0, x, R.SCHEMA_BLAKE3_ROW)


@pytest.mark.parametrize(("codes", "scales", "why"), [
    ([0] * 15, [0], "multiple of 16"),
    ([0] * 16, [0, 0], "scale bytes"),
    ([16] + [0] * 15, [0], "4 bits"),
    ([0] * 16, [256], "one byte"),
])
def test_malformed_nvfp4_rows_are_refused(codes: list[int], scales: list[int], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        R.nvfp4_row_bytes(codes, scales)
