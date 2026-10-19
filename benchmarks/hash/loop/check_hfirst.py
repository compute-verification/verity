"""The committer's two-half leaves (``StepHasher.rows_h`` then ``rows_frame``, which a learning step that keeps no bytes takes so its
windows are hashed in place before the step's length is known) are ``rows_leaves``' bytes, under rows-sha512 and rows-blake3: h from a
RowStep of another domain (the same salt key), the leaves and pads from the step's own, b || c and h alike, with derived and given salts,
at position widths, past one thread block and in capped launches.  job.sh runs it once in each rows-sha512 leaf form."""
from __future__ import annotations

import hashlib
import random

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the rows kernels need a GPU", allow_module_level=True)

from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit import step_rows_b3 as B  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402

PD, CTX, GEO = (hashlib.sha256(t).digest() for t in (b"program", b"ctx", b"geo"))
LB = R.ROW_BYTES


@pytest.fixture(scope="module")
def ext():
    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


def _st() -> int:
    return torch.cuda.current_stream().cuda_stream


def _dev(b: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(b.ljust(16, b"\0")), dtype=torch.uint8).cuda()


def _step(ext, scheme: str, n: int, ks: bytes):
    dom = R.domain(PD, CTX, GEO, n)
    if scheme == B.NAME:
        return ext.RowStep(B.did(dom), ks, B.row_prefix(8 * LB), B.SALT_PREFIX, B.leaf_prefix(), B.SCHEMA, LB, hash="blake3",
                           hash_key=B.HASH_KEY)
    return ext.RowStep(dom.domain_id, ks, R.row_prefix(8 * LB), R.INST.salt_prefix, R.INST.leaf_prefix(R.KEY), R.SCHEMA, LB)


def _tn(scheme: str) -> torch.Tensor:
    return _dev(B.t_table() if scheme == B.NAME else R.t_nibbles(R.KEY))


@pytest.mark.parametrize("scheme", [R.NAME, B.NAME])
@pytest.mark.parametrize("first,m,n_total,given,cap", [(0, 300, 290, False, 0), (0, 300, 290, True, 0), (250, 12, 258, False, 0),
                                                       (65530, 12, 65538, True, 0), ((1 << 24) - 6, 12, (1 << 24) + 2, False, 0),
                                                       (512, 1024, 1536, False, 1), (512, 1024, 1300, True, 1)])
def test_h_then_frame_is_rows_leaves(ext, hasher, scheme, first, m, n_total, given, cap):
    rng = random.Random(first ^ m)
    mod = B if scheme == B.NAME else R
    ks = rng.randbytes(mod.KEY_BYTES)
    n_rows = min(m, n_total - first)
    rows = _dev(rng.randbytes(LB * n_rows))
    salts = _dev(rng.randbytes(mod.SALT_BYTES * n_rows)) if given else None
    tn = _tn(scheme)
    rs, rs_any = _step(ext, scheme, n_total, ks), _step(ext, scheme, 1, ks)
    rs.max_blocks = rs_any.max_blocks = cap
    lv, bc, hs = hasher.rows_leaves(rs, rows, LB * n_rows, first, m, n_total, tn, _st(), salts=salts, want_bc=True, want_h=True)
    h, bc2 = hasher.rows_h(rs_any, rows, LB * n_rows, first, tn, _st(), salts=salts, want_bc=True)
    lv2 = hasher.rows_frame(rs, h, first, m, n_total, _st())
    torch.cuda.synchronize()
    assert torch.equal(h, hs[:n_rows]) and torch.equal(bc2, bc[:n_rows])
    assert torch.equal(lv2, lv)


def test_rows_frame_from_no_h_is_all_pads(ext, hasher):
    ks = bytes(64)
    rs = _step(ext, R.NAME, 3, ks)
    lv, _, _ = hasher.rows_leaves(rs, _dev(b""), 0, 3, 5, 3, _tn(R.NAME), _st())
    lv2 = hasher.rows_frame(rs, torch.empty(0, 64, dtype=torch.uint8, device="cuda"), 3, 5, 3, _st())
    assert torch.equal(lv2, lv)
