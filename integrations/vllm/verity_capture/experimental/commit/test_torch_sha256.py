"""torch SHA-256 backend == hashlib (CPU is enough; the same code runs on CUDA).  EXPERIMENTAL, CMT-1."""
from __future__ import annotations

import hashlib
import os
import struct

import pytest

torch = pytest.importorskip("torch")

from . import engine as E  # noqa: E402
from . import torch_sha256 as TS  # noqa: E402
from .positions import LEAF_BYTES, StepLayout  # noqa: E402

DEV = os.environ.get("CMT_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _rand(n, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, L), dtype=torch.uint8, generator=g)


@pytest.mark.parametrize("L", [0, 1, 55, 56, 63, 64, 90, 103, 119, 120, 200])
def test_sha256_messages_vs_hashlib(L):
    m = _rand(37, L, seed=L).to(DEV)
    d = TS.sha256_messages(m).cpu()
    for i in range(m.shape[0]):
        assert bytes(d[i].numpy().tobytes()) == hashlib.sha256(bytes(m[i].cpu().numpy().tobytes())).digest()


def test_frames_match_production():
    from verity_vllm import hashing
    from verity_vllm.commit_bench.semantic import pos_leaf
    v = _rand(9, LEAF_BYTES, seed=3).to(DEV)
    pl = TS.pos_leaf_digests(v).cpu()
    for i in range(9):
        assert bytes(pl[i].numpy().tobytes()) == pos_leaf(bytes(v[i].cpu().numpy().tobytes()))
    a, b = _rand(5, 32, seed=4).to(DEV), _rand(5, 32, seed=5).to(DEV)
    nd = TS.node_digests(a, b).cpu()
    lf = TS.lift_digests(a).cpu()
    for i in range(5):
        ab = bytes(a[i].cpu().numpy().tobytes()), bytes(b[i].cpu().numpy().tobytes())
        assert bytes(nd[i].numpy().tobytes()) == hashing.node_hash(*ab)
        assert bytes(lf[i].numpy().tobytes()) == hashing.lift_hash(ab[0])


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 17, 1000, 4097])
def test_fold_levels_matches_host(n):
    from verity_vllm.commit_bench import fasttree as ft
    leaves = _rand(n, 32, seed=n).to(DEV)
    lv_t = TS.fold_levels(leaves)
    lv_h = ft.fold_levels([bytes(x.numpy().tobytes()) for x in leaves.cpu()])
    assert len(lv_t) == len(lv_h)
    for a, b in zip(lv_t, lv_h, strict=True):
        assert [bytes(x.numpy().tobytes()) for x in a.cpu()] == b


def _stream(step, n_tok, seed):
    L = StepLayout(step, n_tok)
    g = torch.Generator().manual_seed(seed)
    parts = []
    for i in range(4):
        seg = L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"model.layers.{i}.self_attn.qkv_proj", "bf16", (n_tok, 960))
        parts.append(torch.randint(0, 256, (seg.nbytes,), dtype=torch.uint8, generator=g))
    seg = L.add("tokenselect1/out/tokenselect_v1/selecti32", "sampler", "i32", (1,))
    parts.append(torch.tensor(list(struct.pack("<i", 4242)), dtype=torch.uint8))
    L.freeze()
    s = torch.cat(parts)
    assert s.numel() == L.stream_bytes
    return L, s


@pytest.mark.parametrize("compiled,hlb", [(False, 0), (False, 64), (False, 3)])
def test_torch_backend_root_equals_host_and_opens(compiled, hlb):
    pd = hashlib.sha256(b"program").digest()
    ctx = E.Context("p", "r", 3)
    ch, ct = E.Committer(pd, ctx, backend="host"), E.Committer(pd, ctx, backend="torch", compiled=compiled, host_levels_below=hlb)
    streams = []
    for step, n_tok in enumerate((3, 1, 1)):
        L1, s = _stream(step, n_tok, seed=10 + step)
        L2 = StepLayout.from_json(L1.to_json())
        ch.commit_step(L1, bytes(s.numpy().tobytes()))
        ct.commit_step(L2, s.to(DEV))
        streams.append(s)
    rh, ph = ch.finish()
    rt, pt = ct.finish()
    assert rh == rt and ph["steps"] == pt["steps"]
    for step in range(3):
        for el in (0, 7, 959):
            op = E.open_value(ct, step, "gemm1/out/gemmcoordinate_v1/f2fpbf16", "model.layers.2.self_attn.qkv_proj", el, stream=streams[step].to(DEV))
            assert E.verify_opening(rt, pt, op) == (True, "ok")
            assert E.verify_opening(rh, ph, op) == (True, "ok")
        op = E.open_value(ct, step, "tokenselect1/out/tokenselect_v1/selecti32", "sampler", 0, stream=streams[step].to(DEV))
        assert struct.unpack("<i", op.value)[0] == 4242
        assert E.verify_opening(rt, pt, op)[0]
        bad = op.replace(value=struct.pack("<i", 4243))
        assert not E.verify_opening(rt, pt, bad)[0]
