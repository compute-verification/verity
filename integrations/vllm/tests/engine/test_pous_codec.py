"""The GPU P3 codec (`engine/pous/codec.py`) against the reference `verity_pous` (its `reference` backend), piece by piece and
on a whole 14.4 MB segment.  CUDA-only; the layout test runs anywhere.
"""
from __future__ import annotations

import hashlib
import json
import struct

import pytest
from verity_pous.labels import from_bytes, salt_value, tag, to_bytes
from verity_pous.oracle import Tweak
from verity_pous.oracle import run as run_prog
from verity_pous.params import OPERATING_POINT
from verity_pous.primitives import key_program, key_query, reference
from verity_pous.scheme import P3

from verity_vllm.engine.pous import layout

torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the P3 kernel needs a CUDA device")


@pytest.fixture(scope="module")
def ext(tmp_path_factory):
    from verity_vllm.engine.pous import native

    return native.load(str(tmp_path_factory.mktemp("pous_build")))
P = OPERATING_POINT
M = P.block_bytes


def _rows(t) -> list[bytes]:
    return [bytes(r) for r in t.cpu().tolist()]


def test_layout_is_the_reference_layout_of_the_data_section(tmp_path):
    body = bytes(range(256)) * 3
    header = {"b": {"dtype": "BF16", "shape": [256], "data_offsets": [256, 768]},
              "a": {"dtype": "BF16", "shape": [2, 64], "data_offsets": [0, 256]}, "__metadata__": {"format": "pt"}}
    raw = json.dumps(header).encode()
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)
    ck = layout.read(path)
    assert [(p.name, p.offset, p.size) for p in ck.layout.tensors] == [("a", 0, 256), ("b", 256, 512)]
    assert ck.read_w() == ck.layout.pack({"a": body[:256], "b": body[256:]})
    assert ck.segment_range(["a", "b"]) == (0, 1) and ck.shapes["a"] == (2, 64)
    raw2 = json.dumps({"c": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 16]}}).encode()
    second = tmp_path / "m2.safetensors"
    second.write_bytes(struct.pack("<Q", len(raw2)) + raw2 + bytes(range(16)))
    two = layout.read([path, second])
    assert two.layout.placement("c").offset == 768 and two.read_w()[:784] == body + bytes(range(16))
    with pytest.raises(ValueError, match="two checkpoint files"):
        layout.read([path, path])
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body + b"x")
    with pytest.raises(ValueError, match="trailing"):
        layout.read(path)


@needs_cuda
def test_shake256_rows_match_hashlib(ext):
    g = torch.Generator().manual_seed(1)
    for plen, blen, olen in ((0, 0, 32), (2, 135, 136), (37, 1000, 300), (2, 32812, 32812), (5, 136 * 3 - 5, 137)):
        pre = torch.randint(0, 256, (4, plen), dtype=torch.uint8, generator=g)
        body = torch.randint(0, 256, (4, blen), dtype=torch.uint8, generator=g)
        out = torch.randint(0, 256, (4, olen), dtype=torch.uint8, generator=g)
        want = [bytes(x ^ y for x, y in zip(o, hashlib.shake_256(p + b).digest(olen)))
                for o, p, b in zip(_rows(out), _rows(pre), _rows(body))]
        dev = out.cuda()
        ext.shake256_rows(pre.cuda(), body.cuda(), dev, True)
        assert _rows(dev) == want


@needs_cuda
def test_permutations_and_keys_match_the_reference(ext):
    from verity_vllm.engine.pous import codec

    prims = reference(P)
    g = torch.Generator().manual_seed(2)
    salt = bytes(range(codec.SALT_BYTES))
    tags = codec.segment_tags(salt, [0, 3], "cuda")
    labels = torch.randint(0, 256, (2, P.n, M), dtype=torch.uint8, generator=g).cuda()
    host = [[bytes(r) for r in seg.tolist()] for seg in labels.cpu()]

    perm = getattr(prims.h, "pi2", None) or prims.h.pi
    width = perm.width
    state = torch.randint(0, 256, (2, width // 8), dtype=torch.uint8, generator=g).cuda()
    want = [to_bytes(perm.forward(from_bytes(r, width)), width) for r in _rows(state)]
    ext.feistel_rows(state, state.new_empty((2, 0)), codec.PI_DOMAIN, False)
    assert _rows(state) == want

    vs = torch.tensor([0, 1, 12, 13, 219], device="cuda")
    groups = torch.tensor([0, 1, 1, 0, 1], device="cuda")
    got = _rows(codec.keys(tags[groups], 1, vs, labels, groups))
    for r, (gi, v) in enumerate(zip(groups.tolist(), vs.tolist())):
        t = tag(salt_value(salt), [0, 3][gi])
        ints = [from_bytes(b, P.m) for b in host[gi]]
        key, _ = run_prog(key_program(prims.h, key_query(P, t, 1, v, ints)), prims)
        assert got[r] == to_bytes(key, P.m), (gi, v)

    x = labels[0, :3].clone()
    codec.p_t(x, tags[:1].expand(3, -1), 0, torch.tensor([5, 6, 7], device="cuda"))
    t0 = tag(salt_value(salt), 0)
    assert _rows(x) == [to_bytes(prims.p.forward(Tweak(t0, 0, v), from_bytes(host[0][r], P.m)), P.m) for r, v in enumerate((5, 6, 7))]
    codec.p_t(x, tags[:1].expand(3, -1), 0, torch.tensor([5, 6, 7], device="cuda"), inverse=True)
    assert torch.equal(x, labels[0, :3])


def test_the_default_key_derivation_has_a_gpu_twin_and_others_fail_closed():
    from verity_pous import primitives

    from verity_vllm.engine.pous import codec

    assert codec.MODE == "overwrite"
    sponge = primitives.Primitives(primitives.XorSponge(P, primitives.FeistelPi(P.m + 512)), primitives.FeistelP(P.m))
    assert codec.key_mode(sponge) == "sponge"
    for bad in (primitives.Primitives(primitives.MiyaguchiPreneel(P, primitives.FeistelPi(P.m + 512)), primitives.FeistelP(P.m)),
                primitives.heuristic_arx(P)):
        with pytest.raises(NotImplementedError):
            codec.key_mode(bad)


@needs_cuda
def test_xor_sponge_keys_match_the_reference(ext, monkeypatch):
    """The sponge mode (broken as `H`, kept so both GPU modes are checked) against the reference's `XorSponge`."""
    from verity_pous import primitives

    from verity_vllm.engine.pous import codec

    monkeypatch.setattr(codec, "MODE", "sponge")
    prims = primitives.Primitives(primitives.XorSponge(P, primitives.FeistelPi(P.m + 512)), primitives.FeistelP(P.m))
    g = torch.Generator().manual_seed(4)
    salt = bytes(range(3, 3 + codec.SALT_BYTES))
    labels = torch.randint(0, 256, (1, P.n, M), dtype=torch.uint8, generator=g).cuda()
    ints = [from_bytes(bytes(r), P.m) for r in labels[0].cpu().tolist()]
    vs = torch.tensor([0, 5, 13], device="cuda")
    got = _rows(codec.keys(codec.segment_tags(salt, [2], "cuda")[[0, 0, 0]], 0, vs, labels,
                           torch.zeros(3, dtype=torch.long, device="cuda")))
    t = tag(salt_value(salt), 2)
    for r, v in enumerate(vs.tolist()):
        key, _ = run_prog(key_program(prims.h, key_query(P, t, 0, v, ints)), prims)
        assert got[r] == to_bytes(key, P.m), v


@needs_cuda
def test_segment_encode_and_decode_match_the_reference_codec(ext):
    """The GPU twin of `codec.REFERENCE` against its byte-level `encode_segment` / `decode_segment` on segment 5."""
    from verity_vllm.engine.pous import codec

    from verity_vllm.engine.pous import twins

    ref = codec.REFERENCE
    encode, decode = twins.twin(ref)
    params, prims = P.with_segments(6), reference(P)
    g = torch.Generator().manual_seed(3)
    w = torch.randint(0, 256, (1, P.n, M), dtype=torch.uint8, generator=g)
    salt = bytes(range(7, 7 + codec.SALT_BYTES))
    tags = codec.segment_tags(salt, [5], "cuda")
    c = encode(w.cuda(), tags)
    blocks = ref.encode_segment(params, prims, salt, 5, w[0].numpy().tobytes())
    assert _rows(c[0]) == blocks
    assert decode(c, tags).cpu().numpy().tobytes() == ref.decode_segment(params, prims, salt, 5, blocks) == w[0].numpy().tobytes()
    assert torch.equal(codec.transpose(codec.transpose(c)), c)


def test_the_gpu_twin_is_chosen_by_reference_codec():
    from verity_pous.codec import Codec
    from verity_pous.dense import DENSE

    from verity_vllm.engine.pous import codec, dense, twins

    assert codec.REFERENCE is P3 and twins.twin(P3) == (codec.encode, codec.decode)
    assert twins.twin(DENSE) == (dense.encode, dense.decode) and twins.reference(DENSE.name)[1].block_bytes == 65_536
    with pytest.raises(NotImplementedError):
        twins.twin(Codec("other/v1", P3.encode_segment, P3.decode_segment))


@needs_cuda
def test_feistel_kernels_agree(ext):
    """The one-warp, one-thread, word-I/O and fused Feistel kernels give the same bits on Π₂ at both schemes' widths, and at a
    width whose half is a whole number of Keccak blocks."""
    from verity_vllm.engine.pous import codec

    g = torch.Generator().manual_seed(6)
    for width in (2 * 65_536 + 64, 2 * codec.M + 64, 2 * 136 * 8):
        x = torch.randint(0, 256, (37, width), dtype=torch.uint8, generator=g).cuda()
        empty = x.new_empty((37, 0))
        outs = []
        for thread_rows, words, fused, pair, co in ((0, False, False, False, False), (1, False, False, False, False),
                                                    (1, True, False, False, False), (1, True, True, False, False),
                                                    (1, True, True, True, False), (1, True, True, False, True)):
            y = x.clone()
            ext.feistel_rows(y, empty, codec.PI_DOMAIN, False, thread_rows, words, fused, pair, 1 << 62, co)
            outs.append(y)
        assert all(torch.equal(outs[0], o) for o in outs[1:]), width
        ext.feistel_rows(outs[2], empty, codec.PI_DOMAIN, True, 1, True, True, True)
        assert torch.equal(outs[2], x)


@needs_cuda
def test_dense_twin_matches_the_reference_dense_codec(ext):
    """The dense GPU twin against `verity_pous.dense.DENSE` on short segments (the chain is the same at every length): encode,
    decode, and decode of arbitrary bytes, at the 64 KiB width."""
    from verity_pous import dense as ref

    from verity_vllm.engine.pous import codec, dense

    params = ref.DenseParams(6, 65_536 * 8, segments=4)
    prims = ref.reference(params)
    g = torch.Generator().manual_seed(5)
    w = torch.randint(0, 256, (1, 6, 65_536), dtype=torch.uint8, generator=g)
    salt = bytes(range(11, 11 + codec.SALT_BYTES))
    tags = codec.segment_tags(salt, [3], "cuda")
    c = dense.encode(w.cuda(), tags)
    blocks = ref.DENSE.encode_segment(params, prims, salt, 3, w[0].numpy().tobytes())
    assert _rows(c[0]) == blocks
    assert dense.decode(c, tags).cpu().numpy().tobytes() == w[0].numpy().tobytes()
    junk = torch.randint(0, 256, (1, 6, 65_536), dtype=torch.uint8, generator=g)
    assert (dense.decode(junk.cuda(), tags).cpu().numpy().tobytes()
            == ref.DENSE.decode_segment(params, prims, salt, 3, [bytes(r) for r in junk[0].tolist()]))
