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
from verity_pous.scheme import encode_segment

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

    width = prims.h.pi.width
    state = torch.randint(0, 256, (2, width // 8), dtype=torch.uint8, generator=g).cuda()
    want = [to_bytes(prims.h.pi.forward(from_bytes(r, width)), width) for r in _rows(state)]
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


@needs_cuda
def test_segment_encode_matches_the_reference_and_decodes(ext):
    from verity_vllm.engine.pous import codec

    prims = reference(P)
    g = torch.Generator().manual_seed(3)
    w = torch.randint(0, 256, (1, P.n, M), dtype=torch.uint8, generator=g)
    salt = bytes(range(7, 7 + codec.SALT_BYTES))
    tags = codec.segment_tags(salt, [5], "cuda")
    c = codec.encode(w.cuda(), tags)
    (_, c2), _ = run_prog(encode_segment(P, prims.h, tag(salt_value(salt), 5), [from_bytes(r, P.m) for r in _rows(w[0])]), prims)
    assert _rows(c[0]) == [to_bytes(x, P.m) for x in c2]
    assert torch.equal(codec.decode(c, tags).cpu(), w)
    assert torch.equal(codec.transpose(codec.transpose(c)), c)
