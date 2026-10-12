import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import layout

_spec = importlib.util.spec_from_file_location("wc_codec", HERE.parents[1] / "word_codec" / "codec.py")
codec = sys.modules["wc_codec"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(codec)

SHAPES = [(16, 512), (32, 1024), (48, 512)]


def matrix(n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, k), generator=g, dtype=torch.int64).to(torch.uint8)


@pytest.mark.parametrize("n,k", SHAPES)
def test_fragment_stream_is_codecs(n, k):
    w = matrix(n, k, n * k)
    ref = codec.fragment_stream(bytes(w.flatten().tolist()), n, k)
    got = layout.fragment_stream(w)
    assert got.shape == (len(ref) // 32, 32, 256)
    assert [bytes(r.tolist()) for r in got.reshape(-1, 256)] == ref
    assert torch.equal(layout.unfragment(got, n, k), w)


@pytest.mark.parametrize("n,k", SHAPES)
def test_plain_fragments_are_codecs(n, k):
    w = matrix(n, k, 7 + n)
    got = layout.plain_fragments(w)
    assert bytes(got.tolist()) == codec.plain_fragments(bytes(w.flatten().tolist()), n, k)
    assert torch.equal(layout.unplain(got, n, k), w)


@pytest.mark.parametrize("bits", layout.PACKINGS)
@pytest.mark.parametrize("n,k", SHAPES[:2])
def test_payloads_are_codecs(n, k, bits):
    w = matrix(n, k, 11 * n + bits)
    ref = codec.fragment_payloads(bytes(w.flatten().tolist()), n, k, bits)
    got = layout.payloads(w, bits)
    assert got.tolist() == ref
    assert int(got.max()) < 1 << bits and int(got.min()) >= 0
    assert got.numel() == layout.store_words(n, k, bits)
    assert torch.equal(layout.unpayload(got, n, k, bits), w)


@pytest.mark.parametrize("bits", layout.PACKINGS)
def test_pack_round_trips_extreme_bytes(bits):
    s = torch.full((3, 256), 0xFF, dtype=torch.uint8)
    s[1] = 0
    s[2, ::2] = 0x80
    p = layout.pack(s, bits)
    assert [codec.pack(bytes(r.tolist()), bits) for r in s] == p.tolist()
    assert torch.equal(layout.unpack(p, bits), s)


def test_fp8_input_is_its_bytes():
    w = matrix(16, 512, 3)
    f8 = w.view(torch.float8_e4m3fn)
    assert torch.equal(layout.fragment_stream(f8), layout.fragment_stream(w))


def test_shapes_refused():
    with pytest.raises(ValueError):
        layout.check_shape(24, 512)
    with pytest.raises(ValueError):
        layout.check_shape(16, 500)
    with pytest.raises(ValueError):
        layout.lane_words(64)


def test_llama_shapes_lay_out():
    for n, k in [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]:
        layout.check_shape(n, k)
        assert layout.store_words(n, k, 56) == n * k // 512 // 16 * 37 * 32


def test_decoded_reference_words():
    """Words the codec encodes decode back to `layout.payloads`' values at their store index (the device gate's check)."""
    w = matrix(16, 512, 99)
    p = layout.payloads(w, 56).tolist()
    key = codec.key(bytes(range(32)), 2)
    rnd = random.Random(5)
    for i in rnd.sample(range(len(p)), 16):
        y = codec.encode_word(p[i], 1000 + i, key)
        assert codec.decode_word(y, 1000 + i, key) == p[i]
