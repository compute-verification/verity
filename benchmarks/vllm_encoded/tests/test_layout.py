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


_wspec = importlib.util.spec_from_file_location("wc_wide", HERE.parents[1] / "word_codec" / "wide.py")
wide = sys.modules["wc_wide"] = importlib.util.module_from_spec(_wspec)
_wspec.loader.exec_module(wide)


@pytest.mark.parametrize("bits", sorted(wide.FIELDS))
@pytest.mark.parametrize("n,k", SHAPES[:2])
def test_wide_payloads_are_wides(n, k, bits):
    f = wide.field(bits)
    w = matrix(n, k, 13 * n + bits)
    ref = wide.store_bytes(wide.fragment_payloads(bytes(w.flatten().tolist()), n, k, f), f)
    got = layout.wide_payloads(w, f.payload, f.limbs, chunk=3)
    assert got.dtype == torch.int32 and got.numpy().tobytes() == ref
    assert got.numel() == layout.wide_store_words(n, k, f.payload) * f.limbs
    assert torch.equal(layout.wide_unpayload(got, n, k, f.payload, f.limbs, chunk=3), w)


@pytest.mark.parametrize("bits", sorted(wide.FIELDS))
def test_wide_pack_round_trips_extreme_bytes(bits):
    f = wide.field(bits)
    s = torch.full((3, 256), 0xFF, dtype=torch.uint8)
    s[1] = 0
    s[2, ::2] = 0x80
    v = layout.wide_pack(s, f.payload, f.limbs)
    assert v.shape == (3, f.lane_words(), f.limbs) == (3, layout.wide_lane_words(f.payload), f.limbs)
    for row, words in zip(s, v.tolist()):
        assert [sum(x << (32 * e) for e, x in enumerate(limbs)) for limbs in words] == wide.pack(bytes(row.tolist()),
                                                                                                 f.payload)
    assert torch.equal(layout.wide_unpack(v, f.payload), s)


@pytest.mark.parametrize("bits", sorted(wide.FIELDS))
def test_wide_limb_index_reads_the_store(bits):
    """`layout.wide_word` reads word i of a flat store as `wide.store_bytes` lays it out (the device gate's reader)."""
    f = wide.field(bits)
    w = matrix(16, 1024, 5 + bits)
    lanes = wide.fragment_payloads(bytes(w.flatten().tolist()), 16, 1024, f)
    store = layout.wide_payloads(w, f.payload, f.limbs)
    rnd = random.Random(bits)
    lw = f.lane_words()
    for _ in range(12):
        q, lane, wi = rnd.randrange(len(lanes) // 32), rnd.randrange(32), rnd.randrange(lw)
        i = wide.word_index(q, lane, wi, lw)
        assert layout.wide_word(store, i, f.limbs) == lanes[32 * q + lane][wi]
    assert layout.wide_block(store, 33, lw, f.limbs) == b"".join(y.to_bytes(4 * f.limbs, "little") for y in lanes[33])


_tspec = importlib.util.spec_from_file_location("ct_wide", HERE.parents[1] / "codec_tensor" / "wide.py")
ct = sys.modules["ct_wide"] = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(ct)


def ct_params(name: str) -> tuple:
    f = ct.field(name)
    return f, ct.TASK_WORDS[name], -(-f.n // 32)


@pytest.mark.parametrize("name", sorted(ct.FIELDS))
@pytest.mark.parametrize("n,k", SHAPES[:2])
def test_tensor_limbs_are_codec_tensors(n, k, name):
    """`layout.tensor_limbs` holds each task's `task_payloads` of `codec.plain_fragments`, word w of task q at L·q + w."""
    f, tw, nl = ct_params(name)
    w = matrix(n, k, 17 * n + f.n)
    plain = codec.plain_fragments(bytes(w.flatten().tolist()), n, k)
    ref = [m for q in range(len(plain) // ct.TASK_BYTES)
           for m in ct.task_payloads(plain[ct.TASK_BYTES * q:ct.TASK_BYTES * (q + 1)], f)]
    got = layout.tensor_limbs(w, tw, f.payload_bytes, nl)
    assert got.dtype == torch.int32 and got.numel() == len(ref) * nl
    assert [layout.tensor_limb_word(got, i, nl) for i in range(len(ref))] == ref
    assert torch.equal(layout.tensor_unlimbs(got, n, k, tw, f.payload_bytes, nl), w)


@pytest.mark.parametrize("name", ["m521", "p767"])
def test_tensor_words_decode_to_limbs(name):
    """`encode_tasks`' store, read by `layout.tensor_word`, decodes to `layout.tensor_limbs`' payloads at index base + i
    (the device gate's check)."""
    f, tw, nl = ct_params(name)
    w = matrix(16, 512, f.n)
    key = ct.key(name, bytes(range(32)), 1)
    store = torch.frombuffer(bytearray(ct.encode_tasks(codec.plain_fragments(bytes(w.flatten().tolist()), 16, 512), key, 3)),
                             dtype=torch.uint8)
    limbs = layout.tensor_limbs(w, tw, f.payload_bytes, nl)
    assert store.numel() == tw * f.word_bytes
    for i in random.Random(f.n).sample(range(tw), 8):
        y = layout.tensor_word(store, i, f.word_bytes)
        assert ct.decode_word(y, 3 * tw + i, key) == layout.tensor_limb_word(limbs, i, nl)
