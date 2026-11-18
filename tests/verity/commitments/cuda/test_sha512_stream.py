"""``sha512_stream``'s fill against ``verity.commitments.randomness``: its host twin, built by the host's C++ compiler, writes
the listed stream's blocks bit for bit (the pinned vectors, ``Key.stream`` and ``Key.block``) at every head length a
block takes in one compression, three compressions a leaf, and refuses a longer head or a counter past 2^64 by name."""
from __future__ import annotations

import ctypes
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, cast

import cuda_build
import numpy as np
import pytest

from verity.commitments.cuda.sha512_stream import engine as E
from verity.commitments.randomness import Key, derive

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "verity/commitments/cuda/sha512_stream/stream_host.cpp"
VECTORS = json.loads((ROOT / "tests/verity/commitments/randomness/vectors.json").read_text())
KEY = Key(bytes(range(32)))
#: the device library's build cache, the adapter's
BUILD = Path.home() / ".cache" / "verity" / "sha512_stream"


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    return E.bind(cuda_build.host_lib(SOURCE, (), tmp_path_factory.mktemp("host") / "sha512_stream.so"))


def _blocks(key: Key, index, first: int, leaves: int) -> bytes:
    return b"".join(key.block(index, c) for c in range(3 * first, 3 * (first + leaves)))


def _fill(lib: ctypes.CDLL, key: Key, index, first: int, leaves: int) -> bytes:
    return E.fill(lib, np.zeros((leaves, E.LEAF_BYTES), np.uint8), key, index, first).tobytes()


def _context(doc: dict) -> dict:
    return {n: bytes.fromhex(v["bytes"]) if "bytes" in v else v.get("str", v.get("int")) for n, v in doc.items()}


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: c["domain"])
def test_the_pinned_stream_vectors(lib, case):
    key = derive(bytes.fromhex(case["source"]), case["domain"], _context(case["context"]))
    s = case["stream"]
    n = len(s["bytes"]) // 2
    index = tuple(s["index"]) if isinstance(s["index"], list) else s["index"]
    assert _fill(lib, key, index, 0, -(-n // E.LEAF_BYTES))[:n].hex() == s["bytes"]


def test_every_one_compression_head_length(lib):
    """Heads of 78 to 103 bytes, so the counter starts at every byte of a word and straddles two words where it is not
    word-aligned; a leaf is three whole blocks, so no leaf splits a block."""
    lengths = set()
    for n in range(26):
        index = (bytes(range(n)),)
        lengths.add(len(E.head(KEY, index)))
        assert _fill(lib, KEY, index, 5, 3) == KEY.stream(index, 192 * 8)[192 * 5:]
    assert lengths == set(range(78, E.HEAD_MAX + 1))


def test_random_keys_indices_and_offsets(lib):
    rng = random.Random(11)
    for _ in range(48):
        key = Key(rng.randbytes(32))
        index = tuple(rng.choice((rng.randbytes(rng.randrange(6)), "yat"[:rng.randrange(1, 4)], rng.randrange(1 << 32)))
                      for _ in range(rng.randrange(1, 3)))
        if len(E.head(key, index)) > E.HEAD_MAX:
            continue
        leaves = rng.randrange(1, 5)
        first = rng.choice((rng.randrange(40), rng.randrange(1 << 40), E.LEAVES_MAX - leaves))
        got = _fill(lib, key, index, first, leaves)
        assert got == _blocks(key, index, first, leaves)
        if first < 40:
            assert got == key.stream(index, 192 * (first + leaves))[192 * first:]


def test_three_compressions_a_leaf(lib):
    lib.sxs_compressions(1)
    _fill(lib, KEY, ("y", 4095), 1000, 17)
    assert lib.sxs_compressions(1) == 3 * 17


@pytest.mark.parametrize("index", [(bytes(26),), ("salts", "t", 2**32 - 1)], ids=["104-byte head", "three-part index"])
def test_a_two_compression_head_is_refused(lib, index):
    assert len(E.head(KEY, index)) > E.HEAD_MAX
    out = np.full((2, E.LEAF_BYTES), 0xA5, np.uint8)
    with pytest.raises(E.Refused, match="REFUSED_HEAD") as e:
        E.fill(lib, out, KEY, index, 0)
    assert e.value.name == "REFUSED_HEAD" and (out == 0xA5).all()


@pytest.mark.parametrize("first", [E.LEAVES_MAX - 1, 1 << 64])
def test_a_counter_past_two_to_the_64_is_refused(lib, first):
    out = np.full((2, E.LEAF_BYTES), 0xA5, np.uint8)
    with pytest.raises(E.Refused, match="REFUSED_RANGE"):
        E.fill(lib, out, KEY, "t", first)
    assert (out == 0xA5).all()


def test_the_last_leaf_ends_at_counter_two_to_the_64_less_two(lib):
    """3 (2^64 - 1) / 3 = 2^64 - 1: the leaf after the last would need counters 2^64 - 1 .. 2^64 + 1."""
    got = _fill(lib, KEY, "t", E.LEAVES_MAX - 1, 1)
    assert got[-64:] == KEY.block("t", (1 << 64) - 2) and got == _blocks(KEY, "t", E.LEAVES_MAX - 1, 1)


@pytest.mark.parametrize("out", [np.zeros((2, 191), np.uint8), np.zeros((2, 192), np.int8), np.zeros(384, np.uint8),
                                 np.zeros((2, 384), np.uint8)[:, ::2]])
def test_out_is_contiguous_uint8_rows_of_a_salt(lib, out):
    with pytest.raises(ValueError):
        E.fill(lib, out, KEY, "t", 0)


def test_the_loaders_own_build_agrees(lib, tmp_path):
    own = E.host_lib(tmp_path)
    assert _fill(own, KEY, ("a", 7), 3, 4) == _fill(lib, KEY, ("a", 7), 3, 4) == _blocks(KEY, ("a", 7), 3, 4)


def test_a_one_part_str_index_fixes_every_blocks_message_length(lib):
    """At ``"s"`` every block hashes tag (28) ‖ key (32) ‖ frames (19) ‖ counter (8) = 87 bytes, whatever the key and the
    counter; a fill given another length is refused before it writes."""
    assert E.message_bytes("s") == 87 == len(E.head(KEY, "s")) + 8 == len(E.head(Key(bytes(32)), "s")) + 8
    assert _fill(lib, KEY, "s", 5, 2) == _blocks(KEY, "s", 5, 2)
    out = np.full((2, E.LEAF_BYTES), 0xA5, np.uint8)
    E.fill(lib, out, KEY, "s", 0, length=87)
    assert out.tobytes() == _blocks(KEY, "s", 0, 2)
    out[:] = 0xA5
    with pytest.raises(E.Refused, match="REFUSED_LENGTH"):
        E.fill(lib, out, KEY, ("s", 0), 0, length=87)
    assert (out == 0xA5).all()


def _torch_cuda():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    return torch if torch.cuda.is_available() and cuda_build.gpu_name() is not None else None


TORCH = _torch_cuda()
card = pytest.mark.skipif(TORCH is None, reason="no CUDA device, torch or nvcc")


@pytest.fixture(scope="module")
def device() -> ctypes.CDLL:
    return E.device_lib(BUILD, cuda_build.arch(cast(str, cuda_build.gpu_name())))


def _card_fill(device: ctypes.CDLL, key: Key, index, first: int, leaves: int) -> bytes:
    torch = cast(Any, TORCH)
    out = torch.full((leaves, E.LEAF_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
    E.fill(device, out, key, index, first, stream=torch.cuda.current_stream().cuda_stream)
    return out.cpu().numpy().tobytes()


@card
@pytest.mark.parametrize("leaves, first", [(1, 0), (32, 7), (4096, 123456), (753664, 0), (32, E.LEAVES_MAX - 32)],
                         ids=["one", "decode-call", "prefill-call", "1b-tiles-at-8192", "last"])
def test_the_card_writes_the_host_twins_bytes(lib, device, leaves, first):
    """The sizes Pearl+ fills: a decode call's rows, a prefill's, Llama 3.2 1B's tiles of a pass at m = 8192, and the last
    leaves a key's stream holds."""
    assert _card_fill(device, KEY, ("y", 4095), first, leaves) == _fill(lib, KEY, ("y", 4095), first, leaves)


@card
@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: c["domain"])
def test_the_card_gives_the_pinned_stream_vectors(device, case):
    key = derive(bytes.fromhex(case["source"]), case["domain"], _context(case["context"]))
    s = case["stream"]
    n = len(s["bytes"]) // 2
    index = tuple(s["index"]) if isinstance(s["index"], list) else s["index"]
    assert _card_fill(device, key, index, 0, -(-n // E.LEAF_BYTES))[:n].hex() == s["bytes"]


@card
def test_the_card_refuses_a_capturing_stream(device):
    torch = cast(Any, TORCH)
    out = torch.full((32, E.LEAF_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
    g, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(side), pytest.raises(E.Refused, match="REFUSED_CAPTURE"), torch.cuda.graph(g, stream=side):
        E.fill(device, out, KEY, "a", 0, stream=side.cuda_stream)
    torch.cuda.synchronize()
    assert bool((out == 0xA5).all())


@card
def test_the_card_refuses_a_two_compression_head(device):
    torch = cast(Any, TORCH)
    out = torch.full((2, E.LEAF_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
    with pytest.raises(E.Refused, match="REFUSED_HEAD"):
        E.fill(device, out, KEY, ("salts", "t", 2**32 - 1), 0)
    assert bool((out == 0xA5).all())
