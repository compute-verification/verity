"""The certified stream `stream/v1` (`stream.py`): its AES against the standards' known answers, its layout and digest
against BLAKE3 itself, its openings, its pinned vectors, and the host generator (`stream_host.cpp`) and `stream.cu`'s
addressing held to it.  The device is checked on the node (`run_stream.sh`: `bwstream --mode gates`, then every window's
digest against the host's reference)."""

from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import stream  # noqa: E402
from verity.ml.crypto import blake3  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402

VECTORS = json.loads((HERE / "tests" / "stream_vectors.json").read_text())
CU = (HERE / "stream.cu").read_text()


def test_aes_is_fips_197():
    """FIPS 197 appendix C.1, and the key expansion of appendix A.1."""
    rk = stream.expand_key(bytes.fromhex("000102030405060708090a0b0c0d0e0f"))
    out = stream.encrypt_block(rk, bytes.fromhex("00112233445566778899aabbccddeeff"))
    assert out.hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    rk = stream.expand_key(bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"))
    assert rk[10].hex() == "d014f9a8c9ee2589e13f0cc8b6630ca6"


def test_ctr_is_sp_800_38a():
    """SP 800-38A F.5.1, CTR-AES128.Encrypt: the keystream is ciphertext XOR plaintext."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
                       "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
    ct = bytes.fromhex("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff"
                       "5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee")
    ks = stream.ctr(key, int.from_bytes(bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), "big"), 4)
    assert bytes(a ^ b for a, b in zip(pt, ks)) == ct


def test_ctr_wraps_and_reads_at_any_offset():
    key = bytes(range(16))
    assert stream.ctr(key, (1 << 128) - 1, 2)[16:] == stream.ctr(key, 0, 1)
    whole = stream.stream_bytes(key, 64 * 16)
    assert stream.stream_bytes(key, 128, 512) == whole[512:640]
    with pytest.raises(ValueError):
        stream.stream_bytes(key, 8)


def test_stream_key_is_the_challengers_derivation():
    want = derive(stream.SOURCE, "verity/bandwidth/stream/v1",
                  {"root": stream.ROOT, "salt": stream.SALT, "nonce": stream.NONCE}).stream("aes-128", 16)
    assert stream.stream_key(stream.SOURCE, stream.ROOT, stream.SALT, stream.NONCE) == want
    assert stream.stream_key(stream.SOURCE, stream.ROOT, stream.SALT, bytes(32)) != want


def test_permutation_moves_every_aes_block_once_and_chunks_regenerate():
    key = bytes(range(16, 32))
    s = stream.stream_bytes(key, 3 * stream.GROUP_BYTES)
    m = stream.permuted(s)
    blocks = lambda b: sorted(b[i:i + 16] for i in range(0, len(b), 16))  # noqa: E731
    assert len(m) == len(s) and blocks(m) == blocks(s)
    for i in range(len(m) // stream.CHUNK):
        assert stream.chunk(key, i) == m[stream.CHUNK * i:stream.CHUNK * (i + 1)]
    # the device's coalescing: at load j of compression g, a group's 64 chunks read 64 consecutive AES blocks of S
    for q in range(3):
        for g in range(16):
            for j in range(4):
                got = [stream.counter(64 * q + c, 4 * g + j) for c in range(64)]
                assert got == list(range(4096 * q + 64 * (4 * g + j), 4096 * q + 64 * (4 * g + j) + 64))
    with pytest.raises(ValueError):
        stream.permuted(s[:stream.GROUP_BYTES])


@pytest.mark.parametrize("groups", [2, 3, 5])
def test_digest_is_keyed_blake3_of_the_permuted_stream(groups):
    key, k_a = bytes(range(16)), stream.absorb_key(bytes(32))
    s = stream.stream_bytes(key, groups * stream.GROUP_BYTES)
    d, lv = stream.absorb(k_a, s)
    assert d == blake3.keyed(k_a, stream.permuted(s))
    assert [len(x) for x in lv] == list(stream.sizes(groups * stream.GROUP))
    nodes = stream.group_nodes(lv)
    assert len(nodes) == groups
    # the group nodes are the roots of each group's own 64 chunks, so the final launch needs only them
    for q, node in enumerate(nodes):
        cvs = lv[0][64 * q:64 * (q + 1)]
        while len(cvs) > 1:
            cvs = [stream.parent_cv(k_a, cvs[i], cvs[i + 1]) for i in range(0, len(cvs), 2)]
        assert cvs[0] == node


def test_absorb_key_is_bound_to_the_seed():
    assert stream.absorb_key(bytes(32)) != stream.absorb_key(bytes(31) + b"\x01")
    assert len(stream.absorb_key(bytes(32))) == 32


@pytest.mark.parametrize("groups", [2, 3])
def test_openings_accept_the_stream_and_reject_anything_else(groups):
    key, k_a = bytes(range(16)), stream.absorb_key(bytes(range(32)))
    chunks = groups * stream.GROUP
    d, lv = stream.absorb(k_a, stream.stream_bytes(key, chunks * stream.CHUNK))
    for i in (0, 1, 63, 64, chunks - 2, chunks - 1):
        c, p = stream.chunk(key, i), stream.path(lv, i)
        assert stream.verify(k_a, d, chunks, i, c, p)
        assert not stream.verify(k_a, d, chunks, i, bytes(c[:-1]) + bytes([c[-1] ^ 1]), p)
        assert not stream.verify(k_a, d, chunks, i ^ 1, c, p)
        assert not stream.verify(k_a, d, chunks, i, c, p[:-1])
        assert not stream.verify(k_a, d, chunks, i, c, (*p, p[0]))
        assert not stream.verify(stream.absorb_key(bytes(32)), d, chunks, i, c, p)
    assert not stream.verify(k_a, d, chunks, chunks, stream.chunk(key, 0), stream.path(lv, 0))


def test_a_skipped_chunk_fails_its_opening():
    """A device that never received chunk 5 holds some other bytes there: its digest is not the stream's, and its own
    tree's opening of chunk 5 fails against the chunk the verifier regenerates."""
    key, k_a = bytes(range(16)), stream.absorb_key(bytes(range(32)))
    n = 2 * stream.GROUP_BYTES
    s = bytearray(stream.stream_bytes(key, n))
    d, _ = stream.absorb(k_a, bytes(s))
    for k in range(stream.WORDS):
        b = stream.counter(5, k)
        s[16 * b:16 * (b + 1)] = bytes(16)
    d2, lv2 = stream.absorb(k_a, bytes(s))
    assert d2 != d
    assert not stream.verify(k_a, d2, 2 * stream.GROUP, 5, stream.chunk(key, 5), stream.path(lv2, 5))
    assert stream.verify(k_a, d2, 2 * stream.GROUP, 6, stream.chunk(key, 6), stream.path(lv2, 6))


def test_samples_are_the_windows_draws():
    window = derive(b"v", "verity/bandwidth/window/v1", {"n": b"\x00"})
    got = stream.samples(window, 128, 50)
    assert got == tuple(window.uniform(128, ("stream", t)) for t in range(50))
    assert all(0 <= x < 128 for x in got) and len(set(got)) > 1


def test_fold_is_each_chunks_xor_in_the_streams_order():
    key = bytes(range(16))
    s = stream.stream_bytes(key, 2 * stream.GROUP_BYTES)
    f = stream.fold(s)
    assert len(f) == 16 * len(s) // stream.CHUNK
    for i in (0, 1, 64, 127):
        acc = bytes(16)
        for k in range(stream.WORDS):
            block = s[stream.CHUNK * i + 16 * k:stream.CHUNK * i + 16 * (k + 1)]
            acc = bytes(a ^ b for a, b in zip(acc, block))
        assert f[16 * i:16 * (i + 1)] == acc
    assert stream.fold(s[:stream.CHUNK]) == f[:16]
    with pytest.raises(ValueError):
        stream.fold(s[:stream.CHUNK - 16])


def test_a_wrong_byte_moves_its_folds_byte():
    """A byte the device guessed instead of fetching reaches exactly one fold byte, at its offset in its chunk's block:
    the bound's `⌈X/64⌉` fold bytes over `X` bytes withheld."""
    key = bytes(range(16))
    s = stream.stream_bytes(key, 2 * stream.GROUP_BYTES)
    f = stream.fold(s)
    for at in (0, 5, 1023, 70 * 1024 + 17):
        t = bytearray(s)
        t[at] ^= 0x5A
        g = stream.fold(bytes(t))
        assert [k for k in range(len(f)) if f[k] != g[k]] == [16 * (at // stream.CHUNK) + at % 16]
        assert g[16 * (at // stream.CHUNK) + at % 16] ^ f[16 * (at // stream.CHUNK) + at % 16] == 0x5A


def test_fold_kernels_read_a_chunk_as_the_reference():
    """`fold_chunks` and `fold_live`: lane `l` of chunk `c`'s warp reads uint4 `64c + l` and `64c + 32 + l`, the
    warp's XOR is the chunk's fold, written to `out[c]`."""
    for kernel in ("fold_chunks", "fold_live"):
        body = CU[CU.index(f"__global__ void {kernel}("):]
        body = body[:body.index("\n}\n")]
        assert "p + (c0 + k) * 64 + lane" in body and "p + (c0 + k) * 64 + 32 + lane" in body
        assert "out[c0 + k] = v" in body


def test_pinned_vectors_are_the_reference():
    assert stream.vectors() == VECTORS


def test_stream_cu_constants_are_the_references():
    assert re.search(r"constexpr int GROUP = (\d+), AES = (\d+);", CU).groups() == (str(stream.GROUP), str(stream.AES))
    assert f"constexpr uint64_t CHUNK = {stream.CHUNK}," in CU
    domain = stream.ABSORB_DOMAIN[:-1].decode()
    assert stream.ABSORB_DOMAIN == domain.encode() + b"\x00" and domain == stream.STREAM_DOMAIN
    assert f'ABSORB_DOMAIN[] = "{domain}"' in CU
    assert f'ABSORB_NAME[] = "{stream.ABSORB_NAME}"' in CU


def test_absorb_kernel_and_host_addressing_are_the_layout():
    """`absorb_kernel` reads uint4 `q·GROUP_BYTES/16 + c + (4g + j)·GROUP` for chunk 64q + c at load j of compression g;
    `counter_of`, which the host reference and the verifier's openings read, is `stream.counter`."""
    assert "S + q * (GROUP_BYTES / 16) + c;" in CU
    assert "cur[j] = base[j * GROUP];" in CU and "base[(4 * (g + 1) + j) * GROUP]" in CU
    assert "(chunk / GROUP) * (GROUP_BYTES / AES) + (uint64_t)k * GROUP + chunk % GROUP" in CU
    for q in (0, 1, 7):
        for c in (0, 1, 63):
            for g in range(16):
                for j in range(4):
                    u4 = q * (stream.GROUP_BYTES // 16) + c + (4 * g + j) * stream.GROUP
                    assert u4 == stream.counter(64 * q + c, 4 * g + j)


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    m = re.search(r"^flags\s*:\s*(.*)$", text, re.M)
    return set(m.group(1).split()) if m else set()


@pytest.fixture(scope="module")
def host_lib(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ to build stream_host.cpp")
    if "aes" not in _cpu_flags():
        pytest.skip("this CPU has no AES-NI")
    so = tmp_path_factory.mktemp("stream_host") / "libstreamhost.so"
    subprocess.run([gxx, "-O2", "-mavx512f", "-mavx512bw", "-mvaes", "-maes", "-fPIC", "-shared",
                    str(HERE / "stream_host.cpp"), "-o", str(so), "-ldl"], check=True)
    return ctypes.CDLL(str(so))


@pytest.mark.parametrize("which", ["low", "carry"])
def test_host_generators_are_the_keystream(host_lib, which):
    key = bytes.fromhex(VECTORS["key_s"])
    first, want = VECTORS[f"keystream_{which}_counter"], bytes.fromhex(VECTORS[f"keystream_{which}"])
    blocks = len(want) // 16
    rk = ctypes.create_string_buffer(176)
    host_lib.sg_expand(key, rk)
    assert rk.raw == b"".join(stream.expand_key(key))
    impls = ["aesni"] + (["vaes", "vaes_nt"] if {"vaes", "avx512f", "avx512bw"} <= _cpu_flags() else [])
    for impl in impls:
        raw = ctypes.create_string_buffer(16 * blocks + 64)
        at = (-ctypes.addressof(raw)) % 64
        getattr(host_lib, f"sg_ctr_{impl}")(rk, ctypes.c_uint64(first), ctypes.byref(raw, at), ctypes.c_uint64(blocks))
        assert raw.raw[at:at + 16 * blocks] == want, impl
    one = ctypes.create_string_buffer(16)  # the verifier's single blocks
    for b in (0, 7, blocks - 1):
        host_lib.sg_ctr_aesni(rk, ctypes.c_uint64(first + b), one, ctypes.c_uint64(1))
        assert one.raw == want[16 * b:16 * (b + 1)]
    out = ctypes.create_string_buffer(16 * blocks)
    if host_lib.sg_ctr_openssl(key, ctypes.c_uint64(first), out, ctypes.c_uint64(blocks)) == 0:
        assert out.raw == want
