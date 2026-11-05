"""The MUFU primitives of `verity.ml.circuits.ops.mufu`: ids, descriptor digests, outputs, and the tables they read.

These primitives were registered by the vLLM integration (`verity.ml.definitions.registry.prims`) under the same ids, and its
stored Programs cite them. `PINNED` holds, per primitive, the digests recorded from that registration: the Definition's
descriptor digest, the digest of a program making one call to it, and a fingerprint of its outputs over edge and seeded random
encodings. A primitive whose digest or outputs change is a new version of it, not an edit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import random

import numpy as np
import pytest

from verity.ml.circuits.metadata import definition_digest
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits import library as L
from verity.ml.circuits.ops import mufu as M

PINNED = {  # id: (definition digest, one-call program digest, outputs fingerprint)
    "DivFullRcp_v1": ("8a8820d688d6c921025399b6c3aaa7cc86ee9be02c5df23061c24afbc689b2ec795f73da842826c1923d2dc4b38d4dfd6642fe2c2a3ebefa3bb347fa2b855dac",
                      "df119b4e5672d5ba34d42341d985fc19c044fc142d28f40ff135e675a94c6f3db0ec2fe8bd8124b40922358e573004e74bf4e10eff87337692d2f1b780f3f864",
                      "9204f41f6990ee4cb3d8830bbee15a2c89dbec96c344fcce25f733cf9aab4e2a"),
    "DivFullScaleA_v1": ("e1f74245807daa7c2cf2971e0374ee81d06aa478f33cc0c4eef624448ecaee0d53c4adc1f0e1dd10ce9495d141ba01b80f8a4f7a1820adb52ae0886d5b89d1c9",
                         "516841fb4303306c6e59a28b4f5635e23bb194595985492d0b1108caef0e9321a9de6fdefba9e64d11f64c4dfa66e9f717d947ba73e3857b27da2e36130af43f",
                         "0d4a18ec9a743eece75f31ceac06da273f9f5e0eb3af8c0e2386e37a7c444649"),
    "Fa2InvSum_v1": ("8492eb683d244d906f70cb11eb015da799b2db093be0e810a5ce045846f793dc1440af283609209a5cd6581bc675dfe7f78984efb2978d47d11cac3491a05008",
                     "9d27b9b4415c7f6cb4370aab862d0591c30e837a5ec4ba9069c8765c6decb6e69cdd909dcb4ec563bafc255c6efb971d8dd8eb2fbc3bbdc9cbf189459b162d8d",
                     "7594edc01d89c0d325ae0e4731b97082c19b1112b20b35967feaad481bc0f2ea"),
    "MufuEx2Ftz_v1": ("5ea21fc594f9428f47516441228855fa6e80f95d479b88e61c8c7d400438d524c5257e7ea7c1db427e59c0c0fde3ea11ad078f9aec8e4f7f54086ec427f4f9ac",
                      "44695eb4be8e6b828eb8ac5e248393bb64692ef7b4525eb2e57a01da7139a8dfff079669eb1e5d2979ba30fb01f4e516fe05c74c367161a3026df001ccd95b3c",
                      "59ad46a236c432fd763eb0320096b60c52687950a6af27337c02063c8dc23f88"),
    "MufuRcpFtz_v1": ("1fcac46a7b6b82aea06524707371658f8c6372fa810c9f2140b3221c50aea93e71c329e1790adeadceff2abea0a582d9b3260c4eda0c35a13668cca711578fb2",
                      "271bdb3de3af806339656cd4af1a6d8df0e01935473b8e4f3d80d708443cef22e03a0435612b782d32d3e638457dbd85ab6c42b1edae28c034d3c40a0d066e35",
                      "717ad7815295ed009d1cb6ec296ef5320ed4999c5b040bba37deaf5bbc41e52e"),
    "MufuSqrtFtz_v1": ("b209c0da558f7bf996179b518bc2ea02a5980def393a19ca76d3b8c5b966da51bf2b9cf61492009cfac11edb9fc0e4f443975181931c94e50c6e5d4f4c967f83",
                       "e5513b59bbfd2c05f164bdd5911e9c96eff8342703c4cc35b99ca201c826825a80478f0eb38e755983bacac2a5915125d2d27488ec3c8398e12c23a42d3c5279",
                       "b8351402782d183785b8cc3e1222a9901e55bd355600114cd3e0e0d7ecca6df2"),
    "MufuTanh_v1": ("5a5f80812fb5924444aeada23b0e8964a352848cc26e32813532422c2eb4392aec212f612877ebabf0f146a9c4153af73e4f05c42afbd3be02940700dd0549a3",
                    "c860784cfc0fdc969e65e10275f5f25d33b55307370a7ba00475352c0def0e60386c4a4262f6e4db4bab109e4551592708d7844dd2420128fa792f1d3d0f1cbd",
                    "ad38928341bc38849f8a7f223f27c1589823f9fab8a31ac06244280eb248287b"),
    "RsqrtApprox_v1": ("90008efd80c64dd52b67c19d59bc8dcd185e2fe709d7ab5a04a96710d917f2abcd0b0bc829f4d069b82fefcf244ca816ef381addc92773c2ce243b29bd28759f",
                       "94f2c5d28e59fcdd63aa55b2e9a0b7e58fa9c4e03eff1d55e32bba46d98a83e0586ab6e6c4c16831f3c18d95fc27ea408cd2a8e9d08498b14e84fd3bcc9b6858",
                       "68f760b8f0b114f1a456b2e102a4f2de2272e3f13fe473ce351357da987710ac"),
}

F32_EDGES = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000,
             0xBF800000, 0x3F800001, 0x3EFFFFFF, 0x4B000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
             0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0x33800000, 0x0C000000]
# table ends and the parity boundary; div.full's 2^126 and 2^-126 thresholds; ex2's |x| = 30 and 2^-23 rules and its
# exponent range; tanh's identity and saturation rules
MUFU_EDGES = F32_EDGES + [0x3F7FFFFF, 0x3FFFFFFF, 0x40000000, 0x3F000000, 0x7E800000, 0x7E800001, 0xFE800001, 0x7E7FFFFF,
                          0x00400000, 0x80400000, 0x41F00000, 0x41EFFFFF, 0xC1F00000, 0x43000000, 0xC3000000, 0x42FE0000,
                          0xC3150000, 0xC3160000, 0x3B800000, 0x3B800001, 0x40FFFFFF, 0x41000000, 0xBB800001, 0xC0FFFFFF,
                          0x34000000, 0x33FFFFFF, 0xB4000000]


def _prim(fid: str) -> PrimitiveDefinition:
    return getattr(M, fid.rsplit("_v", 1)[0])


def _inputs(p: PrimitiveDefinition, n_random: int = 4000) -> list[tuple[int, ...]]:
    """The edge grid, then `n_random` seeded random encodings."""
    widths = [t.w for _, t in p.params]
    rng = random.Random(f"mufu-move/{p.id}")
    return list(itertools.product(*[MUFU_EDGES] * len(widths))) + [tuple(rng.getrandbits(w) for w in widths)
                                                                    for _ in range(n_random)]


def _one_call_program(p: PrimitiveDefinition) -> Program:
    sig = p.signature()
    probe = CompositeDefinition(f"Probe{p.name}", 1, (), lambda S: sig, lambda B, S, *args: B.call(p, *args), register=False)
    return Program(bind(probe))


def test_registered_once_under_the_pinned_ids() -> None:
    public = {n for n in M.__all__ if isinstance(getattr(M, n), PrimitiveDefinition)}
    assert {f"{n}_v1" for n in public} == set(PINNED)
    for fid in PINNED:
        assert _prim(fid).id == fid and REGISTRY[fid] is _prim(fid)


@pytest.mark.parametrize("fid", sorted(PINNED))
def test_descriptor_and_program_digests(fid: str) -> None:
    desc = encode_program(_one_call_program(_prim(fid)))
    assert (definition_digest(desc, fid), program_digest(desc)) == PINNED[fid][:2]


@pytest.mark.parametrize("fid", sorted(PINNED))
def test_outputs(fid: str) -> None:
    p = _prim(fid)
    h = hashlib.sha256()
    for x in _inputs(p):
        h.update(json.dumps([list(x), p.evaluate(*x)]).encode())
    assert h.hexdigest() == PINNED[fid][2]


# ---- tables ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", M.DELTA_TABLES)
def test_each_table_is_its_library_table(name: str) -> None:
    entry = L.table(name)
    words = M.table(name)
    assert M.table(entry.sha512) is words and words.dtype == np.uint32 and words.shape == (entry.words,)
    assert not words.flags.writeable
    raw = words.astype("<u4").tobytes()
    assert (hashlib.sha256(raw).hexdigest(), hashlib.sha512(raw).hexdigest()) == (entry.sha256, entry.sha512)
    assert not int(np.bitwise_or.reduce(words)) >> entry.value_bits


def test_a_table_that_is_not_the_library_one_does_not_load(monkeypatch, tmp_path) -> None:
    with open(M.table_path("rcp"), "rb") as f:
        delta = bytearray(lzma.decompress(f.read()))
    delta[12345] ^= 1
    (tmp_path / "mufu_rcp_delta_int8.xz").write_bytes(lzma.compress(bytes(delta), preset=0))
    (tmp_path / "mufu_ex2_delta_int8.xz").write_bytes(lzma.compress(bytes(delta[:-1]), preset=0))
    monkeypatch.setattr(M, "TABLE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="not library table"):
        M._decoded.__wrapped__("rcp")
    with pytest.raises(ValueError, match="differences"):
        M._decoded.__wrapped__("ex2")
    with pytest.raises(KeyError):
        M.table("tanh_mufu")


def test_tanh_reproduces_the_manifest_hardware_sample() -> None:
    manifest = M.mufu_tanh_table_manifest()
    pairs = manifest["sample"]["pairs"]
    assert len(pairs) == manifest["sample"]["n"] == 512
    assert all(M.MufuTanh.evaluate(int(x, 16)) == int(y, 16) for x, y in pairs)
    cov = M.mufu_tanh_coverage()
    assert [s["odd_symmetry_verified"] for s in cov["shards"]] == [True] and "error" not in cov


def test_a_tanh_table_that_is_not_the_pinned_one_does_not_load(monkeypatch) -> None:
    monkeypatch.setattr(M, "MUFU_TANH_TABLE_MANIFEST_SHA512", "0" * 128)
    M.mufu_tanh_shards.cache_clear()
    with pytest.raises(M.MufuTanhUnmeasured, match="pinned"):
        M.MufuTanh.evaluate(0x3F000000)
    assert M.mufu_tanh_coverage()["shards"] == []


# ---- semantics ------------------------------------------------------------------------------------------------------------

def test_stated_semantics() -> None:
    one, inf, nan = 0x3F800000, 0x7F800000, 0x7FFFFFFF
    ex2, rcp, inv = M.MufuEx2Ftz.evaluate, M.MufuRcpFtz.evaluate, M.Fa2InvSum.evaluate
    assert [ex2(x) for x in (0, 0x80000001, 0xFF800000, inf, 0xFFC00001)] == [one, one, 0, inf, nan]
    assert [ex2(x) for x in (0x4E800000, 0xCE800000, 0x43000000, 0xC3160000)] == [inf, 0, inf, 0]  # |x| >= 2^30; 2^128; 2^-150
    assert [ex2(x) for x in (0x41F00000, 0x40000000, 0xC0000000)] == [0x4E800000, 0x40800000, 0x3E800000]
    assert [rcp(x) for x in (one, 0, 0x80000000, inf, 0xFF800000, 0x7F7FFFFF)] == [one, inf, 0xFF800000, 0, 0x80000000, 0]
    assert [inv(x) for x in (0x80000000, 0x7FC00000, 0x40000000)] == [one, one, rcp(0x40000000)]
    sqrt, rsqrt = M.MufuSqrtFtz.evaluate, M.RsqrtApprox.evaluate
    assert [sqrt(x) for x in (0x40800000, 0x80000001, 0xBF800000, inf, 0xFF800000)] == [0x40000000, 0x80000000, nan, inf, nan]
    assert [rsqrt(x) for x in (0x40800000, 0, 0x80000000, inf, 0x80000001)] == [0x3F000000, inf, 0xFF800000, 0, nan]
    assert rsqrt(1) == M.mufu_rsq_bits(0x01000000, M.table("rsq")) + (12 << 23)  # 2^-149: scaled to 2^-125, then 2^12
    div, scale = M.DivFullRcp.evaluate, M.DivFullScaleA.evaluate
    assert [div(x) for x in (0x40000000, 0x7F000000, 0x00400000, inf, 0)] == [rcp(0x40000000), rcp(0x7E000000), rcp(0x0C000000), 0, inf]
    assert [scale(one, b) for b in (0x7F000000, 0x00000001, 0x40000000, 0x7FC00000)] == [0x3E800000, 0x4B800000, one, one]


_TABLE_WORDS = (0x00000000, 0x00400000, 0x3A83126F, 0x3F400000, 0x3F800000, 0x40400000, 0x7E800000, 0x7F800000, 0xBF000000)


def test_each_primitive_declares_the_tables_its_evaluator_reads(monkeypatch):
    """A table an evaluator reads by index is no gate, so only `PrimitiveDefinition.tables` shows it to the guard: every
    primitive here declares exactly the library tables it reads on words of every region."""
    read: list[str] = []
    row, shards = M._row, M.mufu_tanh_shards
    monkeypatch.setattr(M, "_row", lambda name: (read.append(name), row(name))[1])
    monkeypatch.setattr(M, "mufu_tanh_shards", lambda: (read.append("tanh_mufu"), shards())[1])
    prims = [d for d in REGISTRY.defs.values() if isinstance(d, PrimitiveDefinition) and d.evaluate.__module__ == M.__name__]
    assert len(prims) >= 8
    for p in prims:
        read.clear()
        for ws in itertools.product(_TABLE_WORDS, repeat=len(p.params)):
            try:
                p.evaluate(*ws)
            except M.MufuTanhUnmeasured:
                pass
        assert {(n, L.LIBRARY) for n in read} == set(p.tables), p.id
