"""The MUFU primitives of `verity.circuits.ops.mufu`: ids, descriptor digests, outputs, and the tables they read.

These primitives were registered by the vLLM integration (`verity_vllm.definitions.registry.prims`) under the same ids, and its
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

from verity.circuits.codec import definition_digest, encode_program, program_digest
from verity.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits import library as L
from verity.circuits.ops import mufu as M

PINNED = {  # id: (definition digest, one-call program digest, outputs fingerprint)
    "DivFullRcp_v1": ("8a8820d688d6c921025399b6c3aaa7cc86ee9be02c5df23061c24afbc689b2ec795f73da842826c1923d2dc4b38d4dfd6642fe2c2a3ebefa3bb347fa2b855dac",
                      "23962f21958a26f54115030f1262cf0bbc0654ea924e473b2a67691e09181a1e289d1864ffcf9ca308c8c0d8d9851322c89f79b97f0ccb2341fbd6791abb4de9",
                      "9204f41f6990ee4cb3d8830bbee15a2c89dbec96c344fcce25f733cf9aab4e2a"),
    "DivFullScaleA_v1": ("e1f74245807daa7c2cf2971e0374ee81d06aa478f33cc0c4eef624448ecaee0d53c4adc1f0e1dd10ce9495d141ba01b80f8a4f7a1820adb52ae0886d5b89d1c9",
                         "42d6e42d390e30cfe34279bef018fdc196aa097914949b054021f24a4e17fed7e342c022138fe75fb0fcf42a463f46b795221c182cafdc2fe987683ee808dcdd",
                         "0d4a18ec9a743eece75f31ceac06da273f9f5e0eb3af8c0e2386e37a7c444649"),
    "Fa2InvSum_v1": ("8492eb683d244d906f70cb11eb015da799b2db093be0e810a5ce045846f793dc1440af283609209a5cd6581bc675dfe7f78984efb2978d47d11cac3491a05008",
                     "838b0875258aae63cba06986f3843486757284efcf643215f7a0ddb647703eb7af31ad84ae033b057ceacd5808ea19d33f3848f7d68d6a33cab8477d44abbfa7",
                     "7594edc01d89c0d325ae0e4731b97082c19b1112b20b35967feaad481bc0f2ea"),
    "MufuEx2Ftz_v1": ("5ea21fc594f9428f47516441228855fa6e80f95d479b88e61c8c7d400438d524c5257e7ea7c1db427e59c0c0fde3ea11ad078f9aec8e4f7f54086ec427f4f9ac",
                      "69ec589a04ca185c01a72ddd8e4adf7cedd788fb8676c2d47de4362f7a535a661db761594c03204b54db1be1c922707c72b48c343d55ff46127c2789b481ddb7",
                      "59ad46a236c432fd763eb0320096b60c52687950a6af27337c02063c8dc23f88"),
    "MufuRcpFtz_v1": ("1fcac46a7b6b82aea06524707371658f8c6372fa810c9f2140b3221c50aea93e71c329e1790adeadceff2abea0a582d9b3260c4eda0c35a13668cca711578fb2",
                      "9752944f7b407e03efb86c3d69e2e512b7b1a7507bef57b8abb24096b1be9eebd24825b6670abd8f40b7c26a6f03c6615b6aa50c2c3ea6f3e80bb53a7f4c79bc",
                      "717ad7815295ed009d1cb6ec296ef5320ed4999c5b040bba37deaf5bbc41e52e"),
    "MufuSqrtFtz_v1": ("b209c0da558f7bf996179b518bc2ea02a5980def393a19ca76d3b8c5b966da51bf2b9cf61492009cfac11edb9fc0e4f443975181931c94e50c6e5d4f4c967f83",
                       "8e84dc3c5fcc60b6d89ccc603e6016e1b6533113e0f16185dbaad9960ff171bbad4572a9b01188d49127d0d9bcf7663d385f57df964e8ffc7e078e13a94598d1",
                       "b8351402782d183785b8cc3e1222a9901e55bd355600114cd3e0e0d7ecca6df2"),
    "MufuTanh_v1": ("5a5f80812fb5924444aeada23b0e8964a352848cc26e32813532422c2eb4392aec212f612877ebabf0f146a9c4153af73e4f05c42afbd3be02940700dd0549a3",
                    "09e185831daa86cc7b41e46fa0a51cd8946dad93036ebb5c641fbd94d6a5088a84d914389eb09af9151ce6bbc898ad4cf2e55dbd8159cfeda2c10541ae76422b",
                    "ad38928341bc38849f8a7f223f27c1589823f9fab8a31ac06244280eb248287b"),
    "RsqrtApprox_v1": ("90008efd80c64dd52b67c19d59bc8dcd185e2fe709d7ab5a04a96710d917f2abcd0b0bc829f4d069b82fefcf244ca816ef381addc92773c2ce243b29bd28759f",
                       "e7569d024574aa0a854b1309ab8a2c7d8e1e3f91f60057e6f9d82f54c7d958e84b44456a5d69c1b265e8ad2c29cebafde4ebcdfba157544ac9dfb493c0edab99",
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
    assert (definition_digest(desc["definitions"][fid], desc), program_digest(desc)) == PINNED[fid][:2]


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
        unmeasured = []
        for ws in itertools.product(_TABLE_WORDS, repeat=len(p.params)):
            try:
                p.evaluate(*ws)
            except M.MufuTanhUnmeasured as e:
                unmeasured.append(e)
        assert len(unmeasured) < len(_TABLE_WORDS) ** len(p.params), f"{p.id}: every word is in an unmeasured region"
        assert {(n, L.LIBRARY) for n in read} == set(p.tables), p.id
