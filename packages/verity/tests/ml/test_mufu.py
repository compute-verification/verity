"""The MUFU primitives of `verity.ml.mufu`: ids, descriptor digests, outputs, and the tables they read.

These primitives were registered by the vLLM integration (`verity_vllm.program.registry.prims`) under the same ids, and its
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

from verity.ir.annotations import definition_digest
from verity.ir.codec import encode_program, program_digest
from verity.ir.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.program import Program
from verity.ml import library as L
from verity.ml import mufu as M
from verity.proofs import reduce_mufu as R

PINNED = {  # id: (definition digest, one-call program digest, outputs fingerprint)
    "DivFullRcp_v1": ("3b96ef979e97f6ae5d5757069f8fa574fb5aa62c6a2c88ab99ff07da4dbdb170",
                      "cbbddebe9ffb811b3f75c643e2ed1aa47e480b725585ab6500600052050b9875",
                      "9204f41f6990ee4cb3d8830bbee15a2c89dbec96c344fcce25f733cf9aab4e2a"),
    "DivFullScaleA_v1": ("305108267bace43e67d7cf6f66762d841ca4b285e0e4cd005ade80c7f08cd389",
                         "e9011a0e9ea857e0107b846cd3e49c4826ae3b01afcdcfc88dbc5411f8f3d67f",
                         "0d4a18ec9a743eece75f31ceac06da273f9f5e0eb3af8c0e2386e37a7c444649"),
    "Fa2InvSum_v1": ("9bd65175eefd4da57e12704238f7d2fdf6168cbe112c4de45808aaeda1b28a2b",
                     "9a73a24e0ab894f718b076be71534d319a6850f37f038d2ffde4f2b593518fca",
                     "7594edc01d89c0d325ae0e4731b97082c19b1112b20b35967feaad481bc0f2ea"),
    "MufuEx2Ftz_v1": ("bfd71d72d99d6f6e07832227f6f2598746c8a1199e36405f84572abdf44172d1",
                      "cd51675e86870a40512d73170595862b2f213ad325ce220ba991e08381444dcd",
                      "59ad46a236c432fd763eb0320096b60c52687950a6af27337c02063c8dc23f88"),
    "MufuRcpFtz_v1": ("9abb264e2aba44bd54cdea24f9de150de1c4c829f137a494c8a375ff2ce734d0",
                      "5078f4f68a88bf4373a9f28087d83f75a1c66abf2f4340e6e1e56e7c2e661e55",
                      "717ad7815295ed009d1cb6ec296ef5320ed4999c5b040bba37deaf5bbc41e52e"),
    "MufuSqrtFtz_v1": ("b5d461a3f22a16a7763b857c9a35da1b28e5d8f4e1e69feffbd29b97f45d7080",
                       "ccf4ee11ebbad50bde7663e268e40b69a1cca966329552e9f2500bc88677ca4e",
                       "b8351402782d183785b8cc3e1222a9901e55bd355600114cd3e0e0d7ecca6df2"),
    "MufuTanh_v1": ("035f917671f67bfaf821cb44a94b89bb1a62aa357c8291fa5f7dc827f60379e7",
                    "72bee5acd2684cf2ec1cff48e01e7e3fd7fe014442e93527bffddb1bf0ce7f87",
                    "ad38928341bc38849f8a7f223f27c1589823f9fab8a31ac06244280eb248287b"),
    "RsqrtApprox_v1": ("446141811fd3d0a35036ef13f616bdb5df8b5f5442948c81767023edd3dbc8b7",
                       "41d612f5078898a4f75858ef90429d407b012768f7907097dfc5c07f720baf55",
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
    monkeypatch.setattr(M, "MUFU_TANH_TABLE_MANIFEST_SHA256", "0" * 64)
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


def test_the_sp1_guest_mirror_agrees() -> None:
    """`verity.proofs.reduce_mufu` (the SP1 guest's integer-only semantics) against the primitives, on the fingerprint inputs."""
    sqrt_t, rsq_t, rcp_t = M.table("sqrt"), M.table("rsq"), M.table("rcp")

    def row(t, i: int) -> int:
        return (i << 32) | int(t[i])

    for (x,) in _inputs(M.MufuSqrtFtz):
        assert R.mufu_sqrt_tab(x, row(sqrt_t, R.mufu_sqrt_index(x))) == M.MufuSqrtFtz.evaluate(x), hex(x)
        assert R.rsqrt_approx(x, row(rsq_t, R.rsqrt_approx_index(x))) == M.RsqrtApprox.evaluate(x), hex(x)
        assert R.div_full_rcp(x, row(rcp_t, R.div_full_rcp_index(x))) == M.DivFullRcp.evaluate(x), hex(x)
    for a, b in _inputs(M.DivFullScaleA):
        assert R.div_full_scale_a(a, b) == M.DivFullScaleA.evaluate(a, b), (hex(a), hex(b))
