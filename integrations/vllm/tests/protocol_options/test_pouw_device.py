"""`pouw_device`: the host build of `pouw_cuda/ncp2.cuh` commits what `pouw_native` commits, and the gate names the first unit
a record gets wrong."""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import os

import numpy as np
import pytest
from verity_vllm.protocol_options import pouw_circuit as PC
from verity_vllm.protocol_options import pouw_device as PD
from verity_vllm.protocol_options import pouw_native as PN

#: Under `check` (`tools/verity/check/suites.py` sets VERITY_SUITE_ROOT) the CPU build always runs: without a compiler these tests
#: fail there instead of skipping. nvcc and a GPU stay optional (their tests skip by name).
IN_CHECK = bool(os.environ.get("VERITY_SUITE_ROOT"))
pytestmark = pytest.mark.skipif(PD.compiler() is None and not IN_CHECK, reason="needs a C++ compiler")


def ptn(n: int) -> bytes:
    return bytes(i % 251 for i in range(n))


def test_the_sha2s_and_shake256_are_hashlibs_across_block_boundaries():
    rng = np.random.default_rng(1)
    msgs = [rng.bytes(n) for n in (0, 1, 27, 55, 56, 63, 64, 65, 111, 112, 127, 128, 129, 135, 136, 137, 271, 272, 1000)]
    assert PD.hash_many("sha256", msgs) == [hashlib.sha256(m).digest() for m in msgs]
    assert PD.hash_many("sha512", msgs) == [hashlib.sha512(m).digest() for m in msgs]
    for n in (32, 136, 137, 300):
        assert PD.hash_many("shake256", msgs, n) == [hashlib.shake_256(m).digest(n) for m in msgs]


@pytest.mark.parametrize("msg, ds, out, expect", [
    (b"", 0x1F, 32, "1e415f1c5983aff2169217277d17bb538cd945a397ddec541f1ce41af2c1b74c"),               # RFC 9861
    (ptn(17), 0x1F, 32, "9c97d036a3bac819db70ede0ca554ec6e4c2a1a4ffbfd9ec269ca6a111161233"),           # RFC 9861
    (b"\xff\xff\xff", 0x01, 32, "bf323f940494e88ee1c540fe660be8a0c93f43d15ec006998462fa994eed5dab"),   # RFC 9861
    (ptn(17 ** 2), 0x1F, 32, "96c77c279e0126f7fc07c9b07f5cdae1e0be60bdbe10620040e75d7223a624d2"),
    (b"\xff", 0x06, 32, "8ec9c66465ed0d4a6c35d13506718d687a25cb05c74cca1e42501abd83874a67"),
    (ptn(167), 0x1F, 32, "895e142c96269722e14958a4e74055b823472e3a10139241a1a76ec968a4d509"),
    (ptn(168), 0x1F, 32, "ed5bf22a6a67e3cfe1d1f974a9dee10da9da2fe264f55359ec56c16541ac5456"),
])
def test_turboshake128_is_rfc_9861s(msg, ds, out, expect):
    """The first three are RFC 9861's vectors; the rest, around the 168-byte rate, are pycryptodome's."""
    assert PD.hash_many("turboshake128", [msg], out, ds)[0].hex() == expect


def test_turboshake128_squeezes_past_its_rate():
    assert PD.hash_many("turboshake128", [b""], 200)[0][-32:].hex() == "3728a261f2a4be2d4ea83a3b8c3be12fed74555f2410f0d0aa56d0d63967f8e9"


@pytest.mark.parametrize("name, call", PD.gate_calls(), ids=[n for n, _c in PD.gate_calls()])
def test_the_host_build_commits_what_pouw_native_commits(name, call):
    assert PD.first_difference(PN.commit_call(0, "ncp-v2", call), PD.commit_call(0, "ncp-v2", call)) is None


def test_the_gate_calls_reach_the_dequantizations_rounding_and_special_values():
    calls = dict(PD.gate_calls())
    c = calls["z-past-2^24"]
    z = c.a.astype(np.int64) @ c.b.b.astype(np.int64).T
    assert (np.abs(z) > 1 << 24).any() and ((np.abs(z) > 1 << 24) & (z % 2 == 1)).any()
    e = calls["dequant-edges"]
    assert set(PD.F32_EDGES) <= set(e.s.tolist()) and set(PD.F32_EDGES) <= set(e.b.t.tolist())
    assert set(PD.BF16_EDGES) <= set(e.bias.tolist())


@pytest.fixture(scope="module")
def five_strips():
    call = dict(PD.gate_calls())["five-strips"]
    return call, PN.Native(call, "ncp-v2"), PN.commit_call(0, "ncp-v2", call)


@pytest.mark.parametrize("role, where", [("digest", (3,)), ("node", (2,)), ("key", ()), ("form_x", (4,)), ("tile", (2, 1)),
                                         ("dequant", (4, 2, 37))])
def test_the_gate_names_the_first_wrong_unit(five_strips, role, where):
    call, nat, ref = five_strips
    u = nat.index(role, *where)
    bad = PN.commit_call(0, "ncp-v2", call, tamper={(u, 0): int(nat.outputs(u)[0]) ^ 1}, side=ref.side)
    got = PD.first_difference(ref, bad)
    assert got.startswith(f"unit {u} ({role} {where}) has the wrong leaf"), got
    if role == "dequant":
        assert "y[66, 37]" in got and "from z, s, t, bias" in got


def test_the_gate_names_the_first_wrong_y_unit_of_the_weight_side(five_strips):
    call, _nat, ref = five_strips
    side = PN.commit_weight("ncp-v2", call, tamper={(1, 0): int(ref.side.opening(1)[0]) ^ 1})
    assert PD.first_difference(ref, PN.commit_call(0, "ncp-v2", call, side=side)) == "Y unit 1 of the weight side has the wrong leaf"


def test_a_build_refuses_a_weight_side_whose_y_leaves_are_not_its_own(five_strips):
    call, _nat, ref = five_strips
    side = PN.commit_weight("ncp-v2", call, tamper={(0, 5): int(ref.side.opening(0)[5]) ^ 1})
    with pytest.raises(ValueError, match="Y unit leaves are not the weight side's"):
        PD.commit_call(0, "ncp-v2", call, side=side)
    assert PD.first_difference(ref, PD.commit_call(0, "ncp-v2", call, side=ref.side)) is None


def test_the_gate_names_a_wrong_served_entry_and_a_wrong_inputs_digest(five_strips):
    _call, _nat, ref = five_strips
    y = ref.y.copy()
    y[5, 7] ^= 1
    served = dataclasses.replace(ref, y=y)
    assert PD.first_difference(ref, served) == f"served y[5, 7] is {int(y[5, 7]):#06x}, not {int(ref.y[5, 7]):#06x}"
    assert PD.first_difference(ref, dataclasses.replace(ref, inputs=bytes(32))) == "the inputs' digest differs"


def test_the_build_is_keyed_by_its_sources(tmp_path, monkeypatch):
    before = PD.source_digest()
    monkeypatch.setattr(PD, "CFLAGS", PD.CFLAGS + ("-DNCP2_KEYED",))
    assert PD.source_digest() != before


@pytest.fixture(scope="module")
def device():
    """The nvcc build, which compiles and loads without a GPU; the tests that run it need one."""
    if PD.nvcc() is None:
        pytest.skip("needs nvcc")
    return PD.device()


def _has_gpu(dll) -> bool:
    rate = ctypes.c_double(0)
    return dll.ncp2_ceiling(0, 1, 32, 1, ctypes.byref(rate)) == 0


def test_the_device_build_loads_and_without_a_gpu_reports_cuda_errors(device):
    if _has_gpu(device):
        pytest.skip("a CUDA device is present")
    with pytest.raises(ValueError, match="CUDA error"):
        PD.commit_call(0, "ncp-v2", dict(PD.gate_calls())["five-strips"], on="device")


@pytest.mark.parametrize("name, call", PD.gate_calls(), ids=[n for n, _c in PD.gate_calls()])
def test_the_device_build_commits_what_pouw_native_commits(device, name, call):
    if not _has_gpu(device):
        pytest.skip("needs a CUDA device")
    assert PD.first_difference(PN.commit_call(0, "ncp-v2", call), PD.commit_call(0, "ncp-v2", call, on="device")) is None
    msgs = [bytes(range(n % 256)) * (1 + n // 256) for n in (0, 27, 64, 136, 168, 300)]
    for kind in PD.HASHES:
        assert PD.hash_many(kind, msgs, 48, on="device") == PD.hash_many(kind, msgs, 48)


@pytest.mark.parametrize("name, call", PD.gate_calls(), ids=[n for n, _c in PD.gate_calls()])
def test_the_serving_kernels_functions_on_the_cpu_commit_what_pouw_native_commits(name, call):
    """`ncp2_fast.cuh`: the word-stream leaves, the tile hashed as it is computed, the aligned SHA-512 key, the block-wise
    noise and the register units tree, run on the CPU."""
    assert PD.first_difference(PN.commit_call(0, "ncp-v2", call), PD.commit_call(0, "ncp-v2", call, on="twin")) is None


@pytest.mark.parametrize("count", [1, 2, 3, 5, 16, 17, 1000])
def test_the_fast_units_tree_is_pouw_circuits(count):
    leaves = np.random.default_rng(count).bytes(32 * count)
    assert PD.units_root_twin(leaves) == PC.units_root(leaves)


def test_the_low_byte_variant_hashes_each_checkpoints_low_byte(five_strips):
    """The timed-only variant (cheap binding's row 4): the kernel's function on the CPU equals the reference built from
    `pouw_native`'s own tile words."""
    call, nat, _ref = five_strips
    b, _t = nat.side.read(0, call.shape.n)
    y, c0 = nat.side.block(b, 0)
    for rs, cs in ((0, 0), (2, 1), (4, 2)):
        words = nat.outputs(nat.index("tile", rs, cs))
        got = PD.tile_lowbyte_host(nat.x_strip(rs), y[16 * cs:16 * cs + 16], c0[16 * cs:16 * cs + 16], nat.steps)
        assert got == PD.lowbyte_leaf(words)
        assert got == PD.lowbyte_leaf(words ^ 0x100)                     # only the low byte is bound
        assert got != PD.lowbyte_leaf(words ^ np.eye(1, len(words), 5, dtype=np.uint32).ravel())


@pytest.fixture(scope="module")
def fused():
    if PD.nvcc() is None:
        pytest.skip("needs nvcc")
    return PD.fused()


def test_the_fused_build_loads_and_without_a_gpu_reports_cuda_errors(fused, device):
    if _has_gpu(device):
        pytest.skip("a CUDA device is present")
    with pytest.raises(ValueError, match="CUDA error"):
        PD.commit_call(0, "ncp-v2", dict(PD.gate_calls())["five-strips"], on="fused")


@pytest.mark.parametrize("name, call", PD.gate_calls(), ids=[n for n, _c in PD.gate_calls()])
def test_the_fused_kernel_commits_what_pouw_native_commits(fused, device, name, call):
    if not _has_gpu(device):
        pytest.skip("needs a CUDA device")
    ref = PN.commit_call(0, "ncp-v2", call)
    assert PD.first_difference(ref, PD.commit_call(0, "ncp-v2", call, on="fused")) is None
    assert PD.lowbyte_difference(ref, PD.commit_call(0, "ncp-v2", call, on="fused", flags=PD.HASH | PD.LOWBYTE)) is None


def test_the_low_byte_check_names_a_wrong_leaf(five_strips):
    _call, nat, ref = five_strips
    u = nat.index("tile", 1, 1)
    good = ref.leaves[:32 * nat.base["tile"]] + b"".join(PD.lowbyte_leaf(ref.opening(v)) for v in range(nat.base["tile"], nat.base["dequant"])) \
        + ref.leaves[32 * nat.base["dequant"]:]
    rec = lambda leaves: dataclasses.replace(ref, leaves=leaves, root=PC.units_root(leaves))
    assert PD.lowbyte_difference(ref, rec(good)) is None
    bad = good[:32 * u] + bytes(32) + good[32 * u + 32:]
    assert PD.lowbyte_difference(ref, rec(bad)).startswith(f"unit {u} (tile (1, 1)) has the wrong low-byte leaf")
