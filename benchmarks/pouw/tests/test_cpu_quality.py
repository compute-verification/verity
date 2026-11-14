"""`pearl_c_vllm/cpu_quality.py`, the CPU emulation of ``pearl-c-h100-v1-h3``'s served y: the parts it states as exact are
`pearl_c`'s bit for bit (the cast, the line rule, the row statistics), and one tile's served y stays within the device
atoms' rounding of the exact forms'."""
from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C  # noqa: E402
from verity.protocols.accounting.pouw.prover.schemes import pearl_kw as P  # noqa: E402

HARNESS = Path(__file__).resolve().parents[1] / "pearl_c_vllm"
sys.path.insert(0, str(HARNESS))


def _load(name: str):
    key = f"pearl_c_vllm_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HARNESS / f"{name}.py")
        mod = sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


Q = _load("cpu_quality")


def _f32(words):
    return torch.tensor([struct.unpack("<f", struct.pack("<I", w))[0] for w in words], dtype=torch.float32)


def test_e4m3_is_pearls_cast():
    gen = torch.Generator().manual_seed(1)
    mags = torch.cat([torch.rand(20000, generator=gen) * 500, torch.rand(20000, generator=gen) * 2.0 ** -5,
                      torch.tensor([448.0, 464.0, 463.99, 480.0, 1e6, 2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -10, 0.0])])
    x = torch.cat([mags, -mags]).to(torch.float32)
    want = torch.tensor([P.fp8_to_f32(P.f32_to_fp8(float(v))) for v in x], dtype=torch.float32)
    assert torch.equal(Q.e4m3(x), want)


def test_line_codes_are_sample_lines():
    seed = C.S512Seed(hashlib.sha512(b"lines").digest()[:32])
    for norm in (Q.E_NORM, Q.F_NORM):
        raw, want = [], []
        for side, factor, line in ((0, 0, 0), (1, 0, 7), (0, 1, 3), (1, 1, 2047)):
            raw.append(list(C.s512("line", C._s512_line_key(seed) + bytes([side, factor]) + line.to_bytes(4, "little"))))
            want.append([P.fp8_to_f32(c) for c in C.sample_line(seed, side, factor, line, norm)])
        assert torch.equal(Q.line_codes(torch.tensor(raw, dtype=torch.uint8), norm), torch.tensor(want, dtype=torch.float32))


@pytest.mark.parametrize("k", [1024, 2048, 8192])
def test_row_stats_are_s5_stats(k):
    gen = torch.Generator().manual_seed(k)
    x = (torch.randn(6, k, generator=gen) * torch.tensor([1e-3, 0.02, 1.0, 7.0, 1e-30, 0.0])[:, None]).to(torch.bfloat16)
    x[2, 5] = 40.0
    x32 = x.to(torch.float32)
    s, rho, alpha, beta = Q.row_stats(x32)
    for i, row in enumerate(Q._words(x32)):
        assert torch.equal(torch.stack([s[i], rho[i], alpha[i], beta[i]]), _f32(C.s5_stats(row))), i


def _e4m3_codes(gen, shape):
    codes = torch.randint(0, 256, shape, generator=gen)
    codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes)
    codes[..., :3] = torch.tensor([0, 1, 0x80 | 3])
    return codes, torch.tensor([[P.fp8_to_f32(int(c)) for c in row] for row in codes.reshape(-1, shape[-1])],
                               dtype=torch.float32).reshape(shape)


def test_atom_is_hopper_e4m3_k32():
    from verity.circuits.silicon.models import HOPPER_E4M3_K32 as H
    gen = torch.Generator().manual_seed(3)
    lanes = 3000
    ca, a = _e4m3_codes(gen, (lanes, 32))
    cb, b = _e4m3_codes(gen, (lanes, 32))
    acc = torch.randn(lanes, generator=gen) * torch.exp2(torch.randint(-20, 12, (lanes,), generator=gen).float())
    acc[:50] = 0.0
    acc[50:60] = -0.0
    got = Q.atom(acc, a, Q.code_exponent(a), b, Q.code_exponent(b))
    want = _f32([H.step(w, ca[i].tolist(), cb[i].tolist()) for i, w in enumerate(Q._words(acc[None])[0])])
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_chain_is_pearl_c_chain():
    gen = torch.Generator().manual_seed(4)
    ca, a = _e4m3_codes(gen, (5, 512))
    cb, b = _e4m3_codes(gen, (512, 3))
    want = _f32([C.chain(ca[i].tolist(), cb[:, j].tolist()) for i in range(5) for j in range(3)]).view(5, 3)
    assert torch.equal(Q.chain(a, b).view(torch.int32), want.view(torch.int32))


def test_tile_within_the_products_sums():
    """A 16 x 64 tile at k = 1,024 of a heavy-tailed A and a BF16 W: α_A, α_B, A′ and B̃ the exact forms', and the served y
    within 2e-3 of the exact y's RMS (C̃ and the peel are FP32 sums, not the device's truncating adder), with the same error
    against x·Wᵀ to 1%."""
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(16, 1024, generator=gen) * torch.rand(16, 1, generator=gen) * 3).to(torch.bfloat16)
    x[:, 3] *= 20
    w = (torch.randn(64, 1024, generator=gen) * 0.02).to(torch.bfloat16)
    import fast_tile
    undo = fast_tile.install(C.PearlC(forming="v1"))
    try:
        r = Q.reference_tile(x, w, b"test")
    finally:
        undo()
    assert r["alpha_a_equal"] == 1.0 and r["alpha_b_equal"] == 1.0, r
    assert r["a_codes_equal"] == 1.0 and r["b_codes_equal"] == 1.0, r
    assert r["y_emu_vs_ref_rel_rms"] < 2e-3, r
    assert abs(r["y_emu_vs_exact_rel_rms"] / r["y_ref_vs_exact_rel_rms"] - 1) < 0.01, r


def test_join_tiles_the_windows_in_order():
    def part(lo, hi):
        return {"mode": "pearl", "windows": 4, "windows_sha256": "d", "seed": 1, "atoms": "exact", "model": "m",
                "range": [lo, hi], "nll": float(hi - lo), "tokens": 10 * (hi - lo), "seconds": 1.0,
                "per_window_nll": [1.0 * i for i in range(lo, hi)], "top1": list(range(lo, hi)),
                "excluded_a_rows": {"head": {"floor": lo, "live": 0, "rows": 1}}}
    doc = Q.join([part(2, 4), part(0, 2)])
    assert doc["per_window_nll"] == [0.0, 1.0, 2.0, 3.0] and doc["top1"] == [0, 1, 2, 3] and doc["tokens"] == 40
    assert doc["excluded_a_rows"]["head"] == {"floor": 2, "live": 0, "rows": 2}
    with pytest.raises(SystemExit):
        Q.join([part(0, 2), part(3, 4)])
