"""covert_witness's E formats (`prepare.py`): the sparse format's round trip, its two index forms and its refusals, and every
format's rebuild cost, which is at least what te_core's counter sees the decode compute."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from te_core.flops import Counter

_spec = importlib.util.spec_from_file_location("cw_prepare_formats", Path(__file__).resolve().parents[2] / "prepare.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)

N_BIG = 1 << 17


def theta() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {"w": torch.randn(64, 48, generator=g).to(torch.bfloat16), "g": torch.randn(48, generator=g),
            "odd": torch.randn(13, generator=g), "big": torch.randn(N_BIG, generator=g).to(torch.float16),
            "v": torch.randn(7, 5, generator=g)}


def counted(fn) -> int:
    counter = Counter(digests=False)
    with counter:
        fn()
    return counter.setup.macs


def test_sparse_decode_adds_each_value_to_theta_in_fp32_and_leaves_the_rest():
    th = theta()
    idx, val = torch.tensor([5, 0, 3071, 100]), torch.tensor([0.5, -1e-3, 2.0, 3e-2])
    blob = prepare.encode_sparse(th, {"w": (idx, val), "g": (torch.tensor([7]), torch.tensor([1.25]))})
    assert prepare.format_of(blob) == "sparse"
    out = prepare.decode(th, blob, device="cpu")
    want = th["w"].reshape(-1).clone()
    want[idx] = (want[idx].float() + val.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert out["w"].shape == th["w"].shape and prepare.same_bits(out["w"].reshape(-1), want)
    assert out["g"][7] == th["g"][7] + 1.25 and prepare.same_bits(out["g"][:7], th["g"][:7])
    assert all(prepare.same_bits(out[k], th[k]) for k in ("odd", "big"))
    assert prepare.rebuild_macs(th, blob) == 5


def test_varints_round_trip_across_byte_boundaries():
    v = np.array([0, 1, 127, 128, 16383, 16384, 2**21, 2**35, 2**62], dtype=np.uint64)
    b = prepare._varints(v)
    assert len(b) == 1 + 1 + 1 + 2 + 2 + 3 + 4 + 6 + 9
    assert np.array_equal(prepare._unvarints(b, len(v)), v)


def test_the_index_is_varint_gaps_or_a_bitmap_whichever_is_smaller():
    th = theta()
    for idx, form in ((torch.tensor([0, 127, 128, 16383, 16384, N_BIG - 1]), "gaps"), (torch.arange(0, N_BIG, 3), "bitmap")):
        blob = prepare.encode_sparse(th, {"big": (idx, torch.ones(idx.numel()))}, "f16")
        _, header, _ = prepare._unpack(blob)
        ((_, f, nnz, nb),) = header["tensors"]
        gaps = len(prepare._varints(np.diff(idx.numpy(), prepend=-1) - 1))
        assert (f, nnz, nb) == (form, idx.numel(), min(gaps, N_BIG // 8))
        out = prepare.decode(th, blob, device="cpu")["big"]
        assert torch.equal(torch.nonzero(out.view(torch.int16) != th["big"].view(torch.int16)).reshape(-1), idx)


def test_sparse_deltas_encode_a_model_exactly_at_f32():
    th = theta()
    model = {k: v.clone() for k, v in th.items()}
    model["w"].view(-1)[[3, 9, 2000]] = torch.tensor([1.5, -0.25, 3.0]).to(torch.bfloat16)
    model["big"][123456] = 7.0
    d = prepare.sparse_deltas(th, model)
    assert sorted(d) == ["big", "w"] and d["w"][0].tolist() == [3, 9, 2000]
    out = prepare.decode(th, prepare.encode_sparse(th, d, "f32"), device="cpu")
    assert all(prepare.same_bits(out[k], model[k]) for k in th)


def test_the_sparse_encoder_refuses_repeated_or_outside_indices():
    th = theta()
    for idx in ([3, 3], [48], [-1]):
        with pytest.raises(ValueError, match="distinct and inside"):
            prepare.encode_sparse(th, {"g": (torch.tensor(idx), torch.zeros(len(idx)))})


def _blob(entries: list, payload: bytes, dtype: str = "bf16") -> bytes:
    return prepare._pack(b"CWSPAR1\n", {"dtype": dtype, "tensors": entries}, payload)


ONE = b"\x00\x3f"  # 0.5 in bf16
REFUSED = [
    ([["g", "gaps", 1, 1]], bytes([48]) + ONE, "past the tensor"),
    ([["g", "gaps", 1, 2]], b"\x81\x00" + ONE, "overlong"),
    ([["g", "gaps", 1, 1]], b"\x80" + ONE, "count of varints"),
    ([["g", "gaps", 2, 1]], b"\x00" + ONE + ONE, "count of varints"),
    ([["odd", "bitmap", 2, 2]], b"\x01\x80" + ONE + ONE, "past the tensor"),
    ([["odd", "bitmap", 1, 2]], b"\x03\x00" + ONE, "count of entries"),
    ([["odd", "bitmap", 1, 1]], b"\x01" + ONE, "wrong length"),
    ([["w", "gaps", 1, 1], ["g", "gaps", 1, 1]], b"\x00" + ONE + b"\x00" + ONE, "increasing name order"),
    ([["g", "gaps", 1, 1], ["g", "gaps", 1, 1]], b"\x00" + ONE + b"\x00" + ONE, "increasing name order"),
    ([["nope", "gaps", 1, 1]], b"\x00" + ONE, "not one of θ's"),
    ([["g", "gaps", 1]], b"\x00" + ONE, "malformed tensor entries"),
    ([["g", "gaps", 0, 0]], b"", "counts are malformed"),
    ([["g", "runs", 1, 1]], b"\x00" + ONE, "unknown index form"),
    ([["g", "gaps", 1, 1]], b"\x00" + ONE[:1], "too short"),
    ([["g", "gaps", 1, 1]], b"\x00" + ONE + b"\x00", "trailing bytes"),
]


@pytest.mark.parametrize("entries,payload,why", REFUSED)
def test_the_sparse_decoder_refuses_malformed_encodings(entries, payload, why):
    th = theta()
    assert prepare.decode(th, _blob([["g", "gaps", 1, 1]], b"\x00" + ONE), device="cpu")["g"][0] == th["g"][0] + 0.5
    with pytest.raises(ValueError, match=why):
        prepare.decode(th, _blob(entries, payload), device="cpu")


def factors(r: int, d_out: int, d_in: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(r, d_in, generator=g), torch.randn(d_out, r, generator=g) * 1e-2


@pytest.mark.parametrize("bits", [4, 8])
def test_lora2_is_lora_on_its_quantized_factors(bits):
    th = theta()
    a, b = factors(5, 64, 48)
    ah, bh = prepare.lora2_factors(a, b, bits)
    qmax = 2 ** (bits - 1) - 1
    for x, xh in ((a, ah), (b.t(), bh.t())):
        s = x.abs().amax(dim=1, keepdim=True) / qmax
        assert torch.all((x - xh).abs() <= s * (0.5 + qmax * 2.0 ** -10))
        assert torch.all((xh / s).round().abs() <= qmax)
    blob = prepare.encode_lora2({"w": (a, b, 0.5)}, bits, "f32", full={"g": th["g"]})
    assert prepare.format_of(blob) == "lora2"
    out = prepare.decode(th, blob, device="cpu")
    lora1 = prepare.decode(th, prepare.encode_lora({"w": (ah, bh, 0.5)}, "f32", full={"g": th["g"]}), device="cpu")
    assert all(prepare.same_bits(out[k], lora1[k]) for k in th)


def test_lora2_magnitude_sets_each_output_rows_norm():
    th = theta()
    a, b = factors(3, 64, 48)
    m = th["w"].float().norm(dim=1) * torch.linspace(0.5, 2.0, 64)
    out = prepare.decode(th, prepare.encode_lora2({"w": (a, b, 2.0)}, 8, "f32", magnitudes={"w": m}), device="cpu")
    ah, bh = prepare.lora2_factors(a, b, 8)
    assert torch.equal(out["w"], prepare.lora_merge(th["w"], ah, bh, 2.0, m))
    assert torch.allclose(out["w"].float().norm(dim=1), m, rtol=1e-2)
    assert all(prepare.same_bits(out[k], th[k]) for k in th if k != "w")


@pytest.mark.parametrize("bits,ints", [(4, 8 + 11), (8, 15 + 21)])
def test_lora2_stores_the_integers_and_one_f16_scale_per_rank_component(bits, ints):
    a, b = factors(3, 7, 5)
    _, header, payload = prepare._unpack(prepare.encode_lora2({"v": (a, b, 1.0)}, bits))
    assert header["factors"] == [["v", 3, 1.0, bits, False]] and len(payload) == ints + 2 * 2 * 3


def test_lora2_bits_are_per_weight_and_its_encoder_refuses_others():
    a, b = factors(2, 7, 5)
    _, header, _ = prepare._unpack(prepare.encode_lora2({"v": (a, b, 1.0), "w": (*factors(2, 64, 48), 1.0)}, {"v": 4, "w": 8}))
    assert [f[3] for f in header["factors"]] == [4, 8]
    with pytest.raises(ValueError, match="bits must be"):
        prepare.encode_lora2({"v": (a, b, 1.0)}, 2)
    with pytest.raises(ValueError, match="a magnitude needs"):
        prepare.encode_lora2({"v": (a, b, 1.0)}, magnitudes={"w": torch.ones(64)})


def _lora2(entries: list, payload: bytes, full: list = (), dtype: str = "bf16") -> bytes:
    return prepare._pack(b"CWLORA2\n", {"dtype": dtype, "factors": entries, "full": list(full)}, payload)


H1 = b"\x00\x3c"  # 1.0 in f16
V8, V4 = bytes(5) + H1 + bytes(7) + H1, bytes(3) + H1 + bytes(4) + H1
G = bytes(2 * 48)
REFUSED_LORA2 = [
    (prepare._pack(b"CWLORA2\n", {"dtype": "bf16", "factors": []}, b""), "malformed header"),
    (_lora2([["v", 1, 1.0, 8, False]], V8, dtype="f8"), "malformed header"),
    (_lora2([["v", 1, 1.0, 8]], V8), "malformed factor entries"),
    (_lora2([["v", 0, 1.0, 8, False]], b""), "malformed factor entries"),
    (_lora2([["v", 1, 1.0, 5, False]], V8), "malformed factor entries"),
    (_lora2([["v", 1, float("nan"), 8, False]], V8), "malformed factor entries"),
    (_lora2([["v", 1, 1.0, 8, 0]], V8), "malformed factor entries"),
    (_lora2([["w", 1, 1.0, 8, False], ["v", 1, 1.0, 8, False]], V8 + V8), "strictly increasing"),
    (_lora2([["v", 1, 1.0, 8, False], ["v", 1, 1.0, 8, False]], V8 + V8), "strictly increasing"),
    (_lora2([], G + G, full=["g", "g"]), "strictly increasing"),
    (_lora2([["v", 1, 1.0, 8, False]], V8 + bytes(70), full=["v"]), "both in factors and in full"),
    (_lora2([["nope", 1, 1.0, 8, False]], V8), "not one of θ's"),
    (_lora2([["g", 1, 1.0, 8, False]], V8), "not a matrix"),
    (_lora2([["v", 1, 1.0, 8, False]], V8[:-1]), "too short"),
    (_lora2([["v", 1, 1.0, 8, False]], V8 + b"\x00"), "trailing bytes"),
    (_lora2([["v", 1, 1.0, 4, False]], bytes([0, 0, 0x10]) + V4[3:]), "padding nibble"),
    (_lora2([["z", 1, 1.0, 8, True]], bytes(4) + H1 + bytes(3) + H1 + bytes(6)), "norm 0"),
]


@pytest.mark.parametrize("blob,why", REFUSED_LORA2)
def test_the_lora2_decoder_refuses_malformed_encodings(blob, why):
    th = theta() | {"z": torch.zeros(3, 4)}
    assert prepare.same_bits(prepare.decode(th, _lora2([["v", 1, 1.0, 4, False]], V4), device="cpu")["v"], th["v"])
    with pytest.raises(ValueError, match=why):
        prepare.decode(th, blob, device="cpu")


def test_rebuild_macs_per_format_bound_what_the_counter_sees():
    th = theta()
    meta = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in th.items()}
    a, b = torch.randn(4, 48), torch.randn(64, 4)
    m = th["w"].float().norm(dim=1)
    blobs = {
        "lora": (prepare.encode_lora({"w": (a, b, 0.5)}, full={"g": th["g"]}), (4 + 2) * 64 * 48),
        "lora2": (prepare.encode_lora2({"w": (a, b, 0.5)}, 4, magnitudes={"w": m}, full={"g": th["g"]}),
                  4 * (64 + 48) + (4 + 2) * 64 * 48 + 2 * (64 * 48 + 64)),
        "seedlog": (prepare.encode_seedlog(["w", "g"], 1, [0.1, 0.2, 0.3], "f32"), 3 * (64 * 48 + 48)),
        "sparse": (prepare.encode_sparse(th, {"big": (torch.arange(0, N_BIG, 5), torch.ones(N_BIG // 5 + 1))}), N_BIG // 5 + 1),
    }
    assert prepare.rebuild_macs(th, b"CWFULL1\n") == 0
    for fmt, (blob, macs) in blobs.items():
        assert prepare.format_of(blob) == fmt
        assert prepare.rebuild_macs(th, blob) == prepare.rebuild_macs(meta, blob) == macs
        assert counted(lambda blob=blob: prepare.decode(th, blob, device="cpu")) <= macs, fmt
