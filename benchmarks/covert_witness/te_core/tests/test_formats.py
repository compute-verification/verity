"""covert_witness's E formats (`prepare.py`): the sparse, lora2, quant and model formats' round trips, layouts and
refusals, every format's rebuild cost, which is at least what te_core's counter sees the decode compute, and data v3's
training windows across its parts."""
import importlib.util
import json
import lzma
import math
import struct
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


def qarray(shape, bits: int, group: int, seed: int = 2) -> "prepare.QArray":
    g = torch.Generator().manual_seed(seed)
    n = math.prod(shape)
    cb = (torch.arange(2 ** bits) - 2 ** (bits - 1)).to(torch.float16)
    sc = (torch.rand(n // group, generator=g) * 1e-2).to(torch.float16)
    return prepare.QArray(cb, sc, torch.randint(0, 2 ** bits, (n,), generator=g).to(torch.uint8), bits, group, tuple(shape))


def qvalues(q) -> torch.Tensor:
    return (q.codebook.float()[q.codes.long()].view(-1, q.group) * q.scales.float()[:, None]).view(q.shape)


def test_quant_delta_adds_its_codes_values_to_theta_in_fp32():
    th = theta()
    q, qg = qarray((64, 48), 3, 16), qarray((48,), 1, 48, seed=3)
    blob = prepare.encode_quant({"w": {"kind": "delta", "q": q}, "g": {"kind": "delta", "q": qg}}, th)
    assert prepare.format_of(blob) == "quant"
    out = prepare.decode(th, blob, device="cpu")
    assert prepare.same_bits(out["w"], (th["w"].float() + qvalues(q)).to(torch.bfloat16))
    assert prepare.same_bits(out["g"], th["g"] + qvalues(qg))
    assert all(prepare.same_bits(out[k], th[k]) for k in ("odd", "big", "v"))


def test_quant_lora_is_lora_on_its_dequantized_factors_and_full_is_the_tensor():
    th = theta()
    qa, qb = qarray((5, 48), 4, 8), qarray((64, 5), 2, 5, seed=4)
    v = torch.randn(7, 5)
    out = prepare.decode(th, prepare.encode_quant({"w": {"kind": "lora", "a": qa, "b": qb, "scale": 0.5},
                                                   "v": {"kind": "full", "t": v}}, th), device="cpu")
    lora1 = prepare.decode(th, prepare.encode_lora({"w": (qvalues(qa), qvalues(qb), 0.5)}, "f32", full={"v": v}),
                           device="cpu")
    assert all(prepare.same_bits(out[k], lora1[k]) for k in th)


@pytest.mark.parametrize("bits", [1, 2, 3, 5, 7, 8])
def test_quant_codes_are_packed_bits_wide_least_significant_first(bits):
    th = theta()
    q = qarray((13,), bits, 13)
    inner = lzma.decompress(prepare.encode_quant({"odd": {"kind": "delta", "q": q}}, th)[8:])
    (n,) = struct.unpack("<I", inner[:4])
    assert json.loads(inner[4:4 + n]) == {"t": [{"n": "odd", "k": "delta", "b": bits, "g": 13}]}
    codes = inner[4 + n + 2 * 2 ** bits + 2:]
    assert len(codes) == math.ceil(13 * bits / 8)
    stream = int.from_bytes(codes, "little")
    assert [(stream >> (i * bits)) & (2 ** bits - 1) for i in range(13)] == q.codes.tolist()
    assert stream >> (13 * bits) == 0


def test_the_quant_encoder_refuses_codes_that_dont_fit():
    th = theta()
    q = qarray((48,), 2, 16)
    for bad, why in ((q._replace(codes=q.codes + 4), "don't fit"), (q._replace(group=5), "don't fit"),
                     (q._replace(codebook=q.codebook[:3]), "don't fit"), (q._replace(shape=(47,)), "a delta of shape"),
                     (q._replace(scales=q.scales * float("inf")), "finite")):
        with pytest.raises(ValueError, match=why):
            prepare.encode_quant({"g": {"kind": "delta", "q": bad}}, th)
    with pytest.raises(ValueError, match="factors of shapes"):
        prepare.encode_quant({"g": {"kind": "lora", "a": q, "b": q, "scale": 1.0}}, th)


def _quant(entries, payload: bytes, header=None, tail: bytes = b"") -> bytes:
    h = json.dumps(header if header is not None else {"t": entries}, separators=(",", ":")).encode()
    return b"CWQUAN1\n" + lzma.compress(struct.pack("<I", len(h)) + h + payload, **prepare.QUANT_XZ) + tail


D1 = b"\x00\x00\x00\x3c" + b"\x00\x3c" * 6  # a 1-bit codebook {0, 1} and 6 scales of 1.0
OK = ([{"n": "g", "k": "delta", "b": 1, "g": 8}], D1 + bytes(6))
REFUSED_QUANT = [
    (_quant(*OK, header=[]), "malformed entries"),
    (_quant(*OK, header={"t": OK[0], "x": 1}), "malformed entries"),
    (_quant([{"n": "g", "k": "sparse", "b": 1, "g": 8}], OK[1]), "malformed entries"),
    (_quant([{"n": "g", "k": "delta", "b": 1, "g": 8, "x": 0}], OK[1]), "malformed entries"),
    (_quant([{"n": "w", "k": "full"}, {"n": "g", "k": "full"}], b""), "strictly increasing"),
    (_quant([{"n": "g", "k": "full"}, {"n": "g", "k": "full"}], b""), "strictly increasing"),
    (_quant([{"n": "nope", "k": "full"}], b""), "not one of θ's"),
    (_quant([{"n": "i", "k": "delta", "b": 1, "g": 1}], b""), "float tensors"),
    (_quant([{"n": "g", "k": "delta", "b": 0, "g": 8}], OK[1]), "bits or group"),
    (_quant([{"n": "g", "k": "delta", "b": 9, "g": 8}], OK[1]), "bits or group"),
    (_quant([{"n": "g", "k": "delta", "b": True, "g": 8}], OK[1]), "bits or group"),
    (_quant([{"n": "g", "k": "delta", "b": 1, "g": 7}], OK[1]), "bits or group"),
    (_quant([{"n": "g", "k": "lora", "r": 1, "s": 1.0, "A": {"b": 1, "g": 1}, "B": {"b": 1, "g": 1}}], b""), "rank or scale"),
    (_quant([{"n": "v", "k": "lora", "r": 6, "s": 1.0, "A": {"b": 1, "g": 1}, "B": {"b": 1, "g": 1}}], b""), "rank or scale"),
    (_quant([{"n": "v", "k": "lora", "r": 1, "s": float("nan"), "A": {"b": 1, "g": 1}, "B": {"b": 1, "g": 1}}], b""),
     "rank or scale"),
    (_quant([{"n": "v", "k": "lora", "r": 1, "s": 1.0, "A": {"b": 1, "g": 1, "x": 0}, "B": {"b": 1, "g": 1}}], b""),
     "bits or group"),
    (_quant(OK[0], b"\x00\x7c" + D1[2:] + bytes(6)), "finite"),
    (_quant(OK[0], D1 + bytes(5)), "too short"),
    (_quant(OK[0], D1 + bytes(7)), "trailing bytes"),
    (_quant(OK[0], OK[1], tail=b"\x00"), "one xz stream"),
    (_quant(*OK)[:-1], "one xz stream"),
    (b"CWQUAN1\n" + bytes(16), "not an xz stream"),
    (b"CWQUAN1\n" + lzma.compress(bytes(4 << 20), **prepare.QUANT_XZ), "one xz stream"),
    (b"CWQUAN1\n" + lzma.compress(b"\xff\xff\xff\x00", **prepare.QUANT_XZ), "too short"),
]


@pytest.mark.parametrize("blob,why", REFUSED_QUANT, ids=[w for _, w in REFUSED_QUANT])
def test_the_quant_decoder_refuses_malformed_encodings(blob, why):
    th = theta() | {"i": torch.zeros(3, dtype=torch.int32)}
    assert prepare.same_bits(prepare.decode(th, _quant(*OK), device="cpu")["g"], th["g"])
    with pytest.raises(ValueError, match=why):
        prepare.decode(th, blob, device="cpu")


def test_a_quant_padding_bit_is_refused():
    th = theta()
    blob = _quant([{"n": "odd", "k": "delta", "b": 3, "g": 13}], b"\x00\x3c" * 8 + b"\x00\x3c" + bytes(4) + b"\x80")
    with pytest.raises(ValueError, match="padding bit"):
        prepare.decode(th, blob, device="cpu")


def test_rebuild_macs_per_format_bound_what_the_counter_sees():
    th = theta()
    meta = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in th.items()}
    a, b = torch.randn(4, 48), torch.randn(64, 4)
    m = th["w"].float().norm(dim=1)
    quant = {"w": {"kind": "lora", "a": qarray((4, 48), 4, 16), "b": qarray((64, 4), 2, 4), "scale": 0.5},
             "big": {"kind": "delta", "q": qarray((N_BIG,), 2, 128)}, "g": {"kind": "full", "t": th["g"]}}
    blobs = {
        "quant": (prepare.encode_quant(quant, th), 4 * (64 + 48) + (4 + 2) * 64 * 48 + 2 * N_BIG),
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


def lf(n_layer: int, vocab: int = 32000) -> dict:
    from model import Size

    return Size("t", 32, n_layer, 2, 64).config(vocab=vocab)


def lf_theta(config: dict) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return {k: torch.randn(s, generator=g).to(torch.bfloat16) for k, s in prepare.architecture(config).items()}


def test_a_model_format_is_its_own_architecture_with_thetas_tensors_where_e_gives_none(monkeypatch):
    import evaluate

    small, big = lf(1), lf(2)
    th = lf_theta(small)
    rec = {"config": small, "tokenizer": prepare.data_v3.TOKENIZER}
    new = {k: torch.ones(s, dtype=torch.bfloat16) for k, s in prepare.architecture(big).items() if k not in th}
    assert new
    blob = prepare.encode_model(big, new)
    assert prepare.format_of(blob) == "model" and prepare.rebuild_macs(th, blob) == 0
    assert prepare.model_config(rec, blob) == big and prepare.model_config(rec, prepare.encode_full({})) == small
    out = prepare.decode(th, blob, device="cpu")
    assert {k: tuple(v.shape) for k, v in out.items()} == prepare.architecture(big)
    assert all(prepare.same_bits(out[k], new[k] if k in new else th[k]) for k in out)
    m = prepare.model_record(rec, big, out)
    assert m["config"] == big and m["parameters"] == sum(v.numel() for v in out.values())
    assert prepare.model_record(rec, small, th) is rec
    # the evaluation job's cost record reads the grown model's record, scored in the start's tokens
    monkeypatch.setattr(prepare, "held_out", lambda split, tok: (np.zeros(2049, dtype=np.uint16), {"bytes": 8000}))
    monkeypatch.setattr(evaluate, "prepare", prepare)
    cost = evaluate.eval_cost(prepare.SEARCH_POOL, m)
    assert cost["parameters"] == m["parameters"] and cost["set"] == prepare.SEARCH_POOL
    with torch.device("cpu"):
        net = prepare.Net(big).load(out)
    assert net.logits(torch.zeros(1, 4, dtype=torch.long)).shape == (1, 4, 32000)


def test_a_model_format_keeps_thetas_vocabulary_context_and_tying_and_only_its_architectures_tensors():
    small, big = lf(1), lf(2)
    th, rec = lf_theta(small), {"config": small}
    for config in (lf(2, vocab=50304), {**big, "ctx": 2048}, {**big, "tied": False}, {**big, "family": "hf-neox"}, None):
        with pytest.raises(ValueError, match="θ's family, vocab, ctx, tied"):
            prepare.model_config(rec, prepare.encode_model(config, {}))
    shapes = prepare.architecture(big)
    new = {k: torch.ones(s, dtype=torch.bfloat16) for k, s in shapes.items() if k not in th}
    one = sorted(new)[0]
    for given, why in ((new | {"stray": torch.ones(2)}, "not its architecture's tensors"),
                       ({k: v for k, v in new.items() if k != one}, f"{one} is neither E's nor θ's"),
                       (new | {one: torch.ones(3)}, f"{one} is neither E's nor θ's"),
                       (new | {one: torch.ones(shapes[one], dtype=torch.int32)}, f"{one} is neither E's nor θ's")):
        with pytest.raises(ValueError, match=why):
            prepare.decode(th, prepare.encode_model(big, given), device="cpu")


@pytest.mark.parametrize("key, tensors, zero, live, stage", [
    ("prefix", ("prefix_k.weight", "prefix_v.weight", "prefix_gate"), "prefix_gate", "prefix_k.weight", 1),
    ("adapter", ("adapter_norm", "adapter_down.weight", "adapter_up.weight"), "adapter_up.weight", "adapter_down.weight", 7),
])
def test_an_extended_model_at_its_initialisation_computes_thetas_function_exactly(key, tensors, zero, live, stage):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    small = lf(2)
    big = {**small, key: 4}
    # at fp32: the CPU's bf16 products aren't bitwise reproducible from one call to the next, θ's own model included
    torch.manual_seed(1)
    with torch.device("cpu"):
        th = {k: v.detach().float() for k, v in prepare.Net(small).params().items()}
        torch.manual_seed(2)
        init = {k: v.detach().float() for k, v in prepare.Net(big).params().items() if k not in th}
    assert sorted(init) == sorted(f"blocks.{j}.{n}" for j in range(2) for n in tensors)
    assert all(not init[f"blocks.{j}.{zero}"].any() and init[f"blocks.{j}.{live}"].any() for j in range(2))
    out = prepare.decode(th, prepare.encode_model(big, init), device="cpu")
    idx = torch.randint(0, 32000, (2, 16), generator=torch.Generator().manual_seed(3))
    rec = prepare.model_record({"config": small, "tokenizer": prepare.data_v3.TOKENIZER}, big, out)
    at = {n["param"]: n for n in prepare.positions(rec)}
    assert at[f"blocks.1.{live}"]["stage"] == stage and at[f"blocks.1.{live}"]["flops"] == 2 * 4 * 32
    from te_core.flops import forward_macs_per_token
    # CPU's flash attention kernel isn't run-to-run deterministic, and forward_macs_per_token doesn't count its
    # aten._scaled_dot_product_flash_attention_for_cpu, so every forward here runs on the MATH kernel
    with sdpa_kernel(SDPBackend.MATH), torch.device("cpu"):
        start, extended = prepare.Net(small).load(th), prepare.Net(big).load(out)
        assert torch.equal(start.logits(idx), extended.logits(idx))
        extra = forward_macs_per_token(prepare.Net(big), 16, 32000, "cpu") - forward_macs_per_token(prepare.Net(small), 16, 32000, "cpu")
        # the 4 prefix keys' scores and values, or the adapter's down and up, at d = 32 in both blocks
        assert extra >= 2 * 2 * 4 * 32
        # training the zero tensor moves the function, so the extension is live
        out[f"blocks.0.{zero}"] = torch.full_like(out[f"blocks.0.{zero}"], 0.5)
        moved = prepare.Net(big).load(out)
        assert not torch.equal(start.logits(idx), moved.logits(idx))


def test_a_bias_extension_at_zero_biases_reproduces_thetas_function_exactly(monkeypatch):
    """BitFit's extension: `bias` adds a zero vector to each named matrix of every block, so θ's logits come back bit
    for bit (fp32 and under bf16 autocast), its forward costs θ's MACs, every bias has its matrix's position, and other
    leaves are refused."""
    from model import LINEARS
    from te_core.flops import forward_macs_per_token
    from torch.nn.attention import SDPBackend, sdpa_kernel

    small = lf(2)
    th = {k: v.float() for k, v in lf_theta(small).items()}
    rec = {"config": small, "tokenizer": prepare.data_v3.TOKENIZER}
    forward = [{"op": "forward", "tokens": 1}]
    theta_forward = prepare.train_flops(rec | {"tensors": {k: [str(v.dtype), list(v.shape)] for k, v in th.items()}},
                                        {"full": []}, forward)["flops"]
    x = torch.randint(0, 32000, (2, 16), generator=torch.Generator().manual_seed(2))
    # On CPU, oneDNN's bf16 matmul and the flash attention kernel aren't run-to-run deterministic (torch 2.14: about one
    # call in 30 differs on identical inputs), so the comparisons run on kernels that are.
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    for leaves in (["wq", "wv", "w_up", "w_down"], list(LINEARS)):
        biased = {**small, "bias": leaves}
        shapes = prepare.architecture(biased)
        new = {k: torch.zeros(s) for k, s in shapes.items() if k not in th}
        assert sorted(new) == sorted(f"blocks.{i}.{n}.bias" for i in range(2) for n in leaves)
        blob = prepare.encode_model(biased, new)
        assert prepare.model_config(rec, blob) == biased
        out = prepare.decode(th, blob, device="cpu")
        plain, ext = prepare.Net(small).load(th), prepare.Net(biased).load(out)
        assert torch.equal(plain.logits(x), ext.logits(x))
        with sdpa_kernel(SDPBackend.MATH):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert torch.equal(plain.logits(x), ext.logits(x))
            assert forward_macs_per_token(ext, 16, 32000, "cpu") == forward_macs_per_token(plain, 16, 32000, "cpu")
        biased_rec = prepare.model_record(rec, biased, out)
        nodes = {n["param"]: n for n in prepare.positions(biased_rec)}
        for k in new:
            w = nodes[k.removesuffix(".bias") + ".weight"]
            assert (nodes[k]["layer"], nodes[k]["stage"], nodes[k]["flops"]) == (w["layer"], w["stage"], 0)
        assert prepare.train_flops(biased_rec, {"full": sorted(new)}, forward)["flops"] == theta_forward
    for bad in (["embed"], ["wq", "norm"], "wq", 3):
        with pytest.raises(ValueError, match="bias must be a list"):
            prepare.decode(th, prepare.encode_model({**small, "bias": bad}, {}), device="cpu")


def test_every_encoder_is_one_the_harness_and_capture_wrap():
    """Each of prepare's `encode_*` is in te_core.cw's ENCODERS, which the harness wraps to keep each checkpoint's E, and
    in te_core.capture.script's, whose recording leaves the encoding out, so no format's encoding is traced as training."""
    from te_core import cw
    from te_core.capture import script

    names = {n for n in vars(prepare) if n.startswith("encode_") and callable(getattr(prepare, n))}
    assert names == set(cw.ENCODERS) == set(script._ENCODERS)


def test_bpe32k_training_windows_cover_every_parts_windows_and_none_crosses_a_part(tmp_path, monkeypatch):
    lengths = (41, 9, 2, 25)
    (tmp_path / "parts").mkdir()
    for i, n in enumerate(lengths):
        (1000 * i + np.arange(n, dtype=np.uint16)).tofile(tmp_path / "parts" / f"{i:03d}.bin")
    monkeypatch.setattr(prepare.data_v3, "TRAIN", tmp_path)
    c, tok = 4, prepare.data_v3.TOKENIZER

    def windows(n_parts: int) -> list[tuple[int, ...]]:
        return sorted(tuple(range(1000 * i + j * c, 1000 * i + j * c + c + 1)) for i, n in enumerate(lengths[:n_parts])
                      for j in range((n - 1) // c))

    def rows(data) -> list[tuple[int, ...]]:
        return sorted(map(tuple, data.next(len(data.order), device="cpu").tolist()))

    # while it is installed: the parts whose records are in place from 000 on (002's isn't yet)
    for i in (0, 1, 3):
        (tmp_path / "parts" / f"{i:03d}.json").write_text("{}")
    assert rows(prepare.TrainData(tok, ctx=c)) == windows(2)
    assert rows(prepare.TrainData(tok, ctx=c, parts=1)) == windows(1)
    with pytest.raises(SystemExit, match="2 training parts here, not 3"):
        prepare.TrainData(tok, ctx=c, parts=3)
    (tmp_path / "manifest.json").write_text(json.dumps({"parts": [{"index": i} for i in range(len(lengths))]}))
    assert rows(prepare.TrainData(tok, ctx=c)) == windows(4)
