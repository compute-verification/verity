"""`engine.int8_checkpoint`'s v3 (`verity-int8-channel127/v3`): an int8 checkpoint from a block-FP8 base.

The dequant is checked exact against rational arithmetic (and shown inexact in float32 for scales like GLM-5.3's), the row rule
against its definition on the exact values (one rounding of the scale, nearest-even of the exact quotient, ties included), the
two BF16 linears against torch's own float32-to-bfloat16 cast of vLLM's float32 upcast, the container on a small synthetic
two-shard MoE checkpoint, pinned by a golden vector (a change to the rule or the container is a new recipe id), the shapes and
bases it refuses, and `report` on the made pin (zero mismatches, and a tampered word found)."""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I

from tests.engine.test_fp8_checkpoint import _bf16, _write_base_shard

G = 128
VOCAB, H = 16, 256
KV_LORA, INDEX_HEAD = 64, 32


def _codes(n: int, salt: int) -> np.ndarray:
    """Deterministic finite e4m3 codes (no 0x7F / 0xFF)."""
    u = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(salt)) % np.uint64(2 ** 32)
    c = (u % np.uint64(254)).astype(np.uint8)
    return np.where(c >= 0x7F, c + 1, c).astype(np.uint8)


def _scales(n: int, salt: int) -> np.ndarray:
    """Block scales like a real base's (amax / 448, full 24-bit significands), spread over a few decades."""
    u = (np.arange(n, dtype=np.uint64) * np.uint64(40503) + np.uint64(salt)) % np.uint64(10007)
    return (np.float32(1e-4) * np.exp(u.astype(np.float32) / np.float32(10007.0) * np.float32(4.0))).astype(np.float32)


def _fp8(n: int, k: int, salt: int) -> dict[str, tuple[str, list[int], bytes]]:
    nb, kb = -(-n // G), -(-k // G)
    return {"weight": ("F8_E4M3", [n, k], _codes(n * k, salt).tobytes()),
            "scale": ("F32", [nb, kb], _scales(nb * kb, salt).tobytes())}


def _linear(name: str, n: int, k: int, salt: int) -> dict[str, tuple[str, list[int], bytes]]:
    t = _fp8(n, k, salt)
    return {f"{name}.weight": t["weight"], f"{name}.weight_scale_inv": t["scale"]}


def _config(**over) -> dict:
    cfg = {"architectures": ["X"], "hidden_size": H, "kv_lora_rank": KV_LORA, "index_head_dim": INDEX_HEAD, "tie_word_embeddings": False,
           "vocab_size": VOCAB, "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                                         "weight_block_size": [G, G], "modules_to_not_convert": ["lm_head", "model.embed_tokens"]}}
    cfg.update(over)
    return cfg


def _base(d: Path, cfg: dict | None = None) -> Path:
    d.mkdir()
    b = lambda n, salt, scale=0.05: _bf16(n, salt, scale).tobytes()            # noqa: E731
    L = "model.layers.0."
    _write_base_shard(d / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("BF16", [VOCAB, H], b(VOCAB * H, 1)),
        L + "input_layernorm.weight": ("BF16", [H], b(H, 2, 1.0)),
        **_linear(L + "self_attn.q_a_proj", 256, H, 3),
        **_linear(L + "self_attn.kv_a_proj_with_mqa", 136, H, 4),                      # 136 rows: a partial last block row
        **_linear(L + "self_attn.kv_b_proj", 384, KV_LORA, 5),                         # K = kv_lora_rank: BF16
        **_linear(L + "self_attn.indexer.wk", INDEX_HEAD, H, 6),                        # N = index_head_dim: BF16
        L + "self_attn.indexer.weights_proj.weight": ("BF16", [8, H], b(8 * H, 7)),
    })
    _write_base_shard(d / "model-00002-of-00002.safetensors", {
        L + "mlp.experts.0.down_proj.weight_scale_inv": _fp8(H, 128, 8)["scale"],       # a scale before its weight
        L + "mlp.experts.0.down_proj.weight": _fp8(H, 128, 8)["weight"],
        **_linear(L + "mlp.experts.0.gate_proj", 128, H, 9),
        **_linear(L + "mlp.shared_experts.up_proj", 128, H, 10),
        L + "mlp.gate.weight": ("BF16", [8, H], b(8 * H, 11)),
        L + "mlp.gate.e_score_correction_bias": ("F32", [8], np.arange(8, dtype=np.float32).tobytes()),
        "model.norm.weight": ("BF16", [H], b(H, 12, 1.0)),
        "lm_head.weight": ("BF16", [VOCAB, H], b(VOCAB * H, 13)),
    })
    (d / "config.json").write_text(json.dumps(cfg or _config(), indent=2) + "\n")
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (d / "tokenizer.json").write_text('{"model": "t"}')
    return d


def _make(tmp_path: Path, name: str, jobs: int = 1, cfg: dict | None = None) -> dict:
    base = _base(tmp_path / f"base-{name}", cfg)
    return F.make(base, tmp_path / name, sorted(p.name for p in base.glob("*.safetensors")), base_repo="org/fp8", base_revision="0" * 40,
                  recipe=I.INT8_V3, jobs=jobs)


def _tensors(d: Path) -> dict[str, tuple[str, list[int], bytes]]:
    out = {}
    for fn in sorted(p.name for p in d.glob("*.safetensors")):
        start, header = F.read_header(d / fn)
        raw = (d / fn).read_bytes()
        out.update({k: (v["dtype"], v["shape"], raw[start + v["data_offsets"][0]: start + v["data_offsets"][1]])
                    for k, v in header.items() if k != "__metadata__"})
    return out


# ------------------------------------------------------------------------------------------------------------------ numerics

def _rn32(x: Fraction) -> float:
    """x rounded once to float32, nearest-even (subnormals included), from its exact value."""
    if x == 0:
        return 0.0
    a = abs(x)
    e = max(a.numerator.bit_length() - a.denominator.bit_length() - 1, -126)
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    while Fraction(2) ** e > a and e > -126:
        e -= 1
    m = a / Fraction(2) ** (e - 23)
    q = round(m)                                                                 # Fraction's round: half to even
    return math.copysign(q * 2.0 ** (e - 23), x)


def test_the_e4m3_table_is_the_format():
    codes = np.arange(256, dtype=np.uint8)
    v = I.E4M3_VALUES
    assert np.isnan(v[[0x7F, 0xFF]]).all() and np.array_equal(v[np.isfinite(v)], F.e4m3fn_value(codes)[np.isfinite(v)])


def test_the_dequant_is_exact_in_float64_and_not_in_float32():
    codes = _codes(64 * 256, 21).reshape(64, 256)
    s = _scales(1 * 2, 22).reshape(1, 2)
    s[0, 1] = np.float32(2.0 ** -140)                                            # a float32-subnormal scale is exact too
    w = I.fp8_exact(codes, s, 0, G)
    exact = [[Fraction(float(F.e4m3fn_value(codes[i, j]))) * Fraction(float(s[0, j // G])) for j in range(256)] for i in range(0, 64, 7)]
    assert all(Fraction(float(w[i * 7, j])) == exact[i][j] for i in range(len(exact)) for j in range(256))
    w32 = F.e4m3fn_value(codes) * np.repeat(s, G, axis=1)
    assert (w32.astype(np.float64) != w)[:, :G].mean() > 0.5                     # float32 rounds most products of such scales


def test_the_exact_rule_is_one_rounding_of_the_exact_values():
    codes = _codes(40 * 384, 31).reshape(40, 384)
    s = _scales(3, 32).reshape(1, 3)
    w = I.fp8_exact(codes, s, 0, G)
    w[3] = 0                                                                     # an all-zero row: the floor
    sw = I.exact_scales(w)
    q = I.exact_quantize(w, sw)
    floor = Fraction(float(np.float32(1e-30)))
    for n in range(40):
        a = max(Fraction(float(abs(x))) for x in w[n])
        assert float(sw[n, 0]) == _rn32(max(a, floor) / 127)
        sn = Fraction(float(sw[n, 0]))
        assert [int(x) for x in q[n]] == [max(-127, min(127, round(Fraction(float(x)) / sn))) for x in w[n]]
    assert int(np.abs(q.astype(np.int16)).max()) == 127 and not q[3].any()


def test_ties_round_to_even_on_the_exact_quotient():
    # block 0's scale is 127, so a = 127 and s = 1; block 1 (scale 1) holds the ties W / s = +-0.5, 1.5, 2.5, 3.5
    one, half, one5, two5, three5 = 0x38, 0x30, 0x3C, 0x42, 0x46                 # e4m3 codes of 1, 0.5, 1.5, 2.5, 3.5
    codes = np.zeros((8, 256), dtype=np.uint8)
    codes[:, 0] = one
    codes[0, 128:134] = [half, one5, two5, three5, two5 | 0x80, half | 0x80]
    w = I.fp8_exact(codes, np.array([[127.0, 1.0]], dtype=np.float32), 0, G)
    assert w[0, 128:134].tolist() == [0.5, 1.5, 2.5, 3.5, -2.5, -0.5]
    sw = I.exact_scales(w)
    assert sw[0, 0] == 1.0 and I.exact_quantize(w, sw)[0, 128:134].tolist() == [0, 2, 2, 4, -2, 0]


def test_the_bf16_linears_are_vllms_upcast():
    codes = _codes(136 * 256, 41).reshape(136, 256)
    s = _scales(2 * 2, 42).reshape(2, 2)
    got = I.fp8_served_bf16(codes[8:136], s, 8, G)
    w32 = torch.from_numpy(F.e4m3fn_value(codes) * np.repeat(np.repeat(s, G, 0), G, 1)[:136])
    want = w32.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(got, want[8:136])


@pytest.mark.parametrize("bad", ["code", "scale"])
def test_nan_codes_and_non_finite_scales_are_refused(tmp_path, bad):
    base = _base(tmp_path / "base")
    t = _linear("model.layers.0.self_attn.q_a_proj", 256, H, 3)
    codes = np.frombuffer(t["model.layers.0.self_attn.q_a_proj.weight"][2], dtype=np.uint8).copy()
    scale = np.frombuffer(t["model.layers.0.self_attn.q_a_proj.weight_scale_inv"][2], dtype=np.float32).copy()
    if bad == "code":
        codes[77] = 0x7F
    else:
        scale[1] = np.inf
    _write_base_shard(base / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("BF16", [VOCAB, H], _bf16(VOCAB * H, 1).tobytes()),
        "model.layers.0.self_attn.q_a_proj.weight": ("F8_E4M3", [256, H], codes.tobytes()),
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": ("F32", [2, 2], scale.tobytes())})
    with pytest.raises(ValueError, match="NaN code" if bad == "code" else "non-finite block scale"):
        F.make(base, tmp_path / "out", sorted(p.name for p in base.glob("*.safetensors")), base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8_V3)


# ----------------------------------------------------------------------------------------------------------------- container

#: v3's bytes on `_base` (a change here is a new recipe, never an edit of this vector)
GOLDEN_V3_REVISION = "5910827e1183659df49daf7cb2c11cdba11c36c3"
GOLDEN_V3_FILES = {
    "config.json": "1995029dfb9d0507b179b9f05920e52d92fb8e57e42bb98c3f121e2adc289ca6",
    "model-00001-of-00002.safetensors": "12502f815e460d2be874f173e9e769db2c827ae1903e6275223a59511d0d229e",
    "model-00002-of-00002.safetensors": "5b6e28feded11e9a6aaec3c0d917f5c6520171de751c06c413f5641024e732a1",
    "model.safetensors.index.json": "93dd3e04e37e3c2626584ab9e1a6e11502ce74dc9659716131f1d319919244cf",
    "tokenizer.json": "0abac9c6951040d55ef49f52a9af6322c3aa022fac687fa036b7be283ac2512d",
}


def test_v3_golden_vector_repeat_and_jobs(tmp_path):
    a, b, c = _make(tmp_path, "a"), _make(tmp_path, "b"), _make(tmp_path, "c", jobs=2)
    assert a == b == c
    made = {n: d["sha256"] for n, d in a["files"].items()}
    assert (a["revision"], made) == (GOLDEN_V3_REVISION, GOLDEN_V3_FILES)
    recipe_made = {n: d["sha256"] for n, d in a["files"].items() if d["digest_source"] == "recipe"}
    assert a["revision"] == F.pin_revision("org/fp8", "0" * 40, recipe_made, I.INT8_V3)


def test_v3_makes_every_fp8_linear_by_the_row_rule_and_keeps_the_rest(tmp_path):
    _make(tmp_path, "a")
    base, made = _tensors(tmp_path / "base-a"), _tensors(tmp_path / "a")
    L = "model.layers.0."
    int8 = [L + "self_attn.q_a_proj", L + "self_attn.kv_a_proj_with_mqa", L + "mlp.experts.0.down_proj", L + "mlp.experts.0.gate_proj",
            L + "mlp.shared_experts.up_proj"]
    for m in int8:
        _, shape, codes = base[m + ".weight"]
        _, sshape, sraw = base[m + ".weight_scale_inv"]
        w = I.fp8_exact(np.frombuffer(codes, dtype=np.uint8).reshape(shape), np.frombuffer(sraw, dtype=np.float32).reshape(sshape), 0, G)
        dq, qshape, q = made[m + ".weight"]
        ds, ss, s = made[m + ".weight_scale"]
        assert (dq, qshape, ds, ss) == ("I8", shape, "F32", [shape[0], 1]) and m + ".weight_scale_inv" not in made
        assert np.frombuffer(s, dtype=np.float32).tobytes() == I.exact_scales(w).tobytes()
        assert np.frombuffer(q, dtype=np.int8).reshape(shape).tobytes() == I.exact_quantize(w, I.exact_scales(w)).tobytes()
    for m in (L + "self_attn.kv_b_proj", L + "self_attn.indexer.wk"):
        _, shape, codes = base[m + ".weight"]
        _, sshape, sraw = base[m + ".weight_scale_inv"]
        want = I.fp8_served_bf16(np.frombuffer(codes, dtype=np.uint8).reshape(shape), np.frombuffer(sraw, dtype=np.float32).reshape(sshape), 0, G)
        assert made[m + ".weight"] == ("BF16", shape, want.tobytes()) and m + ".weight_scale" not in made and m + ".weight_scale_inv" not in made
    dt, shape, q = made["lm_head.weight"]
    w = (np.frombuffer(base["lm_head.weight"][2], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(VOCAB, H).astype(np.float64)
    assert (dt, shape) == ("I8", [VOCAB, H]) and q == I.exact_quantize(w, I.exact_scales(w)).tobytes()
    assert made["lm_head.weight_scale"] == ("F32", [VOCAB, 1], I.exact_scales(w).tobytes())
    copied = {k for k in base if not k.endswith("_scale_inv")} - {m + ".weight" for m in int8} - {
        L + "self_attn.kv_b_proj.weight", L + "self_attn.indexer.wk.weight", "lm_head.weight"}
    assert copied and all(made[k] == base[k] for k in copied)                   # the embedding, norms, router, weights_proj as they are
    assert set(made) == copied | {m + s for m in int8 + ["lm_head"] for s in (".weight", ".weight_scale")} | {
        L + "self_attn.kv_b_proj.weight", L + "self_attn.indexer.wk.weight"}
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert list(cfg)[:3] == ["architectures", "hidden_size", "kv_lora_rank"]
    assert cfg["quantization_config"] == {"modules_to_not_convert": [
        "model.embed_tokens", L + "mlp.gate", L + "self_attn.indexer.weights_proj", L + "self_attn.indexer.wk", L + "self_attn.kv_b_proj"],
        "quant_method": "verity_int8", "weight_max": 127}
    idx = json.loads((tmp_path / "a" / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(made)


def _with(tmp_path: Path, tensors: dict, cfg: dict | None = None) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    _write_base_shard(base / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [VOCAB, H], _bf16(VOCAB * H, 1).tobytes()),
                                                   "lm_head.weight": ("BF16", [VOCAB, H], _bf16(VOCAB * H, 2).tobytes()), **tensors})
    (base / "config.json").write_text(json.dumps(cfg or _config()))
    return base


def _refuse(base: Path, tmp_path: Path, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40, recipe=I.INT8_V3)


@pytest.mark.parametrize("n, k", [(132, 256), (256, 132)])
def test_v3_refuses_shapes_the_int8_product_cannot_take(tmp_path, n, k):
    _refuse(_with(tmp_path, _linear("model.layers.0.mlp.up_proj", n, k, 3)), tmp_path, "not a multiple of 8 on both axes")


def test_v3_refuses_what_it_cannot_read(tmp_path):
    t = _linear("model.layers.0.mlp.up_proj", 256, H, 3)
    cases = [
        ({"model.layers.0.mlp.up_proj.weight": t["model.layers.0.mlp.up_proj.weight"]}, None, "no F32 .*weight_scale_inv \\[2, 2\\]"),
        ({**t, "model.layers.0.mlp.up_proj.weight_scale_inv": ("F32", [2, 1], np.ones(2, np.float32).tobytes())}, None, "beside it"),
        ({"model.layers.0.mlp.up_proj.weight_scale_inv": t["model.layers.0.mlp.up_proj.weight_scale_inv"]}, None, "no F8_E4M3 weight"),
        ({"x.weight": ("F8_E4M3", [2, 8, 8], bytes(128))}, None, "3-D tensor"),
        ({"x.bias": ("F8_E4M3", [8], bytes(8))}, None, "not a 2-D `.weight`"),
        (t, _config(tie_word_embeddings=True), "tie_word_embeddings"),
        (t, {k: v for k, v in _config().items() if k != "tie_word_embeddings"}, "tie_word_embeddings"),
        (t, _config(quantization_config={"quant_method": "fp8", "fmt": "e4m3"}), "block-FP8 base"),
        (t, _config(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 64]}), "block-FP8 base"),
        (t, _config(quantization_config=None), "block-FP8 base"),
    ]
    for i, (tensors, cfg, why) in enumerate(cases):
        d = tmp_path / str(i)
        d.mkdir()
        _refuse(_with(d, tensors, cfg), d, why)


def test_v3_refuses_a_base_without_its_own_lm_head(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write_base_shard(base / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [VOCAB, H], _bf16(VOCAB * H, 1).tobytes()),
                                                   **_linear("model.layers.0.mlp.up_proj", 256, H, 3)})
    (base / "config.json").write_text(json.dumps(_config()))
    _refuse(base, tmp_path, "no shard holds one")


# ------------------------------------------------------------------------------------------------------------------- report

def _entries(tmp_path: Path, pin: dict) -> tuple[dict, list[dict]]:
    base = {"role": "B", "repo": "org/fp8", "revision": "0" * 40, "local_path": str(tmp_path / "base-a"),
            "safetensors_sha256": {fn: "x" for fn in pin["safetensors_sha256"]}}
    entry = {"role": "B_INT8", "repo": "org/fp8-int8", "revision": pin["revision"], "local_path": str(tmp_path / "a"),
             "recipe": {"id": I.RECIPE_V3, "base_repo": "org/fp8", "base_revision": "0" * 40}}
    return entry, [base, entry]


def test_report_reads_the_pin_against_its_base(tmp_path):
    entry, entries = _entries(tmp_path, _make(tmp_path, "a"))
    doc = I.report(entry, entries, None, jobs=2)
    t, s = doc["tensors"], doc["summary"]
    assert s["mismatch"] == 0 and s["tensors"] == len(t) == 8 and s["int8"]["tensors"] == 6 and s["bf16"]["tensors"] == 2
    q = t["model.layers.0.self_attn.q_a_proj.weight"]
    assert q["kind"] == "int8" and q["base_dtype"] == "F8_E4M3" and 0 < q["rel_fro"] and 0 < q["rel_max"] <= 0.5 / 127 + 1e-9
    assert q["inexact_f32"] > q["elements"] // 2 and t["lm_head.weight"]["inexact_f32"] == 0
    assert len(q["rows"]) == 3 and q["rows"][0]["rel_fro"] >= q["rows"][-1]["rel_fro"]
    assert 0 < t["model.layers.0.self_attn.kv_b_proj.weight"]["rel_max"] <= 2.0 ** -8 * 1.001   # half a bf16 ulp
    assert s["int8"]["worst_rel_fro"][0]["rel_fro"] == max(x["rel_fro"] for x in t.values() if x["kind"] == "int8")
    shard = tmp_path / "a" / "model-00002-of-00002.safetensors"
    start, header = F.read_header(shard)
    raw = bytearray(shard.read_bytes())
    at = start + header["model.layers.0.mlp.shared_experts.up_proj.weight"]["data_offsets"][0] + 5
    raw[at] = (raw[at] + 1) % 256
    shard.write_bytes(bytes(raw))
    assert I.report(entry, entries, None)["tensors"]["model.layers.0.mlp.shared_experts.up_proj.weight"]["mismatch"] == 1


def test_recipe_of_names_v3_and_report_refuses_other_recipes():
    assert I.recipe_of({"recipe": {"id": I.RECIPE_V3}}) is I.INT8_V3 and I.INT8_V3.lm_head
    with pytest.raises(F.UsageError, match="reads a v3 pin"):
        I.report({"role": "X", "recipe": {"id": I.RECIPE}}, [], None)
