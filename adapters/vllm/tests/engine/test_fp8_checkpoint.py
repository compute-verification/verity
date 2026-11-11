"""`engine.fp8_checkpoint`: the offline block-FP8 recipe (`verity-fp8-block128/v1`).

The e4m3fn rounding is checked against exact nearest-even over the whole format, the block rule against its definition, and
the recipe's bytes are pinned by a golden vector (a small synthetic two-shard checkpoint): a change to the rule or the
container changes it, and needs a new recipe id."""
from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.program import model as M


# ------------------------------------------------------------------------------------------------------------------ numerics

def _finite_codes() -> np.ndarray:
    return np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)


def _nearest_even(x: float, grid: dict[Fraction, int]) -> int:
    a = Fraction(abs(x))
    d = min(abs(p - a) for p in grid)
    code = min((grid[p] for p in grid if abs(p - a) == d), key=lambda c: (c & 1, c))
    return code | (0x80 if np.signbit(x) else 0)


def test_e4m3fn_rounds_to_nearest_even_over_the_whole_format():
    codes = _finite_codes()
    values = F.e4m3fn_value(codes)
    assert np.array_equal(F.e4m3fn(values), codes)                           # every finite code round-trips, -0 included
    assert float(values.max()) == 448.0 and float(values[values > 0].min()) == 2.0 ** -9
    grid = {Fraction(float(v)): int(c) for c, v in zip(codes, values) if not np.signbit(v)}
    pos = sorted(grid)
    mids = [float((pos[i] + pos[i + 1]) / 2) for i in range(len(pos) - 1)]    # every tie, each exact in float32
    xs = np.array(mids + [-m for m in mids], dtype=np.float32)
    assert all(Fraction(float(x)) == Fraction(m) for x, m in zip(xs, mids))
    assert np.array_equal(F.e4m3fn(xs), np.array([_nearest_even(float(x), grid) for x in xs], dtype=np.uint8))
    k = np.arange(4096, dtype=np.int64)
    xs = ((k * 7919 % 8191 - 4095).astype(np.float32) / np.float32(4096.0)) * np.exp2((k % 22 - 14).astype(np.float32))
    assert np.array_equal(F.e4m3fn(xs), np.array([_nearest_even(float(x), grid) for x in xs], dtype=np.uint8))
    assert F.e4m3fn(np.array([-0.0, 0.0, -1e-9, 1e-9], dtype=np.float32)).tolist() == [0x80, 0x00, 0x80, 0x00]


@pytest.mark.parametrize("bad", [448.01, np.inf, np.nan])
def test_e4m3fn_refuses_values_outside_the_format(bad):
    with pytest.raises(ValueError):
        F.e4m3fn(np.array([1.0, bad], dtype=np.float32))


def _bf16(n: int, salt: int, scale: float = 1.0) -> np.ndarray:
    """Deterministic bf16 bit patterns without an RNG stream: an integer hash, widened, scaled, truncated to bf16."""
    u = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(salt)) % np.uint64(2 ** 32)
    x = ((u % np.uint64(20001)).astype(np.int64) - 10000).astype(np.float32) / np.float32(10000.0) * np.float32(scale)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def test_block_scales_and_quantize_follow_the_rule():
    w16 = _bf16(256 * 384, 7, 0.05).reshape(256, 384)
    w16[:128, 128:256] = 0                                                     # an all-zero block: s = 1, q = 0
    w16[128:, :128] = _bf16(128 * 128, 11, 1e-4).reshape(128, 128)             # a block of small values (e4m3 subnormals after scaling)
    w16[200, 5] = 0x3F80                                                       # ... with one 1.0 in it
    w = (w16.astype(np.uint32) << 16).view(np.float32)
    s = F.block_scales(w)
    amax = np.abs(w).reshape(2, 128, 3, 128).max(axis=(1, 3))
    assert s.dtype == np.float32 and s.shape == (2, 3)
    assert s[0, 1] == 1.0 and np.array_equal(s[amax > 0], (amax[amax > 0] / np.float32(448)).astype(np.float32))
    q = F.quantize(w, s)
    assert q.dtype == np.uint8 and q.shape == w.shape and not q[:128, 128:256].any()
    qv = F.e4m3fn_value(q)
    blocks = np.abs(qv).reshape(2, 128, 3, 128).max(axis=(1, 3))
    assert np.array_equal(blocks == 448.0, amax > 0)                           # every nonzero block reaches the format's maximum
    S = np.repeat(np.repeat(s, 128, 0), 128, 1)
    assert np.all(np.abs(qv * S - w) <= np.maximum(np.abs(w) * 2.0 ** -4, S * 2.0 ** -10))   # within half an e4m3 step
    assert np.array_equal(q, F.e4m3fn(np.clip(w / S, -448, 448)))


# -------------------------------------------------------------------------------------------------------------- the recipe

def _write_base_shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, at, data = {}, 0, b""
    for name, (dtype, shape, raw) in tensors.items():                        # insertion order, unaligned: a base need not be canonical
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
        data += raw
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


def _base_checkpoint(d: Path) -> Path:
    d.mkdir()
    t = lambda n, salt, scale=0.05: _bf16(n, salt, scale).tobytes()            # noqa: E731
    _write_base_shard(d / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("BF16", [16, 128], t(16 * 128, 1)),
        "model.layers.0.input_layernorm.weight": ("BF16", [128], t(128, 2, 1.0)),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [256, 128], t(256 * 128, 3)),
        "model.layers.0.self_attn.q_proj.bias": ("BF16", [256], t(256, 4)),
        "model.layers.0.self_attn.k_proj.weight": ("F32", [128, 128], (_bf16(128 * 128, 5, 0.05).astype(np.uint32) << 16).tobytes()),
        "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", [64], np.arange(64, dtype=np.float32).tobytes()),
    })
    _write_base_shard(d / "model-00002-of-00002.safetensors", {
        "model.layers.0.mlp.down_proj.weight": ("BF16", [128, 256], t(128 * 256, 7)),
        "model.norm.weight": ("BF16", [128], t(128, 8, 1.0)),
        "lm_head.weight": ("BF16", [16, 128], t(16 * 128, 9)),
    })
    (d / "config.json").write_text('{\n  "architectures": ["X"],\n  "hidden_size": 128,\n  "rms_norm_eps": 1e-06,\n  "torch_dtype": "bfloat16",\n'
                                   '  "vocab_size": 16\n}\n')
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (d / "tokenizer.json").write_text('{"model": "t"}')
    (d / "README.md").write_text("not copied")
    return d


#: the recipe's bytes on `_base_checkpoint` (a change here is a new recipe, `RECIPE`, never an edit of this vector)
GOLDEN_REVISION = "178c5006c9754f08c419447f107c0487ec203d98"
GOLDEN_FILES = {
    "config.json": "32113d05fac5c2a0b0b9cac339bbff8e537e10059cf77558ea7eadf411132326",
    "model-00001-of-00002.safetensors": "95aac92195a9039bb94ca375f5157911bd2e2014a2e5f5f9461b76e67b615156",
    "model-00002-of-00002.safetensors": "5598b9eef3535aa6608f28f1083ef956edf8cb5bd3d74fdfad46a4a184c8262e",
    "model.safetensors.index.json": "cf788ef1bc0ced4fc4d19850bb32ba751e6eaba0cab21989265dedb1a5c08b01",
    "tokenizer.json": "0abac9c6951040d55ef49f52a9af6322c3aa022fac687fa036b7be283ac2512d",
}


def _make(tmp_path: Path, name: str) -> dict:
    base = _base_checkpoint(tmp_path / f"base-{name}")
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    return F.make(base, tmp_path / name, shards, base_repo="org/model", base_revision="0" * 40)


def test_recipe_golden_vector_and_repeat(tmp_path):
    a, b = _make(tmp_path, "a"), _make(tmp_path, "b")
    assert a == b                                                              # the same bytes twice
    made = {n: d["sha256"] for n, d in a["files"].items()}
    assert (a["revision"], made) == (GOLDEN_REVISION, GOLDEN_FILES)
    assert sorted(a["files"]) == ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                                  "model.safetensors.index.json", "tokenizer.json"]
    assert a["files"]["tokenizer.json"]["digest_source"] == "base" and a["files"]["config.json"]["digest_source"] == "recipe"
    assert a["revision"] == F.pin_revision("org/model", "0" * 40, {n: d["sha256"] for n, d in a["files"].items() if d["digest_source"] == "recipe"})


def test_recipe_container_and_config(tmp_path):
    _make(tmp_path, "a")
    out = tmp_path / "a"
    names = {}
    for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        start, header = F.read_header(out / fn)
        assert (start - 8) % 8 == 0 and header.pop("__metadata__") == {"format": "pt"}
        spans = sorted((tuple(v["data_offsets"]), k, v["dtype"]) for k, v in header.items())
        assert spans[0][0][0] == 0 and all(x[0][1] == y[0][0] for x, y in zip(spans, spans[1:]))      # contiguous
        assert all(lo % F._ELEM[dt] == 0 for (lo, _), _, dt in spans)                                  # aligned
        assert (out / fn).stat().st_size == start + spans[-1][0][1]
        names.update({k: (v["dtype"], v["shape"]) for k, v in header.items()})
    assert names["model.layers.0.self_attn.q_proj.weight"] == ("F8_E4M3", [256, 128])
    assert names["model.layers.0.self_attn.q_proj.weight_scale_inv"] == ("F32", [2, 1])
    assert names["model.layers.0.self_attn.k_proj.weight"] == ("F8_E4M3", [128, 128])                # F32 base weights quantize too
    assert names["model.layers.0.mlp.down_proj.weight_scale_inv"] == ("F32", [1, 2])
    for kept in ("model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.rotary_emb.inv_freq", "lm_head.weight",
                 "model.embed_tokens.weight"):
        assert names[kept][0] != "F8_E4M3"
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(names) and idx["metadata"]["total_size"] == sum(
        F._ELEM[dt] * int(np.prod(shape)) for dt, shape in names.values())
    cfg = json.loads((out / "config.json").read_text())
    assert list(cfg)[:5] == ["architectures", "hidden_size", "rms_norm_eps", "torch_dtype", "vocab_size"]   # base key order kept
    assert cfg["quantization_config"] == {"activation_scheme": "dynamic", "fmt": "e4m3", "modules_to_not_convert": ["lm_head"],
                                          "quant_method": "fp8", "weight_block_size": [128, 128]}
    assert M.quant_refusal(cfg["quantization_config"]) is None
    assert not (out / "README.md").exists()


def _one_tensor_base(tmp_path: Path, shape: list[int], config: str) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    n = int(np.prod(shape))
    _write_base_shard(base / "model.safetensors", {"model.embed_tokens.weight": ("BF16", [256, 128], _bf16(256 * 128, 1).tobytes()),
                                                   "model.layers.0.self_attn.q_proj.weight": ("BF16", shape, _bf16(n, 2).tobytes())})
    (base / "config.json").write_text(config)
    return base


@pytest.mark.parametrize("shape, config, why", [
    ([192, 128], '{"vocab_size": 256}', "multiple of the FP8 block 128"),
    ([2, 128, 128], '{"vocab_size": 256}', "3-D tensor"),
    ([128, 128], '{"vocab_size": 16}', "no tensor has the config's vocab_size"),     # an embedding that tiles is never quantized
    ([128, 128], '{"vocab_size": 256, "num_experts": 8}', "mixture-of-experts"),
    ([128, 128], '{"vocab_size": 256, "num_local_experts": 8}', "mixture-of-experts"),
    ([128, 128], '{"vocab_size": 256, "quantization_config": {"quant_method": "awq"}}', "already quantized"),
    ([128, 128], '{}', "no integer vocab_size"),
])
def test_recipe_refuses_what_it_cannot_make(tmp_path, shape, config, why):
    base = _one_tensor_base(tmp_path, shape, config)
    with pytest.raises(ValueError, match=why):
        F.make(base, tmp_path / "out", ["model.safetensors"], base_repo="o/m", base_revision="0" * 40)
