"""`engine.int8_checkpoint`'s `verity-int8-channel127-fp8/v1`: v1's row rule on a block-FP8 MoE base (GLM-5.3's layout).

Each made int8 weight is checked against the rule applied to W = RN32(e4m3fn(q) * s) computed here independently of the
container; the bf16 router and every non-FP8 tensor are copied byte for byte and the base's scales dropped; the shapes the base
can't be read under are refused; `jobs` doesn't change a byte; and a golden vector pins the bytes."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I

from tests.engine.test_fp8_checkpoint import _base_checkpoint, _bf16, _write_base_shard

KEPT = ["lm_head", "model.embed_tokens", "model.layers.0.mlp.gate", "model.norm"]


def _codes(n: int, salt: int) -> np.ndarray:
    """Deterministic e4m3fn codes, every one finite (the NaN codes 0x7F / 0xFF moved to 0x7E / 0xFE)."""
    u = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(salt)) % np.uint64(2 ** 32) % np.uint64(256)).astype(np.uint8)
    return np.where((u & 0x7F) == 0x7F, u - 1, u).astype(np.uint8)


def _scales(n: int, k: int, salt: int) -> np.ndarray:
    nb, kb = -(-n // 128), -(-k // 128)
    return (np.float32(1e-3) * (np.float32(1) + np.arange(nb * kb, dtype=np.float32) * np.float32(salt) / np.float32(7))).reshape(nb, kb)


def _fp8(n: int, k: int, salt: int) -> tuple[np.ndarray, np.ndarray]:
    return _codes(n * k, salt).reshape(n, k), _scales(n, k, salt)


def _dequant(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    n, k = q.shape
    return F.e4m3fn_value(q) * np.repeat(np.repeat(s, 128, axis=0)[:n], 128, axis=1)[:, :k]


TENSORS_1 = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": (256, 128, 3),
    "model.layers.0.mlp.experts.1.gate_proj.weight": (256, 128, 4),
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": (136, 128, 5),          # N = 136: a partial last block row
}
TENSORS_2 = {"model.layers.0.mlp.experts.0.down_proj.weight": (128, 384, 6)}


def _fp8_base(d: Path, kept: list[str] | None = None, drop_scale: str | None = None, scale_shape: list[int] | None = None) -> Path:
    d.mkdir()
    t = lambda n, salt, scale=0.05: _bf16(n, salt, scale).tobytes()            # noqa: E731

    def fp8(tensors: dict) -> dict:
        out = {}
        for name, (n, k, salt) in tensors.items():
            q, s = _fp8(n, k, salt)
            out[name] = ("F8_E4M3", [n, k], q.tobytes())
            if name != drop_scale:
                sh = scale_shape or list(s.shape)
                out[name[: -len("weight")] + F.SCALE_INV] = ("F32", sh, np.resize(s, sh).astype("<f4").tobytes())
        return out

    _write_base_shard(d / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("BF16", [16, 128], t(16 * 128, 1)),
        "model.layers.0.mlp.gate.weight": ("BF16", [8, 128], t(8 * 128, 2)),
        "model.layers.0.mlp.gate.e_score_correction_bias": ("F32", [8], np.arange(8, dtype=np.float32).tobytes()),
        **fp8(TENSORS_1),
    })
    _write_base_shard(d / "model-00002-of-00002.safetensors", {
        **fp8(TENSORS_2),
        "model.norm.weight": ("BF16", [128], t(128, 8, 1.0)),
        "lm_head.weight": ("BF16", [16, 128], t(16 * 128, 9)),
    })
    q = dict(F.QUANTIZATION, modules_to_not_convert=KEPT if kept is None else kept)
    (d / "config.json").write_text(json.dumps({"architectures": ["X"], "n_routed_experts": 2, "vocab_size": 16,
                                               "tie_word_embeddings": False, "quantization_config": q}, indent=2) + "\n")
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (d / "tokenizer.json").write_text('{"model": "t"}')
    return d


def _make(tmp_path: Path, name: str, jobs: int = 1, **kw) -> dict:
    base = _fp8_base(tmp_path / f"base-{name}", **kw)
    return F.make(base, tmp_path / name, sorted(p.name for p in base.glob("*.safetensors")), base_repo="org/model",
                  base_revision="0" * 40, recipe=I.INT8_FP8, jobs=jobs)


def _tensors(d: Path) -> dict[str, tuple[str, list[int], bytes]]:
    out = {}
    for p in sorted(d.glob("*.safetensors")):
        start, header = F.read_header(p)
        raw = p.read_bytes()
        for k, v in header.items():
            if k != "__metadata__":
                lo, hi = v["data_offsets"]
                out[k] = (v["dtype"], v["shape"], raw[start + lo: start + hi])
    return out


#: the recipe's bytes on `_fp8_base` (a change here is a new recipe id, never an edit of this vector)
GOLDEN_REVISION = "86b85168e045999cf30ab1fefe6bc4566b6e077d"
GOLDEN_FILES = {
    "config.json": "d6f9a49411cce9c1489c04b5fb28ca273cae4809cd89b4b3145f59b42989874a",
    "model-00001-of-00002.safetensors": "bfffdac8505bb316e8520e831fe0358c978fd4fb29ef2d66f2ae98e2fd5c7395",
    "model-00002-of-00002.safetensors": "cd00b69d9c5403828061c84eeb92bebcad40172048096f53fc91d213537fcb14",
    "model.safetensors.index.json": "c44b678cb2a5ea95bde0136716e9fd5e4bf0fa63c75405cf0ce7626dd3dc039e",
    "tokenizer.json": "0abac9c6951040d55ef49f52a9af6322c3aa022fac687fa036b7be283ac2512d",
}


def test_golden_vector_repeat_and_jobs(tmp_path):
    a, b = _make(tmp_path, "a"), _make(tmp_path, "b", jobs=2)
    assert a == b
    made = {n: d["sha256"] for n, d in a["files"].items()}
    assert (a["revision"], made) == (GOLDEN_REVISION, GOLDEN_FILES)
    recipe_made = {n: d["sha256"] for n, d in a["files"].items() if d["digest_source"] == "recipe"}
    assert a["revision"] == F.pin_revision("org/model", "0" * 40, recipe_made, I.INT8_FP8) \
        != F.pin_revision("org/model", "0" * 40, recipe_made, I.INT8)


def test_every_fp8_linear_is_v1s_row_rule_on_its_float32_dequantization(tmp_path):
    _make(tmp_path, "a")
    base, out = _tensors(tmp_path / "base-a"), _tensors(tmp_path / "a")
    for name, (n, k, salt) in {**TENSORS_1, **TENSORS_2}.items():
        q8, s8 = _fp8(n, k, salt)
        w = _dequant(q8, s8)
        assert w.dtype == np.float32 and np.isfinite(w).all()
        dt, shape, q = out[name]
        ds, sshape, s = out[name[: -len("weight")] + "weight_scale"]
        assert (dt, shape, ds, sshape) == ("I8", [n, k], "F32", [n, 1])
        assert np.frombuffer(s, dtype="<f4").reshape(n, 1).tobytes() == I.channel_scales(w).tobytes()
        assert np.array_equal(np.frombuffer(q, dtype=np.int8).reshape(n, k), I.quantize(w, I.channel_scales(w)))
        assert name[: -len("weight")] + F.SCALE_INV not in out
    copied = {k for k, v in base.items() if v[0] not in ("F8_E4M3",) and not k.endswith(F.SCALE_INV)}
    assert copied == {"model.embed_tokens.weight", "model.layers.0.mlp.gate.weight", "model.layers.0.mlp.gate.e_score_correction_bias",
                      "model.norm.weight", "lm_head.weight"}
    assert all(out[k] == base[k] for k in copied)                                # the bf16 router, embedding, LM head: as the base has them
    assert set(out) == copied | set(TENSORS_1) | set(TENSORS_2) | {k[: -len("weight")] + "weight_scale" for k in (*TENSORS_1, *TENSORS_2)}
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg["quantization_config"] == {"modules_to_not_convert": KEPT, "quant_method": "verity_int8", "weight_max": 127}
    assert list(cfg) == ["architectures", "n_routed_experts", "vocab_size", "tie_word_embeddings", "quantization_config"]


def test_the_dequantization_is_the_float32_product():
    q = np.array([[0x7E, 0x01, 0x38, 0xB8]], dtype=np.uint8)                   # 448, 2^-9, 1.0, -1.0
    s = np.array([[np.float32(1) / np.float32(3)]], dtype=np.float32)
    assert _dequant(q, s).tolist() == [[np.float32(448) * s[0, 0], np.float32(2.0 ** -9) * s[0, 0], s[0, 0], -s[0, 0]]]


@pytest.mark.parametrize("kw, why", [
    ({"drop_scale": "model.layers.0.mlp.experts.1.gate_proj.weight"}, "F8_E4M3 tensor that is not a 2-D `.weight` with its weight_scale_inv"),
    ({"scale_shape": [3, 1]}, "is not the F32 \\[2, 1\\] block scale"),
])
def test_refuses_an_fp8_weight_it_cannot_read(tmp_path, kw, why):
    with pytest.raises(ValueError, match=why):
        _make(tmp_path, "a", **kw)


def test_refuses_a_base_that_is_not_block_fp8_and_v1_refuses_an_fp8_base(tmp_path):
    bf16 = _base_checkpoint(tmp_path / "bf16")
    with pytest.raises(ValueError, match="reads a block-FP8 base"):
        F.make(bf16, tmp_path / "x", sorted(p.name for p in bf16.glob("*.safetensors")), base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8_FP8)
    fp8 = _fp8_base(tmp_path / "fp8")
    with pytest.raises(ValueError, match="already quantized"):
        F.make(fp8, tmp_path / "y", sorted(p.name for p in fp8.glob("*.safetensors")), base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8)
    cfg = json.loads((fp8 / "config.json").read_text())
    cfg["quantization_config"]["weight_block_size"] = [64, 64]
    (fp8 / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="reads a block-FP8 base"):
        F.make(fp8, tmp_path / "z", sorted(p.name for p in fp8.glob("*.safetensors")), base_repo="o/m", base_revision="0" * 40,
               recipe=I.INT8_FP8)


def test_jobs_leave_v1s_bytes_unchanged(tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    shards = sorted(p.name for p in base.glob("*.safetensors"))
    a = F.make(base, tmp_path / "a", shards, base_repo="o/m", base_revision="0" * 40, recipe=I.INT8)
    b = F.make(base, tmp_path / "b", shards, base_repo="o/m", base_revision="0" * 40, recipe=I.INT8, jobs=2)
    assert a == b


def _local_pull(src: str, dst: str, *flags: str) -> None:
    """`fp8_checkpoint._pull` on this host (no rsync here): a directory's files but the shards, or one file."""
    if src.endswith("/"):
        for p in Path(src).iterdir():
            if not p.name.endswith(".safetensors"):
                shutil.copyfile(p, Path(dst) / p.name)
    else:
        shutil.copyfile(src, dst)


def _pin(base: Path) -> dict:
    shards = {p.name: F._sha256(p) for p in sorted(base.glob("*.safetensors"))}
    return {"repo": "org/model", "revision": "0" * 40, "safetensors_sha256": shards, "config_sha256": F._sha256(base / "config.json")}


def test_a_base_pulled_shard_by_shard_makes_the_same_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "_pull", _local_pull)
    local = _make(tmp_path, "local")
    base = tmp_path / "base-local"
    for jobs in (1, 2):
        pin = F._make_into(tmp_path / f"snap{jobs}", None, _pin(base), I.INT8_FP8, jobs, str(base))
        assert {k: v for k, v in pin.items() if k != "local_path"} == local
        assert sorted(p.name for p in (tmp_path / f"snap{jobs}").iterdir()) == [local["revision"]]   # scratch and tmp are gone


def test_a_base_pulled_from_the_hub_makes_the_same_bytes(tmp_path, monkeypatch):
    local = _make(tmp_path, "local")
    base = tmp_path / "base-local"
    url = f"{F.HUB}/org/model/resolve/{'0' * 40}/"
    seen = []

    def pull(src: str, dst: str, *flags: str) -> None:
        assert src.startswith(url) and not flags
        seen.append(src[len(url):])
        shutil.copyfile(base / src[len(url):], dst)

    monkeypatch.setattr(F, "_pull", pull)
    monkeypatch.setattr(F, "_hub_files", lambda repo, rev: sorted(p.name for p in base.iterdir()) if (repo, rev) == ("org/model", "0" * 40) else [])
    pin = F._make_into(tmp_path / "snap", None, _pin(base), I.INT8_FP8, 1, "hub")
    assert {k: v for k, v in pin.items() if k != "local_path"} == local
    assert sorted(seen) == sorted(p.name for p in base.iterdir())


def test_a_pulled_shard_or_config_off_the_pin_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "_pull", _local_pull)
    base = _fp8_base(tmp_path / "base")
    pin = _pin(base)
    bad = dict(pin, safetensors_sha256=dict(pin["safetensors_sha256"], **{"model-00002-of-00002.safetensors": "0" * 64}))
    with pytest.raises(F.UsageError, match="model-00002-of-00002.safetensors: not the pinned bytes"):
        F._make_into(tmp_path / "a", None, bad, I.INT8_FP8, 2, str(base))
    with pytest.raises(F.UsageError, match="config.json: not the pinned bytes"):
        F._make_into(tmp_path / "b", None, dict(pin, config_sha256="0" * 64), I.INT8_FP8, 1, str(base))
    assert not [p for d in ("a", "b") for p in (tmp_path / d).iterdir() if p.name.startswith(".base")]


def test_recipe_of_names_the_fp8_recipe():
    assert I.recipe_of({"recipe": {"id": I.RECIPE_FP8}}) is I.INT8_FP8 and I.INT8_FP8.fp8_base and not I.INT8_FP8.lm_head
