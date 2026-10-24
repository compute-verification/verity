"""Block-scaled FP8 checkpoints made offline from a pinned BF16 checkpoint: the recipe ``verity-fp8-block128/v1``.

    verity-vllm fp8-checkpoint make --role ROLE [--hf-home DIR]        re-make a registered FP8 pin and check every file's sha256
    verity-vllm fp8-checkpoint pin --base-role ROLE --role ROLE --repo ID [--hf-home DIR]
                                                                         make one and print its new manifest entry

The integration serves this FP8 checkpoint quantization and the exact-int8 one (``program.model.quant_refusal``): vLLM ``Fp8Config`` with
``quant_method: fp8``, ``fmt: e4m3``, ``activation_scheme: dynamic`` and ``weight_block_size: [128, 128]``.  A model whose
authors publish no release in that format gets one from this recipe, pinned in ``manifests/checkpoints.json`` beside its BF16
pin (``recipe``, ``fp8_of``).  The recipe is data-free: the weight scales come from the weights alone and the activations are
quantized at run time, so there is no calibration set and no seed.  Its output is a pure function of the base checkpoint's
bytes, so every host re-makes the pinned bytes.

The recipe makes dense checkpoints only: a mixture of experts (``num_experts`` / ``num_local_experts`` in config.json) is
refused, since its router is a 2-D weight vLLM keeps in bf16 and its experts are 3-D (an FP8 MoE is served from its authors'
release, its routers in ``modules_to_not_convert``).  A recipe with ``moe`` (the int8 ones) makes a MoE whose experts are
stored as 2-D tensors, each expert's projections a linear like any other, and copies its routers, every 2-D tensor shaped
[num_experts, hidden_size] (`routers`), as they are.  The linear weights are
read off the structure, never the tensor names: every 2-D tensor none of whose dimensions is the config's ``vocab_size`` (so
the embedding and ``lm_head`` stay bf16, and ``modules_to_not_convert`` is ``["lm_head"]``).  For every such W [N, K], G = 128:

* W is read exactly as float32 (BF16, F16 and F32 widen exactly);
* for every G x G block b of W (rows G*i.., columns G*j..): a = max |W| over b, and s = a / 448 in float32 (IEEE
  round-to-nearest-even); s = 1 when a = 0;
* the stored weight is q = e4m3fn(W / s): the float32 quotient (round-to-nearest-even), clamped to [-448, 448], rounded to
  the nearest e4m3fn value, ties to even, keeping the sign of zero;
* stored as ``<name>`` F8_E4M3 [N, K] and ``<module>.weight_scale_inv`` F32 [N/G, K/G]; the dequantized weight is q * s.

N and K must be multiples of G (``program.model.dense_layer_weights_type``), else the checkpoint is refused whole, as it is
when no tensor has a ``vocab_size`` dimension or one has more than two.  Every other tensor (the embedding, ``lm_head``,
norms, biases) is copied byte for byte.

The container, so the bytes are the rule's and not a library's:

* the base's shard file names; each shard holds its base tensors plus their scales, laid out by (element size descending,
  name), so every tensor's offset is a multiple of its element size;
* the safetensors header is compact JSON with sorted keys and ``__metadata__ = {"format": "pt"}``, space-padded to a multiple
  of 8 bytes;
* ``model.safetensors.index.json`` (when the base has one) is ``metadata.total_size`` and the weight map, sorted, indent 2;
* ``config.json`` is the base's, key order kept, with ``quantization_config`` appended, indent 2;
* the base's other files (``COPY_PATTERNS``: tokenizer, generation config) are copied unchanged.

The pinned revision is the first 40 hex digits of the sha256 of the recipe id, the base pin and the made files' sha256s, so
the checkpoint resolves like a hub snapshot (``<hf_home>/hub/models--<org>--<name>/snapshots/<revision>/``).

The container and the pins take the per-tensor rule as a `Recipe` (``FP8`` here by default); ``engine.int8_checkpoint``'s are
the others, on the same container (its v3 reads an FP8-block base, with its own shard and config rules).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import config as C
from verity_vllm.config import UsageError, option

RECIPE = "verity-fp8-block128/v1"
BLOCK = 128
E4M3_MAX = 448.0
QUANTIZATION = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [BLOCK, BLOCK]}
#: the base files copied unchanged: the patterns `ops/pod_bootstrap.sh` fetches, less what the recipe writes
COPY_PATTERNS = ("*.json", "*.txt", "*.model", "*.py")
CONFIG = "config.json"
INDEX = "model.safetensors.index.json"
LM_HEAD = "lm_head.weight"
#: rows of W quantized at a time (a multiple of BLOCK): bounds the float32 temporaries to a few hundred MB
STRIPE = 8 * BLOCK
_ELEM = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
         "BOOL": 1}
_WIDEN = {"BF16": lambda b: (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32),
          "F16": lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32),
          "F32": lambda b: np.frombuffer(b, dtype=np.float32).copy()}
_CHUNK = 1 << 24


# ------------------------------------------------------------------------------------------------------------------ numerics

def e4m3fn(x: np.ndarray) -> np.ndarray:
    """float32 -> e4m3fn bytes: nearest value, ties to even, sign of zero kept.  |x| must be <= 448 (callers clamp)."""
    x = np.require(np.asarray(x, dtype=np.float32), requirements="C")
    sign = ((x.view(np.uint32) >> 24) & np.uint32(0x80)).astype(np.uint8)
    a = np.abs(x)
    if a.size and not float(a.max()) <= E4M3_MAX:
        raise ValueError(f"e4m3fn input outside [-{E4M3_MAX}, {E4M3_MAX}] (max |x| {float(a.max())!r})")
    ab = a.view(np.uint32)
    r = (ab + np.uint32(0x7FFFF) + ((ab >> 20) & np.uint32(1))) & np.uint32(0xFFF00000)   # 3 mantissa bits, ties to even
    normal = ((((r >> 23) - np.uint32(120)) << 3) | ((r >> 20) & np.uint32(7)))            # exponent bias 127 -> 7
    subnormal = np.rint(a * np.float32(512.0)).astype(np.uint32)                            # multiples of 2^-9; 8 is 2^-6
    code = np.where(a >= np.float32(2.0 ** -6), normal, subnormal)
    return code.astype(np.uint8) | sign


def e4m3fn_value(code: np.ndarray) -> np.ndarray:
    """e4m3fn bytes -> float32 (the NaN codes 0x7F / 0xFF -> NaN)."""
    c = np.asarray(code, dtype=np.uint8).astype(np.int32)
    e, m = (c >> 3) & 0xF, c & 7
    mag = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.exp2(e - 10.0))
    mag = np.where((c & 0x7F) == 0x7F, np.nan, mag)
    return np.where(c & 0x80, -mag, mag).astype(np.float32)


def block_scales(w: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """s [N/G, K/G] float32 of a float32 [N, K]: max |W| per G x G block / 448, 1 for an all-zero block."""
    n, k = w.shape
    amax = np.abs(w).reshape(n // block, block, k // block, block).max(axis=(1, 3))
    if not np.isfinite(amax).all():
        raise ValueError("non-finite weight")
    s = (amax / np.float32(E4M3_MAX)).astype(np.float32)
    return np.where(amax == 0, np.float32(1.0), s)


def quantize(w: np.ndarray, s: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """q [N, K] e4m3fn bytes of a float32 [N, K] under its block scales `s`."""
    n, k = w.shape
    x = w.reshape(n // block, block, k // block, block) / s[:, None, :, None]
    return e4m3fn(np.clip(x, np.float32(-E4M3_MAX), np.float32(E4M3_MAX))).reshape(n, k)


@dataclass(frozen=True)
class Recipe:
    """What a data-free weight recipe changes in the container: the made weight's dtype, its scale tensor and quantization_config.
    `scales(w)` maps float32 rows of W to their scale rows (`rows_per_scale` weight rows each), `quantize(w, s)` to the stored
    bytes; `refusal(n, k)` names a shape the recipe cannot make (None = makes it).  `of` is the pin's key naming its BF16 base.
    `lm_head`: the recipe also makes `LM_HEAD` [vocab_size, K] by the same rule, the config's `modules_to_not_convert` is empty,
    and a checkpoint with tied embeddings or without its own `LM_HEAD` is refused (the embedding, a gather, stays as it is).
    A recipe over a base of another format (``int8_checkpoint``'s v3, an FP8-block base) brings its own rules for the container's
    three steps: `plan(cfg, recipe)` reads the base config (default `dense_vocab`), `shard(src, dst, plan, recipe)` writes one
    shard (default `make_shard`), `config(base_config, recipe, tensors)` writes config.json from the made tensors' (dtype, shape)
    (default `fp8_config`)."""

    id: str
    weight_dtype: str
    scale_suffix: str
    rows_per_scale: int
    scale_shape: Callable[[int, int], list[int]]
    refusal: Callable[[int, int], str | None]
    scales: Callable[[np.ndarray], np.ndarray]
    quantize: Callable[[np.ndarray, np.ndarray], np.ndarray]
    quantization: dict[str, Any]
    of: str
    #: makes a mixture of experts too (`routers`)
    moe: bool = False
    #: the 2-D shapes copied as they are, not made (a MoE's routers, set by `make`)
    kept: tuple[tuple[int, int], ...] = ()
    lm_head: bool = False
    plan: Callable[[dict[str, Any], "Recipe"], Any] | None = None
    shard: Callable[[Path, Path, Any, "Recipe"], tuple[dict[str, Any], dict[str, str], int, int]] | None = None
    config: Callable[[bytes, "Recipe", dict[str, tuple[str, list[int]]]], bytes] | None = None


FP8 = Recipe(id=RECIPE, weight_dtype="F8_E4M3", scale_suffix="weight_scale_inv", rows_per_scale=BLOCK,
             scale_shape=lambda n, k: [n // BLOCK, k // BLOCK],
             refusal=lambda n, k: f"not a multiple of the FP8 block {BLOCK} on both axes" if n % BLOCK or k % BLOCK else None,
             scales=block_scales, quantize=quantize, quantization=QUANTIZATION, of="fp8_of")


# ---------------------------------------------------------------------------------------------------------------- container

def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    """(byte offset of the data, header) of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return 8 + n, json.loads(f.read(n))


def is_linear(info: dict[str, Any], vocab: int) -> bool:
    """A linear weight of a dense checkpoint: every 2-D tensor but the vocabulary-sized ones (the embedding, `lm_head`)."""
    return len(info["shape"]) == 2 and vocab not in [int(d) for d in info["shape"]]


def scale_name(name: str, recipe: Recipe = FP8) -> str:
    return name[: -len("weight")] + recipe.scale_suffix


def _nbytes(dtype: str, shape: list[int]) -> int:
    return _ELEM[dtype] * int(np.prod(shape, dtype=np.int64))


def shard_layout(header: dict[str, Any], vocab: int, recipe: Recipe = FP8) -> list[tuple[str, str, list[int], str, str]]:
    """The made shard's tensors in file order: (name, dtype, shape, kind, source), kind "copy" | "weight" | "scale"."""
    out = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = [int(d) for d in info["shape"]]
        if len(shape) > 2:
            raise ValueError(f"{name}: a {len(shape)}-D tensor has no place in a dense checkpoint")
        if recipe.lm_head and name == LM_HEAD and shape[0] != vocab:
            raise ValueError(f"{name}: {shape} is not [vocab_size {vocab}, K]")
        if (is_linear(info, vocab) and tuple(shape) not in recipe.kept) or (recipe.lm_head and name == LM_HEAD):
            if info["dtype"] not in _WIDEN or not name.endswith(".weight"):
                raise ValueError(f"{name}: a 2-D {info['dtype']} tensor that is not a float `.weight`")
            n, k = shape
            if why := recipe.refusal(n, k):
                raise ValueError(f"{name}: [{n}][{k}] is {why}")
            out.append((name, recipe.weight_dtype, shape, "weight", name))
            out.append((scale_name(name, recipe), "F32", recipe.scale_shape(n, k), "scale", name))
        else:
            if info["dtype"] not in _ELEM:
                raise ValueError(f"{name}: unknown safetensors dtype {info['dtype']}")
            out.append((name, info["dtype"], shape, "copy", name))
    names = [t[0] for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"scale tensor name collides with a base tensor: {sorted({n for n in names if names.count(n) > 1})[:3]}")
    return sorted(out, key=lambda t: (-_ELEM[t[1]], t[0]))


def encode_header(layout: list[tuple[str, str, list[int], str, str]]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    doc: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offsets, at = {}, 0
    for name, dtype, shape, _, _ in layout:
        n = _nbytes(dtype, shape)
        doc[name] = {"data_offsets": [at, at + n], "dtype": dtype, "shape": shape}
        offsets[name] = (at, at + n)
        at += n
    h = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h, offsets


class _Sink:
    """A file written once, front to back, and hashed as it is written."""

    def __init__(self, path: Path):
        self.f, self.h, self.size = open(path, "wb"), hashlib.sha256(), 0

    def write(self, b: bytes | memoryview) -> None:
        self.f.write(b)
        self.h.update(b)
        self.size += len(b)

    def close(self) -> dict[str, Any]:
        self.f.close()
        return {"sha256": self.h.hexdigest(), "size": self.size}


def _read(f, start: int, end: int) -> bytes:
    f.seek(start)
    b = f.read(end - start)
    if len(b) != end - start:
        raise ValueError(f"truncated safetensors file ({len(b)} of {end - start} bytes)")
    return b


def make_shard(src: Path, dst: Path, vocab: int, recipe: Recipe = FP8) -> tuple[dict[str, Any], dict[str, str], int, int]:
    """Write the recipe's shard of `src` to `dst`: (file digest, tensor -> shard name, vocabulary-sized tensors, data bytes)."""
    base, header = read_header(src)
    layout = shard_layout(header, vocab, recipe)
    r = recipe.rows_per_scale
    head, offsets = encode_header(layout)
    scales: dict[str, np.ndarray] = {}
    with open(src, "rb") as f:
        def weight(name: str, lo: int = 0, hi: int | None = None) -> np.ndarray:
            info = header[name]
            n, k = (int(d) for d in info["shape"])
            hi = n if hi is None else hi
            e = _ELEM[info["dtype"]]
            a, _ = info["data_offsets"]
            return _WIDEN[info["dtype"]](_read(f, base + a + lo * k * e, base + a + hi * k * e)).reshape(hi - lo, k)

        for name, _, _, kind, source in layout:
            if kind == "scale":
                n = int(header[source]["shape"][0])
                scales[source] = np.concatenate([recipe.scales(weight(source, lo, min(lo + STRIPE, n)))
                                                 for lo in range(0, n, STRIPE)])
        sink = _Sink(dst)
        sink.write(head)
        for name, _, _, kind, source in layout:
            if kind == "scale":
                sink.write(scales[source].astype("<f4").tobytes())
            elif kind == "weight":
                n = int(header[source]["shape"][0])
                for lo in range(0, n, STRIPE):
                    hi = min(lo + STRIPE, n)
                    sink.write(recipe.quantize(weight(source, lo, hi), scales[source][lo // r: hi // r]).tobytes())
            else:
                a, b = header[source]["data_offsets"]
                for at in range(a, b, _CHUNK):
                    sink.write(_read(f, base + at, base + min(at + _CHUNK, b)))
        digest = sink.close()
    n_vocab = sum(1 for n, i in header.items() if n != "__metadata__" and len(i["shape"]) == 2 and not is_linear(i, vocab))
    return digest, {name: dst.name for name, *_ in layout}, n_vocab, sum(b - a for a, b in offsets.values())


def dense_vocab(cfg: dict[str, Any], recipe: Recipe = FP8) -> int:
    """The base config's vocabulary size, refusing what the recipe does not make: a quantized base, a MoE (its router is a 2-D
    weight the integration keeps in bf16, and the integration models no FP8 MoE), and under a recipe that makes the LM head, tied
    embeddings (`tie_word_embeddings` true or absent: the LM head would be the embedding's own tensor)."""
    if cfg.get("quantization_config"):
        raise ValueError(f"the base checkpoint is already quantized ({cfg['quantization_config'].get('quant_method')!r})")
    if (cfg.get("num_experts") or cfg.get("num_local_experts")) and not recipe.moe:
        raise ValueError("a mixture-of-experts checkpoint: the recipe makes dense checkpoints only")
    if not isinstance(cfg.get("vocab_size"), int):
        raise ValueError("config.json names no integer vocab_size")
    if recipe.lm_head and cfg.get("tie_word_embeddings", True) is not False:
        raise ValueError(f"tie_word_embeddings is {cfg.get('tie_word_embeddings', '<absent>')!r}: {recipe.id} quantizes the LM head, "
                         "which a tied checkpoint shares with the embedding")
    return cfg["vocab_size"]


def routers(cfg: dict[str, Any], recipe: Recipe = FP8) -> tuple[tuple[int, int], ...]:
    """The routers' shape, [num_experts, hidden_size], of a MoE config under a recipe that makes one (else none): vLLM builds a
    router with no quantization config, so the recipe copies every 2-D tensor of that shape as it is."""
    experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    if not (recipe.moe and experts):
        return ()
    if not isinstance(cfg.get("hidden_size"), int):
        raise ValueError("a mixture-of-experts config.json names no integer hidden_size")
    return ((int(experts), cfg["hidden_size"]),)


def fp8_config(base_config: bytes, recipe: Recipe = FP8) -> bytes:
    cfg = json.loads(base_config)
    dense_vocab(cfg, recipe)
    kept = [] if recipe.lm_head else ["lm_head"]
    cfg["quantization_config"] = dict(sorted(dict(recipe.quantization, modules_to_not_convert=kept).items()))
    return (json.dumps(cfg, indent=2, ensure_ascii=False) + "\n").encode()


def pin_revision(base_repo: str, base_revision: str, made: dict[str, str], recipe: Recipe = FP8) -> str:
    doc = {"base": {"repo": base_repo, "revision": base_revision}, "files": dict(sorted(made.items())), "recipe": recipe.id}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:40]


def map_shards(fn: Callable[..., Any], args: list[tuple[Any, ...]], jobs: int = 1) -> list[Any]:
    """[fn(*a) for a in args], `jobs` at a time in threads (the shards are independent, and numpy's array loops, file reads and
    sha256 release the GIL)."""
    if jobs <= 1 or len(args) <= 1:
        return [fn(*a) for a in args]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(jobs, len(args))) as pool:
        return list(pool.map(lambda a: fn(*a), args))


def make(base_dir: Path, out_dir: Path, shards: list[str], *, base_repo: str, base_revision: str, recipe: Recipe = FP8,
         jobs: int = 1) -> dict[str, Any]:
    """Write the recipe's checkpoint of the base shards `shards` in `base_dir` into `out_dir` (created; must not exist), `jobs`
    shards at a time (the bytes do not depend on it).
    Returns {"revision", "files": {name: {sha256, size, digest_source}}, "config_sha256", "safetensors_sha256"}."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(parents=True)
    shards = sorted(shards)
    if not shards:
        raise ValueError(f"no safetensors shard in {base_dir}")
    base_config = (base_dir / CONFIG).read_bytes()
    plan = (recipe.plan or dense_vocab)(json.loads(base_config), recipe)
    recipe = replace(recipe, kept=routers(json.loads(base_config), recipe))
    files: dict[str, dict[str, Any]] = {}
    weight_map: dict[str, str] = {}
    n_vocab = total = 0
    made = map_shards(recipe.shard or make_shard, [(base_dir / fn, out_dir / fn, plan, recipe) for fn in shards], jobs)
    for fn, (digest, names, v, n) in zip(shards, made):
        files[fn] = dict(digest, digest_source="recipe")
        weight_map.update(names)
        n_vocab += v
        total += n
    vocab = getattr(plan, "vocab", plan)
    if not n_vocab:
        raise ValueError(f"no tensor has the config's vocab_size {vocab} as a dimension: the embedding is not where the recipe looks")
    if recipe.lm_head and LM_HEAD not in weight_map:
        raise ValueError(f"{recipe.id} makes {LM_HEAD}, and no shard holds one")
    if recipe.config is None:
        cfg = fp8_config(base_config, recipe)
    else:
        tensors = {k: (v["dtype"], v["shape"]) for fn in shards for k, v in read_header(out_dir / fn)[1].items() if k != "__metadata__"}
        cfg = recipe.config(base_config, recipe, tensors)
    (out_dir / CONFIG).write_bytes(cfg)
    files[CONFIG] = {"sha256": hashlib.sha256(cfg).hexdigest(), "size": len(cfg), "digest_source": "recipe"}
    if (base_dir / INDEX).exists():
        idx = (json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
                          indent=2, sort_keys=True) + "\n").encode()
        (out_dir / INDEX).write_bytes(idx)
        files[INDEX] = {"sha256": hashlib.sha256(idx).hexdigest(), "size": len(idx), "digest_source": "recipe"}
    made = {n: d["sha256"] for n, d in files.items()}
    for p in sorted(base_dir.rglob("*")):
        rel = p.relative_to(base_dir).as_posix()
        if p.is_file() and rel not in files and rel != INDEX and any(fnmatch.fnmatch(rel, pat) for pat in COPY_PATTERNS):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out_dir / rel)
            files[rel] = {"sha256": _sha256(out_dir / rel), "size": (out_dir / rel).stat().st_size, "digest_source": "base"}
    return {"revision": pin_revision(base_repo, base_revision, made, recipe), "files": dict(sorted(files.items())),
            "config_sha256": files[CONFIG]["sha256"], "safetensors_sha256": {fn: files[fn]["sha256"] for fn in shards}}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# -------------------------------------------------------------------------------------------------------------- the manifest

def snapshot_dir(hf_home: str, repo: str, revision: str) -> Path:
    return Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def base_entry(entry: dict[str, Any], entries: list[dict[str, Any]], recipe: Recipe = FP8) -> dict[str, Any]:
    """The BF16 pin a recipe entry was made from (its `recipe.base_repo` at `recipe.base_revision`, with shard sha256s)."""
    r = entry.get("recipe") or {}
    if r.get("id") != recipe.id:
        raise UsageError(f"{entry.get('role')}: recipe {r.get('id')!r} is not {recipe.id!r}")
    hits = [c for c in entries if c.get("repo") == r.get("base_repo") and c.get("revision") == r.get("base_revision")
            and c.get("safetensors_sha256")]
    if not hits:
        raise UsageError(f"{entry.get('role')}: no pin of {r.get('base_repo')}@{r.get('base_revision')} with shard sha256s")
    return min(hits, key=lambda c: not c.get("config_sha256"))


def verified_base(base: dict[str, Any], hf_home: str | None, jobs: int = 1) -> Path:
    """The base snapshot on this host (under `hf_home`, else the pin's local_path), every pinned shard and the config checked
    against the pin (fail-closed), `jobs` files at a time."""
    d = snapshot_dir(hf_home, base["repo"], base["revision"]) if hf_home else Path(base.get("local_path") or "")
    if not d.is_dir():
        raise UsageError(f"no local snapshot of {base['repo']}@{base['revision'][:12]} at {d} (seed the base pin first)")
    want = dict(base["safetensors_sha256"], **({CONFIG: base["config_sha256"]} if base.get("config_sha256") else {}))
    names = sorted(fn for fn in want if (d / fn).is_file())
    got = dict(zip(names, map_shards(_sha256, [(d / fn,) for fn in names], jobs)))
    bad = [fn for fn, sha in sorted(want.items()) if got.get(fn) != sha]
    if bad:
        raise UsageError(f"{base['repo']}@{base['revision'][:12]} in {d}: not the pinned bytes: {bad}")
    return d


def _make_into(snapshots: Path, base_dir: Path, base: dict[str, Any], recipe: Recipe = FP8, jobs: int = 1) -> dict[str, Any]:
    """Make into `snapshots/<revision>` (written beside it, then renamed into place)."""
    snapshots.mkdir(parents=True, exist_ok=True)
    tmp = snapshots / f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    pin = make(base_dir, tmp, list(base["safetensors_sha256"]), base_repo=base["repo"], base_revision=base["revision"], recipe=recipe,
               jobs=jobs)
    final = snapshots / pin["revision"]
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return dict(pin, local_path=str(final))


def remake(entry: dict[str, Any], entries: list[dict[str, Any]], hf_home: str | None = None,
           recipe: Recipe = FP8, jobs: int = 1) -> tuple[dict[str, Any], list[str]]:
    """Re-make a recipe pin on this host from its BF16 pin (checked first): (the made pin, what differs from `entry`).  The
    snapshot goes to `entry["local_path"]`, or beside the base under `hf_home`."""
    base = base_entry(entry, entries, recipe)
    final = snapshot_dir(hf_home, entry["repo"], entry["revision"]) if hf_home else Path(entry["local_path"])
    pin = _make_into(final.parent, verified_base(base, hf_home, jobs), base, recipe, jobs)
    bad = [fn for fn, d in sorted(entry["files"].items()) if d.get("sha256") and pin["files"].get(fn, {}).get("sha256") != d["sha256"]]
    bad += sorted(set(pin["files"]) - set(entry["files"]))
    if pin["revision"] != entry["revision"]:
        bad.append(f"revision {pin['revision']}")
    return pin, bad


@dataclass(frozen=True, kw_only=True)
class MakeOptions:
    """Re-make a registered FP8 pin (`recipe` in checkpoints.json) from its BF16 pin and check every file's sha256."""

    role: str = option("--role", help="the FP8 pin's role in checkpoints.json")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None,
                                 help="HF cache holding the base snapshot; the FP8 snapshot goes beside it (default: the pins' local_path)")


@dataclass(frozen=True, kw_only=True)
class PinOptions:
    """Make an FP8 checkpoint from a BF16 pin and print its new checkpoints.json entry (to append; the manifest is append-only)."""

    base_role: str = option("--base-role", help="the BF16 pin's role, or its repo id when the role is prose")
    role: str = option("--role", help="the new FP8 pin's role")
    repo: str = option("--repo", help="the new pin's repo id (a local namespace: no hub repo holds these bytes)")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None)
    by: str = option("--by", default="fp8-checkpoint pin", help="resolved_by of the entry")
    at: str = option("--at", help="resolved_at of the entry (UTC, ISO 8601)")


COMMANDS = {"make": MakeOptions, "pin": PinOptions}


def main(a: MakeOptions | PinOptions) -> int:
    return run(a, FP8, "fp8-checkpoint")


def run(a: Any, recipe: Recipe, command: str) -> int:
    """`make` (a `MakeOptions`-shaped `a`: role, checkpoints, hf_home, optionally jobs) or `pin` (`PinOptions`-shaped) under `recipe`."""
    entries = C.checkpoint_entries(a.checkpoints)
    home = a.hf_home or C.checkpoints_hf_home() or ""
    if not hasattr(a, "base_role"):
        entry = next((c for c in entries if c.get("role") == a.role), None)
        if entry is None:
            raise UsageError(f"no checkpoints.json entry with role {a.role!r}")
        pin, bad = remake(entry, entries, a.hf_home, recipe, getattr(a, "jobs", 1))
        print(json.dumps({"role": a.role, "local_path": pin["local_path"], "ok": not bad, "mismatch": bad}))
        return 1 if bad else 0
    base = next((c for c in entries if c.get("role") == a.base_role and c.get("safetensors_sha256")), None) \
        or next((c for c in entries if c.get("repo") == a.base_role and c.get("safetensors_sha256")), None)
    if base is None:
        raise UsageError(f"no BF16 pin with role or repo {a.base_role!r} and shard sha256s")
    if any(c.get("role") == a.role for c in entries):
        raise UsageError(f"role {a.role!r} is already pinned")
    pin = _make_into(snapshot_dir(home, a.repo, "_").parent, verified_base(base, a.hf_home, getattr(a, "jobs", 1)), base, recipe,
                     getattr(a, "jobs", 1))
    local = snapshot_dir(C.checkpoints_hf_home() or home, a.repo, pin["revision"])
    entry = {"config_sha256": pin["config_sha256"], "downloaded": False, "files": pin["files"], recipe.of: base["repo"], "gated": False,
             "local_path": str(local),
             "note": f"made offline by `verity-vllm {command}` ({recipe.id}) from the {base['repo']} pin; no hub repo holds it",
             "recipe": {"base_repo": base["repo"], "base_revision": base["revision"], "id": recipe.id}, "repo": a.repo,
             "resolved_at": a.at, "resolved_by": a.by, "revision": pin["revision"], "role": a.role,
             "safetensors_sha256": pin["safetensors_sha256"]}
    print(json.dumps(entry, indent=1))
    print(f"made {pin['local_path']}", file=sys.stderr)
    return 0
