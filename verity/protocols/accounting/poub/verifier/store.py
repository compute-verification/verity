"""A store a pinned BF16 checkpoint lays out, and the payload salts every store's leaves take: the prover lays its store
out from its own checkpoint (`arena_of`, `Store`, `node.py --model-dir`) and salts its payload tree's leaves under the
verifier's key (`store_salt`). The verifier holds no checkpoint: what the store holds is the developer's choice.
`checkpoint.sh` holds a node's checkpoint to core's pin (`read_pin`, `check`), and `auditor_line.py` reads a pin
(`read_pin`).

~~~text
pin        the checkpoint's files with their SHA-512s, in the auditor's model registry's shape (`verity/protocols/
           compose/device/live_challenger/models.py`): {name, checkpoint: {file: SHA-512 hex}, checkpoint_sha512},
           checkpoint_sha512 the SHA-512 of checkpoint's canonical JSON, or a role's entry of core's pin
           (`models.json`, `read_pin`); every file is held to it before anything reads it (`check`)
arena      one engine's weights (one GPU, tensor parallelism 1): every parameter of vLLM's model with two or more
           dimensions, in its module order, each from a chunk boundary, as the checkpoint's BF16 bytes: embed_tokens;
           each decoder layer's self_attn.qkv_proj (q's, k's and v's rows in turn), o_proj, mlp.gate_up_proj (gate's
           rows, then up's) and down_proj; then lm_head unless the embedding is tied. The rest of the arena is zeros;
           norms and biases (one dimension) stay outside it
store      `engines` arenas of `arena_blocks` each, in GPU order, then `host_blocks` of zero filler: each arena's
           weights chunks labelled (weights, hbm) and its rest (filler, hbm), the host tier (filler, host)
salt       store_salt(key, label, c) = `codec.xof`(m)[0 : 192], SHA512(m ‖ u32be 0) ‖ SHA512(m ‖ u32be 1) ‖
           SHA512(m ‖ u32be 2), m = "verity/store/salt/v1\0" ‖ key ‖ label ‖ u64be c (`SALT_VERSION`)
payload    `audit.payload_tree` over the store, chunk j salted with store_salt(key, PAYLOAD_LABEL, j) under the
           verifier's key (its payload-key message)
~~~
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import codec, receipts

BLOCK = codec.BLOCK_BYTES
SALT_VERSION = "verity/store/salt/v1"
SALT_DOMAIN = SALT_VERSION.encode() + b"\0"
PAYLOAD_LABEL = hashlib.sha512(b"verity/remote-session/payload-salts/v0").digest()
KEY_BYTES = 32
"""The verifier's payload salt key."""
ARCHITECTURES = ("LlamaForCausalLM", "Qwen2ForCausalLM")
"""The models whose vLLM module order `arena` states: a decoder layer's attention, then its MLP, no other 2-D parameter."""
LAYER = (("self_attn.qkv_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
         ("self_attn.o_proj", ("self_attn.o_proj",)),
         ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
         ("mlp.down_proj", ("mlp.down_proj",)))
"""A decoder layer's 2-D parameters in vLLM's module order at TP = 1, each with the checkpoint tensors whose rows it
holds in turn (`MergedColumnParallelLinear`, `QKVParallelLinear`)."""
VOCAB_PADDING = 64
"""vLLM pads an embedding's rows to a multiple of this (`DEFAULT_VOCAB_PADDING_SIZE`)."""
SHA512_HEX = re.compile(r"[0-9a-f]{128}")


class Refused(ValueError):
    """A pin, a checkpoint or a store this module refuses, with why."""


def store_salt(key: bytes, label: bytes, c: int) -> bytes:
    """Salt `c` under `key` and `label` (the module's docstring)."""
    return codec.xof(SALT_DOMAIN + key + label + struct.pack(">Q", c), receipts.SALT_BYTES)


def canonical_json(obj) -> bytes:
    """`verity.circuits.codec.canonical_json`'s bytes, which this protocol may not import."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def checkpoint_sha512(files: dict[str, str]) -> str:
    """A checkpoint's digest, the registry's: SHA-512 of the canonical JSON of its ``{file: SHA-512 hex}``."""
    return hashlib.sha512(canonical_json(files)).hexdigest()


_NUMBER = object()


def _number(literal: str) -> object:
    """A JSON number, which no pin holds, read as an opaque value; refused with an exponent past 2^64."""
    e = literal.lower().partition("e")[2].lstrip("+")
    if e and not e.startswith("-") and (len(e.lstrip("0")) > 20 or int(e) > 2 ** 64):
        raise ValueError(f"the number {literal[:32]} has an exponent past 2^64")
    return _NUMBER


def _constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _loads(text: str):
    """JSON as a pin is read: numbers opaque (`_number`), no NaN or Infinity, and no string holding a lone surrogate
    (its encoding raises)."""
    doc = json.loads(text, parse_int=_number, parse_float=_number, parse_constant=_constant)
    json.dumps(doc, ensure_ascii=False, default=lambda _: None).encode()
    return doc


def _pinned(checkpoint) -> str | None:
    """A registry line's checkpoint as its lines are compared: the canonical JSON of a map to strings, else None."""
    if isinstance(checkpoint, dict) and all(type(h) is str for h in checkpoint.values()):
        return canonical_json(checkpoint)
    return None


def pin_bytes(path: Path) -> bytes:
    """The bytes of the pin file at `path`."""
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise Refused(f"the pin {path} is not readable: {e}") from None


def read_pin(path: Path, role: str | None = None) -> dict:
    """`pin_of` on the file at `path`."""
    return pin_of(pin_bytes(path), str(path), role)


def pin_of(data: bytes, path: str, role: str | None = None) -> dict:
    """A pin in the registry's shape from the bytes of the file at `path`, its `checkpoint_sha512` held to its files:
    {name, checkpoint, checkpoint_sha512}. A registry line names its model `model`, which is read as `name`; a registry
    itself (`models.jsonl`, a line a registration) is read as its lines of `role`, which must pin one checkpoint; core's
    pin (`verity/protocols/accounting/pouw/verifier/lean/check/models.json`, {ROLE: {checkpoint_sha512, files: {file: SHA-512 hex},
    ...}}) as its entry of `role`, named `role`, its `files` the checkpoint."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused(f"the pin {path} is not UTF-8") from None
    form = "{name, checkpoint: {file: SHA-512 hex}, checkpoint_sha512}"
    try:
        doc = _loads(text)
    except ValueError:
        lines = []
        for k, x in enumerate(text.split("\n"), 1):
            if x.strip(" \t\r"):
                try:
                    lines.append(_loads(x))
                except ValueError:
                    raise Refused(f"the pin {path} is neither JSON nor JSON lines: its line {k} is not JSON") from None
        mine = [x for x in lines if isinstance(x, dict) and role is not None and x.get("model", x.get("name")) == role]
        checkpoints = {_pinned(x.get("checkpoint")) for x in mine}
        if len(checkpoints) != 1:
            raise Refused(f"the registry {path} has {len(mine)} lines of {role}, pinning {len(checkpoints)} "
                          "checkpoints: one is the pin") from None
        doc = mine[-1]
    if isinstance(doc, dict) and doc and all(isinstance(v, dict) for v in doc.values()):
        if role not in doc:
            raise Refused(f"the pin {path} pins no role {canonical_json(role).decode()}: its roles are "
                          f"{canonical_json(sorted(doc)).decode()}")
        entry, form = doc[role], "{ROLE: {checkpoint_sha512, files: {file: SHA-512 hex}, ...}}"
        doc = {"name": role, "checkpoint": entry.get("files"), "checkpoint_sha512": entry.get("checkpoint_sha512")}
    name = doc.get("name", doc.get("model")) if isinstance(doc, dict) else None
    files = doc.get("checkpoint") if isinstance(doc, dict) else None
    if (type(name) is not str or not name or not isinstance(files, dict) or not files
            or any(type(f) is not str or not f or "/" in f or f.startswith(".") or type(h) is not str
                   or not SHA512_HEX.fullmatch(h) for f, h in files.items())):
        raise Refused(f"the pin {path} is not {form} with a file name (no directory) a file")
    if doc.get("checkpoint_sha512") != checkpoint_sha512(files):
        raise Refused(f"the pin {path}'s checkpoint_sha512 is not the SHA-512 of its checkpoint's canonical JSON "
                      f"({checkpoint_sha512(files)[:16]}...)")
    if "config.json" not in files or not any(f.endswith(".safetensors") for f in files):
        raise Refused(f"the pin {path} pins no config.json or no .safetensors file: the layout reads both")
    return {"name": name, "checkpoint": dict(files), "checkpoint_sha512": doc["checkpoint_sha512"]}


def file_sha512(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha512").hexdigest()


def pool(workers: int) -> ProcessPoolExecutor:
    """Worker processes, spawned: a caller that runs threads would have a fork copy them mid-flight."""
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


def check(directory: Path, pin: dict, workers: int = 1) -> None:
    """Refused unless each file `pin` names is a regular file in `directory` (no symlink) with the pinned SHA-512."""
    names = sorted(pin["checkpoint"])
    for n in names:
        p = Path(directory) / n
        if p.is_symlink() or not p.is_file():
            raise Refused(f"{p}: not a regular file (no symlink) of the pinned checkpoint {pin['name']}")
    paths = [str(Path(directory) / n) for n in names]
    if workers > 1 and len(paths) > 1:
        with pool(min(workers, len(paths))) as ex:
            got = list(ex.map(file_sha512, paths))
    else:
        got = [file_sha512(p) for p in paths]
    for n, h in zip(names, got):
        if h != pin["checkpoint"][n]:
            raise Refused(f"{Path(directory) / n}: SHA-512 {h[:16]}..., not the pin's {pin['checkpoint'][n][:16]}... of "
                          f"{pin['name']}")


@dataclass(frozen=True)
class Part:
    tensor: str
    file: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Entry:
    name: str
    parts: tuple[Part, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Arena:
    """One engine's arena of weights (the module's docstring): its entries, each from a chunk boundary, and the chunks
    up to the end of the last one's."""
    directory: str
    entries: tuple[Entry, ...]
    weights_blocks: int

    @property
    def weights_bytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def read(self, lo: int, n: int) -> bytes:
        """The arena's bytes lo … lo + n − 1 as an engine holds them, zeros past each entry and past the last."""
        out, fds = bytearray(n), {}
        try:
            for e in self.entries:
                if e.offset >= lo + n or e.offset + e.nbytes <= lo:
                    continue
                at = e.offset
                for p in e.parts:
                    a, b = max(lo, at), min(lo + n, at + p.nbytes)
                    if a < b:
                        if p.file not in fds:
                            fds[p.file] = os.open(p.file, os.O_RDONLY)
                        got = os.pread(fds[p.file], b - a, p.offset + a - at)
                        if len(got) != b - a:
                            raise Refused(f"{p.file}: tensor {p.tensor} is cut short")
                        out[a - lo:b - lo] = got
                    at += p.nbytes
        finally:
            for fd in fds.values():
                os.close(fd)
        return bytes(out)


def tensors(directory: Path, files) -> dict[str, tuple[str, dict, int]]:
    """Each tensor of the `.safetensors` files among `files`: (path, its header entry, its data's first byte); Refused
    when a tensor is in two of them."""
    out: dict[str, tuple[str, dict, int]] = {}
    for name in sorted(f for f in files if f.endswith(".safetensors")):
        path = str(Path(directory) / name)
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for t, meta in header.items():
            if t == "__metadata__":
                continue
            if t in out:
                raise Refused(f"tensor {t} is in both {Path(out[t][0]).name} and {name}")
            out[t] = (path, meta, 8 + n)
    return out


def arena(directory: Path, files) -> Arena:
    """The arena an engine serving the checkpoint in `directory` (its `files`) holds: the layout of the model vLLM
    builds from it (the module's docstring), or Refused for a model or a tensor it doesn't cover."""
    try:
        config = json.loads((Path(directory) / "config.json").read_text())
    except (OSError, ValueError):
        raise Refused(f"{directory} has no readable config.json") from None
    archs = config.get("architectures") or []
    if len(archs) != 1 or archs[0] not in ARCHITECTURES:
        raise Refused(f"config.json's architectures {archs}: one of {ARCHITECTURES}, whose vLLM module order the layout "
                      "states")
    layers, vocab = config.get("num_hidden_layers"), config.get("vocab_size")
    if type(layers) is not int or layers < 1 or type(vocab) is not int or vocab < 1:
        raise Refused("config.json names no num_hidden_layers or vocab_size")
    if vocab % VOCAB_PADDING:
        raise Refused(f"vocab_size {vocab} is not a multiple of {VOCAB_PADDING}: vLLM pads the embedding's rows, which "
                      "the checkpoint doesn't hold")
    found = tensors(directory, files)

    def part(t: str) -> tuple[Part, tuple[int, ...]]:
        if t not in found:
            raise Refused(f"the checkpoint has no tensor {t}")
        path, meta, base = found[t]
        shape = tuple(meta.get("shape") or ())
        lo, hi = meta["data_offsets"]
        if meta.get("dtype") != "BF16" or len(shape) != 2:
            raise Refused(f"tensor {t} is {meta.get('dtype')} {list(shape)}: the layout covers 2-D BF16 weights, which "
                          "vLLM holds as they are at dtype bfloat16")
        if hi - lo != math.prod(shape) * 2:
            raise Refused(f"tensor {t}'s bytes are not its shape {list(shape)}")
        return Part(t, path, base + lo, hi - lo), shape

    def entry(name: str, sources: tuple[str, ...]) -> tuple[str, tuple[Part, ...]]:
        got = [part(t) for t in sources]
        if len({s[1] for _, s in got}) != 1:
            raise Refused(f"{name}'s parts {list(sources)} differ in columns: {[list(s) for _, s in got]}")
        return name, tuple(p for p, _ in got)

    embed = part("model.embed_tokens.weight")
    if embed[1][0] != vocab:
        raise Refused(f"model.embed_tokens.weight has {embed[1][0]} rows, not vocab_size {vocab}")
    plan = [("model.embed_tokens.weight", (embed[0],))]
    for i in range(layers):
        plan += [entry(f"model.layers.{i}.{m}.weight", tuple(f"model.layers.{i}.{p}.weight" for p in ps))
                 for m, ps in LAYER]
    if not config.get("tie_word_embeddings", False):
        head = part("lm_head.weight")
        if head[1] != embed[1]:
            raise Refused(f"lm_head.weight is {list(head[1])}, not the embedding's {list(embed[1])}")
        plan.append(("lm_head.weight", (head[0],)))
    entries, at = [], 0
    for name, parts in plan:
        n = sum(p.nbytes for p in parts)
        entries.append(Entry(name, parts, at, n))
        at += -(-n // BLOCK) * BLOCK
    return Arena(str(directory), tuple(entries), at // BLOCK)


def arena_of(directory: Path) -> Arena:
    """`arena` over every file in `directory`: the prover's own checkpoint, unpinned."""
    return arena(directory, sorted(p.name for p in Path(directory).iterdir() if p.is_file()))


@dataclass(frozen=True)
class Store:
    """`engines` arenas of `arena_blocks` chunks, then `host_blocks` of zero filler (the module's docstring)."""
    arena: Arena
    engines: int
    arena_blocks: int
    host_blocks: int = 0

    def __post_init__(self) -> None:
        if type(self.engines) is not int or self.engines < 1 or type(self.host_blocks) is not int or self.host_blocks < 0:
            raise Refused(f"a store is one or more arenas and no fewer than zero host blocks, not {self.engines} and "
                          f"{self.host_blocks}")
        if type(self.arena_blocks) is not int or self.arena_blocks < self.arena.weights_blocks:
            raise Refused(f"the weights take {self.arena.weights_blocks} blocks, more than an arena's "
                          f"{self.arena_blocks}")

    @property
    def blocks(self) -> int:
        return self.engines * self.arena_blocks + self.host_blocks

    def regions(self) -> list[dict]:
        """The store's layout as a hello carries it (`verifier.regions_of`, `node.ServingNode.layout`)."""
        out, W, A = [], self.arena.weights_blocks, self.arena_blocks
        for e in range(self.engines):
            out.append({"kind": "weights", "tier": "hbm", "first": e * A, "blocks": W})
            if A > W:
                out.append({"kind": "filler", "tier": "hbm", "first": e * A + W, "blocks": A - W})
        if self.host_blocks:
            out.append({"kind": "filler", "tier": "host", "first": self.engines * A, "blocks": self.host_blocks})
        return out

    def payloads(self) -> list[bytes]:
        """Every chunk's payload, a small store's (the CPU backend's)."""
        raw = self.arena.read(0, self.arena.weights_blocks * BLOCK)
        weights = [raw[j * BLOCK:(j + 1) * BLOCK] for j in range(self.arena.weights_blocks)]
        zero = bytes(BLOCK)
        one = weights + [zero] * (self.arena_blocks - len(weights))
        return one * self.engines + [zero] * self.host_blocks

    def shape(self) -> dict:
        return {"engines": self.engines, "arena_blocks": self.arena_blocks, "host_blocks": self.host_blocks,
                "weights_blocks": self.arena.weights_blocks, "weights_bytes": self.arena.weights_bytes,
                "entries": len(self.arena.entries), "blocks": self.blocks}


__all__ = ["Arena", "BLOCK", "KEY_BYTES", "PAYLOAD_LABEL", "Refused", "SALT_DOMAIN", "Store", "arena", "arena_of",
           "check", "checkpoint_sha512", "file_sha512", "pool", "read_pin", "store_salt"]
