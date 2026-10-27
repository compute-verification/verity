"""A registered store bound to a pinned model: what a pinned BF16 checkpoint alone fixes, so the verifier recomputes the
payload root the prover's hello must carry and refuses any other (`Binding`), in the tree whose git hash names the
verifier a period's circuit commits to (`verity/remote-bandwidth/v0 verifier tree <tree>`,
`sampled_proofs.challenger.credited_session`). The decision from what the files hold is Lean's
(`lean/RemoteAudit/Store.lean`), which decides a session's `bound`: the verifier's Lean process takes the pin's bytes
and the weights chunks' digests at startup (`bind`) and decides each hello (`hold`), `Binding` beside it as the
reference, a disagreement refusing the hello (`benchmarks/remote_session/verifier.py`). This is held to it by
`lean/tests/test_binding.py`: the pin from its file's bytes (`pin_of`, `model_pin`), `checkpoint_sha512`, `store_salt`,
`expected_root` from the weights chunks' digests, and `Binding`'s start and `hold`. Reading the files (`check`,
`tensors`, `weight_digests`) and the arena's layout (`arena`) are Python only (`README.md`, "Which side is the
definition").

~~~text
pin        the checkpoint's files with their SHA-512s, in the auditor's model registry's shape (`benchmarks/
           live_challenger/models.py`): {name, checkpoint: {file: SHA-512 hex}, checkpoint_sha512}, checkpoint_sha512
           the SHA-512 of checkpoint's canonical JSON, or a role's entry of core's pin (`models.json`, `read_pin`);
           every file is held to it before anything reads it (`check`)
arena      one engine's weights (one GPU, tensor parallelism 1): every parameter of vLLM's model with two or more
           dimensions, in its module order, each from a chunk boundary, as the checkpoint's BF16 bytes: embed_tokens;
           each decoder layer's self_attn.qkv_proj (q's, k's and v's rows in turn), o_proj, mlp.gate_up_proj (gate's
           rows, then up's) and down_proj; then lm_head unless the embedding is tied. The rest of the arena is zeros;
           norms and biases (one dimension) stay outside it
store      `engines` arenas of `arena_blocks` each, in GPU order, then `host_blocks` of zero filler: each arena's
           weights chunks labelled (weights, hbm) and its rest (filler, hbm), the host tier (filler, host)
payload    `audit.payload_tree` over the store, chunk j salted with store_salt(key, PAYLOAD_LABEL, j) under the
           verifier's key (its payload-key message), so its root follows from the pin, the store's shape and that key
           (`expected_root`, in worker processes: at a whole node's 130M chunks 665 s on 11 cores, the Lean's
           `Store.expectedRoot` 436 s on 11 threads, 23 Oct, `benchmarks/remote_session/binding_bench.py`)
binding    at startup the pin read, every pinned file held to it, the arena laid out and its weights chunks' digests
           computed; at each hello (`Binding.hold`) the store its size makes and that store's payload root under the
           verifier's key, against the hello's, else Refused with why
~~~
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import mmap
import multiprocessing
import os
import re
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from verity.ml.commitments.merkle import merkle_depth

from . import audit, codec, receipts

BLOCK = codec.BLOCK_BYTES
SALT_DOMAIN = b"verity/store/salt/v0\0"
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
DIGESTS_A_TASK = 4096
CHUNK_LOG2 = 16


class Refused(ValueError):
    """A pin, a checkpoint or a store this module refuses, with why."""


def store_salt(key: bytes, label: bytes, c: int) -> bytes:
    return hashlib.shake_256(SALT_DOMAIN + key + label + struct.pack(">Q", c)).digest(receipts.SALT_BYTES)


def canonical_json(obj) -> bytes:
    """`verity.ml.circuits.codec.canonical_json`'s bytes, which this protocol may not import."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def checkpoint_sha512(files: dict[str, str]) -> str:
    """A checkpoint's digest, the registry's: SHA-512 of the canonical JSON of its ``{file: SHA-512 hex}``."""
    return hashlib.sha512(canonical_json(files)).hexdigest()


_NUMBER = object()


def _number(literal: str) -> object:
    """A JSON number, which no pin holds, read as an opaque value; refused with an exponent past 2^64, as Lean's
    parser refuses it."""
    e = literal.lower().partition("e")[2].lstrip("+")
    if e and not e.startswith("-") and (len(e.lstrip("0")) > 20 or int(e) > 2 ** 64):
        raise ValueError(f"the number {literal[:32]} has an exponent past 2^64")
    return _NUMBER


def _constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _loads(text: str):
    """JSON as `RemoteAudit.Store.loads` reads it: numbers opaque (`_number`), no NaN or Infinity, and no string
    holding a lone surrogate, which `json.loads` keeps and Lean's parser reads as U+FFFD (its encoding raises)."""
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


def read_pin(path: Path, role: str | None = None, *, model: bool = False) -> dict:
    """`pin_of` on the file at `path` (`model_pin` with `model`)."""
    return (model_pin if model else pin_of)(pin_bytes(path), str(path), role)


def pin_of(data: bytes, path: str, role: str | None = None) -> dict:
    """A pin in the registry's shape from the bytes of the file at `path`, its `checkpoint_sha512` held to its files:
    {name, checkpoint, checkpoint_sha512}. A registry line names its model `model`, which is read as `name`; a registry
    itself (`models.jsonl`, a line a registration) is read as its lines of `role`, which must pin one checkpoint; core's
    pin (`verity/core/protocols/pouw/lean/check/models.json`, {ROLE: {checkpoint_sha512, files: {file: SHA-512 hex},
    ...}}) as its entry of `role`, named `role`, its `files` the checkpoint. `RemoteAudit.Store.pinOf` is its
    definition, refusals word for word (`lean/tests/test_binding.py`)."""
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
                          "checkpoints: one is the pin")
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


def model_pin(data: bytes, path: str, role: str) -> dict:
    """`pin_of` for the binding of `--model role`: the pin must be `role`'s (`RemoteAudit.Store.modelPin`)."""
    pin = pin_of(data, path, role)
    if pin["name"] != role:
        raise Refused(f"the pin {path} is {pin['name']}'s, not --model {role}'s")
    return pin


def file_sha512(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha512").hexdigest()


def pool(workers: int) -> ProcessPoolExecutor:
    """Worker processes, spawned: the verifier that binds runs threads, which a fork would copy mid-flight."""
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


def _weight_digests(a: Arena, lo: int, n: int) -> bytes:
    raw = a.read(lo * BLOCK, n * BLOCK)
    return b"".join(codec.digest(raw[q * BLOCK:(q + 1) * BLOCK]) for q in range(n))


def weight_digests(a: Arena, path: Path, workers: int = 1) -> None:
    """Each weights chunk's `codec.digest`, 64 bytes a chunk in chunk order, written to `path`."""
    spans = [(lo, min(DIGESTS_A_TASK, a.weights_blocks - lo)) for lo in range(0, a.weights_blocks, DIGESTS_A_TASK)]
    with open(path, "wb") as f:
        if workers > 1 and len(spans) > 1:
            with pool(workers) as ex:
                for got in ex.map(_weight_digests, *zip(*[(a, lo, n) for lo, n in spans])):
                    f.write(got)
        else:
            for lo, n in spans:
                f.write(_weight_digests(a, lo, n))


def _subtree(digests: str, shape: tuple[int, int, int, int], node: bytes, key: bytes, first: int, log2: int) -> bytes:
    """The root of the payload tree's subtree over leaves first … first + 2^log2 − 1 (padding past the store)."""
    engines, A, W, H = shape
    N = engines * A + H
    dom = audit.payload_domain(node, N)
    zero = codec.digest(bytes(BLOCK))
    level = []
    with open(digests, "rb") as f, (mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if W
                                    else contextlib.nullcontext()) as d:
        for j in range(first, first + (1 << log2)):
            if j >= N:
                level.append(dom.pad(j))
                continue
            r = j % A if j < engines * A else W
            x = d[64 * r:64 * (r + 1)] if r < W else zero
            level.append(dom.leaf(j, j, audit.PAYLOAD_SCHEMA, receipts.hidden_digest(x, store_salt(key, PAYLOAD_LABEL, j))))
    return _up(dom, level, 0, first)


def _up(dom, level: list[bytes], depth: int, first: int) -> bytes:
    while len(level) > 1:
        base = first >> (depth + 1)
        level = [dom.node(depth, base + i, level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
        depth += 1
    return level[0]


def expected_root(store: Store, node: bytes, key: bytes, digests: Path, workers: int = 1,
                  chunk_log2: int = CHUNK_LOG2) -> bytes:
    """The payload root of `store` registered by `node` under payload salt key `key` (`audit.payload_tree`'s), from the
    weights' digests (`weight_digests`'s file), each subtree of 2^chunk_log2 leaves in a worker."""
    N = store.blocks
    height = merkle_depth(N)
    c = min(chunk_log2, height)
    shape = (store.engines, store.arena_blocks, store.arena.weights_blocks, store.host_blocks)
    firsts = list(range(0, 1 << height, 1 << c))
    args = [(str(digests), shape, node, key, f, c) for f in firsts]
    if workers > 1 and len(firsts) > 1:
        with pool(min(workers, len(firsts))) as ex:
            roots = list(ex.map(_subtree, *zip(*args), chunksize=max(1, len(firsts) // (8 * workers))))
    else:
        roots = [_subtree(*x) for x in args]
    return _up(audit.payload_domain(node, N), roots, c, 0)


class Binding:
    """The verifier's binding of a registered store to a pinned model (`verifier.py --model`): at startup the pin read
    (`model_pin` of its file's bytes, `pin_data`, its name `role`), every pinned file held to it (`check`), the arena
    laid out (`arena`) and its weights chunks' digests computed (`weight_digests`, the file `digests`); at each hello
    (`hold`), the store the hello's size makes of `engines` arenas of `arena_blocks` and the rest host filler, and its
    payload root under the verifier's key, against the hello's. The Lean's binding (`RemoteAudit.Store.bind`, `hold`)
    takes `pin_data` and `digests`. `record` is what `verifier.json` keeps of it."""

    def __init__(self, role: str, directory: Path, pin_path: Path, engines: int, arena_blocks: int, *,
                 workers: int = 1, scratch: Path | None = None, log=print) -> None:
        t0 = time.monotonic()
        self.pin_data = pin_bytes(pin_path)
        pin = model_pin(self.pin_data, str(pin_path), role)
        check(directory, pin, workers)
        self.role, self.pin, self.pin_path, self.directory = role, pin, Path(pin_path), Path(directory)
        self.arena = arena(directory, pin["checkpoint"])
        Store(self.arena, engines, arena_blocks)
        self.engines, self.arena_blocks, self.workers, self.log = engines, arena_blocks, workers, log
        self._dir = tempfile.TemporaryDirectory(prefix="store-binding-", dir=scratch)
        self.digests = Path(self._dir.name) / "weights.sha512rows"
        weight_digests(self.arena, self.digests, workers)
        self.store: Store | None = None
        self.state: dict = {"prepare_s": time.monotonic() - t0}
        log(f"store: {role} pinned ({pin['checkpoint_sha512'][:16]}...), {len(self.arena.entries)} weights of "
            f"{self.arena.weights_bytes} bytes in {self.arena.weights_blocks} blocks an arena, digested in "
            f"{self.state['prepare_s']:.1f} s")

    def hold(self, node: bytes, blocks: int, root: bytes, key: bytes, layout) -> tuple[tuple[str, str, int, int], ...]:
        """The pinned model's regions (`verifier.regions_of`'s) when the hello's store is its store, its payload root
        `root` the store's under `key`; else Refused with why."""
        A, E = self.arena_blocks, self.engines
        self.state.update(expected_root=None, bound=False)
        if type(blocks) is not int or blocks < E * A:
            raise Refused(f"the hello's store is {blocks} blocks, fewer than the pinned model's {E} arenas of {A}")
        store = self.store = Store(self.arena, E, A, blocks - E * A)
        regions = tuple((r["kind"], r["tier"], r["first"], r["blocks"]) for r in store.regions())
        self.state.update(store.shape(), payload_key=key.hex(), payload_root=root.hex(), expected_root=None,
                          bound=False)
        if layout is not None and tuple(layout) != regions:
            raise Refused(f"the hello's layout {list(layout)[:4]}... is not the pinned model {self.role}'s "
                          f"{list(regions)[:4]}...")
        t0 = time.monotonic()
        want = expected_root(store, node, key, self.digests, self.workers)
        self.state.update(expected_root=want.hex(), recompute_s=time.monotonic() - t0)
        if want != root:
            raise Refused(f"the hello's payload root {root.hex()[:16]}... is not the pinned model {self.role}'s store "
                          f"({E} arenas of {A} blocks, {self.arena.weights_blocks} of weights each, then "
                          f"{store.host_blocks} host filler blocks) under this verifier's payload key, whose root is "
                          f"{want.hex()[:16]}...: register the store `remote_bandwidth.store` lays out, salted with "
                          "the key the verifier sent")
        self.state["bound"] = True
        return regions

    def record(self) -> dict:
        return {"name": self.role, "checkpoint_sha512": self.pin["checkpoint_sha512"], "pin": str(self.pin_path),
                "checkpoint": str(self.directory), "files": len(self.pin["checkpoint"]), "workers": self.workers,
                **self.state}
