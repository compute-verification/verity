"""The auditor's model registry: each served model's weights, committed by the auditor with
its own salts, so a period's weights' root is never the prover's choice.

- **A registration** (`register`, vy-admin's `vy-challenger-gate model-register`, run as `auditor`) takes a model's int8
  checkpoint role (`--model`, a role listed exactly once in the pinned tree's `checkpoints-int8.json` files, `MANIFESTS`:
  vLLM's), the period's registered
  circuit (`--circuit`, the SHA-512 of the window Program's descriptor, `--descriptor FILE`, which it must hash to) and the
  checkpoint's directory (`--checkpoint DIR`). It refuses unless:
  - DIR and every file the manifest lists are regular, not symlinks, and writable by no one but root and `auditor`
    (`check_owned`), so nothing changes them between the hash and the read;
  - each of those files has its pinned size and sha256 (`check_files`), and the SHA-512 of their `{file: SHA-512}` map
    (`checkpoint_sha512`) is the role's pin in core's `PINS` (`verity/core/protocols/pouw/lean/check/models.json`,
    ``{role: {"checkpoint_sha512": hex}}``), which credit holds each open's ``model`` to;
  - the descriptor's linears are exactly the model's decoder linears, ``qkv_proj``, ``o_proj``, ``gate_up_proj`` and
    ``down_proj`` of each of config.json's ``num_hidden_layers`` in turn (`DECODER`, `check_linears`), so the root it
    signs is the function the model computes; or, for a role whose manifest entry declares ``linears`` (an MLP's,
    ``model.layers.0.mlp.up_proj`` then ``model.layers.1.mlp.down_proj``), exactly those in that order
    (`declared_linears`). Only the pinned tree declares them, never the prover.
  It then draws a fresh 32-byte salt key; commits each K's ``<prefix>b/<K>`` from the int8 codes and s_w of the pinned
  shards alone, a tensor in two shards refused (`Checkpoint`), vLLM's fused ``qkv_proj`` and ``gate_up_proj`` their
  parts' rows in turn (`FUSED`, `weights.weights_roots`); keeps the key (`key_path`, auditor's, 0600); and appends one
  line to the registry (`Host.models`, append-only, root's seal): ``{model, circuit, prefix, checkpoint: {file:
  SHA-512}, checkpoint_sha512, source, roots: {K: root}, entries, weights_root, key_sha512, pinned, registered_utc}``.
- **A period-open** with ``--model M`` (or ``--weights-root W``) signs the registry entry's ``weights_root`` and
  ``model: {name, checkpoint_sha512}`` for its circuit (`opened`), and refuses a root the registry has no entry of for
  that circuit: no period's weights' root is the prover's. A model the registry has no entry of under the circuit (a
  remote memory audit's, whose circuit is its verifier tree and which commits no b/<K>) is signed as ``model`` alone,
  its digest core's pin (`PINS`), and refused when the pin lacks it.
- **The openings** are the salt key: a lane's gate key reads it with `model-key --model M --circuit C [--weights-root W]`,
  W the period's signed root, required when the model is registered more than once under C. The weights are public; the
  key salts only their b/<K> rows (`weights.B_NONCE`), never a window's A or z/y.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import weights as Wt
from verity.ml.circuits.codec import canonical_json

MANIFEST = Path("integrations/vllm/manifests/checkpoints-int8.json")
#: the manifests a role may be in, each read from the pinned tree only: vLLM's int8 recipe pins; a role is listed exactly
#: once across them
MANIFESTS = (MANIFEST,)
PINS = Path("verity/core/protocols/pouw/lean/check/models.json")
#: vLLM's merged modules at TP = 1: a fused linear's rows are its parts' rows in turn, its scales the same
FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
#: a decoder layer's int8 linears in vLLM's module order, the window's linear order
DECODER = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
CHUNK = 1 << 24


class Refused(Exception):
    """A registration the auditor refuses, with why."""


def manifest_entry(tree: Path, role: str) -> dict:
    """The entry of int8 checkpoint role `role` in the pinned tree's `MANIFESTS`, where it must appear exactly once across
    them all; else Refused. Both files are read from the pinned tree alone, never from the prover, a request or the
    checkpoint directory. `MANIFEST` is required; a missing later file contributes no role (it can only narrow what
    registers), while one present but unreadable is Refused."""
    entries = []
    for i, path in enumerate(MANIFESTS):
        try:
            doc = json.loads((tree / path).read_text())
        except FileNotFoundError:
            if i == 0:
                raise Refused(f"the pinned tree has no {path}: pin with it in the archive (pin.sh's usage)") from None
            continue
        except (OSError, ValueError):
            raise Refused(f"the pinned tree's {path} is not readable JSON") from None
        listed = doc.get("checkpoints", []) if isinstance(doc, dict) else None
        if not isinstance(listed, list) or not all(isinstance(e, dict) for e in listed):
            raise Refused(f"the pinned tree's {path} has no list of checkpoint entries")
        entries += [(path, e) for e in listed]
    found = [e for _path, e in entries if e.get("role") == role]
    if len(found) != 1:
        where = sorted(str(p) for p, e in entries if e.get("role") == role)
        raise Refused(f"--model {role!r}: {len(found)} entries ({', '.join(where) or 'none'}), not one role of "
                      f"{' and '.join(map(str, MANIFESTS))} together ({sorted(str(e.get('role')) for _p, e in entries)})")
    return found[0]


def pinned_checkpoint(tree: Path, role: str) -> str:
    """Core's pin of `role`'s `checkpoint_sha512` (`PINS` in the pinned tree), or Refused."""
    try:
        pin = json.loads((tree / PINS).read_text())[role]["checkpoint_sha512"]
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused(f"{PINS} in the pinned tree pins no checkpoint_sha512 of {role}: credit refuses a model it doesn't "
                      "pin, so register only a pinned one") from None
    return pin


def check_owned(directory: Path, names: list[str]) -> None:
    """Refused unless `directory` and each of `names` in it are a directory and regular files, no symlink, owned by root or
    this user and writable by no one else: a file the auditor hashes is the one it then reads."""
    me = os.geteuid()
    for p in [directory, *(directory / n for n in names)]:
        try:
            st = os.lstat(p)
        except OSError:
            raise Refused(f"{p}: missing") from None
        kind = stat.S_ISDIR if p == directory else stat.S_ISREG
        if not kind(st.st_mode) or st.st_uid not in (0, me) or st.st_mode & 0o022:
            raise Refused(f"{p}: not a {'directory' if p == directory else 'regular file'} that only root or auditor can "
                          "write (no symlink): copy the checkpoint into a directory auditor owns, mode 0755/0644")


def _digests(path: Path) -> tuple[str, str]:
    s256, s512 = hashlib.sha256(), hashlib.sha512()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            s256.update(b)
            s512.update(b)
    return s256.hexdigest(), s512.hexdigest()


def check_files(directory: Path, entry: dict) -> dict[str, str]:
    """Each file `entry` pins, with its SHA-512, once every one is in `directory` with its pinned size and sha256; else
    Refused."""
    out = {}
    for name, want in sorted(entry["files"].items()):
        p = directory / name
        if not p.is_file() or p.stat().st_size != want["size"]:
            raise Refused(f"{p}: not the pinned file ({want['size']} bytes) of {entry['repo']}@{entry['revision']}")
        s256, s512 = _digests(p)
        if s256 != want["sha256"]:
            raise Refused(f"{p}: sha256 {s256[:16]}..., not the pinned {want['sha256'][:16]}... of {entry['repo']}")
        out[name] = s512
    return out


def checkpoint_sha512(files: dict[str, str]) -> str:
    """A checkpoint's digest: SHA-512 of the canonical JSON of its ``{file: SHA-512 hex}``."""
    return hashlib.sha512(canonical_json(files)).hexdigest()


class Checkpoint:
    """The tensors of a checkpoint's pinned safetensors shards (`files`), read in place (`np.memmap`); Refused when a
    tensor is in two of them."""

    DTYPES = {"I8": np.int8, "F32": np.dtype("<f4")}

    def __init__(self, directory: Path, files: list[str]):
        self.dir = directory
        self.where: dict[str, tuple[Path, dict, int]] = {}
        for name in sorted(f for f in files if f.endswith(".safetensors")):
            f = directory / name
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for t, meta in header.items():
                if t == "__metadata__":
                    continue
                if t in self.where:
                    raise Refused(f"tensor {t} is in both {self.where[t][0].name} and {name}")
                self.where[t] = (f, meta, 8 + n)

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.where:
            raise Refused(f"the checkpoint has no tensor {name}")
        f, t, base = self.where[name]
        dtype = self.DTYPES.get(t["dtype"])
        if dtype is None:
            raise Refused(f"tensor {name} is {t['dtype']}, not one of {sorted(self.DTYPES)}")
        lo, hi = t["data_offsets"]
        shape = tuple(t["shape"])
        if hi - lo != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise Refused(f"tensor {name}'s bytes are not its shape {shape}")
        return np.memmap(f, dtype=dtype, mode="r", offset=base + lo, shape=shape)


def parts(name: str) -> list[str]:
    """The checkpoint modules a window linear `name` (vLLM's module name) reads, in row order: a fused one's parts
    (`FUSED`), any other module itself alone."""
    head, _, last = name.rpartition(".")
    return [f"{head}.{p}" for p in FUSED[last]] if last in FUSED else [name]


def rows_of(ckpt: Checkpoint, name: str, k: int) -> Iterator[bytes]:
    """Linear `name`'s B rows (`weights.row_bytes`): its parts' int8 rows of K codes, each with its f32 s_w."""
    for p in parts(name):
        w, s = ckpt.tensor(f"{p}.weight"), ckpt.tensor(f"{p}.weight_scale")
        if w.ndim != 2 or w.shape[1] != k or s.shape not in ((w.shape[0], 1), (w.shape[0],)):
            raise Refused(f"{p}: weight {w.shape} and scale {s.shape} are not n × {k} int8 codes and n f32 scales")
        s = s.reshape(-1)
        for j in range(w.shape[0]):
            yield Wt.row_bytes(w[j].tobytes(), s[j].tobytes())


def layout(descriptor: bytes, circuit: str) -> tuple[str, list[tuple[str, int, int]]]:
    """(prefix, the window's linears ``(name, K, n)`` in its order) of the descriptor whose SHA-512 is `circuit`."""
    if hashlib.sha512(descriptor).hexdigest() != circuit:
        raise Refused(f"--descriptor's SHA-512 is not --circuit {circuit[:16]}...: the descriptor of the period's circuit")
    lay, why = CR.layout_of(descriptor)
    if why is not None:
        raise Refused(f"--descriptor: {why[1]}")
    return CR.shape_of(descriptor)["prefix"], [(str(a), int(k), int(n)) for a, k, n in lay["linears"]]


def declared_linears(entry: dict) -> list[str] | None:
    """The role's own ordered linears, its pinned `MANIFESTS` entry's ``linears`` (module names, each read by `parts`), or
    None when the entry has none and the role's linears are `DECODER`'s; Refused when the field is not a nonempty list of
    distinct names. Only the pinned tree's entry declares them, never the prover, the request or the checkpoint
    directory: the gate is auditor-trusted, and what it signs is the pinned role's function, not one the prover picks."""
    if "linears" not in entry:
        return None
    names = entry["linears"]
    if (not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names)):
        raise Refused(f"{entry.get('role')}'s manifest entry declares linears that are not a nonempty list of distinct "
                      "module names: fix the role's entry in the pinned tree")
    return names


def check_linears(linears: list[tuple[str, int, int]], config: dict | None, declared: list[str] | None = None) -> None:
    """Refused unless the window's linears are the role's `declared` ones (`declared_linears`) in that order, or, when it
    declares none, `DECODER` of each of `config`'s ``num_hidden_layers`` in turn; no other."""
    got = [name for name, _k, _n in linears]
    if declared is not None:
        want, what = declared, "the role's declared linears"
    else:
        layers = (config or {}).get("num_hidden_layers")
        if type(layers) is not int or layers < 1:
            raise Refused("the checkpoint's config.json names no num_hidden_layers")
        want = [f"model.layers.{i}.{m}" for i in range(layers) for m in DECODER]
        what = f"the model's {layers} decoder layers' {', '.join(DECODER)}"
    if got != want:
        at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        raise Refused(f"the descriptor's linears are not {what} in turn: linear {at} is "
                      f"{got[at] if at < len(got) else 'missing'}, not {want[at] if at < len(want) else 'past the last'}")


def register(tree: Path, role: str, circuit: str, descriptor: bytes, checkpoint: Path, key: bytes, *, pinned: str,
             now: str, log: Callable[[str], None] = lambda s: None) -> dict:
    """The registry line of `role`'s weights under `circuit`, committed with salt key `key`; or Refused."""
    entry = manifest_entry(tree, role)
    pin = pinned_checkpoint(tree, role)
    prefix, linears = layout(descriptor, circuit)
    names = sorted(entry["files"])
    check_owned(checkpoint, names)
    log(f"checking {len(names)} files of {entry['repo']}@{entry['revision'][:10]} in {checkpoint}")
    files = check_files(checkpoint, entry)
    digest = checkpoint_sha512(files)
    if digest != pin:
        raise Refused(f"the checkpoint's SHA-512 {digest[:16]}... is not {PINS}'s pin {pin[:16]}... of {role}")
    declared = declared_linears(entry)
    config = None
    if declared is None:
        try:
            config = json.loads((checkpoint / "config.json").read_text())
        except (OSError, ValueError):
            raise Refused("the checkpoint has no readable config.json") from None
    check_linears(linears, config, declared)
    ckpt = Checkpoint(checkpoint, names)
    shapes = [(n, k) for _name, k, n in linears]
    try:
        roots = Wt.weights_roots(circuit, prefix, shapes, lambda w: rows_of(ckpt, linears[w][0], linears[w][1]), key,
                                 progress=lambda k, j: log(f"K = {k}: row {j}"))
    except ValueError as e:
        raise Refused(f"the checkpoint's rows are not the window's linears: {e}") from None
    leaves = {k: sum(n for n, kw in shapes if kw == k) for k in roots}
    return {"model": role, "circuit": circuit, "prefix": prefix, "checkpoint": files, "checkpoint_sha512": digest,
            "source": {"repo": entry["repo"], "revision": entry["revision"], "recipe": entry.get("recipe")},
            "roots": {str(k): roots[k].hex() for k in sorted(roots)},
            "entries": [Wt.entry(circuit, prefix, k, leaves[k], roots[k]) for k in sorted(roots)],
            "weights_root": Wt.weights_root(roots).hex(), "key_sha512": hashlib.sha512(key).hexdigest(),
            "pinned": pinned, "registered_utc": now}


def read(path: Path) -> list[dict]:
    try:
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    except FileNotFoundError:
        return []


def resolve(lines: list[dict], circuit: str, model: str | None, weights_root: str | None, *,
            unique: bool = False) -> dict | None:
    """The registry entry a period-open's ``--model`` and ``--weights-root`` name for its circuit (with the model alone,
    its newest; with `unique`, Refused when the model has more than one), None when it names neither; Refused when the
    registry has none, or the two disagree."""
    if model is None and weights_root is None:
        return None
    mine = [e for e in lines if e["circuit"] == circuit and (model is None or e["model"] == model)
            and (weights_root is None or e["weights_root"] == weights_root)]
    if not mine:
        raise Refused(f"the model registry has no entry of {model or 'any model'} under circuit {circuit[:16]}..."
                      + (f" with weights' root {weights_root[:16]}..." if weights_root else "")
                      + ": a period's weights' root is the auditor's registration (vy-admin's model-register), never "
                        "the prover's")
    if unique and len({e["weights_root"] for e in mine}) > 1:
        raise Refused(f"{model} is registered {len(mine)} times under circuit {circuit[:16]}...: name the period's signed "
                      "root with --weights-root")
    return mine[-1]


def opened(lines: list[dict], tree: Path, circuit: str, model: str | None, weights_root: str | None) -> dict:
    """What a period-open signs for its model: a registry entry's ``weights_root`` and ``model: {name,
    checkpoint_sha512}`` when the registry has one for the circuit (`resolve`); for a model the registry has none of under
    the circuit (a remote memory audit's, whose circuit is its verifier tree), ``model`` alone, its digest core's pin
    (`pinned_checkpoint`), with no ``weights_root``; nothing when the open names neither. Refused for a root the registry
    lacks, or a model neither it nor core's pin has."""
    if model is None and weights_root is None:
        return {}
    try:
        e = resolve(lines, circuit, model, weights_root)
    except Refused:
        if weights_root is not None or model is None:
            raise
        return {"model": {"name": model, "checkpoint_sha512": pinned_checkpoint(tree, model)}}
    return {"weights_root": e["weights_root"], "model": {"name": e["model"], "checkpoint_sha512": e["checkpoint_sha512"]}}


def key_path(root: Path, e: dict) -> Path:
    return root / "models" / f"{e['model']}-{e['circuit'][:32]}-{e['key_sha512'][:16]}.key"
