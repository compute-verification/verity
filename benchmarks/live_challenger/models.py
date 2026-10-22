"""The auditor's model registry (the #3201 recheck's decision 1): each served model's weights, committed by the auditor with
its own salts, so a period's weights' root is never the prover's choice.

- **A registration** (`register`, vy-admin's `vy-challenger-gate model-register`, run as `auditor`) takes a model's int8
  checkpoint role (`--model`, a role of `checkpoints-int8.json` in the pinned tree, `MANIFEST`), the period's declared
  circuit (`--circuit`, the SHA-512 of the window Program's descriptor, `--descriptor FILE`, which it must hash to) and the
  checkpoint's snapshot directory (`--checkpoint DIR`). It refuses unless every file the manifest lists has its pinned
  sha256 (`check_files`); reads the window's linears and prefix from the descriptor (`credit.layout_of`, `shape_of`);
  draws a fresh 32-byte salt key; commits each K's ``<prefix>b/<K>`` from the checkpoint's int8 codes and s_w
  (`weights.weights_roots`, vLLM's fused ``qkv_proj`` and ``gate_up_proj`` their parts' rows in turn, `FUSED`); keeps
  the key (`key_path`, auditor's, 0600); and appends one line to the registry (`Host.models`, append-only, root's seal):
  ``{model, circuit, prefix, checkpoint: {file: SHA-512}, checkpoint_sha512, roots: {K: root}, entries, weights_root,
  key_sha512, pinned, registered_utc}``.
- **A period-open** with ``--model M`` (or ``--weights-root W``) signs the registry entry's ``weights_root`` and
  ``model: {name, checkpoint_sha512}`` for its circuit (`resolve`), and refuses a model or a root the registry has no
  entry of for that circuit: no period's weights' root is the prover's.
- **The openings** are the salt key: a lane's gate key reads it with `model-key --model M --circuit C`. The weights are
  public; the key salts only their b/<K> rows (`weights.B_NONCE`).
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import weights as Wt
from verity.ml.circuits.codec import canonical_json

MANIFEST = Path("integrations/vllm/manifests/checkpoints-int8.json")
#: vLLM's merged modules at TP = 1: a fused linear's rows are its parts' rows in turn, its scales the same
FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
CHUNK = 1 << 24


class Refused(Exception):
    """A registration the auditor refuses, with why."""


def manifest_entry(tree: Path, role: str) -> dict:
    """The pinned tree's `MANIFEST` entry of int8 checkpoint role `role`, or Refused."""
    try:
        doc = json.loads((tree / MANIFEST).read_text())
    except (OSError, ValueError):
        raise Refused(f"the pinned tree has no {MANIFEST}: pin with it in the archive (pin.sh's usage)") from None
    found = [e for e in doc.get("checkpoints", []) if e.get("role") == role]
    if len(found) != 1:
        raise Refused(f"--model {role!r}: one role of {MANIFEST} ({sorted(e.get('role') for e in doc.get('checkpoints', []))})")
    return found[0]


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


class Checkpoint:
    """A safetensors snapshot's tensors, read in place (`np.memmap`)."""

    DTYPES = {"I8": np.int8, "F32": np.dtype("<f4")}

    def __init__(self, directory: Path):
        self.dir = directory
        files = sorted(directory.glob("*.safetensors"))
        self.where: dict[str, tuple[Path, dict, int]] = {}
        for f in files:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for name, t in header.items():
                if name != "__metadata__":
                    self.where[name] = (f, t, 8 + n)

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
    """The checkpoint modules a window linear `name` (vLLM's module name) reads, in row order."""
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


def register(tree: Path, role: str, circuit: str, descriptor: bytes, checkpoint: Path, key: bytes, *, pinned: str,
             now: str, log: Callable[[str], None] = lambda s: None) -> dict:
    """The registry line of `role`'s weights under `circuit`, committed with salt key `key`; or Refused."""
    entry = manifest_entry(tree, role)
    prefix, linears = layout(descriptor, circuit)
    log(f"checking {len(entry['files'])} files of {entry['repo']}@{entry['revision'][:10]} in {checkpoint}")
    files = check_files(checkpoint, entry)
    ckpt = Checkpoint(checkpoint)
    shapes = [(n, k) for _name, k, n in linears]
    try:
        roots = Wt.weights_roots(circuit, prefix, shapes, lambda w: rows_of(ckpt, linears[w][0], linears[w][1]), key,
                                 progress=lambda k, j: log(f"K = {k}: row {j}"))
    except ValueError as e:
        raise Refused(f"the checkpoint's rows are not the window's linears: {e}") from None
    leaves = {k: sum(n for n, kw in shapes if kw == k) for k in roots}
    return {"model": role, "circuit": circuit, "prefix": prefix, "checkpoint": files,
            "checkpoint_sha512": hashlib.sha512(canonical_json(files)).hexdigest(),
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


def resolve(lines: list[dict], circuit: str, model: str | None, weights_root: str | None) -> dict | None:
    """The registry entry a period-open's ``--model`` and ``--weights-root`` name for its circuit (the newest of that
    model), None when it names neither; Refused when the registry has none, or the two disagree."""
    if model is None and weights_root is None:
        return None
    mine = [e for e in lines if e["circuit"] == circuit and (model is None or e["model"] == model)
            and (weights_root is None or e["weights_root"] == weights_root)]
    if not mine:
        raise Refused(f"the model registry has no entry of {model or 'any model'} under circuit {circuit[:16]}..."
                      + (f" with weights' root {weights_root[:16]}..." if weights_root else "")
                      + ": a period's weights' root is the auditor's registration (vy-admin's model-register), never "
                        "the prover's")
    return mine[-1]


def key_path(root: Path, e: dict) -> Path:
    return root / "models" / f"{e['model']}-{e['circuit'][:32]}-{e['key_sha512'][:16]}.key"
