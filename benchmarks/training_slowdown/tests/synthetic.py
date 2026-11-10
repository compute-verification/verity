"""A tiny data and start, built in a test's directory in the layout a node holds them in: a byte-level tokenizer of 259
tokens, a stream of three parts (six windows at the start's context), a search pool of six full windows and a final
set of two, an lf start of one block, and a registry naming both (`tiny_registry`)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training_slowdown.data import Data, bytes_to_unicode
from training_slowdown.model import LM
from training_slowdown.starts import Start

EOT = 0
MERGES = ("ab", "abc")
VOCAB = 1 + 256 + len(MERGES)
SHAPE = {"family": "lf", "d_model": 16, "n_layer": 1, "n_head": 2, "d_ff": 32, "vocab": VOCAB, "ctx": 1024, "tied": True,
         "parameters": 16 * 16 * 4 + 3 * 16 * 32 + 2 * 16 + (VOCAB + 1) * 16, "tokens": 0}
#: each token's bytes: end-of-text none, a byte one, a merge its letters
TOKEN_BYTES = np.array([0] + [1] * 256 + [len(m) for m in MERGES], dtype=np.int64)
PARTS = (2100, 1100, 3100)
SETS = {"search": 6 * 1024 + 300, "final": 2 * 1024 + 50}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tokens(rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` token ids, a document starting at every 97th."""
    t = rng.integers(1, VOCAB, n).astype(np.uint16)
    t[::97] = EOT
    return t


def tiny_data(root: Path) -> Data:
    rng = np.random.default_rng(0)
    train = root / "train"
    (train / "parts").mkdir(parents=True)
    enc = bytes_to_unicode()
    vocab = {"<|endoftext|>": EOT} | {enc[b]: 1 + b for b in range(256)}
    vocab |= {"".join(enc[c] for c in m.encode()): 257 + i for i, m in enumerate(MERGES)}
    tok = json.dumps({"model": {"vocab": vocab}}).encode()
    (train / "tokenizer.json").write_bytes(tok)
    selection = b'{"sources": {}}\n'
    man = {"tokenizer": "tiny", "vocab": VOCAB, "selection": {"file": "selection.json", "sha256": sha256(selection)},
           "parts": []}
    for i, n in enumerate(PARTS):
        (train / "parts" / f"{i:03d}.bin").write_bytes(tokens(rng, n).tobytes())
        man["parts"].append({"index": i, "tokens": n})
    raw = json.dumps(man).encode()
    (train / "manifest.json").write_bytes(raw)
    sets, pins = {}, {}
    for split, n in SETS.items():
        d = root / split
        d.mkdir()
        t = tokens(rng, n)
        (d / "tiny.bin").write_bytes(t.tobytes())
        pins[split] = sha256(t.tobytes())
        (d / "manifest.json").write_text(json.dumps({"tokenizers": {"tiny": {"sha256": pins[split], "tokens": n}},
                                                     "bytes": int(TOKEN_BYTES[t].sum()), "documents": int((t == EOT).sum())}))
        sets[split] = d
    return Data("tiny", "tiny", VOCAB, EOT, train, sets, sha256(raw), sha256(tok), sha256(selection), pins)


def write_safetensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    names = {torch.float32: "F32", torch.bfloat16: "BF16"}
    header, blobs, at = {}, [], 0
    for k in sorted(state):
        t = state[k].detach().contiguous().cpu()
        b = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def tiny_start(root: Path, name: str = "tiny") -> tuple[Start, dict[str, torch.Tensor]]:
    """An installed start under `root` and its registry entry."""
    torch.manual_seed(0)
    theta = {k: v.detach().clone() for k, v in LM.of(SHAPE).state_dict().items()}
    d = root / name
    d.mkdir(parents=True)
    write_safetensors(d / "theta.safetensors", theta)
    digest = "d" * 64
    rec = {"name": name, "digest": digest, "provenance": {"c0": {"macs": 12_345}, "search_pool": {"bpb": 7.5}}}
    (d / "start.json").write_text(json.dumps(rec))
    s = Start(name, "tiny", "art:" + "0" * 64, f"starts/{name}", sha256((d / "theta.safetensors").read_bytes()),
              sha256((d / "start.json").read_bytes()), digest, SHAPE, 12_345.0, 7.5)
    return s, theta


def _toml(v) -> str:
    if isinstance(v, dict):
        return "{ " + ", ".join(f'"{k}" = {_toml(x)}' for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return json.dumps(str(v) if isinstance(v, Path) else v)


def tiny_registry(root: Path) -> Path:
    """`root/starts.toml`, naming the tiny data and start built under `root`."""
    data = tiny_data(root / "data")
    s, _ = tiny_start(root / "starts")
    d = {"tokenizer": data.tokenizer, "vocab": data.vocab, "eot": data.eot, "train": data.train,
         "manifest_sha256": data.manifest_sha256, "tokenizer_sha256": data.tokenizer_sha256,
         "selection_sha256": data.selection_sha256}
    d |= {split: data.sets[split] for split in data.sets} | {f"{k}_sha256": v for k, v in data.set_sha256.items()}
    entry = {k: v for k, v in s.entry().items() if k != "name"}
    lines = [f"installed = {_toml(root / 'starts')}", "", "[data.tiny]", *(f"{k} = {_toml(v)}" for k, v in d.items()),
             "", f"[starts.{s.name}]", *(f"{k} = {_toml(v)}" for k, v in entry.items())]
    (root / "starts.toml").write_text("\n".join(lines) + "\n")
    return root / "starts.toml"


def write_config(path: Path, **over) -> Path:
    """An attack config of the tiny start: `over` replaces a top-level key (None drops it)."""
    cfg = {"start": "tiny", "route": "subset", "target": {"gain": 1.1}, "budget": {"steps": 3, "batch": 2},
           "seeds": {"search": [0], "final": []}, "policy": "console", "dials": {"trainable": "blocks.0.w_*", "lr": 1e-2}}
    cfg |= over
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k} = {_toml(v)}\n" for k, v in cfg.items() if v is not None and k != "dials")
                    + ("[dials]\n" + "".join(f"{k} = {_toml(v)}\n" for k, v in cfg["dials"].items())
                       if cfg.get("dials") is not None else ""))
    return path
