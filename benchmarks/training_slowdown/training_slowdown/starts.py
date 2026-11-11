"""The starts registry, `starts.toml`: each start by its store artifact, its files' sha256, its shape, the MACs of its
own training (C0) and its loss on the search pool, and the data it was trained on (`training_slowdown.data`).

`start(name)` is a registered start, refusing a name the registry lacks. `load(start, root)` reads its θ from an
installed copy, `root/NAME/{theta.safetensors,start.json}` (the registry's `installed` on a node; `research data
fetch ART --path PATH/...` gets the same directory from the store), and refuses one whose files aren't the registry's
or whose start.json records another digest, C0 or loss. θ's file is read here (`tensors`), without the safetensors
package.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REGISTRY = Path(__file__).with_name("starts.toml")


def registry(path: Path = REGISTRY) -> dict:
    return tomllib.loads(Path(path).read_text())


@dataclass(frozen=True)
class Start:
    name: str
    data: str
    art: str
    path: str
    theta_file_sha256: str
    start_json_sha256: str
    digest: str
    shape: dict
    c0_macs: float
    l0_bpb: float

    def entry(self) -> dict:
        """The start as a run's record states it."""
        return asdict(self)


def start(name: str, reg: dict | None = None) -> Start:
    reg = registry() if reg is None else reg
    if name not in reg["starts"]:
        raise SystemExit(f"starts: no start {name!r} in {REGISTRY.name} (have {', '.join(reg['starts'])})")
    return Start(name=name, **reg["starts"][name])


#: safetensors' dtype names
DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}


def read_file(path: Path) -> bytearray:
    buf = bytearray(Path(path).stat().st_size)
    with open(path, "rb") as f, memoryview(buf) as view:
        got = 0
        while got < len(buf) and (n := f.readinto(view[got:])):
            got += n
    return buf


def tensors(buf: bytearray) -> dict[str, torch.Tensor]:
    """The tensors of a safetensors file's bytes (an 8-byte little-endian header length, a JSON header naming each
    tensor's dtype, shape and byte range, then the bytes), as views of `buf`."""
    n = int.from_bytes(buf[:8], "little")
    out = {}
    for name, t in json.loads(bytes(buf[8: 8 + n])).items():
        if name == "__metadata__":
            continue
        lo, hi = 8 + n + t["data_offsets"][0], 8 + n + t["data_offsets"][1]
        dtype = DTYPES[t["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        if hi == lo:
            flat = torch.empty(0, dtype=dtype)
        elif lo % size:
            flat = torch.frombuffer(bytearray(buf[lo:hi]), dtype=dtype)
        else:
            flat = torch.frombuffer(buf, dtype=dtype, count=(hi - lo) // size, offset=lo)
        out[name] = flat.reshape(t["shape"])
    return out


def load(s: Start, root: Path | None = None) -> dict[str, torch.Tensor]:
    """θ of `s`, from its installed directory under `root` (default the registry's `installed`), checked."""
    d = Path(root if root is not None else registry()["installed"]) / s.name
    raw = {}
    for f, want in (("start.json", s.start_json_sha256), ("theta.safetensors", s.theta_file_sha256)):
        if not (d / f).is_file():
            raise SystemExit(f"starts: {d / f} is missing (`research data fetch {s.art} --path {s.path}/{f}`)")
        raw[f] = read_file(d / f)
        if (got := hashlib.sha256(raw[f]).hexdigest()) != want:
            raise SystemExit(f"starts: {d / f} has sha256 {got}, the registry's {s.name} {want}")
    rec = json.loads(bytes(raw["start.json"]))
    p = rec.get("provenance", {})
    found = (rec.get("digest"), float(p.get("c0", {}).get("macs", "nan")), p.get("search_pool", {}).get("bpb"))
    if found != (s.digest, s.c0_macs, s.l0_bpb):
        raise SystemExit(f"starts: {d / 'start.json'} records digest, C0 and loss {found}, the registry "
                         f"{(s.digest, s.c0_macs, s.l0_bpb)}")
    return tensors(raw["theta.safetensors"])
