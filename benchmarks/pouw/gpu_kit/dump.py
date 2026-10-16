"""The one dump format, read and written: `bench.cuh`'s `gk::Dump` writes it from the device, a twin writes it here.

A dump is a directory: NAME.bin holds a buffer's bytes, and dump.jsonl has a line a buffer, ``{"name", "file", "bytes",
"layout"}``. Layout ``bytes`` is the buffer as the device holds it; ``sha512-be`` is SHA-512 digests as the hash's
big-endian output bytes, whatever word order the device kept them in (EGEMM dumped its digests' native uint64 words, so no
reference matched: 63f78c35a). A bench's stdout holds its timed items as ``{"gk": "item", ...}`` lines (`items`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VERSION = "gpu-kit-dump/v1"
INDEX = "dump.jsonl"
LAYOUTS = ("bytes", "sha512-be")


class DumpError(ValueError):
    pass


@dataclass(frozen=True)
class Buffer:
    name: str
    layout: str
    data: bytes


def digests_from_words(words: bytes) -> bytes:
    """Digests held as native (little-endian) uint64 words -> SHA-512's bytes: what `gk::Dump::digests` writes."""
    if len(words) % 64:
        raise DumpError(f"{len(words)} bytes is not a whole number of 64-byte digests")
    return b"".join(words[i:i + 8][::-1] for i in range(0, len(words), 8))


def write(directory: Path, name: str, data: bytes, layout: str = "bytes") -> None:
    if layout not in LAYOUTS:
        raise DumpError(f"layout {layout!r}: one of {LAYOUTS}")
    if layout == "sha512-be" and len(data) % 64:
        raise DumpError(f"{name}: {len(data)} bytes is not a whole number of 64-byte digests")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.bin").write_bytes(data)
    with (directory / INDEX).open("a") as f:
        f.write(json.dumps({"name": name, "file": f"{name}.bin", "bytes": len(data), "layout": layout}) + "\n")


def read(directory: Path) -> dict[str, Buffer]:
    """Every buffer of a dump, each checked against its index line (size, layout, one line a name)."""
    directory = Path(directory)
    index = directory / INDEX
    if not index.exists():
        raise DumpError(f"{directory}: no {INDEX} (write the dump with bench.cuh's gk::Dump or dump.write)")
    out: dict[str, Buffer] = {}
    for n, line in enumerate(index.read_text().splitlines(), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        if e["name"] in out:
            raise DumpError(f"{index}:{n}: {e['name']!r} dumped twice; give each buffer its own name")
        if e["layout"] not in LAYOUTS:
            raise DumpError(f"{index}:{n}: layout {e['layout']!r}: one of {LAYOUTS}")
        data = (directory / e["file"]).read_bytes()
        if len(data) != e["bytes"]:
            raise DumpError(f"{index}:{n}: {e['file']} holds {len(data)} bytes, its line says {e['bytes']} (cut short)")
        out[e["name"]] = Buffer(e["name"], e["layout"], data)
    return out


def items(stdout: str) -> list[dict]:
    """The timed items a bench printed (`gk::items_jsonl`)."""
    out = []
    for line in stdout.splitlines():
        if line.startswith('{"gk": "item"'):
            out.append(json.loads(line))
    return out
