"""One safetensors file of a snapshot's linear weights at the given layers, the tensors bench.py and gate.py read, for a node
that has no copy of the snapshot (send the file with the run and pass its directory as --snapshot).

    python subset.py --snapshot DIR --out OUT [--layers 0]

Writes OUT/layers.safetensors and OUT/subset.json (each tensor's shape and the file's SHA-256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_check import LINEARS, Shards  # noqa: E402

NAMES = {v: k for k, v in {"I8": "int8", "F32": "float32", "BF16": "uint16", "I32": "int32"}.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sh = Shards(Path(a.snapshot))
    names = [f"model.layers.{int(l)}.{p}.weight" for l in a.layers.split(",") for parts in LINEARS.values() for p in parts]
    arrays = [sh.get(n) for n in names]
    header, at = {}, 0
    for n, x in zip(names, arrays):
        header[n] = {"dtype": NAMES[x.dtype.name], "shape": list(x.shape), "data_offsets": [at, at + x.nbytes]}
        at += x.nbytes
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path = out / "layers.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head)
        for x in arrays:
            f.write(x.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "subset.json").write_text(json.dumps({"snapshot": a.snapshot, "sha256": digest, "bytes": path.stat().st_size,
                                                 "tensors": {n: h["shape"] for n, h in header.items()}}, indent=1) + "\n")
    print(digest, path.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
