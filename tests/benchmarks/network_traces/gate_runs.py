"""A gate run's files for the leak guard's tests: its witness (the attested lines, the cells' salts, an attestation
indented, a deployment equal to the public one) beside its public files, and the archives and run dirs that hold them."""
from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path

from benchmarks.network_traces import leak_guard as LG

GATE = "c20261027T000000Z-0000"
_rnd = random.Random(7)
SALTS = _rnd.randbytes(16 * 8)
LINES = [{"window": w, "head": _rnd.randbytes(32).hex(), "sig": _rnd.randbytes(96).hex()} for w in range(4)]
CERTIFIED = "".join(json.dumps(d) + "\n" for d in LINES).encode()
DEPLOYMENT = json.dumps({"session": "s0", "windows": 4}).encode()


def _sig_piece() -> tuple[bytes, int]:
    """An aligned piece of certified.jsonl inside the first line's signature, which no public file holds."""
    at = CERTIFIED.index(LINES[0]["sig"].encode())
    o = -(-at // LG.TEXT) * LG.TEXT
    return CERTIFIED[o:o + LG.TEXT], o


#: A witness file holding a public attestation as a verifier writes it, indented: public only through `reprinted`.
INDENTED = json.dumps({"window": LINES[0]["window"], "head": LINES[0]["head"]}, indent=1).encode()


def _gate(root: Path) -> Path:
    """A gate run: its witness (the attested lines, the cells' salts, an attestation indented, a deployment equal to the
    public one) beside its public files (the attestations, each line's window and head, the deployment, end.json and
    request.json)."""
    g = root / GATE
    for name, b in {"witness/certified.jsonl": CERTIFIED, "witness/cells.salts": SALTS, "witness/check.json": INDENTED,
                    "witness/deployment.json": DEPLOYMENT, "out/deployment.json": DEPLOYMENT,
                    "out/certificates.jsonl": "".join(json.dumps({"window": d["window"], "head": d["head"]}) + "\n"
                                                      for d in LINES).encode(),
                    "end.json": b'{"files": {}}', "request.json": b'{"minutes": 1}'}.items():
        (g / name).parent.mkdir(parents=True, exist_ok=True)
        (g / name).write_bytes(b)
    return g


def _tar(path: Path, members: dict[str, bytes], mode: str = "w:gz") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as t:
        for name, b in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    return path


def _put(run: Path, files: dict[str, bytes]) -> None:
    for name, b in files.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_bytes(b)
