"""Pinned vectors for the tile checks' public input regions (`anchors.TileAnchors.regions`), the `--public-inputs` content a
verifier derives for one unit, against which C-Flock's file is mapped: ``uv run python
protocols/pouw/tests/pouw_regions_vectors.py`` rewrites ``tests/vectors/pouw_regions.json``.  A change to that file is a
change to a region's layout or derivation and needs a reason in the PR.

Inputs are derived, not stored: a case's salt, root_A and root_B are SHA-256("verity/pouw/regions-vectors/v1", case, part),
its matmul index 3 and weight id 9, and the scheme's `noise` gives the seeds.  Each region is pinned by the SHA-256 of its
bytes (`anchors.region_bytes`: its u16 words little-endian), and an E region by its bytes too; each `--public-inputs` file
(`TileAnchors.public_inputs`, C-Flock's ``verity/flock-public-inputs/v1``) by its header line and its body's SHA-256."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity_pouw.circuit import pc4, pc8
from verity_pouw.circuit.anchors import TileAnchors, region_bytes
from verity_pouw.protocol import Context, Matmul, Shape
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes.pearl_c_device import DEVICES

OUT = Path(__file__).with_name("vectors") / "pouw_regions.json"
ROWS, COLS = (7, 300), (2, 41)
#: (case, circuit, k, device); each case's units are row_a 1 and row_b 0 of a 2 x k x 2 block on rows ROWS and COLS
CASES = (("pc4-64", "pc4", 64, None), ("pc8-h100-128", "pc8", 128, "h100"), ("pc8-sm120-128", "pc8", 128, "sm120"))
UNITS = (("row_a", 1), ("row_b", 0))
#: each case's `--public-inputs` files: a statement of two instances per row role (`TileAnchors.public_inputs`)
FILES = (("row_a", (0, 1)), ("row_b", (1, 0)))


def part(case: str, name: str) -> bytes:
    return hashlib.sha256(f"verity/pouw/regions-vectors/v1\0{case}\0{name}".encode()).digest()


def anchors(case: str, circuit: str, k: int, dev: str | None) -> TileAnchors:
    ctx = Context(part(case, "salt"), Matmul(3, Shape(2, k, 2), 9), part(case, "root_a"), part(case, "root_b"))
    if circuit == "pc4":
        return TileAnchors.of(F.PearlC4("nvfp4"), ctx, pc4.Shape(2, k, 2, tile=(1, 1)), ROWS, COLS)
    return TileAnchors.of(C.PearlC(forming="v1", device=DEVICES[dev]), ctx, pc8.Shape(2, k, 2, tile=(1, 1), dev=dev),
                          ROWS, COLS)


def case_vector(case: str, circuit: str, k: int, dev: str | None) -> dict:
    t = anchors(case, circuit, k, dev)
    units = []
    for role, index in UNITS:
        regions = []
        for port, words in t.regions(role, index).items():
            anchor = t.shape.ports[role]["regions"][port]
            d = t.shape.derivations[anchor]
            r = {"port": port, "anchor": anchor, "seed": d.seed, "derive": d.rule, "bits": d.bits, "n_words": len(words),
                 "bytes_sha256": hashlib.sha256(region_bytes(words)).hexdigest()}
            if d.per_row:
                r["bytes"] = region_bytes(words).hex()
            regions.append(r)
        units.append({"role": role, "index": index, "regions": regions})
    files = []
    for role, indexes in FILES:
        head, body = t.public_inputs(role, indexes).split(b"\n", 1)
        files.append({"role": role, "indexes": list(indexes), "header": head.decode(),
                      "body_sha256": hashlib.sha256(body).hexdigest()})
    return {"case": case, "circuit": circuit, "k": k, "dev": dev, "salt": part(case, "salt").hex(),
            "root_a": part(case, "root_a").hex(), "root_b": part(case, "root_b").hex(), "matmul": 3, "weight": 9,
            "seed_a": t.seed_a.hex(), "seed_b": t.seed_b.hex(), "rows": list(ROWS), "cols": list(COLS), "units": units,
            "public_inputs": files}


def vectors() -> dict:
    return {"schema": "verity-pouw/regions-vectors/v1", "word_bits": 16, "region_word_bits": 128,
            "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
