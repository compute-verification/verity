"""Pinned vectors for the tile checks' public input regions (`anchors.TileAnchors.regions`), the `--public-inputs` content a
verifier derives for one unit, against which C-Flock's file is mapped: ``uv run python
verity/core/protocols/pouw/tests/pouw_regions_vectors.py`` rewrites ``tests/vectors/pouw_regions.json``.  A change to that file is a
change to a region's layout or derivation and needs a reason in the PR.

Inputs are derived, not stored: a case's salt, root_A and root_B are SHA-256("verity/pouw/regions-vectors/v1", case, part),
its matmul index 3 and weight id 9, and the scheme's `noise` gives the seeds.  Each region is pinned by the SHA-256 of its
bytes (`anchors.region_bytes`: its u16 words little-endian), and an E region by its bytes too; each `--public-inputs` file
(`TileAnchors.public_inputs`, C-Flock's ``verity/flock-public-inputs/v1``) by its header line and its body's SHA-256.
``row_k_cases`` pins the cut's shapes (`rowk`, ``Shape.row_k``): each piece's regions, a form's its segment of F_A and F_B
under the slice rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.protocols.pouw.circuit import pc4, pc8, rowk
from verity.ml.protocols.pouw.circuit.anchors import TileAnchors, region_bytes
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c4 as F
from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES

OUT = Path(__file__).with_name("vectors") / "pouw_regions.json"
ROWS, COLS = (7, 300), (2, 41)
#: (case, circuit, k, device); each case's units are row_a 1 and row_b 0 of a 2 x k x 2 block on rows ROWS and COLS
CASES = (("pc4-64", "pc4", 64, None), ("pc8-h100-128", "pc8", 128, "h100"), ("pc8-sm120-128", "pc8", 128, "sm120"))
UNITS = (("row_a", 1), ("row_b", 0))
#: each case's `--public-inputs` files: a statement of two instances per row role (`TileAnchors.public_inputs`)
FILES = (("row_a", (0, 1)), ("row_b", (1, 0)))
#: (case, circuit, k, device, row_k): the cut's shapes (`rowk`); each case's units are every piece of row 1 of A and row 0
#: of B that reads a region, a first form at segment 0 and a form at segment 1, and its files two forms' statements
ROWK_CASES = (("pc4-rowk-128", "pc4", 128, None, 64), ("pc8-h100-rowk-256", "pc8", 256, "h100", 128),
              ("pc8-sm120-rowk-256", "pc8", 256, "sm120", 128))
ROWK_FILES = (("row_a_form", (0, 1), 1), ("row_b_first", (1, 0), 0))


def part(case: str, name: str) -> bytes:
    return hashlib.sha256(f"verity/pouw/regions-vectors/v1\0{case}\0{name}".encode()).digest()


def anchors(case: str, circuit: str, k: int, dev: str | None, row_k: int | None = None) -> TileAnchors:
    ctx = Context(part(case, "salt"), Matmul(3, Shape(2, k, 2), 9), part(case, "root_a"), part(case, "root_b"))
    if circuit == "pc4":
        return TileAnchors.of(F.PearlC4("nvfp4"), ctx, pc4.Shape(2, k, 2, tile=(1, 1), row_k=row_k), ROWS, COLS)
    return TileAnchors.of(C.PearlC(forming="v1", device=DEVICES[dev]), ctx,
                          pc8.Shape(2, k, 2, tile=(1, 1), dev=dev, row_k=row_k), ROWS, COLS)


def _regions(t: TileAnchors, role: str, index: int, segment: int | None = None) -> list[dict]:
    regions = []
    for port, words in t.regions(role, index, segment).items():
        anchor = t.shape.ports[role]["regions"][port]
        d = t.shape.derivations[anchor]
        r = {"port": port, "anchor": anchor, "seed": d.seed, "derive": d.rule, "bits": d.bits, "n_words": len(words),
             "bytes_sha256": hashlib.sha256(region_bytes(words)).hexdigest()}
        if d.per_row:
            r["bytes"] = region_bytes(words).hex()
        regions.append(r)
    return regions


def _file(t: TileAnchors, role: str, indexes: tuple[int, ...], segment: int | None = None) -> dict:
    head, body = t.public_inputs(role, indexes, segment).split(b"\n", 1)
    return {"role": role, "indexes": list(indexes), "header": head.decode(), "body_sha256": hashlib.sha256(body).hexdigest()}


def _case(t: TileAnchors, case: str, circuit: str, k: int, dev: str | None) -> dict:
    return {"case": case, "circuit": circuit, "k": k, "dev": dev, "salt": part(case, "salt").hex(),
            "root_a": part(case, "root_a").hex(), "root_b": part(case, "root_b").hex(), "matmul": 3, "weight": 9,
            "seed_a": t.seed_a.hex(), "seed_b": t.seed_b.hex(), "rows": list(ROWS), "cols": list(COLS)}


def case_vector(case: str, circuit: str, k: int, dev: str | None) -> dict:
    t = anchors(case, circuit, k, dev)
    units = [{"role": role, "index": index, "regions": _regions(t, role, index)} for role, index in UNITS]
    return _case(t, case, circuit, k, dev) | {"units": units, "public_inputs": [_file(t, *f) for f in FILES]}


def rowk_case_vector(case: str, circuit: str, k: int, dev: str | None, row_k: int) -> dict:
    t = anchors(case, circuit, k, dev, row_k)
    units = []
    for role in rowk.ROLES["check"]:
        if t.shape.ports[role]["regions"]:
            index = 1 if role.startswith("row_a") else 0
            segment = 1 if role.endswith("_form") else 0 if role.endswith("_first") else None
            units.append({"role": role, "index": index, "segment": segment, "regions": _regions(t, role, index, segment)})
    files = [_file(t, role, indexes, segment) | {"segment": segment} for role, indexes, segment in ROWK_FILES]
    return _case(t, case, circuit, k, dev) | {"row_k": row_k, "units": units, "public_inputs": files}


def vectors() -> dict:
    return {"schema": "verity-pouw/regions-vectors/v1", "word_bits": 16, "region_word_bits": 128,
            "cases": [case_vector(*c) for c in CASES], "row_k_cases": [rowk_case_vector(*c) for c in ROWK_CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
