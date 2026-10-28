"""A CPU model of J10's banded noise as read from SharedNoise.lean (a band of g column tiles a row block).
Credit's banded reference (verity/core/protocols/pouw/circuit/banded.py's banded_reference, at `CA_SHA`) equals
this model on its pinned vectors and on leader and non-leader receipts; a banded gate holds the kernels to that code
itself (`ca_reference`, j10ca_ref.py under the tree `ca_tree` finds) and is labelled `CA_LABEL`, and a run
without that tree is labelled `LABEL`.

Unit u = (I, J) of a call, u = I ncb + J, sits at position base + u (as today), its band b = J // g and its band leader
unit (I, b g). seed_j(v) = seeds.seed(salt, base + v, root_a, root_b) and σ_R(v) = seeds.receipt_seed(...) as today.
* Row streams are shared along the band: unit (I, J)'s row i is its leader's h1tile row i, the blocks i RB .. of
  seed_j(leader)'s stream (h1tile.TileShape(32, 32, K)): up to g units read one row stream.
* Column streams stay per unit (they carry I): the unit's own h1tile columns, blocks 32 RB + j CB .. of seed_j(u).
* Fold keys stay per unit: h1tile.fold_keys(σ_R(u)).
With g = 1 every unit is its own leader and this is h1tile.reference exactly (`self_check`). The unit's z and receipt
on that noise are v3's (j10r_ca.unit on h1v3_ca: h1tile's word functions and v3's unpeeled stream_cell).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

import h1v3_ca as V3
import j10r_ca as J10

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD
from verity_config.settings import of

settings = of(__file__)
LABEL = "noise=j10-banded (j10b_ref's reading of SharedNoise.lean; not credit's banded reference)"
NOISE = "j10-banded/j10b_ref-reading"
CA_SHA = "1a4d1de024"
CA_LABEL = f"noise=j10-banded (credit's banded reference @ {CA_SHA})"
HERE = Path(__file__).resolve().parent
_CA: dict = {}


def ca_tree() -> str | None:
    """The tree of credit's banded reference (at CA_SHA): $J10_CA_TREE, else the run's inputs/banded-CA_SHA.tar.gz
    (`git archive CA_SHA verity | gzip`) extracted once a process; None without either."""
    if "tree" not in _CA:
        tree = settings.J10_CA_TREE
        arc = Path(settings.RESEARCH_RUN_DIR or ".") / "inputs" / f"banded-{CA_SHA}.tar.gz"
        if not tree and arc.is_file():
            tree = tempfile.mkdtemp(prefix="j10ca-")
            with tarfile.open(arc) as t:
                t.extractall(tree, filter="data")
        ok = tree and (Path(tree) / "verity/core/protocols/pouw/circuit/banded.py").is_file()
        _CA["tree"] = tree if ok else None
    return _CA["tree"]


def ca_reference(units: list[dict], workers: int = 8) -> dict:
    """credit's banded_reference on `units` (j10ca_ref.py's IN) under ca_tree(): {"tree", "sha", "units", ...}, or
    {"tree": None} without the tree, or with "error"."""
    tree = ca_tree()
    if not tree:
        return {"tree": None}
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(src, "w") as f:
            json.dump(units, f)
        p = subprocess.run([sys.executable, str(HERE / "j10ca_ref.py"), src, dst, str(workers)],
                           env={**os.environ, "PYTHONPATH": tree}, capture_output=True, text=True)
        if p.returncode:
            return {"tree": tree, "sha": CA_SHA, "error": p.stderr[-2000:]}
        with open(dst) as f:
            return {"tree": tree, "sha": CA_SHA, **json.load(f)}


def leader(u: int, ncb: int, g: int) -> int:
    """Unit u's band leader: the first unit of its row tile's band of g column tiles."""
    ri, cj = divmod(u, ncb)
    return ri * ncb + (cj // g) * g


def rows_of(seed_l: bytes, lay: HT.TileShape) -> list:
    """h1tile.noise's rows from the band leader's seed_j: per row (dither bits, digit coins)."""
    k, m = lay.k, lay.m
    out = []
    for i in range(lay.r):
        s = HT._stream(seed_l, lay.row(i), lay.row_bits)
        out.append((s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(m - 1)]))
    return out


def cols_of(seed_j: bytes, lay: HT.TileShape) -> list:
    """h1tile.noise's columns from the unit's own seed_j: per column (coin bits, U's int8 encodings)."""
    k = lay.k
    out = []
    for j in range(lay.c):
        s = HT._stream(seed_j, lay.col(j), lay.col_bits)
        out.append((s[:k], [HT.u129(HT._value(s[k + 8 * t:k + 8 * t + 8])) for t in range(32)]))
    return out


def band_noise(salt: bytes, pos: int, lead_pos: int, roots: tuple[bytes, bytes], k: int) -> tuple[list, list]:
    """The unit at `pos` whose band leader sits at `lead_pos`: (rows, columns) in h1tile.noise's format."""
    lay = HT.TileShape(32, 32, k)
    return rows_of(SD.seed(salt, lead_pos, *roots), lay), cols_of(SD.seed(salt, pos, *roots), lay)


def unit(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, pos: int, lead_pos: int,
         roots: tuple[bytes, bytes]) -> tuple[list[int], list[int]]:
    """v3's (z row-major, receipt) of one 32 x 32 unit (int8 encodings) on its banded noise and its own fold keys."""
    k = len(a[0])
    rows, cols = band_noise(salt, pos, lead_pos, roots, k)
    keys = HT.fold_keys(SD.receipt_seed(salt, pos, *roots), HT.TileShape(32, 32, k))
    return J10.unit(a, b, rows, cols, keys)


def reference(a: np.ndarray, b: np.ndarray, salt: bytes, base: int, roots: tuple[bytes, bytes], units: Sequence[int],
              g: int) -> dict[int, tuple[list[int], list[int], list, list]]:
    """(z, receipt, rows, columns) at the listed units of a call: A [rows, K] and B [N, K] int8, bands of g column tiles."""
    ncb, k = b.shape[0] // 32, a.shape[1]
    lay = HT.TileShape(32, 32, k)
    out = {}
    for u in units:
        ri, cj = divmod(u, ncb)
        rows, cols = band_noise(salt, base + u, base + leader(u, ncb, g), roots, k)
        keys = HT.fold_keys(SD.receipt_seed(salt, base + u, *roots), lay)
        ar = (a[32 * ri:32 * ri + 32].astype(np.int64) & 0xFF).tolist()
        br = (b[32 * cj:32 * cj + 32].astype(np.int64) & 0xFF).tolist()
        z, r = J10.unit(ar, br, rows, cols, keys)
        out[u] = (z, r, rows, cols)
    return out


def row_block(seed_l: bytes, i: int, c: int, k: int) -> list[int]:
    """Row i's c-th noise chunk from the leader's seed_j, as 16 big-endian halves (seeds.halves): what the kernels'
    band workspace holds at (row i, chunk c)."""
    return SD.halves(SD.noise_block(seed_l, HT.TileShape(32, 32, k).row(i) + c))


def col_block(seed_j: bytes, j: int, c: int, k: int) -> list[int]:
    return SD.halves(SD.noise_block(seed_j, HT.TileShape(32, 32, k).col(j) + c))


def self_check(seed: int = 7) -> dict:
    """g = 1 is h1tile.reference (v3's words, h1v3_ca.reference(stream=True)) on every unit of a 32 x 64 x 512 call;
    at g = 2 the leader (unit 0) is still per-unit, and unit 1 reads unit 0's rows and its own columns and keys."""
    import random
    rng = random.Random(seed)
    k, ncb = 512, 2
    a = np.array([[rng.randrange(-128, 128) for _ in range(k)] for _ in range(32)], dtype=np.int8)
    b = np.array([[rng.randrange(-128, 128) for _ in range(k)] for _ in range(32 * ncb)], dtype=np.int8)
    salt, roots, base = rng.randbytes(64), (rng.randbytes(64), rng.randbytes(64)), rng.getrandbits(40)
    a8 = (a.astype(np.int64) & 0xFF).tolist()
    bt = [(b[32 * j:32 * j + 32].astype(np.int64) & 0xFF).tolist() for j in range(ncb)]
    per = [V3.reference(a8, bt[j], salt, base + j, roots, stream=True) for j in range(ncb)]
    g1 = reference(a, b, salt, base, roots, range(ncb), 1)
    g2 = reference(a, b, salt, base, roots, range(ncb), 2)
    lay = HT.TileShape(32, 32, k)
    r0, c0 = HT.noise(SD.seed(salt, base, *roots), lay)
    _, c1 = HT.noise(SD.seed(salt, base + 1, *roots), lay)
    doc = {"g1_is_h1tile": all(list(g1[j][:2]) == list(per[j]) for j in range(ncb)),
           "g2_leader_is_h1tile": list(g2[0][:2]) == list(per[0]),
           "g2_unit1_rows_are_leaders": g2[1][2] == r0, "g2_unit1_cols_own": g2[1][3] == c1 and c1 != c0,
           "g2_unit1_differs": list(g2[1][:2]) != list(per[1])}
    doc["ok"] = all(doc.values())
    return doc


if __name__ == "__main__":
    import json
    print(json.dumps(self_check()))
