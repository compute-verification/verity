"""The prover's openings of a Pearl+ pass from its directory (``pass-NNNNN``, the manifest's layout: `manifest`): a tile's
A and B rows and transcript leaf (`Opened.open`, as `clear_audit.Prover.open` has them) and its real rows of the call's
committed y (`Opened.open_y`), each read from the pass's files (`Files`) and opened in the tree as the device laid it out
(`opening`). ``verify_run`` replays a pass through them."""
from __future__ import annotations

import hashlib
import mmap
from pathlib import Path

import numpy as np
from verity.commitments import Opening, hm96

HM96_SALT = hm96.SHA512.salt_bytes


def pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def opening(tree: bytes, count: int, pos: int, value: bytes, nb: int = 32) -> Opening:
    """Position `pos`'s path in a tree laid out as the device writes it: level 0 padded to a power of two, then each level
    above it, `nb` bytes a node (64 under hm96-sha512)."""
    path, off, width, cur = [], 0, pow2(count), pos
    while width > 1:
        s = off + (cur ^ 1)
        path.append(bytes(tree[nb * s:nb * s + nb]))
        off, width, cur = off + width, width // 2, cur >> 1
    return Opening(pos, value, tuple(path))


class Files:
    """The pass's files, each read once (large ones memory-mapped) and held to its size and, where the manifest has one, its
    SHA-256."""

    def __init__(self, base: Path):
        self.base, self.cache = base, {}

    def __call__(self, spec: dict):
        key = spec["path"]
        if key not in self.cache:
            with open(self.base / key, "rb") as fh:
                data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if spec["bytes"] > 1 << 24 else fh.read()
            if len(data) != spec["bytes"]:
                raise ValueError(f"{key}: {len(data)} bytes; the manifest says {spec['bytes']}")
            if "sha256" in spec and hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"{key}: its SHA-256 is not the manifest's")
            self.cache[key] = data
        return self.cache[key]


def f32_row(data, k: int, i: int) -> bytes:
    """Row i of a BF16 row file as the committed FP32 words (BF16 widened, exact)."""
    return (np.frombuffer(data[2 * k * i:2 * k * (i + 1)], "<u2").astype("<u4") << 16).tobytes()


class Opened:
    """The prover's side of the audit from the pass's files: `open`, as `clear_audit.Prover` has it. ``salted``: the
    scheme commits under hm96-sha512, each leaf followed by its kept salt; ``tile_opening``: what `open` returns,
    `clear_audit.TileOpening` as the caller's verifier reads it; ``epoch``, in `open` and `open_y`, is an `audit.Epoch`."""

    def __init__(self, man: dict, files: Files, refs, salted: bool, tile_opening: type):
        self.man, self.files, self.refs, self.hm96, self.tile_opening = man, files, refs, salted, tile_opening
        self.nb = 64 if self.hm96 else 32

    def _salted(self, value: bytes, spec: dict | None, i: int) -> bytes:
        """``value`` followed by leaf ``i``'s salt from the kept salts at manifest entry ``spec``."""
        return value + bytes(self.files(spec)[HM96_SALT * i:HM96_SALT * (i + 1)]) if self.hm96 else value

    def open(self, epoch, tile_id: int):
        ref, nb, sv = self.refs[tile_id], self.nb, self._salted
        u, rec = epoch.workload.matmuls[ref.matmul], self.man["matmuls"][ref.matmul]
        wt = self.man["weights"][str(u.weight)]
        a_rows, a_tree = self.files(rec["rows"]), self.files(rec["tree"])
        b_rows, b_tree = self.files(wt["rows"]), self.files(wt["tree"])
        c = self.man["commitment"]
        leaves = self.files(self.man["leaves"])
        a_salts, b_salts, t_salts = (rec["salts"], wt["salts"], self.man["salts_t"]) if self.hm96 else (None, None, None)
        return self.tile_opening(ref, tuple(opening(a_tree, u.shape.m, i, sv(f32_row(a_rows, u.shape.k, i), a_salts, i), nb)
                                      for i in ref.tile.rows),
                           tuple(opening(b_tree, u.shape.n, j, sv(f32_row(b_rows, u.shape.k, j), b_salts, j), nb)
                                 for j in ref.tile.cols),
                           opening(self.files(self.man["tree_t"]), c["count"], tile_id,
                                   sv(bytes(leaves[nb * tile_id:nb * tile_id + nb]), t_salts, tile_id), nb))

    def open_y(self, epoch, tile_id: int) -> tuple[Opening, ...]:
        """The tile's real rows of its call's committed y (`served_y`; filler rows aren't served)."""
        ref = self.refs[tile_id]
        u, rec = epoch.workload.matmuls[ref.matmul], self.man["matmuls"][ref.matmul]
        rows, tree, real = self.files(rec["y"]["rows"]), self.files(rec["y"]["tree"]), rec["real_rows"]
        salts = rec["y"]["salts"] if self.hm96 else None
        return tuple(opening(tree, real, i, self._salted(f32_row(rows, u.shape.n, i), salts, i), self.nb)
                     for i in ref.tile.rows if i < real)


__all__ = ["Files", "Opened", "f32_row", "opening", "pow2"]
