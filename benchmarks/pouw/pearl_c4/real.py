"""Pearl-C4's reference replay on real activations (CPU; a development shortcut: the protocol's check is a ZK proof,
PROTOCOL.md), per linear of a `pearlc_capture.py` directory: X, the linear's input rows (float32 holding BF16 exactly),
and its weight W (the model's BF16 safetensors); or of an `fp4_v3_real.capture` npz file (X and W of each linear as
float32 holding BF16), judged in the clear as the scheme (`verity.core.protocols.pouw.schemes.pearl_c4`) judges a unit:

* admission per row of X and of W (`row_passes`, D-NF), over the full k and over the split unit, and the rows with a
  non-finite word (D-SB rejects the transcript);
* the split unit's screened blocks (`dead_blocks`, 9.5ρ), and the rows D-SS's retired one-in-64 cap would have rejected;
* on --tiles 64 x 64 grid tiles of the split unit, drawn by a seeded RNG: the verifier's forming with D-SB
  (`PearlC4._side`) and the replayed debit of the tile's credited words against the cap (`tile_cap`), item by item.

Shortcuts: the split is a stand-in, not a pinned rule (the ⌊k/100⌋ channels with the largest max|x| over the captured rows
leave the unit, ties to the lower channel; the kept ones are zero-padded to a multiple of 128); the noise is keyed by
labels, not an epoch's beacon; NVFP4 only.

    python real.py CAPTURE_DIR OUT_DIR --layers 'layers\\.(0|8)\\.' --tiles 2 --workers 8 --budget-min 12

Each task's JSON in OUT_DIR is its checkpoint.  A run starts tasks while its budget lasts, then exits 99 while tasks
remain (one fill chunk), and 0 once `summary.json` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_c_work as W
from verity.core.protocols.pouw.schemes import pearl_kw as P

TILE, ROWS_PER_TASK, ALIGN = 64, 2048, 128
SECONDS_8K = {"admit": 30.0, "tile": 150.0}     # a task's first estimate at k = 8,192; measured ones replace it


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def x_codes(path: Path | np.ndarray, label: str = "") -> np.ndarray:
    """X (or an npz capture's W) as BF16 codes (m x k uint16); it must hold BF16 values exactly."""
    bits = np.ascontiguousarray(np.load(path) if isinstance(path, Path) else path, dtype=np.float32).view(np.uint32)
    if (bits & 0xFFFF).any():
        raise SystemExit(f"{label or path}: not BF16-exact")
    return (bits >> 16).astype(np.uint16)


def npz_manifest(path: Path) -> dict:
    """A manifest for an `fp4_v3_real.capture` npz (``layers.<i>.<linear>.x`` and ``.w``, float32 holding BF16): per
    linear ``model.layers.<i>.<linear>``, its X's shape from the member's header (nothing is loaded)."""
    def shape(z, member: str) -> list[int]:
        fmt = np.lib.format
        with z.zip.open(member) as f:
            read = fmt.read_array_header_1_0 if fmt.read_magic(f) == (1, 0) else fmt.read_array_header_2_0
            return list(read(f)[0])

    inputs = {}
    with np.load(path) as z:
        for member in z.zip.namelist():
            if member.endswith(".x.npy"):
                name = "model." + member[:-len(".x.npy")]
                inputs[name] = {"file": name + ".npy", "shape": shape(z, member),
                                "w_rows": shape(z, member[:-len(".x.npy")] + ".w.npy")[0]}
    return {"model": path.name, "npz": str(path), "inputs": inputs}


def capture_sha256(job: "Job") -> str:
    """The capture's identity: its manifest's sha256, or an npz's own (hashed once, kept in the output directory)."""
    if not job.npz:
        return sha256(job.capture / "manifest.json")
    mark = job.out / "capture.sha256"
    if not mark.exists():
        mark.write_text(sha256(job.capture))
    return mark.read_text()


def w_codes(snapshot: Path, name: str) -> np.ndarray:
    """W as BF16 codes (n x k uint16), memory-mapped from its safetensors shard."""
    tensor = name + ".weight"
    path = snapshot / json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"][tensor]
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n))[tensor]
    if meta["dtype"] != "BF16":
        raise SystemExit(f"{tensor}: {meta['dtype']}, not BF16")
    return np.memmap(path, dtype="<u2", mode="r", offset=8 + n + meta["data_offsets"][0], shape=tuple(meta["shape"]))


def split_side(x: np.ndarray) -> list[int]:
    """The stand-in split: the ⌊k/100⌋ channels with the largest max|x|, ties to the lower channel."""
    k = x.shape[1]
    mags = np.abs((x.astype(np.uint32) << 16).view(np.float32)).max(axis=0)
    return sorted(int(c) for c in np.lexsort((np.arange(k), -mags))[:k // 100])


def unit_rows(codes: np.ndarray, keep: np.ndarray, k_unit: int) -> list[bytes]:
    out = np.zeros((codes.shape[0], k_unit), dtype="<u2")
    out[:, :len(keep)] = codes[:, keep]
    return [r.tobytes() for r in out]


class Job:
    """A `pearlc_capture.py` directory, or an `fp4_v3_real.capture` npz file (X and W of each linear)."""

    def __init__(self, capture: Path, out: Path, seed: int):
        self.capture, self.out, self.seed = capture, out, seed
        self.npz = capture.suffix == ".npz"
        self.man = npz_manifest(capture) if self.npz else json.loads((capture / "manifest.json").read_text())
        self.snapshot = None if self.npz else Path(self.man["snapshot"])

    def _member(self, name: str, part: str) -> np.ndarray:
        key = name[len("model."):] + "." + part
        with np.load(self.capture) as z:
            return x_codes(z[key], f"{self.capture}:{key}")

    def x(self, name: str) -> np.ndarray:
        if self.npz:
            return self._member(name, "x")
        return x_codes(self.capture / self.man["inputs"][name]["file"])

    def w(self, name: str) -> np.ndarray:
        return self._member(name, "w") if self.npz else w_codes(self.snapshot, name)

    def keep(self, name: str) -> tuple[np.ndarray, int]:
        """(kept channels, k_unit) of `name`'s split unit."""
        k = self.man["inputs"][name]["shape"][-1]
        side = json.loads((self.out / f"split-{Path(self.man['inputs'][name]['file']).stem}.json").read_text())["side"]
        keep = np.setdiff1d(np.arange(k), np.array(side, dtype=np.int64))
        return keep, -(-len(keep) // ALIGN) * ALIGN

    def operand(self, name: str, side: str) -> np.ndarray:
        return self.x(name) if side == "x" else self.w(name)


def admit(job: Job, name: str, side: str, r0: int, r1: int) -> dict:
    """D-NF over full and split rows, non-finite rows, and the split rows' screened blocks."""
    codes = np.asarray(job.operand(name, side)[r0:r1])
    keep, k_unit = job.keep(name)
    k = codes.shape[1]
    got = {"rows": r1 - r0, "non_finite": 0, "dnf_fail_full": 0, "dnf_fail_split": 0, "screened_blocks": 0, "blocks": 0,
           "rows_screened": 0, "rows_old_cap": 0}
    for full, split in zip((r.tobytes() for r in codes), unit_rows(codes, keep, k_unit)):
        try:
            x_full, x_split = F.parse_row(full, k), F.parse_row(split, k_unit)
        except P.Reject:
            got["non_finite"] += 1
            continue
        got["dnf_fail_full"] += not F.row_passes(F.NVFP4, x_full)
        got["dnf_fail_split"] += not F.row_passes(F.NVFP4, x_split)
        dead = F.dead_blocks(F.NVFP4, x_split)
        got["screened_blocks"] += sum(dead)
        got["blocks"] += len(dead)
        got["rows_screened"] += any(dead)
        got["rows_old_cap"] += 64 * sum(dead) > len(dead)
    return got


def tile(job: Job, name: str, t: int) -> dict:
    """One seeded grid tile of the split unit through the verifier's forming (D-SB) and its replayed debit."""
    x, w = job.x(name), job.w(name)
    keep, k_unit = job.keep(name)
    (m, _), n = x.shape, w.shape[0]
    rng = random.Random(f"{job.seed}/{name}/{t}")
    ti, tj = rng.randrange(m // TILE), rng.randrange(n // TILE)
    rows, cols = list(range(ti * TILE, (ti + 1) * TILE)), list(range(tj * TILE, (tj + 1) * TILE))
    fast = os.environ.get("PEARL_C4_F1") == "fast"
    if fast:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import f1_fast
    s = F.PearlC4("nvfp4", f1=f1_fast.row_units if fast else None)
    unit = Matmul(0, Shape(m, k_unit, n), 0)
    lab = lambda part: hashlib.sha256(f"pearl-c4-real/{name}/{t}/{part}".encode()).digest()  # noqa: E731
    noise = s.noise(Context(lab("salt"), unit, lab("root_a"), lab("root_b")))
    got = {"tile": [ti, tj], "m": m, "n": n, "k_kept": len(keep), "k_unit": k_unit, "f1": "fast" if fast else "exact"}
    try:
        a, e_a = s._side(unit_rows(x[rows], keep, k_unit), rows, F.ROLE_EA, F.ROLE_FA, noise.seed_a, noise.seed_b, k_unit)
        b, e_b = s._side(unit_rows(np.asarray(w[cols]), keep, k_unit), cols, F.ROLE_EB, F.ROLE_FB, noise.seed_b,
                         noise.seed_b, k_unit)
    except P.Reject as e:
        return got | {"rejected": str(e)}
    cred_a = [p for p in range(TILE) if F.row_passes(F.NVFP4, a.x[p])]
    cred_b = [p for p in range(TILE) if F.row_passes(F.NVFP4, b.x[p])]
    got |= {"credited": [len(cred_a), len(cred_b)], "rejected": ""}
    if not (cred_a and cred_b):
        return got
    debit, cap = s.tile_cap(W.UnitRecord(unit, m, len(keep)), SimpleNamespace(a=a, b=b, e_a=e_a, e_b=e_b), cred_a, cred_b)
    return got | {"debit": s.last_debit.to_json(), "cap": str(cap), "ratio": float(debit / cap)}


def run(task: tuple) -> tuple[tuple, dict]:
    kind, capture, out, seed, *rest = task
    job, t0 = Job(Path(capture), Path(out), seed), time.monotonic()
    got = admit(job, *rest) if kind == "admit" else tile(job, *rest)
    return task, got | {"seconds": round(time.monotonic() - t0, 1)}


def task_file(task: tuple) -> str:
    return f"{task[0]}-" + "-".join(str(v) for v in task[4:]).replace("model.layers.", "L") + ".json"


def summarize(job: Job, tasks: list[tuple]) -> dict:
    """Admission summed per input file (X) and per weight (W), and every tile's result per linear."""
    inputs: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    tiles: dict[str, list] = {}
    for task in tasks:
        got = json.loads((job.out / task_file(task)).read_text())
        name = task[4]
        if task[0] == "tile":
            tiles.setdefault(name, []).append(got)
            continue
        acc = (inputs.setdefault(job.man["inputs"][name]["file"], {}) if task[5] == "x" else weights.setdefault(name, {}))
        for key, v in got.items():
            if key != "seconds":
                acc[key] = acc.get(key, 0) + v
    flat = [t for ts in tiles.values() for t in ts]
    ratios = [t["ratio"] for t in flat if "ratio" in t]
    return {"schema": "pouw/pearl-c4-real/v0", "capture": str(job.capture),
            "manifest_sha256": capture_sha256(job), "model": job.man.get("model"),
            "revision": job.man.get("revision"), "seed": job.seed, "inputs": inputs,
            "weights": weights, "tiles": tiles, "tiles_debited": len(ratios), "max_ratio": max(ratios, default=None),
            "rejected_tiles": sum(bool(t.get("rejected")) for t in flat), "over_cap_tiles": sum(r > 1 for r in ratios)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", default="", help="measure the linears whose module name this regex matches")
    ap.add_argument("--tiles", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-min", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20261930)
    ap.add_argument("--f1", choices=("exact", "fast"), default="exact",
                    help="F1′'s evaluation: the rule's own, or f1_fast's cheap sound one (staged, pending the assessor's rating)")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    os.environ["PEARL_C4_F1"] = args.f1
    args.out.mkdir(parents=True, exist_ok=True)
    job = Job(args.capture, args.out, args.seed)
    names = [n for n in job.man["inputs"] if re.search(args.layers, n)]
    for name in names:
        ent = job.man["inputs"][name]
        mark = args.out / f"split-{Path(ent['file']).stem}.json"
        if not mark.exists():
            if job.npz:
                digest = capture_sha256(job)
            else:
                f, digest = args.capture / ent["file"], ent["sha256"]
                if sha256(f) != digest:
                    raise SystemExit(f"{f}: sha256 differs from the manifest")
            side = split_side(job.x(name))
            mark.write_text(json.dumps({"file": ent["file"], "sha256": digest, "k": ent["shape"][-1], "side": side}))
    common = (str(args.capture), str(args.out), args.seed)
    tasks, seen_x = [], set()
    for name in names:
        ent = job.man["inputs"][name]
        m = int(np.prod(ent["shape"][:-1]))
        if ent["file"] not in seen_x:
            seen_x.add(ent["file"])
            tasks += [("admit", *common, name, "x", r, min(r + ROWS_PER_TASK, m)) for r in range(0, m, ROWS_PER_TASK)]
        n = ent["w_rows"] if job.npz else w_codes(job.snapshot, name).shape[0]
        tasks += [("admit", *common, name, "w", r, min(r + ROWS_PER_TASK, n)) for r in range(0, n, ROWS_PER_TASK)]
        tasks += [("tile", *common, name, t) for t in range(args.tiles)]
    k_of = {name: job.man["inputs"][name]["shape"][-1] for name in names}
    pending = [t for t in tasks if not (args.out / task_file(t)).exists()]
    measured: dict[tuple, float] = {}
    for t in tasks:
        f = args.out / task_file(t)
        if f.exists():
            key = (t[0], k_of[t[4]])
            measured[key] = max(measured.get(key, 0.0), json.loads(f.read_text())["seconds"])

    def estimate(t: tuple) -> float:
        k = k_of[t[4]]
        return measured.get((t[0], k), SECONDS_8K[t[0]] * k / 8192)

    budget, started = 60 * args.budget_min, 0

    def fits(t: tuple) -> bool:
        return not started or time.monotonic() - t0 + estimate(t) < budget

    with ProcessPoolExecutor(args.workers) as pool:
        flight: dict = {}
        while pending or flight:
            while pending and len(flight) < args.workers and fits(pending[0]):
                t = pending.pop(0)
                flight[pool.submit(run, t)] = t
                started += 1
            if not flight:
                break
            done, _ = wait(flight, return_when=FIRST_COMPLETED)
            for fut in done:
                t, got = fut.result()
                del flight[fut]
                tmp = args.out / (task_file(t) + ".tmp")
                tmp.write_text(json.dumps(got))
                tmp.replace(args.out / task_file(t))
                measured[(t[0], k_of[t[4]])] = max(measured.get((t[0], k_of[t[4]]), 0.0), got["seconds"])
                print(f"{task_file(t)} {got['seconds']}s", flush=True)
    if pending:
        print(f"{len(pending)} tasks remain", flush=True)
        return 99
    (args.out / "summary.json").write_text(json.dumps(summarize(job, tasks), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
