"""The frozen input sets of the four non-A100 Table 2 rows as one binary file each, for the SP1 format guests.

    python -m verity_sp1.format_instances --format fp8-ada --n 4096 --procs 16 --out DIR/fp8-ada.bin

Run from the repository root with the workspace ``PYTHONPATH`` (``research env --source .``); it imports the
B-Ligero instance generators, so it needs numpy and torch and runs on a pod, never on the laptop.

The four sets are synthetic and pinned by recipe (``verity_numerical.bench.contract.*_INSTANCES_*``,
``verity_numerical.bench.tables.FROZEN_INSTANCES``): VU ``i`` draws its operands from
``np.random.default_rng([seed, i])`` and records its accumulators through the model.  This module calls the very
functions the B-Ligero cells proved -- ``relchain._instance`` for the grouped-sum relations (``bf16-hopper``,
``fp8-hopper``, ``fp8-ada``) and ``fp4.chain.instances_fp4`` for ``fp4-nvf4`` -- and checks the recipe digest against
the frozen ref before writing anything.

File (``sp1-format-instances/v1``)::

    b"VYSP1FI\\x01" || u32 LE header length || header (UTF-8 JSON) || x block || W block || y block

The blocks are VU-major: ``n`` rows of ``row_bytes`` (x, then W), then ``n`` words of ``y_bytes`` (little-endian).
A row is the format's operand bytes in ascending k: BF16 words as ``u16`` LE (``row_bytes = 2 K``), E4M3 bytes
(``K``), NVFP4 firings of 64 element codes (one byte each) then 4 UE4M3 scale bytes (``24 x 68``).  ``y`` is the
relation's public word: ``f32_to_bf16`` of the final accumulator for ``bf16-hopper`` (``u16``), the final FP32
accumulator word otherwise (``u32``).  The header carries the frozen ref (``instances``), the target name, the
shape and the sha256 of each block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MAGIC = b"VYSP1FI\x01"
SCHEMA = "sp1-format-instances/v1"
K_VU = 1536

#: format -> (target attribute of verity.ml.circuits.targets.target, row bytes, y bytes, relation)
FORMATS = {
    "bf16-hopper": ("BF16_HOPPER", 2 * K_VU, 2, "bf16-hopper"),
    "fp8-hopper": ("FP8_HOPPER_WGMMA", K_VU, 4, "fp8-hopper"),
    "fp8-ada": ("FP8_ADA_MMA", K_VU, 4, "fp8-ada"),
    "fp4-nvf4": ("NVFP4_SM120", 24 * 68, 4, None),
}


def _groupsum_chunk(args: tuple[str, int, int, int]) -> list[tuple[bytes, bytes, int]]:
    """VUs ``[lo, hi)`` of a grouped-sum relation through ``relchain._instance`` (operands, model chain)."""
    import numpy as np
    from archive.direct.ligero.relations import relation
    from archive.direct.ligero.relchain import _instance
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word

    name, seed, lo, hi = args
    rel = relation(name)
    dtype = "<u2" if rel.word_dtype == "<u2" else "u1"
    out = []
    for i in range(lo, hi):
        a, b, accs, y_final = _instance((name, seed, i))
        assert len(accs) == rel.steps and accs[-1] == y_final
        y = f32_to_bf16_word(y_final) if name == "bf16-hopper" else y_final
        out.append((np.asarray(a, dtype=dtype).tobytes(), np.asarray(b, dtype=dtype).tobytes(), int(y)))
    return out


def _nvfp4_chunk(args: tuple[str, int, int, int]) -> list[tuple[bytes, bytes, int]]:
    """VUs ``[lo, hi)`` of ``fp4-nvf4`` through ``fp4.chain.instances_fp4``."""
    import numpy as np
    from archive.direct.ligero.fp4.chain import instances_fp4

    _, seed, lo, hi = args
    out = []
    for A, B, accs, y_final in instances_fp4(hi - lo, seed=seed, start=lo):
        A, B = np.asarray(A), np.asarray(B)
        assert A.shape == (24, 68) and B.shape == (24, 68) and int(A.max()) < 256 and int(B.max()) < 256
        assert len(accs) == 24 and accs[-1] == y_final
        out.append((A.astype("u1").tobytes(), B.astype("u1").tobytes(), int(y_final)))
    return out


def recipe(fmt: str) -> tuple[object, dict, int, str]:
    """(target, frozen ref, seed, recipe digest) of ``fmt``; the digest must equal the frozen ref's manifest_sha256."""
    from verity.ml.circuits.targets import target as targets
    from verity_numerical.bench.tables import FROZEN_INSTANCES

    tgt = getattr(targets, FORMATS[fmt][0])
    ref = FROZEN_INSTANCES[tgt.name]
    if fmt == "fp4-nvf4":
        from archive.direct.ligero.fp4.chain import INSTANCE_SEED, instances_digest
        seed, digest = INSTANCE_SEED, instances_digest(ref["range"][1])
    else:
        from archive.direct.ligero.relations import relation
        from archive.direct.ligero.relchain import instances_digest
        rel = relation(FORMATS[fmt][3])
        seed, digest = rel.instance_seed, instances_digest(rel, ref["range"][1])
    if digest != ref["manifest_sha256"]:
        raise SystemExit(f"{fmt}: recipe digest {digest} != frozen {ref['manifest_sha256']}")
    return tgt, ref, seed, digest


def generate(fmt: str, n: int, procs: int) -> tuple[dict, bytes, bytes, bytes]:
    tgt, ref, seed, _ = recipe(fmt)
    _, row_bytes, y_bytes, rel = FORMATS[fmt]
    chunk = max(1, n // (4 * max(1, procs)))
    jobs = [(rel or fmt, seed, lo, min(n, lo + chunk)) for lo in range(0, n, chunk)]
    fn = _nvfp4_chunk if fmt == "fp4-nvf4" else _groupsum_chunk
    if procs > 1:
        with ProcessPoolExecutor(procs) as ex:
            parts = list(ex.map(fn, jobs))
    else:
        parts = [fn(j) for j in jobs]
    vus = [v for part in parts for v in part]
    assert len(vus) == n
    x = b"".join(v[0] for v in vus)
    w = b"".join(v[1] for v in vus)
    y = b"".join(v[2].to_bytes(y_bytes, "little") for v in vus)
    assert len(x) == len(w) == n * row_bytes and len(y) == n * y_bytes
    header = {
        "schema": SCHEMA, "format": fmt, "target": tgt.name, "instances": ref, "n": n, "K": K_VU,
        "row_bytes": row_bytes, "y_bytes": y_bytes,
        "x_sha256": hashlib.sha256(x).hexdigest(), "w_sha256": hashlib.sha256(w).hexdigest(),
        "y_sha256": hashlib.sha256(y).hexdigest(),
        "generator": "verity_sp1.format_instances (relchain._instance / fp4.chain.instances_fp4)",
    }
    return header, x, w, y


def write(path: Path, header: dict, x: bytes, w: bytes, y: bytes) -> None:
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(h)) + h + x + w + y)


def read(path: Path) -> tuple[dict, bytes, bytes, bytes]:
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not an {SCHEMA} file")
    (hl,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12:12 + hl])
    n, rb, yb = header["n"], header["row_bytes"], header["y_bytes"]
    o = 12 + hl
    x, w, y = data[o:o + n * rb], data[o + n * rb:o + 2 * n * rb], data[o + 2 * n * rb:o + 2 * n * rb + n * yb]
    if len(y) != n * yb or o + 2 * n * rb + n * yb != len(data):
        raise ValueError(f"{path}: truncated or oversized")
    return header, x, w, y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--format", required=True, choices=sorted(FORMATS))
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    header, x, w, y = generate(args.format, args.n, args.procs)
    write(args.out, header, x, w, y)
    print(json.dumps({**header, "path": str(args.out), "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
