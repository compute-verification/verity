"""Pearl-C4's device-check fixture (`pearl_c4/fixture.py`) by its command line, as `build.sh` runs it: a format's
directory holds the files its manifest names, at their sizes and digests, and on rows and columns across the tile its
buffers are the scalar reference's (`verity.protocols.pouw.prover.schemes.pearl_c4`): the seeds, both operands' formed
codes, scale bytes, α and a_E, the peel factors P_A and P_B, and C̃ and U."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.protocols.pouw.prover.schemes import pearl_c4 as F
from verity.protocols.pouw.verifier.protocol import Context, Matmul, Shape, Tile

FIXTURE = Path(__file__).resolve().parents[1] / "pearl_c4" / "fixture.py"
M, N, K = 128, 128, 1024
ROWS, COLS = (0, 77, 127), (1, 64)


@pytest.fixture(scope="module", params=["nvfp4", "mxfp4"])
def written(request, tmp_path_factory) -> tuple[str, Path, dict, str]:
    """One format's directory as `fixture.py OUT --fmt FMT` writes it: the format, the directory, its manifest and what
    the run printed."""
    out = tmp_path_factory.mktemp("check")
    p = subprocess.run([sys.executable, str(FIXTURE), str(out), "--fmt", request.param], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert [d.name for d in out.iterdir()] == [request.param]
    return request.param, out / request.param, json.loads((out / request.param / "manifest.json").read_text()), p.stdout


def read(d: Path, name: str, dtype: object, *shape: int) -> np.ndarray:
    return np.frombuffer((d / f"{name}.bin").read_bytes(), dtype).reshape(*shape)


def codes(packed: np.ndarray) -> np.ndarray:
    """(rows, k/2) bytes -> (rows, k) 4-bit codes, the low nibble first."""
    return np.stack([packed & 0xF, packed >> 4], axis=2).reshape(packed.shape[0], -1)


def test_the_directory_holds_what_its_manifest_names(written):
    fmt, d, manifest, stdout = written
    s = F.PearlC4(fmt, hashing="h0")
    assert stdout == f"{fmt}: 35 files, admissible True, {s.name} at ({M}, {N}, {K})\n"
    assert {k: v for k, v in manifest.items() if k != "files"} == {
        "scheme": s.name, "fmt": fmt, "shape": {"m": M, "n": N, "k": K}, "block": s.fmt.block, "admissible": True,
        "roles": {"ea": 0, "fa": 1, "eb": 2, "fb": 3}}
    assert len(manifest["files"]) == 35
    assert {f.name for f in d.iterdir()} == {"manifest.json", *(f"{name}.bin" for name in manifest["files"])}
    for name, entry in manifest["files"].items():
        data = (d / f"{name}.bin").read_bytes()
        assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, name


def test_its_buffers_are_the_scalar_reference_s(written):
    fmt, d, _, _ = written
    s = F.PearlC4(fmt, hashing="h0")
    ctx = Context((d / "salt.bin").read_bytes(), Matmul(0, Shape(M, K, N), 1), (d / "root_a.bin").read_bytes(),
                  (d / "root_b.bin").read_bytes())
    noise = s.noise(ctx)
    assert (noise.seed_a, noise.seed_b) == ((d / "seed_a.bin").read_bytes(), (d / "seed_b.bin").read_bytes())
    xa, xb = read(d, "xa", "<u2", M, K), read(d, "xb", "<u2", N, K)
    ops = s.encode(ctx, noise, [xa[i].tobytes() for i in ROWS], [xb[j].tobytes() for j in COLS], Tile(ROWS, COLS))
    assert ops is not None
    checked = s.checked(ctx, ops)
    for side, rows, count in (("a", ROWS, M), ("b", COLS, N)):
        formed, at = getattr(ops, side), list(rows)
        assert codes(read(d, f"{side}_codes", np.uint8, count, K // 2))[at].tolist() == formed.codes, side
        scales = read(d, f"{side}_scales", np.uint8, K // 128, count, 128 // s.fmt.block).transpose(1, 0, 2)
        assert scales.reshape(count, -1)[at].tolist() == formed.scales, side
        assert read(d, f"{side}_alpha", "<u4", count)[at].tolist() == formed.alpha, side
        assert read(d, f"{side}_ae", np.uint8, count)[at].tolist() == formed.ae, side
    assert read(d, "pa", "<u2", M, -1)[list(ROWS)].tolist() == ops.pa
    assert read(d, "pb", "<u2", N, -1)[list(COLS)].tolist() == ops.pb
    assert read(d, "c", "<u4", M, N)[np.ix_(ROWS, COLS)].tolist() == checked.c
    assert read(d, "u", "<u4", M, N)[np.ix_(ROWS, COLS)].tolist() == checked.u
