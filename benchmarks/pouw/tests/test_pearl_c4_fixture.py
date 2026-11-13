"""Pearl-C4's device-check fixture (`pearl_c4/fixture.py`) by its command line, as `build.sh` runs it: a format's
directory holds the files its manifest names, at their sizes and digests, and each buffer but its three inputs (A's
rows, B's rows and the salt) is held to what the verifier and the scalar reference
(`verity.protocols.pouw.prover.schemes.pearl_c4`) make of those inputs.  The roots and domain ids are
`audit.commit_rows`' over the rows, the seeds the scheme's noise of those roots, and the noise lines, both bases and G
the seeds'.  On rows and columns across the tile: both operands' formed codes, scale bytes, α, 1/α and a_E, the peel
factors P_A and P_B, B̃·F_A, C̃ and U, and the two unscaled outputs.  The message digests and tile leaves are the
scheme's over the directory's own C̃ and U, at every cell."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.commitments import MerkleTree
from verity.definitions.silicon import fp32
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_c4 as F
from verity.protocols.pouw.prover.schemes import pearl_kw as P
from verity.protocols.pouw.verifier import audit as AU
from verity.protocols.pouw.verifier.protocol import Context, Matmul, Shape, Tile

FIXTURE = Path(__file__).resolve().parents[1] / "pearl_c4" / "fixture.py"
M, N, K = 128, 128, 1024
WEIGHT = 1
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


def trees(d: Path, s: F.PearlC4) -> dict[str, MerkleTree]:
    """The verifier's trees over the directory's rows: A's as matmul 0's activations under the salt, B's as the
    weight's."""
    salt = (d / "salt.bin").read_bytes()
    a_rows, b_rows = ([r.tobytes() for r in read(d, name, "<u2", count, K)] for name, count in (("xa", M), ("xb", N)))
    return {"a": AU.commit_rows(AU.activation_binding(salt, 0), 0, a_rows, s.commitment_hash, public=True),
            "b": AU.commit_rows(AU.weight_binding(WEIGHT), -2, b_rows, s.commitment_hash, public=True)}


def context(d: Path, s: F.PearlC4) -> Context:
    """The matmul's context from the directory's inputs alone: its roots are `trees`', never the directory's."""
    made = trees(d, s)
    return Context((d / "salt.bin").read_bytes(), Matmul(0, Shape(M, K, N), WEIGHT), made["a"].commitment.root,
                   made["b"].commitment.root)


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


def test_its_roots_are_the_verifier_s_trees_over_its_rows(written):
    fmt, d, _, _ = written
    for side, tree in trees(d, F.PearlC4(fmt, hashing="h0")).items():
        assert (d / f"dom_{side}.bin").read_bytes() == tree.domain.domain_id, side
        assert (d / f"root_{side}.bin").read_bytes() == tree.commitment.root, side


def test_its_lines_bases_and_g_are_its_seeds(written):
    """Every noise line as the reference samples it (line i of A's seed for row i of A; of B̃'s seed for row j of B and
    for position t of F_A and F_B), both bases transposed, and G = F_Bᵀ·F_A."""
    fmt, d, _, _ = written
    s = F.PearlC4(fmt, hashing="h0")
    noise = s.noise(context(d, s))
    for name, seed, role, count in (("ea", noise.seed_a, F.ROLE_EA, M), ("eb", noise.seed_b, F.ROLE_EB, N),
                                    ("fa", noise.seed_b, F.ROLE_FA, K), ("fb", noise.seed_b, F.ROLE_FB, K)):
        lines = codes(read(d, f"lines_{name}", np.uint8, count, F.R // 2))
        assert lines.tolist() == [F.line(seed, role, i) for i in range(count)], name
    for name, role in (("fat", F.ROLE_FA), ("fbt", F.ROLE_FB)):
        basis = np.array(F.basis(noise.seed_b, role, K))
        assert codes(read(d, name, np.uint8, F.R, K // 2)).tolist() == basis.T.tolist(), name
    assert read(d, "g", "<u4", F.R, F.R).tolist() == [list(row) for row in F.gram(s.fmt, noise.seed_b, K)]


def test_its_buffers_are_the_scalar_reference_s(written):
    """The unscale has no scalar reference: it is pearl_c4.cu's, y = bf16((w · 1/α_A) · 1/α_B) with 1/α one FP32
    division and each product one FP32 multiply, nearest even, here on `fp32`'s words."""
    fmt, d, _, _ = written
    s = F.PearlC4(fmt, hashing="h0")
    ctx = context(d, s)
    noise = s.noise(ctx)
    assert (noise.seed_a, noise.seed_b) == ((d / "seed_a.bin").read_bytes(), (d / "seed_b.bin").read_bytes())
    xa, xb = read(d, "xa", "<u2", M, K), read(d, "xb", "<u2", N, K)
    ops = s.encode(ctx, noise, [xa[i].tobytes() for i in ROWS], [xb[j].tobytes() for j in COLS], Tile(ROWS, COLS))
    assert ops is not None
    checked = s.checked(ctx, ops)
    inv = {}
    for side, rows, count in (("a", ROWS, M), ("b", COLS, N)):
        formed, at = getattr(ops, side), list(rows)
        assert codes(read(d, f"{side}_codes", np.uint8, count, K // 2))[at].tolist() == formed.codes, side
        scales = read(d, f"{side}_scales", np.uint8, K // 128, count, 128 // s.fmt.block).transpose(1, 0, 2)
        assert scales.reshape(count, -1)[at].tolist() == formed.scales, side
        assert read(d, f"{side}_alpha", "<u4", count)[at].tolist() == formed.alpha, side
        assert read(d, f"{side}_ae", np.uint8, count)[at].tolist() == formed.ae, side
        inv[side] = [fp32.div(F.ONE, alpha) for alpha in formed.alpha]
        assert read(d, f"{side}_inv", "<u4", count)[at].tolist() == inv[side], side
    assert read(d, "pa", "<u2", M, -1)[list(ROWS)].tolist() == ops.pa
    assert read(d, "pb", "<u2", N, -1)[list(COLS)].tolist() == ops.pb
    fa, ones = F.basis(noise.seed_b, F.ROLE_FA, K), [s.fmt.one] * (K // s.fmt.block)
    assert read(d, "bf", "<u4", N, F.R)[list(COLS)].tolist() == [
        [F.chain(s.fmt, b_codes, [line[q] for line in fa], b_scales, ones) for q in range(F.R)]
        for b_codes, b_scales in zip(ops.b.codes, ops.b.scales)]
    for name, y, words in (("c", "y_plain", checked.c), ("u", "y", checked.u)):
        assert read(d, name, "<u4", M, N)[np.ix_(ROWS, COLS)].tolist() == words, name
        assert read(d, y, "<u2", M, N)[np.ix_(ROWS, COLS)].tolist() == [
            [P.f32_to_bf16(P.f32_of_bits(fp32.mul(fp32.mul(w, ia), ib))) for w, ib in zip(row, inv["b"])]
            for row, ia in zip(words, inv["a"])], y


def test_its_digests_and_leaves_are_the_scheme_s_over_its_checked_words(written):
    """Each (row, 32-column chunk) message's digest and each tile's leaf, over the directory's C̃ and U: every cell of
    both enters one message, so a digest or a leaf of other words than the ones beside it fails here."""
    fmt, d, _, _ = written
    s = F.PearlC4(fmt, hashing="h0")
    ctx = context(d, s)
    noise = s.noise(ctx)
    c, u = read(d, "c", "<u4", M, N), read(d, "u", "<u4", M, N)
    digests = read(d, "digests", np.uint8, M, N // C.CHUNK, 32)
    leaves = read(d, "leaves", np.uint8, M // F.TILE, N // F.TILE, 32)
    for i in range(M // F.TILE):
        for j in range(N // F.TILE):
            rows, cols = slice(F.TILE * i, F.TILE * (i + 1)), slice(F.TILE * j, F.TILE * (j + 1))
            tile = C.Checked(c[rows, cols].tolist(), u[rows, cols].tolist())
            chunks = slice(cols.start // C.CHUNK, cols.stop // C.CHUNK)
            assert digests[rows, chunks].tobytes() == b"".join(C.message_digest(m) for m in C.messages(tile)), (i, j)
            assert leaves[i, j].tobytes() == s.digest(ctx, noise, tile), (i, j)
