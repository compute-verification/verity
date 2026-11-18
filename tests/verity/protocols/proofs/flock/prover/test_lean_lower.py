"""`Flock.Lower.lower` on the periods' chain's product (`versions.block_circuit` of ``Gemm_v2{K=16, N=16}`` on Ampere's dot,
whose Boolean bodies' batches read listed references), compiled (`flock-lower-check`): every composite Definition's
template `CALL`s within `Defn.callBound`, and each output bit of the template Lean writes, its rows unrolled and solved in
Lean and the file read and evaluated by `tmpl.evaluate`, equal to the Python lowering's (`UnitStager`'s templated unit, the
file the prover proves) on the chain's two member-0 units and 62 random instances, and to the trace's outputs on those
two. The descriptor is 650 KB, past what the interpreter's `#guard`s take."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_lean_verifier import LEAN, lake_build
from verity.circuits import population as PN
from verity.circuits.codec import canonical_json, descriptor_id, encode_program
from verity.circuits.ops.boolean import versions as V
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import tmpl as T

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ROOT = Path(__file__).resolve().parents[6]
LAYOUT = "iv/hybrid/4"
LANES = 64


def _chain():
    spec = importlib.util.spec_from_file_location("periods_test_interface",
                                                  ROOT / "tests/verity/commitments/periods/test_interface.py")
    ti = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ti)
    return ti


def _bits(words: np.ndarray) -> np.ndarray:
    """Each 16-bit word's bits, low first, word after word."""
    w = np.asarray(words, np.uint16).reshape(-1)
    return ((w[:, None] >> np.arange(16, dtype=np.uint16)) & 1).reshape(-1).astype(np.uint64)


def _lanes(rows: np.ndarray) -> np.ndarray:
    """``(instances, bits)`` 0/1 as ``(bits, 1)`` words, instance ``l`` lane bit ``l``."""
    return np.bitwise_or.reduce(rows.T.astype(np.uint64) << np.arange(rows.shape[0], dtype=np.uint64), axis=1)[:, None]


def _unlane(words: np.ndarray, n: int) -> np.ndarray:
    return ((np.asarray(words, np.uint64).reshape(-1, 1) >> np.arange(n, dtype=np.uint64)) & 1).T.astype(np.uint8)


@pytest.mark.slow
def test_lean_lowered_gemm_computes_the_python_lowering(tmp_path):
    ti = _chain()
    wp = PN.population_program(ti.GEMM, 1)
    word = next(n.fn for n in wp.root.body.nodes if n.fn.id.startswith("Gemm"))
    fn = V.block_circuit(word, V.boolean_table(wp))
    desc = tmp_path / "gemm.json"
    desc.write_bytes(canonical_json(encode_program(PN.population_program(fn, 1))))

    (xs, ws), (ys,) = ti.chain_values()
    rng = np.random.default_rng(1)
    member0 = [np.concatenate([_bits(xs[u]), _bits(ws[u])]) for u in (0, 1)]
    rows = np.stack(member0 + [rng.integers(0, 2, 256 + 4096).astype(np.uint64) for _ in range(LANES - 2)])
    words = _lanes(rows)
    (tmp_path / "in.txt").write_text("".join(f"{int(v)}\n" for v in words[:, 0]))

    lake_build(LEAN, "flock-lower-check")
    p = subprocess.run([str(LEAN / ".lake" / "build" / "bin" / "flock-lower-check"), str(desc), descriptor_id(fn),
                        "256,4096", "256", str(tmp_path / "in.txt"), str(tmp_path / "lean.tmpl"), str(tmp_path / "out.txt")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    lean = _unlane([int(v) for v in (tmp_path / "out.txt").read_text().split()], LANES)

    stager = CS.UnitStager(word, f"unit/{word.id}", [("x", SCHEMA_HM96_SHA512_ROW_V2, 256, ["x"]),
                                                     ("w", SCHEMA_HM96_SHA512_ROW_V2, 4096, ["w"])],
                           [(SCHEMA_HM96_SHA512_ROW_V2, (0,))], layout=LAYOUT)
    python = _unlane(T.evaluate(T.parse(FC.template_of(stager.text)), words)[:, 0], LANES)
    read = _unlane(T.evaluate(T.parse((tmp_path / "lean.tmpl").read_text()), words)[:, 0], LANES)

    assert np.array_equal(python[:2], np.stack([_bits(ys[u]) for u in (0, 1)])), "the Python lowering is not the trace"
    assert np.array_equal(lean, python), f"Lean's template differs from the Python lowering at {np.argwhere(lean != python)[:8]}"
    assert np.array_equal(read, python), "Lean's file, read and evaluated by tmpl.evaluate, differs from the Python lowering"
