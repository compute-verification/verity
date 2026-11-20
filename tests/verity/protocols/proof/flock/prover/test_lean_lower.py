"""`Flock.Lower.lower` on the chain's first block as `restrict` writes it (C_S, word level, its unit ``Gemm_v2{K=16, N=16}``
on Ampere's dot, lifted to bits by the lowering, whose Boolean bodies' batches read listed references), compiled
(`flock-lower-check`): every composite Definition's template `CALL`s within `Lower.callBound` of its lift, and each output
bit of the unit's template Lean writes, its rows unrolled and solved in Lean and the file read and unrolled by
`tmpl.unroll` and solved here, equal to the Python lowering's (`UnitStager`'s templated unit) on C_S's two instances and 62 random ones,
and to the trace's outputs on those two. The descriptor is 650 KB, past what the interpreter's `#guard`s take."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_lean_verifier import LEAN, lake_build
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_id
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import tmpl as T

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ROOT = Path(__file__).resolve().parents[6]
LAYOUT = "iv/hybrid/4"
LANES = 64


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def first_block(tmp_path: Path):
    """C_S of the chain's first window member (`restrict`), its unit's word Definition, and its instances' rows and outputs."""
    M = _load("one_circuit_chain", "tests/verity/protocols/proof/spot/verifier/test_one_circuit_chain.py")
    r = M.registered()
    m0 = r["window"].window.members[0]
    block = R.restrict(r["C"], r["partition"], [(m0.base, m0.base + m0.n)])
    rows, outs = M.instances(block, M.by_gate(block))
    key, = {descriptor_id(nd.fn) for _k, nd, _n in M._units(R.program(block.descriptor))}
    program, _q = M.circuit().program()
    word = next(nd.fn for nd in program.root.body.nodes if nd.form == "batch" and nd.fn.id.startswith("Gemm"))
    assert descriptor_id(word) == key, (descriptor_id(word), key)
    desc = tmp_path / "c_s.json"
    desc.write_bytes(block.descriptor)
    return desc, word, rows, outs


def _bits(words: np.ndarray) -> np.ndarray:
    """Each 16-bit word's bits, low first, word after word."""
    w = np.asarray(words, np.uint16).reshape(-1)
    return ((w[:, None] >> np.arange(16, dtype=np.uint16)) & 1).reshape(-1).astype(np.uint64)


def _lanes(rows: np.ndarray) -> np.ndarray:
    """``(instances, bits)`` 0/1 as ``(bits, 1)`` words, instance ``l`` lane bit ``l``."""
    return np.bitwise_or.reduce(rows.T.astype(np.uint64) << np.arange(rows.shape[0], dtype=np.uint64), axis=1)[:, None]


def _solve(t, words: np.ndarray, outs: int) -> np.ndarray:
    """Template ``t`` unrolled (`tmpl.unroll`) and its rows solved in column order, 64 instances a word: input bit ``i`` at
    column ``i``, the constant all ones, column ``c`` ``(A·z)(B·z)``, an unwritten one 0; its first ``outs`` output bits."""
    f = T.unroll(t)
    (out_col, _w), = T._out_groups(t)
    ones = np.uint64(2 ** 64 - 1)
    z = np.zeros(f.useful, np.uint64)
    z[:len(words)] = words
    z[f.const] = ones
    for c in range(len(words), f.useful):
        if c == f.const:
            continue
        a, b = f.row(c)
        if a or b:
            z[c] = np.bitwise_xor.reduce(z[list(a)]) & np.bitwise_xor.reduce(z[list(b)]) if a and b else np.uint64(0)
    return z[out_col:out_col + outs]


def _unlane(words: np.ndarray, n: int) -> np.ndarray:
    return ((np.asarray(words, np.uint64).reshape(-1, 1) >> np.arange(n, dtype=np.uint64)) & 1).T.astype(np.uint8)


@pytest.mark.slow
def test_lean_lowered_gemm_computes_the_python_lowering(tmp_path):
    desc, word, rows, outs = first_block(tmp_path)
    assert list(rows) == ["x", "w"] and list(outs) == ["o0"], (list(rows), list(outs))
    n = len(outs["o0"])
    rng = np.random.default_rng(1)
    member0 = [np.concatenate([_bits(rows["x"][u]), _bits(rows["w"][u])]) for u in range(n)]
    insts = np.stack(member0 + [rng.integers(0, 2, 256 + 4096).astype(np.uint64) for _ in range(LANES - n)])
    words = _lanes(insts)
    (tmp_path / "in.txt").write_text("".join(f"{int(v)}\n" for v in words[:, 0]))

    lake_build(LEAN, "flock-lower-check")
    p = subprocess.run([str(LEAN / ".lake" / "build" / "bin" / "flock-lower-check"), str(desc), descriptor_id(word),
                        "256,4096", "256", str(tmp_path / "in.txt"), str(tmp_path / "lean.tmpl"), str(tmp_path / "out.txt")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert "REFUSED" not in p.stdout, p.stdout[-4000:]
    out_bits = 16 * outs["o0"].shape[1]
    got = [int(v) for v in (tmp_path / "out.txt").read_text().split()]
    assert len(got) == out_bits, f"Lean's template gives {len(got)} output bits, not {out_bits}"
    lean = _unlane(got, LANES)

    stager = CS.UnitStager(word, f"unit/{word.id}", [("x", SCHEMA_HM96_SHA512_ROW_V2, 256, ["x"]),
                                                     ("w", SCHEMA_HM96_SHA512_ROW_V2, 4096, ["w"])],
                           [(SCHEMA_HM96_SHA512_ROW_V2, (0,))], layout=LAYOUT)
    python = _unlane(_solve(T.parse(FC.template_of(stager.text)), words[:, 0], out_bits), LANES)
    read = _unlane(_solve(T.parse((tmp_path / "lean.tmpl").read_text()), words[:, 0], out_bits), LANES)
    trace = np.stack([_bits(outs["o0"][u]) for u in range(n)])

    for name, v in (("Lean's", lean), ("the Python lowering's", python), ("Lean's file read by tmpl", read)):
        assert v.shape == (LANES, out_bits), f"{name} outputs have shape {v.shape}, not {(LANES, out_bits)}"
    assert trace.shape == (n, out_bits) and n >= 1
    assert np.array_equal(python[:n], trace), "the Python lowering is not the trace"
    assert np.array_equal(lean, python), f"Lean's template differs from the Python lowering at {np.argwhere(lean != python)[:8]}"
    assert np.array_equal(read, python), "Lean's file, read and unrolled by tmpl, differs from the Python lowering"
