"""PoUW's tile-check rows (`verity_pouw.circuit.leaves`) are C-Flock's rows byte for byte: every serial row's vector of
``protocols/pouw/tests/vectors/pouw_rows.json``, recomputed from its case's words and salt through the port C-Flock stages
for the row (`circuit._port`: ``n_words`` u16 leaves, a `sha512/row/v1` row of whole blocks under ``hm96-sha512/row/v1`` or
a bit row of ``16 n_words`` bits under ``hm96-sha512/row/v2``), gives the pinned row bytes, digest, commit string
(`circuit._commit_strings`) and tree leaf; and one vector of each row version and a tile's give the same commit string and
tree leaf from the Lean verifier's own row constants (`Flock.HmRow`).  The per-call rows under ``hm96-sha512/row-seg/v1``
wait for C-Flock's segmented port (`AWAITING`)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
from verity_flock import circuit as FC
from verity_pouw.circuit import leaves as L

POUW_TESTS = Path(__file__).resolve().parents[3] / "protocols" / "pouw" / "tests"
LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
_spec = importlib.util.spec_from_file_location("pouw_rows_vectors", POUW_TESTS / "pouw_rows_vectors.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
PINNED = json.loads((POUW_TESTS / "vectors" / "pouw_rows.json").read_text())
#: row schemas C-Flock has no port for yet: the segmented rows (proofs' stage 2 of `sha512/row-seg/v1`, after #757)
AWAITING = {SCHEMA_HM96_SHA512_ROW_SEG: "awaiting C-Flock's hm96-sha512/row-seg/v1 port (proofs' row-seg stage 2)"}


def _case(case: dict) -> tuple[list[int], bytes]:
    """The case's 16-bit words (PoUW's schema encoder of its derived elements) and its salt."""
    row, k = case["row"], case["k"]
    words = V.words(row, V.inputs(case["case"], row, tuple(k) if row == "tile" else k))
    return words, V.shake(case["case"], "salt", L.INST.salt_bytes)


def _flock_port(case: dict) -> FC.Port:
    n = case["n_words"]
    return FC._port(case["case"], n, None if case["row_schema"] == SCHEMA_HM96_SHA512_ROW else 16 * n)


def test_every_schema_has_a_flock_row():
    """Every row and role of PoUW's has a pinned vector, and its schema is a C-Flock port's or one `AWAITING` names."""
    assert {(c["row"], V.ROLES[c["role"]]) for c in PINNED["cases"]} == set(L.ROW_SCHEMA)
    assert {row for row, _role in L.ROW_SCHEMA} == set(L.SCHEMAS)
    assert set(L.ROW_SCHEMA.values()) <= {SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, *AWAITING}
    assert PINNED["key_sha256"] == hashlib.sha256(hm96.SHA512.default_key).hexdigest()


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.skip(reason=AWAITING[c["row_schema"]]))
                                  if c["row_schema"] in AWAITING else c for c in PINNED["cases"]], ids=lambda c: c["case"])
def test_pouw_rows_are_flock_rows(case):
    """C-Flock's port for the row commits exactly the pinned leaf: its schema is the vector's ``row_schema``, its leaves'
    bytes hash to ``row_bytes_sha256``, its row digest is ``x``, its ``b || c`` is ``bc``, and hm96's tree leaf over that
    is ``tree_leaf``."""
    words, salt = _case(case)
    port = _flock_port(case)
    assert case["row_schema"] == L.ROW_SCHEMA[case["row"], V.ROLES[case["role"]]] == port.schema
    assert len(words) == case["n_words"] == port.words and port.n_bits == 16 * len(words)
    rows = np.array([words], dtype="<u2")
    raw = rows[0].tobytes()[:port.bytes]
    assert hashlib.sha256(raw).hexdigest() == case["row_bytes_sha256"]
    assert hashlib.sha256(salt).hexdigest() == case["salt_sha256"]
    assert port.layout().digest(raw).hex() == case["x"]
    bc = FC._commit_strings([port], {port.name: rows}, {port.name: [salt]})[port.name][0]
    assert bc.hex() == case["bc"]
    inst = hm96.SHA512
    assert inst.tree_leaf(inst.default_key, bc).hex() == case["tree_leaf"]


LEAF_OF = r"""import Flock
open Flock HmRow

/-- A row's `b ‖ c` and tree leaf from the verifier's constants (as `test_lean_verifier`'s `bcOf`, for a `sha512/row/v1`
port of `words` u16 words or a bit row of `16 words` bits): block `k`'s column `t` is `constBit`'s constant, else the row's
bit; the blocks compressed from `midstateOf`, the digest under hm96's default key with `salt`. -/
def leafOf (v2 : Bool) (words : Nat) (row salt : String) : String := Id.run do
  let row := match unhex row with | .ok b => b | .error _ => ByteArray.empty
  let salt := match unhex salt with | .ok b => b | .error _ => ByteArray.empty
  let p : Port := { name := "p", words := words, chunks := 0, nb := 0, bits := 16 * words, v2 := v2 }
  let tl := tailOf p
  let mut cv := midstateOf p
  for k in [0:(p.bytes + 17 + 127) / 128] do
    let mut blk := ByteArray.mk (Array.replicate 128 0)
    for t in [0:1024] do
      let (o, b) := blockBit t
      let v := match constBit p tl k t with
        | some c => c
        | none => ((row.get! (128 * k + o)).toNat >>> b) % 2 == 1
      if v then blk := blk.set! o (blk.get! o ||| ((1 : UInt8) <<< b.toUInt8))
    cv := Sha512.compress cv blk 0
  let x : ByteArray := ⟨Array.ofFn (n := 64) fun i => (cv[i.val / 8]! >>> (8 * (7 - i.val % 8)).toUInt64).toUInt8⟩
  let bc := Hm96.Default512.commitString x salt
  return hexOf bc ++ " " ++ hexOf (Hm96.Default512.treeLeaf bc)
"""

#: a v1 row, a packed v2 row whose bytes end inside a block, and a tile digest (a v2 row of no compression); the Lean
#: verifier has no row-seg row yet either
LEAN_CASES = ("bf16-w-64", "nvfp4-p-w-64", "tile-2x33")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_verifiers_row_constants_give_the_pinned_leaves(tmp_path):
    """The Lean verifier's row constants for the port (`HmRow.prefixOf`, `midstateOf`, `constBit` past the row's bits,
    `tailOf`), compressed over the row's bytes, give the pinned ``b || c`` and tree leaf of a v1 row, a v2 row and a tile."""
    cases = [c for c in PINNED["cases"] if c["case"] in LEAN_CASES]
    assert [c["case"] for c in cases] == list(LEAN_CASES)
    lines = []
    for case in cases:
        words, salt = _case(case)
        v2 = "true" if case["row_schema"] == SCHEMA_HM96_SHA512_ROW_V2 else "false"
        lines.append(f'#eval leafOf {v2} {len(words)} "{np.array(words, dtype="<u2").tobytes().hex()}" "{salt.hex()}"\n')
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "PouwRows.lean"
    f.write_text(LEAF_OF + "".join(lines))
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines() == [f'"{c["bc"]} {c["tree_leaf"]}"' for c in cases]
