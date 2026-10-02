"""PoUW's tile-check rows (`verity_pouw.circuit.leaves`) are C-Flock's rows byte for byte: every row's vector of
``protocols/pouw/tests/vectors/pouw_rows.json``, recomputed from its case's words and salt through the port C-Flock stages
for the row (`circuit._port`: ``n_words`` u16 leaves, a bit row of ``16 n_words`` bits under ``hm96-sha512/row/v2`` or
``hm96-sha512/row-seg/v1``), gives the pinned row bytes, digest, commit string (`circuit._commit_strings`) and tree leaf;
one vector of each row version and a tile's give the same commit string and tree leaf from the Lean verifier's own row
constants (`Flock.HmRow`).  The row a PoUW row unit writes as several output ports (A′_i ‖ P_A,i, Pearl-C's fp8-p row),
staged as one segmented output row (META ``out_row_ports``), is PoUW's row: its writer's and its reader's commit strings
are PoUW's, on C-Flock's writer/reader pairs (`test_lean_verifier.ROW_SEG_PAIRS`).  Each unit's output rows
(`leaves.statement_rows`) are the rows C-Flock's composer gives its output ports, but for the ones `AWAITING` names."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lean_verifier import ROW_SEG_PAIRS, _bit_row_toy, _public_bcs, _seg_writer_toy
from verity.commitments import hm96
from verity.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
from verity.ir.types import Tuple
from verity_flock import circuit as FC
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import pc4, pc8

POUW_TESTS = Path(__file__).resolve().parents[3] / "protocols" / "pouw" / "tests"
LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
_spec = importlib.util.spec_from_file_location("pouw_rows_vectors", POUW_TESTS / "pouw_rows_vectors.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
PINNED = json.loads((POUW_TESTS / "vectors" / "pouw_rows.json").read_text())


def _case(case: dict) -> tuple[list[int], bytes]:
    """The case's 16-bit words (PoUW's schema encoder of its derived elements) and its salt."""
    row, k = case["row"], case["k"]
    words = V.words(row, V.inputs(case["case"], row, tuple(k) if row == "tile" else k))
    return words, V.shake(case["case"], "salt", L.INST.salt_bytes)


def _flock_port(case: dict) -> FC.Port:
    n = case["n_words"]
    return FC._port(case["case"], n, 16 * n, case["row_schema"] == SCHEMA_HM96_SHA512_ROW_SEG)


def test_every_schema_has_a_flock_row():
    """Every row and role of PoUW's has a pinned vector, and its schema is one C-Flock's META ``row_schemas`` names."""
    assert {(c["row"], V.ROLES[c["role"]]) for c in PINNED["cases"]} == set(L.ROW_SCHEMA)
    assert {row for row, _role in L.ROW_SCHEMA} == set(L.SCHEMAS)
    assert set(L.ROW_SCHEMA.values()) == {SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG}
    assert PINNED["key_sha256"] == hashlib.sha256(hm96.SHA512.default_key).hexdigest()


@pytest.mark.parametrize("case", PINNED["cases"], ids=lambda c: c["case"])
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
    assert port.digest(raw).hex() == case["x"]
    bc = FC._commit_strings([port], {port.name: rows}, {port.name: [salt]})[port.name][0]
    assert bc.hex() == case["bc"]
    inst = hm96.SHA512
    assert inst.tree_leaf(inst.default_key, bc).hex() == case["tree_leaf"]


def _out_words(fn) -> list[int]:
    """Each output port's words: its leaves' bits, 16 a word."""
    out = fn.signature()[1]
    return [sum(t.leaf_widths()) // 16 for t in (out.fields if isinstance(out, Tuple) else (out,))]


def _elements(row: str, k: int, words: list[int]) -> list[int]:
    """The row's words as `leaves.words_*` gives them from its elements (A′'s codes and scales, or the fp8 codes, and P)."""
    n = k // 4 if row == "nvfp4-p" else k // 2
    per, bits = (4, 4) if row == "nvfp4-p" else (2, 8)
    codes = [(w >> (bits * q)) & ((1 << bits) - 1) for w in words[:n] for q in range(per)]
    p = words[-L.P_WORDS:]
    if row == "fp8-p":
        return L.words_fp8_p(codes, p)
    scales = [(w >> (8 * q)) & 0xFF for w in words[n:n + k // 32] for q in range(2)]
    return L.words_nvfp4_p(codes, scales, p)


#: each writer/reader pair's PoUW row and unit: A′_i ‖ P_A,i of `Pc4RowA` at k = 4,096 and Pearl-C's fp8-p row of `Pc8RowA`
#: at k = 1,024
PAIRS = {"row-seg-pair/nvfp4-p-k4096": ("nvfp4-p", 4096, pc4), "row-seg-pair/fp8-p-k1024": ("fp8-p", 1024, pc8)}


@pytest.mark.parametrize("vid", list(ROW_SEG_PAIRS))
def test_a_row_unit_writes_its_row_seg_row_as_its_reader_commits_it(vid, tmp_path):
    """A row unit's per-call row written as its output ports (codes, scales and P: `Pc4RowA` at k = 4,096 writes A′_i ‖
    P_A,i as 1,024 + 128 + 64 words, `Pc8RowA` at k = 1,024 its fp8-p row as 512 + 64) is C-Flock's writer of the pair,
    and PoUW's statement rows are its META: the writer's output row and the tile's input row are both `statement_rows`'
    [row-seg, bits] entry for the row, one run of the unit's first output ports.  On each instance the row's words are
    PoUW's row of their elements (`leaves.words_*`), and PoUW's commit string of them under the writer's salt
    (`leaves.commit`, role X) is the writer's ``b || c`` and the reader's; instance 0's row digest and tree leaf are the
    pair's pinned ones."""
    row, k, mod = PAIRS[vid]
    out_words, want = ROW_SEG_PAIRS[vid]
    unit = mod.templates(k)["row_a"]
    rows = L.statement_rows(mod.PORTS["row_a"], k)
    entry = [L.ROW_SCHEMA[row, ROLE_X], 16 * L.ROW_WORDS[row](k)]
    assert _out_words(unit)[:len(out_words)] == out_words and rows["out_row_ports"][0] == len(out_words)
    assert rows["row_schemas"][1] == entry
    tile = L.statement_rows(mod.PORTS["tile"], k)
    assert tile["row_schemas"][0] == entry

    dw, dr = tmp_path / "writer", tmp_path / "reader"
    text, got, salts = _seg_writer_toy(out_words, dw)
    rtext = _bit_row_toy([entry[1]], dr, seg=frozenset({"p0"}), rows={"p0": got}, salts={"p0": salts})
    meta, rmeta = (json.loads(t.split("\n", 2)[1][5:]) for t in (text, rtext))
    assert meta["row_schemas"][1] == entry == rmeta["row_schemas"][0]
    assert meta["out_row_ports"] == [len(out_words)]
    bw, br = _public_bcs(dw / "pub.bin", 2), _public_bcs(dr / "pub.bin", 2)
    for i in range(len(got)):
        words = [int(v) for v in got[i]]
        assert _elements(row, k, words) == words, (vid, i)
        bc = L.commit(row, ROLE_X, words, salts[i])
        assert bw[i][1] == br[i][0] == bc, (vid, i)
    words0 = [int(v) for v in got[0]]
    assert L.row_digest(row, ROLE_X, words0).hex() == want["x"]
    assert L.tree_leaf(L.commit(row, ROLE_X, words0, salts[0])).hex() == want["tree_leaf"]


_PADDED = ("C-Flock pads a lone output port's v2 row to whole SHA-512 blocks (64 words); PoUW's row is a v2 row of exactly "
           "its words")
_RUNS = "C-Flock commits a run of output ports (META out_row_ports) as a row-seg row only; PoUW's registered row is a v2 row"
#: the output rows C-Flock's composer can't commit as PoUW's rows yet, and why (proofs' stage 2 after #849)
AWAITING = {("ok", ROLE_X): _PADDED, ("ok", ROLE_W): _PADDED, ("tile", ROLE_X): _PADDED,
            ("nvfp4-p", ROLE_W): _RUNS, ("fp8-p", ROLE_W): _RUNS}
#: each check's hiding unit templates (`PORTS` are theirs) at a small depth
UNITS = {"pc4": (lambda k: pc4.templates(k, pc4.TILE, True), 64),
         "pc8": (lambda k: pc8.templates(k, pc8.TILE, "sm120", True), 128)}
MODS = {"pc4": pc4, "pc8": pc8}
OUT_ROWS = [(m, role, j, out[0], out[1]) for m, mod in MODS.items() for role, ports in mod.PORTS.items()
            for j, out in enumerate(r for r in ports["rows"] if isinstance(r[2][0], int))]


@pytest.mark.parametrize("m, role, j, row, row_role", [
    pytest.param(*o, marks=pytest.mark.skip(reason=AWAITING[o[3], o[4]])) if (o[3], o[4]) in AWAITING else o for o in OUT_ROWS],
    ids=[f"{m}-{role}-{row}" for m, role, _j, row, _r in OUT_ROWS])
def test_a_units_output_rows_are_the_rows_flock_composes(m, role, j, row, row_role):
    """Output row ``j`` of the unit (`leaves.statement_rows`: its schema, bits and run of output ports) is the row C-Flock's
    composer gives the unit's output ports (`circuit._out_ports`, a bit-row statement with each row-seg run in
    ``seg_out_rows``): the same schema, bits and run."""
    templates, k = UNITS[m]
    fn = templates(k)[role]
    rows = L.statement_rows(MODS[m].PORTS[role], k)
    n_out = len(rows["out_row_ports"])
    outs = [[f"o{i}", w] for i, w in enumerate(_out_words(fn))]
    starts = np.cumsum([0] + rows["out_row_ports"])
    runs = [tuple(nm for nm, _w in outs[starts[i]:starts[i + 1]])
            for i, (s, _b) in enumerate(rows["row_schemas"][-n_out:]) if s == SCHEMA_HM96_SHA512_ROW_SEG]
    got, counts = FC._out_ports(outs, True, runs)
    assert [counts[j], [got[j].schema, got[j].bits]] == [rows["out_row_ports"][j], rows["row_schemas"][-n_out + j]]


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

#: a registered bf16 row (a v2 row of whole blocks), a packed v2 row whose bytes end inside a block, and a tile digest (a
#: v2 row of no compression); a row-seg row's leaf in Lean is `test_lean_verifier`'s writer/reader pairs'
LEAN_CASES = ("bf16-w-64", "nvfp4-p-w-64", "tile-2x33")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_verifiers_row_constants_give_the_pinned_leaves(tmp_path):
    """The Lean verifier's row constants for the port (`HmRow.prefixOf`, `midstateOf`, `constBit` past the row's bits,
    `tailOf`), compressed over the row's bytes, give the pinned ``b || c`` and tree leaf of two v2 rows and a tile."""
    cases = [c for c in PINNED["cases"] if c["case"] in LEAN_CASES]
    assert [c["case"] for c in cases] == list(LEAN_CASES)
    assert {c["row_schema"] for c in cases} == {SCHEMA_HM96_SHA512_ROW_V2}
    lines = []
    for case in cases:
        words, salt = _case(case)
        lines.append(f'#eval leafOf true {len(words)} "{np.array(words, dtype="<u2").tobytes().hex()}" "{salt.hex()}"\n')
    subprocess.run(["lake", "build", "Flock"], cwd=LEAN, check=True, capture_output=True)
    f = tmp_path / "PouwRows.lean"
    f.write_text(LEAF_OF + "".join(lines))
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.splitlines() == [f'"{c["bc"]} {c["tree_leaf"]}"' for c in cases]
