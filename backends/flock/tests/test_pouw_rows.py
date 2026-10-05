"""PoUW's tile-check rows (`verity.protocols.accounting.work.pouw.circuit.leaves`) are C-Flock's rows byte for byte: every row's vector of
``verity/protocols/accounting/work/pouw/tests/vectors/pouw_rows.json``, recomputed from its case's words and salt through the port C-Flock stages
for the row (`circuit._port`: ``n_words`` u16 leaves, a bit row of ``16 n_words`` bits under ``hm96-sha512/row/v2`` or
``hm96-sha512/row-seg/v1``), gives the pinned row bytes, digest, commit string (`circuit._commit_strings`) and tree leaf;
one vector of each row version and a tile's give the same commit string and tree leaf from the Lean verifier's own row
constants (`Flock.HmRow`).  Every output row a PoUW unit writes is PoUW's row of exactly its words: a per-call row written
as several output ports (A′_i ‖ P_A,i, Pearl-C's fp8-p row) as one segmented row, and the ok word, the tile digest and a
registered row written as several ports (B̃_j ‖ P_B,j, the fp8-p row) as one v2 row; their writers' and readers' commit
strings are PoUW's on C-Flock's writer/reader pairs (`test_lean_verifier.ROW_SEG_PAIRS`, ``ROW_V2_PAIRS``).  Each unit's
output rows (`leaves.statement_rows`) are the rows C-Flock's composer gives its output ports, and each of the six units'
statement rows compose: META is `statement_rows`', and every staged row's ``b || c`` is PoUW's commit string."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lean_verifier import ROW_SEG_PAIRS, ROW_V2_PAIRS, _bit_row_toy, _public_bcs, _seg_writer_toy
from verity.primitives.commitments import hm96
from verity.primitives.commitments.rowleaf import (ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2, row_bytes,
                                        sha512_row_v2_digest)
from verity.primitives.circuits.types import Tuple
from verity_flock import circuit as FC
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.circuit import pc4, pc8

POUW_TESTS = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "accounting" / "work" / "pouw" / "tests"
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


#: each check's hiding unit templates (`PORTS` are theirs) and a small depth
UNITS = {"pc4": (lambda k: pc4.templates(k, pc4.TILE, True), 64),
         "pc8": (lambda k: pc8.templates(k, pc8.TILE, "sm120", True), 128)}
MODS = {"pc4": pc4, "pc8": pc8}
OUT_ROWS = [(m, role, j, out[0], out[1]) for m, mod in MODS.items() for role, ports in mod.PORTS.items()
            for j, out in enumerate(r for r in ports["rows"] if isinstance(r[2][0], int))]


def _out_runs(ports: dict) -> tuple[tuple, tuple]:
    """A unit's output rows as `circuit.compose` takes them, over output ports ``o0, o1, …``: the row-seg runs
    (``seg_out_rows``) and the v2 rows of exactly their words, one port or several (``v2_out_rows``)."""
    seg, v2 = [], []
    for row, role, ps in ports["rows"]:
        if isinstance(ps[0], int):
            (seg if L.ROW_SCHEMA[row, role] == SCHEMA_HM96_SHA512_ROW_SEG else v2).append(tuple(f"o{p}" for p in ps))
    return tuple(seg), tuple(v2)


@pytest.mark.parametrize("m, role, j, row, row_role", OUT_ROWS, ids=[f"{m}-{role}-{row}" for m, role, _j, row, _r in OUT_ROWS])
def test_a_units_output_rows_are_the_rows_flock_composes(m, role, j, row, row_role):
    """Output row ``j`` of the unit (`leaves.statement_rows`: its schema and bits) is the row C-Flock's composer gives the
    unit's output ports (`circuit._out_ports`, a bit-row statement with each row-seg run in ``seg_out_rows`` and every
    other output row in ``v2_out_rows``), of the run of output ports `PORTS` names: a row of exactly its words, the ok word
    and the tile digest included."""
    templates, k = UNITS[m]
    ports = MODS[m].PORTS[role]
    rows = L.statement_rows(ports, k)
    outs = [[f"o{i}", w] for i, w in enumerate(_out_words(templates(k)[role]))]
    got, counts = FC._out_ports(outs, True, *_out_runs(ports))
    n_out = len(counts)
    run = [r for r in ports["rows"] if isinstance(r[2][0], int)][j][2]
    assert (row, row_role) == [r[:2] for r in ports["rows"] if isinstance(r[2][0], int)][j]
    assert [counts[j], [got[j].schema, got[j].bits]] == [len(run), rows["row_schemas"][-n_out + j]]
    assert got[j].bits == 16 * got[j].words == 16 * L.ROW_WORDS[row](k)


#: each exact-v2 writer/reader pair's PoUW row, its role, depth, check and writing unit: an ok row (`Pc4RowA`'s), a tile
#: digest (`Pc4TileHidden`'s), and the registered B̃_j ‖ P_B,j and fp8-p rows at k = 1,024 (`Pc4RowB`'s, `Pc8RowB`'s)
V2_PAIRS = {"row-v2-pair/ok": ("ok", ROLE_X, 1024, "pc4", "row_a"), "row-v2-pair/tile": ("tile", ROLE_X, 1024, "pc4", "tile"),
            "row-v2-pair/nvfp4-p-w-k1024": ("nvfp4-p", ROLE_W, 1024, "pc4", "row_b"),
            "row-v2-pair/fp8-p-w-k1024": ("fp8-p", ROLE_W, 1024, "pc8", "row_b")}


@pytest.mark.parametrize("vid", list(ROW_V2_PAIRS))
def test_a_units_v2_output_row_is_written_as_its_reader_commits_it(vid, tmp_path):
    """A unit's v2 output row of exactly its words, one port's (the 1-word ok row, the 16-word tile digest) or a run's (the
    registered B̃_j ‖ P_B,j as 256 + 32 + 64 words and fp8-p row as 512 + 64 at k = 1,024), is C-Flock's writer of the
    exact-v2 pair: the unit's output ports for the row are the pair's, the writer's output row is `statement_rows`' [v2,
    bits] entry for it, and so is the reader's input row (for a registered row, the tile's B row).  On each instance PoUW's
    commit string of the row's words under the writer's salt is the writer's ``b || c`` and the reader's, and instance 0's
    row digest and tree leaf are the pair's pinned ones.  The pair's ok word is not an admission value (bits 1-15 set), so
    PoUW's `commit` refuses it and its leaf is the v2 row's of the word under ok's schema and length."""
    row, role, k, m, unit = V2_PAIRS[vid]
    out_words, want = ROW_V2_PAIRS[vid]
    templates, mod = UNITS[m][0], MODS[m]
    ports = mod.PORTS[unit]
    (run,) = [ps for r, rl, ps in ports["rows"] if (r, rl) == (row, role) and isinstance(ps[0], int)]
    assert [_out_words(templates(k)[unit])[p] for p in run] == out_words
    entry = [L.ROW_SCHEMA[row, role], 16 * L.ROW_WORDS[row](k)]
    assert entry[0] == SCHEMA_HM96_SHA512_ROW_V2
    rows = L.statement_rows(ports, k)
    assert entry in rows["row_schemas"][-len(_out_runs(ports)[0] + _out_runs(ports)[1]):]
    if role == ROLE_W:
        assert L.statement_rows(mod.PORTS["tile"], k)["row_schemas"][1] == entry

    dw, dr = tmp_path / "writer", tmp_path / "reader"
    text, got, salts = _seg_writer_toy(out_words, dw, seg=False)
    rtext = _bit_row_toy([entry[1]], dr, rows={"p0": got}, salts={"p0": salts})
    meta, rmeta = (json.loads(t.split("\n", 2)[1][5:]) for t in (text, rtext))
    assert meta["row_schemas"][1] == entry == rmeta["row_schemas"][0]
    bw, br = _public_bcs(dw / "pub.bin", 2), _public_bcs(dr / "pub.bin", 2)

    def commit(words, salt):
        if row != "ok":
            return L.commit(row, role, words, salt)
        if words[0] not in (0, 1):
            with pytest.raises(ValueError, match="an ok row is one word"):
                L.commit(row, role, words, salt)
        return L.INST.commit_string(L.KEY, sha512_row_v2_digest(row_bytes(words, 16), entry[1]), salt)

    for i in range(len(got)):
        words = [int(v) for v in got[i]]
        if row in ("nvfp4-p", "fp8-p"):
            assert _elements(row, k, words) == words, (vid, i)
        assert bw[i][1] == br[i][0] == commit(words, salts[i]), (vid, i)
    words0 = [int(v) for v in got[0]]
    assert sha512_row_v2_digest(row_bytes(words0, 16), entry[1]).hex() == want["x"]
    if row != "ok":
        assert L.row_digest(row, role, words0).hex() == want["x"]
    assert L.tree_leaf(commit(words0, salts[0])).hex() == want["tree_leaf"]


def _unit_toy(m: str, role: str, d: Path, n: int = 2) -> tuple[dict, dict, list, dict]:
    """A statement of unit ``role`` of check ``m``'s rows at its small depth (`UNITS`, a 2 × 1 tile), staged in ``d``: its
    input rows `statement_rows`' (rows ``r0, r1, …``, each segmented row in ``seg_ports``), its output ports the unit's
    outputs (``o0, o1, …`` of the template's words), committed as its rows (`_out_runs`).  The gates are a toy: output word
    ``o`` is the XOR of input words ``o`` and ``o + 1`` (mod the input words) rotated by ``o mod 16``, but for the ok word,
    which is AND(bit 0, bit 1) of input word 0 at bit 0 and zero above, so an admission value.  Returns META, each input
    row's words, the output words and every row's salts."""
    from types import SimpleNamespace

    from verity.primitives.circuits.types import Array, Value
    from verity_flock import class_statement as CS
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    templates, k = UNITS[m]
    ports, reads = MODS[m].PORTS[role], {p: 2 for p in ("ac", "as", "pa")}
    rows = L.statement_rows(ports, k, reads)
    out_words = _out_words(templates(k)[role])
    seg_runs, v2_runs = _out_runs(ports)
    ins = rows["row_schemas"][:len(rows["row_schemas"]) - len(seg_runs) - len(v2_runs)]
    in_words = [bits // 16 for _s, bits in ins]
    (ok_port,) = [ps[0] for r, _rl, ps in ports["rows"] if r == "ok"] or [None]
    C = gf2.Circuit()
    leaves = [C.input(f"in{j}", 16) for j in range(sum(in_words))]
    at = np.cumsum([0] + out_words)
    for o in range(sum(out_words)):
        if ok_port is not None and o == at[ok_port]:
            out = [C.AND(leaves[0][0], leaves[0][1])] + [gf2.ZERO] * 15
        else:
            a, b, r = leaves[o % len(leaves)], leaves[(o + 1) % len(leaves)], o % 16
            out = [a[t] ^ b[(t + r) % 16] for t in range(16)]
        C.output(f"out{o}", out)
    W = sum(out_words)
    units = IL.Units(target=None, in_widths=[16] * len(leaves), out_widths=[16] * W,
                     in_src=[[("leaf", p) for p in range(len(leaves))]], out_src=[[("ret", o) for o in range(W)]],
                     signature=("toy",), gates=[], arg_gates=[], ret_gates=[])
    name = f"unit-class/pouw-{m}-{role}"
    sig = (tuple((f"r{i}", Array(w, Value(16))) for i, w in enumerate(in_words)),
           Tuple(tuple(Array(w, Value(16)) for w in out_words)))
    low = IL.Lowering(name, SimpleNamespace(id=name, signature=lambda: sig), units, C,
                      IL.layout(C, [f"in{j}" for j in range(len(leaves))], [f"out{o}" for o in range(W)]))
    CS.cache_slot_circuits()
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    content = json.dumps([m, role, k]).encode()
    sub = SimpleNamespace(id=name, template=SimpleNamespace(id="unit-class"), inputs=[port(f"r{i}") for i in range(len(ins))],
                          outputs=[port(f"o{j}") for j in range(len(out_words))],
                          program_digests=lambda statics: {name: hashlib.sha256(content).hexdigest()})
    iset = SimpleNamespace(subcircuit=sub, name=name.replace("/", "-"), content_digest=hashlib.sha256(content).hexdigest())
    seg = frozenset(f"r{i}" for i, (s, _b) in enumerate(ins) if s == SCHEMA_HM96_SHA512_ROW_SEG)
    text = FC.compose(low, sub, port_bits=[b for _s, b in ins], seg_ports=seg, seg_out_rows=seg_runs, v2_out_rows=v2_runs)
    rng = np.random.default_rng(7)
    in_rows = {f"r{i}": rng.integers(0, 1 << 16, (n, w), dtype=np.uint64).astype(np.uint16) for i, w in enumerate(in_words)}
    got, _cut, ok = low.evaluate(np.concatenate(list(in_rows.values()), axis=1).astype(np.uint64))
    assert int(ok) == (1 << n) - 1
    got = got.astype(np.uint16)
    outs = {f"o{j}": got[:, at[j]:at[j + 1]] for j in range(len(out_words))}
    meta = json.loads(text.split("\n", 2)[1][5:])
    salts = FC.stage_salts(iset, 0, n, FC._meta_ports(meta), b"k" * 32)
    d.mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    FC.write(text, iset, 0, n, d / "inst.bin", d / "pub.bin", b"k" * 32, rows=in_rows, outs=outs, salts=salts)
    return meta, in_rows, got, salts


@pytest.mark.parametrize("m, role", [(m, r) for m in MODS for r in ("row_a", "row_b", "tile")])
def test_each_unit_composes_its_statement_rows(m, role, tmp_path):
    """Each of the six units' statement rows compose in C-Flock (`circuit.compose`, over a toy unit with the unit's ports,
    `_unit_toy`): META's ``row_schemas`` and ``out_row_ports`` are `statement_rows`' (a 2 × 1 tile: two A rows, one B row),
    and every row's ``b || c`` in the staged public file is PoUW's commit string (`leaves.commit`) of that row's words in its
    role, under the row's staged salt: the input rows', and each output row's of its run of output words.  The anchors are
    public input regions, not rows, so the toy reads none."""
    meta, in_rows, got, salts = _unit_toy(m, role, tmp_path)
    templates, k = UNITS[m]
    ports = MODS[m].PORTS[role]
    rows = L.statement_rows(ports, k, {p: 2 for p in ("ac", "as", "pa")})
    assert meta["row_schemas"] == rows["row_schemas"]
    assert meta.get("out_row_ports") == rows.get("out_row_ports")
    names = [p[0] for p in meta["ports"]]
    which = [(r, rl) for r, rl, ps in ports["rows"] if isinstance(ps[0], str)
             for _ in range({"ac": 2, "as": 2, "pa": 2}.get(ps[0], 1))]
    at = np.cumsum([0] + _out_words(templates(k)[role]))
    outs = [(r, rl, ps) for r, rl, ps in ports["rows"] if isinstance(ps[0], int)]
    bcs = _public_bcs(tmp_path / "pub.bin", len(names))
    for i in range(len(got)):
        for p, (r, rl) in enumerate(which):
            words = [int(v) for v in in_rows[names[p]][i]]
            assert bcs[i][p] == L.commit(r, rl, words, salts[names[p]][i]), (m, role, i, names[p])
        for j, (r, rl, ps) in enumerate(outs):
            words = [int(v) for v in got[i, at[ps[0]]:at[ps[-1] + 1]]]
            nm = names[len(which) + j]
            assert nm == f"o{ps[0]}"
            assert bcs[i][len(which) + j] == L.commit(r, rl, words, salts[nm][i]), (m, role, i, nm)


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
