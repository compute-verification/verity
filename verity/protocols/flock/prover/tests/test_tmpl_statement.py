"""The templated statement (``flock-circuit/tmpl``, `circuit.compose`'s ``layout``): the flat statement with its unit given as
the unit's ``flock-ir-tmpl/v2`` template, in its own section right after META (`circuit.template_section`), META ``layout``
the mode, and no ``CIRCUIT unit`` section, which the Lean verifier reads with ``HmOut.parseTmpl``.

`tmpl.layout`, the mirror of ``Tmpl.layoutM`` that puts the unit's range at its committed slot, is held to
``Flock.Test.TmplLayout``'s vectors and to a Definition two call paths reach, which is where the modes differ. The Boolean
toy's templated statement unrolls to its flat unit and differs from the flat statement's META in the templated keys alone.
The Lean verifier derives from it the class `circuit.class_digest` computes and the regions, Δ and partition check it
derives from the flat statement. It is outside the verifier's ``Tags.all``, in its own list (``Tags.templated``), and
the Lean test skips unless the verifier knows it, as TMPL1's does (``FLOCK_VERIFY_TMPL``: its flock-verify)."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_class_statement_unit_stager import _toy
from test_struct_lower import TOY_READS, toy_reads  # noqa: F401
from verity.circuits.defs import bind
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_config.settings import of
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import struct_lower as S
from verity_flock import tmpl as T

LEAN = Path(__file__).resolve().parents[2] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
MODE = "iv/hybrid/4"

#: ``Flock.Test.Tmpl``'s unit: two copies of a pair (each two copies of a two-input AND and their sum) at 128, bound to input
#: bits that step by one, its outputs at 256 and 257 and a row at 300
HEADER = "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 2 1 1 2"
AND2 = ["DEF 1 2 1 0", "0 1 2 1 1"]
PAIR = ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 2 3 0 1 0"]
ROWS = ["0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399", "257 1 133 1 399", "300 3 0 1 0 1 400"]


def _tmpl(*calls: str, rows: list[str] = ROWS) -> T.Tmpl:
    return T.parse("\n".join([HEADER, *AND2, *PAIR, f"DEF 400 0 {len(rows)} {len(calls)}", *rows, *calls]) + "\n")


def _map(t: T.Tmpl, mode: str) -> tuple[int, list[int]]:
    lays = T.layout(t, mode)
    at = T.pos_of(t, lays)
    return lays[t.root].size, [at(x) for x in range(t.defs[t.root].cols)]


def _sections(text: str) -> dict[str, list[str]]:
    lines, out, at = text.split("\n"), {}, 2
    if lines[at].startswith(f"{FC.TMPL_SECTION} "):
        at += 1 + int(lines[at].split(" ")[1])
    while at < len(lines) and lines[at]:
        _c, name, n = lines[at].split(" ")
        out[name] = lines[at + 1:at + 1 + int(n)]
        at += 1 + int(n)
    return out


@pytest.mark.parametrize("mode", ["iv", "iv/strict", "iv/hybrid", MODE])
def test_layout_places_each_column_where_the_lean_layout_does(mode):
    """``Flock.Test.TmplLayout``'s vectors under every mode: the pair's two copies take two lanes at 128 (columns 128 to 133
    at 128, 130, 132, 129, 131, 133), the pair is laid out as 3 columns aligned to 2, one copy keeps its columns, two pairs
    at 250 push the output group to 384 and the constant to 527; and the two layouts it refuses, by name."""
    moved = {129: 130, 130: 132, 131: 129, 132: 131}
    t = _tmpl("CALL 1 2 128 3 1 2 0 1 1 0")
    assert _map(t, mode) == (400, [moved.get(x, x) for x in range(400)])
    assert (T.layout(t, mode)[1].size, T.layout(t, mode)[1].align) == (3, 2)
    assert _map(_tmpl("CALL 1 1 128 3 1 2 0 1 1 0"), mode) == (400, list(range(400)))
    size, at = _map(_tmpl("CALL 1 2 250 3 1 2 0 1 1 0"), mode)
    assert (size, at[256], at[399]) == (528, 384, 527)
    with pytest.raises(ValueError, match="output group at word 2 does not keep its columns at a multiple of its width"):
        T.layout(_tmpl("CALL 1 2 253 3 1 2 0 1 1 0", rows=[ROWS[i] for i in (0, 1, 2, 5)]), mode)
    with pytest.raises(ValueError, match="a call in the input rows or an output group to a definition laid out apart"):
        T.layout(_tmpl("CALL 1 1 0 3 1 1 399 0 2 0", rows=ROWS[3:]), mode)


def test_a_definition_two_call_paths_reach_is_placed_at_its_modes_alignment():
    """The pair called again, once, at 141: two call paths reach it, so strict, and hybrid at T = 0, place its copy at its size's
    power of two (144), lenient and hybrid at T >= 1 at its own alignment (142); the root's size is 528 in each."""
    t = _tmpl("CALL 1 2 128 3 1 2 0 1 1 0", "CALL 1 1 141 3 1 2 0 1 1 0")
    for mode, nb in (("iv", 142), ("iv/strict", 144), ("iv/hybrid", 144), ("iv/hybrid/1", 142), (MODE, 142)):
        size, at = _map(t, mode)
        assert size == 528 and at[141:144] == [nb, nb + 1, nb + 2], mode


def test_a_layout_mode_is_one_the_verifier_reads():
    assert [T.mode_of(m) for m in ("iv", "iv/strict", "iv/hybrid", MODE)] == [("lenient", 0), ("strict", 0), ("hybrid", 0), ("hybrid", 4)]
    for bad in ("iv/hybrid/", "iv/hybrid/x", "interleaved", "iv/strict/4"):
        with pytest.raises(ValueError, match="is not a layout mode"):
            T.mode_of(bad)


@pytest.mark.slow
def test_the_templated_statement_is_the_flat_one_with_its_unit_as_its_template(monkeypatch):
    """The toy's templated statement: format and statement the templated ones, META the flat statement's but for ``format``,
    ``statement`` and ``layout``; its template section the unit's template, which `tmpl.unroll` takes to the flat file's
    unit section line for line; its other sections the flat file's but the unit's; its unit range's slot the laid-out unit's
    (a layout one column past the flat unit's slot moves ``unit_log`` and the range to the next); and a size class's
    ``definition`` is refused."""
    args, _rows, _kw = _toy(0)
    flat, tm = CS.UnitStager(*args), CS.UnitStager(*args, layout=MODE)
    assert tm.text.split("\n", 1)[0] == f"{FC.TMPL_FORMAT} {args[1]}" and flat.text.split("\n", 1)[0] == f"{FC.FORMAT} {args[1]}"
    assert (tm.meta["format"], tm.meta["statement"], tm.meta["layout"]) == (FC.TMPL_FORMAT, FC.TMPL_STATEMENT, MODE)
    assert sorted(k for k in set(flat.meta) | set(tm.meta) if flat.meta.get(k) != tm.meta.get(k)) == ["format", "layout", "statement"]
    sections = _sections(flat.text)
    assert _sections(tm.text) == {k: v for k, v in sections.items() if k != "unit"}
    t = T.parse(FC.template_of(tm.text))
    assert T.flat_lines(T.unroll(t)) == sections["unit"]
    size = T.layout(t, MODE)[t.root].size
    unit = [r for r in tm.meta["ranges"] if r["net"] == "unit"]
    assert tm.meta["unit_log"] == FC._slot_log(size) and [r["slot_log"] for r in unit] == [FC._slot_log(size)]
    assert tm.class_sha512 == FC.class_digest(tm.text) != flat.class_sha512

    layout = T.layout
    past = 1 << FC._slot_log(size)

    def wider(t, mode):
        lays = layout(t, mode)
        return [*lays[:t.root], dataclasses.replace(lays[t.root], size=past + 1), *lays[t.root + 1:]]

    monkeypatch.setattr(T, "layout", wider)
    moved = CS.UnitStager(*args, layout=MODE).meta
    assert moved["unit_log"] == FC._slot_log(size) + 1
    assert [r["slot_log"] for r in moved["ranges"] if r["net"] == "unit"] == [FC._slot_log(size) + 1]
    monkeypatch.setattr(T, "layout", layout)
    with pytest.raises(ValueError, match="a unit class \\(META definition\\) has no templated statement"):
        FC.compose(tm.low, tm.sub, port_bits=[b for _r, _s, b, _ps in args[2]], v2_out_rows=(("o0",),), definition="d", layout=MODE)


def test_the_pin_and_class_hash_the_file_a_chunk_at_a_time(monkeypatch):
    """`circuit.digest` and `circuit.class_digest`, which hash a templated file (its template section most of it) in chunks
    rather than whole copies, are SHA-512 of the file and of the file with META's binding keys removed and re-serialized,
    across chunk boundaries anywhere, a non-ASCII section included; `circuit.template_of` reads the template section back."""
    import hashlib

    meta = {"format": FC.TMPL_FORMAT, "layout": MODE, "partition": {"digest": "p"}, "program_sha512": "q", "z": "é"}
    lines = "".join(f"{i} 1 {i} 1 {i + 1}\n" for i in range(50))
    rest = f"{FC.template_section(lines)}CIRCUIT hm96 1\nÿ 0\n"
    text = f"{FC.TMPL_FORMAT} u\nMETA {json.dumps(meta)}\n{rest}"
    kept = {k: v for k, v in meta.items() if k not in FC.BINDING_KEYS}
    want = (hashlib.sha512(text.encode()).hexdigest(),
            hashlib.sha512((f"{FC.TMPL_FORMAT} u\nMETA {json.dumps(kept, sort_keys=True, separators=(',', ':'))}\n"
                            f"{rest}").encode()).hexdigest())
    assert FC.meta_of(text) == meta and FC.template_of(text) == lines and rest.startswith(f"{FC.TMPL_SECTION} 50\n")
    for chunk in (1, 7, 64, 1 << 24):
        monkeypatch.setattr(FC, "_HASH_CHUNK", chunk)
        assert (FC.digest(text), FC.class_digest(text)) == want


@pytest.mark.slow
def test_a_structural_stager_writes_the_templated_statement_of_struct_lowers_template(tmp_path):
    """`UnitStager`'s ``structural`` on the toy: its template section is `struct_lower`'s template on the stager's ports (the
    flat unit never built), ordered, its unit range that template's slot; the outputs a round stages and META ``dummy`` are
    the flat stager's, from the template (`TmplLowering.evaluate`); it is the emitted templated statement but for the
    template section, the unit's slot and META's pins that read them; and a structural stager without a layout is
    refused."""
    args, rows, _kw = _toy(0)
    flat, tm = CS.UnitStager(*args, layout=MODE), CS.UnitStager(*args, layout=MODE, structural=True)
    assert tm.low.circuit is None and isinstance(tm.low, CS.TmplLowering)
    t = T.parse(FC.template_of(tm.text))
    assert T.write(t) == T.write(tm.low.template) and T.ordered(t)["ok"]
    assert tm.meta["unit_log"] == FC._slot_log(T.layout(t, MODE)[t.root].size)
    assert tm.meta["dummy"] == flat.meta["dummy"]
    diff = sorted(k for k in set(flat.meta) | set(tm.meta) if flat.meta.get(k) != tm.meta.get(k))
    assert set(diff) <= {"unit_sha256", "partition", "unit_log", "k_log", "ranges", "vus_per_block"}
    assert FC.template_of(tm.text) != FC.template_of(flat.text)
    a, b = flat.stage(rows, tmp_path / "flat", b"k" * 32), tm.stage(rows, tmp_path / "struct", b"k" * 32)
    assert all(np.array_equal(a["outs"][o], b["outs"][o]) for o in a["outs"])
    assert (tmp_path / "struct" / "circuit.txt").read_text() == tm.text
    with pytest.raises(ValueError, match="a structural unit is written as its template: name its layout"):
        CS.UnitStager(*args, structural=True)


def _structural_is_flat(args: tuple, rows: dict, out: Path) -> CS.UnitStager:
    """The structural stager's statement of ``args`` against the flat stager's at the same layout: its template ordered and
    computing the flat lowering's bits on random inputs (`struct_lower.check`), META the flat one's but for the template and
    what reads it, and the outputs a round stages on ``rows`` the flat ones. circuit-check's structural check
    (``--structural``) holds a template to the flat proof block gate for gate."""
    flat, tm = CS.UnitStager(*args, layout=MODE), CS.UnitStager(*args, layout=MODE, structural=True)
    assert isinstance(tm.low, CS.TmplLowering) and T.ordered(tm.low.template)["ok"]
    assert tm.meta["dummy"] == flat.meta["dummy"]
    diff = sorted(k for k in set(flat.meta) | set(tm.meta) if flat.meta.get(k) != tm.meta.get(k))
    assert set(diff) <= {"template", "unit_sha256", "partition", "unit_log", "k_log", "ranges", "vus_per_block"}
    a, b = flat.stage(rows, out / "flat", b"k" * 32), tm.stage(rows, out / "struct", b"k" * 32)
    assert all(np.array_equal(a["outs"][o], b["outs"][o]) for o in a["outs"])
    assert S.check(args[0])["mismatched_bits"] == 0
    return tm


@pytest.mark.slow
@pytest.mark.usefixtures("toy_reads")
def test_a_structural_stager_lays_a_table_read_out_as_the_flat_stager_does(tmp_path):
    """`test_struct_lower`'s toy, whose unit reads a library table three times, as a structural statement (beside a row
    spanning SHA-512 blocks, as a composed circuit needs one): the flat statement's by `_structural_is_flat`, with the read's
    rows once in the template, in the table-reading primitive's one definition."""
    from verity.circuits.boolean import BIT
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.types import Array, Value

    unit = bind(CompositeDefinition("TmplToyReads", 1, (), lambda _S: ((("x", Value(8)), ("y", Value(8)), ("w", Array(2048, BIT))),
                                                                       Array(4, Value(8))),
                                    lambda B, _S, x, y, w: B.call(TOY_READS, x, y), register=False))
    args = (unit, "unit/struct-reads", [("xy", V2, 16, ["x", "y"]), ("w", V2, 2048, ["w"])], [(V2, (0,))])
    rng = np.random.default_rng(5)
    rows = {"xy": rng.integers(0, 1 << 16, (5, 1), dtype=np.uint16), "w": rng.integers(0, 1 << 16, (5, 128), dtype=np.uint16)}
    t = _structural_is_flat(args, rows, tmp_path).low.template
    refs = [sum(len(a) + len(b) for _c, a, b in d.rows) for d in t.defs]
    read = max((e for e in range(len(t.defs)) if e != t.root), key=refs.__getitem__)
    assert sum(c.count for d in t.defs for c in d.calls if c.e == read) == 3


#: the served types struct_lower lays out at small statics: FlashInfer's FP8 linear (two rcp reads a group), SparseMla's
#: combine (ex2 and rcp), the fused RMSNorm (rsq) and the multimem all-reduce (none); each has a row of at least 2,048 bits,
#: since a statement needs a row whose SHA-512 compresses a block before its last
def _served(name: str):
    from verity.definitions.registry import cuda_rmsnorm, deepgemm, hopper, sparse_mla_split, tp_allreduce

    return {"FlashInferFp8Linear": lambda: bind(deepgemm.FlashInferFp8Linear, K=128, N=1),
            "SparseMlaCombine": lambda: bind(sparse_mla_split.SparseMlaCombine, NSPLIT=2, DV=32, INV=hopper.Fa3InvSum),
            "RMSNormFusedCudaVector8": lambda: bind(cuda_rmsnorm.RMSNormFusedCudaVector8, N=64, EPS=1e-5, BLOCK=8),
            "AllReduceMultimem": lambda: bind(tp_allreduce.AllReduceMultimem, WORLD=2, SIZE=64, TREES=((0, 1), (0, 1)),
                                              FTZ=False)}[name]()


SERVED = ("FlashInferFp8Linear", "SparseMlaCombine", "RMSNormFusedCudaVector8", "AllReduceMultimem")


@pytest.mark.skipif(not of(__file__).VERITY_FLOCK_STRUCT_TYPES,
                    reason="stages each served type flat and structural, the MUFU reads' 2^23- and 2^24-word tables inline: "
                           "about ten minutes and 40 GB; set VERITY_FLOCK_STRUCT_TYPES=1")
@pytest.mark.parametrize("name", SERVED)
def test_a_served_type_reading_tables_is_staged_structurally_as_its_flat_statement(name, tmp_path):
    """Each served type in `SERVED`, at small statics (`_served`): its structural statement is its flat one
    (`_structural_is_flat`), a parameter a row of its u16 words, on finite bf16 rows and random ones."""
    fn = _served(name)
    params, ret = fn.signature()
    in_rows = [(nm, V2, sum(t.leaf_widths()), [nm]) for nm, t in params]
    assert all(b % 16 == 0 for _r, _s, b, _p in in_rows)
    outs = [(V2, (j,)) for j in range(len(ret.fields) if hasattr(ret, "fields") else 1)]
    rng = np.random.default_rng(11)
    rows = {r: np.concatenate([rng.integers(0, 1 << 16, (2, b // 16), dtype=np.uint16),
                               (rng.standard_normal((3, b // 16)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)])
            for r, _s, b, _p in in_rows}
    _structural_is_flat((fn, f"unit/struct-{name}", in_rows, outs), rows, tmp_path)


#: the auditor's host runs the lowering on numpy and Flock's Python alone
NO_TORCH = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] in ("torch", "triton", "verity_numerical"):
            raise ImportError(f"{name}: the auditor's lowering runs without it")
sys.meta_path.insert(0, _Block())
"""
WRITE = """
from verity.circuits.boolean import BIT
from verity.circuits.boolean.forms import add
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import bind
from verity.circuits.types import Array
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
d = bind(definition("TmplNoTorch", 1, lambda C, a, b, w: add(C, a, b, width=16, carry_out=True) + [C.AND(w[0], w[2047])],
                    (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(18, BIT), register=False))
CS.UnitStager(d, "unit/tmpl-no-torch", [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])], [(V2, (0,))], circuit=OUT,
              layout=MODE)
"""


@pytest.mark.slow
def test_the_templated_statement_is_written_without_torch(tmp_path):
    """`UnitStager`'s templated statement, the file the auditor's lowering writes on a host with numpy and no torch, is
    written with torch (and the numerical benchmarks) unimportable, byte for byte the file it writes with them."""
    import sys
    prog = NO_TORCH + f"from pathlib import Path\nOUT, MODE = Path({str(tmp_path / 'a.txt')!r}), {MODE!r}\n" + WRITE
    p = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    exec(WRITE, {"OUT": tmp_path / "b.txt", "MODE": MODE})
    a = (tmp_path / "a.txt").read_text()
    assert a.startswith(f"{FC.TMPL_FORMAT} unit/tmpl-no-torch\n") and a == (tmp_path / "b.txt").read_text()


def test_a_unit_past_k_max_is_refused_before_its_template_is_written(monkeypatch, tmp_path):
    """A templated unit whose block passes ``K_MAX`` is refused before compose writes its template's text, which for a
    served lane is gigabytes (the K=14336 lane's 16.5M lines took a refused compose from 12.8 to 15.1 GiB). At 19, below
    every block's 2^20 and above the unit's own slot, the refusal is compose's, not `UnitStager`'s before lowering."""
    def written(_t):
        raise AssertionError("the template was written before the block was checked")

    monkeypatch.setattr(FC, "K_MAX", 19)
    monkeypatch.setattr(T, "write", written)
    with pytest.raises(ValueError, match="unit/tmpl-no-torch: an instance needs a block of 2\\^\\d+ bits \\(at most 2\\^19\\)"):
        exec(WRITE, {"OUT": tmp_path / "c.txt", "MODE": MODE})


def _verifier() -> Path:
    if b := os.environ.get("FLOCK_VERIFY_TMPL"):
        return Path(b)
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return BIN


@pytest.fixture(scope="module")
def verifier() -> Path:
    b = _verifier()
    p = subprocess.run([str(b), "statement", "--statement", FC.TMPL_STATEMENT, "--circuit", os.devnull, "--public", os.devnull],
                       capture_output=True, text=True)
    if "unknown statement" in p.stderr:
        pytest.skip(f"{b} does not know {FC.TMPL_STATEMENT} (not in its Tags.templated; FLOCK_VERIFY_TMPL: TMPL1's flock-verify)")
    return b


def _statement(b: Path, st, rows, out: Path, statement: str, flags: list[str], indices=None) -> dict:
    st.stage(rows, out, b"k" * 32, **({"indices": indices} if indices is not None else {}))
    p = subprocess.run([str(b), "statement", "--statement", statement, "--circuit", str(out / "circuit.txt"), "--public",
                        str(out / "pub.bin"), *flags], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.parametrize("bound", [False, True], ids=["unbound", "bound"])
def test_lean_reads_the_templated_statement_as_the_flat_one(verifier, bound, tmp_path):
    """The Lean verifier's statement of the toy, templated and flat, unbound and bound to a template query's partition (its
    units derived from the verifier's own copy of the Program): the templated statement is the templated tags', its class
    `circuit.class_digest`'s (``HmRow.classOf`` of its canonical META and the rest), and its instance count, regions, Δ,
    partition and partition check the flat statement's."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity.circuits.codec import canonical_json, encode_program

    args, rows, _kw = _toy(0)
    flags, kw, indices = [], {}, None
    if bound:
        from test_class_statement_unit_stager import TOY

        program = P.mixed_population_program([(TOY, 3), (TOY, 5)])
        part = P.partition(program, P.template_instances_query([TOY]))
        (tmp_path / "program.json").write_bytes(canonical_json(encode_program(program)))
        (tmp_path / "partition.json").write_text(json.dumps(part))
        flags = ["--program", str(tmp_path / "program.json"), "--partition", str(tmp_path / "partition.json")]
        kw, indices = {"bind": {"partition": P.digest(part), "program": part["program"]}}, range(3, 8)
    flat, tm = CS.UnitStager(*args, **kw), CS.UnitStager(*args, layout=MODE, **kw)
    a = _statement(verifier, flat, rows, tmp_path / "flat", FC.STATEMENT, flags, indices)
    b = _statement(verifier, tm, rows, tmp_path / "tmpl", FC.TMPL_STATEMENT, flags, indices)
    assert b["statement"] == FC.TMPL_STATEMENT and a["statement"] == FC.STATEMENT
    assert b["class_sha512"] == tm.class_sha512 and a["class_sha512"] == flat.class_sha512
    assert b["circuit_sha512"] == tm.pin
    same = ("m", "regions", "delta", "partition", "partition_check", "program_sha512")
    assert {k: b[k] for k in same} == {k: a[k] for k in same}
    if bound:
        assert b["partition_check"] == a["partition_check"] is not None
