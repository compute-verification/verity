"""The templated statement (``flock-circuit/tmpl``, `circuit.compose`'s ``layout``): the flat statement with its unit given as
the unit's ``flock-ir-tmpl/v2`` template, META ``template`` its lines and ``layout`` the mode, and no ``CIRCUIT unit``
section, which the Lean verifier reads with ``HmOut.parseTmpl``.

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
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import tmpl as T

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
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
    ``statement``, ``template`` (the unit's template, which `tmpl.unroll` takes to the flat file's unit section line for line)
    and ``layout``; its sections the flat file's but the unit's; its unit range's slot the laid-out unit's (a layout one
    column past the flat unit's slot moves ``unit_log`` and the range to the next); and a unit class's ``definition`` is
    refused."""
    args, _rows, _kw = _toy(0)
    flat, tm = CS.UnitStager(*args), CS.UnitStager(*args, layout=MODE)
    assert tm.text.split("\n", 1)[0] == f"{FC.TMPL_FORMAT} {args[1]}" and flat.text.split("\n", 1)[0] == f"{FC.FORMAT} {args[1]}"
    assert (tm.meta["format"], tm.meta["statement"], tm.meta["layout"]) == (FC.TMPL_FORMAT, FC.TMPL_STATEMENT, MODE)
    assert sorted(k for k in set(flat.meta) | set(tm.meta) if flat.meta.get(k) != tm.meta.get(k)) == ["format", "layout", "statement", "template"]
    sections = _sections(flat.text)
    assert _sections(tm.text) == {k: v for k, v in sections.items() if k != "unit"}
    t = T.parse("\n".join(tm.meta["template"]) + "\n")
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
    """`circuit.digest` and `circuit.class_digest`, which hash a templated file's META (most of the file) in chunks rather
    than whole copies, are SHA-512 of the file and of the file with META's binding keys removed and re-serialized, across
    chunk boundaries anywhere, a non-ASCII section included."""
    import hashlib

    meta = {"format": FC.TMPL_FORMAT, "template": [f"{i} 1 {i} 1 {i + 1}" for i in range(50)], "partition": {"digest": "p"},
            "program_sha512": "q", "z": "é"}
    text = f"{FC.TMPL_FORMAT} u\nMETA {json.dumps(meta)}\nCIRCUIT hm96 1\nÿ 0\n"
    kept = {k: v for k, v in meta.items() if k not in FC.BINDING_KEYS}
    want = (hashlib.sha512(text.encode()).hexdigest(),
            hashlib.sha512((f"{FC.TMPL_FORMAT} u\nMETA {json.dumps(kept, sort_keys=True, separators=(',', ':'))}\n"
                            "CIRCUIT hm96 1\nÿ 0\n").encode()).hexdigest())
    assert FC.meta_of(text) == meta
    for chunk in (1, 7, 64, 1 << 24):
        monkeypatch.setattr(FC, "_HASH_CHUNK", chunk)
        assert (FC.digest(text), FC.class_digest(text)) == want


@pytest.mark.slow
def test_a_structural_stager_writes_the_templated_statement_of_struct_lowers_template(tmp_path):
    """`UnitStager`'s ``structural`` on the toy: META ``template`` is `struct_lower`'s template on the stager's ports (the
    flat unit never built), ordered, its unit range that template's slot; the outputs a round stages and META ``dummy`` are
    the flat stager's, from the template (`TmplLowering.evaluate`); META is the emitted templated statement's but for the
    template, the unit's slot and the pins that read them; and a structural stager without a layout is refused."""
    args, rows, _kw = _toy(0)
    flat, tm = CS.UnitStager(*args, layout=MODE), CS.UnitStager(*args, layout=MODE, structural=True)
    assert tm.low.circuit is None and isinstance(tm.low, CS.TmplLowering)
    t = T.parse("\n".join(tm.meta["template"]) + "\n")
    assert T.write(t) == T.write(tm.low.template) and T.ordered(t)["ok"]
    assert tm.meta["unit_log"] == FC._slot_log(T.layout(t, MODE)[t.root].size)
    assert tm.meta["dummy"] == flat.meta["dummy"]
    diff = sorted(k for k in set(flat.meta) | set(tm.meta) if flat.meta.get(k) != tm.meta.get(k))
    assert set(diff) <= {"template", "unit_sha256", "partition", "unit_log", "k_log", "ranges", "vus_per_block"} and "template" in diff
    a, b = flat.stage(rows, tmp_path / "flat", b"k" * 32), tm.stage(rows, tmp_path / "struct", b"k" * 32)
    assert all(np.array_equal(a["outs"][o], b["outs"][o]) for o in a["outs"])
    assert (tmp_path / "struct" / "circuit.txt").read_text() == tm.text
    with pytest.raises(ValueError, match="a structural unit is written as its template: name its layout"):
        CS.UnitStager(*args, structural=True)


#: the auditor's host runs the lowering on numpy and C-Flock's Python alone
NO_TORCH = """
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] in ("torch", "triton", "verity_numerical"):
            raise ImportError(f"{name}: the auditor's lowering runs without it")
sys.meta_path.insert(0, _Block())
"""
WRITE = """
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.forms import add
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
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
    served lane is gigabytes (the K=14336 lane's 16.5M lines took a refused compose from 12.8 to 15.1 GiB)."""
    def written(_t):
        raise AssertionError("the template was written before the block was checked")

    monkeypatch.setattr(FC, "K_MAX", 10)
    monkeypatch.setattr(T, "write", written)
    with pytest.raises(ValueError, match="unit/tmpl-no-torch: an instance needs a block of 2\\^\\d+ bits \\(at most 2\\^10\\)"):
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
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits.codec import canonical_json, encode_program

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
