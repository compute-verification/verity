"""stage.py's census types: `parse` of census:NAME (the census name verbatim) and of a CENSUS key's statics, and `main`
staging one through `stage_census` on a stub of `class_statement.UnitStager` (nothing is lowered or proved): the serving
rows' template the name resolves, its rows' words, the size ``full`` and ``full/D`` resolve to at the block's k_log (never
under one block's units), and stage.json's record. By default: the Program and partition object the statement binds (the
verifier's copies beside it), the stager's ``bind`` and unit indices, and a ``full`` statement's circuit rebound to the
Program of the count its k_log gives; with ``--unbound`` none of them. And the retired types: each refused by name, with
what stages in its place."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load
from verity.circuits import partition_object as PO
from verity.circuits.codec import program_digest
from verity.circuits.types import Array, Value

SILU = "SiluMul_v2{I=256}"


@pytest.fixture(scope="module")
def stage():
    with pytest.MonkeyPatch.context() as mp:
        yield load(mp, "stage.py")


def test_parse(stage):
    name, cen, S = stage.parse(f"census:{SILU}")
    assert (name, S, cen.name(S)) == ("census", {}, SILU)
    assert stage.parse("census:Fam")[1].name({}) == "Fam"
    name, cen, S = stage.parse("gq:K=2048")
    assert (name, S, cen.name(S)) == ("gq", {"K": 2048, "G": 128}, "Fp8GroupQuantUe8m0_v1{K=2048,G=128}")
    _, cen, S = stage.parse("mla-split-part")
    assert cen.name(S) == ('SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
                           'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}}')
    for bad in ("census:", "census:{A=1}", "census:Fam{A=1"):
        with pytest.raises(SystemExit, match="census:NAME takes a census name"):
            stage.parse(bad)
    with pytest.raises(SystemExit, match="gq has no static 'X'"):
        stage.parse("gq:X=1")


def test_census_leaves_bits(stage):
    """1-bit leaves (each of `EmbeddingRead_v1`'s parameters) come as seeded uniform bits, sixteen a u16 word, leaf j the low
    bit of word j / 16 on, as `unit_ports` concatenates a row's leaves; a count off whole words is refused."""
    ty = Array(4, Array(16, Value(1)))
    got = stage.census_leaves(np.random.default_rng(7), ty, 3)
    assert got.dtype == np.dtype("<u2") and got.shape == (3, 4)
    assert (got == stage.census_leaves(np.random.default_rng(7), ty, 3)).all()
    bits = np.unpackbits(got.view(np.uint8), axis=1, bitorder="little")
    assert (bits == np.random.default_rng(7).integers(0, 2, (3, 64), dtype=np.uint8)).all()
    with pytest.raises(SystemExit, match=r"24 leaves of \[1\] bits"):
        stage.census_leaves(np.random.default_rng(7), Array(24, Value(1)), 3)


class Stager:
    """`UnitStager`'s face to `stage_census`: its META (k_log 32 and 4 units a block, so a statement holds 32 instances),
    the rows it is handed, and the files a staging writes."""

    seen: dict = {}

    def __init__(self, d, name, in_rows, out_rows, circuit, **stmt):
        self.meta = {"k_log": 32, "vus_per_block": 4}
        self.low = SimpleNamespace(layout=SimpleNamespace(counts={"ands": 5, "rows": 6, "nnz_a": 7, "nnz_b": 8}))
        self.pin, self.class_sha512 = "p" * 128, "c" * 128
        Stager.seen.update(definition=d.id, name=name, stmt=stmt, circuit=circuit)

    def stage(self, rows, out, salt_key):
        Stager.seen["rows"] = {r: v.shape for r, v in rows.items()}
        for f in ("circuit.txt", "inst.bin", "pub.bin"):
            (out / f).write_bytes(b"xy")
        return {"meta": {"k_log": 32, "vus_per_block": 4}, "ands": 5, "unit_nnz": 9}


@pytest.mark.parametrize(("spec", "n", "want"), [(f"census:{SILU}", "full", 32), (f"census:{SILU}", "full/4", 8),
                                                 (f"census:{SILU}", "full/64", 4), ("gq:K=2048", "full/2", 16),
                                                 (f"census:{SILU}", "3", 3)])
def test_main_unbound(stage, monkeypatch, tmp_path, capsys, spec, n, want):
    """--unbound: the statement under the provisional unit cover, its stager given no ``bind`` and nothing written beside it."""
    monkeypatch.setattr(stage.CS, "UnitStager", Stager)
    monkeypatch.setattr(sys, "argv", ["stage.py", spec, n, str(tmp_path), "--unbound"])
    stage.main()
    rec = json.loads((tmp_path / "stage.json").read_text())
    assert rec == json.loads(capsys.readouterr().out)
    census = stage.parse(spec)[1].name(stage.parse(spec)[2])
    assert (rec["census"], rec["definition"], rec["n"]) == (census, census, want)
    assert (tmp_path / "pin.txt").read_text() == "p" * 128 and rec["class_sha512"] == "c" * 128
    assert rec["nets"] == [{"net": "unit", "ands": 5, "rows": 6, "nnz_a": 7, "nnz_b": 8}]
    assert (rec["k_log"], rec["vus_per_block"], rec["circuit_bytes"], rec["inst_bytes"]) == (32, 4, 2, 2)
    assert Stager.seen["circuit"] == tmp_path / "circuit.txt" and Stager.seen["stmt"] == rec["statement"]
    assert Stager.seen["rows"] == {r: (want, b // 16) for r, _s, b, _ps in rec["in_rows"]}
    assert "bind" not in rec and not (tmp_path / "partition.json").exists() and not (tmp_path / "program.json").exists()


def circuit_text(bind: dict) -> bytes:
    """A circuit file's head as `circuit.compose` writes a binding: META, the second line, its two values."""
    meta = {"k_log": 32, "partition": {"digest": bind["partition"], "rule": "verity/partition/v1"},
            "program_sha512": bind["program"]}
    return b"verity/flock-circuit\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")).encode() + b"\ngates\n"


class BoundStager(Stager):
    """`UnitStager` given ``bind``: circuit.txt written once, as it is composed, bound to it; a round takes its units'
    indices and writes the public file's head (the circuit's pin as the stager holds it, the indices)."""

    def __init__(self, d, name, in_rows, out_rows, circuit, bind, **stmt):
        super().__init__(d, name, in_rows, out_rows, circuit, **stmt)
        Stager.seen["bind"] = dict(bind)
        circuit.parent.mkdir(parents=True, exist_ok=True)
        circuit.write_bytes(circuit_text(bind))
        self.pin = hashlib.sha512(circuit.read_bytes()).hexdigest()

    def stage(self, rows, out, salt_key, indices):
        Stager.seen.update(rows={r: v.shape for r, v in rows.items()}, indices=list(indices))
        (out / "inst.bin").write_bytes(b"xy")
        (out / "pub.bin").write_bytes(json.dumps({"circuit_sha512": self.pin, "units": {"indices": list(indices)}}).encode() + b"\n")
        return {"meta": {"k_log": 32, "vus_per_block": 4}, "ands": 5, "unit_nnz": 9}


@pytest.mark.parametrize(("n", "want", "composed"), [("3", 3, "own"), ("full/4", 8, "unbound"), ("8", 8, "own")])
def test_main_bound(stage, monkeypatch, tmp_path, capsys, n, want, composed):
    """By default the statement binds the population Program of its ``want`` instances and that Program's partition under
    `Q_template_instances` v0, both written beside it. A count binds as the circuit is composed; ``full/4`` is composed
    bound to `UNBOUND` and rebound, to the file a count of 8 composes."""
    monkeypatch.setattr(stage.CS, "UnitStager", BoundStager)
    monkeypatch.setattr(sys, "argv", ["stage.py", f"census:{SILU}", n, str(tmp_path)])
    stage.main()
    rec = json.loads(capsys.readouterr().out)
    part, desc = json.loads((tmp_path / "partition.json").read_text()), json.loads((tmp_path / "program.json").read_text())
    digests = {"partition": PO.digest(part), "program": part["program"]}
    assert PO.validate(part).ok and part["query"] == PO.template_instances_query([SILU])
    assert desc["program_fn"] == f'TemplatePopulations_v1{{templates=["{SILU}"],counts=[{want}]}}'
    assert program_digest(desc) == part["program"]
    assert rec["bind"] == {**digests, "query": part["query"], "population": want, "indices": [0, want],
                           "program_bytes": (tmp_path / "program.json").stat().st_size}
    assert Stager.seen["bind"] == (digests if composed == "own" else stage.UNBOUND)
    assert (rec["n"], Stager.seen["indices"]) == (want, list(range(want)))
    assert (tmp_path / "circuit.txt").read_bytes() == circuit_text(digests)
    assert rec["circuit_sha512"] == (tmp_path / "pin.txt").read_text() == hashlib.sha512(circuit_text(digests)).hexdigest()


def test_rebind_refuses(stage, tmp_path):
    """`rebind` rewrites only a file whose META holds the old binding's two values once each, and nowhere after META."""
    f, new = tmp_path / "circuit.txt", {"partition": "1" * 128, "program": "2" * 128}
    text = circuit_text(stage.UNBOUND)
    f.write_bytes(text)
    assert stage.rebind(f, stage.UNBOUND, new) == hashlib.sha512(circuit_text(new)).hexdigest() and f.read_bytes() == circuit_text(new)
    with pytest.raises(SystemExit, match="META does not hold the binding"):
        stage.rebind(f, stage.UNBOUND, new)
    part = stage.UNBOUND["partition"].encode()
    for bad, why in ((text.replace(b'"program_sha512"', b'"x":"' + part + b'","program_sha512"'), "META does not hold the binding"),
                     (text.replace(b"\nMETA ", b"\nHEAD "), "META does not hold the binding"),
                     (text + part + b"\n", "outside META's two values")):
        f.write_bytes(bad)
        with pytest.raises(SystemExit, match=why):
            stage.rebind(f, stage.UNBOUND, new)
        assert f.read_bytes() == bad


@pytest.mark.parametrize(("argv", "why"), [([f"census:{SILU}", "full/0"], "invalid literal"),
                                           (["s8-coord", "full"], "is for a CENSUS type"),
                                           (["s8-coord", "2", "--bind"], "usage: stage.py"),
                                           (["moe-w13", "2"], "moe-w13 is retired"),
                                           (["census:SiluMul_v2", "2"], "lacks the static")])
def test_main_refuses(stage, monkeypatch, tmp_path, argv, why):
    monkeypatch.setattr(stage.CS, "UnitStager", Stager)
    monkeypatch.setattr(sys, "argv", ["stage.py", *argv, str(tmp_path)])
    with pytest.raises((SystemExit, ValueError), match=why):
        stage.main()
    assert not (tmp_path / "stage.json").exists()


#: the 18 types that were `Q_word` classes of a word Definition
RETIRED = {"moe-gemmw", "moe-chunk", "moe-w13", "dsa-krow", "dsa-logits", "moe-sum", "silu-mul", "rope", "dsa-topk", "s8-linear",
           "router", "dsa-q", "rmsnorm-cuda", "sparse-mla-head", "token-select", "moe-w2", "rmsnorm-triton", "silu-quant"}


def test_retired(stage):
    """Each retired type is refused by name, and the refusal names what stages in its place (all of it, or one of two for
    `RETIRED_EITHER`), or says that nothing does; each
    replacement is a type `parse` takes, and a census replacement is a serving rows' template whose statement `verify --zk`
    takes (`chunk_rows.zk_refusal`). The census's own head block at 64 keys reads in lookup slots, which `verify --zk`
    refuses, and its refusal names the Boolean version `RETIRED` stages instead."""
    from verity_vllm.pipeline import chunk_rows as CR
    from verity_vllm.pipeline.serving_rows import template

    assert set(stage.RETIRED) == RETIRED
    for name, (instead, why) in stage.RETIRED.items():
        with pytest.raises(SystemExit) as stop:
            stage.parse(f"{name}:K=2")
        said = str(stop.value)
        assert said.startswith(f"stage.py: {name} is retired") and why in said
        assert all(r in said for r in instead) and (bool(instead) != ("nothing stages in its place" in said)), said
        assert (" or ".join(instead) in said) == (name in stage.RETIRED_EITHER) or len(instead) < 2, said
        for r in instead:
            got, cen, S = stage.parse(r)
            assert got not in stage.RETIRED
            if isinstance(cen, stage.Census):
                assert CR.zk_refusal(template(cen.name(S))) is None, r
    assert not {name for name, (instead, _) in stage.RETIRED.items() if not instead} ^ {"moe-w2", "rmsnorm-triton", "silu-quant"}
    block = stage.RETIRED["sparse-mla-head"][0][0].removeprefix("census:")
    census = block.replace("_v2", "_v1")
    assert CR.zk_refusal(template(census)).endswith(f"name its Boolean version {block}")
