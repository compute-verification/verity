"""`compose_circuit.py --cache` and a5's own composers through it: a machine composes each template once, and a window's composes
can be made before its timed audit (`a5.warm`). A circuit composed bound to one partition and program, kept, and taken bound to
another (`restamp`) is byte for byte, with compose's line, the circuit composed bound to that one, its caller never composing,
whatever the reads' chunks split; taken bound to the same one it is the kept file, linked and never read, and a sampled proof
block's copy that changed is refused before a prover reads it (`a5.held`); a cache key moves with what the compose reads (the
template, the reads' layout, the Definition's program, the composer's inputs), any composing file, compose_circuit's own and
the composer's among them, any data file of the packages and a lowering flag, and with nothing else; a circuit whose reads are
in slots gets its MUFU tables beside it whether composed or restamped; and a kept circuit that changed, a binding not in the
file once, or a binding that is not two SHA-512 digests is refused, writing nothing. A toy unit Definition."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.forms import add
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402
import compose_circuit as CC  # noqa: E402

#: a + b over 16 bits with its carry, and w[0] & w[2047]: w's row spans SHA-512 blocks, as a composed circuit needs one row to
TOY = bind(definition("ComposeCacheToy", 1, lambda C, a, b, w: add(C, a, b, width=16, carry_out=True) + [C.AND(w[0], w[2047])],
                      (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(18, BIT), register=False))
ROWS = [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])]
B1 = {"partition": "a1" * 64, "program": "b2" * 64}
B2 = {"partition": "c3" * 64, "program": "d4" * 64}


def _compose(binding: dict, calls: list | None = None):
    """compose_circuit's ``compose`` for the toy: the circuit at its path composed bound to ``binding``, -> its line."""
    def compose(out: Path) -> dict:
        if calls is not None:
            calls.append(binding)
        st = CS.UnitStager(TOY, "unit/compose-cache-toy", ROWS, [(V2, (0,))], circuit=out, bind=binding)
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"],
                "meta": st.meta}
    return compose


def _own(calls: list | None = None, inputs: object = ("unit/compose-cache-toy", ROWS)):
    """A caller's composer for a5 (``(out, partition, program) -> line``, records_audit's form) that says what it composes from."""
    def compose(out: Path, partition: str, program: str) -> dict:
        if calls is not None:
            calls.append((partition, program))
        line = _compose({"partition": partition, "program": program})(Path(out))
        return {k: v for k, v in line.items() if k != "meta"}
    compose.inputs = inputs
    return compose


@pytest.mark.slow
@pytest.mark.parametrize("split", ["whole", "partition", "program"])
def test_a_kept_circuit_taken_with_another_binding_is_the_circuit_composed_bound_to_it(split, tmp_path, monkeypatch):
    """The first caller composes and keeps; a later one bound elsewhere gets the bytes and line a fresh compose bound there
    gives (another pin, the one class) without composing, and one bound as the first gets the first's line and the kept file
    itself, linked; with the reads' chunks ending inside the partition's value or the program's, the same."""
    cache, calls = tmp_path / "cache", []
    first = CC.cached(cache, "toy", tmp_path / "r0" / "circuit.txt", B1, _compose(B1, calls))
    at = CC.binding_at(tmp_path / "r0" / "circuit.txt", B1, first["pin"])
    if split != "whole":
        monkeypatch.setattr(CC, "_CHUNK", at[split] + 40)
        assert CC.binding_at(tmp_path / "r0" / "circuit.txt", B1, first["pin"]) == at
    later = CC.cached(cache, "toy", tmp_path / "r1" / "circuit.txt", B2, _compose(B2, calls))
    again = CC.cached(cache, "toy", tmp_path / "r2" / "circuit.txt", B1, _compose(B1, calls))
    assert calls == [B1]
    fresh = _compose(B2)(tmp_path / "fresh" / "circuit.txt")
    assert (tmp_path / "r1" / "circuit.txt").read_bytes() == (tmp_path / "fresh" / "circuit.txt").read_bytes()
    assert later == fresh and later["pin"] != first["pin"] and later["class_sha512"] == first["class_sha512"]
    assert again == first and (tmp_path / "r2" / "circuit.txt").samefile(cache / "toy" / "circuit.txt")
    assert first["meta"] == CC.head(tmp_path / "r0" / "circuit.txt") and later["meta"] == CC.head(tmp_path / "r1" / "circuit.txt")


def test_a_cache_key_moves_with_what_the_compose_reads_and_nothing_else(tmp_path, monkeypatch):
    """The key is what the compose reads (the caller's ``what``), compose_circuit's own bytes', the caller's composer file's,
    every file's of the packages and the lowering flags': another template, the reads in slots, a changed composer, a changed
    code or data file or a ``FLOCK_*`` flag moves it; a file under ``tests/`` or ``__pycache__``, a doc, or ``FLOCK_WORK`` (a
    place), does not. ``--reads slots`` is refused without ``--template``."""
    pkg = tmp_path / "composecachekeypkg"
    for d in ("sub", "tests", "__pycache__"):
        (pkg / d).mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "lower.py").write_text("X = 1\n")
    composer = tmp_path / "compose_circuit.py"
    composer.write_bytes(Path(CC.__file__).read_bytes())
    monkeypatch.syspath_prepend(str(tmp_path))
    for k in [k for k in os.environ if k.startswith("FLOCK_")]:
        monkeypatch.delenv(k)

    def key(t="T{N=1}", reads="inline"):
        CC.tree_digest.cache_clear()
        return CC.cache_key(["unit/serving-rows", t, reads], ("composecachekeypkg",), (composer,))

    k = key()
    assert key() == k != key("T{N=2}") and key(reads="slots") not in (k, key("T{N=2}"))
    (pkg / "tests" / "test_lower.py").write_text("X = 3\n")
    (pkg / "__pycache__" / "lower.py").write_text("X = 3\n")
    (pkg / "sub" / "README.md").write_text("the lowering\n")
    monkeypatch.setenv("FLOCK_WORK", str(tmp_path))
    assert key() == k
    (pkg / "sub" / "table.json").write_text("[1, 2]")
    k2 = key()
    assert k2 != k
    (pkg / "sub" / "lower.py").write_text("X = 2\n")
    k3 = key()
    assert k3 != k2
    composer.write_bytes(composer.read_bytes() + b"\n")
    k4 = key()
    assert k4 != k3
    monkeypatch.setenv("FLOCK_PLAIN_LEAVES", "1")
    assert key() != k4
    monkeypatch.setattr(sys, "argv", ["compose_circuit.py", "--input-set", str(tmp_path), "--out", str(tmp_path / "c.txt"),
                                      "--reads", "slots"])
    with pytest.raises(SystemExit) as e:
        CC.main()
    assert e.value.code == 2


def test_an_own_composers_key_moves_with_its_inputs_its_definition_and_its_file(tmp_path):
    """a5's key for a caller's composer (`compose_key`): another statement, another template id, another Definition (its
    program digest) or another composer file each move it; a composer that says nothing of its inputs is not cached."""
    k = a5.compose_key(_own(), TOY, "Toy")
    assert a5.compose_key(_own(), TOY, "Toy") == k
    assert a5.compose_key(_own(inputs=("unit/compose-cache-toy", ROWS[:1])), TOY, "Toy") != k
    assert a5.compose_key(_own(), TOY, "Toy{N=2}") != k
    other = bind(definition("ComposeCacheToy2", 1, lambda C, a: [C.AND(a[0], a[1])], (("a", Array(2, BIT)),), Array(1, BIT),
                            register=False))
    assert a5.compose_key(_own(), other, "Toy") != k
    src = tmp_path / "composer_elsewhere.py"
    src.write_text("def compose(out, partition, program):\n    pass\n")
    spec = importlib.util.spec_from_file_location("composer_elsewhere", src)
    elsewhere = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elsewhere)
    elsewhere.compose.inputs = ("unit/compose-cache-toy", ROWS)
    assert a5.compose_key(elsewhere.compose, TOY, "Toy") != k
    plain = _own()
    del plain.inputs
    assert a5.compose_key(plain, TOY, "Toy") is None


@pytest.mark.slow
def test_a_window_warmed_before_its_audit_stages_from_the_cache_reading_no_circuit(tmp_path, monkeypatch):
    """`a5.warm` composes each template a window's members name once, in forked processes, bound to the window's partition and
    program; the audit's staging (`own_composed`) then takes each from the cache without composing and its line's META, pin
    and class without reading the circuit (`circuit_of`); a sampled proof block's circuit is read whole once (`held`), and one
    that changed after it was kept is refused there."""
    sets = {t: (_own(), TOY) for t in ("Toy{N=1}", "Toy{N=2}")}
    mine = SimpleNamespace(members=[SimpleNamespace(template=t) for t in ("Toy{N=1}", "Toy{N=2}", "Toy{N=1}")],
                           partition_digest=B1["partition"], partition={"program": B1["program"]})
    a = SimpleNamespace(compose_cache=str(tmp_path / "cache"))
    got = a5.warm(a, sets, mine, jobs=2)
    assert [g["template"] for g in got] == ["Toy{N=1}", "Toy{N=2}"] and all(g["kept"] is False for g in got)
    assert [g["kept"] for g in a5.warm(a, sets, mine, jobs=1)] == [True, True]
    calls: list = []
    out = tmp_path / "audit" / "m0" / "circuit.txt"
    made, kept = a5.own_composed(a, _own(calls), TOY, "Toy{N=1}", out, B1["partition"], B1["program"])
    assert calls == [] and kept is True and made["pin"] == got[0]["pin"]
    monkeypatch.setattr(a5.FI, "flock_circuit", lambda b: pytest.fail("staging read a kept circuit"))
    read: set[Path] = set()
    circ = a5.circuit_of(out, made, read)
    assert read == set() and circ.meta == CC.head(out)
    monkeypatch.undo()
    a5.held(out, circ, read)
    assert read == {out}
    a5.held(out, circ, read)
    bad = bytearray(out.read_bytes())
    bad[-2] ^= 1
    other = tmp_path / "audit" / "m1" / "circuit.txt"
    other.parent.mkdir(parents=True)
    other.write_bytes(bad)
    with pytest.raises(RuntimeError, match="not the circuit its compose's line names"):
        a5.held(other, circ, read)
    with pytest.raises(ValueError, match="--compose-cache"):
        a5.warm(SimpleNamespace(compose_cache=None), sets, mine)


@pytest.mark.slow
def test_a_slots_circuit_gets_its_tables_composed_or_restamped(tmp_path, monkeypatch, capsys):
    """``--template --reads slots --cache`` writes the MUFU tables (`ir_lower.write_tables`) beside the circuit it composes, the
    cache keeps them with it and links them beside a restamped circuit; ``--reads inline`` writes none. The toy as the
    template, through a stand-in for `verity_vllm`'s template lookup, which this suite doesn't read."""
    pipeline = ModuleType("verity_vllm.pipeline")
    pipeline.serving_rows = SimpleNamespace(template=lambda tid: SimpleNamespace(fn=TOY))
    pipeline.chunk_rows = SimpleNamespace(unit_rows=lambda tp: (ROWS, [(V2, (0,))]))
    monkeypatch.setitem(sys.modules, "verity_vllm", ModuleType("verity_vllm"))
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline", pipeline)
    monkeypatch.setattr(CC, "cache_key", lambda what: f"{what[1]}-{what[2]}")
    written: list[Path] = []

    def write_tables(d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "rsq.u32").write_bytes(b"\x01\x00\x00\x00")
        written.append(d)

    monkeypatch.setattr(CC.FC.IL, "write_tables", write_tables)

    def compose(out: Path, binding: dict, reads: str) -> None:
        monkeypatch.setattr(sys, "argv", ["compose_circuit.py", "--template", "Toy", "--out", str(out), "--reads", reads,
                                          "--cache", str(tmp_path / "cache"), *(f"--{k}={v}" for k, v in binding.items())])
        assert CC.main() == 0

    compose(tmp_path / "r0" / "circuit.txt", B1, "slots")
    compose(tmp_path / "r1" / "circuit.txt", B2, "slots")
    assert "restamped" in capsys.readouterr().err
    compose(tmp_path / "r2" / "circuit.txt", B1, "inline")
    assert len(written) == 1 and (tmp_path / "cache" / "Toy-slots" / "tables" / "rsq.u32").is_file()
    assert all((tmp_path / r / "tables" / "rsq.u32").read_bytes() == b"\x01\x00\x00\x00" for r in ("r0", "r1"))
    assert not (tmp_path / "r2" / "tables").exists()


@pytest.mark.slow
def test_a_changed_kept_circuit_and_a_binding_not_found_once_are_refused(tmp_path):
    """A binding the file does not hold, or a pin not the file's, is refused by `binding_at`; a binding that is not two
    lowercase SHA-512 digests is refused before the entry is read; and a kept circuit changed by one bit is refused as it is
    copied, the caller's file never written."""
    cache, out = tmp_path / "cache", tmp_path / "r0" / "circuit.txt"
    made = CC.cached(cache, "toy", out, B1, _compose(B1))
    with pytest.raises(RuntimeError, match="once each"):
        CC.binding_at(out, B2, made["pin"])
    with pytest.raises(RuntimeError, match="not the pin"):
        CC.binding_at(out, B1, "00" * 64)
    r1 = tmp_path / "r1" / "circuit.txt"
    with pytest.raises(ValueError, match="128 lowercase hex"):
        CC.cached(cache, "toy", r1, {**B2, "partition": "C3" * 64}, _compose(B2))
    kept = bytearray((cache / "toy" / "circuit.txt").read_bytes())
    kept[-2] ^= 1
    (cache / "toy" / "circuit.txt").write_bytes(kept)
    with pytest.raises(RuntimeError, match="not the pin"):
        CC.cached(cache, "toy", r1, B2, _compose(B2))
    assert sorted(p.name for p in r1.parent.iterdir()) == []


def test_a_compose_that_fails_keeps_nothing(tmp_path):
    """A compose that raises leaves no entry and no temporary directory, and the next caller composes."""
    def boom(out: Path) -> dict:
        out.write_text("half")
        raise RuntimeError("compose failed")

    with pytest.raises(RuntimeError, match="compose failed"):
        CC.cached(tmp_path / "cache", "toy", tmp_path / "r0" / "circuit.txt", B1, boom)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["toy.lock"]
