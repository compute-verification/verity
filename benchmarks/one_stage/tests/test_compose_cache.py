"""`compose_circuit.py --cache`: the ranks a machine audits compose each template once. A circuit composed bound to one partition
and program, kept, and taken bound to another (`restamp`) is byte for byte, with compose's line, the circuit composed bound to
that one, its caller never composing, whatever the reads' chunks split; a cache key moves with the template, the reads' layout
and any composing file, compose_circuit's own among them, and with nothing else; a circuit whose reads are in slots gets its
MUFU tables beside it whether composed or restamped; and a kept circuit that changed, a binding
not in the file once, or a binding that is not two SHA-512 digests is refused, writing nothing. A toy unit Definition."""

from __future__ import annotations

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
import compose_circuit as CC  # noqa: E402

#: a + b over 16 bits with its carry, and w[0] & w[2047]: w's row spans SHA-512 blocks, as a composed circuit needs one row to
TOY = bind(definition("ComposeCacheToy", 1, lambda C, a, b, w: add(C, a, b, width=16, carry_out=True) + [C.AND(w[0], w[2047])],
                      (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(18, BIT), register=False))
ROWS = [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])]
B1 = {"partition": "a1" * 64, "program": "b2" * 64}
B2 = {"partition": "c3" * 64, "program": "d4" * 64}


def _compose(out: Path, binding: dict, calls: list | None = None):
    """compose_circuit's ``compose`` for the toy: ``out`` composed bound to ``binding``, -> its line."""
    def compose() -> dict:
        if calls is not None:
            calls.append(binding)
        st = CS.UnitStager(TOY, "unit/compose-cache-toy", ROWS, [(V2, (0,))], circuit=out, bind=binding)
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}
    return compose


@pytest.mark.slow
@pytest.mark.parametrize("split", ["whole", "partition", "program"])
def test_a_kept_circuit_taken_with_another_binding_is_the_circuit_composed_bound_to_it(split, tmp_path, monkeypatch):
    """The first caller composes and keeps; a later one bound elsewhere gets the bytes and line a fresh compose bound there
    gives (another pin, the one class) without composing, and one bound as the first gets the first's; with the reads' chunks
    ending inside the partition's value or the program's, the same."""
    cache, calls = tmp_path / "cache", []
    first = CC.cached(cache, "toy", tmp_path / "r0" / "circuit.txt", B1, _compose(tmp_path / "r0" / "circuit.txt", B1, calls))
    at = CC.binding_at(tmp_path / "r0" / "circuit.txt", B1, first["pin"])
    if split != "whole":
        monkeypatch.setattr(CC, "_CHUNK", at[split] + 40)
        assert CC.binding_at(tmp_path / "r0" / "circuit.txt", B1, first["pin"]) == at
    later = CC.cached(cache, "toy", tmp_path / "r1" / "circuit.txt", B2, _compose(tmp_path / "r1" / "circuit.txt", B2, calls))
    again = CC.cached(cache, "toy", tmp_path / "r2" / "circuit.txt", B1, _compose(tmp_path / "r2" / "circuit.txt", B1, calls))
    assert calls == [B1]
    fresh = _compose(tmp_path / "fresh" / "circuit.txt", B2)()
    assert (tmp_path / "r1" / "circuit.txt").read_bytes() == (tmp_path / "fresh" / "circuit.txt").read_bytes()
    assert later == fresh and later["pin"] != first["pin"] and later["class_sha512"] == first["class_sha512"]
    assert again == first and (tmp_path / "r2" / "circuit.txt").read_bytes() == (tmp_path / "r0" / "circuit.txt").read_bytes()


def test_a_cache_key_moves_with_the_template_and_with_any_composing_file(tmp_path, monkeypatch):
    """The key is the template's, the reads', compose_circuit's own bytes' and every ``.py`` file's of the packages: another
    template, the reads in slots, a changed composer or a changed file moves it; a file that is not Python, or one in
    ``__pycache__``, does not. ``--reads slots`` is refused without ``--template``."""
    pkg = tmp_path / "composecachekeypkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__pycache__").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "lower.py").write_text("X = 1\n")
    composer = tmp_path / "compose_circuit.py"
    composer.write_bytes(Path(CC.__file__).read_bytes())
    monkeypatch.syspath_prepend(str(tmp_path))
    key = lambda t="T{N=1}", reads="inline": CC.cache_key(t, reads, ("composecachekeypkg",), composer)  # noqa: E731
    k = key()
    assert key() == k != key("T{N=2}") and key(reads="slots") not in (k, key("T{N=2}"))
    (pkg / "notes.txt").write_text("not code")
    (pkg / "__pycache__" / "lower.py").write_text("X = 3\n")
    assert key() == k
    (pkg / "sub" / "lower.py").write_text("X = 2\n")
    k2 = key()
    assert k2 != k
    composer.write_bytes(composer.read_bytes() + b"\n")
    assert key() != k2
    monkeypatch.setattr(sys, "argv", ["compose_circuit.py", "--input-set", str(tmp_path), "--out", str(tmp_path / "c.txt"),
                                      "--reads", "slots"])
    with pytest.raises(SystemExit) as e:
        CC.main()
    assert e.value.code == 2


@pytest.mark.slow
def test_a_slots_circuit_gets_its_tables_composed_or_restamped(tmp_path, monkeypatch, capsys):
    """``--template --reads slots --cache`` writes the MUFU tables (`ir_lower.write_tables`) beside the circuit when it composes
    and when it restamps a kept circuit, which copies circuit.txt alone; ``--reads inline`` writes none. The toy as the
    template, through a stand-in for `verity_vllm`'s template lookup, which this suite doesn't read."""
    pipeline = ModuleType("verity_vllm.pipeline")
    pipeline.serving_rows = SimpleNamespace(template=lambda tid: SimpleNamespace(fn=TOY))
    pipeline.chunk_rows = SimpleNamespace(unit_rows=lambda tp: (ROWS, [(V2, (0,))]))
    monkeypatch.setitem(sys.modules, "verity_vllm", ModuleType("verity_vllm"))
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline", pipeline)
    monkeypatch.setattr(CC, "cache_key", lambda t, reads="inline": f"{t}-{reads}")
    written: list[Path] = []
    monkeypatch.setattr(CC.FC.IL, "write_tables", written.append)

    def compose(out: Path, binding: dict, reads: str) -> None:
        monkeypatch.setattr(sys, "argv", ["compose_circuit.py", "--template", "Toy", "--out", str(out), "--reads", reads,
                                          "--cache", str(tmp_path / "cache"), *(f"--{k}={v}" for k, v in binding.items())])
        assert CC.main() == 0

    compose(tmp_path / "r0" / "circuit.txt", B1, "slots")
    compose(tmp_path / "r1" / "circuit.txt", B2, "slots")
    assert "restamped" in capsys.readouterr().err
    compose(tmp_path / "r2" / "circuit.txt", B1, "inline")
    assert written == [tmp_path / "r0" / "tables", tmp_path / "r1" / "tables"]


@pytest.mark.slow
def test_a_changed_kept_circuit_and_a_binding_not_found_once_are_refused(tmp_path):
    """A binding the file does not hold, or a pin not the file's, is refused by `binding_at`; a binding that is not two
    lowercase SHA-512 digests is refused before the entry is read; and a kept circuit changed by one bit is refused as it is
    copied, the caller's file never written."""
    cache, out = tmp_path / "cache", tmp_path / "r0" / "circuit.txt"
    made = CC.cached(cache, "toy", out, B1, _compose(out, B1))
    with pytest.raises(RuntimeError, match="once each"):
        CC.binding_at(out, B2, made["pin"])
    with pytest.raises(RuntimeError, match="not the pin"):
        CC.binding_at(out, B1, "00" * 64)
    r1 = tmp_path / "r1" / "circuit.txt"
    with pytest.raises(ValueError, match="128 lowercase hex"):
        CC.cached(cache, "toy", r1, {**B2, "partition": "C3" * 64}, _compose(r1, B2))
    kept = bytearray((cache / "toy" / "circuit.txt").read_bytes())
    kept[-2] ^= 1
    (cache / "toy" / "circuit.txt").write_bytes(kept)
    with pytest.raises(RuntimeError, match="not the pin"):
        CC.cached(cache, "toy", r1, B2, _compose(r1, B2))
    assert sorted(p.name for p in r1.parent.iterdir()) == []
