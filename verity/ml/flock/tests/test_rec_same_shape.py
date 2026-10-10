"""`verity_flock.rec_vstage.same_shape`, the prover's refusal of a private circuit of another shape than its class's, before
anything of its statement is committed or registered (the recursive audit's ``shapeOf``, ``Proofs/Flock/Recursive/
ExecClass.lean``). The shape is the circuit file (its SHA-512) and the blocks of the instance count: a statement with
another circuit, or as many instances in another number of blocks, is refused by name; one whose count differs inside the same
blocks is not, since ``shapeOf`` drops ``n``. `rec_vstage stage --class --statement` refuses before it reads the session, and
``shape`` exits 2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from verity_flock import rec_shape as RS
from verity_flock import rec_vstage as S


def _circuit(d: Path, body: bytes = b"body\n", k_log: int = 22, vus_per_block: int = 2) -> Path:
    p = d / "circuit.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"flock-circuit unit-class/test\nMETA " + json.dumps({"k_log": k_log, "vus_per_block": vus_per_block}).encode()
                  + b"\n" + body)
    return p


def _public(d: Path, n: int) -> Path:
    p = d / f"pub-{n}.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps({"instances": n}).encode() + b"\n" + bytes(16 * n))
    return p


@pytest.fixture
def cls(tmp_path) -> tuple[Path, Path]:
    """A class at 64 instances, two a block: 32 blocks, nbl 5, m 27."""
    return _circuit(tmp_path / "cls"), _public(tmp_path / "cls", 64)


def test_a_statement_is_its_circuit_file_and_its_blocks(cls):
    got = S.statement_shape(*cls)
    assert (got["nbl"], got["m"], got["n"]) == (5, 27, 64)
    assert got["circuit_sha512"] == hashlib.sha512(cls[0].read_bytes()).hexdigest()


@pytest.mark.parametrize("n", [63, 62, 33])
def test_another_count_in_the_same_blocks_is_the_same_shape(cls, tmp_path, n):
    """``shapeOf`` keeps the blocks, not ``n``: 33 to 64 instances, two a block, are 17 to 32 blocks, all nbl 5."""
    res = S.same_shape(cls, (cls[0], _public(tmp_path / "p", n)))
    assert res["statement"]["n"] == n and res["statement"]["nbl"] == res["class"]["nbl"] == 5


@pytest.mark.parametrize("n", [65, 32, 1])
def test_another_number_of_blocks_is_refused(cls, tmp_path, n):
    with pytest.raises(RS.Refused, match="SHAPE-REFUSED: .* another blocks than its class's"):
        S.same_shape(cls, (cls[0], _public(tmp_path / "p", n)))


def test_the_fewest_blocks_are_three(tmp_path):
    """``nbl = max 3 ⌈log2 blocks⌉``: one instance and sixteen, two a block, are one and eight blocks, both nbl 3."""
    c = _circuit(tmp_path)
    assert S.same_shape((c, _public(tmp_path / "a", 1)), (c, _public(tmp_path / "b", 16)))["class"]["nbl"] == 3


@pytest.mark.parametrize("change", ["body", "k_log", "vus_per_block"])
def test_another_circuit_is_refused(cls, tmp_path, change):
    """One byte of the circuit's body, or META's k_log or vus_per_block, is another circuit file at the same blocks."""
    kw = {"body": {"body": b"bodz\n"}, "k_log": {"k_log": 21}, "vus_per_block": {"vus_per_block": 1}}[change]
    p = (_circuit(tmp_path / "p", **kw), _public(tmp_path / "p", 64 // (2 if change == "vus_per_block" else 1)))
    with pytest.raises(RS.Refused, match=r"SHAPE-REFUSED: .* another circuit \(its file's SHA-512\) than its class's"):
        S.same_shape(cls, p)


def test_stage_refuses_another_shape_before_it_reads_the_session(cls, tmp_path):
    """The check comes first: no inner session at all, and nothing is written under OUT."""
    out = tmp_path / "out"
    with pytest.raises(RS.Refused, match="another blocks"):
        S.stage(tmp_path / "no-inner", {}, out, cls=cls, statement=(cls[0], _public(tmp_path / "p", 65)))
    assert not out.exists()
    with pytest.raises(ValueError, match="go together"):
        S.stage(tmp_path / "no-inner", {}, out, cls=cls)


def test_the_command_exits_2_on_another_shape(cls, tmp_path, capsys):
    assert S.main(["shape", *map(str, cls), str(cls[0]), str(_public(tmp_path / "p", 63))]) == 0
    assert json.loads(capsys.readouterr().out)["statement"]["n"] == 63
    assert S.main(["shape", *map(str, cls), str(cls[0]), str(_public(tmp_path / "p", 65))]) == 2
    assert "another blocks" in json.loads(capsys.readouterr().out)["refused"]
    out, fold = tmp_path / "out", tmp_path / "fold.json"
    fold.write_text("{}")
    assert S.main(["stage", str(tmp_path / "no-inner"), str(fold), str(out), "--class", *map(str, cls),
                   "--statement", str(cls[0]), str(_public(tmp_path / "p", 65))]) == 2
    assert not out.exists()
