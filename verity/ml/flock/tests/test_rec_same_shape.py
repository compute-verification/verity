"""`verity_flock.rec_vstage.same_shape`, the prover's refusal of a private circuit of another shape than its class's, before
anything of its statement is committed or registered (the recursive audit's ``shapeOf``, ``Proofs/Flock/Recursive/
ExecClass.lean``). The shape is the circuit file (its SHA-512) and the blocks of the instance count: a statement with
another circuit, or as many instances in another number of blocks, is refused by name; one whose count differs inside the same
blocks is not, since ``shapeOf`` drops ``n``. `rec_vstage stage --class --statement` refuses before it reads the session, and
``shape`` exits 2.

The recorded classes' statements (`RECORDED`) are read from the evidence store: 6720's three in every run, 2f10's and 80e4's
with ``VERITY_FLOCK_SHAPE_RECORDED=1`` (about 830 MB), and the 8090 classes' (2922, 322a: a 31.8 GB circuit file each) with
``VERITY_FLOCK_SHAPE_RECORDED=all``."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from verity_flock import rec_shape as RS
from verity_flock import rec_vstage as S

FLOCK = Path(__file__).resolve().parents[1]


def _circuit(d: Path, body: bytes = b"body\n", k_log: int = 22, vus_per_block: int = 2) -> Path:
    p = d / "circuit.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"flock-circuit unit-class/test\nMETA " + json.dumps({"k_log": k_log, "vus_per_block": vus_per_block}).encode()
                  + b"\n" + body)
    return p


def _public(circuit: Path, d: Path, n: int, header: dict | None = None) -> Path:
    p = d / f"pub-{n}.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    h = {"circuit_sha512": hashlib.sha512(circuit.read_bytes()).hexdigest(), "instances": n, **(header or {})}
    p.write_bytes(json.dumps(h).encode() + b"\n" + bytes(16 * n))
    return p


@pytest.fixture
def cls(tmp_path) -> tuple[Path, Path]:
    """A class at 64 instances, two a block: 32 blocks, nbl 5, m 27."""
    c = _circuit(tmp_path / "cls")
    return c, _public(c, tmp_path / "cls", 64)


def test_a_statement_is_its_circuit_file_and_its_blocks(cls):
    got = S.statement_shape(*cls)
    assert (got["nbl"], got["m"], got["n"]) == (5, 27, 64)
    assert got["circuit_sha512"] == hashlib.sha512(cls[0].read_bytes()).hexdigest()


@pytest.mark.parametrize("n", [63, 62, 33])
def test_another_count_in_the_same_blocks_is_the_same_shape(cls, tmp_path, n):
    """``shapeOf`` keeps the blocks, not ``n``: 33 to 64 instances, two a block, are 17 to 32 blocks, all nbl 5."""
    res = S.same_shape(cls, (cls[0], _public(cls[0], tmp_path / "p", n)))
    assert res["statement"]["n"] == n and res["statement"]["nbl"] == res["class"]["nbl"] == 5


@pytest.mark.parametrize("n", [65, 32, 1])
def test_another_number_of_blocks_is_refused(cls, tmp_path, n):
    with pytest.raises(RS.Refused, match="SHAPE-REFUSED: .* another blocks than its class's"):
        S.same_shape(cls, (cls[0], _public(cls[0], tmp_path / "p", n)))


def test_the_fewest_blocks_are_three(tmp_path):
    """``nbl = max 3 ⌈log2 blocks⌉``: one instance and sixteen, two a block, are one and eight blocks, both nbl 3."""
    c = _circuit(tmp_path)
    assert S.same_shape((c, _public(c, tmp_path / "a", 1)), (c, _public(c, tmp_path / "b", 16)))["class"]["nbl"] == 3


@pytest.mark.parametrize("change", ["body", "k_log", "vus_per_block"])
def test_another_circuit_is_refused(cls, tmp_path, change):
    """One byte of the circuit's body, or META's k_log or vus_per_block, is another circuit file at the same blocks."""
    kw = {"body": {"body": b"bodz\n"}, "k_log": {"k_log": 21}, "vus_per_block": {"vus_per_block": 1}}[change]
    c = _circuit(tmp_path / "p", **kw)
    p = (c, _public(c, tmp_path / "p", 64 // (2 if change == "vus_per_block" else 1)))
    with pytest.raises(RS.Refused, match=r"SHAPE-REFUSED: .* another circuit \(its file's SHA-512\) than its class's"):
        S.same_shape(cls, p)


@pytest.mark.parametrize("header", [{"circuit_sha512": "00" * 64}, {"circuit_sha512": None}])
def test_a_public_file_staged_for_another_circuit_is_refused(cls, tmp_path, header):
    """``HmRow.loadPublic`` sets up no statement from a public file that names another circuit, or none."""
    p = (cls[0], _public(cls[0], tmp_path / "p", 64, header))
    with pytest.raises(RS.Refused, match="SHAPE-REFUSED: .* is staged for another circuit than"):
        S.same_shape(cls, p)
    with pytest.raises(RS.Refused, match="is staged for another circuit than"):
        S.same_shape(p, cls)


def test_stage_refuses_another_shape_before_it_reads_the_session(cls, tmp_path):
    """The check comes first: no inner session at all, and nothing is written under OUT."""
    out = tmp_path / "out"
    with pytest.raises(RS.Refused, match="another blocks"):
        S.stage(tmp_path / "no-inner", {}, out, cls=cls, statement=(cls[0], _public(cls[0], tmp_path / "p", 65)))
    assert not out.exists()
    with pytest.raises(ValueError, match="go together"):
        S.stage(tmp_path / "no-inner", {}, out, cls=cls)


def test_the_command_exits_2_on_another_shape(cls, tmp_path, capsys):
    assert S.main(["shape", *map(str, cls), str(cls[0]), str(_public(cls[0], tmp_path / "p", 63))]) == 0
    assert json.loads(capsys.readouterr().out)["statement"]["n"] == 63
    assert S.main(["shape", *map(str, cls), str(cls[0]), str(_public(cls[0], tmp_path / "p", 65))]) == 2
    assert "another blocks" in json.loads(capsys.readouterr().out)["refused"]
    out, fold = tmp_path / "out", tmp_path / "fold.json"
    fold.write_text("{}")
    assert S.main(["stage", str(tmp_path / "no-inner"), str(fold), str(out), "--class", *map(str, cls),
                   "--statement", str(cls[0]), str(_public(cls[0], tmp_path / "p", 65))]) == 2
    assert not out.exists()


E8F4 = "art:e8f4f446"
U16 = f"{E8F4}/16-16-8"
U1024 = f"{E8F4}/1024-1024-512"
SC16 = "art:7dd2eb2f/out/rec-v0/sc-16-16-8-stage"
PROG = "out/rec-stage/f32mulftz-8090-64-32-n32-prog"
RESTAGE = "art:1564ad51/out/rec-stage/f32mulftz-8090-64-32-n32-restage-6692"
#: Each recorded class's statements, (circuit, public file) by store path. A class's statements share its circuit file: 322a's
#: other/circuit.txt is stage/'s bytes (sha256 95bed417…), and 2922's other statement names stage/'s circuit.
RECORDED = {
    "6720": (f"{U16}/circuit.txt", f"{U16}/n64-stage/pub-64.bin"),
    "6720 other": (f"{U16}/circuit.txt", f"{U16}/n64-other/pub-64.bin"),
    "6720 at 8": (f"{U16}/circuit.txt", f"{U16}/n8-stage/pub-8.bin"),
    "2f10": (f"{SC16}/circuit.txt", f"{SC16}/pub-8.bin"),
    "80e4": (f"{U1024}/circuit.txt", f"{U1024}/n8-stage/pub-8.bin"),
    "80e4 other": (f"{U1024}/circuit.txt", f"{U1024}/n8-other/pub-8.bin"),
    "2922": (f"art:7dd2eb2f/{PROG}/stage/circuit.txt", f"art:9379a0b2/{PROG}/stage/pub-32.bin"),
    "2922 other": (f"art:7dd2eb2f/{PROG}/stage/circuit.txt", f"art:9379a0b2/{PROG}/other/pub-32.bin"),
    "322a": (f"{RESTAGE}/stage/circuit.txt", f"{RESTAGE}/stage/pub-32.bin"),
    "322a other": (f"{RESTAGE}/stage/circuit.txt", f"{RESTAGE}/other/pub-32.bin"),
}
#: (class, inner statement, the same shape, the tier that reads them): every recorded class against its own statements, and a
#: private circuit of another shape in each way: blocks alone (6720 at 8 instances), the circuit alone (2f10, 322a), both (80e4).
CASES = [
    ("6720", "6720", True, ""), ("6720", "6720 other", True, ""), ("6720", "6720 at 8", False, ""),
    ("2f10", "2f10", True, "1"), ("2f10", "6720 at 8", False, "1"),
    ("80e4", "80e4", True, "1"), ("80e4", "80e4 other", True, "1"), ("80e4", "6720 other", False, "1"),
    ("2922", "2922", True, "all"), ("2922", "2922 other", True, "all"),
    ("322a", "322a", True, "all"), ("322a", "322a other", True, "all"), ("322a", "2922 other", False, "all"),
]
TIERS = {"": 0, "1": 1, "all": 2}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("store")


def _fetch(store: Path, ref: str) -> Path:
    """``art:<id>/<path>`` under store/<id>/<path>, fetching only that file and only once."""
    art, _, path = ref.partition("/")
    d = store / art.split(":", 1)[1]
    if not (d / path).exists():
        sys.path.insert(0, str(FLOCK / "verifier"))
        import ci

        try:
            subprocess.run([sys.executable, "-m", "research", "data", "fetch", art, "--to", str(d), "--path", path], check=True,
                           capture_output=True, env=ci.store_env())
        except (subprocess.CalledProcessError, OSError) as e:
            pytest.skip(f"the evidence store is not reachable: {e}")
    return d / path


def _python(cls: tuple[Path, Path], statement: tuple[Path, Path]) -> bool:
    try:
        S.same_shape(cls, statement)
    except RS.Refused:
        return False
    return True


@pytest.mark.slow
@pytest.mark.parametrize("cls_, statement, same, tier", CASES, ids=[f"{c}-{s}" for c, s, _, _ in CASES])
def test_each_recorded_class_takes_its_statements_and_refuses_another_shape(store, cls_, statement, same, tier):
    want = os.environ.get("VERITY_FLOCK_SHAPE_RECORDED", "")
    if TIERS[tier] > TIERS.get(want, 0):
        pytest.skip(f"VERITY_FLOCK_SHAPE_RECORDED={tier} reads this class's statements from the evidence store: about 830 MB "
                    "at 1, a 31.8 GB circuit file per 8090 class at all")
    c, s = (tuple(_fetch(store, r) for r in RECORDED[k]) for k in (cls_, statement))
    assert _python(c, s) is same
