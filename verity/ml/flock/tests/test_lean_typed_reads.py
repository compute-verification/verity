"""Item 1e for a template that reads tables (typed attention, `attention-head/d64-bn16/sm80-fa2-bf16`): each placed read is
a part whose layout `table/v2` generates, and its product rows' B side is the table itself. The statement names each table
by the SHA-512 of its words and never carries them; the verifier holds its own copy (`Typed.hold`), a library table found
by its name and checked against that SHA-512 before any use."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: typed attention (4 instances) staged by `typed_statement.stage` with its `rows/`, and the ex2 and rcp tables it reads
FIXTURE = "art:e1673794"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def attn(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci  # noqa: PLC0415

    try:
        stage = ci.fetch(f"{FIXTURE}/stage", tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return stage


def _statement(stage: Path, tables: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit/types@210d32e1", "--circuit",
                           str(stage / "circuit.txt"), "--public", str(stage / "pub-4.bin"), "--tables", str(tables)],
                          capture_output=True, text=True)


def test_a_table_whose_words_are_not_its_name_is_refused(attn, tmp_path):
    shutil.copytree(attn / "tables", tmp_path / "tables")
    ex2 = tmp_path / "tables" / "ex2.u32"
    b = bytearray(ex2.read_bytes())
    b[4 * 12345] ^= 1
    ex2.write_bytes(bytes(b))
    p = _statement(attn, tmp_path / "tables")
    assert p.returncode != 0 and "table ex2: its words are not the table its SHA-512 names" in p.stdout + p.stderr, p.stdout


def test_a_table_not_given_is_refused(attn, tmp_path):
    (tmp_path / "tables").mkdir()
    shutil.copy(attn / "tables" / "rcp.u32", tmp_path / "tables")
    p = _statement(attn, tmp_path / "tables")
    assert p.returncode != 0 and "table ex2: not given" in p.stdout + p.stderr, p.stdout


@pytest.fixture(scope="module")
def part(attn, tmp_path_factory):
    """`tests/lean/TypedRowsPart.lean` built against the verifier's own build, as a package that requires it by path; run
    from the verifier's directory, so that it takes the verifier's toolchain."""
    pkg = tmp_path_factory.mktemp("typed-rows-part")
    (pkg / "lakefile.toml").write_text(
        f'name = "typed_rows_part"\ndefaultTargets = ["typed-rows-part"]\n\n[[require]]\nname = "flock_verifier"\n'
        f'path = "{LEAN}"\n\n[[lean_exe]]\nname = "typed-rows-part"\nroot = "TypedRowsPart"\n')
    shutil.copy(Path(__file__).resolve().parent / "lean" / "TypedRowsPart.lean", pkg)
    p = subprocess.run(["lake", "-d", str(pkg), "build"], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return pkg / ".lake" / "build" / "bin" / "typed-rows-part"


@pytest.mark.slow
@pytest.mark.parametrize("which", ["rows", "order0", "order1", "parts"])
def test_the_verifiers_rows_are_the_writers(attn, part, tmp_path, which):
    """Through `deriveChecked`, the rows are the writer's `rows/` byte for byte: the root, the tensor-core part, each read's
    generated layout with its `READ` line (ex2 k 23, rcp k 24), and `delta.txt`. Deriving and checking the full template
    takes the verifier about 14 minutes in one process, so `typed-rows` runs here in four parts, side by side
    (`TypedRowsPart.lean`, whose theorems make them pass exactly when `typed-rows` does): `rows` checks every layout and
    writes the rows, `order0` and `order1` check the unit's logical order, and `parts` its parts. Each part derives the
    layouts again, so each still takes minutes and is marked `slow` (`tools/verity/check/suites.py --quick` leaves it out);
    GEMM's derivation (`test_lean_typed_template.py`) is not."""
    p = subprocess.run([str(part), which, str(attn / "circuit.txt"), "--tables", str(attn / "tables"),
                        "--out", str(tmp_path / "rows")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    if which == "rows":
        want = sorted((attn / "rows").iterdir())
        for f in want:
            assert (tmp_path / "rows" / f.name).read_bytes() == f.read_bytes(), f.name
