"""The move of circuit-check's bindings and pins into Definition modules' sidecars (`migrate_bindings.py`), on a synthetic
`targets.py` whose bindings name real Definitions: each binding goes to the module that defines what it binds, with only the
imports and locals it reads; what no module of the sidecars' packages owns, or what a core sidecar could only bind by importing
circuit-check, stays; and the functions it writes make the roots the original made."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCRIPT = HERE.parent / "migrate_bindings.py"


def _script():
    spec = importlib.util.spec_from_file_location("migrate_bindings", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MB = _script()

TARGETS = '''
from typing import Any, Callable

TINY = {"H": 32}


def _local() -> Any:
    from verity.ml.circuits.ops import prims as P

    return P.F32ToBf16Rn


def _core_roots() -> list[Any]:
    """Core's GEMM Definitions at K = 64 over the Ampere step, and two casts."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.ops import prims as P

    A = P.AmpereBF16TcDot16
    rows = [bind(d, K=64, DOT=A) for d in (G.DotBf16, G.GemmCoordinate)]
    rows += [x for x in (P.E4m3ToF32, G.DotBf16)]
    return [P.F32ToE2m1Sat, bind(G.Gemm, K=32, N=3, DOT=A), _local()] + rows


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Tiny configurations."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates import merkle as M
    from verity_vllm.program.registry import b1

    n = TINY["H"] // 2
    return [
        # the residual at half the tiny width
        ("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=n)),
        ("MerkleRead_v1", lambda: bind(M.MerkleRead, ARITY=2, D=TINY["H"] // 32)),
    ]


def _kernel_roots() -> list[Any]:
    return []


#: the parameters each template is checked at
TEMPLATE_PARAMS = {
    "rope-head": [{"D": 64}],
    # no such template
    "no-such-template": [{"D": 1}],
}

__all__ = ["TEMPLATE_PARAMS", "TINY"]
'''.lstrip("\n")


def _run(text: str, name: str) -> list[Any]:
    """Root function ``name`` of binding file ``text``, called."""
    ns: dict[str, Any] = {"__name__": "synthetic"}
    exec(compile("from typing import Any, Callable\n" + text, "<binding>", "exec"), ns)
    return ns[name]()


@pytest.fixture
def planned(tmp_path, monkeypatch):
    p = tmp_path / "targets.py"
    p.write_text(TARGETS)
    monkeypatch.setattr(MB, "TARGETS", p)
    src = MB.Source(TARGETS)
    doc = MB._doc_digest(MB.ast.get_docstring(src.top["_core_roots"]))
    monkeypatch.setattr(MB, "DOCS", {"_core_roots": (doc, {"verity.ml.circuits.ops.gemm": "Core's GEMM Definitions at K = 64."})})
    return MB.plan()


def test_each_binding_goes_to_the_module_that_defines_what_it_binds(planned):
    moved, _targets, summary = planned
    assert {m: sorted(v) for m, v in moved.items()} == {
        "verity.ml.circuits.ops.gemm": ["_core_roots"], "verity.ml.circuits.ops.prims": ["_core_roots"],
        "verity_vllm.program.registry.b1": ["_model_roots"], "verity_numerical.bench.templates": ["TEMPLATE_PARAMS"]}
    assert summary["undocumented"] == ["verity.ml.circuits.ops.prims#_core_roots"]


def test_a_split_root_function_makes_the_roots_the_original_made(planned):
    """The comprehension over two modules' Definitions is split between them; each part imports only what it reads, and the
    local ``A`` goes where it is read."""
    moved, targets, _summary = planned
    gemm, prims = moved["verity.ml.circuits.ops.gemm"]["_core_roots"], moved["verity.ml.circuits.ops.prims"]["_core_roots"]
    assert "for x in (P.E4m3ToF32,)" in prims and "for x in (G.DotBf16,)" in gemm
    assert "import gemm" not in prims and "defs import bind" not in prims and "A = " not in prims
    assert "A = P.AmpereBF16TcDot16" in gemm and '"""Core\'s GEMM Definitions at K = 64."""' in gemm
    assert "_local()" in targets and "def _core_roots" in targets and "Core's GEMM Definitions at K = 64 over" in targets
    original = {fn.id for fn in _run(TARGETS, "_core_roots")}
    stayed = {fn.id for fn in _run(targets, "_core_roots")}
    assert {fn.id for fn in _run(gemm, "_core_roots")} | {fn.id for fn in _run(prims, "_core_roots")} | stayed == original
    assert stayed == {"F32ToBf16Rn_v1"} and "def _kernel_roots" in targets


def test_a_model_entry_moves_with_its_comment_and_what_it_reads_and_a_core_one_reading_the_checker_stays(planned):
    moved, targets, summary = planned
    b1 = moved["verity_vllm.program.registry.b1"]["_model_roots"]
    assert "# the residual at half the tiny width" in b1 and "from circuit_check.targets import TINY" in b1
    assert 'n = TINY["H"] // 2' in b1 and "merkle" not in b1
    assert [label for label, _make in _run(b1, "_model_roots")] == ["ResidualAdd_v1"]
    assert '("MerkleRead_v1"' in targets and '("ResidualAdd_v1"' not in targets
    assert any("circuit_check.targets' TINY" in s for s in summary["stays"])


def test_a_template_entry_moves_and_an_unknown_one_stays(planned):
    moved, targets, _summary = planned
    t = moved["verity_numerical.bench.templates"]["TEMPLATE_PARAMS"]
    assert t.startswith("#: the parameters each template is checked at\nTEMPLATE_PARAMS = {") and '"rope-head"' in t
    assert '"no-such-template"' not in t and '"no-such-template"' in targets and "# no such template" in targets
    assert '"TEMPLATE_PARAMS"' in targets


def test_a_docstring_the_table_was_not_written_against_is_an_error(tmp_path, monkeypatch):
    p = tmp_path / "targets.py"
    p.write_text(TARGETS)
    monkeypatch.setattr(MB, "TARGETS", p)
    monkeypatch.setattr(MB, "DOCS", {"_core_roots": ("0" * 16, {})})
    with pytest.raises(ValueError, match="_core_roots's docstring is not the one `DOCS` splits"):
        MB.plan()


def test_a_migrated_targets_py_has_nothing_to_move(tmp_path, monkeypatch):
    text = TARGETS.split("\n\n\ndef _core_roots")[0] + "\n\n\ndef _kernel_roots() -> list[Any]:\n    return []\n"
    p = tmp_path / "targets.py"
    p.write_text(text)
    monkeypatch.setattr(MB, "TARGETS", p)
    moved, targets, _summary = MB.plan()
    assert moved == {} and targets == text


def test_the_task_lists_differences_ignore_which_file_names_a_unit():
    t = {"definitions": [{"id": "A_v1", "as_call": False, "gates": 3, "units": ["_core_roots"]}], "templates": [],
         "root_errors": [], "pins": {"pieces": {"A_v1": {"and": 1}}}}
    moved = {**t, "definitions": [{**t["definitions"][0], "units": ["verity.ml.circuits.ops.gemm.circuit-check#_core_roots"]}]}
    assert MB.differences(t, moved) == []
    regated = {**t, "definitions": [{**t["definitions"][0], "gates": 4}]}
    assert MB.differences(t, regated) == [f"definitions: A_v1: {t['definitions'][0]} became {regated['definitions'][0]}"]
    assert MB.differences(t, {**t, "pins": {"pieces": {}}}) == ["pins changed"]


#: the source roots the migration reads and writes, copied whole
COPIED = ("tools/verity/circuit_check", ".", "integrations/vllm/verity_vllm", "verity/core/protocols/pouw",
          "benchmarks/numerical/python", "verity/ml/flock/python")


@pytest.mark.slow
def test_the_migration_keeps_the_task_list(tmp_path):
    """Run on a copy of this tree, the migration leaves the task list unchanged (`migrate_bindings.py --verify`: targets,
    statics, gate counts, root errors, every pin, and each target's binding units but for their file).  It catches a
    binding the migration moves wrong, which the synthetic tests cannot, and costs two catalog builds (minutes) only when
    there is something to move: on a migrated tree the dry run finds nothing and the check ends there."""
    env = {**os.environ, "PYTHONPATH": ""}
    dry = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True, text=True, cwd=REPO, env=env)
    assert dry.returncode == 0, dry.stderr
    summary = json.loads(dry.stdout)
    if not summary["modules"]:
        return
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "target", ".lake", "build")
    for rel in COPIED:
        shutil.copytree(REPO / rel, tmp_path / rel, ignore=ignore, symlinks=True)
    run = subprocess.run([sys.executable, str(tmp_path / "tools/verity/circuit_check/migrate_bindings.py"), "--verify"],
                         capture_output=True, text=True, cwd=tmp_path, env=env)
    assert run.returncode == 0, run.stderr[-4000:]
