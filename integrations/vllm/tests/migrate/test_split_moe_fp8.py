"""`tools/migrate/split_moe_fp8.py` on synthetic trees: what it moves and rewrites (from-imports, aliases, function-level and
relative imports, strings and comments, `__all__`, the imports a move leaves unused), that a rerun changes nothing, and each loud
refusal (a half-applied split, a name in neither home, a moved name the source still uses, a moved name read by string, `import *`,
an edit whose text is gone)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/migrate/split_moe_fp8.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_moe_fp8", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["split_moe_fp8"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load()
OLD = "verity_vllm.program." + "registry"
NEW = "verity_vllm.program.kernels"
PKG = "pkg/verity_vllm"
REG, KER = f"{PKG}/program/registry", f"{PKG}/program/kernels"

FAM = f'''"""The family."""
from __future__ import annotations

import numpy as np

from {OLD} import prims as P
from {OLD}.prims import bits_of

__all__ = ["Defn", "twin", "VOCAB"]

K = 3
C = bits_of(np.float32(1.0))


def Defn(x):
    return P.add(x, K)


# The twin of `Defn`.
def twin(x):
    return np.float32(x) + K


VOCAB = {{"a": Defn}}
'''

FILES = {
    f"{PKG}/__init__.py": "",
    f"{PKG}/program/__init__.py": "",
    f"{REG}/__init__.py": "",
    f"{KER}/__init__.py": "",
    f"{REG}/prims.py": "def add(x, y):\n    return x + y\n\n\ndef bits_of(x):\n    return int(x)\n",
    f"{REG}/fam.py": FAM,
    f"{REG}/other.py": "from .fam import twin\n\nY = twin(2)\n",
    "pkg/tests/test_a.py": f"from {OLD}.fam import Defn, twin  # noqa: F401\n",
    "pkg/tests/test_b.py": (f'"""`registry.fam.twin`, `{OLD}.fam:twin`, fam.VOCAB and fam.Defn."""\n'
                            f"from {OLD} import fam as F\n\n"
                            "A = F.twin(1) + F.Defn(1)\n"
                            "B = f\"{F.Defn(2)} is fam.twin's\"  # fam.twin\n"),
    "pkg/tests/test_c.py": f"import {OLD}.fam as _F\n\nV = _F.VOCAB\n",
    "pkg/tests/test_d.py": f"def f():\n    from {OLD}.fam import twin\n    return twin(0)\n",
}

MOVES = [S.Move(f"{OLD}.fam", f"{NEW}.fam_twins", ("twin",)), S.Move(f"{OLD}.fam", f"{OLD}.fam_vocab", ("VOCAB",))]
DESTINATIONS = {f"{NEW}.fam_twins": '"""Twins."""', f"{OLD}.fam_vocab": '"""Vocabulary."""'}
EDITS = [S.Edit(f"{REG}/fam.py", "C = bits_of(np.float32(1.0))", "C = bits_of(1.0)")]


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _run(root: Path, moves=MOVES, edits=EDITS, keep=None) -> S.Plan:
    return S.run(root, moves, DESTINATIONS, edits, keep or {})


def _apply(root: Path, **kw) -> S.Plan:
    plan = _run(root, **kw)
    plan.write()
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return plan


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text()


def test_the_split_moves_the_names_and_rewrites_every_reference(tmp_path):
    root = _tree(tmp_path, FILES)
    _apply(root)

    assert _read(root, f"{KER}/fam_twins.py") == (
        '"""Twins."""\nfrom __future__ import annotations\n\nimport numpy as np\n\n'
        f"from {OLD}.fam import K\n\n\n"
        "# The twin of `Defn`.\ndef twin(x):\n    return np.float32(x) + K\n")
    assert _read(root, f"{REG}/fam_vocab.py") == (
        f'"""Vocabulary."""\nfrom __future__ import annotations\n\nfrom {OLD}.fam import Defn\n\n\nVOCAB = {{"a": Defn}}\n')
    fam = _read(root, f"{REG}/fam.py")
    assert "import numpy" not in fam and "C = bits_of(1.0)" in fam
    assert '__all__ = ["Defn"]' in fam
    assert "def twin" not in fam and "VOCAB" not in fam and fam.endswith("return P.add(x, K)\n")

    assert _read(root, f"{REG}/other.py").startswith(f"from {NEW}.fam_twins import twin\n")
    assert [" ".join(s.split()) for s in _read(root, "pkg/tests/test_a.py").splitlines()] == [
        f"from {OLD}.fam import Defn # noqa: F401", f"from {NEW}.fam_twins import twin # noqa: F401"]
    b = _read(root, "pkg/tests/test_b.py")
    assert f"from {OLD} import fam as F\nfrom {NEW} import fam_twins\n" in b
    assert "A = fam_twins.twin(1) + F.Defn(1)" in b
    assert '"""`kernels.fam_twins.twin`, `' + NEW + '.fam_twins.twin`, fam_vocab.VOCAB and fam.Defn."""' in b
    assert "is fam_twins.twin's\"  # fam_twins.twin" in b
    assert _read(root, "pkg/tests/test_c.py") == f"from {OLD} import fam_vocab as _fam_vocab\n\nV = _fam_vocab.VOCAB\n"
    assert f"    from {NEW}.fam_twins import twin\n" in _read(root, "pkg/tests/test_d.py")

    for rel in [*FILES, f"{KER}/fam_twins.py", f"{REG}/fam_vocab.py"]:
        compile(_read(root, rel), rel, "exec")


def test_a_rerun_changes_nothing(tmp_path):
    root = _tree(tmp_path, FILES)
    _apply(root)
    plan = _run(root)
    assert plan.changed == {}
    assert sorted(plan.log) == [f"already split: {NEW}.fam_twins", f"already split: {OLD}.fam_vocab"]


def test_check_reports_without_writing(tmp_path, monkeypatch):
    root = _tree(tmp_path, FILES)
    monkeypatch.setattr(S.run, "__defaults__", (root, MOVES, DESTINATIONS, EDITS, {}))
    assert S.main(["--root", str(root), "--check"]) == 1
    assert _read(root, f"{REG}/fam.py") == FAM
    assert S.main(["--root", str(root)]) == 0
    assert S.main(["--root", str(root), "--check"]) == 0


def test_a_late_reference_on_a_split_tree_is_rewritten(tmp_path):
    root = _tree(tmp_path, FILES)
    _apply(root)
    _tree(root, {"pkg/tests/test_e.py": f"from {OLD}.fam import twin\n"})
    _apply(root)
    assert _read(root, "pkg/tests/test_e.py") == f"from {NEW}.fam_twins import twin\n"


@pytest.mark.parametrize("last", [False, True])
def test_the_keep_entry_goes_after_the_kernels_entries_once(tmp_path, last):
    after = "" if last else ',\n  "verity_vllm.x": "x"'
    keep_file = ('{\n "schema": "vllm-dead-code-keep/v1",\n "modules": {\n  "verity_vllm.a": "a",\n'
                 f'  "{NEW}.k": "k"{after}\n }}\n}}\n')
    root = _tree(tmp_path, {**FILES, S.KEEP_FILE: keep_file})
    _apply(root, keep={f"{NEW}.fam_twins": "the twins"})
    lines = _read(root, S.KEEP_FILE).splitlines()
    assert lines[4:6] == [f'  "{NEW}.k": "k",', f'  "{NEW}.fam_twins": "the twins"' + ("" if last else ",")]
    assert _run(root, keep={f"{NEW}.fam_twins": "the twins"}).changed == {}


def _refused(root: Path, match: str, **kw) -> None:
    before = {p: p.read_text() for p in root.rglob("*.py")}
    with pytest.raises(S.SplitError, match=match):
        _run(root, **kw).write()
    assert {p: p.read_text() for p in root.rglob("*.py")} == before


def test_a_half_applied_split_is_refused(tmp_path):
    root = _tree(tmp_path, FILES)
    _apply(root)
    _tree(root, {f"{REG}/fam.py": _read(root, f"{REG}/fam.py") + "\n\ndef twin(x):\n    return x\n"})
    _refused(root, "partially applied or diverged")


def test_a_name_in_neither_home_is_refused(tmp_path):
    root = _tree(tmp_path, FILES)
    _refused(root, r"fam\.nope \(source: False, destination: False\)",
             moves=[S.Move(f"{OLD}.fam", f"{NEW}.fam_twins", ("twin", "nope"))])


def test_a_moved_name_the_source_still_uses_is_refused(tmp_path):
    root = _tree(tmp_path, {**FILES, f"{REG}/fam.py": FAM + "\n\ndef uses():\n    return twin(1)\n"})
    _refused(root, r"still uses the moved names \['twin'\]")


def test_a_moved_name_read_by_string_is_refused(tmp_path):
    root = _tree(tmp_path, {**FILES, "pkg/tests/test_e.py": f"from {OLD} import fam\n\nT = getattr(fam, 'twin')\n"})
    _refused(root, r"getattr\(fam, 'twin'\) names a moved name by string")


def test_a_star_import_is_refused(tmp_path):
    root = _tree(tmp_path, {**FILES, "pkg/tests/test_e.py": f"from {OLD}.fam import *  # noqa: F403\n"})
    _refused(root, "import \\*")


def test_an_edit_whose_text_is_gone_is_refused(tmp_path):
    root = _tree(tmp_path, FILES)
    _refused(root, "neither the text to replace nor its replacement",
             edits=[S.Edit(f"{REG}/fam.py", "C = bits_of(np.float32(2.0))", "C = bits_of(2.0)")])
