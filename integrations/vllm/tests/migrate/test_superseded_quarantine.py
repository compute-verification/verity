"""`tools/migrate/superseded_quarantine.py` on synthetic trees: it deletes only the versions its `DELETE` lists (with their
circuit-check binding and pins) and refuses everything else, moves the quarantine lanes core does not bind and rewrites what
names them, refuses when core's library code imports a lane that would move, and changes nothing when run again."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "superseded_quarantine.py"


def _load():
    spec = importlib.util.spec_from_file_location("superseded_quarantine", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
Q = "verity_vllm.program.registry.quarantine"
QDIR = "integrations/vllm/verity_vllm/program/registry/quarantine"
REG = "integrations/vllm/verity_vllm/program/registry"
PKG = f"{M.DEST}/{M.PACKAGE}"

FILES = {
    "pyproject.toml": '[project]\nname = "w"\ndependencies = ["verity"]\n\n[tool.uv.workspace]\nmembers = ["packages/verity", '
                      '"integrations/vllm"]\n\n[tool.uv.sources]\nverity = { workspace = true }\ntorch = { index = "cpu" }\n',
    "integrations/vllm/pyproject.toml": '[project]\nname = "verity-vllm"\n\n[tool.verity.tests]\n# Test-only: fixtures.\n'
                                        'inputs = ["fixtures/artifacts.json",\n          ".gitignore"]\n',
    f"{REG}/__init__.py": "",
    f"{REG}/b1.py": ('from verity.ir.defs import composite\n\n\n@composite("Old", 1, ["N"], None)\ndef OldV1(S, x):\n    return x\n\n\n'
                     '@composite("Old", 2, ["N"], None)\ndef Old(S, x):\n    return x\n'),
    f"{REG}/vocab.py": f"from {Q}.ln import layer_norm as LN\n",
    f"{QDIR}/__init__.py": '"""the old quarantine"""\n',
    f"{QDIR}/act/__init__.py": f"from {Q}.act import silu_mul  # noqa: F401\n",
    f"{QDIR}/act/silu_mul.py": 'SEMANTIC_ID = "act/SiluMul@v2"\n',
    f"{QDIR}/collective/__init__.py": "",
    f"{QDIR}/dense/__init__.py": "",
    f"{QDIR}/ov_moe/__init__.py": "",
    f"{QDIR}/ln/__init__.py": "",
    f"{QDIR}/ln/layer_norm.py": 'SEMANTIC_ID = "ln/LayerNormAten@v1"\n',
    "integrations/vllm/tests/test_lanes.py": (f"from {Q}.act import silu_mul as S\nfrom {Q} import collective as QC\n"
                                              f"from {Q}.ln import layer_norm\nMODULE = \"{Q}.act.silu_mul\"\n"),
    "integrations/vllm/tests/program/fixtures_query.py": f"def f():\n    from {Q}.ov_moe import moe as Q\n    return Q\n",
    "integrations/vllm/tests/lint/allowlists/p11_names.json": (
        '{\n  "entries": [\n'
        '    {"file": "verity_vllm/program/registry/quarantine/act/silu_mul.py", "kind": "doc-round"},\n'
        '    {"file": "verity_vllm/program/registry/quarantine/ln/__init__.py", "kind": "doc-date"}\n  ]\n}\n'),
    "integrations/vllm/tests/dead_code_keep.json": (f'{{\n "modules": {{\n  "{Q}.ln.layer_norm": "stays",\n'
                                                    f'  "{Q}.act.silu_mul": "moves"\n }}\n}}\n'),
    "integrations/vllm/verity_vllm/engine/profiles/quarantine/REC.json": f'{{"adapter": "{Q}.act.silu_mul_difftest"}}\n',
    "tools/circuit_check/src/circuit_check/targets.py": ('def _model_roots():\n    from verity_vllm.program.registry import b1\n'
                                                         '    return [\n        # the superseded one\n'
                                                         '        ("Old_v1", lambda: bind(b1.OldV1, N=8)),\n'
                                                         '        ("Old_v2", lambda: bind(b1.Old, N=8)),\n    ]\n'),
    "tools/circuit_check/src/circuit_check/pins.json": json.dumps(
        {"definitions": {"Old_v1{N=8}": {"and": 3}, "Old_v2{N=8}": {"and": 4}}}, indent=1, sort_keys=True) + "\n",
}


def _tree(root: Path, extra: dict[str, str] | None = None) -> Path:
    for rel, text in {**FILES, **(extra or {})}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def _contents(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.parts}


def _apply(root: Path, *only: str, dest: str | None = None) -> int:
    return M.main(["apply", "--root", str(root), *(["--only", *only] if only else []), *(["--dest", dest] if dest else [])])


def test_apply_moves_the_lanes_core_does_not_bind_and_rewrites_what_names_them(tmp_path, capsys):
    root = _tree(tmp_path)
    assert _apply(root) == 0
    for lane in M.MOVE:
        assert (root / PKG / lane / "__init__.py").is_file() and not (root / QDIR / lane).exists()
    assert (root / QDIR / "ln" / "layer_norm.py").is_file()
    assert (root / QDIR / "__init__.py").read_text() == M.core_doc(M.DEST) and (root / PKG / "__init__.py").read_text() == M.PACKAGE_DOC
    assert (root / PKG / "act" / "__init__.py").read_text() == f"from {M.PACKAGE}.act import silu_mul  # noqa: F401\n"
    assert (root / "integrations/vllm/tests/test_lanes.py").read_text() == (
        f"from {M.PACKAGE}.act import silu_mul as S\nfrom {M.PACKAGE} import collective as QC\n"
        f"from {Q}.ln import layer_norm\nMODULE = \"{M.PACKAGE}.act.silu_mul\"\n")
    assert "from verity_vllm.program.registry import moe as Q" in (root / "integrations/vllm/tests/program/fixtures_query.py").read_text()

    entries = json.loads((root / "integrations/vllm/tests/lint/allowlists/p11_names.json").read_text())["entries"]
    assert [e["file"] for e in entries] == ["verity_vllm/program/registry/quarantine/ln/__init__.py"]
    assert json.loads((root / "integrations/vllm/tests/dead_code_keep.json").read_text())["modules"] == {f"{Q}.ln.layer_norm": "stays"}
    assert (root / "integrations/vllm/verity_vllm/engine/profiles/quarantine/REC.json").read_text() == FILES[
        "integrations/vllm/verity_vllm/engine/profiles/quarantine/REC.json"]

    _assert_member(root, M.DEST)

    before = _contents(root)
    capsys.readouterr()
    assert _apply(root) == 0
    assert capsys.readouterr().out.strip() == "nothing to do"
    assert _contents(root) == before


def _assert_member(root: Path, dest: str) -> None:
    import tomllib
    top = tomllib.loads((root / "pyproject.toml").read_text())
    assert dest in top["tool"]["uv"]["workspace"]["members"]
    assert M.DISTRIBUTION in top["project"]["dependencies"]
    assert top["tool"]["uv"]["sources"][M.DISTRIBUTION] == {"workspace": True}
    vllm = tomllib.loads((root / "integrations/vllm/pyproject.toml").read_text())
    assert vllm["tool"]["verity"]["tests"]["inputs"] == [dest, "fixtures/artifacts.json", ".gitignore"]
    pyproject = tomllib.loads((root / dest / "pyproject.toml").read_text())
    assert pyproject["project"]["name"] == M.DISTRIBUTION
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [M.PACKAGE]


def test_apply_puts_the_package_where_dest_says(tmp_path):
    root = _tree(tmp_path)
    assert _apply(root, dest="experimental/elsewhere") == 0
    for lane in M.MOVE:
        assert (root / "experimental/elsewhere" / M.PACKAGE / lane / "__init__.py").is_file()
    assert not (root / M.DEST).exists()
    assert (root / QDIR / "__init__.py").read_text() == M.core_doc("experimental/elsewhere")
    _assert_member(root, "experimental/elsewhere")


@pytest.mark.parametrize("dest", ["/abs/path", "experimental/../x", "experimental/./x", "integrations/vllm/q", "packages/q", ""])
def test_apply_refuses_a_dest_that_is_not_a_directory_outside_core(tmp_path, capsys, dest):
    root = _tree(tmp_path)
    before = _contents(root)
    assert M.main(["apply", "--root", str(root), "--dest", dest]) == 2
    assert "--dest" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_a_second_copy_of_the_package_elsewhere(tmp_path, capsys):
    root = _tree(tmp_path)
    assert _apply(root) == 0
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    before = _contents(root)
    assert _apply(root, dest="experimental/elsewhere") == 2
    assert f"{M.PACKAGE} already lives in {M.DEST}" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_while_core_library_code_imports_a_lane_that_would_move(tmp_path, capsys):
    root = _tree(tmp_path, {f"{REG}/uses_act.py": f"from {Q}.act import silu_mul\n"})
    before = _contents(root)
    assert _apply(root) == 2
    assert f"{REG}/uses_act.py" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_an_import_of_moved_and_staying_lanes_together(tmp_path, capsys):
    root = _tree(tmp_path, {"integrations/vllm/tests/test_mixed.py": f"from {Q} import act, ln\n"})
    before = _contents(root)
    assert _apply(root) == 2
    assert "moved and staying" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_a_version_not_in_its_list(tmp_path, capsys):
    root = _tree(tmp_path)
    before = _contents(root)
    assert _apply(root, "Old_v1") == 2
    assert "not in DELETE: Old_v1" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_a_kept_version(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path)
    monkeypatch.setattr(M, "DELETE", {"Serve_v1": M.Retirement("s", "t", "r", "verity@0")})
    assert _apply(root, "Serve_v1") == 2
    assert "both DELETE and KEPT: Serve_v1" in capsys.readouterr().err


def test_delete_removes_the_definition_its_circuit_check_binding_and_its_pins(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(M, "DELETE", {"Old_v1": M.Retirement("old-v1", "old-v2", "replaced.", "verity@abc")})
    assert _apply(root, "Old_v1") == 0
    b1 = (root / REG / "b1.py").read_text()
    assert "OldV1" not in b1 and '@composite("Old", 2' in b1
    targets = (root / "tools/circuit_check/src/circuit_check/targets.py").read_text()
    assert "Old_v1" not in targets and "superseded one" not in targets and '("Old_v2"' in targets
    assert json.loads((root / "tools/circuit_check/src/circuit_check/pins.json").read_text())["definitions"] == {"Old_v2{N=8}": {"and": 4}}


def test_delete_refuses_a_version_something_still_references(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path, {f"{REG}/hopper.py": "from verity_vllm.program.registry import b1\nPORT = b1.OldV1\n",
                            "integrations/vllm/manifests/map.json": '{"definitions": ["Old_v1"]}\n'})
    monkeypatch.setattr(M, "DELETE", {"Old_v1": M.Retirement("old-v1", "old-v2", "replaced.", "verity@abc")})
    before = _contents(root)
    assert _apply(root, "Old_v1") == 2
    err = capsys.readouterr().err
    assert f"{REG}/hopper.py:2: OldV1" in err and "integrations/vllm/manifests/map.json:1: Old_v1" in err
    assert _contents(root) == before


def test_commands_name_each_deletion_s_approach_entry(monkeypatch, capsys):
    assert M.main(["commands"]) == 0 and capsys.readouterr().out.strip() == "no deletions: no approach entries needed"
    monkeypatch.setattr(M, "DELETE", {"Old_v1": M.Retirement("old-v1", "old-v2", "Old_v2 replaces it.", "verity@abc")})
    assert M.commands()[1] == ('research notes approach circuits/old-v1 superseded --superseded-by circuits/old-v2 --by circuits '
                               '--reason "Old_v2 replaces it." --cite verity@abc   # Old_v1')


def test_verify_allows_only_the_deleted_versions_to_go():
    before = {"suite": {"A_v1{N=8}": "a", "Old_v1{N=8}": "o", "Old_v1/x": "o2"}, "registry": {"A_v1": "g", "Old_v1": "u"}, "root_errors": []}
    gone = {"suite": {"A_v1{N=8}": "a"}, "registry": {"A_v1": "g"}, "root_errors": []}
    assert M.verify(before, before, []) == []
    assert M.verify(before, gone, ["Old_v1"]) == []
    assert M.verify(before, gone, []) == ["suite: lost Old_v1/x", "suite: lost Old_v1{N=8}", "registry: lost Old_v1"]
    assert M.verify(before, before, ["Old_v1"]) == ["suite: deleted but present Old_v1/x", "suite: deleted but present Old_v1{N=8}",
                                                   "registry: deleted but present Old_v1"]
    changed = {**before, "suite": {**before["suite"], "A_v1{N=8}": "b"}}
    assert M.verify(before, changed, []) == ["suite: digest changed A_v1{N=8}: a -> b"]


def test_the_lists_are_disjoint_and_name_every_reviewed_version():
    assert not set(M.DELETE) & set(M.KEPT) and not set(M.MOVE) & set(M.STAY)
    assert set(M.KEPT) == {"RMSNormFusedCuda_v1", "SiluMul_v1", "SiluMulBf16_v1", "SiluMul_v3", "SiluMulBf16_v3", "Attention_v1",
                           "AttentionHead_v1", "AttnBlock_v1", "Serve_v1", "Serve_v2", "Serve_v3", "LayerPre_v1", "LayerPost_v1",
                           "Final_v1"}


@pytest.mark.parametrize("target, fam", [("Serve_v2{C={d0},LP=3}", "Serve_v2"), ("Attention_v6/x", "Attention_v6"), ("F_v1", "F_v1")])
def test_family(target, fam):
    assert M.family(target) == fam
