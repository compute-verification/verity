"""`tools/migrate/superseded_quarantine.py` on synthetic trees: it deletes only the versions its `DELETE` lists (with their
circuit-check binding and pins) and refuses everything else, promotes the lanes core binds into the registry (moving or merging
their modules, rewriting their importers, re-keying or dropping the lists keyed by them) and refuses whatever would leave a name of
them behind, moves the quarantine lanes core does not bind and rewrites what names them, refuses when core's library code imports
a lane that would move, and changes nothing when run again.  The promotion here is a synthetic one (`PROMOTE` below)."""
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
REAL_PROMOTED_LANES = M.PROMOTED_LANES
R = "verity_vllm.program.registry"

PROMOTE = (M.Promotion("ln/norm.py", "norm.py", edits=((f"from {Q}.ln._common import BF16\n", "BF16 = 16\n"),)),
           M.Promotion("ln/gelu.py", "act.py"))
EDITS = (("integrations/vllm/tests/test_ln.py", f"from {Q}.ln import ALL\n", "ALL = ()\n"),)


@pytest.fixture(autouse=True)
def _synthetic_promotion(monkeypatch):
    monkeypatch.setattr(M, "PROMOTE", PROMOTE)
    monkeypatch.setattr(M, "PROMOTED_IDS", ("Norm_v1", "Gelu_v1"))
    monkeypatch.setattr(M, "PROMOTED_LANES", ("ln",))
    monkeypatch.setattr(M, "promote_edits", lambda dest: EDITS)


FILES = {
    "pyproject.toml": '[project]\nname = "w"\ndependencies = ["verity"]\n\n[tool.uv.workspace]\nmembers = ["verity", '
                      '"integrations/vllm"]\n\n[tool.uv.sources]\nverity = { workspace = true }\ntorch = { index = "cpu" }\n',
    "integrations/vllm/pyproject.toml": '[project]\nname = "verity-vllm"\n\n[tool.verity.tests]\n# Test-only: fixtures.\n'
                                        'inputs = ["fixtures/artifacts.json",\n          ".gitignore"]\n',
    f"{REG}/__init__.py": "",
    f"{REG}/b1.py": ('from verity.ml.circuits.defs import composite\n\n\n@composite("Old", 1, ["N"], None)\ndef OldV1(S, x):\n    return x\n\n\n'
                     '@composite("Old", 2, ["N"], None)\ndef Old(S, x):\n    return x\n'),
    f"{REG}/vocab.py": f"from {Q}.ln import norm as LN\n",
    f"{REG}/act.py": "from verity.ml.circuits.defs import composite\n\nBF16 = 16\n\n\ndef silu(x):\n    return x\n",
    f"{QDIR}/__init__.py": '"""the old quarantine"""\n',
    f"{QDIR}/act/__init__.py": f"from {Q}.act import silu_mul  # noqa: F401\n",
    f"{QDIR}/act/silu_mul.py": 'SEMANTIC_ID = "act/SiluMul@v2"\n',
    f"{QDIR}/collective/__init__.py": "",
    f"{QDIR}/dense/__init__.py": "",
    f"{QDIR}/ov_moe/__init__.py": "",
    f"{QDIR}/ln/__init__.py": f"from {Q}.ln import gelu, norm  # noqa: F401\n\nALL = [norm.Norm, gelu.Gelu]\n",
    f"{QDIR}/ln/_common.py": "BF16 = 16\n",
    f"{QDIR}/ln/norm.py": (f'"""the norm"""\nfrom verity.ml.circuits.defs import composite\nfrom {Q}.ln._common import BF16\n\n'
                           'SEMANTIC_ID = "ln/Norm@v1"\n\n\n@composite("Norm", 1, ["N"], None)\ndef Norm(B, S, x):\n    return BF16\n'),
    f"{QDIR}/ln/gelu.py": (f'"""the erf GELU"""\nfrom verity.ml.circuits.defs import composite\nfrom {Q}.ln._common import BF16\n\n'
                           'SEMANTIC_ID = "ln/Gelu@v1"\n\n\n# one word\n@composite("Gelu", 1, [], None)\ndef Gelu(B, S, x):\n'
                           '    return BF16\n\n\n__all__ = ["Gelu"]\n'),
    "integrations/vllm/tests/test_lanes.py": (f"from {Q}.act import silu_mul as S\nfrom {Q} import collective as QC\n"
                                              f"from {Q}.ln import gelu, norm\nMODULE = \"{Q}.act.silu_mul\"\nNORM = \"{Q}.ln.norm\"\n"),
    "integrations/vllm/tests/test_ln.py": f"from {Q}.ln import ALL\nfrom {Q}.ln.gelu import Gelu  # the GELU\n",
    "integrations/vllm/tests/program/fixtures_query.py": f"def f():\n    from {Q}.ov_moe import moe as Q\n    return Q\n",
    "integrations/vllm/tests/lint/allowlists/p11_names.json": (
        '{\n  "entries": [\n'
        '    {"file": "verity_vllm/program/registry/quarantine/act/silu_mul.py", "kind": "doc-round"},\n'
        '    {"file": "verity_vllm/program/registry/quarantine/ln/__init__.py", "kind": "doc-date"}\n  ]\n}\n'),
    "integrations/vllm/tests/lint/allowlists/p10_size.json": (
        '{\n  "entries": [\n'
        '    {"file": "verity_vllm/program/registry/quarantine/ln/norm.py", "kind": "module", "count": 900},\n'
        '    {"file": "verity_vllm/program/registry/quarantine/ln/gelu.py", "kind": "module", "count": 801}\n  ]\n}\n'),
    "integrations/vllm/tests/dead_code_keep.json": (f'{{\n "modules": {{\n  "verity_vllm.protocols.linear": "kept",\n  "{Q}.ln.norm": "promoted",\n'
                                                    f'  "{Q}.act.silu_mul": "moves"\n }}\n}}\n'),
    "integrations/vllm/verity_vllm/engine/profiles/quarantine/REC.json": f'{{"adapter": "{Q}.act.silu_mul_difftest"}}\n',
    "tools/verity/circuit_check/src/circuit_check/targets.py": ('def _model_roots():\n    from verity_vllm.program.registry import b1\n'
                                                         '    return [\n        # the superseded one\n'
                                                         '        ("Old_v1", lambda: bind(b1.OldV1, N=8)),\n'
                                                         '        ("Old_v2", lambda: bind(b1.Old, N=8)),\n    ]\n'),
    "tools/verity/circuit_check/src/circuit_check/pins.json": json.dumps(
        {"definitions": {"Old_v1{N=8}": {"and": 3}, "Old_v2{N=8}": {"and": 4}}}, indent=1, sort_keys=True) + "\n",
}


def _tree(root: Path, extra: dict[str, str] | None = None) -> Path:
    for rel, text in {**FILES, **(extra or {})}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "tree"], check=True)
    return root


def _contents(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file() and ".git" not in p.parts}


def _apply(root: Path, *only: str, dest: str | None = None, promote_only: bool = False) -> int:
    return M.main(["apply", "--root", str(root), *(["--only", *only] if only else []), *(["--dest", dest] if dest else []),
                   *(["--promote-only"] if promote_only else [])])


def test_promote_only_promotes_and_moves_no_lane(tmp_path, capsys):
    root = _tree(tmp_path)
    assert _apply(root, promote_only=True) == 0
    for lane in M.MOVE:
        assert (root / QDIR / lane / "__init__.py").is_file()
    assert not (root / "experimental").exists() and not (root / QDIR / "ln").exists()
    assert (root / QDIR / "__init__.py").read_text() == FILES[f"{QDIR}/__init__.py"]
    for rel in ("pyproject.toml", "integrations/vllm/pyproject.toml", "integrations/vllm/tests/program/fixtures_query.py"):
        assert (root / rel).read_text() == FILES[rel]
    assert (root / REG / "norm.py").is_file() and (root / REG / "vocab.py").read_text() == f"from {R} import norm as LN\n"
    assert (root / "integrations/vllm/tests/test_lanes.py").read_text() == (
        f"from {Q}.act import silu_mul as S\nfrom {Q} import collective as QC\n"
        f"from {R} import act as gelu, norm\nMODULE = \"{Q}.act.silu_mul\"\nNORM = \"{R}.norm\"\n")
    entries = json.loads((root / "integrations/vllm/tests/lint/allowlists/p11_names.json").read_text())["entries"]
    assert entries == [{"file": "verity_vllm/program/registry/quarantine/act/silu_mul.py", "kind": "doc-round"}]
    assert json.loads((root / "integrations/vllm/tests/dead_code_keep.json").read_text())["modules"] == {
        "verity_vllm.protocols.linear": "kept", f"{Q}.act.silu_mul": "moves"}

    before = _contents(root)
    capsys.readouterr()
    assert _apply(root, promote_only=True) == 0
    assert capsys.readouterr().out.strip() == "nothing to do"
    assert _contents(root) == before


def test_promote_only_reads_the_unqualified_names_the_destination_binds(tmp_path, monkeypatch, capsys):
    gelu = FILES[f"{QDIR}/ln/gelu.py"].replace("from verity.ml.circuits.defs import composite\n",
                                                "from verity.ml.circuits.defs import composite\nimport prims as P\n")
    gelu = gelu.replace("    return BF16\n\n\n__all__", "    return P.BF16\n\n\n__all__")
    monkeypatch.setattr(M, "PROMOTE", (PROMOTE[0], M.Promotion("ln/gelu.py", "act.py", unqualify=("P",))))
    root = _tree(tmp_path, {f"{QDIR}/ln/gelu.py": gelu})
    assert _apply(root, promote_only=True) == 0
    assert (root / REG / "act.py").read_text().endswith("def Gelu(B, S, x):\n    return BF16\n")

    root = _tree(tmp_path / "unbound", {f"{QDIR}/ln/gelu.py": gelu.replace("P.BF16", "P.F32")})
    assert _apply(root, promote_only=True) == 2
    assert "act.py does not bind F32, which ln/gelu.py reads as P.<name>" in capsys.readouterr().err


def test_apply_promotes_and_moves_the_lanes_and_rewrites_what_names_them(tmp_path, capsys):
    root = _tree(tmp_path)
    assert _apply(root) == 0
    for lane in M.MOVE:
        assert (root / PKG / lane / "__init__.py").is_file() and not (root / QDIR / lane).exists()
    assert not (root / QDIR / "ln").exists()
    assert (root / QDIR / "__init__.py").read_text() == M.core_doc(M.DEST) and (root / PKG / "__init__.py").read_text() == M.PACKAGE_DOC
    assert (root / PKG / "act" / "__init__.py").read_text() == f"from {M.PACKAGE}.act import silu_mul  # noqa: F401\n"
    assert (root / "integrations/vllm/tests/test_lanes.py").read_text() == (
        f"from {M.PACKAGE}.act import silu_mul as S\nfrom {M.PACKAGE} import collective as QC\n"
        f"from {R} import act as gelu, norm\nMODULE = \"{M.PACKAGE}.act.silu_mul\"\nNORM = \"{R}.norm\"\n")
    assert "from verity_vllm.program.registry import moe as Q" in (root / "integrations/vllm/tests/program/fixtures_query.py").read_text()

    assert (root / REG / "norm.py").read_text() == FILES[f"{QDIR}/ln/norm.py"].replace(f"from {Q}.ln._common import BF16\n", "BF16 = 16\n")
    assert (root / REG / "act.py").read_text() == FILES[f"{REG}/act.py"] + (
        '\n\n# one word\n@composite("Gelu", 1, [], None)\ndef Gelu(B, S, x):\n    return BF16\n')
    assert (root / REG / "vocab.py").read_text() == f"from {R} import norm as LN\n"
    assert (root / "integrations/vllm/tests/test_ln.py").read_text() == f"ALL = ()\nfrom {R}.act import Gelu  # the GELU\n"
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"], check=True, capture_output=True, text=True).stdout.split()
    assert f"{REG}/norm.py" in tracked and not [t for t in tracked if t.startswith(f"{QDIR}/ln/")]

    sizes = json.loads((root / "integrations/vllm/tests/lint/allowlists/p10_size.json").read_text())["entries"]
    assert sizes == [{"file": "verity_vllm/program/registry/norm.py", "kind": "module", "count": 900}]
    entries = json.loads((root / "integrations/vllm/tests/lint/allowlists/p11_names.json").read_text())["entries"]
    assert entries == []
    assert json.loads((root / "integrations/vllm/tests/dead_code_keep.json").read_text())["modules"] == {"verity_vllm.protocols.linear": "kept"}
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


def test_apply_refuses_an_import_of_moved_and_other_lanes_together(tmp_path, capsys):
    root = _tree(tmp_path, {"integrations/vllm/tests/test_mixed.py": f"from {Q} import act, ln\n"})
    before = _contents(root)
    assert _apply(root) == 2
    assert "moved and other lanes" in capsys.readouterr().err
    assert _contents(root) == before


@pytest.mark.parametrize("extra, edits, says", [
    ({}, (), f"`from {Q}.ln import ALL`: ALL is not a promoted module of lane ln"),
    ({"integrations/vllm/tests/test_pkg.py": f"from {Q} import ln\n"}, EDITS, "imports a promoted lane's package"),
    ({f"{QDIR}/ln/extra.py": "X = 1\n"}, EDITS, f"lane ln keeps {QDIR}/ln/extra.py, which is not promoted"),
    ({"integrations/vllm/README.md": "see `registry/quarantine/ln/norm.py`\n"}, EDITS,
     "would still name a promoted lane:\n  integrations/vllm/README.md:1: see `registry/quarantine/ln/norm.py`"),
    ({f"{REG}/act.py": "from verity.ml.circuits.defs import composite\nBF16 = 16\nGelu = None\n"}, EDITS, "Gelu already bound in act.py"),
    ({f"{REG}/act.py": "from verity.ml.circuits.defs import composite\n"}, EDITS, "act.py does not bind BF16"),
    ({}, (("integrations/vllm/tests/test_ln.py", "no such text", ""),), "expected one 'no such text', found 0"),
])
def test_apply_refuses_a_promotion_that_would_leave_a_name_behind_or_clash(tmp_path, monkeypatch, capsys, extra, edits, says):
    monkeypatch.setattr(M, "promote_edits", lambda dest: edits)
    root = _tree(tmp_path, extra)
    before = _contents(root)
    assert _apply(root) == 2
    assert says in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_a_promoted_module_that_declares_an_unlisted_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(M, "PROMOTED_IDS", ("Norm_v1",))
    root = _tree(tmp_path)
    before = _contents(root)
    assert _apply(root) == 2
    assert "declare ['Gelu_v1', 'Norm_v1'], not PROMOTED_IDS ['Norm_v1']" in capsys.readouterr().err
    assert _contents(root) == before


def test_apply_refuses_a_partly_promoted_lane(tmp_path, capsys):
    root = _tree(tmp_path)
    (root / QDIR / "ln" / "gelu.py").unlink()
    assert _apply(root) == 2
    assert "partly promoted: ln/gelu.py gone" in capsys.readouterr().err


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
    targets = (root / "tools/verity/circuit_check/src/circuit_check/targets.py").read_text()
    assert "Old_v1" not in targets and "superseded one" not in targets and '("Old_v2"' in targets
    assert json.loads((root / "tools/verity/circuit_check/src/circuit_check/pins.json").read_text())["definitions"] == {"Old_v2{N=8}": {"and": 4}}


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


def test_verify_allows_new_ids_only_where_the_promoted_targets_reach():
    before = {"suite": {"A_v1{N=8}": "a"}, "registry": {"A_v1": "g", "P_v1": "p"}, "root_errors": []}
    after = {"suite": {"A_v1{N=8}": "a", "P_v1{N=4}": "q", "Const32[0x1]_v1": "c"}, "registry": {"A_v1": "g", "P_v1": "p", "Const32[0x1]_v1": "c"},
             "root_errors": [], "promoted_closure": ["Const32[0x1]_v1", "P_v1{N=4}"]}
    assert M.verify(before, after, []) == []
    assert M.verify(before, {**after, "promoted_closure": ["P_v1{N=4}"]}, []) == ["suite: new Const32[0x1]_v1", "registry: new Const32[0x1]_v1"]
    assert M.verify(before, {**after, "registry": {**after["registry"], "P_v1": "p2"}}, []) == ["registry: digest changed P_v1: p -> p2"]


def test_the_lists_are_disjoint_and_name_every_reviewed_version():
    assert not set(M.DELETE) & set(M.KEPT) and not set(M.MOVE) & set(REAL_PROMOTED_LANES)
    assert set(M.KEPT) == {"RMSNormFusedCuda_v1", "SiluMul_v1", "SiluMulBf16_v1", "SiluMul_v3", "SiluMulBf16_v3", "Attention_v1",
                           "AttentionHead_v1", "AttnBlock_v1", "Serve_v1", "Serve_v2", "Serve_v3", "LayerPre_v1", "LayerPost_v1",
                           "Final_v1"}


@pytest.mark.parametrize("target, fam", [("Serve_v2{C={d0},LP=3}", "Serve_v2"), ("Attention_v6/x", "Attention_v6"), ("F_v1", "F_v1")])
def test_family(target, fam):
    assert M.family(target) == fam


def test_add_entries_puts_each_new_entry_in_its_sorted_place_once():
    def e(f, d):
        return {"file": f, "kind": "gpu", "symbol": "<module>", "detail": d}
    text = '{\n  "rule": "P8",\n  "entries": [\n' + ",\n".join("    " + json.dumps(x) for x in (e("a.py", "h100"), e("c.py", "a100"))) + "\n  ]\n}\n"
    out = M._add_entries(text, (e("b.py", "l40s"), e("a.py", "h100")), "p08")
    assert json.loads(out)["entries"] == [e("a.py", "h100"), e("b.py", "l40s"), e("c.py", "a100")]
    assert out.splitlines()[4] == '    {"file": "b.py", "kind": "gpu", "symbol": "<module>", "detail": "l40s"},'
    assert M._add_entries(out, (e("b.py", "l40s"),), "p08") == out
    unsorted = M._add_entries(text.replace("a.py", "d.py"), (e("b.py", "l40s"),), "p08")
    assert json.loads(unsorted)["entries"] == [e("b.py", "l40s"), e("d.py", "h100"), e("c.py", "a100")]
    with pytest.raises(M.Refused, match="not one entry per line"):
        M._add_entries(text.replace(",\n    ", ", "), (), "p08")
