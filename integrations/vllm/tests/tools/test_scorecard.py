"""`tools/scorecard.py`: every number on a toy tree with one of each thing it counts, its CLI, and on the real tree its
size and settings counts agree with the P10 and P7 allowlists."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

#: the toy tree's package, written through a name so the repository's module-string scan doesn't read these strings as ours
TOY = "verity_vllm"

TOOLS = Path(__file__).resolve().parents[2] / "tools"
SCRIPT = TOOLS / "scorecard.py"
_spec = importlib.util.spec_from_file_location("scorecard", SCRIPT)
SC = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("scorecard", SC)
_spec.loader.exec_module(SC)

LIMITS = Path(__file__).resolve().parents[1] / "lint" / "allowlists"


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _toy(root: Path) -> Path:
    pkg = "integrations/vllm/verity_vllm"
    _write(root, f"{pkg}/__init__.py", '"""Root."""\nfrom verity_vllm.a import f\n__all__ = ["f"]\n')
    _write(root, f"{pkg}/a.py", '"""A."""\nimport os\n\n\ndef f(x: int) -> int:\n    """F."""\n'
                                '    return int(os.environ.get("VERITY_A", x))\n\n\nclass K:\n    def m(self, y):\n'
                                '        return y\n\n\ndef _g():\n    pass\n')
    _write(root, f"{pkg}/b/__init__.py", 'import os\nfrom os import environ\nenviron["VERITY_B"] = "1"\n'
                                         'if "VERITY_C" in environ:\n    pass\nos.environ.setdefault("VERITY_D", "0")\n')
    _write(root, f"{pkg}/settings.py", '"""Settings."""\nimport os\nX = os.environ.get("VERITY_X")\n')
    body = "".join(f"    x{i} = {i}\n" for i in range(120))
    _write(root, f"{pkg}/big.py", '"""Big."""\n\n\ndef _long():\n' + body + "".join(f"Y{i} = {i}\n" for i in range(330)))
    _write(root, f"{pkg}/tests/test_skipped.py", "import os\nos.environ['VERITY_T'] = '1'\n")
    _write(root, f"{pkg}/test_skipped.py", "import os\nos.environ['VERITY_T'] = '1'\n")
    _write(root, "integrations/vllm/tests/lint/allowlists/p99_demo.json", json.dumps({"entries": [{"a": 1}, {"b": 2}]}))
    _write(root, "integrations/vllm/tests/by_name_allowlist.json", json.dumps({"_about": "x", "entries": [{}]}))
    _write(root, "integrations/vllm/tests/fixture.json", json.dumps({"entries": [{}]}))
    _write(root, "integrations/vllm/tests/test_inside.py", "import verity_vllm.big\n")
    _write(root, "benchmarks/x.py", 'import importlib\nimport verity_vllm\nfrom verity_vllm.a import f, _g\n'
                                    f'from verity_vllm import b\nM = "{TOY}.a:f"\n'
                                    f'importlib.import_module("{TOY}.b")\n')
    _write(root, "tests/test_vllm_public_api.py", "import verity_vllm.big\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_each_number_on_a_toy_tree(tmp_path: Path) -> None:
    card = SC.scorecard(_toy(tmp_path), top=2)
    assert card["commit"] is None
    assert card["ratchets"] == {"by_name_allowlist": {"file": "tests/by_name_allowlist.json", "entries": 1},
                                "p99_demo": {"file": "tests/lint/allowlists/p99_demo.json", "entries": 2}}
    s = card["size"]
    assert (s["modules"], s["modules_over_800"], s["modules_over_600"], s["modules_over_400"]) == (5, 0, 0, 1)
    assert (s["functions_over_150"], s["functions_over_100"]) == (0, 1)
    assert s["largest"][0] == {"file": "verity_vllm/big.py", "lines": 454}
    o = card["outside_imports"]
    assert o["module_list"] == ["verity_vllm", "verity_vllm.a", "verity_vllm.b"]
    assert o["private_list"] == ["verity_vllm.a._g"]
    assert (o["modules"], o["private_names"], o["sites"], o["files"]) == (3, 1, 6, 1)
    pkg = {"modules": 5, "modules_undocumented": 1, "public_defs": 2, "undocumented": 1, "functions": 2, "untyped": 1}
    assert card["docs_types"] == {"package": pkg, "interface": {**pkg, "modules": 3}}
    e = card["settings"]
    assert e["settings_module"] == "verity_vllm/settings.py"
    assert (e["sites"], e["reads"], e["writes"], e["modules"], e["variables"]) == (4, 2, 2, 2, 4)
    assert e["by_module"] == {"verity_vllm/b/__init__.py": 3, "verity_vllm/a.py": 1}


def test_the_cli_prints_the_card_and_its_json(tmp_path: Path) -> None:
    root = _toy(tmp_path)
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *a],   # noqa: E731
                                    capture_output=True, text=True, check=True).stdout
    assert json.loads(run("--json")) == SC.scorecard(root)
    text = run()
    assert "distinct modules imported" in text and "p99_demo" in text and "verity_vllm/settings.py" in text


def test_env_uses_names_the_variable_and_the_mode() -> None:
    src = ("import os\nimport os as o\nfrom os import environ, getenv\na = os.environ['A']\nos.environ['B'] = '1'\n"
           "del os.environ['C']\nd = getenv('D')\ne = o.environ.get('E')\nos.environ.update({})\nf = dict(environ)\n"
           "os.putenv('G', '1')\n")
    assert [(v, m) for v, m, _ in SC.env_uses(src)] == [
        ("A", "read"), ("B", "write"), ("C", "write"), ("D", "read"), ("E", "read"), ("os.environ.update", "write"),
        ("os.environ", "read"), ("G", "write")]


def _p07_env_owners() -> frozenset[str]:
    """P7's ``ENV_OWNERS``, the modules it lets read the environment, as its source states them."""
    tree = ast.parse((LIMITS.parent / "test_p07_declared_inputs.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "ENV_OWNERS":
            return frozenset(ast.literal_eval(node.value.args[0]))
    raise AssertionError("tests/lint/test_p07_declared_inputs.py states no ENV_OWNERS")


def _package_entries(name: str) -> list[dict]:
    """An allowlist's entries in the package: the lints also hold the Definitions that moved to ``verity/ml/definitions/``,
    which the scorecard doesn't count."""
    entries = json.loads((LIMITS / f"{name}.json").read_text(encoding="utf-8"))["entries"]
    return [e for e in entries if e["file"].startswith(f"{SC.PACKAGE}/")]


@pytest.mark.slow
def test_the_real_tree_agrees_with_the_p10_and_p7_allowlists() -> None:
    """On the real package (about 6 s): the scorecard's size bands and environment uses are the numbers the P10 and P7
    ratchets hold for it, so a scorecard that drifted from what the lints enforce fails here, which the toy tree can't
    show."""
    card = SC.scorecard(TOOLS.parents[2], outside=False)
    p10 = _package_entries("p10_size")
    assert card["size"]["modules_over_800"] == sum(e["kind"] == "module" for e in p10)
    assert card["size"]["functions_over_150"] == sum(e["kind"] == "function" for e in p10)
    p07 = _package_entries("p07_declared_inputs")
    owners = _p07_env_owners()
    by_module = card["settings"]["by_module"]
    assert card["settings"]["sites"] - sum(by_module.get(f, 0) for f in owners) == \
        sum(e.get("count", 1) for e in p07 if e["kind"] == "environ")
