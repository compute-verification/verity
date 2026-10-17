"""`tools/check/move_check.py`: which commit ranges it certifies as pure moves, on tiny git repositories."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GIT = shutil.which("git")
spec = importlib.util.spec_from_file_location("check_move_check", ROOT / "tools" / "verity" / "check" / "move_check.py")
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}      # not this machine's git config
PYPROJECT = 'name = "pous"\n[tool.verity.tests]\ninputs = ["protocols/pous/data", "census/hardware.json"]\n'
LAKEFILE = '[[require]]\nname = "verity"\npath = "../../../packages/verity/lean"\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=ENV).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for p, text in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _move(repo: Path, old: str, new: str) -> None:
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "mv", old, new)


def _check(repo: Path, base: str, extra: dict[str, str] | None = None) -> dict:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "move")
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), extra, cwd=str(repo))


BASE_FILES = {
    "protocols/pous/src/verity_pous/__init__.py": "",
    "protocols/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 1\n",
    "protocols/pous/lean/Pous.lean": "theorem t : 1 = 1 := rfl\n",
    "protocols/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abc"}}\n',
    "protocols/pouw/src/verity_pouw/x.py": "X = 1\n",
    "tests/test_pous.py": "from verity_pous.codec import encode\nimport verity_pous.codec as c\n\n\ndef test():\n    assert encode(1) == 2\n",
}


def test_pure_rename_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert len(r["renames"]) == 4 and not r["edits"]
    # the map is the moved directory, not `protocols`, since `protocols/pouw` stayed
    assert r["map"] == {"protocols/pous": "security_proofs/pous"}


def test_rename_with_a_one_byte_edit_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 2\n"})
    r = _check(repo, base)
    assert not r["certified"]
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/src/verity_pous/codec.py" and f["old"] == "protocols/pous/src/verity_pous/codec.py"
    assert "-    return x + 1" in f["hunk"] and "+    return x + 2" in f["hunk"]


def test_pyproject_path_edit_is_certified(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT,
                                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml": PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data"),
                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["security_proofs/pous", "protocols/pouw"]\n'})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"pyproject.toml", "security_proofs/pous/pyproject.toml"}


def test_path_edit_to_a_directory_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "pyproject.toml": 'members = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"pyproject.toml": 'members = ["security_proofs/pous", "security_proofs/pouw"]\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "pyproject.toml" and "security_proofs/pouw" in f["hunk"]


def test_non_path_edit_in_pyproject_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml":
                  PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data").replace('"pous"', '"pous2"')})
    r = _check(repo, base)
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/pyproject.toml" and f["reason"] == "edit beyond path strings"
    assert f["hunk"].splitlines()[1:] == ['-name = "pous"', '+name = "pous2"']


def test_import_line_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous", "pous")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "tests/test_pous.py", "kind": "import"}]


def test_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "pous.other")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tests/test_pous.py" and f["reason"] == "edit beyond import lines"


def test_a_new_file_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/NEW.md": "hello\n"})
    [f] = _check(repo, base)["refusals"]
    assert (f["path"], f["status"], f["reason"]) == ("security_proofs/pous/NEW.md", "A", "added")


def test_names_with_spaces_and_odd_characters(tmp_path):
    files = {"old dir/a file.txt": "one\n", "old dir/tab\there.txt": "two\n", 'old dir/quo"te.txt': "three\n"}
    repo, base = _repo(tmp_path, files)
    _move(repo, "old dir", "new dir")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert sorted(x["new"] for x in r["renames"]) == ["new dir/a file.txt", 'new dir/quo"te.txt', "new dir/tab\there.txt"]
    assert r["map"] == {"old dir": "new dir"}


def test_lean_package_move_with_relative_require(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "packages/verity/lean", "verity/lean")
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lakefile.toml": LAKEFILE.replace("../../../packages/verity/lean", "../../verity/lean")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "security_proofs/pous/lakefile.toml", "old": "protocols/pous/lean/lakefile.toml", "kind": "path"}]


def test_byte_identical_lakefile_whose_relative_require_broke_is_refused(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    [f] = _check(repo, base)["refusals"]
    assert f["reason"] == "relative path now resolves elsewhere"


def test_lean_audit_pin_change_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abd"}}\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "security_proofs/pous/lean/lean-audit.json"


def test_explicit_map_and_cli(tmp_path, capsys, monkeypatch):
    repo, base = _repo(tmp_path, {**BASE_FILES, "AGENTS.md": "See `protocols/pous/lean/Pous.lean`.\n"})
    _write(repo, {"AGENTS.md": "See `elsewhere/Pous.lean`.\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "doc")
    monkeypatch.chdir(repo)
    assert MC.main([base, "HEAD", "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["certified"]
    assert MC.main([base, "HEAD", "--map", "protocols/pous/lean=elsewhere"]) == 0
    assert capsys.readouterr().out.startswith("move-check: certified")


SPEC = "packages/verity/lean"
PROOFS = "security_proofs/core"
LAKEFILE_SPEC = ('name = "verity"\ndefaultTargets = ["Verity"]\n\n[[require]]\nname = "mathlib"\n'
                 'git = "https://github.com/leanprover-community/mathlib4"\nrev = "abc"\n\n# Verity core\'s Lean.\n'
                 '[[lean_lib]]\nname = "Verity"\n')
LAKEFILE_PROOFS = ('name = "coreproofs"\ndefaultTargets = ["CoreProofs"]\n\n[[require]]\nname = "mathlib"\n'
                   'git = "https://github.com/leanprover-community/mathlib4"\nrev = "abc"\n\n[[require]]\nname = "verity"\n'
                   'path = "../../packages/verity/lean"\n\n# The proofs of core\'s guarantees.\n[[lean_lib]]\nname = "CoreProofs"\n'
                   'globs = ["CoreProofs.+"]\n')
MATHLIB = {"url": "https://github.com/leanprover-community/mathlib4", "type": "git", "name": "mathlib", "rev": "abc",
           "inherited": False, "configFile": "lakefile.lean"}
VERITY_PATH = {"type": "path", "scope": "", "name": "verity", "manifestFile": "lake-manifest.json", "inherited": False,
               "dir": "../../packages/verity/lean", "configFile": "lakefile.toml"}
PINS = {"Verity.SecurityProofs.G": {"signature": ["Verity.SecurityProofs.G : Verity.G"], "assumptions": [], "type_hash": "aa"}}
READS = {"Verity.Guarantees": {"digest": "dd", "definitions": {"Verity.G": "ee"}}}
FOO = "import Specs.Core.Guarantees\n\nnamespace Proofs.Core\n\ntheorem G : Verity.G := rfl\n\nend Proofs.Core\n"
BAR = ("import Verity.SecurityProofs.Foo\nimport Mathlib.Tactic\n\nnamespace Proofs.Core\n\n"
       "theorem H : 1 = 1 := rfl\n\nend Verity.SecurityProofs\n")


def _manifest(name: str, *packages: dict) -> str:
    return json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": list(packages), "name": name}, indent=1)


def _lock(**fields) -> str:
    return json.dumps(fields, indent=2) + "\n"


SPEC_FILES = {
    f"{SPEC}/lakefile.toml": LAKEFILE_SPEC, f"{SPEC}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
    f"{SPEC}/lake-manifest.json": _manifest("verity", MATHLIB),
    f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS),
    f"{SPEC}/Verity.lean": "import Definitions.Core\nimport Specs.Core.Guarantees\nimport Proofs.Core\n",
    f"{SPEC}/Verity/Protocol.lean": "def Verity.p := 1\n",
    f"{SPEC}/Verity/Guarantees.lean": "import Definitions.Core\n\ndef Verity.G : Prop := Verity.p = 1\n",
    f"{SPEC}/Verity/SecurityProofs.lean": "import Verity.SecurityProofs.Foo\nimport Verity.SecurityProofs.Bar\n",
    f"{SPEC}/Verity/SecurityProofs/Foo.lean": FOO,
    f"{SPEC}/Verity/SecurityProofs/Bar.lean": BAR,
}


def _split(tmp_path: Path, bar: str | None = None, pins: dict | None = None, files: dict | None = None,
           base: dict | None = None) -> tuple[Path, str, dict]:
    """Core's proofs split out to `security_proofs/core` under the module root `CoreProofs`; the spec and its umbrella stay."""
    repo, base = _repo(tmp_path, {**SPEC_FILES, **(base or {})})
    for m in ("Foo", "Bar"):
        _move(repo, f"{SPEC}/Verity/SecurityProofs/{m}.lean", f"{PROOFS}/CoreProofs/{m}.lean")
    split = {
        f"{PROOFS}/CoreProofs/Bar.lean": bar if bar is not None else BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo"),
        f"{SPEC}/Verity/SecurityProofs.lean": "import CoreProofs.Foo\nimport CoreProofs.Bar\n",
        f"{SPEC}/lean-audit.json": _lock(roots=["Verity.Protocol", "Verity.Guarantees", "Verity.SecurityProofs"],
                                         layers={"Verity.Protocol": []}, pins=pins or PINS, reads=READS,
                                         proved_in=["../../../security_proofs/core"]),
        f"{PROOFS}/lakefile.toml": LAKEFILE_PROOFS, f"{PROOFS}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        f"{PROOFS}/lake-manifest.json": _manifest("coreproofs", VERITY_PATH, MATHLIB),
        f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], layers={"Verity.Protocol": []}),
    }
    _write(repo, {**split, **(files or {})})
    return repo, base, _check(repo, base)


def test_lean_split_is_certified(tmp_path):
    _, _, r = _split(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"Verity.SecurityProofs.Bar": "CoreProofs.Bar", "Verity.SecurityProofs.Foo": "CoreProofs.Foo"}
    assert r["renames"] == [{"old": f"{SPEC}/Verity/SecurityProofs/Foo.lean", "new": f"{PROOFS}/CoreProofs/Foo.lean"}]
    kinds = {e["path"]: e["kind"] for e in r["edits"]}
    assert kinds == {f"{PROOFS}/CoreProofs/Bar.lean": "lean", f"{SPEC}/Verity/SecurityProofs.lean": "lean",
                     f"{SPEC}/lean-audit.json": "lock", **{f"{PROOFS}/{n}": "copy" for n in sorted(MC.LAKE_COPIES)
                                                             if n != "lakefile.lean"}}


def test_module_names_follow_src_dir(tmp_path):
    """A nested package with `srcDir = ".."` names its modules from the directory above it, as Lake does."""
    repo, base = _repo(tmp_path, {"verity/Security/lakefile.toml": 'name = "security"\n[[lean_lib]]\nname = "Spec"\n',
                                  "verity/Security/Spec/Pouw.lean": "",
                                  "verity/Security/Proofs/lakefile.toml": 'name = "proofs"\nsrcDir = ".."\n',
                                  "verity/Security/Proofs/Pouw/End.lean": "",
                                  "p/lakefile.toml": '[[lean_lib]]\nname = "P"\nsrcDir = "src"\n', "p/src/P/A.lean": "",
                                  "p/Check.lean": ""})
    tree = MC.Tree(base, str(repo))
    assert tree.module("verity/Security/Spec/Pouw.lean") == "Spec.Pouw"
    assert tree.module("verity/Security/Proofs/Pouw/End.lean") == "Proofs.Pouw.End"
    assert tree.module("p/src/P/A.lean") == "P.A" and tree.module("p/Check.lean") == "Check"


def test_lean_split_that_changes_a_proof_line_is_refused(tmp_path):
    bar = BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo").replace(":= rfl", ":= by rfl")
    [f] = _split(tmp_path, bar=bar)[2]["refusals"]
    assert f["path"] == f"{PROOFS}/CoreProofs/Bar.lean" and "+theorem H : 1 = 1 := by rfl" in f["hunk"]


def test_lean_split_that_changes_a_guarantee_record_is_refused(tmp_path):
    pins = {"Verity.SecurityProofs.G": {**PINS["Verity.SecurityProofs.G"], "type_hash": "ab"}}
    [f] = _split(tmp_path, pins=pins)[2]["refusals"]
    assert f["path"] == f"{SPEC}/lean-audit.json" and f["reason"].startswith("`pins` changed")
    assert '"type_hash": "aa"' in f["hunk"] and '"type_hash": "ab"' in f["hunk"]


def test_lean_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    bar = BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Other")
    [f] = _split(tmp_path, bar=bar)[2]["refusals"]
    assert f["path"] == f"{PROOFS}/CoreProofs/Bar.lean" and "+import CoreProofs.Other" in f["hunk"]


def test_proofs_package_lakefile_with_a_new_git_dependency_is_refused(tmp_path):
    lakefile = LAKEFILE_PROOFS + '\n[[require]]\nname = "extra"\ngit = "https://example.com/extra"\n'
    [f] = _split(tmp_path, files={f"{PROOFS}/lakefile.toml": lakefile})[2]["refusals"]
    assert f["path"] == f"{PROOFS}/lakefile.toml" and '+git = "https://example.com/extra"' in f["hunk"]


def test_proofs_package_lock_that_lists_guarantees_is_refused(tmp_path):
    [f] = _split(tmp_path, files={f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], pins=PINS)})[2]["refusals"]
    assert f["path"] == f"{PROOFS}/lean-audit.json" and "lists no guarantees" in f["reason"]


def test_spec_lock_proved_in_that_names_no_package_is_refused(tmp_path):
    repo, base = _repo(tmp_path, SPEC_FILES)
    _write(repo, {f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS,
                                                   proved_in=["../../security_proofs/core"])})
    [f] = _check(repo, base)["refusals"]
    assert "`proved_in` entry" in f["reason"]


#: the numerical bench's lowerings, a registry move-check reads wherever the tree keeps it
LOWERINGS = next(p for p in MC.PY_PATH_FILES if p.endswith("/bench/lowerings.py"))
REGISTRY = 'REGISTRY = {\n    "flock_pure": "backends.flock.tool:FLOCK_PURE",\n    "pouw_vllm": "benchmarks.pouw.tool:POUW_VLLM",\n}\n'
REGISTRY_FILES = {"backends/flock/tool.py": "FLOCK_PURE = 1\n", "backends/flock/prove.py": "",
                  "benchmarks/pouw/tool.py": "POUW_VLLM = 1\n",
                  "tools/research/src/research/store/tools_registry.py": REGISTRY,
                  LOWERINGS:
                      'DIRECTORIES = {\n    "C-Flock": REPO / "backends" / "flock" / "templates",\n}\n',
                  "tools/research/src/research/store/parity.py":
                      'argv = [sys.executable, "-m", f"backends.flock.{mod}", "--root", str(root)]\n'}


def test_tools_registry_dotted_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, REGISTRY_FILES)
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/store/tools_registry.py": REGISTRY.replace("backends.flock", "catalog.flock"),
                  LOWERINGS:
                      'DIRECTORIES = {\n    "C-Flock": REPO / "catalog" / "flock" / "templates",\n}\n',
                  "tools/research/src/research/store/parity.py":
                      'argv = [sys.executable, "-m", f"catalog.flock.{mod}", "--root", str(root)]\n'})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"tools/research/src/research/store/tools_registry.py",
                                               LOWERINGS,
                                               "tools/research/src/research/store/parity.py"}


def test_tools_registry_non_path_change_is_refused(tmp_path):
    repo, base = _repo(tmp_path, REGISTRY_FILES)
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/store/tools_registry.py":
                  REGISTRY.replace("backends.flock", "catalog.flock").replace("FLOCK_PURE", "FLOCK_FAST")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tools/research/src/research/store/tools_registry.py" and "FLOCK_FAST" in f["hunk"]


def test_module_argument_outside_a_dash_m_line_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**REGISTRY_FILES, "tools/research/src/research/x.py": 'NAME = "backends.flock.tool"\n'})
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/x.py": 'NAME = "catalog.flock.tool"\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tools/research/src/research/x.py"


WORKSPACE = '[tool.uv.workspace]\nmembers = ["protocols/pous", "protocols/pouw"]\n\n[tool.uv.sources]\npous = { workspace = true }\n'


def _new_member(tmp_path: Path, members: str, sources: str = "") -> dict:
    repo, base = _repo(tmp_path, {**BASE_FILES, "pyproject.toml": WORKSPACE})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"pyproject.toml": WORKSPACE.replace('"protocols/pous"', '"security_proofs/pous"')
                  .replace('"protocols/pouw"]', members + "]") + sources})
    return _check(repo, base)


def test_workspace_member_for_a_new_package_is_certified(tmp_path):
    r = _new_member(tmp_path, '"protocols/pouw", "security_proofs/pous/src"')
    assert r["certified"], r["refusals"]


def test_workspace_member_for_an_old_or_missing_directory_is_refused(tmp_path):
    for i, extra in enumerate(('"protocols/pouw", "tests"', '"protocols/pouw", "nowhere"')):
        [f] = _new_member(tmp_path / str(i), extra)["refusals"]
        assert f["path"] == "pyproject.toml"


def test_uv_source_for_a_package_not_created_by_the_range_is_refused(tmp_path):
    [f] = _new_member(tmp_path, '"protocols/pouw"', "other = { workspace = true }\n")["refusals"]
    assert f["path"] == "pyproject.toml" and "+other = { workspace = true }" in f["hunk"]


LEAN_PACKAGES = 'PACKAGES = [\n    ROOT / "packages" / "verity" / "lean",\n]\n'


def test_lean_packages_entry_for_a_new_lake_package_is_certified(tmp_path):
    entry = '    ROOT / "security_proofs" / "core",\n'
    _, _, r = _split(tmp_path, base={"tests/test_lean_packages.py": LEAN_PACKAGES},
                     files={"tests/test_lean_packages.py": LEAN_PACKAGES.replace("]\n", entry + "]\n")})
    assert r["certified"], r["refusals"]
    assert {"path": "tests/test_lean_packages.py", "kind": "path"} in r["edits"]


def test_lean_packages_entry_for_an_existing_package_is_refused(tmp_path):
    entry = '    ROOT / "packages" / "verity" / "lean",\n'
    _, _, r = _split(tmp_path, base={"tests/test_lean_packages.py": LEAN_PACKAGES},
                     files={"tests/test_lean_packages.py": LEAN_PACKAGES.replace("]\n", entry + "]\n")})
    [f] = r["refusals"]
    assert f["path"] == "tests/test_lean_packages.py"


RUN = {"why": "random instances", "args": ["{difftest/instances.jsonl}", "10000"],
       "data": {"difftest/instances.jsonl": "c048"}, "generate": ["python3", "scripts/difftest_vectors.py", "{repo}", "--out", "{out}"]}


def _runs_split(tmp_path: Path, proofs_runs: dict | None, spec_runs: dict | None = None) -> dict:
    """The split, with core's lock holding a difftest whose runner module and generator move to the proofs package."""
    base_files = {**SPEC_FILES, f"{SPEC}/Verity/SecurityProofs/Difftest.lean": "def main : IO Unit := pure ()\n",
                  f"{SPEC}/scripts/difftest_vectors.py": "print(1)\n",
                  f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS,
                                                   runs={"Verity.SecurityProofs.Difftest": RUN})}
    repo, base = _repo(tmp_path, base_files)
    for m in ("Foo", "Bar", "Difftest"):
        _move(repo, f"{SPEC}/Verity/SecurityProofs/{m}.lean", f"{PROOFS}/CoreProofs/{m}.lean")
    _move(repo, f"{SPEC}/scripts", f"{PROOFS}/scripts")
    spec_lock = dict(roots=["Verity.Protocol", "Verity.Guarantees", "Verity.SecurityProofs"], layers={"Verity.Protocol": []},
                     pins=PINS, reads=READS, proved_in=["../../../security_proofs/core"])
    _write(repo, {
        f"{PROOFS}/CoreProofs/Bar.lean": BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo"),
        f"{SPEC}/Verity/SecurityProofs.lean": "import CoreProofs.Foo\nimport CoreProofs.Bar\n",
        f"{SPEC}/lean-audit.json": _lock(**spec_lock, **({"runs": spec_runs} if spec_runs is not None else {})),
        f"{PROOFS}/lakefile.toml": LAKEFILE_PROOFS, f"{PROOFS}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        f"{PROOFS}/lake-manifest.json": _manifest("coreproofs", VERITY_PATH, MATHLIB),
        f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], layers={"Verity.Protocol": []},
                                           **({"runs": proofs_runs} if proofs_runs is not None else {})),
    })
    return _check(repo, base)


def test_runs_entry_moved_to_the_proofs_lock_is_certified(tmp_path):
    r = _runs_split(tmp_path, {"CoreProofs.Difftest": RUN})
    assert r["certified"], r["refusals"]


def test_runs_entry_that_vanishes_is_refused(tmp_path):
    [f] = _runs_split(tmp_path, None)["refusals"]
    assert f["path"] == f"{SPEC}/lean-audit.json" and "is in no lock at HEAD" in f["reason"]


def test_runs_entry_with_a_changed_data_hash_is_refused(tmp_path):
    [f] = _runs_split(tmp_path, {"CoreProofs.Difftest": {**RUN, "data": {"difftest/instances.jsonl": "c049"}}})["refusals"]
    assert f["path"] == f"{PROOFS}/lean-audit.json" and "changed beyond moved paths" in f["reason"] and "c049" in f["hunk"]


def test_runs_entry_duplicated_into_two_locks_is_refused(tmp_path):
    r = _runs_split(tmp_path, {"CoreProofs.Difftest": RUN}, spec_runs={"CoreProofs.Difftest": RUN})
    assert any("locks at HEAD" in f["reason"] for f in r["refusals"])
