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


def test_path_after_a_checkout_variable_is_certified(tmp_path):
    """`tools/move/layout.py` rewrites a moved path after `$REPO/` or `${ROOT}/` (a shell script's checkout), so the check
    reads it from the root there too; a moved path inside a longer one is still not the move's."""
    run = 'SRC=$REPO/protocols/pous/src:${ROOT}/protocols/pous\nOUT=$W/x/protocols/pous\n'
    repo, base = _repo(tmp_path, {**BASE_FILES, "run.sh": run})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"run.sh": run.replace("$REPO/protocols", "$REPO/security_proofs").replace("}/protocols", "}/security_proofs")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"run.sh": run.replace("protocols/pous", "security_proofs/pous")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "run.sh" and "+OUT=$W/x/security_proofs/pous" in f["hunk"]


def test_non_path_edit_in_pyproject_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml":
                  PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data").replace('"pous"', '"pous2"')})
    r = _check(repo, base)
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/pyproject.toml" and f["reason"] == "edit beyond moved paths and module names"
    assert f["hunk"].splitlines()[1:] == ['-name = "pous"', '+name = "pous2"']


def test_import_line_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous", "pous")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "tests/test_pous.py", "kind": "python"}]


def test_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "pous.other")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tests/test_pous.py" and f["reason"] == "edit beyond moved paths and module names"


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


#: the numerical bench's lowerings, which name a directory in a pathlib chain
LOWERINGS = "benchmarks/numerical/python/verity_numerical/bench/lowerings.py"
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


def test_dotted_name_no_rename_explains_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**REGISTRY_FILES, "tools/research/src/research/x.py": 'NAME = "backends.flock.tool"\n'})
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/x.py": 'NAME = "catalog.other.tool"\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tools/research/src/research/x.py" and "catalog.other.tool" in f["hunk"]


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
    assert {"path": "tests/test_lean_packages.py", "kind": "python"} in r["edits"]


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


WORD_MAP = ('[words]\n"Models/Core" = "Definitions/Core"\n"Models.Core" = "Definitions.Core"\n"Trusted" = "Checked"\n'
            '"S/Models/" = "S/Definitions/"\n\'"Models" / "Core"\' = \'"Definitions" / "Core"\'\n'
            "'Models\\.Core' = 'Definitions\\.Core'\n'\"Models\" / \"Hash\"' = '\"Definitions\" / \"Hash\"'\n"
            '\n[rewrite]\nexclude = []\n')
WORD_BASE = {
    "tools/move/demo_map.toml": WORD_MAP,
    "S/lakefile.toml": 'name = "s"\n\n[[lean_lib]]\nname = "Models"\n',
    "S/Models/Core/A.lean": "import Models.Core.B\n\n/-! Trusted: `Models/Core/B.lean` states `Models.Core.B`. -/\n",
    "S/Models/Core/B.lean": "def x := 1\n",
    "S/lean-audit.json": _lock(roots=["Models"], layers={"Models.Core": {"imports": ["Models.Core"]}},
                               reads={"Models.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}}, dependencies={"m": "1"}),
    "notes.md": "The core's statements are `S/Models/Core/` (`Models.Core`). Trusted.\n",
    "more.md": "`S/Models/` states them.\n",
    "t.py": 'CORE = ROOT / "S" / "Models" / "Core"\nHASH = ROOT / "S" / "Models" / "Hash"\nB = r"Models\\.Core\\.B"\n',
    "p/TRUSTED.sha256": "",
    "tools/research/src/research/store/moves.json": json.dumps({"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}]},
                                                               indent=2) + "\n",
}


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _worded(tmp_path: Path, base_pins: str | None = None, base_extra: dict | None = None, **over: str) -> dict:
    """`Models/Core/` moved to `Definitions/Core/` with the map's words applied, as `tools/move/rename.py` does it."""
    base_files = {**WORD_BASE, **(base_extra or {}),
                  "p/TRUSTED.sha256": base_pins or f"{_sha(WORD_BASE['S/Models/Core/A.lean'])}  ../S/Models/Core/A.lean\n"}
    repo, base = _repo(tmp_path, base_files)
    for m in ("A", "B"):
        _move(repo, f"S/Models/Core/{m}.lean", f"S/Definitions/Core/{m}.lean")
    a = "import Definitions.Core.B\n\n/-! Trusted: `Definitions/Core/B.lean` states `Definitions.Core.B`. -/\n"
    head = {
        "S/Definitions/Core/A.lean": a,
        "S/lean-audit.json": _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                                   reads={"Definitions.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}},
                                   dependencies={"m": "2"}),
        "notes.md": "The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Trusted.\n",
        "more.md": "`S/Definitions/` states them.\n",
        "t.py": 'CORE = ROOT / "S" / "Definitions" / "Core"\nHASH = ROOT / "S" / "Models" / "Hash"\nB = r"Definitions\\.Core\\.B"\n',
        "p/TRUSTED.sha256": f"{_sha(a)}  ../S/Definitions/Core/A.lean\n",
        "tools/research/src/research/store/moves.json": json.dumps(
            {"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}, {"paths": {"S/Models/": "S/Definitions/"}}]},
            indent=2) + "\n",
    }
    _write(repo, {**head, **over})
    return _check(repo, base)


def test_a_word_map_move_is_certified(tmp_path):
    """What `tools/move/rename.py` writes: the map's words in any text file, a pin's digest following its file, the record of
    the move appended to `moves.json`, a lock's `reads` keyed by the moved module and its `dependencies` record."""
    r = _worded(tmp_path)
    assert r["certified"], r["refusals"]
    kinds = {e["path"]: e["kind"] for e in r["edits"]}
    assert kinds["notes.md"] == kinds["more.md"] == kinds["S/Definitions/Core/A.lean"] == kinds["t.py"] == "words"
    assert kinds["p/TRUSTED.sha256"] == "text"


def test_a_list_entry_the_words_repeat_may_go(tmp_path):
    """Two directories the range merged, both listed: the words list the one twice, and the repeat may go; any other entry
    stays."""
    old = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Models/Core/",\n    "S/Proofs/"\n  ]\n}\n'
    once = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Proofs/"\n  ]\n}\n'
    (tmp_path / "ok").mkdir()
    r = _worded(tmp_path / "ok", base_extra={"owners.json": old}, **{"owners.json": once})
    assert r["certified"], r["refusals"]
    (tmp_path / "dropped").mkdir()
    dropped = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Definitions/Core/"\n  ]\n}\n'
    assert not _worded(tmp_path / "dropped", base_extra={"owners.json": old}, **{"owners.json": dropped})["certified"]


def test_a_chain_or_regex_spelling_is_kept_only_with_the_word_it_spells(tmp_path):
    """`"Models" / "Hash"` spells `Models/Hash`, which no kept word renames (nothing under `Models/Hash` moved), so the map
    doesn't explain renaming it."""
    base = {"t.py": 'CORE = S / "Models" / "Core"\nHASH = S / "Models" / "Hash"\nB = r"Models\\.Core\\.B"\n'}
    kept = 'CORE = S / "Definitions" / "Core"\nHASH = S / "Models" / "Hash"\nB = r"Definitions\\.Core\\.B"\n'
    (tmp_path / "kept").mkdir()
    r = _worded(tmp_path / "kept", base_extra=base, **{"t.py": kept})
    assert r["certified"], r["refusals"]
    (tmp_path / "hash").mkdir()
    [f] = _worded(tmp_path / "hash", base_extra=base, **{"t.py": kept.replace('"Models" / "Hash"', '"Definitions" / "Hash"')})["refusals"]
    assert f["path"] == "t.py"


def test_a_word_that_renames_nothing_the_range_moved_is_refused(tmp_path):
    """`Trusted` is in the map, but no path or module the range moved has it, so the map doesn't explain renaming it."""
    [f] = _worded(tmp_path, **{"notes.md": "The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Checked.\n"})["refusals"]
    assert f["path"] == "notes.md" and "+The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Checked." in f["hunk"]


DECL_BASE = {
    "tools/move/sample_map.toml": '[words]\n"DrawIsCoin" = "SampleIsCoin"\n\n[[words_in]]\nfiles = ["P/S/Law.lean", '
                                  '"P/S/Use.lean"]\nwords = { "escape" = "acceptProb" }\n\n[rewrite]\n'
                                  'exclude = ["tools/move/**", "**/CHANGELOG.md"]\n',
    "P/lakefile.toml": 'name = "p"\n',
    "P/S/Law.lean": "/-- The coin is a draw. -/\ndef DrawIsCoin : Prop := True\n\n/-- An escape. -/\ndef escape : Nat := 1\n",
    "P/S/Use.lean": "import S.Law\n\ntheorem t : DrawIsCoin := trivial\ntheorem u : escape = 1 := rfl\n",
    "P/lean-audit.json": _lock(assumptions={"DrawIsCoin": {"owner": "@lean"}},
                               reads={"S.Law": {"digest": "dd", "definitions": {"DrawIsCoin": "ee", "escape": "ff"}}}),
    "notes.md": "`DrawIsCoin` holds; an escape is rare.\n",
    "CHANGELOG.md": "Added `DrawIsCoin`.\n",
}
DECL_HEAD = {
    "P/S/Law.lean": "/-- The coin is a draw. -/\ndef SampleIsCoin : Prop := True\n\n/-- An acceptProb. -/\ndef acceptProb : Nat := 1\n",
    "P/S/Use.lean": "import S.Law\n\ntheorem t : SampleIsCoin := trivial\ntheorem u : acceptProb = 1 := rfl\n",
    "P/lean-audit.json": _lock(assumptions={"SampleIsCoin": {"owner": "@lean"}},
                               reads={"S.Law": {"digest": "dd", "definitions": {"DrawIsCoin": "ee", "escape": "ff"}}}),
    "notes.md": "`SampleIsCoin` holds; an escape is rare.\n",
}


def _declared(tmp_path: Path, base_extra: dict | None = None, **over: str) -> dict:
    """`DrawIsCoin` and the scoped `escape` renamed as `tools/move/rename.py` renames them: no path moves."""
    repo, base = _repo(tmp_path, {**DECL_BASE, **(base_extra or {})})
    _write(repo, {**DECL_HEAD, **over})
    return _check(repo, base)


def test_a_lean_declaration_rename_is_certified(tmp_path):
    """A word that renames a declaration of an edited `.lean` file to a name the base never spells is kept, as is a
    `[[words_in]]` entry's in its files; a lock's records keep the old names for `check.py --update --moved`."""
    r = _declared(tmp_path)
    assert r["certified"], r["refusals"]
    assert {e["path"]: e["kind"] for e in r["edits"]} == dict.fromkeys(["P/S/Law.lean", "P/S/Use.lean", "P/lean-audit.json",
                                                                        "notes.md"], "words")


def test_a_declaration_rename_onto_a_name_the_base_spells_is_refused(tmp_path):
    """`SampleIsCoin` is already a name at the base, so renaming `DrawIsCoin` to it would merge two names."""
    r = _declared(tmp_path, base_extra={"other.md": "`SampleIsCoin` is another premise.\n"})
    assert {f["path"] for f in r["refusals"]} == {"P/S/Law.lean", "P/S/Use.lean", "P/lean-audit.json", "notes.md"}


def test_a_scoped_word_renames_only_in_its_files(tmp_path):
    """`escape` is renamed only in the files its `[[words_in]]` entry names; a prose `escape` elsewhere stays."""
    [f] = _declared(tmp_path, **{"notes.md": "`SampleIsCoin` holds; an acceptProb is rare.\n"})["refusals"]
    assert f["path"] == "notes.md"


def test_a_stale_word_of_another_map_is_not_kept(tmp_path):
    """An earlier move's map also renames `DrawIsCoin`, to a name the base no longer spells; the range didn't make that
    rename, so its word is not applied."""
    stale = '[words]\n"DrawIsCoin" = "CoinIsDrawn"\n\n[rewrite]\nexclude = ["tools/move/**"]\n'
    r = _declared(tmp_path, base_extra={"tools/move/a_stale_map.toml": stale})
    assert r["certified"], r["refusals"]


def test_a_word_no_edited_lean_file_declares_is_refused(tmp_path):
    """Renaming a declaration's mentions while its `.lean` files stay as they were renames no declaration."""
    repo, base = _repo(tmp_path, DECL_BASE)
    _write(repo, {"notes.md": DECL_HEAD["notes.md"]})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "notes.md"


def test_lean_declared_reads_declarations_constructors_and_fields():
    text = ("/-- `def notMe` in a docstring. -/\n@[simp] private theorem a_b : True := trivial\nnamespace Law.X\n"
            "structure S where\n  draw : Nat\n  h : draw = 1 := by rfl\ninductive T\n  | leaf\n  | node (l : T) : T\n"
            "def f : T → Nat\n  | .leaf => 0\n  | x => 1\n-- def notMeEither\nnoncomputable def escape := 1\n")
    assert MC.lean_declared(text) == ["a_b", "Law.X", "S", "draw", "h", "T", "leaf", "node", "f", "escape"]


AREAS = {
    "tools/move/areas_map.toml": '[words]\n"S/Area" = "S/Area/Check"\n"S.Area" = "S.Area.Check"\n',
    "P/lakefile.toml": 'name = "p"\n',
    "P/S/Area.lean": "import S.Area.Check\n",
    "P/S/Area/Check.lean": "import S.Area.Grid\n\ntheorem t : True := trivial\n",
    "P/S/Area/Grid.lean": "def g := 1\n",
    MC.MOVES: json.dumps({"about": "x", "moves": []}, indent=2) + "\n",
}
AREAS_MOVED = {"P/S/Area.lean": "P/S/Area/Check.lean", "P/S/Area/Check.lean": "P/S/Area/Check/Check.lean",
               "P/S/Area/Grid.lean": "P/S/Area/Check/Grid.lean"}


def _areas(tmp_path: Path, **over: str) -> dict:
    """`S.Area` becomes `S.Area.Check` and its `Check` `S.Area.Check.Check`: a file lands where another left, and an
    aggregator whose every line the words rename goes where the map sends it, as `tools/move/rename.py` writes them."""
    repo, base = _repo(tmp_path, AREAS)
    _move(repo, "P/S/Area/Grid.lean", "P/S/Area/Check/Grid.lean")
    _move(repo, "P/S/Area/Check.lean", "P/S/Area/Check/Check.lean")
    _move(repo, "P/S/Area.lean", "P/S/Area/Check.lean")
    _write(repo, {"P/S/Area/Check.lean": "import S.Area.Check.Check\n",
                  "P/S/Area/Check/Check.lean": "import S.Area.Check.Grid\n\ntheorem t : True := trivial\n",
                  MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": AREAS_MOVED}]}, indent=2) + "\n", **over})
    return _check(repo, base)


def test_a_file_landing_where_another_left_is_a_move(tmp_path):
    r = _areas(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"S.Area": "S.Area.Check", "S.Area.Check": "S.Area.Check.Check",
                                 "S.Area.Grid": "S.Area.Check.Grid"}


def test_a_file_landing_where_another_left_is_recorded_where_it_went(tmp_path):
    swapped = {**AREAS_MOVED, "P/S/Area/Check.lean": "P/S/Area/Grid.lean"}
    [f] = _areas(tmp_path, **{MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": swapped}]}, indent=2) + "\n"})["refusals"]
    assert f["path"] == MC.MOVES


def test_a_file_landing_where_another_left_is_judged_like_any_edit(tmp_path):
    [f] = _areas(tmp_path, **{"P/S/Area/Check.lean": "import S.Area.Check.Check\ntheorem u : True := trivial\n"})["refusals"]
    assert f["path"] == "P/S/Area/Check.lean" and f.get("old") == "P/S/Area.lean"


def test_a_pin_whose_digest_is_not_its_file_is_refused(tmp_path):
    [f] = _worded(tmp_path, **{"p/TRUSTED.sha256": f"{'0' * 64}  ../S/Definitions/Core/A.lean\n"})["refusals"]
    assert f["path"] == "p/TRUSTED.sha256"


def test_pins_sorted_anew_after_their_paths_moved_are_a_move(tmp_path):
    """The command that writes a pin file sorts its paths, so a regeneration after the move may order them anew."""
    a, b = "import Definitions.Core.B\n\n/-! Trusted: `Definitions/Core/B.lean` states `Definitions.Core.B`. -/\n", "def x := 1\n"
    base_pins = (f"{_sha(WORD_BASE['S/Models/Core/A.lean'])}  ../S/Models/Core/A.lean\n"
                 f"{_sha(b)}  ../S/Models/Core/B.lean\n")
    swapped = f"{_sha(b)}  ../S/Definitions/Core/B.lean\n{_sha(a)}  ../S/Definitions/Core/A.lean\n"
    (tmp_path / "ok").mkdir()
    r = _worded(tmp_path / "ok", base_pins, **{"p/TRUSTED.sha256": swapped})
    assert r["certified"], r["refusals"]
    (tmp_path / "dropped").mkdir()
    dropped = f"{_sha(a)}  ../S/Definitions/Core/A.lean\n"
    assert not _worded(tmp_path / "dropped", base_pins, **{"p/TRUSTED.sha256": dropped})["certified"]


def test_a_moves_record_of_a_path_that_did_not_move_whole_is_refused(tmp_path):
    moves = json.dumps({"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}, {"paths": {"S/": "T/"}}]}, indent=2) + "\n"
    [f] = _worded(tmp_path, **{"tools/research/src/research/store/moves.json": moves})["refusals"]
    assert f["path"] == "tools/research/src/research/store/moves.json" and "T/Models/Core/A.lean" in f["reason"]


SPLIT_DIR = {
    "p/circuit/__init__.py": "",
    "p/circuit/a.py": "from p.circuit.live import T\n\nA = T + 1\n",
    "p/circuit/live.py": "T = 1\n",
    "p/circuit/a.sidecar.py": "from p.circuit.a import A\n",
    "p/other.py": "X = 1\n",
    "suite/pyproject.toml": 'inputs = ["p/circuit", "p/other.py"]\n',
    MC.MOVES: json.dumps({"about": "x", "moves": []}, indent=2) + "\n",
}


def _split_dir(tmp_path: Path, paths: dict[str, str] | None) -> dict:
    """`p/circuit/` moves to `q/circuit/` but for `live.py`, which goes to `p/`: no rename pairs the directory, and the
    one-line sidecar whose import is rewritten is below git's similarity. ``paths`` is the entry appended to `moves.json`."""
    repo, base = _repo(tmp_path, SPLIT_DIR)
    _move(repo, "p/circuit/live.py", "p/live.py")
    for f in ("__init__.py", "a.py", "a.sidecar.py"):
        _move(repo, f"p/circuit/{f}", f"q/circuit/{f}")
    _write(repo, {"q/circuit/a.py": "from p.live import T\n\nA = T + 1\n", "q/circuit/a.sidecar.py": "from q.circuit.a import A\n",
                  "suite/pyproject.toml": 'inputs = ["q/circuit", "p/other.py"]\n'})
    if paths is not None:
        _write(repo, {MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": paths}]}, indent=2) + "\n"})
    return _check(repo, base)


def test_the_appended_moves_entry_pairs_what_no_rename_does(tmp_path):
    r = _split_dir(tmp_path, {"p/circuit/live.py": "p/live.py", "p/circuit/": "q/circuit/"})
    assert r["certified"], r["refusals"]
    assert r["map"]["p/circuit"] == "q/circuit"
    assert {e["path"] for e in r["edits"]} >= {"suite/pyproject.toml", "q/circuit/a.sidecar.py", MC.MOVES}


def test_without_the_entry_a_split_directory_is_refused(tmp_path):
    paths = {f["path"] for f in _split_dir(tmp_path, None)["refusals"]}
    assert {"suite/pyproject.toml", "q/circuit/a.sidecar.py"} <= paths


def test_an_entry_the_range_does_not_bear_out_adds_nothing(tmp_path):
    """An entry is a map only where the range moved each file it names to its place: `r/circuit/` is nowhere at HEAD."""
    r = _split_dir(tmp_path, {"p/circuit/live.py": "p/live.py", "p/circuit/": "r/circuit/"})
    assert "p/circuit" not in r["map"]
    assert {MC.MOVES, "suite/pyproject.toml"} <= {f["path"] for f in r["refusals"]}


def test_a_moves_file_that_is_not_json_declares_nothing_and_is_refused(tmp_path):
    repo, base = _repo(tmp_path, SPLIT_DIR)
    _write(repo, {MC.MOVES: '{"about": "x", "moves": [\n'})
    r = _check(repo, base)
    assert [(f["path"], f["reason"]) for f in r["refusals"]] == [(MC.MOVES, "not JSON")]


def test_a_lock_field_the_writer_wraps_anew_is_still_a_move(tmp_path):
    """`check.py --update` wraps a list a renamed module lengthened past its line: the same value, laid out anew."""
    lock = _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                 reads={"Definitions.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}}, dependencies={"m": "2"})
    wrapped = lock.replace('[\n        "Definitions.Core"\n      ]', '["Definitions.Core"]')
    assert wrapped != lock
    r = _worded(tmp_path, **{"S/lean-audit.json": wrapped})
    assert r["certified"], r["refusals"]


def test_a_reads_record_that_changes_with_its_key_is_refused(tmp_path):
    lock = _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                 reads={"Definitions.Core.B": {"digest": "de", "definitions": {"x": "ee"}}}, dependencies={"m": "2"})
    [f] = _worded(tmp_path, **{"S/lean-audit.json": lock})["refusals"]
    assert f["reason"].startswith("`reads` changed")


PROPS = 'name = "props"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Props"\nroots = ["Props.A", "Props.Proofs.A"]\n'
PROPS_PATH = {"type": "path", "scope": "", "name": "props", "manifestFile": "lake-manifest.json", "inherited": False,
              "dir": "../Sec/Props", "configFile": "lakefile.toml"}
PP_PATH = {**PROPS_PATH, "name": "pp", "dir": "../Sec/Proofs/Props"}


def _in_place(tmp_path: Path, **over: str) -> dict:
    """`Props.Proofs` split out of package `props` into a new package `pp` inside `Sec/Proofs/`: `props` drops its roots,
    `q`, which imports the proof, requires `pp`, and a suite's inputs name `pp`'s Lake files."""
    repo, base = _repo(tmp_path, {
        "Sec/Props/lakefile.toml": PROPS, "Sec/Props/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        "Sec/Props/lake-manifest.json": _manifest("props"), "Sec/Props/lean-audit.json": _lock(roots=["Props.A", "Props.Proofs.A"]),
        "Sec/Props/A.lean": "def a := 1\n", "Sec/Props/Proofs/A.lean": "import Props.A\n\ntheorem t : a = 1 := rfl\n",
        "q/lakefile.toml": 'name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n',
        "q/lake-manifest.json": _manifest("q", PROPS_PATH), "q/Q.lean": "import Props.Proofs.A\n",
        "x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n]\n'})
    _move(repo, "Sec/Props/Proofs/A.lean", "Sec/Proofs/Props/A.lean")
    _write(repo, {
        "Sec/Proofs/Props/lakefile.toml": ('name = "pp"\nsrcDir = "../.."\n\n[[require]]\nname = "props"\npath = "../../Props"\n'
                                           '\n[[lean_lib]]\nname = "PP"\nroots = ["Proofs.Props.A"]\n'),
        "Sec/Proofs/Props/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        "Sec/Proofs/Props/lake-manifest.json": _manifest("pp", {**PROPS_PATH, "dir": "../../Props"}),
        "Sec/Proofs/Props/lean-audit.json": _lock(roots=["Proofs.Props.A"], guarantees={}),
        "Sec/Props/lakefile.toml": PROPS.replace(', "Props.Proofs.A"', ""),
        "Sec/Props/lean-audit.json": _lock(roots=["Props.A"], proved_in=["../Proofs/Props"]),
        "q/lakefile.toml": ('name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n\n# the proofs\n[[require]]\n'
                            'name = "pp"\npath = "../Sec/Proofs/Props"\n'),
        "q/lake-manifest.json": _manifest("q", PROPS_PATH, PP_PATH), "q/Q.lean": "import Proofs.Props.A\n",
        "x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n    "Sec/Proofs/Props/lakefile.toml", "Sec/Proofs/Props/lean-toolchain",\n]\n',
        **over})
    return _check(repo, base)


def test_a_split_into_a_package_inside_another_tree_is_certified(tmp_path):
    r = _in_place(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"Props.Proofs.A": "Proofs.Props.A"}


def test_an_in_place_lakefile_that_drops_a_root_that_did_not_move_is_refused(tmp_path):
    [f] = _in_place(tmp_path, **{"Sec/Props/lakefile.toml": PROPS.replace('"Props.A", ', "").replace(', "Props.Proofs.A"', "")})["refusals"]
    assert f["path"] == "Sec/Props/lakefile.toml"


def test_an_in_place_require_of_an_existing_package_is_refused(tmp_path):
    lakefile = 'name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n\n[[require]]\nname = "q2"\npath = "../Sec"\n'
    r = _in_place(tmp_path, **{"q/lakefile.toml": lakefile})
    assert [f["path"] for f in r["refusals"]] == ["q/lakefile.toml"]


def test_an_inputs_line_naming_an_existing_package_is_refused(tmp_path):
    r = _in_place(tmp_path, **{"x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n    "Sec/Props/lean-toolchain",\n]\n'})
    assert [f["path"] for f in r["refusals"]] == ["x/pyproject.toml"]


def test_a_fold_into_an_existing_package_moves_its_roots(tmp_path):
    """`Props.Proofs` folded into `Sec/Proofs`'s existing package (srcDir `..`): `props` drops the roots and `proofs` gains them."""
    proofs = 'name = "proofs"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\nroots = ["Proofs.B"]\n'
    repo, base = _repo(tmp_path, {
        "Sec/Props/lakefile.toml": PROPS, "Sec/Props/A.lean": "def a := 1\n",
        "Sec/Props/Proofs/A.lean": "import Props.A\n\ntheorem t : a = 1 := rfl\n",
        "Sec/Proofs/lakefile.toml": proofs, "Sec/Proofs/B.lean": "import Props.Proofs.A\n"})
    _move(repo, "Sec/Props/Proofs/A.lean", "Sec/Proofs/Props/A.lean")
    _write(repo, {"Sec/Props/lakefile.toml": PROPS.replace(', "Props.Proofs.A"', ""),
                  "Sec/Proofs/lakefile.toml": proofs.replace('"Proofs.B"]', '"Proofs.B", "Proofs.Props.A"]'),
                  "Sec/Proofs/B.lean": "import Proofs.Props.A\n"})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"Sec/Proofs/lakefile.toml": proofs.replace('"Proofs.B"]', '"Proofs.B", "Proofs.Props.A", "Proofs.C"]')})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "Sec/Proofs/lakefile.toml"


#: a package below `pkg/` that moves deeper, the shape of the vLLM integration's first move
APP = {"pkg/app/__init__.py": "", "pkg/app/store/__init__.py": "", "pkg/app/store/remote.py": "class Session:\n    pass\n",
       "pkg/app/store/host.py": "x = 1\n", "pkg/app/linear/__init__.py": "", "pkg/app/linear/api.py": "y = 2\n",
       "pkg/tests/store/__init__.py": "", "pkg/tests/store/test_remote.py": "ROOT = Path(__file__).resolve().parents[2]\n",
       ".gitignore": "pkg/tests/store/fixtures/periods\n"}
APP_MOVES = [("pkg/app/store", "pkg/app/protocols/bandwidth"), ("pkg/app/linear", "pkg/app/protocols/linear"),
             ("pkg/tests/store", "pkg/tests/protocols/bandwidth")]


def _app(tmp_path: Path, files: dict[str, str], edits: dict[str, str]) -> dict:
    repo, base = _repo(tmp_path, {**APP, **files})
    for old, new in APP_MOVES:
        _move(repo, old, new)
    _write(repo, edits)
    return _check(repo, base)


def test_prose_paths_and_module_names_in_any_text_file_are_certified(tmp_path):
    r = _app(tmp_path, {
        "README.md": "The store (`pkg/app/store/`, `app/store/remote.py`) is `app.store.remote.Session`.\n"
                     "Its test is `tests/store/test_remote.py`.\n",
        "run.sh": 'git show "$SHA:pkg/app/store/host.py" > host.py  # app.linear.api\n',
        "keep.json": '{"app.linear.api": "pkg/app/linear/api.py"}\n',
        "pkg/app/doc.py": '"""Through `app.store.remote.Session` (pkg/app/store/remote.py)."""\n'}, {
        "README.md": "The store (`pkg/app/protocols/bandwidth/`, `app/protocols/bandwidth/remote.py`) is "
                     "`app.protocols.bandwidth.remote.Session`.\nIts test is `tests/store/test_remote.py`.\n",
        "run.sh": 'git show "$SHA:pkg/app/protocols/bandwidth/host.py" > host.py  # app.protocols.linear.api\n',
        "keep.json": '{"app.protocols.linear.api": "pkg/app/protocols/linear/api.py"}\n',
        "pkg/app/doc.py": '"""Through `app.protocols.bandwidth.remote.Session` (pkg/app/protocols/bandwidth/remote.py)."""\n',
        "pkg/tests/protocols/bandwidth/test_remote.py": "ROOT = Path(__file__).resolve().parents[3]\n",
        ".gitignore": "pkg/tests/protocols/bandwidth/fixtures/periods\n"})
    assert r["certified"], r["refusals"]
    assert {e["path"]: e["kind"] for e in r["edits"]} == {
        "README.md": "text", "run.sh": "text", "keep.json": "text", "pkg/app/doc.py": "python", ".gitignore": "text",
        "pkg/tests/protocols/bandwidth/test_remote.py": "python"}


PY_OLD = ("from app import linear, store as ST\nfrom app.store import remote as R\n\n"
          'HOST = Path(__file__).resolve().parents[0] / "store" / "host.py"\nUP = Path(R.__file__).parents[1]\n')
PY_NEW = ("from app.protocols import linear\nfrom app.protocols import bandwidth as ST\nfrom app.protocols.bandwidth import remote as R\n\n"
          'HOST = Path(__file__).resolve().parents[0] / "protocols" / "bandwidth" / "host.py"\nUP = Path(R.__file__).parents[2]\n')


def test_moved_modules_imported_by_name_and_anchors_with_a_path_are_certified(tmp_path):
    r = _app(tmp_path, {"pkg/app/use.py": PY_OLD}, {"pkg/app/use.py": PY_NEW})
    assert r["certified"], r["refusals"]


def test_imports_in_a_shell_heredoc_and_in_a_string_literal_are_certified(tmp_path):
    r = _app(tmp_path, {"run.sh": "python - <<'EOF'\nfrom app import linear, store as ST\nEOF\n",
                        "pkg/app/gen.py": 'SRC = (\n    "from app.store import remote as R\\n"\n)\n'},
             {"run.sh": "python - <<'EOF'\nfrom app.protocols import linear\nfrom app.protocols import bandwidth as ST\nEOF\n",
              "pkg/app/gen.py": 'SRC = (\n    "from app.protocols.bandwidth import remote as R\\n"\n)\n'})
    assert r["certified"], r["refusals"]


def test_an_import_in_a_string_literal_that_binds_another_name_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/gen.py": 'SRC = (\n    "from app.store import remote as R\\n"\n)\n'},
               {"pkg/app/gen.py": 'SRC = (\n    "from app.protocols.bandwidth import remote as S\\n"\n)\n'})["refusals"]
    assert f["path"] == "pkg/app/gen.py" and "as S" in f["hunk"]


def test_an_import_that_binds_another_name_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/use.py": PY_OLD},
               {"pkg/app/use.py": PY_NEW.replace("bandwidth as ST", "bandwidth as BW")})["refusals"]
    assert f["path"] == "pkg/app/use.py" and "BW" in f["hunk"]


PAREN_OLD = "from app import store as ST\nfrom app.store.remote import (Session,\n                              Other)\n"


def test_a_parenthesized_import_continued_on_an_unchanged_line_is_certified(tmp_path):
    r = _app(tmp_path, {"pkg/app/use.py": PAREN_OLD}, {"pkg/app/use.py": PAREN_OLD.replace(
        "from app import store as ST", "from app.protocols import bandwidth as ST").replace(
        "app.store.remote", "app.protocols.bandwidth.remote")})
    assert r["certified"], r["refusals"]


def test_a_parenthesized_import_of_a_module_that_did_not_move_there_is_refused(tmp_path):
    for old in (PAREN_OLD, PAREN_OLD.replace("import (Session,", "import (")):
        new = old.replace("from app import store as ST", "from app.protocols import bandwidth as ST").replace(
            "app.store.remote", "app.protocols.linear.api")
        [f] = _app(tmp_path / str(len(old)), {"pkg/app/use.py": old}, {"pkg/app/use.py": new})["refusals"]
        assert f["path"] == "pkg/app/use.py" and "linear.api" in f["hunk"]


def test_a_module_anchor_that_names_another_directory_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/use.py": PY_OLD},
               {"pkg/app/use.py": PY_NEW.replace("parents[2]", "parents[3]")})["refusals"]
    assert f["path"] == "pkg/app/use.py" and "parents[3]" in f["hunk"]


def test_a_word_that_names_no_moved_path_is_refused_in_prose(tmp_path):
    [f] = _app(tmp_path, {"README.md": "The store is `app.store`.\n"},
               {"README.md": "The store is `app.protocols.storage`.\n"})["refusals"]
    assert f["path"] == "README.md" and f["reason"] == "edit beyond moved paths and module names"


WORD_OLD = {"examples/toy/run.sh": "echo toy\n",
            "README.md": "The end-to-end examples (`examples/toy/run.sh`) are the milestone's.\n",
            "registry.json": '{"examples": ["who runs `examples/`?"], "owns": ["examples/toy/"]}\n',
            "pkg/app/ladder.py": 'def examples(root):\n    return sorted((root / "examples").iterdir())\n'}
WORD_NEW = {"README.md": "The end-to-end examples (`demos/toy/run.sh`) are the milestone's.\n",
            "registry.json": '{"examples": ["who runs `demos/`?"], "owns": ["demos/toy/"]}\n',
            "pkg/app/ladder.py": 'def examples(root):\n    return sorted((root / "demos").iterdir())\n'}


def _word(tmp_path: Path, edits: dict[str, str]) -> dict:
    repo, base = _repo(tmp_path, WORD_OLD)
    _move(repo, "examples/toy/run.sh", "demos/toy/run.sh")
    _write(repo, edits)
    return _check(repo, base)


def test_a_moved_top_level_name_left_standing_as_a_word_is_certified(tmp_path):
    r = _word(tmp_path, WORD_NEW)
    assert r["certified"], r["refusals"]


def test_a_moved_top_level_name_rewritten_to_anything_else_is_refused(tmp_path):
    [f] = _word(tmp_path, {**WORD_NEW, "README.md": WORD_NEW["README.md"].replace("end-to-end examples", "end-to-end samples")})[
        "refusals"]
    assert f["path"] == "README.md" and "samples" in f["hunk"]


def test_anchor_that_names_another_directory_is_refused(tmp_path):
    [f] = _app(tmp_path, {}, {"pkg/tests/protocols/bandwidth/test_remote.py": "ROOT = Path(__file__).resolve().parents[4]\n"})[
        "refusals"]
    assert f["path"] == "pkg/tests/protocols/bandwidth/test_remote.py" and "parents[4]" in f["hunk"]


SH_BODY = "".join(f'echo "step {i} of the session, unchanged by the move"\n' for i in range(12))
SH_OLD = ('#!/usr/bin/env bash\nSELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")\n'
          'ROOT=$(cd "$(dirname "$SELF")/../../.." && pwd)\ncd "$(dirname "${BASH_SOURCE[0]}")/../../.."\n'
          'bash "$(dirname "$0")/../../../pkg/app/store/host.py"\n' + SH_BODY)


def _shell(tmp_path: Path, new: str) -> dict:
    """A script three directories down moves to two (the streamed wipe's `demo.sh` to `examples/wipe/`)."""
    repo, base = _repo(tmp_path, {**APP, "bench/pous/wipe/demo.sh": SH_OLD})
    for old, new_dir in APP_MOVES:
        _move(repo, old, new_dir)
    _move(repo, "bench/pous/wipe/demo.sh", "examples/wipe/demo.sh")
    _write(repo, {"examples/wipe/demo.sh": new})
    return _check(repo, base)


SH_NEW = ('#!/usr/bin/env bash\nSELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")\n'
          'ROOT=$(cd "$(dirname "$SELF")/../.." && pwd)\ncd "$(dirname "${BASH_SOURCE[0]}")/../.."\n'
          'bash "$(dirname "$0")/../../pkg/app/protocols/bandwidth/host.py"\n' + SH_BODY)


def test_a_shell_scripts_paths_from_its_own_directory_recounted_are_certified(tmp_path):
    r = _shell(tmp_path, SH_NEW)
    assert r["certified"], r["refusals"]


def test_a_shell_path_from_its_own_directory_that_names_another_one_is_refused(tmp_path):
    [f] = _shell(tmp_path, SH_NEW.replace('"$SELF")/../..', '"$SELF")/..'))["refusals"]
    assert f["path"] == "examples/wipe/demo.sh" and '"$SELF")/.." &&' in f["hunk"]


def test_a_shell_path_from_a_variable_that_is_not_the_scripts_own_path_is_refused(tmp_path):
    other = SH_OLD.replace('SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")', 'SELF=$NVCC')
    repo, base = _repo(tmp_path, {**APP, "bench/pous/wipe/demo.sh": other})
    _move(repo, "bench/pous/wipe/demo.sh", "examples/wipe/demo.sh")
    _write(repo, {"examples/wipe/demo.sh": other.replace('"$SELF")/../../..', '"$SELF")/../..')})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "examples/wipe/demo.sh" and "SELF" in f["hunk"]


def test_relative_import_rewritten_absolute_is_certified(tmp_path):
    body = "".join(f"\n\ndef f{i}():\n    return put({i})\n" for i in range(8))
    files = {"pkg/app/options/__init__.py": "", "pkg/app/options/pearl.py": "from .pearl_host import put\n" + body,
             "pkg/app/options/pearl_host.py": "def put():\n    pass\n"}
    repo, base = _repo(tmp_path, {**APP, **files})
    _move(repo, "pkg/app/options/pearl.py", "pkg/app/protocols/pearl/__init__.py")
    _move(repo, "pkg/app/options/pearl_host.py", "pkg/app/protocols/pearl/host.py")
    _move(repo, "pkg/app/options/__init__.py", "pkg/app/protocols/__init__.py")
    _write(repo, {"pkg/app/protocols/pearl/__init__.py": "from app.protocols.pearl.host import put\n" + body})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"pkg/app/protocols/pearl/__init__.py": "from app.protocols.pearl.other import put\n" + body})
    assert _check(repo, base)["refusals"]


def test_identical_files_git_pairs_across_directories_follow_their_directory():
    renames = [("a/__init__.py", "y/b/__init__.py"), ("b/__init__.py", "x/a/__init__.py"),
               ("a/m.py", "x/a/m.py"), ("b/n.py", "y/b/n.py")]
    base = ["a/__init__.py", "a/m.py", "b/__init__.py", "b/n.py"]
    added = {"x/a/__init__.py", "x/a/m.py", "y/b/__init__.py", "y/b/n.py"}
    assert MC.derive_map(renames, base, set(base), added) == {"a": "x/a", "b": "y/b"}


MOVES_DOC = {"about": "the moves", "moves": [{"paths": {"old/": "older/"}}]}


def _moves(tmp_path: Path, paths: dict[str, str]) -> dict:
    files = {MC.MOVES: json.dumps(MOVES_DOC, indent=2) + "\n"}
    return _app(tmp_path, files, {MC.MOVES: json.dumps({**MOVES_DOC, "moves": [*MOVES_DOC["moves"], {"paths": paths}]},
                                                       indent=2) + "\n"})


def test_moves_entry_the_range_agrees_with_is_certified(tmp_path):
    r = _moves(tmp_path, {"pkg/app/store/": "pkg/app/protocols/bandwidth/", "pkg/app/linear/api.py": "pkg/app/protocols/linear/api.py",
                          "pkg/app/store/added_on_a_branch.py": "pkg/app/protocols/bandwidth/added_on_a_branch.py"})
    assert r["certified"], r["refusals"]
    assert {"path": MC.MOVES, "kind": "moves"} in r["edits"]


def test_moves_entry_the_range_contradicts_is_refused(tmp_path):
    [f] = _moves(tmp_path, {"pkg/app/store/": "pkg/app/protocols/storage/"})["refusals"]
    assert f["path"] == MC.MOVES and "pkg/app/protocols/storage/" in f["reason"]
