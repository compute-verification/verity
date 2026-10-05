"""`tools/lean/split.py`: a package's spec and its proofs computed from its sources and lock, a spec that imports a proof
refused, and the split carried out as a pure move on a fixture checkout (the lock's guarantees and reads byte for byte, the
proofs under a root of their own and the spec's lock naming them in `proved_in`, the proofs package requiring the spec by
path, a dependent requiring the proofs and importing them by their new names, `lean-deps.json` re-keyed to the same bundles);
then built with Lake and audited when `lake` is on the PATH."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_split", HERE.parent / "split.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
A = S.A
TOOLCHAIN = (A.FIXTURES / "control" / "lean-toolchain").read_text()
LAKE = shutil.which("lake") or (str(Path.home() / ".elan" / "bin" / "lake") if (Path.home() / ".elan" / "bin" / "lake").exists()
                                else None)

PKG = "protocols/p/lean"
FILES = {
    "P.lean": "import P.Protocol\nimport P.Guarantees\nimport P.SecurityProofs\n",
    "P/Protocol.lean": "/-! The definitions. -/\nnamespace P\ndef f (n : Nat) : Nat := n + 0\nend P\n",
    "P/Guarantees.lean": "import P.Protocol\nnamespace P.Guarantees\ndef G : Prop := ∀ n, P.f n = n\nend P.Guarantees\n",
    "P/SecurityProofs.lean": "import P.Guarantees\nimport P.SecurityProofs.Lemma\nnamespace P.SecurityProofs\n"
                             "theorem t : P.Guarantees.G := fun n => P.SecurityProofs.Lemma.l n\nend P.SecurityProofs\n",
    "P/SecurityProofs/Lemma.lean": "import P.Protocol\nnamespace P.SecurityProofs.Lemma\n"
                                   "theorem l (n : Nat) : P.f n = n := rfl\nend P.SecurityProofs.Lemma\n",
    "CheckAxioms.lean": "import P\n#print axioms P.SecurityProofs.t\n",
}
LAKEFILE = 'name = "p"\ndefaultTargets = ["P"]\n\n[leanOptions]\nautoImplicit = false\n\n[[lean_lib]]\nname = "P"\n'
DEPENDENT = "protocols/q/lean"
DEP_FILES = {"Q.lean": "import P.SecurityProofs.Lemma\nimport P.Guarantees\n"
                       "theorem Q.q : P.Guarantees.G := fun n => P.SecurityProofs.Lemma.l n\n"}
DEP_LAKEFILE = ('name = "q"\ndefaultTargets = ["Q"]\n\n[[require]]\nname = "p"\npath = "../../p/lean"\n\n# Q\'s library.\n'
                '[[lean_lib]]\nname = "Q"\n')
MATHLIB = ('[[require]]\nname = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib4"\n'
           'rev = "5ed2965256430c3649e86755f9576b54eca72435"\n\n')
FETCHED = {"type": "git", "subDir": None, "scope": "", "name": "mathlib", "manifestFile": "lake-manifest.json",
           "inherited": False, "configFile": "lakefile.lean", "url": "https://github.com/leanprover-community/mathlib4",
           "rev": "5ed2965256430c3649e86755f9576b54eca72435", "inputRev": "5ed2965256430c3649e86755f9576b54eca72435"}


def manifest(name: str, packages: list[dict]) -> str:
    return json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": packages, "name": name,
                       "lakeDir": ".lake", "fixedToolchain": False}, indent=1) + "\n"


def write(root: Path, base: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / base / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / base / rel).write_text(text)


def checkout(tmp: Path, mathlib: bool = False, extra: dict[str, str] | None = None) -> Path:
    """A git checkout with package P (spec, proofs and an exempt axioms list) and package Q, which requires P by path and
    imports one of P's lemmas; with ``mathlib``, both also require Mathlib (planned, never built)."""
    root = tmp / "repo"
    fetched = [FETCHED] if mathlib else []
    write(root, PKG, FILES | (extra or {}) | {
        "lakefile.toml": LAKEFILE.replace("\n[[lean_lib]]", "\n" + (MATHLIB if mathlib else "") + "[[lean_lib]]"),
        "lean-toolchain": TOOLCHAIN, "lake-manifest.json": manifest("p", fetched)})
    write(root, DEPENDENT, DEP_FILES | {
        "lakefile.toml": DEP_LAKEFILE.replace("[[require]]", (MATHLIB if mathlib else "") + "[[require]]", 1),
        "lean-toolchain": TOOLCHAIN,
        "lake-manifest.json": manifest("q", [S.entry("p", "../../p/lean", False), *({**f, "inherited": False} for f in fetched)])})
    policy = {"roots": ["P"], "exempt": {"CheckAxioms": "the axioms list"}, "assumptions": ["P.Assumptions"],
              "meaning": ["P.Protocol", "P.Guarantees"],
              "layers": {"P.Protocol": {"imports": ["P.Protocol"], "except": [], "why": "trusted text"},
                         "P.SecurityProofs.Lemma": {"imports": ["P.Protocol"], "except": [], "why": "a lemma of the definitions"}},
              "guarantees": {"P.SecurityProofs.t": {}},
              "reads": {"P.Protocol": {"pins": ["P.SecurityProofs.t"], "digest": "0" * 64}}}
    A.dump(root / PKG, policy)
    (root / "tools" / "lean").mkdir(parents=True)
    (root / "README.md").write_text(f"The lemma: `{PKG}/P/SecurityProofs/Lemma.lean`; the proofs: `{PKG}/P/SecurityProofs/`.\n"
                                    f"In the package: `P/SecurityProofs/Lemma.lean`; `CheckAxioms.lean`.\n")
    (root / "protocols" / "p" / "PROTOCOL.md").write_text("Proved in `lean/P/SecurityProofs/Lemma.lean`, not `lean/P/Protocol.lean`.\n")
    (root / "protocols" / "p" / "py").mkdir()
    (root / "protocols" / "p" / "py" / "m.py").write_text('"""Proved in `lean/P/SecurityProofs/` and `p/lean/P/SecurityProofs.lean`."""\n')
    (root / PKG / "TRUSTED.sha256").write_text("".join(
        f"{hashlib.sha256((root / PKG / f).read_bytes()).hexdigest()}  {f}\n" for f in ("lakefile.toml", "P/Protocol.lean")))
    (root / PKG / "README.md").write_text("P/Protocol.lean               the definitions\nP/SecurityProofs/Lemma.lean   the lemma\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if mathlib:
        pins = {S.deps_key(root / p): {"package": p, "sha256": n * 64, "bytes": 1, "run": "r0"}
                for p, n in ((PKG, "a"), (DEPENDENT, "b"))}
        S.write_json(root / S.LEAN_DEPS, {"deps": pins, "schema": "lean-deps-pin/v1"})
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def test_the_partition(tmp_path):
    root = checkout(tmp_path)
    p = S.plan(root / PKG, "p", [], root)
    assert p["spec"] == ["P.Guarantees", "P.Protocol"]
    assert p["proofs"] == ["CheckAxioms", "P", "P.SecurityProofs", "P.SecurityProofs.Lemma"]
    assert p["spec_roots"] == ["P.Guarantees", "P.Protocol"]
    assert p["dest"] == "security_proofs/p"
    assert p["dependents"] == [{"package": DEPENDENT, "imports": ["P.SecurityProofs.Lemma"]}]
    assert p["rename"] == {"P": "PProofs", "P.SecurityProofs": "PProofs.SecurityProofs",
                           "P.SecurityProofs.Lemma": "PProofs.SecurityProofs.Lemma"}
    assert p["proof_roots"] == ["PProofs"] and p["proofs_lib"] == "PProofs"
    assert p["lock_spec"]["guarantees"] == A.load(root / PKG)["guarantees"]
    assert p["lock_spec"]["proved_in"] == ["../../../security_proofs/p"]
    assert "exempt" not in p["lock_spec"] and p["lock_proofs"]["exempt"] == {"CheckAxioms": "the axioms list"}
    assert list(p["lock_spec"]["layers"]) == ["P.Protocol"] and list(p["lock_proofs"]["layers"]) == ["PProofs.SecurityProofs.Lemma"]


def test_a_reads_exemption_of_moved_proofs_names_their_new_modules(tmp_path):
    root = checkout(tmp_path)
    A.dump(root / PKG, A.load(root / PKG) | {"reads_exempt": {"P.SecurityProofs.Lemma": "witnesses", "Other": "theirs"}})
    p = S.plan(root / PKG, "p", [], root)
    assert p["lock_spec"]["reads_exempt"] == {"PProofs.SecurityProofs.Lemma": "witnesses", "Other": "theirs"}


def test_a_layer_that_allows_proofs_allows_them_under_their_new_root():
    spec, proofs = {"P.Protocol", "P.Guarantees"}, {"P", "P.SecurityProofs", "P.SecurityProofs.Lemma"}
    assert S.renamed_prefixes(["Mathlib", "P.Protocol", "P.SecurityProofs", "P", "Q"], proofs, spec, {"P"}) == \
        ["Mathlib", "P.Protocol", "PProofs.SecurityProofs", "P", "PProofs", "Q"]


def test_a_proofs_root_already_taken_is_refused(tmp_path):
    root = checkout(tmp_path, extra={"PProofs.lean": "import P.Protocol\n"})
    with pytest.raises(S.Refused, match="the proofs' root PProofs is already a root"):
        S.plan(root / PKG, "p", [], root)


def test_a_spec_that_imports_a_proof_is_refused(tmp_path):
    root = checkout(tmp_path, extra={"P/Guarantees.lean": "import P.Protocol\nimport P.SecurityProofs.Lemma\n"
                                                          "namespace P.Guarantees\ndef G : Prop := ∀ n, P.f n = n\nend P.Guarantees\n"})
    with pytest.raises(S.Refused, match="P.Guarantees imports P.SecurityProofs.Lemma"):
        S.plan(root / PKG, "p", [], root)
    assert S.main([str(root / PKG), "--area", "p", "--root", str(root)]) == 1
    kept = S.plan(root / PKG, "p", ["P.SecurityProofs.Lemma"], root)
    assert "P.SecurityProofs.Lemma" in kept["spec"] and "P.SecurityProofs.Lemma" in kept["spec_roots"]


def test_the_dry_run_changes_nothing(tmp_path, capsys):
    root = checkout(tmp_path)
    assert S.main([str(root / PKG), "--area", "p", "--root", str(root)]) == 0
    assert "mv protocols/p/lean/P/SecurityProofs/Lemma.lean -> security_proofs/p/PProofs/SecurityProofs/Lemma.lean" in capsys.readouterr().out
    assert subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout == ""


def test_the_split_is_a_pure_move(tmp_path):
    root = checkout(tmp_path, mathlib=True)
    lock = (root / PKG / "lean-audit.json").read_text()
    pins_before = json.loads((root / S.LEAN_DEPS).read_text())["deps"]
    p = S.plan(root / PKG, "p", [], root)
    S.describe(p)
    notes = S.apply(p, root)
    dest = root / "security_proofs" / "p"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
    assert "protocols/p/lean/P/SecurityProofs/Lemma.lean -> security_proofs/p/PProofs/SecurityProofs/Lemma.lean" in status
    assert (dest / "PProofs.lean").exists() and not (root / PKG / "P.lean").exists()
    assert (dest / "PProofs.lean").read_text() == "import P.Protocol\nimport P.Guarantees\nimport PProofs.SecurityProofs\n"
    assert (dest / "CheckAxioms.lean").read_text() == "import PProofs\n#print axioms P.SecurityProofs.t\n"
    assert (root / DEPENDENT / "Q.lean").read_text().startswith("import PProofs.SecurityProofs.Lemma\nimport P.Guarantees\n")
    assert (dest / "lean-toolchain").read_text() == TOOLCHAIN
    new_spec = json.loads((root / PKG / "lean-audit.json").read_text())
    old = json.loads(lock)
    assert new_spec["roots"] == ["P.Guarantees", "P.Protocol"]
    assert new_spec["proved_in"] == ["../../../security_proofs/p"]
    assert {k: v for k, v in new_spec.items() if k not in ("roots", "proved_in", "exempt", "layers")} == \
        {k: v for k, v in old.items() if k not in ("roots", "exempt", "layers")}
    assert json.dumps(new_spec["guarantees"]) == json.dumps(old["guarantees"])
    assert (root / PKG / "lean-audit.json").read_text() == A.text(new_spec) and '"pins": ["P.SecurityProofs.t"]' in lock
    cfg = tomllib.loads((dest / "lakefile.toml").read_text())
    assert [(r["name"], r.get("path"), r.get("rev")) for r in cfg["require"]] == \
        [("mathlib", None, FETCHED["rev"]), ("p", "../../protocols/p/lean", None)]
    assert cfg["leanOptions"] == {"autoImplicit": False} and cfg["lean_lib"] == [{"name": "PProofs", "roots": ["PProofs"]}]
    assert tomllib.loads((root / PKG / "lakefile.toml").read_text())["lean_lib"] == [{"name": "P", "roots": ["P.Guarantees", "P.Protocol"]}]
    m = json.loads((dest / "lake-manifest.json").read_text())
    assert m["name"] == "security_proofs_p" and [e["name"] for e in m["packages"]] == ["p", "mathlib"]
    assert A.load(dest) == {"roots": ["PProofs"], "exempt": {"CheckAxioms": "the axioms list"},
                            "layers": {"PProofs.SecurityProofs.Lemma": old["layers"]["P.SecurityProofs.Lemma"]}, "guarantees": {}}
    assert (root / DEPENDENT / "lakefile.toml").read_text().endswith(
        'path = "../../p/lean"\n\n[[require]]\nname = "security_proofs_p"\npath = "../../../security_proofs/p"\n\n'
        '# Q\'s library.\n[[lean_lib]]\nname = "Q"\n')
    q = tomllib.loads((root / DEPENDENT / "lakefile.toml").read_text())
    assert [(r["name"], r.get("path")) for r in q["require"]][1:] == [("p", "../../p/lean"),
                                                                       ("security_proofs_p", "../../../security_proofs/p")]
    qm = json.loads((root / DEPENDENT / "lake-manifest.json").read_text())
    assert [(e["name"], e.get("dir"), e["inherited"]) for e in qm["packages"]] == [
        ("p", "../../p/lean", False), ("security_proofs_p", "../../../security_proofs/p", False), ("mathlib", None, False)]
    pins = json.loads((root / S.LEAN_DEPS).read_text())["deps"]
    assert pins[S.deps_key(dest)] == pins_before[S.deps_key(root / PKG)] | {"package": "security_proofs/p"}
    assert pins[S.deps_key(root / DEPENDENT)]["sha256"] == "b" * 64 and len(pins) == 3
    readme = (root / "README.md").read_text()
    assert "`security_proofs/p/PProofs/SecurityProofs/Lemma.lean`" in readme and "`security_proofs/p/PProofs/SecurityProofs/`" in readme
    assert (root / "protocols" / "p" / "PROTOCOL.md").read_text() == \
        "Proved in `security_proofs/p/PProofs/SecurityProofs/Lemma.lean`, not `lean/P/Protocol.lean`.\n"
    assert (root / "protocols" / "p" / "py" / "m.py").read_text() == \
        '"""Proved in `security_proofs/p/PProofs/SecurityProofs/` and `security_proofs/p/PProofs/SecurityProofs.lean`."""\n'
    assert (root / PKG / "README.md").read_text().endswith("\nP/SecurityProofs/Lemma.lean   the lemma\n")
    assert [n for n in notes if n.startswith("read: ")] == [
        f"read: {PKG}/README.md:2: P/SecurityProofs/Lemma.lean   the lemma",
        f"read: {PKG}/TRUSTED.sha256: lakefile.toml changed; regenerate the list"]


LAKE_WRITTEN = """{"version": "1.2.0",
 "packagesDir": ".lake/packages",
 "packages":
 [{"type": "path",
   "scope": "",
   "name": "verity",
   "manifestFile": "lake-manifest.json",
   "inherited": false,
   "dir": "../../../packages/verity/lean",
   "configFile": "lakefile.toml"},
  {"url": "https://github.com/leanprover-community/mathlib4",
   "type": "git",
   "subDir": null,
   "scope": "",
   "rev": "5ed2965256430c3649e86755f9576b54eca72435",
   "name": "mathlib",
   "manifestFile": "lake-manifest.json",
   "inputRev": "5ed2965256430c3649e86755f9576b54eca72435",
   "inherited": false,
   "configFile": "lakefile.lean"}],
 "name": "nci",
 "lakeDir": ".lake",
 "fixedToolchain": false}
"""


def test_a_manifest_is_written_as_lake_writes_it():
    assert S.lake_json(json.loads(LAKE_WRITTEN)) + "\n" == LAKE_WRITTEN
    assert S.lake_json({"version": "1.2.0", "packages": [], "name": "p"}) == '{"version": "1.2.0", "packages": [], "name": "p"}'


def test_deps_key_is_lean_audit_s(tmp_path):
    """`lean_audit.deps_key`, run from its source (importing `lean_audit` loads `tools/check/common.py`, outside this suite)."""
    tree = ast.parse((HERE.parent / "lean_audit.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "deps_key")
    scope = {"hashlib": hashlib, "json": json, "Path": Path}
    exec(compile(ast.Module([fn], []), "lean_audit.py", "exec"), scope)
    root = checkout(tmp_path, mathlib=True)
    assert S.deps_key(root / PKG) == scope["deps_key"](root / PKG) is not None
    assert S.deps_key(root / PKG / "P") is None


@pytest.mark.slow
@pytest.mark.skipif(LAKE is None, reason="needs lake")
def test_the_split_builds(tmp_path):
    """What only Lake shows: the spec, the proofs and a dependent build after the split, the proofs under their own root
    find the spec's modules in the other package, Lake keeps the manifests the split wrote, and `lake env lean` (the
    search path `Facts.lean` and `Replay.lean` read with) elaborates a proof file that imports spec and proof modules."""
    root = checkout(tmp_path)
    env = A.lake_env()
    S.apply(S.plan(root / PKG, "p", [], root), root)
    dest = root / "security_proofs" / "p"
    for d in (root / PKG, dest, root / DEPENDENT):
        r = subprocess.run([LAKE, "build"], cwd=d, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    assert subprocess.run(["git", "diff", "--stat"], cwd=root, capture_output=True, text=True).stdout == ""
    r = subprocess.run([LAKE, "env", "lean", "PProofs.lean"], cwd=dest, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
