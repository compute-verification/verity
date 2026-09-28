"""The suite's inputs under `Security` (`pyproject.toml`) cover what its Lean reads there. Lake and `lean` run outside the read
guard, so nothing else fails when a module a test builds (`lake build Proofs.Flock.Verifier`) or imports (in a scratch file it
hands to `lake env lean`, or a file of `tests/lean/`) reaches a file of `Security` that the list leaves out. These tests read the
`import` lines and the lakefiles alone, without Lean."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SUITE = TESTS.parent
ROOT = SUITE.parents[2]
SECURITY = ROOT / "Security"
#: the package `lake build` runs in (test_lean_verifier.py's `PROOFS`)
PROOFS = SECURITY / "Proofs"
#: what Lake reads of each package it loads
PACKAGE_FILES = ("lakefile.toml", "lake-manifest.json", "lean-toolchain")
#: the workspace dependency whose directory `suites.py` keys this suite on, which holds the verifier's and core's Lake packages
DEPENDENCY = "verity"
IMPORT = re.compile(r"^\s*(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?(.*)$")


def declared() -> list[str]:
    return tomllib.loads((SUITE / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["inputs"]


def under(path: str, prefixes: list[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def packages() -> tuple[dict[Path, dict], list[Path]]:
    """({directory: lakefile} of `security_proofs` and each package under `Security` it requires by path, transitively; the
    packages outside `Security` it so requires)."""
    seen: dict[Path, dict] = {}
    todo = [PROOFS]
    while todo:
        d = todo.pop()
        if d not in seen:
            seen[d] = tomllib.loads((d / "lakefile.toml").read_text())
            todo += [(d / r["path"]).resolve() for r in seen[d].get("require", []) if "path" in r]
    return {d: c for d, c in seen.items() if d.is_relative_to(SECURITY)}, sorted(d for d in seen if not d.is_relative_to(SECURITY))


def header_imports(text: str) -> list[str]:
    """The modules a Lean file's header imports: its `import` lines before the first other command, past comments."""
    out, depth = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if depth:
            depth += s.count("/-") - s.count("-/")
        elif s.startswith("/-"):
            depth = s.count("/-") - s.count("-/")
        elif s and not s.startswith("--"):
            if not (m := IMPORT.match(line)):
                break
            out += m.group(1).split("--")[0].split()
    return out


def roots(libs: set[str]) -> set[str]:
    """The modules of the `Security` libraries (``libs``) a test builds (`["lake", "build", M]`) or writes an import of, and
    those a file of `tests/lean/` imports."""
    named = re.compile(r"""(?:\bimport\s+|"build",\s*")((?:%s)(?:\.[\w']+)+)""" % "|".join(map(re.escape, sorted(libs))))
    out = {m for p in sorted(TESTS.glob("*.py")) if p != Path(__file__).resolve() for m in named.findall(p.read_text())}
    return out | {m for p in sorted((TESTS / "lean").glob("*.lean")) for m in header_imports(p.read_text())
                  if m.partition(".")[0] in libs}


def closure(start: set[str], sources: list[Path]) -> tuple[dict[str, Path], set[str]]:
    """({module: file} of ``start`` and what they import, transitively, that a `Security` package's sources hold; the modules of
    ``start`` none holds)."""
    out: dict[str, Path] = {}
    todo = sorted(start)
    while todo:
        m = todo.pop()
        if m in out:
            continue
        f = next((f for s in sources if (f := s / (m.replace(".", "/") + ".lean")).is_file()), None)
        if f is None:
            continue
        out[m] = f
        todo += header_imports(f.read_text())
    return out, start - out.keys()


def test_the_lean_the_tests_build_and_import_is_declared():
    """Every file of `Security` in the import closure of the modules the tests build or import is under the suite's inputs, and
    the closure holds `Proofs.Flock.Verifier`, which `build_proofs_verifier` builds. A file left out would let its change reuse a
    pass that never compiled it."""
    pkgs, _ = packages()
    sources = sorted({(d / conf.get("srcDir", ".")).resolve() for d, conf in pkgs.items()})
    libs = {lib["name"] for conf in pkgs.values() for lib in conf.get("lean_lib", [])}
    start = roots(libs)
    mods, unheld = closure(start, sources)
    assert "Proofs.Flock.Verifier" in start and not unheld, f"the tests name modules no Security package holds: {sorted(unheld)}"
    inputs = declared()
    missing = sorted(rel(f) for f in mods.values() if not under(rel(f), inputs))
    assert not missing, f"add to [tool.verity.tests] inputs in verity/ml/flock/pyproject.toml: {missing}"


def test_the_packages_lake_loads_are_declared():
    """`lake build` in `security_proofs` loads it and every package it requires by path: their lakefiles, manifests and
    toolchains under `Security` are among the inputs, and the packages outside `Security` lie under an input or the dependency
    `verity`'s directory."""
    pkgs, outside = packages()
    inputs = declared()
    missing = sorted(r for d in pkgs for f in PACKAGE_FILES if (d / f).is_file() and not under(r := rel(d / f), inputs))
    assert not missing, f"add to [tool.verity.tests] inputs in verity/ml/flock/pyproject.toml: {missing}"
    assert {SECURITY, SECURITY / "Properties"} <= pkgs.keys()
    unkeyed = sorted(rel(p) for p in outside if not under(rel(p), [*inputs, DEPENDENCY]))
    assert not unkeyed, f"security_proofs requires packages this suite's key leaves out: {unkeyed}"
