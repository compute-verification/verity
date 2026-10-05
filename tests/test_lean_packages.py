"""The Lean packages in the tree (AGENTS.md "Lean"): each is audited (a `lean-audit.json` beside its lakefile, which `check`'s
`lean-audit` step reads), and all of them build with one toolchain and one Mathlib, so a bump is one change to all of them."""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".lake", ".git", ".venv", "node_modules", "__pycache__"}


def lake_packages() -> list[Path]:
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if "lakefile.toml" in files or "lakefile.lean" in files:
            out.append(Path(d))
    return out


def test_every_lake_package_is_audited():
    pkgs = lake_packages()
    assert len(pkgs) >= 3
    missing = [str(p.relative_to(ROOT)) for p in pkgs if not (p / "lean-audit.json").exists()]
    assert not missing, f"Lake packages without a lean-audit.json (tools/lean/README.md): {missing}"


def test_one_toolchain():
    pins = {str(p.relative_to(ROOT)): (p / "lean-toolchain").read_text().strip() for p in lake_packages()}
    assert len(set(pins.values())) == 1, pins


def test_one_mathlib():
    revs = {}
    for p in lake_packages():
        manifest = p / "lake-manifest.json"
        if manifest.exists():
            for dep in json.loads(manifest.read_text())["packages"]:
                if dep["name"] == "mathlib":
                    revs[str(p.relative_to(ROOT))] = dep["rev"]
    assert revs and len(set(revs.values())) == 1, revs


def test_the_randomness_assumption_cites_this_toolchain():
    """A3 (`soundness/ASSUMPTIONS.md`, `Assumptions.UniformRandomBytes`) cites what the toolchain's runtime reads for
    `IO.getRandomBytes` (`lean_io_get_random_bytes`), so a toolchain bump fails here until someone re-reads it and updates
    the citation."""
    soundness = ROOT / "backends/flock/verifier/lean/soundness"
    version = (soundness / "lean-toolchain").read_text().strip().rpartition(":")[2]
    for doc in ("ASSUMPTIONS.md", "FlockSoundness/Assumptions.lean"):
        assert f"Lean {version}'s runtime" in (soundness / doc).read_text(), doc


def _module_roots(pkg: Path):
    """(source directory, root module) for each library and executable of `pkg`, as Lake resolves them."""
    cfg = tomllib.loads((pkg / "lakefile.toml").read_text())
    src = pkg / cfg.get("srcDir", ".")
    for lib in cfg.get("lean_lib", []):
        for r in lib.get("roots", [lib["name"]]) + [g.rstrip("+*").rstrip(".") for g in lib.get("globs", [])]:
            yield (src / lib.get("srcDir", ".")).resolve(), r
    for exe in cfg.get("lean_exe", []):
        yield (src / exe.get("srcDir", ".")).resolve(), exe.get("root", exe["name"])


def test_lean_module_roots_are_disjoint():
    """No module name is claimed from two source directories, by two packages or by one. Lake resolves an import by name
    across every library in the workspace and doesn't report a duplicate: a library whose roots overlap another's builds
    the other's module from its own directory, without a warning. Within one package, roots in different source
    directories may share a namespace but not nest (`Pouw.Protocol` and `Pouw.Guarantees` may; `Pouw` and
    `Pouw.Guarantees` may not). Across packages they may not even share a first component: with `Pk.A` in one package
    and `Pk.B` in another, `lake build` passes but `lake env lean` looks for every `Pk.*` under the first package's
    build directory and fails to find `Pk.B`."""
    claims = []
    for pkg in lake_packages():
        assert (pkg / "lakefile.toml").exists(), f"{pkg.relative_to(ROOT)}: only a lakefile.toml can be read here"
        claims += [(pkg, src, root) for src, root in _module_roots(pkg)]

    def nested(a: str, b: str) -> bool:
        a, b = a.split("."), b.split(".")
        return a[: len(b)] == b or b[: len(a)] == a

    def clash(x, y) -> bool:
        if x[0] != y[0]:
            return x[2].split(".")[0] == y[2].split(".")[0]
        return x[1] != y[1] and nested(x[2], y[2])

    clashes = [f"{x[2]} ({x[0].relative_to(ROOT)}, {x[1].relative_to(ROOT)}) and "
               f"{y[2]} ({y[0].relative_to(ROOT)}, {y[1].relative_to(ROOT)})"
               for x, y in itertools.combinations(claims, 2) if clash(x, y)]
    assert len(claims) >= 10
    assert not clashes, clashes


def test_every_record_merges_with_the_driver():
    """`.gitattributes` sends every `lean-audit.json` to `tools/lean/merge.py`, and each record's `reads` digests are the ones
    the driver recomputes from their definitions."""
    spec = importlib.util.spec_from_file_location("lean_merge", ROOT / "tools" / "lean" / "merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    assert f"lean-audit.json merge={merge.NAME}" in (ROOT / ".gitattributes").read_text().splitlines()
    stale = [f"{p.relative_to(ROOT)}: {m}" for p in lake_packages()
             for m, g in json.loads((p / "lean-audit.json").read_text()).get("reads", {}).items()
             if g["digest"] != merge.A.module_digest(merge.A.definitions(g))]
    assert not stale, (f"reads digests that are not the SHA-256 of their definitions (`python tools/lean/merge.py --normalize "
                       f"FILE` recomputes them): {stale}")


CORE = ROOT / "verity" / "lean"
#: the Lake names of the protocols' packages, so the rule below can't go vacuous when they move
PROTOCOL_PACKAGES = {"nci", "networktiming", "pous", "pouw"}


def _requires(pkg: Path) -> list[dict]:
    return tomllib.loads((pkg / "lakefile.toml").read_text()).get("require", []) if (pkg / "lakefile.toml").exists() else []


def test_core_package_requires_only_mathlib():
    """Core's Lean (`verity/lean`) is the model every protocol states its guarantees in, so it requires nothing of
    the repository: a path dependency here would put a protocol or a backend under every guarantee."""
    assert CORE in lake_packages()
    assert [r["name"] for r in _requires(CORE)] == ["mathlib"]


def test_protocol_packages_require_core_and_protocols_only():
    """A protocol's Lean package may require core's and other protocols' packages (the ruling of Oct 1 in
    `.agents/skills/lean-proofs/SKILL.md`), never a backend's: backends instantiate protocols, not the reverse."""
    protocols = [pkg for pkg in lake_packages() if pkg.relative_to(ROOT).parts[:2] == ("verity", "protocols")]
    names = {tomllib.loads((pkg / "lakefile.toml").read_text())["name"] for pkg in protocols}
    assert PROTOCOL_PACKAGES <= names, f"the protocols' Lake packages aren't under verity/protocols/: {PROTOCOL_PACKAGES - names}"
    bad = []
    for pkg in protocols:
        for r in _requires(pkg):
            if "path" in r:
                target = (pkg / r["path"]).resolve()
                if target != CORE and target.relative_to(ROOT).parts[:2] != ("verity", "protocols"):
                    bad.append(f"{pkg.relative_to(ROOT)} requires {target.relative_to(ROOT)}")
    assert not bad, bad
