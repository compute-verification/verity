"""The Lean packages in the tree (AGENTS.md "Lean"): each is audited (a `lean-audit.json` beside its lakefile, which `check`'s
`lean-audit` step reads), and all of them build with one toolchain and one Mathlib, so a bump is one change to all of them."""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
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
    assert not missing, f"Lake packages without a lean-audit.json (tools/verity/lean/README.md): {missing}"


def test_one_toolchain():
    """The jobs' base image installs the same one."""
    pins = {str(p.relative_to(ROOT)): (p / "lean-toolchain").read_text().strip() for p in lake_packages()}
    image = ROOT / "tools/research/src/research/pods/nebius/image/Dockerfile"
    pins[str(image.relative_to(ROOT))] = re.search(r"^ARG LEAN_TOOLCHAIN=(\S+)", image.read_text(), re.MULTILINE).group(1)
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
    """A3 (Flock soundness's `ASSUMPTIONS.md`, `Assumptions.UniformRandomBytes`) cites what the toolchain's runtime reads
    for `IO.getRandomBytes` (`lean_io_get_random_bytes`), so a toolchain bump fails here until someone re-reads it and
    updates the citation."""
    proofs = ROOT / "Security/Proofs"
    version = (proofs / "lean-toolchain").read_text().strip().rpartition(":")[2]
    for doc in ("ASSUMPTIONS.md", "Assumptions.lean"):
        assert f"Lean {version}'s runtime" in (proofs / "Flock/Soundness" / doc).read_text(), doc


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
    """`.gitattributes` sends every `lean-audit.json` to `tools/verity/lean/merge.py`, and each record's `reads` digests are the ones
    the driver recomputes from their definitions."""
    spec = importlib.util.spec_from_file_location("lean_merge", ROOT / "tools" / "verity" / "lean" / "merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    assert f"lean-audit.json merge={merge.NAME}" in (ROOT / ".gitattributes").read_text().splitlines()
    stale = [f"{p.relative_to(ROOT)}: {m}" for p in lake_packages()
             for m, g in json.loads((p / "lean-audit.json").read_text()).get("reads", {}).items()
             if g["digest"] != merge.A.module_digest(merge.A.definitions(g))]
    assert not stale, (f"reads digests that are not the SHA-256 of their definitions (`python tools/verity/lean/merge.py --normalize "
                       f"FILE` recomputes them): {stale}")


def test_every_fetching_package_has_a_pinned_bundle():
    """A Lake package that fetches dependencies has a bundle in `lean-deps.json` for its manifest's fetched entries and its
    toolchain (`deps_key`), which is what `check`'s preflight on node 1 restores; without one main's full check can't start.
    A change to a fetched entry needs its pin: a cold `lean_audit.py --export` run, then `--pin RUN`; a
    `path` entry changes no key."""
    spec = importlib.util.spec_from_file_location("lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    pinned = json.loads((ROOT / audit.PIN).read_text())["deps"]
    missing = [f"{p.relative_to(ROOT)} ({k[:12]})" for p in lake_packages() if (k := audit.deps_key(p)) and k not in pinned]
    assert not missing, f"Lake packages whose dependencies {audit.PIN} pins no bundle for: {missing}"


SECURITY = ROOT / "Security"
PROOFS = SECURITY / "Proofs"


def _requires(pkg: Path) -> list[dict]:
    return tomllib.loads((pkg / "lakefile.toml").read_text()).get("require", []) if (pkg / "lakefile.toml").exists() else []


def test_security_package_requires_only_mathlib():
    """The security analysis's trusted text (`Security`: Definitions and Models) is what every guarantee is stated
    in, so it requires nothing of the repository and no ArkLib: a path dependency here would put a component or a proof
    under every guarantee."""
    assert SECURITY in lake_packages()
    assert [r["name"] for r in _requires(SECURITY)] == ["mathlib"]


def test_proofs_require_the_security_package():
    """The proofs (`Security/Proofs`, srcDir `..`) prove what `security` states, so they require it by path."""
    assert PROOFS in lake_packages()
    paths = {(PROOFS / r["path"]).resolve() for r in _requires(PROOFS) if "path" in r}
    assert SECURITY in paths
    assert tomllib.loads((PROOFS / "lakefile.toml").read_text()).get("srcDir") == ".."


def test_code_never_requires_the_security_analysis():
    """Code (Flock's verifier, the PoUS grader, the certifier's program) stays with its component and never imports
    `Security/`: the analysis reads the code, never the reverse."""
    code = [pkg for pkg in lake_packages() if not pkg.is_relative_to(SECURITY) and not pkg.is_relative_to(ROOT / "tools")]
    assert code, "no code packages outside Security"
    bad = [f"{pkg.relative_to(ROOT)} requires {(pkg / r['path']).resolve().relative_to(ROOT)}"
           for pkg in code for r in _requires(pkg) if "path" in r and (pkg / r["path"]).resolve().is_relative_to(SECURITY)]
    assert not bad, bad


def test_every_upstream_watch_is_scanned():
    """The audit's upstream pass reads only a lock's `scan` packages, so a `watch` list with nothing to scan checks none of
    its entries and passes (a lock that loses its `scan` silently stops watching). Each lock that
    watches a dependency for its assumptions scans one, and the proofs' lock scans ArkLib."""
    bare = [str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob("lean-audit.json")) if ".lake" not in p.parts
            and (up := json.loads(p.read_text()).get("upstream", {})).get("watch") and not up.get("scan")]
    assert not bare, bare
    assert "Arklib" in json.loads((PROOFS / "lean-audit.json").read_text())["upstream"]["scan"]


def test_every_open_conjecture_names_a_test_that_exists():
    """An open entry of a lock's `shape.conjectures` names its falsification test as a pytest node id, `PATH::NAME` from
    the repository's root, and that file defines the test, so each conjecture's test runs."""
    bad = []
    for p in sorted(ROOT.rglob("lean-audit.json")):
        if ".lake" in p.parts:
            continue
        for name, c in json.loads(p.read_text()).get("shape", {}).get("conjectures", {}).items():
            if c.get("status") != "open":
                continue
            path, _, test = str(c.get("test", "")).partition("::")
            node = test.split("[")[0]
            if not (ROOT / path).is_file() or not re.search(rf"^def {re.escape(node)}\(", (ROOT / path).read_text(), re.M):
                bad.append(f"{p.relative_to(ROOT)}: {name}'s test {c.get('test')!r} is not PATH::NAME of a test in the tree")
    assert not bad, bad
