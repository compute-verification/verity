"""The Lean packages in the tree (AGENTS.md "Lean"): each is audited (a `lean-audit.json` beside its lakefile, which `check`'s
`lean-audit` step reads), and all of them build with one toolchain and one Mathlib, so a bump is one change to all of them."""
from __future__ import annotations

import importlib.util
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


CORE = ROOT / "packages" / "verity" / "lean"
#: the Lake names of the protocols' packages, so the rule below can't go vacuous when they move
PROTOCOL_PACKAGES = {"nci", "networktiming", "pous", "pouw"}


def _requires(pkg: Path) -> list[dict]:
    return tomllib.loads((pkg / "lakefile.toml").read_text()).get("require", []) if (pkg / "lakefile.toml").exists() else []


def test_core_package_requires_only_mathlib():
    """Core's Lean (`packages/verity/lean`) is the model every protocol states its guarantees in, so it requires nothing of
    the repository: a path dependency here would put a protocol or a backend under every guarantee."""
    assert CORE in lake_packages()
    assert [r["name"] for r in _requires(CORE)] == ["mathlib"]


def test_protocol_packages_require_core_and_protocols_only():
    """A protocol's Lean package may require core's and other protocols' packages (the ruling of Oct 1 in
    `.agents/skills/lean-proofs/SKILL.md`), never a backend's: backends instantiate protocols, not the reverse."""
    protocols = [pkg for pkg in lake_packages() if pkg.relative_to(ROOT).parts[0] == "protocols"]
    names = {tomllib.loads((pkg / "lakefile.toml").read_text())["name"] for pkg in protocols}
    assert PROTOCOL_PACKAGES <= names, f"the protocols' Lake packages aren't under protocols/: {PROTOCOL_PACKAGES - names}"
    bad = []
    for pkg in protocols:
        for r in _requires(pkg):
            if "path" in r:
                target = (pkg / r["path"]).resolve()
                if target != CORE and target.relative_to(ROOT).parts[0] != "protocols":
                    bad.append(f"{pkg.relative_to(ROOT)} requires {target.relative_to(ROOT)}")
    assert not bad, bad
