"""`lean_split.py`, the script that splits PoUS's Lean package into its spec and `security_proofs/pous`, holds on this tree:
the spec is separable, every module is placed, the proofs' globs claim exactly the proof modules, and the lock splits with
every non-package section unchanged. A failure here means the split would refuse when it runs."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pous_lean_split", HERE / "lean_split.py")
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)


def _matches(glob: str, m: str) -> bool:
    return m.startswith(glob[:-2] + ".") if glob.endswith(".+") else m == glob


@pytest.fixture(scope="module")
def plan():
    return split.Plan(split.REPO)


def test_the_spec_separates(plan):
    assert not plan.done and not plan.partial
    assert not plan.problems, "\n".join(plan.problems)
    assert plan.spec | plan.proofs | {m for m, p in plan.placed.items() if p == "stay"} == set(plan.mods)
    assert set(split.SPEC_EXTRA) <= plan.spec and "Pous" in plan.proofs


def test_the_proofs_globs_claim_exactly_the_proof_modules(plan):
    globs = plan.globs()
    claimed = {m for m in plan.mods if any(_matches(g, m) for g in globs)}
    assert claimed == plan.proofs - set(split.PROOF_EXTRA)


def test_the_lock_splits_only_its_package_sections(plan):
    for k, v in plan.lock.items():
        if k not in split.PACKAGE_SECTIONS:
            assert plan.spec_lock[k] == v, k
    assert set(plan.spec_lock["exempt"]) | set(plan.proofs_lock.get("exempt", {})) == set(plan.lock["exempt"])
    assert plan.proofs_lock["roots"] == ["Pous"]
    assert plan.spec_lock.get("dependencies") == plan.lock.get("dependencies") == plan.proofs_lock.get("dependencies")


def test_the_manifest_layout_is_lakes(plan):
    text = (plan.spec_dir / "lake-manifest.json").read_text()
    assert split.manifest_text(split.json.loads(text)) == text


def _copy(tmp: Path) -> Path:
    src = split.REPO / split.SPEC
    dst = tmp / split.SPEC
    for f in [*src.rglob("*.lean"), *(src / n for n in ("lean-audit.json", "TRUSTED.sha256", "grade.sh"))]:
        if ".lake" not in f.relative_to(src).parts:
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.relative_to(src))
    return tmp


def test_a_spec_module_importing_a_proof_is_refused(tmp_path):
    repo = _copy(tmp_path)
    f = repo / split.SPEC / "Pous/Protocol/Model/Inc.lean"
    f.write_text("import Pous.SecurityProofs.P2.Count\n" + f.read_text())
    assert "spec module Pous.Protocol.Model.Inc imports Pous.SecurityProofs.P2.Count, which is not spec" in \
        split.Plan(repo).problems


def test_an_unplaced_module_and_a_pin_reading_a_proof_are_refused(tmp_path):
    repo = _copy(tmp_path)
    (repo / split.SPEC / "PousScratch.lean").write_text("def x := 1\n")
    problems = split.Plan(repo).problems
    assert any(p.startswith("PousScratch (PousScratch.lean): placed by no rule") for p in problems)
    sanity = repo / split.SPEC / "Pous/SecurityProofs/Sanity.lean"
    sanity.rename(repo / split.SPEC / "Pous/SecurityProofs/Sanity2.lean")
    problems = split.Plan(repo).problems
    assert "pin Pous.junk_not_meets reads Pous.junk, declared in Pous.SecurityProofs.Sanity2, which is not spec" in problems


def test_a_split_tree_is_left_alone(tmp_path):
    repo = _copy(tmp_path)
    plan = split.Plan(repo)
    for a, b in plan.moves():
        (repo / b).parent.mkdir(parents=True, exist_ok=True)
        (repo / a).rename(repo / b)
    assert split.Plan(repo).partial
    (repo / split.PROOFS / "lakefile.toml").write_text("")
    assert split.Plan(repo).done and split.main(["--repo", str(repo)]) == 0
