"""`tools/move_map.txt` covers the tree (lane vllm-skeleton).

The map is the layout of record for the move-only commit: every tracked file of the integration must resolve through it, no two
files may land on one path, a DROP must be a docstring-only `__init__.py`, and the explicit literal rewrites must still find their
literal (or their file must be gone).  On a tree that is already moved, the same assertions hold with zero moves (idempotence).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

relayout = pytest.importorskip("relayout")

pytestmark = pytest.mark.skipif(not (ROOT / ".git").exists() and not (ROOT.parent.parent / ".git").exists(),
                                reason="needs the git tree (git ls-files)")


@pytest.fixture(scope="module")
def mm():
    return relayout.load_map()


@pytest.fixture(scope="module")
def plan(mm):
    return relayout.build_plan(ROOT, mm)


def test_every_tracked_file_is_mapped(plan):
    assert plan.unmapped == [], f"{len(plan.unmapped)} tracked files have no home in the map: {plan.unmapped[:20]}"


def test_no_two_files_land_on_one_path(plan):
    assert plan.collisions == {}, plan.collisions


def test_drop_only_removes_docstring_only_inits(plan):
    assert plan.drop_refused == [], plan.drop_refused
    assert all(f.endswith("__init__.py") for f in plan.drops), plan.drops


def test_every_move_stays_inside_the_skeleton(plan):
    roots = ("verity_vllm/", "tests/", "workloads/", "manifests/", "fixtures/", "data/", "docs/")
    bad = [(o, n) for o, n, _ in plan.moves if not n.startswith(roots) and "/" in n]
    assert bad == [], bad
    # the v1-era package names do not survive as directories
    assert not any(n.startswith(("verity_capture/", "verity_vllm_adapter/", "verity_vllm_numerics/")) for _, n, _ in plan.moves)


def test_lane_tags_name_known_lanes(mm):
    known = {"vllm-dead-code", "vllm-profiles-generic", "vllm-tp-n", "vllm-v2-default", "vllm-v2-gm-scale", "vllm-v2-memory",
             "vllm-v2-research-tools", "vllm-cleanup-1", "vllm-cleanup-2", "ghost"}
    lanes = {lane.lstrip("+") for _, lane in mm.exact.values() if lane} | {lane for _, _, lane in mm.prefix if lane}
    assert lanes <= known, lanes - known


def test_explicit_rewrites_find_their_literal(mm, plan):
    """A REWRITE line whose file exists must still match verbatim (a lane edited the line -> the map needs updating); a file that
    was already moved must not hold the OLD literal any more."""
    new_of = {o: n for o, n, _ in plan.moves}
    stale = []
    for f, old, new in mm.rewrites:
        p = ROOT / f
        if p.exists():
            text = p.read_text(errors="replace")
            if old not in text and not (new == "<DEL>" or new in text):    # neither the old literal nor the rewritten one
                stale.append((f, old))
        else:
            q = ROOT / new_of.get(f, f)
            if q.exists() and old in q.read_text(errors="replace"):
                stale.append((str(q.relative_to(ROOT)), old))
    assert stale == [], stale


def test_moved_tree_is_a_fixed_point(mm, plan):
    """Simulate the move: resolving every NEW path through the map must give back the same path."""
    for _, new, _ in plan.moves:
        r = mm.resolve(new)
        assert r is not None and r[0] == new, (new, r)


def test_tests_mirror_the_package(plan):
    """Every test lands under tests/<subpackage>/ for a subpackage the map creates (or tests/regression, tests/query, tests/acquire)."""
    subs = {n.split("/")[1] for _, n, _ in plan.moves if n.startswith("verity_vllm/") and n.count("/") >= 2}
    subs |= {"regression", "query", "acquire"}
    for _, new, _ in plan.moves:
        if new.startswith("tests/") and os.path.basename(new).startswith("test_"):
            assert new.split("/")[1] in subs, new
