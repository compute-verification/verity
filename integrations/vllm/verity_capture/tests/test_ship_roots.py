"""[R17 rel2 2026-09-21, rel M-1320] `data/hf_configs` ships to pods and lives in the R17 sparse worktrees.

The configuration-derived profiles (`profiles.generic` since P6; before it gen_moe_olmoe, gen_dense2_*, gen_ln_*, dense_generic) and their tests read
`data/hf_configs/<ROLE>.config.json` from the tree.  Before this fix the directory was (a) excluded from the rel sparse worktree
(`!/out/gen/*` with no re-include) and (b) absent from `record_v5/ship.sh` ROOTS, so on every pod tree since r17-int-1
`test_gen_ov_moe*.py` errored at collection (OLMOE.config.json absent) and `test_gen_dense2.py` / `test_gen_ln.py` skipped.
Torch-free: text checks on the two files + one `ship.sh --pack` into a temp dir, read back with tar."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHIP = REPO / "record_v5" / "ship.sh"
PATTERNS = REPO / "out" / "gen" / "r17" / "sparse-patterns.txt"
HF_DIR = REPO / "data" / "hf_configs"
ROLES = ("OLMOE", "LLAMA32_1B", "GEMMA2_2B", "PHI3_MINI", "PYTHIA160M", "B7")


def _roots() -> list[str]:
    m = re.search(r'^ROOTS="([^"]*)"', SHIP.read_text(), re.M)
    assert m, "record_v5/ship.sh: no ROOTS= line"
    return shlex.split(m.group(1))


def test_ship_roots_name_out_gen_hf_configs():
    roots = _roots()
    assert "data/hf_configs" in roots
    # the roots stay distinct and the existing ones are untouched (the worker package must contain what the row runner reads)
    assert len(roots) == len(set(roots))
    for r in ("verity_capture", "record_v5", "verity.ir", "workloads"):
        assert r in roots


def test_r17_sparse_patterns_re_include_hf_configs():
    lines = [ln.strip() for ln in PATTERNS.read_text().splitlines() if ln.strip()]
    assert "!/out/gen/*" in lines, "the exclusion this line re-includes under is gone: re-read the pattern file"
    assert "/data/hf_configs/" in lines
    # a re-include must come AFTER the exclusion it overrides (git sparse-checkout --no-cone: last match wins)
    assert lines.index("/data/hf_configs/") > lines.index("!/out/gen/*")


def test_tracked_hf_configs_are_the_roles_the_profiles_read():
    if not HF_DIR.is_dir():
        pytest.skip("data/hf_configs not in this checkout (pre-fix sparse worktree or pod tree shipped before the int-9 roots)")
    have = {p.name for p in HF_DIR.glob("*.config.json")}
    for role in ROLES:
        assert f"{role}.config.json" in have, f"{role}.config.json missing from data/hf_configs"
    for p in HF_DIR.glob("*.config.json"):
        json.loads(p.read_text())     # every shipped config is a JSON document (the profiles json.load them at import)


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("tar") is None, reason="needs git + tar")
def test_ship_pack_carries_out_gen_hf_configs(tmp_path):
    """`ship.sh --pack <dir>` (the runner's snapshot form; the same ROOTS as the in-place ship) must put every tracked hf config into
    src.tgz -- from this worktree when it has the directory, else from SHIP_FALLBACK_REPO, recorded in EXPORT.json."""
    fallback = os.environ.get("SHIP_FALLBACK_REPO", "/Users/danielreuter/projects/veritor")
    if not HF_DIR.is_dir() and not (Path(fallback) / "data" / "hf_configs").is_dir():
        pytest.skip("neither this checkout nor the fallback repo has data/hf_configs")
    if subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True).returncode != 0:
        pytest.skip("not a git checkout (pod tree)")
    out = tmp_path / "pack"
    r = subprocess.run(["bash", str(SHIP), "--pack", str(out)], cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr + r.stdout
    with tarfile.open(out / "src.tgz", "r:gz") as tf:
        names = set(tf.getnames())
    for role in ROLES:
        assert f"data/hf_configs/{role}.config.json" in names, f"{role} not in the shipped package"
    export = json.loads((out / "EXPORT.json").read_text())
    fb = export.get("fallback_roots")
    if HF_DIR.is_dir():
        assert fb is None or "data/hf_configs" not in fb["roots"], "the worktree has the directory: it must ship from HERE, not the fallback"
    else:
        assert fb is not None and "data/hf_configs" in fb["roots"], "shipped from the fallback repo: EXPORT.json must record it"
