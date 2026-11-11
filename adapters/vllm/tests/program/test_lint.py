"""The new query path decides nothing by a name pattern: no `.startswith(` anywhere under `verity_vllm/query/` or in its CLI command (`pipeline/manifest.py`; v1's family / module
tables were all `family.startswith(...)` / `leaf in _LINEAR_LEAVES` checks), and no reference model family is named."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "verity_vllm"
QUERY_DIR = PKG / "query"
MODEL_NAMES = ("llama", "qwen", "gemma", "olmoe", "mixtral", "smollm", "mistral", "deepseek")
FILES = sorted([*QUERY_DIR.glob("*.py"), PKG / "pipeline" / "manifest.py"])


def test_query_package_exists():
    assert FILES, f"no modules under {QUERY_DIR}"
    assert {f.name for f in FILES} >= {"program_view.py", "module_body.py", "required.py", "manifest.py"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_startswith(path: Path):
    src = path.read_text()
    hits = [i + 1 for i, line in enumerate(src.splitlines()) if ".startswith(" in line]
    assert not hits, f"{path.name}: `.startswith(` at lines {hits} -- a name-pattern decision"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_model_names(path: Path):
    src = path.read_text().lower()
    hits = sorted({m for m in MODEL_NAMES if re.search(rf"\b{m}", src)})
    assert not hits, f"{path.name} names model families {hits}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_v1_membership_tables(path: Path):
    """The v1 by-name tables are not imported or re-declared here (the bridge imports v1 only for the identity encoding + kernel geometry)."""
    src = path.read_text()
    for name in ("_LINEAR_LEAVES", "_DENSE_INTERIOR", "MEMBER_RULES", "FIRST_NORM_SUFFIX", "FAMILY_OF_MODULE", "members_for(", "is_epilogue_interior", "epilogue_modules"):
        assert name not in src, f"{path.name} uses v1's by-name table {name}"
