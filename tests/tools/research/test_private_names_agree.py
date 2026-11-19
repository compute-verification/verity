"""The two readers of the tree's one list of private names, the vLLM adapter's matcher (`private_names.py`, loaded by path:
`research` imports no package of the tree) and custody's (`research.store.private`), give the same answer on every case."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from research.store import private as P

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("adapter_private_names", ROOT / "adapters/vllm/verity_vllm/pipeline/private_names.py")
ADAPTER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ADAPTER)

#: (path relative to a run's or a row's directory, private by the list?)
CASES = [
    ("tokens.json", True),                                          # a plain name
    ("requests.jsonl", True),
    ("commit/tokens_p0.json", True),                                # a glob, one directory down
    ("serve/private/server/ids.3.jsonl.gz", True),                  # a glob, nested directories
    ("a/b/c/step_trace.4127.jsonl.gz", True),
    ("y/7.salts", True),
    ("load_0.jsonl", True),
    ("commit/replay_bundle_p0/commit/values.json", True),           # under a kept-apart directory
    ("rows/r1/commit/replay_slim_p2/x/y/slim.json", True),          # nested under one
    ("replay_bundle_p1.tar", True),                                 # a file whose own name has the prefix
    ("commit/verdict.json", False),                                 # not private
    ("build_summary.json", False),
    ("commit/replay_bundle/values.json", False),                    # a directory short of the prefix
    ("notes/tokens.json.txt", False),                               # a name the glob doesn't match whole
    ("timeline.md", False),
]


@pytest.mark.parametrize("rel,expected", CASES, ids=[r for r, _ in CASES])
def test_both_readers_answer_alike_and_as_expected(tmp_path, rel, expected):
    doc = json.loads((ROOT / P.TREE_NAMES).read_text())
    adapter = ADAPTER.private(rel, ADAPTER.load(ROOT / P.TREE_NAMES))
    custody = P.rule_for(rel, tmp_path / rel, rules=P.names_of(doc, P.TREE_NAMES)) is not None
    floor = P.rule_for(rel, tmp_path / rel, rules=P.NAMES) is not None
    assert adapter == custody == floor == expected, f"{rel}: adapter {adapter}, custody {custody}, custody's floor {floor}"


def test_both_read_the_one_list():
    assert ADAPTER.LIST == ROOT / P.TREE_NAMES and P.floor_text() == ADAPTER.LIST.read_text()
