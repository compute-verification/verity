"""stoch_recompute's Program S per request from the Build's request results: a readable result gives its shape's `sampling.splits`, and
an unreadable one is named with its error (the check makes it a FAIL reason), never skipped as if the shape had no Build result.
Torch-free."""
from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.check.replay.stoch_recompute import _program_splits


def _build(tmp_path: Path) -> tuple[Path, Path]:
    b = tmp_path / "build"
    (b / "build_request_a").mkdir(parents=True)
    (b / "build_request_a" / "result.json").write_text(json.dumps({"ok": True, "wrapper": "request", "LP": 4, "T": 7,
                                                                   "sampling": {"splits": 3}}))
    (b / "build_request_b").mkdir()
    (b / "build_request_b" / "result.json").write_text('{"ok": true, "wrapper": "request", "LP": 5,')
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 4, "max_tokens": 8},
                                           {"request_id": "r1", "prompt_len": 5, "max_tokens": 8}]}))
    return b, wl


def test_unreadable_build_result_is_named(tmp_path: Path) -> None:
    b, wl = _build(tmp_path)
    splits, unreadable = _program_splits(b, wl)
    assert splits == {"r0": 3, "r1": None}
    assert len(unreadable) == 1
    assert unreadable[0].startswith(f"{b / 'build_request_b' / 'result.json'}: JSONDecodeError(")


def test_no_build_reads_nothing(tmp_path: Path) -> None:
    assert _program_splits(None, None) == ({}, [])
