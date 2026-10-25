"""`tools/advice_compile.py --window K`'s inputs: the certified bodies are window K's own requests', so a user the run served
again in a later window keeps the prompt it had in K."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "advice_compile.py"
_spec = importlib.util.spec_from_file_location("advice_compile", SCRIPT)
AC = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("advice_compile", AC)
_spec.loader.exec_module(AC)


def _run(tmp_path: Path, requests: list[dict]) -> Path:
    (tmp_path / "serve.json").write_text(json.dumps({"model": "m", "revision": "r", "target": {}, "default_seed": 7}))
    for k in (0, 3):
        (tmp_path / "advice" / f"w{k}").mkdir(parents=True)
        (tmp_path / "advice" / f"w{k}" / "registration.json").write_text(json.dumps({"window": k}))
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in requests))
    return tmp_path


def _q(user: str, windows: list[int], prompt: list[int], max_tokens: int) -> dict:
    return {"user": user, "http_id": f"h-{user}-{windows}", "windows": windows, "prompt_ids": prompt,
            "ingress": {"brought": {"max_tokens": max_tokens}}}


def test_a_window_reads_its_own_requests_when_a_user_is_served_again(tmp_path):
    run = _run(tmp_path, [_q("u", [0], [1, 2, 3], 8), _q("v", [0, 1], [9], 4), _q("u", [3], [4, 5], 16)])
    _, meta, bodies, reg = AC.served_inputs(run, 0)
    assert bodies == {"u": {"max_tokens": 8, "prompt": [1, 2, 3]}, "v": {"max_tokens": 4, "prompt": [9]}}
    assert reg == {"window": 0} and meta["default_seed"] == 7
    _, _, bodies, _ = AC.served_inputs(run, 3)
    assert bodies == {"u": {"max_tokens": 16, "prompt": [4, 5]}}


def test_a_user_with_two_requests_in_one_window_is_refused(tmp_path):
    run = _run(tmp_path, [_q("u", [3], [1], 8), _q("u", [3], [2], 8)])
    with pytest.raises(ValueError, match="user 'u' has two requests in window 3"):
        AC.served_inputs(run, 3)
