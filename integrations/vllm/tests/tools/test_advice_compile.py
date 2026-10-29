"""`tools/advice_compile.py --window K`'s inputs: the certified bodies are window K's own requests', so a user the run served
again in a later window keeps the prompt it had in K; K's carry is the close of the last window before it that ran steps; and the
default seed is the public one (`engine.served_seed`), whatever the run's server says it filled in."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from tests.pipeline.test_window_advice import W1, W2
from verity_vllm.pipeline import window_advice as WA

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


def test_the_registered_default_seed_is_the_public_one_and_a_run_that_used_another_is_refused():
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    assert AC.registered_seed({"default_seed": DEFAULT_SEED}) == DEFAULT_SEED
    for seed in (7, 0, None, str(DEFAULT_SEED), float(DEFAULT_SEED)):
        why = f"filled in default seed {seed!r}, and the registered one is engine.served_seed."
        with pytest.raises(ValueError, match=re.escape(why)):
            AC.registered_seed({"default_seed": seed})
    with pytest.raises(ValueError, match="filled in default seed None"):
        AC.registered_seed({})


def test_a_user_with_two_requests_in_one_window_is_refused(tmp_path):
    run = _run(tmp_path, [_q("u", [3], [1], 8), _q("u", [3], [2], 8)])
    with pytest.raises(ValueError, match="user 'u' has two requests in window 3"):
        AC.served_inputs(run, 3)


def _windows(tmp_path: Path, windows: list[int], empty: list[int], advice: dict[int, WA.Advice]) -> Path:
    (tmp_path / "windows.jsonl").write_text("".join(json.dumps({"window": k, "empty": k in empty}) + "\n" for k in windows))
    for k, a in advice.items():
        (tmp_path / "advice" / f"w{k}" / "private").mkdir(parents=True)
        (tmp_path / "advice" / f"w{k}" / "private" / "advice.bin").write_bytes(WA.encode(a))
    return tmp_path


def test_a_window_carries_in_what_the_last_window_that_ran_steps_left(tmp_path):
    run = _windows(tmp_path, [0, 1, 2], [1], {0: W1})
    assert AC.carry_of(run, 0, WA.encode(W1)) == {"in": [], "out": ["a"], "follows": None}
    assert AC.carry_of(run, 2, WA.encode(W2)) == {"in": ["a"], "out": [], "follows": 0}


def test_a_carry_the_last_window_didnt_leave_is_refused(tmp_path):
    run = _windows(tmp_path, [0, 1], [], {0: W2})
    with pytest.raises(ValueError, match=r"window 1 doesn't open where window 0 closed: carried request 0 \('a'\) .* not as the last"):
        AC.carry_of(run, 1, WA.encode(W2))


def test_a_carry_with_no_advice_before_it_is_refused(tmp_path):
    run = _windows(tmp_path, [0, 1], [], {})
    with pytest.raises(ValueError, match="window 1 carries 1 requests in, 'a' first, and window 0, the one before it, has no advice"):
        AC.carry_of(run, 1, WA.encode(W2))
    with pytest.raises(ValueError, match="no window ran before it"):
        AC.carry_of(run, 0, WA.encode(W2))
