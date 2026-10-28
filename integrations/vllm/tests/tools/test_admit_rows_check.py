"""`tools/admit_rows_check.py` on a serve dir of two windows: a step whose new rows are in admission order passes, and one
laid out as endpoint 2's window 237 was (the later-admitted, shorter prompt's first row first) is an inversion, named, in
both the step trace and the committed rows."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "admit_rows_check.py"
_spec = importlib.util.spec_from_file_location("admit_rows_check", SCRIPT)
AR = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("admit_rows_check", AR)
_spec.loader.exec_module(AR)


def serve_dir(tmp_path: Path, w1_rows: list[str]) -> Path:
    """Window 0: a (40 tokens) then b (10), each admitted and first scheduled alone in turn, a decoding at b's step. Window 1:
    c (128) then d (81) admitted, first scheduled in one step beside b's decode, laid out as ``w1_rows``."""
    prompt = {"a": 40, "b": 10, "c": 128, "d": 81}
    trace = [{"kind": "header", "schema": "verity-vllm/step-trace/v1", "config": {}},
             *({"kind": "arrive", "request": r, "before_step": 0, "prompt_len": n} for r, n in prompt.items()),
             {"kind": "window", "window": 0, "event": "open", "before_step": 0},
             {"kind": "step", "step": 0, "order": "runner", "rows": [["a", 0, 40]]},
             {"kind": "step", "step": 1, "order": "runner", "rows": [["a", 40, 1], ["b", 0, 10]]},
             {"kind": "window", "window": 0, "event": "close", "before_step": 2},
             {"kind": "window", "window": 1, "event": "open", "before_step": 2},
             {"kind": "step", "step": 2, "order": "runner", "rows": [["b", 10, 1], *([r, 0, prompt[r]] for r in w1_rows)]}]
    lo, rows = 1, {"a": [[0, 0, 0, 40, 0, 40, "prefill"], [0, 1, 0, 1, 40, 40, "decode"]], "b": [[0, 1, 1, 11, 0, 10, "prefill"]]}
    for r in w1_rows:
        rows[r] = [[1, 0, lo, lo + prompt[r], 0, prompt[r], "prefill"]]
        lo += prompt[r]
    reqs = [{"engine_id": r, "user": f"u.{r}", "admitted": i, "release": ["run", i], "prompt_ids": [0] * prompt[r], "rows": rows[r]}
            for i, r in enumerate("abcd")]
    (tmp_path / "advice_trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in trace))
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reqs))
    (tmp_path / "serve.json").write_text(json.dumps({"admit": "release-order", "step_rows": "scheduler"}))
    return tmp_path


def test_new_rows_in_admission_order_pass(tmp_path):
    rep = AR.check(serve_dir(tmp_path, ["c", "d"]))
    assert rep["passed"] and rep["admitted_in_release_order"] and rep["released"] == 4
    for src in ("trace", "committed"):
        assert rep[src]["inversions"] == 0 and rep[src]["numbering_is_admission"] and set(rep[src]["windows"]) == {"0", "1"}
    assert rep["layout"]["decode_after_prefill"] == [] and rep["layout"]["mixed"] == 2


def test_a_shorter_later_prompt_first_is_an_inversion(tmp_path, capsys):
    rep = AR.check(serve_dir(tmp_path, ["d", "c"]))
    assert not rep["passed"]
    for src in ("trace", "committed"):
        w1 = rep[src]["windows"]["1"]
        assert (w1["inversions"], w1["same_step"]) == (1, 1) and rep[src]["windows"]["0"]["inversions"] == 0
        assert [x["user"] for x in w1["examples"][0]] == ["u.d", "u.c"]
    assert AR.main([str(tmp_path)]) == 1 and "ADMIT-ROWS FAIL" in capsys.readouterr().out


def test_a_missing_file_is_its_own_exit(tmp_path):
    assert AR.main([str(tmp_path)]) == 2
