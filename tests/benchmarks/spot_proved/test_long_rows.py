"""`long_rows.split`: a replay's steps, rows and sampled tokens are compared with the serve on each side of the long-row split
(more than `keys` keys), a token's side read from its prompt length and position, and an aborted request's rows are compared
while its tokens, which its client never received, are not."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _long_rows():
    spec = importlib.util.spec_from_file_location("spot_long_rows", HERE.parent / "long_rows.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _window() -> dict:
    # request a: a 6-key prompt, decoded to 9 keys; b: a 3-key prompt, decoded to 4; c: 8 keys of prompt, aborted after step 1
    steps = [{"step": 0, "rows": [["a", 0, 6], ["b", 0, 3]]}, {"step": 1, "rows": [["a", 6, 1], ["b", 3, 1], ["c", 0, 8]]},
             {"step": 2, "rows": [["a", 7, 1], ["c", 8, 1]]}, {"step": 3, "rows": [["a", 8, 1]]}]
    reqs = [{"engine_id": "a", "prompt_ids": [0] * 6, "served": [5, 6, 7, 8]}, {"engine_id": "b", "prompt_ids": [0] * 3, "served": [1, 2]},
            {"engine_id": "c", "prompt_ids": [0] * 8, "served": []}]
    return {"steps": steps, "requests": reqs, "aborts": [(2, "c")]}


def test_a_replay_equal_to_the_serve_differs_nowhere_and_each_side_is_counted() -> None:
    lr, w = _long_rows(), _window()
    r = lr.split(w, {s["step"]: copy.deepcopy(s["rows"]) for s in w["steps"]}, {q["engine_id"]: list(q["served"]) for q in w["requests"]}, 6)
    assert {k: r["long"][k] for k in ("steps", "rows", "tokens", "requests")} == {"steps": 3, "rows": 5, "tokens": 3, "requests": 1}
    assert {k: r["short"][k] for k in ("steps", "rows", "tokens", "requests")} == {"steps": 1, "rows": 3, "tokens": 3, "requests": 2}
    assert all(r[s][k] == 0 for s in ("long", "short") for k in ("steps_differing", "rows_differing", "tokens_differing"))
    assert r["aborted"] == [{"request": "c", "after_step": 2, "prompt": 8, "rows": 2, "rows_reproduced": 2}]
    assert r["served"]["decode_max_keys"] == 9 and r["served"]["prefill_max_keys"] == 8
    assert r["served"]["longest_request_decoded_to_keys"] == 9


def test_a_long_token_and_a_missing_row_count_on_their_own_side() -> None:
    lr, w = _long_rows(), _window()
    mine = {s["step"]: copy.deepcopy(s["rows"]) for s in w["steps"]}
    mine[2].remove(["c", 8, 1])
    tokens = {q["engine_id"]: list(q["served"]) for q in w["requests"]}
    tokens["a"][2] ^= 1
    tokens["c"] = [9, 9]
    r = lr.split(w, mine, tokens, 6)
    assert (r["long"]["rows_differing"], r["long"]["steps_differing"], r["long"]["tokens_differing"]) == (1, 1, 1)
    assert (r["short"]["rows_differing"], r["short"]["tokens_differing"]) == (0, 0)
    assert r["aborted"][0]["rows_reproduced"] == 1
