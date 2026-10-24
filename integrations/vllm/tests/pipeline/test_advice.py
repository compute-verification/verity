"""`verity-vllm advice`: the advice of a run recorded on the stand-in scheduler (`tests.engine.fake_scheduler`) under varied
traffic, its round trip, its windows, and the refusals when the step trace and the Commit record disagree."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, drive
from verity_vllm.config import UsageError
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_draft as I1


def _served(tmp_path, *, seed: int = 0, n: int = 12, preempt_at=None, gap: bool = True):
    """A run of `n` requests with varied prompt and output lengths, some ending at EOS, arriving while others decode, with an idle
    gap before the last two when `gap`; writes step_trace.jsonl and requests.jsonl under tmp_path."""
    rng = random.Random(seed)
    reqs, arrivals, at = [], {}, 0
    for i in range(n):
        prompt, cap = rng.randint(1, 40), rng.randint(1, 12)
        outputs = [rng.randint(3, 99) for _ in range(cap)]
        if rng.random() < 0.5:
            outputs[rng.randrange(cap)] = 2                     # EOS before the cap
        r = FakeRequest(f"e{i}", [rng.randint(3, 99) for _ in range(prompt)], outputs, cap,
                        sampling_params=SimpleNamespace(temperature=0.0, max_tokens=cap))
        at += 60 if gap and i == n - 2 else rng.randint(0, 3)
        arrivals.setdefault(at, []).append(r)
        reqs.append(r)
    return _run(tmp_path, arrivals, preempt_at=preempt_at)


def _run(tmp_path, arrivals: dict[int, list[FakeRequest]], **sched_kw) -> list[FakeRequest]:
    sched = FakeScheduler(budget=16, max_seqs=4, **sched_kw)
    trace = ST.StepTrace(tmp_path / "step_trace.jsonl").install(sched)
    drive(sched, arrivals)
    trace.uninstall()
    reqs = [r for rs in arrivals.values() for r in rs]
    with (tmp_path / "requests.jsonl").open("w") as f:
        for r in reqs:
            f.write(json.dumps({"request": r.request_id, "user": f"user-{r.request_id}", "prompt_ids": r.prompt_token_ids,
                                "output_ids": r.emitted}) + "\n")
    return reqs


def test_a_varied_run_gives_advice_that_round_trips_with_a_window_per_idle_gap(tmp_path):
    reqs = _served(tmp_path)
    rep = ADV.write(tmp_path)
    assert rep["roundtrip"] is True
    a = I1.decode((tmp_path / "advice" / "advice.bin").read_bytes())
    assert [q.id for q in a.requests] == [f"user-e{i}" for i in range(len(reqs))]
    assert [(q.prompt_len, q.output_len) for q in a.requests] == [(len(r.prompt_token_ids), len(r.emitted)) for r in reqs]
    assert {q.finish for q in a.requests} == {"eos", "length"}
    assert [q.finish == "eos" for q in a.requests] == [r.status == "stopped" for r in reqs]
    c = rep["counts"]
    assert c["chunked_prefill_rows"] > 0 and c["mixed_steps"] > 0 and c["arrivals_during_decode"] > 0
    assert len(rep["windows"]) == 2 and rep["windows"][0]["window"][1] == rep["windows"][1]["window"][0]
    assert sum(w["steps"] for w in rep["windows"]) == c["steps"]
    assert all(0 < w["bits"] < rep["advice"]["bits"] for w in rep["windows"])
    # every scheduled token is in exactly one row, and each request's rows tile [0, prompt + output - 1)
    for i, q in enumerate(a.requests):
        spans = [(r.start, r.n) for s in a.steps for r in s.rows if r.request == i]
        assert sum(n for _, n in spans) == q.prompt_len + q.output_len - 1
        assert [s for s, _ in spans] == [sum(n for _, n in spans[:k]) for k in range(len(spans))]


def test_a_preempted_request_shows_a_zero_length_row_and_recomputes_from_zero(tmp_path):
    _run(tmp_path, {0: [FakeRequest("e0", [5] * 4, [6] * 6, 6), FakeRequest("e1", [5] * 4, [6] * 6, 6)]}, preempt_at={2: "e0"})
    rep = ADV.write(tmp_path)
    a = I1.decode((tmp_path / "advice" / "advice.bin").read_bytes())
    pre = [(s, r) for s, st in enumerate(a.steps) for r in st.rows if r.preempted]
    assert rep["roundtrip"] and rep["counts"]["preempted_rows"] == 1 and pre[0][1].n == 0
    s = pre[0][0]
    assert next(r for st in a.steps[s + 1:] for r in st.rows if r.request == 0).start == 0


@pytest.mark.parametrize("seed", range(4))
def test_the_round_trip_holds_over_seeds(tmp_path, seed):
    _served(tmp_path, seed=seed, n=16)
    assert ADV.write(tmp_path)["roundtrip"] is True


def test_a_commit_record_whose_output_disagrees_with_the_trace_is_refused(tmp_path):
    _served(tmp_path, n=4)
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    doc = json.loads(lines[1])
    doc["output_ids"] = doc["output_ids"] + [7]
    lines[1] = json.dumps(doc)
    (tmp_path / "requests.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="request e1: the trace emitted .* the Commit record holds"):
        ADV.write(tmp_path)


def test_a_commit_record_missing_a_request_is_refused(tmp_path):
    _served(tmp_path, n=4)
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    (tmp_path / "requests.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="request e3 is in the trace but not in the Commit record"):
        ADV.write(tmp_path)


def test_a_row_that_skips_or_repeats_positions_is_refused(tmp_path):
    _served(tmp_path, n=4, gap=False)
    recs = ST.load(tmp_path / "step_trace.jsonl")
    step = next(r for r in recs if r["kind"] == "step" and any(n == 1 and s > 0 for _, s, n in r["rows"]))
    row = next(row for row in step["rows"] if row[2] == 1 and row[1] > 0)
    row[1] -= 1                                                 # one position computed twice
    with pytest.raises(ValueError, match=f"request {row[0]}'s row starts at {row[1]}, not where its last row ended"):
        ADV.from_trace(recs)


def test_the_command_writes_the_advice_and_exits_zero(tmp_path, capsys):
    from verity_vllm.pipeline import cli
    _served(tmp_path, n=6)
    assert cli.main(["advice", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert "round trip equal" in capsys.readouterr().out
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"advice.bin", "advice.json", "report.json"}


def test_the_command_reports_a_disagreeing_run_as_a_usage_error(tmp_path):
    _served(tmp_path, n=4)
    (tmp_path / "requests.jsonl").write_text("")
    with pytest.raises(UsageError, match="the trace but not in the Commit record"):
        ADV.main(ADV.Options(run=str(tmp_path)))
