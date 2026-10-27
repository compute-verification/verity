"""`verity-vllm advice`: I1's advice of a run recorded on the stand-in scheduler (`tests.engine.fake_scheduler`) under varied
traffic, its round trip, its windows, prefix-cache sources and resumes, and the refusals when the step trace and the Commit
record disagree or a finish is one advice v0 can't say (an abort, which a Commit window's advice says)."""
from __future__ import annotations

import itertools
import json
import random
import re
from types import SimpleNamespace

import pytest

from tests.engine.fake_scheduler import FakeRequest, FakeScheduler, drive
from verity.core.protocols.certifier import advice as I1
from verity_vllm.config import UsageError
from verity_vllm.engine import step_trace as ST
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import window_advice as WA

C = ADV.registered_compiler("test-compiler", "ab" * 64)


def _served(tmp_path, *, seed: int = 0, n: int = 12, gap: bool = True, **sched_kw):
    """A run of `n` requests with varied prompt and output lengths, some ending at EOS, arriving while others decode, with an idle
    gap before the last two when `gap`; writes step_trace.jsonl and requests.jsonl under tmp_path."""
    rng = random.Random(seed)
    arrivals, at = {}, 0
    for i in range(n):
        prompt, cap = rng.randint(1, 40), rng.randint(1, 12)
        outputs = [rng.randint(3, 99) for _ in range(cap)]
        if rng.random() < 0.5:
            outputs[rng.randrange(cap)] = 2                     # EOS before the cap
        r = FakeRequest(f"e{i}", [rng.randint(3, 99) for _ in range(prompt)], outputs, cap,
                        sampling_params=SimpleNamespace(temperature=0.0, max_tokens=cap))
        at += 60 if gap and i == n - 2 else rng.randint(0, 3)
        arrivals.setdefault(at, []).append(r)
    return _run(tmp_path, arrivals, **sched_kw)


def _run(tmp_path, arrivals: dict[int, list[FakeRequest]], *, abort_at=None, clock: bool = True, **sched_kw) -> list[FakeRequest]:
    sched = FakeScheduler(budget=16, max_seqs=4, **sched_kw)
    ticks = itertools.count()
    trace = ST.StepTrace(tmp_path / "step_trace.jsonl", clock=(lambda: 0.05 * next(ticks)) if clock else None).install(sched)
    drive(sched, arrivals, abort_at=abort_at)
    trace.uninstall()
    reqs = [r for rs in arrivals.values() for r in rs]
    with (tmp_path / "requests.jsonl").open("w") as f:
        for r in reqs:
            f.write(json.dumps({"request": r.request_id, "user": f"user-{r.request_id}", "prompt_ids": r.prompt_token_ids,
                                "output_ids": r.emitted}) + "\n")
    return reqs


def _advice(tmp_path) -> I1.Advice:
    return I1.decode((tmp_path / "advice" / "private" / "advice.bin").read_bytes())


def test_a_varied_run_gives_i1_advice_that_round_trips_with_its_bits_charged_to_windows(tmp_path):
    reqs = _served(tmp_path)
    rep = ADV.write(tmp_path, T=5, compiler=C)
    assert rep["roundtrip"] is True
    a = _advice(tmp_path)
    by_id = {f"user-{r.request_id}": r for r in reqs}
    assert sorted(q.id for q in a.requests) == sorted(by_id)
    assert [(q.prompt_len, q.output_len) for q in a.requests] == [(len(by_id[q.id].prompt_token_ids), len(by_id[q.id].emitted))
                                                                  for q in a.requests]
    assert {q.finish for q in a.requests} == {"eos", "length"}
    assert all((q.finish == "eos") == (by_id[q.id].status == "stopped") for q in a.requests)
    c = rep["counts"]
    assert c["chunked_prefill_rows"] > 0 and c["mixed_steps"] > 0 and c["arrivals_during_decode"] > 0
    assert c["outputs"] == len(a.release) == sum(len(r.emitted) for r in reqs)
    assert rep["release"] == {"rule": "engine clock", "tau_s": 0.1}
    assert len(rep["windows"]) > 1 and sum(w["bits"] for w in rep["windows"]) == rep["advice"]["bits"] == I1.total_bits(a)
    assert sum(w["outputs"] for w in rep["windows"]) == len(a.release)
    # every scheduled token is in exactly one row, and each request's rows tile [0, prompt + output - 1)
    for i, q in enumerate(a.requests):
        spans = [(r.start, r.n) for s in a.steps for r in s.rows if r.request == i]
        assert sum(n for _, n in spans) == q.prompt_len + q.output_len - 1
        assert [s for s, _ in spans] == [sum(n for _, n in spans[:k]) for k in range(len(spans))]


def test_a_preempted_request_resumes_on_a_row_marked_preempted_from_position_zero(tmp_path):
    _run(tmp_path, {0: [FakeRequest("e0", [5] * 4, [6] * 6, 6), FakeRequest("e1", [5] * 4, [6] * 6, 6)]}, preempt_at={2: "e0"})
    rep = ADV.write(tmp_path, compiler=C)
    a = _advice(tmp_path)
    pre = [r for st in a.steps for r in st.rows if r.preempted]
    assert rep["roundtrip"] and rep["counts"]["preempted_rows"] == 1
    assert pre[0].request == 0 and pre[0].start == pre[0].cache_hit == 0 and pre[0].n > 1 and pre[0].kind == "prefill"


def test_a_prefix_cache_hit_names_the_request_whose_kv_it_reuses(tmp_path):
    shared = [7, 8, 9, 10, 11, 12, 13, 14]
    _run(tmp_path, {0: [FakeRequest("e0", shared + [3, 4], [6] * 3, 3)], 5: [FakeRequest("e1", shared + [5], [6] * 3, 3)]},
         prefix_block=4)
    rep = ADV.write(tmp_path, compiler=C)
    a = _advice(tmp_path)
    first = next(r for st in a.steps for r in st.rows if r.request == 1)
    assert rep["roundtrip"] and rep["counts"]["cache_hit_rows"] == 1
    assert first.start == first.cache_hit == 8 and first.sources == (I1.Source(0, 8),)


def test_a_resume_under_prefix_caching_reuses_its_own_kv(tmp_path):
    _run(tmp_path, {0: [FakeRequest("e0", list(range(3, 13)), [6] * 8, 8), FakeRequest("e1", [5] * 4, [6] * 8, 8)]},
         preempt_at={3: "e0"}, prefix_block=4)
    rep = ADV.write(tmp_path, compiler=C)
    a = _advice(tmp_path)
    pre = next(r for st in a.steps for r in st.rows if r.preempted)
    assert rep["roundtrip"] and a.requests[pre.request].id == "user-e0" and pre.start == pre.cache_hit > 0
    assert pre.sources == (I1.Source(pre.request, pre.cache_hit),)


@pytest.mark.parametrize("seed", range(4))
def test_the_round_trip_holds_over_seeds_with_prefix_caching_and_a_preemption(tmp_path, seed):
    _served(tmp_path, seed=seed, n=16, prefix_block=4, preempt_at={6: "e1"})
    assert ADV.write(tmp_path, compiler=C)["roundtrip"] is True


def test_a_trace_without_a_clock_releases_each_output_at_its_step(tmp_path):
    _served(tmp_path, n=4, gap=False, clock=False)
    rep = ADV.write(tmp_path, compiler=C)
    a = _advice(tmp_path)
    assert rep["release"] == {"rule": "step index"}
    assert list(a.release) == [o.step for o in I1.outputs(a)]


def test_an_aborted_request_is_a_commit_windows_finish_and_a_whole_sessions_refusal(tmp_path):
    _run(tmp_path, {0: [FakeRequest("e0", [5] * 4, [6] * 9, 9), FakeRequest("e1", [5] * 4, [6] * 6, 6)]}, abort_at={3: "e0"})
    with pytest.raises(ValueError, match=r"request e0 finished by abort, which a whole session's advice \(I1 v0\) can't say"):
        ADV.write(tmp_path, compiler=C)
    recs, io = ST.load(tmp_path / "step_trace.jsonl"), ADV.io_record(tmp_path / "requests.jsonl")
    a = ADV.from_trace(recs, io, carry=()).advice
    # aborted after the step that sampled its third output (the stand-in schedules synchronously: no row past it)
    assert [(q.id, q.output_len, q.finish) for q in a.requests] == [("user-e0", 3, "abort"), ("user-e1", 6, "length")]
    assert a.version == WA.ABORTS and WA.decode(WA.encode(a)) == a and 0 not in {c.request for c in WA.close(a)}
    step = next(r for r in recs if r["kind"] == "step" and r["step"] == 3)
    step["rows"].append(["e0", 6, 1])                           # a row after the step it was freed at
    with pytest.raises(ValueError, match=r"step 3: request e0 has a row 1 steps after it finished \(abort after step 2, outside"):
        ADV.from_trace(recs, io, carry=())


def _behind_a_long_prompt(tmp_path, abort_at: dict[int, str]) -> tuple[list[dict], dict]:
    """e0's 40-token prompt takes the 16-token budget for three steps: aborted after the first, it is mid-prefill; e1, waiting
    behind it, has never run."""
    _run(tmp_path, {0: [FakeRequest("e0", [5] * 40, [6] * 3, 3), FakeRequest("e1", [5] * 4, [6] * 3, 3)]}, abort_at=abort_at)
    return ST.load(tmp_path / "step_trace.jsonl"), ADV.io_record(tmp_path / "requests.jsonl")


def test_an_abort_mid_prefill_before_the_first_output_is_refused_by_name(tmp_path):
    recs, io = _behind_a_long_prompt(tmp_path, {1: "e0"})
    assert len(io["e0"]["output_ids"]) == 0 and any(row[0] == "e0" for r in recs if r["kind"] == "step" for row in r["rows"])
    with pytest.raises(ValueError) as e:
        ADV.from_trace(recs, io, carry=())
    assert str(e.value) == ("request e0 was aborted before its first output, which the advice refuses: an aborted request's "
                            "Program ends at its last kept output, and it kept none")


def test_a_request_aborted_before_it_is_ever_scheduled_is_dropped_from_the_advice_and_kept_in_the_commit_record(tmp_path):
    """e1 has no row and no output, so the advice is e0's alone, a Commit window's and a whole session's alike, and the Commit
    record keeps e1; a Commit record or a trace that gives e1 an output is refused."""
    recs, io = _behind_a_long_prompt(tmp_path, {1: "e1"})
    assert [r["reason"] for r in recs if r["kind"] == "finish" and r["request"] == "e1"] == ["abort"] and io["e1"]["output_ids"] == []
    assert not any(row[0] == "e1" for r in recs if r["kind"] == "step" for row in r["rows"])
    for carry in ((), None):
        b = ADV.from_trace(recs, io, carry=carry)
        assert [(q.id, q.output_len, q.finish) for q in b.advice.requests] == [("user-e0", 3, "length")] and b.dropped == ("e1",)
        assert len(b.advice.release) == 3 and b.advice.version == 0
    rep = ADV.write(tmp_path, compiler=C)
    assert rep["roundtrip"] and rep["counts"]["requests"] == 1 and rep["counts"]["aborted_unscheduled"] == 1
    assert [(q.id, q.output_len) for q in _advice(tmp_path).requests] == [("user-e0", 3)]
    gone = "request e1 was aborted before it was ever scheduled, which the advice drops, and the trace emitted"
    for e1, why in [({"output_ids": [6]}, "0 tokens for it, the Commit record 1, finished by abort"),
                    ({"finish": {"reason": "abort", "stop_reason": None, "k": 2}}, "0 tokens for it, the Commit record 0, finished by abort after 2 ids"),
                    ({"finish": {"reason": "length", "stop_reason": None}}, "0 tokens for it, the Commit record 0, finished by length")]:
        with pytest.raises(ValueError) as e:
            ADV.from_trace(recs, {**io, "e1": {**io["e1"], **e1}}, carry=())
        assert str(e.value) == f"{gone} {why}"
    next(r for r in recs if r["kind"] == "step" and r["step"] == 0)["emitted"]["e1"] = 1
    with pytest.raises(ValueError) as e:
        ADV.from_trace(recs, io, carry=())
    assert str(e.value) == f"{gone} 1 tokens for it, the Commit record 0, finished by abort"


def test_a_request_that_arrived_and_never_ran_unaborted_is_refused(tmp_path):
    recs, io = _behind_a_long_prompt(tmp_path, {1: "e1"})
    recs = [r for r in recs if not (r["kind"] == "finish" and r["request"] == "e1")]
    with pytest.raises(ValueError, match=r"^1 requests arrived and never ran: \['e1'\]$"):
        ADV.from_trace(recs, io, carry=())


def test_an_abort_of_a_request_waiting_after_a_preemption_is_refused_by_name(tmp_path):
    _run(tmp_path, {0: [FakeRequest("e0", [5] * 4, [6] * 6, 6), FakeRequest("e1", [5] * 4, [6] * 6, 6)]}, preempt_at={2: "e0"},
         abort_at={3: "e0"})
    recs, io = ST.load(tmp_path / "step_trace.jsonl"), ADV.io_record(tmp_path / "requests.jsonl")
    assert len(io["e0"]["output_ids"]) == 2 and "e0" in next(r for r in recs if r["kind"] == "step" and r["step"] == 2)["preempted"]
    with pytest.raises(ValueError) as e:
        ADV.from_trace(recs, io, carry=())
    assert str(e.value) == ("request e0 was aborted after step 2 while it waited after a preemption, which the advice refuses: it "
                            "has no row in the step it is freed at")


def test_a_commit_record_whose_output_disagrees_with_the_trace_is_refused(tmp_path):
    _served(tmp_path, n=4)
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    doc = json.loads(lines[1])
    doc["output_ids"] = doc["output_ids"] + [7]
    lines[1] = json.dumps(doc)
    (tmp_path / "requests.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="request e1: the trace emitted .* the Commit record holds"):
        ADV.write(tmp_path, compiler=C)


def test_a_commit_record_missing_a_request_is_refused(tmp_path):
    _served(tmp_path, n=4)
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    (tmp_path / "requests.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="request e3 is in the trace but not in the Commit record"):
        ADV.write(tmp_path, compiler=C)


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
    out = tmp_path / "out"
    assert cli.main(["advice", str(tmp_path), "--out", str(out), "--T", "50", "--tau", "0.2", "--compiler-digest", "ab" * 64]) == 0
    assert "round trip equal" in capsys.readouterr().out
    assert {p.name for p in out.iterdir()} == {"registration.json", "private"}
    assert {p.name for p in (out / "private").iterdir()} == {"advice.bin", "advice.json", "opening.json", "report.json"}
    rep = json.loads((out / "private" / "report.json").read_text())
    assert rep["T"] == 50 and rep["release"]["tau_s"] == 0.2
    assert rep["registration"] == json.loads((out / "registration.json").read_text())
    capped = ADV.registered_compiler(ADV.COMPILER, "ab" * 64)
    assert rep["registration"]["binding"]["compiler"] == {"name": capped.name, "digest": capped.digest}
    assert rep["registration"]["compiler"]["inner"] == {"name": ADV.COMPILER, "digest": "ab" * 64}
    private = out / "private"
    assert I1.from_json(json.loads((private / "advice.json").read_text())) == I1.decode((private / "advice.bin").read_bytes())


def test_the_registration_records_the_committers_weights_roots_and_refuses_anything_else(tmp_path):
    """``--weights-of-record``: the registration records the committer's registered root for each weight parameter of the
    window circuit (its weights registration's, ``registered_roots``, what `interface.input_count` holds the verifier's pins
    to); without it, none. A weights record (the verifier's shapes, no root), a registration of no value, one whose values
    aren't weight parameters' entries with a root of 64 bytes, or one of another model than the run's ``serve.json`` names,
    is a usage error."""
    from verity.ml.windows import interface as IF
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.records import weights_registration as WRG

    _served(tmp_path, n=4)
    roots = {"weights/model.norm.weight@64": "cd" * 64, "weights/model.embed_tokens.weight@64": "ef" * 64}
    values = {p: {"value": p, "root": r, "leaves": 1, "words": 64, "domain": "01" * 64} for p, r in roots.items()}
    registration = {"schema": WRG.SCHEMA, "program": "ab" * 64, "model": {"repo": "org/m", "revision": "r"},
                    "grain": {"gemm_words": 32768, "rows": 256, "select": 768}, "values": values, "unrooted": {}}
    (tmp_path / "wr.json").write_text(json.dumps(registration))
    assert ADV.main(ADV.Options(run=str(tmp_path), out=str(tmp_path / "a"), compiler_digest="ab" * 64,
                                weights_of_record=str(tmp_path / "wr.json"))) == 0
    reg = json.loads((tmp_path / "a" / "registration.json").read_text())
    assert reg["registered_roots"] == IF.registered_roots(reg) == roots
    assert ADV.main(ADV.Options(run=str(tmp_path), out=str(tmp_path / "b"), compiler_digest="ab" * 64)) == 0
    assert "registered_roots" not in json.loads((tmp_path / "b" / "registration.json").read_text())
    p = "weights/model.norm.weight@64"
    record = {"schema": WR.SCHEMA, "of_record": True, "root_of_record": "cd" * 32,
              "registered_entries": {q: {k: v for k, v in e.items() if k != "root"} for q, e in values.items()}}
    for bad, serve, why in [(record, None, "never a weights record"),
                            ({**registration, "values": {}}, None, "'values' is empty"),
                            ({**registration, "values": {"x": {**values[p], "value": "x"}}}, None, "registers ['x'], not weight"),
                            ({**registration, "values": {**values, p: {**values[p], "root": "cd" * 32}}}, None,
                             f"'values.{p}' doesn't name its parameter with a root of 64 bytes"),
                            (registration, {"model": "org/other", "revision": "r"}, "the run served org/other@r")]:
        (tmp_path / "wr.json").write_text(json.dumps(bad))
        if serve:
            (tmp_path / "serve.json").write_text(json.dumps(serve))
        with pytest.raises(UsageError, match=re.escape(why)):
            ADV.main(ADV.Options(run=str(tmp_path), out=str(tmp_path / "c"), compiler_digest="ab" * 64,
                                 weights_of_record=str(tmp_path / "wr.json")))


def test_the_command_reports_a_disagreeing_run_as_a_usage_error(tmp_path):
    _served(tmp_path, n=4)
    (tmp_path / "requests.jsonl").write_text("")
    with pytest.raises(UsageError, match="the trace but not in the Commit record"):
        ADV.main(ADV.Options(run=str(tmp_path), compiler_digest="ab" * 64))
