"""The scan's cost (`result.py`, `run.sh`) on a scan's kept files: the two numbers are the round's sessions and staging,
and its pinned Lean verifier processes' CPU seconds, over the group's execution phases, with the verifier's wall
beside its number and not in it; each session's stages sum to its process; the
guard holds only on an exit 0 with every execution phase's public part the pinned one; a result whose guard does not
hold has no measurement; and a comparison refuses two results whose identities differ.

No scan runs here and the evidence store is not reached: `kept` writes files in the form a scan keeps them (the keys
of its summary and answers, the lines of its sessions' logs) with stand-in figures chosen so every expected sum is
exact, `store` stands in for the `research data` commands, and `run.sh` runs against a stand-in for `scan_zk.sh` in a
copy of the benchmark under `tmp_path`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.scan_cost import result as R  # noqa: E402

RUN, OTHER = "r20000101-000000-aaaa", "r20000101-000001-bbbb"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
SOURCE = "the synthetic run (scan_traffic.serve), signed with the driver's own key"
#: a class's T, as its public part's blocks give it (2T blocks an execution phase)
TICKS = {cls: len(pin["proved"]) // 2 for cls, pin in json.loads(R.PUBLIC.read_text(encoding="utf-8")).items()}


def members(ticks: int, group: int) -> list[tuple[str, int]]:
    """The honest round's members with their blocks, as a scan at a group of `group` names them."""
    scans = [(f"ingress-{g}", ticks - (g == 0) - (g == group - 1)) for g in range(group)]
    return [("first", 1), *scans, ("last", 1), *((f"egress-{g}", ticks) for g in range(group))]


def kept(out: Path, cls: str = "test", group: int = 1, cores: int = 21, pin: bool = True, public: dict | None = None) -> Path:
    """A scan's kept files under `out`. Member i's prover process takes 12 + i s: 2 s to Register, 4 s from there to the
    repetitions, 2 s of level 0, 3 + i s of repetitions and a 1 s tail; its staging takes 0.5 s; its Lean verifier
    takes 4 + i s of wall, of which 1 s reading inputs, 2 s of setup and 0.5 + i/10 s of checks, and 3 + i s of CPU.
    Another round's verifier process, 9 s, is in `lean.jsonl` beside the honest round's."""
    ticks, names = TICKS[cls], members(TICKS[cls], group)
    timing, rnd, base = {}, {}, 0
    for i, (name, n) in enumerate(names):
        logs = out / "sczk" / R.ROUND / f"session-{name}"
        logs.mkdir(parents=True)
        live = {"e2e_s": 10 + i, "tp_s": 6 + i, "prove_total_s": 5 + i, "prove_s": [1 + i, 2], "witness_s": 0.25,
                "wait_e2e_s": 0.5, "k_log": 20, "dense_m": 22,
                "buckets": [{"t.prove": 1 + i, "t.upstream": i, "t.zk_inner": 0.75, "t.zk_pads_commit": 0.25},
                            {"t.prove": 2, "t.upstream": 1, "t.zk_inner": 0.75, "t.zk_pads_commit": 0.25}]}
        (logs / "prove.log").write_text(
            f"$ flock-circuit prove --zk\n# rc 0 in {12 + i} s\n--- stdout\nLIVE\t{json.dumps(live)}\n--- stderr\n"
            f"statement unit/scan-{name} instances=1 blocks={n} built in 0.5s\n", encoding="utf-8")
        (logs / "lean-verify.log").write_text(
            f"$ verifier verify\n# rc 0 in {4 + i} s\n--- stdout\nVERDICT {{}}\n--- stderr\ninputs: 1000 ms\n"
            f"setup (sampled): 2000 ms\n/tmp/scan/sessions/{name}: {2500 + 100 * i} ms\n", encoding="utf-8")
        timing[name] = {"session_s": 12 + i, "lean_verify_s": 4 + i, "timing": {"k_log": 20, "dense_m": 22}}
        rnd[name] = {"stage_s": 0.5}
        base += n
    summary = {"T": ticks, "group": group, "setup": cls, "source": SOURCE,
               "params": {"bucket_ms": 100},
               "members": {R.ROUND: [{"name": name, "n": n} for name, n in names]},
               "stagers": {f"joint/{k}": {"lower_s": 3.0} for k in ("first", "ingress", "egress")},
               "rounds": {R.ROUND: {**rnd, "audit": {"timing": dict(reversed(timing.items()))}}}}
    (out / "sczk" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    answer = {"verdict": "accept", "reason": None, "commitment": "c0", "public": public or R.pinned(cls, group)}
    (out / "sczk" / "answers.json").write_text(json.dumps({"windows": [{"answer": answer}] * group}), encoding="utf-8")
    (out / "host.txt").write_text(f"CPU(s): 64\nthreads {cores}\n", encoding="utf-8")
    (out / "rss.jsonl").write_text("".join(json.dumps({"command": ["prove"], "rc": 0, "wall_s": 12, "peak_gb": gb}) + "\n"
                                           for gb in (1.5, 3.5)), encoding="utf-8")
    if pin:
        rows = [{"command": [c], "archive": a, "core": 8, "rc": 0, "wall_s": w, "cpu_s": cpu, "peak_gb": 1.0}
                for i, (name, _) in enumerate(names)
                for c, a, w, cpu in (("archive-put", None, 1, 1),
                                     ("verify", f"sczk/{R.ROUND}/session-{name}/archive", 4 + i, 3 + i))]
        rows.append({**rows[-1], "archive": "sczk/chain/session-first/archive", "wall_s": 9, "cpu_s": 9})
        (out / "lean.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return out


def store(docs: dict):
    """The `research data` commands over `docs`, a run's document each."""
    def call(argv, **_):
        if argv[:3] == ["research", "data", "show"]:
            if argv[3] not in docs:
                return SimpleNamespace(returncode=1, stdout="", stderr="no such run")
            return SimpleNamespace(returncode=0, stdout=json.dumps({"attempt": {"outputs": {"run_record": f"art:{argv[3]}"}}}),
                                   stderr="")
        assert argv[:3] == ["research", "data", "fetch"] and argv[argv.index("--path") + 1] == f"out/{R.DOC}"
        to = Path(argv[argv.index("--to") + 1]) / "out"
        to.mkdir(parents=True)
        (to / R.DOC).write_text(json.dumps(docs[argv[3].removeprefix("art:")]), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return call


def doc_of(tmp_path: Path, name: str, run: str = RUN, **kw) -> dict:
    return R.document(kept(tmp_path / name, **kw), 0, run, "d" * 40)


def test_the_two_numbers_are_the_round_over_its_execution_phases(tmp_path):
    doc = doc_of(tmp_path, "a")
    assert doc["identity"] == {"class": "test", "sample": "every unit", "ticks": 16, "tick_ms": 100, "source": SOURCE,
                               "cores": 21, "verifier_cores": 1}
    assert (doc["group"], doc["phase_s"], doc["guard"]) == (1, 1.6, {"holds": True, "scan_exit": 0, "why": []})
    # four sessions of 12, 13, 14 and 15 s and 4 × 0.5 s of staging over 1.6 s; a Lean verifier of 3, 4, 5 and 6 s of
    # CPU and 4, 5, 6 and 7 s of wall
    assert doc["numbers"] == {"proving": 35.0, "proving_s_a_tick": 3.5, "verifier": 11.25, "verifier_wall": 13.75}
    assert doc["round"] == {"sessions_s": 54, "staging_s": 2.0, "to_register_s": 8, "register_to_reps_s": 16, "level0_s": 8,
                            "reps_s": 18, "tail_s": 4, "lean_cpu_s": 18, "lean_s": 22, "lean_inputs_s": 4.0,
                            "lean_setup_s": 8.0, "lean_check_s": 2.6}
    assert doc["lowering_s"] == {"first": 3.0, "ingress": 3.0, "egress": 3.0}
    # the whole scan's verifier processes, the other round's among them; the number is the honest round's alone
    assert (doc["peak_gb"], doc["verifier"]) == (3.5, {"cores": 1, "processes": 5, "wall_s": 31, "cpu_s": 27})


def test_a_session_is_read_in_the_rounds_order_and_its_stages_sum_to_its_process(tmp_path):
    doc = doc_of(tmp_path, "a")
    assert [(s["member"], s["blocks"]) for s in doc["sessions"]] == [("first", 1), ("ingress-0", 14), ("last", 1), ("egress-0", 16)]
    for i, s in enumerate(doc["sessions"]):
        assert sum(s["prover"][k] for k in R.STAGES) == s["prover"]["process_s"] == s["session_s"] == 12 + i
        assert s["prover"] | {"process_s": 0} == {
            "process_s": 0, "to_register_s": 2, "register_to_reps_s": 4, "level0_s": 2, "reps_s": 3 + i, "tail_s": 1,
            "statement_built_s": 0.5, "witness_s": 0.25, "wait_s": 0.5, "upstream_s": 1 + i, "zk_inner_s": 1.5, "pads_s": 0.5}
        assert s["lean"] == {"process_s": 4 + i, "inputs_s": 1.0, "setup_s": 2.0, "check_s": round(0.5 + i / 10, 3)}


def test_a_log_without_its_lines_gives_no_stage_and_the_number_stands(tmp_path):
    out = kept(tmp_path / "a")
    (out / "sczk" / R.ROUND / "session-last" / "prove.log").write_text("$ prove\n# rc 0 in 14 s\n", encoding="utf-8")
    (out / "sczk" / R.ROUND / "session-last" / "lean-verify.log").unlink()
    doc = R.document(out, 0, RUN)
    assert doc["sessions"][2]["prover"] == {"process_s": 14.0}
    assert doc["sessions"][2]["lean"] == {"process_s": None, "inputs_s": None, "setup_s": None, "check_s": None}
    assert doc["numbers"] == {"proving": 35.0, "proving_s_a_tick": 3.5, "verifier": 11.25, "verifier_wall": 13.75}
    assert doc["round"]["reps_s"] is None and doc["round"]["lean_setup_s"] is None
    names = [m["name"] for m in R.measurements(doc)]
    assert "sessions" in names and "the repetitions" not in names and "Lean verifier wall: setup" not in names


def test_a_group_is_its_execution_phases_and_every_blocks_answer(tmp_path):
    doc = doc_of(tmp_path, "a", group=2)
    assert [(s["member"], s["blocks"]) for s in doc["sessions"]] == [
        ("first", 1), ("ingress-0", 15), ("ingress-1", 15), ("last", 1), ("egress-0", 16), ("egress-1", 16)]
    assert (doc["group"], doc["phase_s"], doc["guard"]["holds"]) == (2, 3.2, True)
    assert doc["numbers"]["proving"] == 28.125          # six sessions of 12 to 17 s and 6 × 0.5 s of staging over 3.2 s
    assert R.pinned("test", 2)["proved"] == list(range(64)) and R.pinned("test", 1)["proved"] == list(range(32))
    one = R.document(kept(tmp_path / "b", group=2, public=R.pinned("test", 1)), 0, RUN)
    lists = ", ".join(sorted(k for k, v in R.pinned("test", 1).items() if v == list(range(32))))
    assert lists.endswith("proved") and lists != "proved"
    assert one["guard"]["why"] == [f"execution phase {i}'s public part is not public.json's for test in {lists}"
                                   for i in (0, 1)]
    assert one["identity"]["sample"] == "32 of 64 units"


def test_an_unpinned_verifier_has_no_number_and_says_why(tmp_path):
    doc = doc_of(tmp_path, "a", pin=False)
    assert doc["identity"]["verifier_cores"] is None and doc["numbers"]["verifier"] is None
    assert doc["numbers"]["proving"] == 35.0 and (doc["round"]["lean_s"], doc["numbers"]["verifier_wall"]) == (22, 13.75)
    m = next(m for m in R.measurements(doc) if m["name"] == R.VERIFIER)
    assert (m["value"], m["why"], m["inputs"]) == (None, R.NOT_PINNED, [])
    assert "not pinned" in R.table(doc)


def test_a_pinned_verifier_with_no_line_for_a_session_has_no_number(tmp_path):
    out = kept(tmp_path / "a")
    lines = (out / "lean.jsonl").read_text(encoding="utf-8").splitlines()
    (out / "lean.jsonl").write_text("".join(x + "\n" for x in lines if "session-last/" not in x), encoding="utf-8")
    doc = R.document(out, 0, RUN)
    assert (doc["identity"]["verifier_cores"], doc["round"]["lean_cpu_s"], doc["numbers"]["verifier"]) == (1, None, None)
    m = next(m for m in R.measurements(doc) if m["name"] == R.VERIFIER)
    assert (m["value"], m["why"]) == (None, R.NO_LINE)


@pytest.mark.parametrize(("change", "why"), [
    ({"bound": 1}, "execution phase 0's public part is not public.json's for test in bound"),
    ({"proved": list(range(31))}, "execution phase 0's public part is not public.json's for test in proved"),
    ({"rests_on_period_check": []}, "execution phase 0's public part is not public.json's for test in rests_on_period_check"),
])
def test_the_guard_refuses_another_public_part(tmp_path, change, why):
    doc = R.document(kept(tmp_path / "a", public={**R.pinned("test", 1), **change}), 0, RUN)
    assert doc["guard"] == {"holds": False, "scan_exit": 0, "why": [why]}


def test_the_guard_refuses_a_failed_scan_a_reject_and_a_missing_answer(tmp_path):
    out = kept(tmp_path / "a")
    assert R.document(out, 3, RUN)["guard"] == {"holds": False, "scan_exit": 3, "why": ["the scan exited 3"]}
    answers = out / "sczk" / "answers.json"
    answers.write_text(json.dumps({"windows": [{"answer": {"verdict": "reject", "reason": "a block failed", "public": None}}]}),
                       encoding="utf-8")
    assert R.document(out, 0, RUN)["guard"]["why"] == ["execution phase 0's answer is 'reject': a block failed"]
    answers.write_text(json.dumps({"windows": []}), encoding="utf-8")
    assert R.document(out, 0, RUN)["guard"]["why"] == ["answers.json holds 0 execution phases' answers, and the group is 1"]
    answers.unlink()
    [why] = R.document(out, 0, RUN)["guard"]["why"]
    assert "answers.json can't be read" in why
    assert R.guard(out, {"setup": "w0", "group": 1}, 0)["why"][0].startswith("public.json pins no public part for the class w0")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    monkeypatch.setattr(R.kit, "host_stamp", lambda: dict(STAMP))
    return tmp_path / "run"


def test_write_is_the_document_and_the_runs_result(run_dir, capsys):
    kept(run_dir / "out")
    assert R.main(["write", str(run_dir), "--scan-exit", "0"]) == 0
    doc = json.loads((run_dir / "out" / R.DOC).read_text(encoding="utf-8"))
    assert (doc["schema"], doc["run"], doc["commit"]) == (R.SCHEMA, RUN, "d" * 40)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["schema"] == R.kit.RESULT_SCHEMA and result["run_id"] == RUN
    assert result["workload_fingerprint"] == {"benchmark": "scan_cost", **doc["identity"], "group": 1}
    assert result["validation"] == {"status": "passed", "checker": "benchmarks/scan_cost/result.py",
                                    "detail": "the guard holds", "evidence": ["out/scan_cost.json", "out/sczk/answers.json"]}
    assert result["measurement_files"] == ["out/scan_cost.json"]
    got = {m["name"]: (m["value"], m["unit"]) for m in result["measurements"]}
    assert got[R.PROVING] == (35.0, "proving seconds a second of execution phase")
    assert got[R.VERIFIER] == (11.25, "Lean verifier CPU seconds a second of execution phase, on one core")
    assert got["Lean verifier wall seconds a second of execution phase"] == (13.75, "s")
    assert got["proving seconds a tick"] == (3.5, "s") and got["execution phases"] == (1.6, "s")
    assert {k: v[0] for k, v in got.items() if v[1] == "s a round"} == {
        "sessions": 54, "staging": 2.0, "to Register": 8, "Register to the repetitions": 16, "level 0": 8,
        "the repetitions": 18, "the tail": 4, "Lean verifier CPU": 18, "Lean verifier wall": 22,
        "Lean verifier wall: inputs": 4.0, "Lean verifier wall: setup": 8.0, "Lean verifier wall: checks": 2.6}
    by = {m["name"]: m["inputs"] for m in result["measurements"]}
    assert by[R.VERIFIER] == ["Lean verifier CPU", "execution phases"]
    names = {m["name"] for m in result["measurements"]}
    assert all(set(m["inputs"]) <= names for m in result["measurements"])
    text = capsys.readouterr().out
    assert "guard: holds" in text and "proving seconds a second of execution phase: 35.0 (3.5 s a tick)" in text
    assert "the round      32     54.0      2.0" in text


def test_a_result_whose_guard_does_not_hold_has_no_measurement(run_dir, capsys):
    kept(run_dir / "out", public={**R.pinned("test", 1), "bound": 2})
    assert R.main(["write", str(run_dir), "--scan-exit", "0"]) == 1
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["measurements"] == [] and result["validation"]["status"] == "failed"
    assert result["validation"]["detail"] == ("the guard does not hold: execution phase 0's public part is not "
                                              "public.json's for test in bound")
    assert json.loads((run_dir / "out" / R.DOC).read_text(encoding="utf-8"))["guard"]["holds"] is False
    assert "guard: does not hold" in capsys.readouterr().out


def test_a_round_without_a_sessions_seconds_has_no_result(run_dir, capsys):
    out = kept(run_dir / "out")
    summary = json.loads((out / "sczk" / "summary.json").read_text(encoding="utf-8"))
    del summary["rounds"][R.ROUND]["last"]["stage_s"]
    (out / "sczk" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert R.main(["write", str(run_dir), "--scan-exit", "0"]) == 1
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert (result["validation"]["status"], result["measurements"]) == ("failed", [])
    assert result["validation"]["detail"] == "the guard holds, and the summary lacks a session's seconds or staging: no number"
    del summary["rounds"][R.ROUND]["audit"]["timing"]["last"]["session_s"]
    (out / "sczk" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(R.Refused, match="lack what the honest round is read by \\(KeyError: 'session_s'\\)"):
        R.document(out, 0, RUN)
    assert "the summary lacks" in capsys.readouterr().out


def test_a_kept_file_that_does_not_parse_is_refused_not_passed_over(tmp_path):
    out = kept(tmp_path / "a")
    with (out / "lean.jsonl").open("a", encoding="utf-8") as f:
        f.write("{half a line\n")
    with pytest.raises(R.Refused, match="lack what the honest round is read by \\(JSONDecodeError: "):
        R.document(out, 0, RUN)


def test_a_scan_that_left_no_round_fails_with_no_document(run_dir, capsys):
    (run_dir / "out").mkdir(parents=True)
    assert R.main(["write", str(run_dir), "--scan-exit", "4"]) == 1
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert (result["validation"]["status"], result["measurements"], result["measurement_files"]) == ("failed", [], [])
    assert result["validation"]["detail"].startswith("no round to measure: ") and not (run_dir / "out" / R.DOC).exists()
    assert result["workload_fingerprint"] == {"benchmark": "scan_cost"}
    assert capsys.readouterr().out.startswith("scan_cost: no round to measure: ")


def test_a_comparison_refuses_two_results_whose_identities_differ(tmp_path):
    base = doc_of(tmp_path, "a")
    for field, kw in (("cores", {"cores": 23}), ("verifier_cores", {"pin": False}), ("class", {"cls": "w63k4"})):
        with pytest.raises(R.Refused, match=f"identities differ.*{field}: "):
            R.compare(base, doc_of(tmp_path, f"b-{field}", OTHER, **kw))
    bad = R.document(kept(tmp_path / "c"), 2, OTHER)
    with pytest.raises(R.Refused, match="the candidate's guard does not hold \\(the scan exited 2\\)"):
        R.compare(base, bad)
    with pytest.raises(R.Refused, match="the baseline's guard does not hold"):
        R.compare(bad, base)


def test_a_comparison_at_one_identity_is_a_second_of_execution_phase_beside_another(tmp_path):
    text = R.compare(doc_of(tmp_path, "a"), doc_of(tmp_path, "b", OTHER, group=2))
    assert text.splitlines()[:3] == [
        f"scan_cost: {OTHER} (group 2) beside {RUN} (group 1), at one identity",
        "  proving seconds a second of execution phase: 28.1 beside 35.0, 0.804 of the baseline",
        "  Lean verifier CPU seconds a second of execution phase, on one core: 10.3 beside 11.2, 0.917 of the baseline"]
    assert "    sessions: 27.2 beside 33.8" in text and "    Lean verifier wall: setup: 3.8 beside 5.0" in text


def test_the_baseline_names_a_run_and_holds_no_number(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "BASELINE", tmp_path / "baseline.json")
    with pytest.raises(R.Refused, match="no baseline is pinned yet"):
        R.baseline()
    base, cand = doc_of(tmp_path, "a"), doc_of(tmp_path, "b", OTHER, group=2)
    (tmp_path / "a" / R.DOC).write_text(json.dumps(base), encoding="utf-8")
    assert R.main(["baseline", "--lock", str(tmp_path / "a")]) == 0
    assert R.baseline() == {"schema": R.BASELINE_SCHEMA, "run": RUN, "commit": "d" * 40, "identity": base["identity"],
                            "group": 1}
    capsys.readouterr()
    # the candidate, a run's id, beside the baseline's run: both documents from the store
    assert R.main(["compare", OTHER], call=store({RUN: base, OTHER: cand})) == 0
    assert capsys.readouterr().out.startswith(f"scan_cost: {OTHER} (group 2) beside {RUN} (group 1), at one identity\n")
    # another identity is refused on the baseline file alone, before the baseline's document is fetched
    other = doc_of(tmp_path, "c", OTHER, cores=48)
    assert R.main(["compare", OTHER], call=store({OTHER: other})) == 2
    err = capsys.readouterr().err
    assert "identities differ" in err and "cores: 21 and 48" in err
    for bad in (R.document(kept(tmp_path / "d"), 1, RUN), doc_of(tmp_path, "e", "local")):
        with pytest.raises(R.Refused, match="a baseline is a run whose guard holds"):
            R.lock(bad)


def test_a_baseline_has_both_numbers(tmp_path, monkeypatch, capsys):
    """A document whose guard holds and that lacks a number is no baseline: a candidate would have nothing to be
    compared with on it. Neither an unpinned verifier's run nor one with no proving seconds is written."""
    monkeypatch.setattr(R, "BASELINE", tmp_path / "baseline.json")
    unpinned = doc_of(tmp_path, "a", pin=False)
    assert unpinned["guard"]["holds"] and unpinned["numbers"]["proving"] is not None
    with pytest.raises(R.Refused) as e:
        R.lock(unpinned)
    assert str(e.value) == f"a baseline has both numbers: this document has no {R.VERIFIER} ({R.NOT_PINNED})"
    both = doc_of(tmp_path, "b")
    assert None not in (both["numbers"]["proving"], both["numbers"]["verifier"])
    with pytest.raises(R.Refused, match=f"this document has no {R.PROVING}$"):
        R.lock(both | {"numbers": both["numbers"] | {"proving": None}})
    neither = unpinned | {"numbers": unpinned["numbers"] | {"proving": None}}
    with pytest.raises(R.Refused, match=f"has no {R.PROVING} and no {R.VERIFIER} \\("):
        R.lock(neither)
    (tmp_path / "a" / R.DOC).write_text(json.dumps(unpinned), encoding="utf-8")
    assert R.main(["baseline", "--lock", str(tmp_path / "a")]) == 2
    assert "scan_cost: a baseline has both numbers" in capsys.readouterr().err and not R.BASELINE.exists()
    assert R.lock(both)["run"] == RUN and R.baseline()["run"] == RUN


def test_a_document_is_a_file_a_directory_or_a_run(tmp_path, capsys):
    doc = doc_of(tmp_path, "a")
    (tmp_path / "a" / R.DOC).write_text(json.dumps(doc), encoding="utf-8")
    assert R.opened(str(tmp_path / "a" / R.DOC)) == R.opened(str(tmp_path / "a")) == R.opened(RUN, store({RUN: doc})) == doc
    assert R.main(["table", str(tmp_path / "a")]) == 0
    text = capsys.readouterr().out
    assert "  Lean verifier CPU seconds a second of execution phase, on one core: 11.2; its wall beside it: 13.8\n" in text
    assert "the round, in seconds" in text and "first           1     12.0" in text
    with pytest.raises(R.Refused, match="names no run record \\(no such run\\)"):
        R.opened(OTHER, store({}))
    with pytest.raises(R.Refused, match="no scan_cost.json, no directory holding one and no run's id"):
        R.opened(str(tmp_path / "nowhere"))
    (tmp_path / "x.json").write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert R.main(["table", str(tmp_path / "x.json")]) == 2 and "is no scan-cost/v1 document" in capsys.readouterr().err


def test_the_metrics_are_written_under_their_ids():
    assert (R.PROVING, R.VERIFIER) == ("network-accounting/scan-proving-cost", "network-accounting/scan-verifier-cost")


SCAN = """#!/usr/bin/env bash
# a stand-in for the scan: the kept files, the threads it would size its provers by (nproc, as the scan's job
# does) and the settings it was started with
mkdir -p "$RESEARCH_RUN_DIR/out" && cp -r "$KEPT"/. "$RESEARCH_RUN_DIR/out"/
printf 'threads %s\\n' "$(nproc)" > "$RESEARCH_RUN_DIR/out/host.txt"
printf '%s\\n' "class=${SC_CLASS-} mem=${SC_MEM_GB-} rss=${SC_RSS-} one=${SC_LEAN_ONE-} dry=${SC_DRY-unset} other=${SC_OTHER-unset} group=${SC_GROUP-unset}" \\
  > "$RESEARCH_RUN_DIR/out/seen.txt"
exit "${SCAN_EXIT:-0}"
"""


@pytest.fixture
def tree(tmp_path):
    """A copy of the benchmark's runner, writer and pin beside a stand-in scan, with the tree's own harness and
    settings."""
    top = tmp_path / "tree"
    (top / "benchmarks" / "network_traces").mkdir(parents=True)
    (top / "benchmarks" / "scan_cost").mkdir()
    for name in ("run.sh", "result.py", "public.json"):
        shutil.copy2(HERE.parent / name, top / "benchmarks" / "scan_cost" / name)
    (top / "benchmarks" / "network_traces" / "scan_zk.sh").write_text(SCAN, encoding="utf-8")
    (top / "benchmarks" / "harness").symlink_to(ROOT / "benchmarks" / "harness")
    (top / "verity").symlink_to(ROOT / "verity")
    return top


def run_sh(tree: Path, run_dir: Path, **env: str) -> subprocess.CompletedProcess:
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("SC_", "RESEARCH_"))}
    return subprocess.run(["bash", str(tree / "benchmarks" / "scan_cost" / "run.sh")], capture_output=True, text=True,
                          env={**keep, "RESEARCH_RUN_DIR": str(run_dir), "RESEARCH_RUN_ID": RUN,
                               "RESEARCH_SOURCE_SHA": "d" * 40, **env}, check=False)


def test_run_sh_pins_the_job_runs_the_scan_at_the_identity_and_writes_the_result(tree, tmp_path):
    files = kept(tmp_path / "kept", cls="w63k4")
    # a caller's OMP_NUM_THREADS, which nproc would answer with, is cleared with every setting that is not the benchmark's
    r = run_sh(tree, tmp_path / "run", KEPT=str(files), SC_CORES="1", SC_DRY="1", SC_OTHER="4", OMP_NUM_THREADS="7")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "run" / "out" / "seen.txt").read_text(encoding="utf-8") == (
        "class=w63k4 mem=128 rss=1 one=1 dry=unset other=unset group=unset\n")
    result = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    # the scan saw one core, the one run.sh pinned it to, whatever this machine has
    assert result["workload_fingerprint"] == {
        "benchmark": "scan_cost", "class": "w63k4", "sample": "every unit", "ticks": 32, "tick_ms": 100, "source": SOURCE,
        "cores": 1, "verifier_cores": 1, "group": 1}
    assert result["validation"]["status"] == "passed"
    got = {m["name"]: m["value"] for m in result["measurements"]}
    assert (got[R.PROVING], got[R.VERIFIER], got["execution phases"]) == (17.5, 5.625, 3.2)
    assert "scan_cost: pinned to 1 cores" in r.stdout and "guard: holds" in r.stdout


def test_run_sh_removes_the_verifiers_circuit_text_and_nothing_else_the_scan_kept(tree, tmp_path):
    """The scan keeps the verifier's copy of each session's circuit with its other files, and those are most of a run's
    bytes. The runner removes them once the scan has ended, whatever the scan's exit, and every other kept file stays:
    the session's logs, the verifier's other files, and a circuit's text outside a verifier's directory."""
    files = kept(tmp_path / "kept", cls="w63k4")
    stays = [f"sczk/{R.ROUND}/verifier/program.json", f"sczk/{R.ROUND}/verifier/first-session-l0-circuit.rep0.bin",
             f"sczk/{R.ROUND}/stage-first/first-circuit.txt", "sczk/chain/verifier/notes.txt"]
    goes = [f"sczk/{R.ROUND}/verifier/first-circuit.txt", f"sczk/{R.ROUND}/verifier/ingress-0-circuit.txt",
            "sczk/chain/verifier/egress-0-circuit.txt"]
    for rel in stays + goes:
        (files / rel).parent.mkdir(parents=True, exist_ok=True)
        (files / rel).write_text("x" * 64, encoding="utf-8")
    before = {p.relative_to(files).as_posix() for p in files.rglob("*") if p.is_file()}
    for name, exit_ in (("held", 0), ("failed", 7)):
        r = run_sh(tree, tmp_path / name, KEPT=str(files), SC_CORES="1", SCAN_EXIT=str(exit_))
        assert r.returncode == exit_, r.stdout + r.stderr
        out = tmp_path / name / "out"
        after = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        assert before - after == set(goes) and all((out / rel).is_file() for rel in stays)
        assert "scan_cost: removed the verifier's copies of the circuits from the kept files" in r.stdout
    assert json.loads((tmp_path / "held" / "result.json").read_text(encoding="utf-8"))["validation"]["status"] == "passed"


def test_run_sh_exits_as_the_scan_did_and_refuses_a_job_with_too_few_cores(tree, tmp_path):
    files = kept(tmp_path / "kept", cls="w63k4")
    r = run_sh(tree, tmp_path / "failed", KEPT=str(files), SC_CORES="1", SCAN_EXIT="7")
    assert r.returncode == 7, r.stdout + r.stderr
    result = json.loads((tmp_path / "failed" / "result.json").read_text(encoding="utf-8"))
    assert result["measurements"] == [] and result["validation"]["detail"] == "the guard does not hold: the scan exited 7"
    r = run_sh(tree, tmp_path / "few", KEPT=str(files), SC_CORES="100000")
    assert r.returncode == 2 and "the benchmark's identity is 100000" in r.stderr
    assert not (tmp_path / "few").exists()
    r = run_sh(tree, tmp_path / "few", KEPT=str(files), SC_CORES="many")
    assert r.returncode == 2 and "SC_CORES=many is not a count of cores" in r.stderr
