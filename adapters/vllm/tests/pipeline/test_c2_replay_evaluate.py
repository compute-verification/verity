"""`verity_vllm.pipeline.c2_replay.evaluate`, the replay and linkage records of one instrumented run, with each step that reads
Programs or committed values replaced by a stand-in: what it hands the linkage and the sampled replay, what it puts on the
replay's record after them, and what a plan and a planner's plan return instead.  The executed prefix of record, its facts
(`executed_prefix_kw`) and the reconciliation of the not-executed class (`reconcile`) are the real ones
(`verity_vllm.commit.executed_prefix`, `check.replay.population`), on two requests of which one stopped at an EOS.  CPU only;
the replay itself is `tests/check/test_sampled_replay.py`."""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.c2_replay as C2
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.commit import executed_prefix as EPX

Record = dict[str, Any]
Lines = Callable[[], list[str]]

#: two requests of four steps each: `a` runs to its `max_tokens`, `b` stops at the EOS (token 2) it serves second
REQS: list[Record] = [{"request_id": "a", "prompt_token_ids": [1, 9], "max_tokens": 4, "seed": 11},
                      {"request_id": "b", "prompt_token_ids": [1], "arrive_step": 1}]
TOKENS: Record = {"a": [5, 6, 7, 8], "b": [5, 2]}
LABELS = {"a": "r0", "b": "r1"}
STOP: Record = {"eos": [2], "lag": 0, "declared_lag": 0}
#: a binding whose token record links both requests' served lengths: the not-executed class applies with no fault
LINKED: Record = {"coverage": {"ok": True, "expected_n": 8, "bound_n": 8, "missing_n": 0, "commit_account_link": {
    "ok": True, "required": True, "lag_attested": True, "problems": [], "record": {"path": "commit_tokens.json", "sha256": "ab" * 32},
    "served_by_request": {"r0": 4, "r1": 2}, "tokens_committed": {"r0": True, "r1": True}}}}
LINKAGE: Record = {"result": True, "horizon": {"r0": {"linked_steps": 4}, "r1": {"linked_steps": 2}}}
#: what the replay's population says it left out: `b`'s two rows past its stop, in one row family
POPULATION: Record = {"applied": True, "by_request": {"r0": {"n_exec": 4, "arrive": 0}, "r1": {"n_exec": 2, "arrive": 1}},
                      "totals": {"by_family": {"instance_outputs": {"N_excluded": 2}}}}
CLASSIFIED: Record = {"by_family": {"instance_outputs": 2}, "kind_by_family": {"instance_outputs": EPX.KIND_ROW}}
OPENED: Record = {"reads": 6, "leaves": 12, "bytes": 96, "seconds": 0.5, "steps_bound": 5, "not_retained": 0, "failed": 0}


@dataclass(frozen=True)
class Weights:
    """A weights provider that only says which prefix it was made for."""
    prefix: str

    def __call__(self, *a: Any) -> None:
        return None


@pytest.fixture
def said(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> Lines:
    """The lines `c2_replay` has logged since the last call, each after its level; it prints none."""
    caplog.set_level(logging.INFO, logger=C2.__name__)

    def lines() -> list[str]:
        assert capsys.readouterr().out == ""
        got = [f"{r.levelname} {r.getMessage()}" for r in caplog.records if r.name == C2.__name__]
        caplog.clear()
        return got
    return lines


def executed(tokens: Record | None = None) -> Record:
    """The executed prefix of record of the two requests, as `evaluate` derives it."""
    served = TOKENS if tokens is None else tokens
    return EPX.executed_prefix_of_record(["r0", "r1"], {LABELS[k]: v for k, v in served.items()}, caps={"r0": 4, "r1": 4}, eos_of_record=[2], lag=0)


def sources(out: Path, binding: Record | None = None, tokens: Record | None = None, **args: Any) -> C2.ReplaySources:
    (out / "binding_map_p0.json").write_text('{"entries": 5}')
    options: Record = {"serving_rows": None, "label": "row", "replay_seed": 7, "replay_per_stratum": 2, "replay_max": 9, "replay_k": None}
    commit: Record = copy.deepcopy({"binding": {"population": {"manifest": {"not_executed_by_rule": CLASSIFIED}}} | (LINKED if binding is None else binding)})
    return C2.ReplaySources(
        args=SimpleNamespace(**options | args), out=out, pair=0, com=SimpleNamespace(name="stub"), rc=SimpleNamespace(run_root=bytes(64)),
        row={"commit": commit}, reqs=REQS, sampling={"max_tokens": 4, "seed": 3}, tokens=TOKENS if tokens is None else tokens,
        manifest={"manifest_digest": "m" * 8}, stop_of_record=STOP, ids_full=None, versions={}, components={}, vc_result=None,
        weights_of=Weights, weights_check=lambda: {"equal": True}, tl=None)


@pytest.mark.parametrize("k", [None, 3])
def test_an_eos_stopped_request_s_facts_of_record_bound_the_replay_s_population(tmp_path: Path, said: Lines, k: int | None) -> None:
    s = sources(tmp_path, replay_k=k)
    kw, facts = C2.executed_prefix_kw(s, executed(), LINKAGE, {"r0": 0, "r1": 1}, LABELS)
    assert set(kw) == {"executed_prefix"} and set(kw["executed_prefix"]) == {"r0", "r1"}
    r0, r1 = kw["executed_prefix"]["r0"], kw["executed_prefix"]["r1"]
    assert (r0["a_r"], r0["n_exec"], r0["cap"], r0["faults"]) == (4, 4, 4, [])
    assert (r1["a_r"], r1["n_exec"], r1["cap"], r1["arrive"], r1["faults"]) == (2, 2, 4, 1, [])
    assert facts["r1"]["served"] == [5, 2] and facts["r1"]["eos_ids"] == [2] and facts["r1"]["linked"] is True
    assert said() == ["INFO [delta] not-executed class -> sampled_replay(executed_prefix=): r0 a_r 4 n_exec 4 cap 4, r1 a_r 2 n_exec 2 cap 4; no faults"]


def test_a_prefix_no_account_links_is_passed_on_with_its_faults_named(tmp_path: Path, said: Lines) -> None:
    s = sources(tmp_path, binding={})
    kw, _facts = C2.executed_prefix_kw(s, executed(), LINKAGE, {"r0": 0, "r1": 1}, LABELS)
    faults = {k: v["faults"] for k, v in kw["executed_prefix"].items()}
    assert all(len(f) == 3 and "a_r is not linked" in f[1] for f in faults.values())
    line, = said()
    assert line == ("WARNING [delta] not-executed class -> sampled_replay(executed_prefix=): r0 a_r 4 n_exec 4 cap 4, r1 a_r 2 n_exec 2 cap 4; "
                    f"FAULTS {faults}")


def test_with_no_early_stop_the_facts_pass_quietly_and_a_sample_of_k_takes_none(tmp_path: Path, said: Lines) -> None:
    full = {"a": [5, 6, 7, 8], "b": [5, 6, 7, 8]}
    linked = {"coverage": LINKED["coverage"] | {"commit_account_link": LINKED["coverage"]["commit_account_link"] | {"served_by_request": {"r0": 4, "r1": 4}}}}
    linkage = {"result": True, "horizon": {"r0": {"linked_steps": 4}, "r1": {"linked_steps": 4}}}
    kw, _facts = C2.executed_prefix_kw(sources(tmp_path, linked, full), executed(full), linkage, {"r0": 0, "r1": 1}, LABELS)
    assert [(v["n_exec"], v["faults"]) for v in kw["executed_prefix"].values()] == [(4, []), (4, [])]
    kw, _facts = C2.executed_prefix_kw(sources(tmp_path, linked, full, replay_k=3), executed(full), linkage, {"r0": 0, "r1": 1}, LABELS)
    assert kw == {"executed_prefix": {}}
    assert said() == []


def test_a_replay_that_takes_no_executed_prefix_is_given_none_and_an_early_stop_is_warned_of(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    monkeypatch.setattr(C2.SR, "sampled_replay", lambda progs, bmap, reader, weights, plan=False: {})
    full = {"a": [5, 6, 7, 8], "b": [5, 6, 7, 8]}
    assert C2.executed_prefix_kw(sources(tmp_path, tokens=full), executed(full), LINKAGE, {"r0": 0, "r1": 1}, LABELS)[0] == {}
    assert said() == []
    kw, facts = C2.executed_prefix_kw(sources(tmp_path), executed(), LINKAGE, {"r0": 0, "r1": 1}, LABELS)
    assert kw == {} and facts["r1"]["a_r"] == 2
    warned = ("WARNING [delta] WARNING: this tree's sampled_replay has no executed_prefix=: the EOS-stopped request's rows past the stop are "
              "sampled as ordinary strata and grade 'not evaluated' -> C2 partial, never PASS")
    assert said() == [warned]


def reconciled(s: C2.ReplaySources, population: Record | None, kw: Record | None = None, tokens: Record | None = None) -> Record:
    sr: Record = {"population": {"not_executed_by_rule": population}}
    C2.reconcile(s, sr, {}, None, executed(tokens), {"executed_prefix": {}} if kw is None else kw)
    return sr


def test_the_commit_s_and_the_population_s_not_executed_class_are_reconciled_on_the_record(tmp_path: Path, said: Lines) -> None:
    s = sources(tmp_path)
    s.row["commit"]["coverage"] = {"ok": True}
    sr = reconciled(s, POPULATION)
    rec = sr["not_executed_reconciliation"]
    assert rec is s.row["commit"]["not_executed_reconciliation"] and rec["ok"] and not rec["problems"]
    assert rec["n_exec"] == {"r0": {"commit": 4, "population": 4, "equal": True}, "r1": {"commit": 2, "population": 2, "equal": True}}
    assert rec["by_family"] == {"instance_outputs": {"commit": 2, "population": 2, "equal": True}}
    assert s.row["commit"]["coverage"] == {"ok": True}
    assert said() == ["INFO [delta] not-executed reconciliation (pair 0): AGREE n_exec r0 4/4, r1 2/2; row families [('instance_outputs', 2, 2)]"]


def test_a_disagreement_fails_the_commit_s_coverage_by_name(tmp_path: Path, said: Lines) -> None:
    s = sources(tmp_path)
    s.row["commit"]["coverage"] = {"ok": True}
    one_short = POPULATION | {"totals": {"by_family": {"instance_outputs": {"N_excluded": 1}}}}
    rec = reconciled(s, one_short)["not_executed_reconciliation"]
    why = "instance_outputs: the Commit classified 2 identities not-executed-by-rule, the population excluded 1"
    assert not rec["ok"] and rec["problems"] == [why]
    assert s.row["commit"]["coverage"] == {"ok": False, "error": f"one mechanism: {why}"}
    line = "WARNING [delta] not-executed reconciliation (pair 0): DISAGREE n_exec r0 4/4, r1 2/2; row families [('instance_outputs', 2, 1)]"
    assert said() == [f"{line}; PROBLEMS {[why]}"]
    failed = sources(tmp_path)
    failed.row["commit"]["coverage"] = {"ok": False, "error": "an identity is unbound"}
    reconciled(failed, one_short)
    assert failed.row["commit"]["coverage"] == {"ok": False, "error": "an identity is unbound"}, "an earlier failure keeps its own reason"
    absent = sources(tmp_path)
    assert not reconciled(absent, None)["not_executed_reconciliation"]["ok"] and "coverage" not in absent.row["commit"]


def test_nothing_is_reconciled_without_the_class_or_without_an_early_stop(tmp_path: Path, said: Lines) -> None:
    full = {"a": [5, 6, 7, 8], "b": [5, 6, 7, 8]}
    for s, kw, tokens in ((sources(tmp_path), {}, None), (sources(tmp_path, tokens=full), None, full)):
        assert reconciled(s, POPULATION, kw, tokens) == {"population": {"not_executed_by_rule": POPULATION}}
        assert "not_executed_reconciliation" not in s.row["commit"]
    assert said() == []


def test_the_seed_is_the_prover_s_else_derived_from_the_challenge_and_the_run_root_else_from_the_run_root_alone() -> None:
    rc = SimpleNamespace(run_root=bytes(range(64)))
    assert C2.seed_of(SimpleNamespace(replay_seed=0, replay_challenge="c0ffee"), rc) == (0, RC.PROVER_SEED_SOURCE, "prover-chosen", None)
    seed, source = C2.SR.challenge_seed("c0ffee", rc.run_root)
    assert C2.seed_of(SimpleNamespace(replay_seed=None, replay_challenge="c0ffee"), rc) == (seed, source, "challenge", "c0ffee")
    assert C2.seed_of(SimpleNamespace(), rc) == (C2.CH.root_seed(rc.run_root), RC.ROOT_SEED_SOURCE, "run-root", None)
    assert {"prover-chosen", "challenge", "run-root"} <= set(RC.SEED_FORMS)


class Steps:
    """`evaluate`'s steps that read Programs or committed values, each a stand-in that keeps what it was called with."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.replays: list[Record] = []
        self.reader = SimpleNamespace(record=lambda: dict(OPENED))
        # each Program's one row is of a family with no row kernel and no delegate: a row as `evaluate` may read it, which the
        # replay evaluates nothing of
        self.loaded: Record = {"r0": {"instances": {"rows": [{"spec": "NotReplayed_v1{request=0}"}]}, "dir": "/programs/r0", "result": {"T": 3}},
                               "r1": {"instances": {"rows": [{"spec": "NotReplayed_v1{request=1}"}]}, "dir": "/programs/r1", "result": {"T": 3}}}
        self.progs: Record = {k: v["instances"] for k, v in self.loaded.items()}
        monkeypatch.setattr(C2, "programs_of_record", self.programs_of_record)
        monkeypatch.setattr(C2.SR, "ProgramIndex", self.index)
        monkeypatch.setattr(C2.SRW, "readers", self.readers)
        monkeypatch.setattr(C2.SR, "boundary_linkage", self.boundary_linkage)
        monkeypatch.setattr(C2.SR, "sampled_replay", self.sampled_replay)
        monkeypatch.setattr(C2.SD, "touch", lambda com, reads: self.calls.append(("touch", com.name, reads)))

    def programs_of_record(self, s: C2.ReplaySources, labels: dict[str, str], *, definitions: bool = True) -> tuple[Record, dict[str, str]]:
        self.calls.append(("programs", labels, definitions))
        return self.loaded, {"r0": "d" * 8, "r1": "e" * 8}

    def index(self, instances: Record, program_dir: str | None = None) -> SimpleNamespace:
        self.calls.append(("index", instances, program_dir))
        return SimpleNamespace(prefix="model.", embed_op_path=lambda: "model/embed")

    def readers(self, request: str | None, com: Any, rc: Any, weights: Any, programs: Record, bmap: Record, arrivals: dict[str, int],
                loaded: Record, manifest: Any, row: str | None, digests: dict[str, str]) -> tuple[Any, Any]:
        self.calls.append(("readers", request, com.name, weights, programs, bmap, arrivals, loaded, manifest, row, digests))
        return weights, self.reader

    def boundary_linkage(self, bmap: Record, reader: Any, requests: list[Record], **kw: Any) -> Record:
        self.calls.append(("linkage", bmap, reader, requests, kw))
        return dict(LINKAGE)

    def sampled_replay(self, progs: Record, bmap: Record, reader: Any, weights: Any, plan: bool = False, executed_prefix: Record | None = None,
                       **kw: Any) -> Record:
        self.calls.append(("replay", progs, bmap, reader, weights, plan))
        self.replays.append({"executed_prefix": executed_prefix} | kw)
        if plan:
            return {"plan": True, "picks": 4}
        return {"result": True, "sample": {"seed": kw["seed"], "evaluated": 4}, "population": {"not_executed_by_rule": POPULATION},
                "value_reads_in_children": 3}


def test_the_replay_and_the_linkage_are_given_the_run_s_facts_of_record_and_the_record_names_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    steps, s, held, exported = Steps(monkeypatch), sources(tmp_path, retain="host"), Record(), list[tuple[Any, ...]]()
    s.after_replay, s.reads = lambda *a: exported.append(a), {}
    sr, lk = C2.evaluate(s, held)
    arr = {"r0": 0, "r1": 1}
    assert [c[0] for c in steps.calls] == ["programs", "index", "readers", "linkage", "replay", "replay"]
    assert steps.calls[0] == ("programs", LABELS, True)
    assert steps.calls[1] == ("index", {"rows": [{"spec": "NotReplayed_v1{request=0}"}]}, "/programs/r0")
    assert steps.calls[2] == ("readers", None, "stub", Weights("model."), steps.progs, {"entries": 5}, arr, steps.loaded,
                              {"manifest_digest": "m" * 8}, "row", {"r0": "d" * 8, "r1": "e" * 8})
    linked = steps.calls[3]
    assert linked[1:3] == ({"entries": 5}, steps.reader) and linked[4] == {"embed_op_path": "model/embed", "eos_of_record": [2], "eos_lag_rows": 0}
    assert linked[3] == [
        {"request_id": "r0", "arrive_step": 0, "prompt_token_ids": [1, 9], "cached": 0, "seed": 11, "max_tokens": 4, "program_steps": 4,
         "steps": [0, 1, 2, 3]},
        {"request_id": "r1", "arrive_step": 1, "prompt_token_ids": [1], "cached": 0, "seed": 3, "max_tokens": 4, "program_steps": 4,
         "steps": [1, 2]}], "each request is linked over its executed prefix: b's two steps, from its arrival"
    assert steps.calls[4] == ("replay", steps.progs, {"entries": 5}, steps.reader, Weights("model."), False)
    assert steps.calls[5] == steps.calls[4][:5] + (True,), "the sample is taken again, read only, for the reads this process must see"
    kw = steps.replays[0]
    assert steps.replays[1] == kw
    assert set(kw.pop("executed_prefix")) == {"r0", "r1"}
    assert kw == {"arrivals": arr, "per_stratum": 2, "seed": 7, "seed_source": RC.PROVER_SEED_SOURCE, "seed_challenge": None, "max_evaluate": 9,
                  "weights_check": {"equal": True}, "targets": [], "replay_k": None, "replay_draw": "family", "workers": None, "retention": "host",
                  "ring_slots": None, "manifest": {"manifest_digest": "m" * 8}, "worker_peak": None,
                  "program_dirs": {"r0": "/programs/r0", "r1": "/programs/r1"}, "cached_prefixes": {},
                  "certified": {"r0": {"request_id": "a", "seed": 11}, "r1": {"request_id": "b", "seed": 3}}}
    assert lk == LINKAGE and s.reads == {"plan": True, "picks": 4}
    assert sr["sample"] == {"seed": 7, "evaluated": 4, "seed_form": "prover-chosen"}
    assert sr["programs"] == {"r0": {"dir": "/programs/r0", "program_digest": "d" * 8}, "r1": {"dir": "/programs/r1", "program_digest": "e" * 8}}
    epr = sr["executed_prefix_of_record"]
    assert epr["r1"] == {"served_output_len": 2, "executed_prefix_len": 2, "cap": 4, "cap_exhausted": False, "eos_stop": True}
    assert (epr["lag"], epr["ok"], epr["problems"], epr["mechanism"]) == (0, True, [], "population.not_executed_by_rule")
    assert epr["facts_of_record"]["r1"]["a_r"] == 2
    assert sr["not_executed_reconciliation"]["ok"] and s.row["commit"]["not_executed_reconciliation"] is sr["not_executed_reconciliation"]
    assert sr["value_source"] == OPENED | {"forked_children": 3} and "value_reads_in_children" not in sr
    assert s.row["commit"]["opened_values"]["replay_linkage"] == sr["value_source"]
    assert set(held) == {"bm", "loaded", "digs", "progs", "pi0", "rd"} and held["rd"] is steps.reader
    (told,) = exported
    assert told[0] is sr and told[1:5] == (steps.progs, {"entries": 5}, steps.reader, Weights("model.")) and told[5] == arr
    assert told[7:] == (steps.loaded, {"r0": "d" * 8, "r1": "e" * 8})
    assert said() == [
        "INFO [delta] replay/linkage horizons = the executed prefix of record: {'r0': 4, 'r1': 2} (LAG 0; ok)",
        "INFO [delta] not-executed class -> sampled_replay(executed_prefix=): r0 a_r 4 n_exec 4 cap 4, r1 a_r 2 n_exec 2 cap 4; no faults",
        "INFO [delta] not-executed reconciliation (pair 0): AGREE n_exec r0 4/4, r1 2/2; row families [('instance_outputs', 2, 2)]",
        ("INFO [delta] C2 SAMPLED REPLAY + LINKAGE opened values (pair 0): 6 reads, 12 leaves, 96 B verified against the run root in 0.5 s (5 "
         "steps bound); not retained 0, failed 0; forked evaluators 3")]


def test_arrivals_the_engine_s_schedule_moved_are_the_replay_s_and_the_linkage_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                said: Lines) -> None:
    steps, s = Steps(monkeypatch), sources(tmp_path, replay_seed=None, replay_challenge="c0ffee")
    s.row["commit"]["binding"]["coverage"]["schedule"] = {"use": "observed", "observed_arrive_steps": {"r0": 0, "r1": 2},
                                                          "declared_arrive_steps": {"r0": 0, "r1": 1}}
    sr, _lk = C2.evaluate(s, {})
    arr = {"r0": 0, "r1": 2}
    assert steps.calls[2][6] == arr and steps.replays[0]["arrivals"] == arr
    assert [r["steps"] for r in steps.calls[3][3]] == [[0, 1, 2, 3], [2, 3]]
    seed, source = C2.SR.challenge_seed("c0ffee", bytes(64))
    assert (steps.replays[0]["seed"], steps.replays[0]["seed_source"], steps.replays[0]["seed_challenge"]) == (seed, source, "c0ffee")
    assert sr["sample"]["seed_form"] == "challenge"
    assert said()[0] == "INFO [delta] replay/linkage arrivals follow the engine schedule of record: {'r0': 0, 'r1': 2} (declared {'r0': 0, 'r1': 1})"


def test_a_plan_makes_the_sample_s_reads_and_evaluates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps, s = Steps(monkeypatch), sources(tmp_path)
    s.weights_check = lambda: pytest.fail("a plan checks no weights")
    held: Record = {}
    assert C2.plan(s) == {"replay": {"plan": True, "picks": 4}, "linkage": True}
    assert steps.calls[0] == ("programs", LABELS, False), "a plan evaluates no row, so it checks no Definition"
    assert [c[0] for c in steps.calls] == ["programs", "index", "readers", "linkage", "replay"] and steps.calls[4][5] is True
    assert steps.replays[0]["weights_check"] is None
    assert C2.evaluate(s, held, plan=True) == ({"plan": True, "picks": 4}, LINKAGE)
    assert "opened_values" not in s.row["commit"] and "not_executed_reconciliation" not in s.row["commit"]


def test_a_planner_s_plan_is_made_from_its_programs_and_only_the_reads_are_made_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps, s, asked = Steps(monkeypatch), sources(tmp_path), list[Record]()

    def plan(request: Record) -> Record:
        asked.append(request)
        return {"plan": {"picks": 4}, "reads": ["read0", "read1", "read2"]}
    planner = SimpleNamespace(ready=lambda: {"program_steps": {"r0": 4, "r1": 4}, "embed_op_path": "model/embed", "prepare_s": 1.5}, plan=plan)
    sr, lk = C2.evaluate(s, {}, plan=True, planner=planner)
    assert [c[0] for c in steps.calls] == ["readers", "linkage", "touch"], "no Program is loaded here and no sample taken"
    assert steps.calls[0][3:6] == (C2.no_weights, {}, {"entries": 5}) and steps.calls[0][7] == {}
    with pytest.raises(RuntimeError, match="the Programs are a planner's, whose plan evaluates no row"):
        steps.calls[0][3]("model.layers.0.weight")
    assert steps.calls[1][4]["embed_op_path"] == "model/embed" and [r["steps"] for r in steps.calls[1][3]] == [[0, 1, 2, 3], [1, 2]]
    assert steps.calls[2] == ("touch", "stub", ["read0", "read1", "read2"])
    (request,) = asked
    assert set(request.pop("sr_kw")["executed_prefix"]) == {"r0", "r1"}
    assert request == {"binding_map": str(tmp_path / "binding_map_p0.json"), "arrivals": {"r0": 0, "r1": 1}, "seed": 7,
                       "seed_source": RC.PROVER_SEED_SOURCE, "seed_challenge": None}
    sample_s, reads_s = sr.pop("draw_s"), sr.pop("reads_s")
    assert sr == {"picks": 4, "prepare_s": 1.5, "reads": 3} and sample_s >= 0 and reads_s >= 0 and lk == LINKAGE


def test_for_a_planner_that_is_not_ready_the_programs_are_loaded_here_and_the_plan_is_still_its(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps, s = Steps(monkeypatch), sources(tmp_path)
    planner = SimpleNamespace(ready=lambda: None, plan=lambda request: {"plan": {"picks": 1}, "reads": []})
    sr, _lk = C2.evaluate(s, {}, plan=True, planner=planner)
    assert [c[0] for c in steps.calls] == ["programs", "index", "readers", "linkage", "touch"]
    assert steps.calls[2][3] == Weights("model.") and sr["picks"] == 1 and sr["prepare_s"] is None
