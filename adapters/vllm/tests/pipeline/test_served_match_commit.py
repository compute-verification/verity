"""`served_match.ServedCommit`, the served row in a Commit's treatment process, on a CPU.

The source (`ServedCompiledSource`) is a stand-in that records what the row asks of it.  What the row reads off an engine
and a committer is stood in too, each a function with tests of its own: `facts_of`, `served_weights`, `build_here` and
`check_committed` (`test_served_match_facts.py`), the source's `linkage` and the check's `population`.  So these cases show
what the row does around them: what it refuses before vLLM compiles, what it asks of the source and in which order, the two
files it writes when it builds again and its exit when the compile artifact is not the Build of record's, and one run's
C2 record.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.acquire import served_artifact as SA
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.acquire.sources import served_compiled_source
from verity_vllm.check.replay import sample, served_check
from verity_vllm.config import UsageError
from verity_vllm.engine import env as engine_env
from verity_vllm.pipeline import served_match as sm
from verity_vllm.query.manifest.format import SERVED_IDS

WINNERS = {"k0/a.best_config": {"BLOCK_M": 64}}
ARTIFACT = SA.artifact_of({"sha-a": {"config": {"num_warps": 4}}, "sha-b": {"config": {"num_warps": 8}}}, WINNERS)
OTHER = SA.artifact_of({"sha-a": {"config": {"num_warps": 2}}, "sha-c": {"config": {"num_warps": 1}}}, {"k1/b.best_config": {"BLOCK_M": 32}})
DIFFERENCES = ["kernels sha-a: pinned {'num_warps': 4}, loaded {'num_warps': 2}", "kernels sha-b: pinned {'num_warps': 8}, loaded absent",
               "kernels sha-c: pinned absent, loaded {'num_warps': 1}", "winners k0/a.best_config: pinned {'BLOCK_M': 64}, loaded absent",
               "winners k1/b.best_config: pinned absent, loaded {'BLOCK_M': 32}"]
WORKLOAD = {"requests": [{"prompt_token_ids": [1, 2], "max_tokens": 3, "arrive_step": 0}]}
SCHEDULE = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 2, "max_tokens": 3}]

RC = SimpleNamespace(run_root=bytes(range(64)))
COM = SimpleNamespace(name="a committer")
REQS = [{"request_id": "req-a", "prompt_token_ids": [1, 2]}, {"request_id": 7, "prompt_token_ids": (3,)}]
CHAIN = {"checked": 1, "mismatches": 0, "first": [], "passed": True}
TALLY = {"gemm": [5, 5], SERVED_IDS: [3, 3]}


def _build(**over: Any) -> dict[str, Any]:
    """A Build of record as far as the row reads it; its step plan is not in step order."""
    return {"program_digest": "p" * 128, "manifest_digest": "m" * 128, "complete": True, "artifact": ARTIFACT, "populations": {"identities": 7},
            "steps_plan": [{"step": 1, "requests": [{"request_id": "r0"}, {"request_id": "r1"}]}, {"step": 0, "requests": [{"request_id": "r0"}]}],
            **over}


def _report(**over: Any) -> dict[str, Any]:
    """The check's report of a run whose every value replayed."""
    return {"ok": True, "tally": TALLY, "steps": {"0": {SERVED_IDS: {"ok": True}}, "1": {SERVED_IDS: {"ok": True}}}, "failures": [],
            "steps_checked": [0, 1], "steps_unrun": [], "rows_per_request": 16, "picks": {"r0": [0, 1]}, **over}


class _Source:
    """`ServedCompiledSource` as the row drives it: each call in order, one captured slot, no error."""

    made: list[_Source] = []

    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.calls: list[tuple[Any, ...]] = []
        self.errors: list[str] = []
        self.slots: dict[tuple[str, ...], Any] = {("c00.input/ids",): object()}
        _Source.made.append(self)

    def install_hooks(self) -> None:
        self.calls.append(("install_hooks",))

    def note_engine(self, model: Any) -> None:
        self.calls.append(("note_engine", model))

    def use_build(self, build: dict[str, Any]) -> None:
        self.calls.append(("use_build", build))

    def attach(self, com: Any) -> None:
        self.calls.append(("attach", com))

    def summary(self) -> dict[str, Any]:
        return {"stats": {"slots": len(self.slots)}}


@pytest.fixture
def process(monkeypatch, tmp_path):
    """A process whose source is the stand-in and which no compile switch is set in; called, it is given its own compile dirs
    under the test's directory and returns the Inductor one."""
    monkeypatch.setattr(_Source, "made", [])
    monkeypatch.setattr(served_compiled_source, "ServedCompiledSource", _Source)
    for name in ("TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", engine_env.INDUCTOR_DIR_MARK, *engine_env.COMPILE_FRESH):
        monkeypatch.setenv(name, "")    # set first, so that the test's end takes out what the row sets
        monkeypatch.delenv(name)

    def own() -> Any:
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
        monkeypatch.setenv(engine_env.INDUCTOR_DIR_MARK, "1")
        return tmp_path / "inductor"

    return own


def _fresh() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in engine_env.COMPILE_FRESH}


def test_a_process_that_was_not_given_its_own_compile_dirs_is_refused_before_anything_is_hooked(process):
    with pytest.raises(UsageError) as stop:
        sm.ServedCommit(_build(), None)
    assert str(stop.value) == ("--compiled-taps served: the Build reads this process's own compile artifact; export its own "
                               f"TORCHINDUCTOR_CACHE_DIR and VLLM_CACHE_ROOT with {engine_env.INDUCTOR_DIR_MARK}=1 (ops/served_commit.sh)")
    assert _Source.made == [] and set(_fresh().values()) == {None}


@pytest.mark.parametrize("artifact", [None, {"kernels": {}}, {"winners": {}}, "a digest"])
def test_a_build_of_record_without_a_whole_compile_artifact_is_refused_before_anything_is_hooked(process, artifact):
    process()
    with pytest.raises(UsageError, match="records no compile artifact"):
        sm.ServedCommit(_build(artifact=artifact), None)
    assert _Source.made == [] and set(_fresh().values()) == {None}


@pytest.mark.parametrize("fault", [None, "compiled_stale_slot"])
def test_the_row_pins_the_builds_artifact_compiles_fresh_and_hooks_its_source_with_the_fault(process, fault):
    process()
    build = _build()
    row = sm.ServedCommit(build, fault)
    src, = _Source.made
    assert row.src is src and (src.fault, src.calls) == (fault, [("install_hooks",)])
    assert row.build is build and row.pin is build["artifact"] and (row.facts, row.weight) == ({}, None)
    assert _fresh() == engine_env.COMPILE_FRESH


def _row(process: Any, **build: Any) -> sm.ServedCommit:
    process()
    return sm.ServedCommit(_build(**build), None)


def _engine(monkeypatch) -> tuple[SimpleNamespace, dict[str, Any], Any]:
    """An engine, with the facts and the weights `facts_of` and `served_weights` are stood in to read off it."""
    eng, facts = SimpleNamespace(model=object()), {"case": "A_CASE", "cache_dir": "/a/cache"}

    def weight(name: str) -> str:
        return name

    def facts_of(e: Any) -> dict[str, Any]:
        assert e is eng
        return facts

    def served_weights(model: Any) -> Any:
        assert model is eng.model
        return weight

    monkeypatch.setattr(sm, "facts_of", facts_of)
    monkeypatch.setattr(sm, "served_weights", served_weights)
    return eng, facts, weight


def test_once_the_engine_is_up_the_source_notes_it_then_takes_the_build_and_the_row_holds_its_facts_and_weights(monkeypatch, process):
    row = _row(process)
    eng, facts, weight = _engine(monkeypatch)
    assert row.engine(eng) == {"compiled_source": {"stats": {"slots": 1}}, "served_facts": facts}
    assert row.src.calls == [("install_hooks",), ("note_engine", eng.model), ("use_build", row.build)]
    assert row.src.calls[2][1] is row.build and row.facts is facts and row.weight is weight


@pytest.mark.parametrize(("errors", "slots", "said"), [
    (["a hook failed"], {("s",): 1}, r"no captured slots / errors \['a hook failed'\]"),
    ([], {}, r"no captured slots / errors \[\]"),
])
def test_an_engine_whose_source_captured_no_slot_or_met_an_error_is_refused(monkeypatch, process, errors, slots, said):
    row = _row(process)
    row.src.errors, row.src.slots = errors, slots
    with pytest.raises(UsageError, match=said):
        row.engine(_engine(monkeypatch)[0])


def test_the_source_is_attached_to_a_native_collect_committer_and_to_no_other(process):
    row = _row(process)
    for com in (SimpleNamespace(name="a wrapping committer"), SimpleNamespace(name="half", _nc=object()),
                SimpleNamespace(name="other half", attach_source=print)):
        with pytest.raises(UsageError, match=f"needs the native_collect committer \\(got {com.name}\\)"):
            row.attach(com)
    com = SimpleNamespace(name="native_collect", _nc=object(), attach_source=print)
    row.attach(com)
    assert row.src.calls == [("install_hooks",), ("attach", com)]


def _rebuild(monkeypatch, row: sm.ServedCommit, built: Any, out: Any) -> Any:
    """`row.rebuild` where `build_here` gives ``built``, or raises it; what `build_here` was asked is checked."""
    row.facts = {"case": "A_CASE"}

    def build_here(src: Any, facts: dict[str, Any], schedule: list[dict[str, Any]], steps: int, to: str) -> dict[str, Any]:
        assert src is row.src and facts is row.facts and (schedule, steps, to) == (SCHEDULE, 3, str(out))
        if isinstance(built, Exception):
            raise built
        return built

    monkeypatch.setattr(sm, "build_here", build_here)
    return row.rebuild(WORKLOAD, str(out))


def _files(out: Any) -> tuple[Any, Any]:
    return json.loads((out / "build_rebuilt.json").read_text()), json.loads((out / "artifact_check.json").read_text())


def test_a_process_whose_build_read_the_artifact_of_record_writes_both_files_and_returns_its_build(monkeypatch, process, tmp_path, capsys):
    built = _build(program_digest="q" * 128)
    assert _rebuild(monkeypatch, _row(process), built, tmp_path) is built
    check = {"pinned": ARTIFACT["digest"], "loaded": ARTIFACT["digest"], "equal": True, "differences": []}
    assert _files(tmp_path) == (built, check)
    assert (tmp_path / "build_rebuilt.json").read_text() == json.dumps(built, indent=1)
    assert (tmp_path / "artifact_check.json").read_text() == json.dumps(check, indent=1)
    assert capsys.readouterr().out == f"[delta] compile artifact {ARTIFACT['digest'][:16]}: this process's is the Build of record's\n"


def test_a_process_whose_build_read_another_artifact_writes_both_files_with_each_difference_and_exits_3(monkeypatch, process, tmp_path, capsys):
    built = _build(artifact=OTHER)
    with pytest.raises(SystemExit) as stop:
        _rebuild(monkeypatch, _row(process), built, tmp_path)
    assert stop.value.code == 3
    assert _files(tmp_path) == (built, {"pinned": ARTIFACT["digest"], "loaded": OTHER["digest"], "equal": False, "differences": DIFFERENCES})
    assert capsys.readouterr().out == (f"[delta] REFUSE: this process loaded compile artifact {OTHER['digest'][:16]}, not the Build of record's "
                                       f"{ARTIFACT['digest'][:16]} (5 differences: {DIFFERENCES[:3]})\n")


def test_a_build_that_stops_on_anything_but_a_refusal_of_the_artifact_stops_the_row_and_writes_no_file(monkeypatch, process, tmp_path):
    with pytest.raises(UsageError, match="^2 computation_graph.py under /a/cache$"):
        _rebuild(monkeypatch, _row(process), UsageError("2 computation_graph.py under /a/cache"), tmp_path)
    assert list(tmp_path.iterdir()) == []


REFUSED: dict[str, Any] = {"program_digest": None, "manifest_digest": None, "complete": False, "unmodelled": {"build_refused": "no template models kernel k9"},
                           "populations": {"identities": 0}, "identities": []}


def test_a_refused_build_is_recorded_and_its_process_is_held_to_the_winners_of_record_alone(monkeypatch, process, tmp_path, capsys):
    row, inductor = _row(process), process()
    (inductor / "k0").mkdir(parents=True)
    (inductor / "k0" / "a.best_config").write_text(json.dumps({"BLOCK_M": 64, "time_taken_ms": 9}))
    assert _rebuild(monkeypatch, row, Unmodelled("no template models kernel k9"), tmp_path) == REFUSED
    assert _files(tmp_path) == (REFUSED, {"pinned": ARTIFACT["digest"], "loaded": None, "equal": True, "differences": []})
    assert capsys.readouterr().out == f"[delta] compile artifact {ARTIFACT['digest'][:16]}: this process's is the Build of record's\n"


def test_a_refused_build_in_a_process_whose_own_compile_dirs_are_gone_is_refused_by_name(monkeypatch, process, tmp_path):
    row = _row(process)
    monkeypatch.delenv(engine_env.INDUCTOR_DIR_MARK)
    with pytest.raises(UsageError, match="^this process was not given its own compile dirs "):
        _rebuild(monkeypatch, row, Unmodelled("no template models kernel k9"), tmp_path)
    assert [f.name for f in tmp_path.iterdir()] == ["build_rebuilt.json"]


def test_a_refused_build_whose_process_has_other_winners_exits_3(monkeypatch, process, tmp_path, capsys):
    row, inductor = _row(process), process()
    (inductor / "k1").mkdir(parents=True)
    (inductor / "k1" / "b.best_config").write_text(json.dumps({"BLOCK_M": 32}))
    with pytest.raises(SystemExit) as stop:
        _rebuild(monkeypatch, row, Unmodelled("no template models kernel k9"), tmp_path)
    differences = ["winners k0/a.best_config: pinned {'BLOCK_M': 64}, loaded absent", "winners k1/b.best_config: pinned absent, loaded {'BLOCK_M': 32}"]
    assert stop.value.code == 3
    assert _files(tmp_path) == (REFUSED, {"pinned": ARTIFACT["digest"], "loaded": None, "equal": False, "differences": differences})
    assert capsys.readouterr().out == ("[delta] REFUSE: this process loaded compile artifact of a refused Bui, not the Build of record's "
                                       f"{ARTIFACT['digest'][:16]} (2 differences: {differences})\n")


def _c2(monkeypatch, row: sm.ServedCommit, report: Any, chain: dict[str, Any] = CHAIN, *, challenge: str | None = "abc",
        replay_seed: int | None = None, weight: Any = print) -> tuple[dict[str, Any], dict[str, Any]]:
    """One run's C2 where the engine's weights are ``weight``, the check gives ``report``, or raises it, and the committed chain
    is ``chain``: the record, and what the check and the population were asked."""
    asked: dict[str, Any] = {}
    row.weight = weight

    def check_committed(build: Any, com: Any, weight: Any, prompts: Any, order: Any, *, seed: int | None) -> dict[str, Any]:
        asked["check"] = (build, com, weight, prompts, order, seed)
        if isinstance(report, Exception):
            raise report
        return report

    def linkage(com: Any) -> dict[str, Any]:
        assert com is COM
        return chain

    def population(build: Any, rep: Any, *, ranks: list[int]) -> dict[str, Any]:
        asked["population"] = (build, rep, ranks)
        return {"query_total": 3}

    monkeypatch.setattr(sm, "check_committed", check_committed)
    monkeypatch.setattr(served_compiled_source, "linkage", linkage)
    monkeypatch.setattr(served_check, "population", population)
    return row.c2(COM, RC, REQS, challenge, replay_seed), asked


REPLAY_METHOD = ("sampled-exact-replay: the served Build's Program (acquire.served) on the committed values word for word "
                 "(served_check.py): every call of every piece on every row; each GEMM and attention op on a seeded sample of "
                 "each request's rows; the logits and the sampled ids of every request")
LINKAGE_METHOD = ("per (engine step, request): the committed ids are the declared prompt at the request's first step and its "
                  "previous step's sampled id at every later step (served_check inputs); committed sampled(s) == committed ids(s+1)")


def test_a_run_whose_every_value_replays_and_whose_ids_link_passes_c2_under_the_challenges_seed(monkeypatch, process):
    row, report = _row(process), _report()
    out, asked = _c2(monkeypatch, row, report)
    seed, source = sample.challenge_seed("abc", RC.run_root)
    build, com, weight, prompts, order, used = asked["check"]
    assert build is row.build and com is COM and weight is print and used == seed
    assert (prompts, order) == ({"r0": [1, 2], "r1": [3]}, {"req-a": "r0", "7": "r1"})
    assert asked["population"][0] is row.build and asked["population"][1] is report and asked["population"][2] == [0]
    check = {"ok": True, "checked": 8, "equal": 8, "launches_checked": 8, "launches_exact": 8, "error": None, "seed": seed, "tally": TALLY,
             "failures": [], "steps_checked": [0, 1], "steps_unrun": []}
    assert out == {
        "components": {
            "sampled_replay": {
                "result": True, "seed": seed, "seed_source": source, "seed_form": "challenge",
                "sample": {"seed": seed, "seed_source": source, "seed_form": "challenge", "challenge": "abc", "rows_per_request": 16,
                           "picks": {"r0": [0, 1]}},
                "population": {"n": 7, "digest": "m" * 128, "not_evaluable": {}, "query_total": 3},
                "n": 8, "exact": 8, "tally": TALLY, "steps_checked": [0, 1], "steps_unrun": [], "mismatch": [], "errors": [],
                "why": "every value replays word for word from the committed values", "method": REPLAY_METHOD},
            "boundary_linkage": {
                "result": True, "pairs": 3, "unlinked_n": 0, "linked": 3, "unlinked": [], "horizon": {"r0": [0, 1], "r1": [1, 1]},
                "prompt_linked": {"r0": True, "r1": True}, "embed_op_path": SERVED_IDS, "committed_chain": CHAIN, "why": None,
                "method": LINKAGE_METHOD}},
        "ok": True,
        "line": f"OK 8/8 values exact over steps [0, 1] (unrun []), seed challenge {seed}; linkage 3/3, committed chain {CHAIN}; "
                "source stats {'slots': 1}; first failures [] ",
        "value_correspondence": {"population": {"identities": 7}, "n": 8, "seed": seed, "compiled_check": check}}


@pytest.mark.parametrize(("challenge", "replay_seed", "seed", "form"), [
    ("abc", 5, 5, "prover-chosen"),
    (None, None, 0x0001020304050607, "run-root"),
    ("", None, 0x0001020304050607, "run-root"),
])
def test_the_seed_is_the_one_given_else_the_challenges_else_the_run_roots_and_only_a_challenge_that_seeds_is_recorded(
        monkeypatch, process, challenge, replay_seed, seed, form):
    out, asked = _c2(monkeypatch, _row(process), _report(), challenge=challenge, replay_seed=replay_seed)
    replay = out["components"]["sampled_replay"]
    assert asked["check"][5] == seed == replay["seed"] == replay["sample"]["seed"] == out["value_correspondence"]["seed"]
    assert (replay["seed_form"], replay["sample"]["seed_form"], replay["sample"]["challenge"]) == (form, form, None)
    assert f"seed {form} {seed}; " in out["line"]


@pytest.mark.parametrize(("steps", "unlinked", "prompt_linked"), [
    ({"0": {SERVED_IDS: {"ok": True}}, "1": {SERVED_IDS: {"ok": False}}}, [(1, "r0"), (1, "r1")], {"r0": True, "r1": False}),
    ({"1": {SERVED_IDS: {"ok": True}}}, [(0, "r0")], {"r0": False, "r1": True}),
    ({"0": {}, "1": {SERVED_IDS: {"ok": True}}}, [(0, "r0")], {"r0": False, "r1": True}),
    ({"0": {SERVED_IDS: {}}, "1": {SERVED_IDS: {"ok": True}}}, [(0, "r0")], {"r0": False, "r1": True}),
    ({"0": {SERVED_IDS: None}, "1": None}, [(0, "r0"), (1, "r0"), (1, "r1")], {"r0": False, "r1": False}),
])
def test_a_step_whose_ids_the_check_did_not_pass_unlinks_each_of_its_requests_and_fails_c2(monkeypatch, process, steps, unlinked, prompt_linked):
    out, _ = _c2(monkeypatch, _row(process), _report(steps=steps))
    linkage = out["components"]["boundary_linkage"]
    assert linkage["unlinked"] == [{"step": s, "request": r} for s, r in unlinked] and linkage["prompt_linked"] == prompt_linked
    assert (linkage["result"], linkage["pairs"], linkage["unlinked_n"], linkage["linked"]) == (False, 3, len(unlinked), 3 - len(unlinked))
    assert linkage["why"] == f"{len(unlinked)} (step, request) boundaries unlinked; committed chain {CHAIN}"
    assert linkage["horizon"] == {"r0": [0, 1], "r1": [1, 1]}
    assert out["components"]["sampled_replay"]["result"] is True and out["ok"] is False
    assert out["line"].startswith("OK 8/8 ") and f"; linkage {3 - len(unlinked)}/3, " in out["line"]


def test_a_committed_chain_that_did_not_pass_or_a_build_with_no_step_fails_the_linkage(monkeypatch, process):
    chain = {"checked": 0, "mismatches": 0, "first": [], "passed": False}
    out, _ = _c2(monkeypatch, _row(process), _report(), chain)
    linkage = out["components"]["boundary_linkage"]
    assert (linkage["result"], linkage["unlinked"], linkage["committed_chain"]) == (False, [], chain) and out["ok"] is False
    assert linkage["why"] == f"0 (step, request) boundaries unlinked; committed chain {chain}"
    out, _ = _c2(monkeypatch, _row(process, steps_plan=[]), _report())
    linkage = out["components"]["boundary_linkage"]
    assert (linkage["result"], linkage["pairs"], linkage["horizon"], linkage["prompt_linked"]) == (False, 0, {}, {}) and out["ok"] is False
    assert out["components"]["sampled_replay"]["result"] is True


def test_more_than_sixteen_unlinked_boundaries_are_counted_and_the_first_sixteen_listed(monkeypatch, process):
    plan = [{"step": s, "requests": [{"request_id": "r0"}]} for s in range(20)]
    out, _ = _c2(monkeypatch, _row(process, steps_plan=plan), _report(steps={}))
    linkage = out["components"]["boundary_linkage"]
    assert (linkage["unlinked_n"], linkage["linked"], linkage["horizon"]) == (20, 0, {"r0": [0, 19]})
    assert linkage["unlinked"] == [{"step": s, "request": "r0"} for s in range(16)]


FAILURES = [{"step": 0, "value": "a"}, {"step": 1, "value": "b"}, {"step": 1, "value": "c"}]


@pytest.mark.parametrize(("report", "errors", "why", "line"), [
    ({"ok": False, "tally": {"gemm": [4, 5], SERVED_IDS: [3, 3]}, "failures": FAILURES}, [], "see mismatch/errors",
     (f"FAIL 7/8 values exact over steps [0, 1] (unrun []), seed challenge @; linkage 3/3, committed chain {CHAIN}; source stats {{'slots': 1}}; "
      f"first failures {FAILURES[:2]} ")),
    ({}, ["a slot was read stale"], "see mismatch/errors", None),
    ({"error": "the Program names a value the run did not commit"}, [], "the Program names a value the run did not commit", None),
    ({"ok": None}, [], "see mismatch/errors", None),
])
def test_a_report_that_is_not_ok_an_error_in_it_or_an_error_of_the_source_fails_the_replay_and_c2(monkeypatch, process, report, errors, why, line):
    row = _row(process)
    row.src.errors = errors
    report = _report(**report)
    out, _ = _c2(monkeypatch, row, report)
    replay, check = out["components"]["sampled_replay"], out["value_correspondence"]["compiled_check"]
    n, exact = (sum(t[i] for t in report["tally"].values()) for i in (1, 0))
    assert (replay["result"], check["ok"], out["ok"], out["components"]["boundary_linkage"]["result"]) == (False, False, False, True)
    assert (replay["why"], replay["errors"], replay["mismatch"], check["error"]) == (why, errors, report["failures"], report.get("error"))
    assert (replay["n"], replay["exact"]) == (n, exact) == (check["checked"], check["equal"]) == (check["launches_checked"], check["launches_exact"])
    assert out["value_correspondence"]["n"] == n
    assert out["line"].startswith(f"FAIL {exact}/{n} ") and out["line"].endswith(f" {report.get('error') or ''}")
    assert line is None or out["line"] == line.replace("@", str(replay["seed"]))


def test_a_check_that_raises_is_recorded_with_its_error_and_its_traceback_and_fails_c2(monkeypatch, process):
    out, asked = _c2(monkeypatch, _row(process), ValueError("no such value"))
    replay, linkage = out["components"]["sampled_replay"], out["components"]["boundary_linkage"]
    check = out["value_correspondence"]["compiled_check"]
    report = asked["population"][1]
    assert report["traceback"].rstrip().endswith("ValueError: no such value") and "in check_committed" in report["traceback"]
    assert report == {"ok": False, "error": "ValueError('no such value')", "traceback": report["traceback"], "tally": {}, "steps": {},
                      "failures": []}
    assert (replay["result"], replay["why"], replay["n"], replay["exact"], replay["tally"]) == (False, "ValueError('no such value')", 0, 0, {})
    assert (replay["steps_checked"], replay["steps_unrun"], replay["mismatch"]) == (None, None, [])
    assert (replay["sample"]["rows_per_request"], replay["sample"]["picks"]) == (None, None)
    assert check == {"ok": False, "checked": 0, "equal": 0, "launches_checked": 0, "launches_exact": 0, "error": "ValueError('no such value')",
                     "seed": replay["seed"], "tally": {}, "failures": [], "steps_checked": None, "steps_unrun": None}
    assert (linkage["result"], linkage["unlinked_n"], linkage["prompt_linked"]) == (False, 3, {"r0": False, "r1": False}) and out["ok"] is False
    assert out["line"] == (f"FAIL 0/0 values exact over steps None (unrun None), seed challenge {replay['seed']}; linkage 0/3, committed chain "
                           f"{CHAIN}; source stats {{'slots': 1}}; first failures [] ValueError('no such value')\n{report['traceback']}")


def test_a_c2_before_the_engine_was_read_asks_no_check_and_fails_with_that_said(monkeypatch, process):
    out, asked = _c2(monkeypatch, _row(process), _report(), weight=None)
    error = """UsageError("--compiled-taps served: a run's C2 before `engine` read the served weights")"""
    assert "check" not in asked and asked["population"][1]["error"] == error
    assert (out["ok"], out["components"]["sampled_replay"]["why"], out["value_correspondence"]["compiled_check"]["error"]) == (False, error, error)


def test_a_long_error_is_recorded_by_its_first_300_characters_and_the_tracebacks_last_1200(monkeypatch, process):
    out, asked = _c2(monkeypatch, _row(process), ValueError("x" * 2000))
    report = asked["population"][1]
    assert report["error"] == ("ValueError('" + "x" * 2000)[:300] == out["components"]["sampled_replay"]["why"]
    assert len(report["traceback"]) == 1200 and report["traceback"].endswith("x" * 1000 + "\n")
