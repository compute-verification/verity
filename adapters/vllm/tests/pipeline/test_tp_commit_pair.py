"""A pair of `tp-commit` (`verity_vllm.pipeline.tp.commit`) whose ranks all finalized: the TP run root the driver makes of the ranks'
roots, the positions it has every rank open under the pair's coins, each rank's verdict with its binding and its population
openings, and what the pair leaves on disk (`finalize_pair<N>.json`, `declaration_rank<r>.json`, `binding_map_rank<r>.json`,
`coverage.json` and the run's row).  With it, what a rank of `verity_vllm.pipeline.tp.rank_worker` answers to the three calls
the driver sends it after its root: `tp2_commit_open`, `tp2_commit_open_population` and `tp2_commit_binding_map_dump`.
CPU-only: `tp.commit.main` over the stand-in LLM of `tests.pipeline.test_tp_commit_options`, whose rank calls go to two
`TP2CaptureWorkerExtension`s.  A rank's committer is the library's host committer with no engine under it: it commits the
rank's stream of collective members when the run ends and states, as a committer on an engine does, the tensors it expected
and the identity check of its steps."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.pipeline import test_tp_commit_options as TO
from verity_vllm.commit import binding as BD
from verity_vllm.commit.challenge import challenge_positions
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.engine.profiles import layer_classes as LC
from verity_vllm.pipeline import commit as PC
from verity_vllm.pipeline import commit_coins as CC
from verity_vllm.pipeline.tp import commit as TPCM
from verity_vllm.pipeline.tp import run_root as RT
from verity_vllm.pipeline.tp import warmup as WU
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension

driver = TO.driver

SITES = [("m.proj", LC.ROW_LINEAR), ("m.logits", LC.LOGITS)]
STEPS = 3
#: one pair, four openings a rank, no cross-rank pass (it reads a GPU's properties), and the class the host committer declares
PAIR = ("--pairs", "1", "--openings", "4", "--xrank-per-stratum", "0", "--required-classes", "instance_outputs_native")
RANK_CALLS = ("tp2_commit_install", "tp2_commit_finalize", "tp2_commit_open", "tp2_commit_open_population", "tp2_commit_binding_map_dump",
              "tp2_commit_reset")


def _program(rank: int) -> bytes:
    return hashlib.sha256(b"the Program of rank %d" % rank).digest()


class _Committer:
    """A rank's committer with no engine under it.  `install` starts a run on a new host committer, `finalize` commits the run's
    values (at each step the rank's part of `m.proj` and its shard of `m.logits`) and states that every tensor it expected is
    committed and that each step's identities are the layout's; ``says`` overrides what it states.  Every other read goes to
    the host committer of the run."""

    def __init__(self, rank: int) -> None:
        self.rank, self.runs, self.says = rank, 0, dict[str, Any]()
        self.install()

    def install(self) -> None:
        self.com = NativeHostCommitter(None, run_id=f"rank{self.rank}-run{self.runs}", hash_threads=1, program_digest=_program(self.rank))

    def reset(self) -> None:
        """Nothing is hooked, so nothing is undone."""

    uninstall = reset

    def finalize(self) -> Any:
        rc = self.com.commit_offline([[(f"m.proj/part_rank{self.rank}", bytes([16 * self.rank + step + 1]) * 100),
                                       (f"m.logits/shard_rank{self.rank}", bytes([self.runs + 1]) * 150)] for step in range(STEPS)])
        self.runs += 1
        tensors = sum(s.n_tensors for s in rc.steps)
        rc.extra.update({"expected_tensors": tensors, "committed_tensors": tensors, "identity_check": {"ok": True, "steps_checked": STEPS}} | self.says)
        return rc

    def __getattr__(self, name: str) -> Any:
        return getattr(self.com, name)


def _rank(rank: int) -> TP2CaptureWorkerExtension:
    eng: Any = SimpleNamespace(runner=SimpleNamespace(), model=torch.nn.Linear(2, 2))
    w = TP2CaptureWorkerExtension()
    w._tp2_commit = {"rank": rank, "world": 2, "program_digest": _program(rank).hex(), "com": _Committer(rank),
                     "tps": SimpleNamespace(sites=list(SITES)), "eng": eng, "base": PC.baseline_state(eng), "rc": None}
    return w


class _Pair:
    """Two ranks under `tp-commit`'s stand-in LLM.  ``run(*flags)`` runs the Commit with `PAIR` and ``flags`` to its end -> its exit
    code; ``sent`` is each rank call the driver made, with its kwargs; ``alter(method, rank, change)`` changes what that rank
    answers to that call before the driver reads it."""

    def __init__(self, llm: Any, out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.out, self.ranks = out, [_rank(0), _rank(1)]
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self._changes: dict[tuple[str, int], Callable[[dict[str, Any]], None]] = {}
        self._stand_in = llm.collective_rpc
        llm.to_the_end = True
        monkeypatch.setattr(llm, "collective_rpc", self._rpc)

    def _rpc(self, method: Any, timeout: float | None = None, args: tuple = (), kwargs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if method is WU.warming:
            return [method(w, *args) for w in self.ranks]
        if method not in RANK_CALLS:
            return self._stand_in(method, timeout=timeout, args=args, kwargs=kwargs)
        self.sent.append((method, dict(kwargs or {})))
        answers = [getattr(w, method)(*args, **(kwargs or {})) for w in self.ranks]
        for r, answer in enumerate(answers):
            if (method, r) in self._changes:
                self._changes[method, r](answer)
        return answers

    def alter(self, method: str, rank: int, change: Callable[[dict[str, Any]], None]) -> None:
        self._changes[method, rank] = change

    def run(self, *flags: str) -> int:
        return TPCM.main(TO._options(self.out, PAIR + flags))

    def roots(self) -> dict[int, bytes]:
        """Each rank's root of its last run."""
        return {r: w._tp2_commit["rc"].run_root for r, w in enumerate(self.ranks)}

    def read(self, name: str) -> Any:
        return json.loads((self.out / name).read_text())

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.out / "runs.jsonl").read_text().splitlines()]


@pytest.fixture
def pair(driver, monkeypatch) -> _Pair:
    llm, out = driver
    return _Pair(llm, out, monkeypatch)


def _binding(**fields: Any) -> Callable[[dict[str, Any]], None]:
    """A change of a rank's finalize answer: its binding (a run of record's; a warm-up has none) gets ``fields``."""
    def change(answer: dict[str, Any]) -> None:
        if answer["binding"] is not None:
            answer["binding"] = answer["binding"] | fields
    return change


def test_a_pair_every_rank_finalized_passes_under_the_root_of_the_ranks_roots(pair):
    assert pair.run() == 0
    control, run = pair.rows()
    assert (control["arm"], control.get("commit"), run["arm"], run["tokens_equal_ref"]) == ("control", None, "instrumented", True)
    c, roots = run["commit"], pair.roots()
    assert c["tp_run_root"] == RT.tp_run_root(2, roots).hex() and c["world"] == 2
    assert c["commitment_vector"] == [{"rank": r, "run_root": roots[r].hex(), "leaves": STEPS, "steps": STEPS} for r in (0, 1)]
    assert (c["coins"]["source"], c["coins"]["root"], sorted(c["coins"]["keys"])) == \
        ("in-process", c["tp_run_root"], ["opening-identities", "opening-positions", "replay"])
    assert (c["leaves"], c["bytes_bound"], c["tensors"], c["openings"]) == (6, 1500, 12, {"n": 8, "verified": 8})
    assert (c["commit_pass"], c["commit_fail_reason"], c["serving_rows"]) == (True, "PASS", None)
    assert sorted(c["per_rank"]) == ["0", "1"]
    fin = pair.read("finalize_pair0.json")
    for r, f in enumerate(fin):
        per = c["per_rank"][str(r)]
        assert (f["rank"], f["run_root"], per["run_root"]) == (r, roots[r].hex(), roots[r].hex())
        assert (per["commit_pass"], per["commit_fail_reason"]) == (True, "PASS")
        assert [per[k] for k in ("steps", "leaves", "bytes_bound", "tensors")] == [STEPS, STEPS, 750, 6] == [f[k] for k in ("steps", "leaves", "bytes_bound", "tensors")]
        assert per["extra"] == {k: f[k] for k in ("expected_tensors", "committed_tensors", "errors_n", "errors", "collector_failure", "identity_check",
                                                  "expected_manifest", "bounded_staging")}
        assert (per["extra"]["expected_tensors"], per["extra"]["errors_n"], per["extra"]["identity_check"]["ok"]) == (6, 0, True)
        assert (per["openings"]["n"], per["openings"]["verified"], per["openings"]["error"]) == (4, 4, None)
        assert per["openings"]["bytes_mean"] > 0 and per["openings"]["generate_s"] >= 0 and per["openings"]["verify_s"] >= 0
        assert per["classes"] == f["classes"] and sorted(per["classes"]) == ["instance_outputs_native"]
        assert per["binding"] == f["binding"] and per["binding"]["coverage"]["ok"] is True and "binding_note" not in per
        assert (per["attribution"], per["match_oracle"], per["sampled_replay"], per["boundary_linkage"], per["weights_pin"]) == (None,) * 5
        assert per["cross_rank_collectives"] == {"result": None, "picks": None, "equal": None, "mismatch_n": None, "missing_n": None,
                                                 "why": "not run (--xrank-per-stratum 0)"}
        assert pair.read(f"declaration_rank{r}.json") == f["declaration"]
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"], summary["pairs_run"]) == (True, [True], 1)


def test_every_rank_opens_the_positions_and_the_identities_the_pairs_coins_sample(pair):
    assert pair.run() == 0
    c, roots = pair.rows()[1]["commit"], pair.roots()
    root, claimed = bytes.fromhex(c["tp_run_root"]), {str(r): roots[r].hex() for r in (0, 1)}
    positions_from, identities_from = CC.tp_keys(c["coins"])
    opened = [kw for method, kw in pair.sent if method == "tp2_commit_open"]
    assert opened == [{"positions_by_rank": {str(r): [list(p) for p in challenge_positions(root, [1] * STEPS, 4, salt=f"0/rank{r}".encode(),
                                                                                           source=positions_from)] for r in (0, 1)},
                       "claimed_roots": claimed}]
    assert all(len(p) == 4 for p in opened[0]["positions_by_rank"].values())
    # one call per rank's entry of the commitment vector, each rank verifying against its own
    assert [kw for method, kw in pair.sent if method == "tp2_commit_open_population"] == [
        {"k": 4, "salt": "0/population", "combined_root_hex": root.hex(), "source_hex": identities_from.hex(), "claimed_root_hex": claimed[r]} for r in "01"]
    for r in "01":
        sampled = c["per_rank"][r]["openings_population"]
        assert (sampled["rank"], sampled["claimed_root"], sampled["claimed_root_is_own"], sampled["context"]) == (int(r), claimed[r], True, {"ok": True, "why": "PASS"})
        assert sampled["n"] == sampled["opened"] == sampled["verified"] >= 4 and (sampled["unbound_n"], sampled["unbound"]) == (0, [])
        assert (sampled["late_step_in_sample"], sampled["families"]) == (True, ["logits", "proj"])
        assert sampled["map_digest"] == c["per_rank"][r]["binding"]["map_digest"] and sampled["open_verify_s"] >= 0
    # the warm-up, then the pair's instrumented run: each installed, finalized and reset; only the pair's is opened and dumped
    assert [method for method, _ in pair.sent] == [
        "tp2_commit_install", "tp2_commit_finalize", "tp2_commit_reset",
        "tp2_commit_install", "tp2_commit_finalize", "tp2_commit_open", "tp2_commit_binding_map_dump", "tp2_commit_open_population",
        "tp2_commit_open_population", "tp2_commit_reset"]
    assert [f["binding"] for f in pair.read("finalize_warmup.json")] == [None, None]


def test_the_pairs_coverage_record_is_the_ranks_bindings_and_population_openings(pair, capsys):
    assert pair.run() == 0
    per_rank, cov = pair.rows()[1]["commit"]["per_rank"], pair.read("coverage.json")
    bindings = {r: per_rank[r]["binding"] for r in "01"}
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"], cov["skipped"], cov["replay_pending_ranks"]) == ("PASS", True, None, None, None)
    assert (cov["world"], cov["ranks_present"], cov["population_source"], cov["required_manifest"], cov["rules_version"]) == \
        (2, [0, 1], "structural", None, "r15-tp04-v1")
    assert cov["checker"].startswith("verity/coverage/binding-r14 (bind BIND-01 verity_vllm.commit.binding, per TP rank;")
    assert cov["manifest_digests_per_rank"] == {r: b["population"]["digest"] for r, b in bindings.items()}
    assert cov["manifest_digest"] == "+".join(bindings[r]["population"]["digest"][:16] for r in "01")
    assert (cov["manifest_rows_per_rank"], cov["structural_coverage_ok_per_rank"]) == (None, None)
    assert cov["population_n_per_rank"] == {"0": 6, "1": 6}
    assert cov["binding_map_digests_per_rank"] == {r: b["map_digest"] for r, b in bindings.items()}
    assert cov["binding_map_digest"] == "+".join(bindings[r]["map_digest"][:16] for r in "01")
    assert cov["program_digests_per_rank"] == {str(r): _program(r).hex() for r in (0, 1)}
    assert cov["coverage_per_rank"] == {r: {"ok": True, "expected_n": 6, "bound_n": 6, "missing_n": 0, "extra_n": 0, "physical_copies_n": 0} for r in "01"}
    sampled = {r: per_rank[r]["openings_population"] for r in "01"}
    assert cov["openings_per_rank"] == sampled and cov["openings_population"] == "expected population"
    assert cov["openings_checked"] == cov["openings_ok"] == sum(d["opened"] for d in sampled.values()) and cov["openings_unbound"] == 0
    assert (cov["late_step_in_sample"], cov["attribution_per_rank"], cov["attribution_reference"], cov["retention"]) == (True, {"0": None, "1": None}, None, "host")
    assert cov["negatives"] == pair.read("versions.json")["negatives"] and cov["negatives"]["wrong_root_rank"] is None
    # each rank's binding map is beside the record, under the name the record gives it
    assert cov["binding_map_files"] == {"0": "binding_map_rank0.json", "1": "binding_map_rank1.json"}
    maps = {r: pair.read(cov["binding_map_files"][r]) for r in "01"}
    assert {r: m["digest"] for r, m in maps.items()} == cov["binding_map_digests_per_rank"] and {m["entries_n"] for m in maps.values()} == {6}
    assert f"[tp2commit] binding maps: {[(r, maps[str(r)]['digest'][:16], 6) for r in (0, 1)]}" in capsys.readouterr().out.splitlines()


def test_a_wrong_root_for_one_rank_in_the_vector_fails_that_ranks_openings(pair):
    assert pair.run("--wrong-root-rank", "1") == 1
    c, roots = pair.rows()[1]["commit"], pair.roots()
    wrong = bytes([roots[1][0] ^ 0xFF]) + roots[1][1:]
    assert [e["run_root"] for e in c["commitment_vector"]] == [roots[0].hex(), wrong.hex()]
    assert c["tp_run_root"] == RT.tp_run_root(2, {0: roots[0], 1: wrong}).hex()
    fin = pair.read("finalize_pair0.json")
    assert (fin[1]["run_root"], fin[1]["run_root_actual"], "run_root_actual" in fin[0]) == (wrong.hex(), roots[1].hex(), False)
    assert [kw["claimed_roots"] for method, kw in pair.sent if method == "tp2_commit_open"] == [{"0": roots[0].hex(), "1": wrong.hex()}]
    honest, cheated = c["per_rank"]["0"], c["per_rank"]["1"]
    assert (honest["commit_pass"], honest["openings"]["verified"], honest["openings_population"]["claimed_root_is_own"]) == (True, 4, True)
    assert (cheated["openings"]["n"], cheated["openings"]["verified"], c["openings"]) == (4, 0, {"n": 8, "verified": 4})
    sampled = cheated["openings_population"]
    assert (sampled["claimed_root"], sampled["claimed_root_is_own"], sampled["verified"], sampled["context"]["ok"]) == (wrong.hex(), False, 0, False)
    assert cheated["commit_pass"] is False
    assert cheated["commit_fail_reason"] == f"population openings: 0/{sampled['opened']} verified, 0 unbound (None)"
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, f"rank 1: openings verified 0 != checked 4; rank 1: {cheated['commit_fail_reason']}")
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"]) == ("FAIL", False, c["commit_fail_reason"])
    assert cov["negatives"]["wrong_root_rank"] == 1 and pair.read("summary.json")["pass"] is False


def test_a_rank_whose_openings_verify_under_a_root_that_is_not_its_own_is_named(pair):
    """The vector's wrong root is caught even where the rank answers that every opening verified."""
    pair.alter("tp2_commit_open", 1, lambda answer: answer.update(verified=answer["n"]))
    pair.alter("tp2_commit_open_population", 1, lambda answer: answer.update(verified=answer["opened"], context={"ok": True, "why": "PASS"}))
    assert pair.run("--wrong-root-rank", "1") == 1
    c = pair.rows()[1]["commit"]
    wrong = c["commitment_vector"][1]["run_root"]
    assert c["commit_fail_reason"] == f"rank 1: claimed run root for rank 1 {wrong[:16]} != the rank's root (openings verified 4/4 against the claim)"
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["1"]["commit_pass"]) == (True, False)


@pytest.mark.parametrize("says,why", [
    ({"committed_tensors": 5}, "expected_tensors 6 != committed_tensors 5 (refused/lost acquisitions -> incomplete root)"),
    ({"errors_n": 1, "errors": ["a step's hook raised"]}, "1 collector error(s): a step's hook raised"),
    ({"identity_check": {"ok": False, "violations_n": 1, "violations": [{"step": 2}]}}, "identity check: 1 step(s) differ from the expected manifest (first: step 2 "),
], ids=["a-tensor-short", "a-collector-error", "a-step-not-the-layouts"])
def test_what_one_ranks_committer_states_fails_that_rank_alone(pair, says, why):
    pair.ranks[1]._tp2_commit["com"].says = says
    assert pair.run() == 1
    c = pair.rows()[1]["commit"]
    assert (c["per_rank"]["0"]["commit_pass"], c["per_rank"]["1"]["commit_pass"], c["commit_pass"]) == (True, False, False)
    assert c["per_rank"]["1"]["commit_fail_reason"].startswith(why) and c["commit_fail_reason"] == f"rank 1: {c['per_rank']['1']['commit_fail_reason']}"
    assert {k: c["per_rank"]["1"]["extra"][k] for k in says} == says
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"], cov["coverage_per_rank"]["1"]["ok"]) == ("FAIL", False, c["commit_fail_reason"], True)


@pytest.mark.parametrize("fields,why,note", [
    ({"error": "no layout on this rank"}, "binding map / coverage error: no layout on this rank", "no layout on this rank"),
    ({"coverage": {"ok": False, "missing_n": 2, "expected_n": 6, "missing": [["m.proj", 2]]}},
     "identity coverage: 2 of 6 required identities have no binding (first: ['m.proj', 2])", None),
    ({"context": {"ok": False, "why": "the map is of another run"}}, "binding map context: the map is of another run", None),
    ({"coverage": {"ok": True, "error": "two identities share a leaf"}}, "identity coverage: two identities share a leaf", None),
], ids=["no-map", "identities-missing", "another-runs-map", "coverage-error"])
def test_a_ranks_binding_gates_its_verdict(pair, fields, why, note):
    pair.alter("tp2_commit_finalize", 0, _binding(**fields))
    assert pair.run() == 1
    c = pair.rows()[1]["commit"]
    failed = c["per_rank"]["0"]
    assert (failed["commit_pass"], failed["commit_fail_reason"], failed.get("binding_note")) == (False, why, note)
    assert (c["per_rank"]["1"]["commit_pass"], c["commit_pass"], c["commit_fail_reason"]) == (True, False, f"rank 0: {why}")
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"]) == ("FAIL", False, f"rank 0: {why}")


def test_a_binding_no_rank_ran_is_not_a_coverage_verdict_and_fails_no_rank(pair, capsys):
    """Without a required manifest a rank that skipped its binding still passes on its openings; the coverage record says it
    was not run, and no rank opens a population it has no map for."""
    for w in pair.ranks:
        del w._tp2_commit["program_digest"]
    assert pair.run() == 0
    why = "no program_digest for this rank (pass --build-rank<r> or --required-manifest to the driver)"
    c, cov = pair.rows()[1]["commit"], pair.read("coverage.json")
    assert [(c["per_rank"][r]["commit_pass"], c["per_rank"][r]["binding_note"]) for r in "01"] == [(True, why)] * 2
    assert all("openings_population" not in c["per_rank"][r] for r in "01") and "tp2_commit_open_population" not in [method for method, _ in pair.sent]
    assert (cov["verdict"], cov["identity_coverage"], cov["skipped"]) == ("NOT_RUN", None, [why] * 2)
    assert (cov["openings_population"], cov["openings_checked"], cov["openings_ok"], cov["openings_per_rank"], cov["late_step_in_sample"]) == (None, 0, 0, {}, None)
    assert cov["coverage_per_rank"]["0"] == {"ok": None, "expected_n": None, "bound_n": None, "missing_n": None, "extra_n": None, "physical_copies_n": None}
    assert (cov["binding_map_digest"], cov["manifest_digest"], cov["population_n_per_rank"]) == ("None+None", "None+None", {"0": None, "1": None})
    assert "[tp2commit] binding maps: [(0, 'no binding map f', None), (1, 'no binding map f', None)]" in capsys.readouterr().out.splitlines()


def test_one_rank_without_a_map_keeps_every_rank_from_the_population_openings(pair):
    """As the driver is: the Commit passes, and its coverage record says FAIL with no reason, since one rank has no coverage."""
    del pair.ranks[1]._tp2_commit["program_digest"]
    assert pair.run() == 0
    c, cov = pair.rows()[1]["commit"], pair.read("coverage.json")
    assert (c["commit_pass"], c["commit_fail_reason"]) == (True, "PASS") and all("openings_population" not in c["per_rank"][r] for r in "01")
    assert (cov["verdict"], cov["identity_coverage"], cov["reason"], cov["openings_population"]) == ("FAIL", False, None, None)
    assert cov["skipped"] == ["no program_digest for this rank (pass --build-rank<r> or --required-manifest to the driver)"]
    assert {r: cov["coverage_per_rank"][r]["ok"] for r in "01"} == {"0": True, "1": None}


@pytest.mark.parametrize("answer,why", [
    ({"unbound_n": 1}, "population openings: {opened}/{opened} verified, 1 unbound (None)"),
    ({"verified": 0}, "population openings: 0/{opened} verified, 0 unbound (None)"),
    ({"opened": 0, "verified": 0, "error": "no population / binding map on this rank"},
     "population openings: 0/0 verified, 0 unbound (no population / binding map on this rank)"),
    ({"context": {"ok": False, "why": "the map names another root"}}, "binding map context vs claimed root: the map names another root"),
], ids=["an-identity-unbound", "none-verified", "none-opened", "another-roots-map"])
def test_a_ranks_population_openings_gate_its_verdict(pair, answer, why):
    opened: list[int] = []

    def change(sampled: dict[str, Any]) -> None:
        opened.append(sampled["opened"])
        sampled.update(answer)

    pair.alter("tp2_commit_open_population", 1, change)
    assert pair.run() == 1
    c = pair.rows()[1]["commit"]
    why = why.format(opened=opened[0])
    assert (c["per_rank"]["1"]["commit_pass"], c["per_rank"]["1"]["commit_fail_reason"]) == (False, why)
    assert (c["per_rank"]["0"]["commit_pass"], c["commit_pass"], c["commit_fail_reason"]) == (True, False, f"rank 1: {why}")
    cov = pair.read("coverage.json")
    assert (cov["verdict"], cov["reason"], cov["openings_unbound"]) == ("FAIL", f"rank 1: {why}", answer.get("unbound_n", 0))


def test_tokens_that_differ_from_the_controls_fail_every_rank(pair, monkeypatch):
    runs = iter([TO.RUN, TO.RUN, TO.RUN | {"tokens": {"q0": [5, 6, 7, 9]}}])       # the warm-up, the control, the instrumented run
    monkeypatch.setattr(PC, "run_workload", lambda eng, requests, sampling: dict(next(runs)))
    assert pair.run() == 1
    run = pair.rows()[1]
    assert run["tokens_equal_ref"] is False
    assert run["commit"]["commit_fail_reason"] == "rank 0: control vs instrumented tokens differ; rank 1: control vs instrumented tokens differ"


def test_a_second_pair_has_its_own_root_and_coins_and_a_coverage_record_only_when_it_fails(pair):
    pair.alter("tp2_commit_open", 1, lambda answer: answer.update(verified=3) if pair.ranks[1]._tp2_commit["com"].runs == 3 else None)
    assert pair.run("--pairs", "2") == 1
    first, second = (row["commit"] for row in pair.rows() if row["arm"] == "instrumented")
    assert [(row["pair"], row["arm"]) for row in pair.rows()] == [(0, "control"), (0, "instrumented"), (1, "instrumented")]
    assert first["tp_run_root"] != second["tp_run_root"] and first["coins"]["keys"] != second["coins"]["keys"]
    assert second["tp_run_root"] == RT.tp_run_root(2, pair.roots()).hex() == second["coins"]["root"]
    assert [f["run_root"] for f in pair.read("finalize_pair1.json")] == [root.hex() for root in pair.roots().values()]
    assert (first["commit_pass"], second["commit_pass"], second["commit_fail_reason"]) == (True, False, "rank 1: openings verified 3 != checked 4")
    assert pair.read("coverage.json")["verdict"] == "PASS"
    cov = pair.read("coverage_pair1.json")
    assert (cov["verdict"], cov["reason"]) == ("FAIL", "rank 1: openings verified 3 != checked 4")
    # the positions and the identities of the second pair are sampled under its own salt; the declarations and the maps are the first pair's
    positions_from, identities_from = CC.tp_keys(second["coins"])
    opened = [kw for method, kw in pair.sent if method == "tp2_commit_open"]
    assert opened[1]["positions_by_rank"]["1"] == [list(p) for p in challenge_positions(bytes.fromhex(second["tp_run_root"]), [1] * STEPS, 4,
                                                                                         salt=b"1/rank1", source=positions_from)]
    assert [(kw["salt"], kw["source_hex"]) for method, kw in pair.sent if method == "tp2_commit_open_population"][2:] == [("1/population", identities_from.hex())] * 2
    assert [method for method, _ in pair.sent].count("tp2_commit_binding_map_dump") == 1
    assert sorted(p.name for p in pair.out.glob("coverage*.json")) == ["coverage.json", "coverage_pair1.json"]


def test_a_second_pair_that_passes_writes_no_coverage_record_of_its_own(pair):
    assert pair.run("--pairs", "2") == 0
    commits = [row["commit"] for row in pair.rows() if row["arm"] == "instrumented"]
    assert [(c["commit_pass"], c["commit_fail_reason"], c["openings"]) for c in commits] == [(True, "PASS", {"n": 8, "verified": 8})] * 2
    assert commits[0]["tp_run_root"] != commits[1]["tp_run_root"]
    assert sorted(p.name for p in pair.out.glob("coverage*.json")) == ["coverage.json"]
    maps = [{r: c["per_rank"][r]["binding"]["map_digest"] for r in "01"} for c in commits]
    assert pair.read("coverage.json")["binding_map_digests_per_rank"] == maps[0] != maps[1]            # the record is the first pair's
    assert sorted(p.name for p in pair.out.glob("finalize_pair*.json")) == ["finalize_pair0.json", "finalize_pair1.json"]
    summary = pair.read("summary.json")
    assert (summary["pass"], summary["commit_pass"], summary["pairs_run"]) == (True, [True, True], 2)


def _finalized(rank: int = 1) -> tuple[TP2CaptureWorkerExtension, Any]:
    """(a rank after its run of record, the run's commitment)"""
    w = _rank(rank)
    w.tp2_commit_install()
    w.tp2_commit_finalize()
    return w, w._tp2_commit["rc"]


def test_a_rank_opens_its_positions_against_the_root_the_driver_claims():
    w, rc = _finalized()
    positions = {"1": [[0, 0], [2, 0]], "0": [[1, 0]]}
    own = w.tp2_commit_open(positions_by_rank=positions)
    assert (own["rank"], own["n"], own["verified"], own["claimed_root"], own["claimed_root_is_own"]) == (1, 2, 2, rc.run_root.hex(), True)
    assert own["bytes_mean"] > 250 and own["generate_s"] >= 0 and own["verify_s"] >= 0
    assert w.tp2_commit_open(positions_by_rank=positions, claimed_roots={"1": rc.run_root.hex(), "0": "00" * 64})["claimed_root_is_own"] is True
    other = w.tp2_commit_open(positions_by_rank=positions, claimed_roots={"1": "00" * 64})
    assert (other["n"], other["verified"], other["claimed_root"], other["claimed_root_is_own"]) == (2, 0, "00" * 64, False)
    assert rc.run_root != bytes(64)                                  # the rank's own commitment is not rewritten
    assert w.tp2_commit_open(positions_by_rank={"0": [[0, 0]]}) == {"rank": 1, "n": 0, "verified": 0, "error": "no positions for this rank"}
    w.tp2_commit_install()
    assert w.tp2_commit_open(positions_by_rank=positions) == {"rank": 1, "n": 0, "verified": 0, "error": "no run commitment on this rank"}


def test_a_rank_opens_identities_of_its_population_sampled_under_the_combined_root():
    w, rc = _finalized()
    st = w._tp2_commit
    opened: list[list[tuple[int, int]]] = []

    def open_(positions: list[tuple[int, int]]) -> Any:
        opened.append(list(positions))
        return st["com"].com.open(positions)

    st["com"].open = open_
    asked: dict[str, Any] = {"k": 16, "salt": "0/population", "combined_root_hex": "ab" * 64, "source_hex": "cd" * 64}
    own = w.tp2_commit_open_population(**asked)
    sampled = BD.challenge_identities(st["pop"], st["bmap"], bytes.fromhex("ab" * 64), 16, salt=b"0/population", source=bytes.fromhex("cd" * 64))
    assert opened == [[(identity[2], leaf) for identity, _, leaf in sampled]] and own["n"] == len(sampled) == 16 + 1 + 2
    assert (own["rank"], own["claimed_root"], own["claimed_root_is_own"], own["context"]) == (1, rc.run_root.hex(), True, {"ok": True, "why": "PASS"})
    assert own["n"] == own["opened"] == own["verified"] and (own["unbound_n"], own["unbound"]) == (0, [])
    assert (own["late_step_in_sample"], own["families"], own["map_digest"]) == (True, ["logits", "proj"], st["bmap"]["digest"])
    again = w.tp2_commit_open_population(**asked)
    assert {k: v for k, v in again.items() if k != "open_verify_s"} == {k: v for k, v in own.items() if k != "open_verify_s"}
    other = w.tp2_commit_open_population(**asked, claimed_root_hex="00" * 64)
    assert (other["opened"], other["verified"], other["claimed_root"], other["claimed_root_is_own"]) == (own["opened"], 0, "00" * 64, False)
    assert other["context"]["ok"] is False and other["context"]["why"]
    # an identity of the population with no entry in the map is counted, not opened
    held = dict(st["bmap"])
    st["bmap"] = dict(held, entries=[])
    unbound = w.tp2_commit_open_population(**asked)
    assert (unbound["n"], unbound["unbound_n"], unbound["opened"], unbound["verified"], len(unbound["unbound"])) == (own["n"], own["n"], 0, 0, 5)
    assert {u["output_member"] for u in unbound["unbound"]} <= {"part_rank1", "shard_rank1"} and {u["rank"] for u in unbound["unbound"]} == {1}
    for lacks in ("pop", "bmap", "rc"):
        st.update(bmap=held, pop=st["pop"], rc=rc)
        kept, st[lacks] = st[lacks], None
        assert w.tp2_commit_open_population(**asked) == {"rank": 1, "n": 0, "verified": 0, "error": "no population / binding map on this rank"}
        st[lacks] = kept


def test_a_rank_writes_its_binding_map_beside_the_drivers_evidence(tmp_path):
    w, _ = _finalized()
    st = w._tp2_commit
    wrote = w.tp2_commit_binding_map_dump(out_dir=str(tmp_path / "commit"))
    assert wrote == {"rank": 1, "path": str(tmp_path / "commit" / "binding_map_rank1.json"), "digest": st["bmap"]["digest"], "entries_n": 6}
    assert json.loads(Path(wrote["path"]).read_text()) == json.loads(json.dumps(st["bmap"], default=str))
    st["bmap"] = None
    assert w.tp2_commit_binding_map_dump(out_dir=str(tmp_path / "none")) == {"rank": 1, "skipped": "no binding map for this run"}
    assert not (tmp_path / "none").exists()
