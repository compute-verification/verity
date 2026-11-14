"""What `tp-commit` (`verity_vllm.pipeline.tp.commit`) does with its options before its first measured run: the communicator size it
builds its engine with, the negatives and the replay config every rank's committer is made with, the environment it puts on the
record, and the refusals that end it before a rank commits: a deferred replay with no manifest, two spellings of the ranks' Builds,
a manifest whose rank Program no Build vouches for, names another Program or is of another world, a flip with no index and another
rank's shard with no checkpoint.  With them, the rank's Program digest as `build_program_digest` reads it from a Build dir.
CPU-only: `tp.commit.main` over a stand-in LLM of two ranks, which stops the Commit where it makes the ranks' committers."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import fake_vllm
from tests.pipeline import served_ranks
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import commit as TPCM

# imported before any stand-in is installed: the module arms its switches on the vLLM it finds at import
from verity_vllm.pipeline.tp import rank_worker
from verity_vllm.query.manifest.format import components_fold_digest

pytest.importorskip("torch")

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4},
                         {"request_id": "q1", "prompt_token_ids": [4, 5], "arrive_step": 1, "seed": 7}],
            "sampling": {"max_tokens": 2, "seed": 3}, "sha256": "w", "execution": "bi-eager"}
RUN = {"wall_s": 0.5, "iterations": 4, "step_wall_s": [0.2, 0.1, 0.1, 0.1], "step_tokens_out": [1, 1, 1, 1], "generated_tokens": 4, "prompt_tokens": 3,
       "tokens": {"q0": [5, 6, 7, 8]}}
A, B, C = "a" * 128, "b" * 128, "c" * 128


class _Made(Exception):
    """The ranks' committers were asked for: every option the Commit reads before its first run is read by then."""


class _LLM:
    """`vllm.LLM` over two ranks: it keeps the kwargs it was built with and what every rank's committer is made with, where it
    stops the Commit unless ``to_the_end``; every other rpc answers one dict per rank, a rank that committed and passed."""

    def __init__(self) -> None:
        self.kw: dict[str, Any] = {}
        self.made: dict[str, Any] = {}
        self.to_the_end = False

    def __call__(self, **kw: Any) -> _LLM:
        self.kw = kw
        return self

    def collective_rpc(self, method: Any, timeout: float | None = None, args: tuple = (), kwargs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if method == "tp2_commit_make":
            self.made = dict(kwargs or {})
            if not self.to_the_end:
                raise _Made
        if (served := served_ranks.answer(method, WORKLOAD)) is not None:
            return served
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "verdict": "PASS", "cpu_s": {}} for r in (0, 1)]


@pytest.fixture
def driver(tmp_path, monkeypatch) -> tuple[_LLM, Path]:
    """`tp-commit`'s engine and what it reads around it, stood in -> (the stand-in LLM, the Commit's out dir).  Inductor's config is
    unloaded for the test: another test in the same worker may have imported it before the engine's pin, which
    `engine.env.require_inductor` refuses."""
    from verity_vllm import protocols as PO
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL

    llm = _LLM()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    for name in ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS"):
        monkeypatch.delenv(name, raising=False)
    fake_vllm.install(monkeypatch).set("vllm:LLM", llm)
    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
    for owner, name, value in [(prof, "apply_env", lambda: {}), (prof, "checkpoint_for", lambda case, path: {"repo": "org/model"}),
                               (prof, "engine_kwargs", lambda cp, **kw: dict(kw)), (va, "load_workload", lambda path, max_tokens: WORKLOAD),
                               (va, "target_of_workload", lambda wl, *a: None), (va, "case_of", lambda wl, case: case),
                               (va, "throwaway_requests", lambda eng, reqs, seed=0: reqs),
                               (PO, "commit_guard", lambda *a, **k: None), (PC, "git_head", lambda: None),
                               (PC, "run_workload", lambda eng, requests, sampling: dict(RUN)), (TPCM, "tree_of_record", lambda: {}),
                               (GL, "_ENVIRON", []), (GL, "_HELD", [])]:
        monkeypatch.setattr(owner, name, value)
    return llm, tmp_path / "commit"


def _options(out: Path, flags: tuple[str, ...]) -> TPCM.Options:
    return cli.parse("tp-commit", ["--workload", str(out.parent / "w.json"), "--out", str(out), "--case", "C", "--warmup", "0", *flags], environ={})


@pytest.fixture
def commit(driver):
    """``commit(*flags)`` runs ``tp-commit --workload W --out <tmp>/commit --case C --warmup 0 FLAGS`` to where it makes the ranks'
    committers -> (the stand-in LLM, the Commit's out dir)."""
    llm, out = driver

    def run(*flags: str) -> tuple[_LLM, Path]:
        with pytest.raises(_Made):
            TPCM.main(_options(out, flags))
        return llm, out

    return run


@pytest.fixture
def whole_commit(driver):
    """``whole_commit(*flags)`` runs the same Commit to its end -> (its exit code, its out dir)."""
    llm, out = driver

    def run(*flags: str) -> tuple[int, Path]:
        llm.to_the_end = True
        return TPCM.main(_options(out, flags)), out

    return run


def _versions(out: Path) -> dict[str, Any]:
    return json.loads((out / "versions.json").read_text())


def _build(root: Path, rank: int, digest: str) -> str:
    """A rank's Build dir of a one-request row: `result.json` names its request Program."""
    d = root / f"build_rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"program": {"digest": digest}}))
    return str(d)


def _manifest(root: Path, per_rank: dict[str, str], ranks: int = 2) -> str:
    """A merged TP manifest of ``ranks`` ranks naming each rank's Program; not under the query of record, so it binds no partition."""
    p = root / "manifest.json"
    p.write_text(json.dumps({"program_digest_per_rank": per_rank, "ranks": ranks, "query": {"query_id": "a query off the record"},
                             "manifest_digest": "m" * 64}))
    return str(p)


def test_world_is_the_communicator_size_whatever_tp_says(commit):
    llm, out = commit("--tp", "4", "--world", "2")
    assert llm.kw["tensor_parallel_size"] == 2 and llm.kw["max_num_seqs"] == 8 and llm.kw["enable_prefix_caching"] is False
    assert llm.kw["worker_extension_cls"] == rank_worker.EXTENSION      # every rank's Worker is made with the TP worker's methods
    assert _versions(out)["tp"] == 2


@pytest.mark.parametrize("flags,relabel,swap", [
    ((), None, None),
    (("--relabel-rank", "0:1"), {"0": 1}, None),
    (("--swap-requests", "1:q0:q1"), None, {"1": ["q0", "q1"]}),
    (("--relabel-rank", "1:0", "--swap-requests", "0:q1:q0:2-5"), {"1": 0}, {"0": ["q1", "q0", 2, 5]}),
], ids=["none", "relabel", "swap", "relabel and a swap on steps 2 to 5"])
def test_every_rank_s_committer_is_made_with_the_negatives_as_parsed(commit, flags, relabel, swap):
    llm, out = commit(*flags)
    assert (llm.made["tp_relabel"], llm.made["tp_swap_requests"]) == (relabel, swap)
    said = dict(zip(flags[::2], flags[1::2]))
    negatives = _versions(out)["negatives"]
    assert (negatives["relabel_rank"], negatives["swap_requests"]) == (said.get("--relabel-rank"), said.get("--swap-requests"))
    assert (llm.made["request_order"], llm.made["request_ids"]) == (["q0", "q1"], {"q0": "r0", "q1": "r1"})
    assert (llm.made["arrivals"], llm.made["n_requests"]) == ({"r0": 0, "r1": 1}, 2)
    limits = [v for k, v in llm.made.items() if k not in ("arrivals", "request_ids") and isinstance(v, dict) and set(v) == {"r0", "r1"}]
    assert limits == [{"r0": 4, "r1": 2}]                # each request's token limit by its id: its own, else the sampling's
    assert llm.made["replay"] is None and llm.made["required_manifest"] is None and llm.made["program_digest"] == {"0": None, "1": None}


def test_a_flip_of_one_weight_reaches_the_ranks_parsed_and_one_with_no_index_is_refused(commit):
    llm, out = commit("--flip-weight", "1:layers.0.w:3")
    assert llm.made["flip_weight"] == {"rank": 1, "param": "layers.0.w", "index": 3, "value": 1.0}
    assert _versions(out)["negatives"]["flip_weight"] == llm.made["flip_weight"]
    llm, _ = commit("--flip-weight", "0:w:5:-2.5")
    assert llm.made["flip_weight"] == {"rank": 0, "param": "w", "index": 5, "value": -2.5}
    with pytest.raises(SystemExit, match=r"--flip-weight <rank>:<param>:<index>\[:<value>\]"):
        commit("--flip-weight", "0:w")


def test_another_rank_s_shard_needs_the_checkpoint_it_is_composed_from(commit):
    with pytest.raises(SystemExit, match="--serve-shard-from needs --checkpoint-dir"):
        commit("--serve-shard-from", "0:w:1")


def test_the_record_holds_the_environment_of_the_engine_s_families_and_no_secret(commit, monkeypatch):
    """``VLLM_BATCH_INVARIANT`` is the workload's execution (bi-eager), which the Commit pinned before its engine."""
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NATIVE_COLLECT_TOKEN", "a secret")
    monkeypatch.setenv("VERITY_OTHER", "not of these families")
    env = _versions(commit()[1])["env"]
    assert {k: env.get(k) for k in ("VLLM_BATCH_INVARIANT", "NCCL_DEBUG", "NATIVE_COLLECT_TOKEN", "VERITY_OTHER")} == {
        "VLLM_BATCH_INVARIANT": "1", "NCCL_DEBUG": "INFO", "NATIVE_COLLECT_TOKEN": "<redacted>", "VERITY_OTHER": None}
    assert all(k.startswith(("NCCL_", "VLLM_", "CUBLAS", "NATIVE_COLLECT")) for k in env)


@pytest.mark.parametrize("flags", [("--replay-deferred",), ("--replay-deferred", "--required-manifest", "m.json", "--no-sampled-replay")],
                         ids=["no manifest", "no sampled replay"])
def test_a_deferred_replay_with_no_replay_to_defer_is_refused_before_the_engine(commit, flags):
    with pytest.raises(SystemExit, match="--replay-deferred defers the ranks' sampled replay: it needs --required-manifest and no --no-sampled-replay"):
        commit(*flags)


def test_the_ranks_builds_take_one_spelling_and_one_dir_per_rank(commit, tmp_path):
    d0, d1 = _build(tmp_path, 0, A), _build(tmp_path, 1, B)
    for flags in (("--build-rank0", d0, "--build-rank1", d1), ("--build-rank", d0, "--build-rank", d1)):
        llm, out = commit(*flags)
        assert llm.made["program_digest"] == {"0": A, "1": B}
        versions = _versions(out)
        assert versions["build_program_digest"] == {"0": A, "1": B}
        assert versions["build_program_digest_kind"] == {"0": "request Program", "1": "request Program"}
    with pytest.raises(SystemExit, match=r"--build-rank \(rank order\) and --build-rank0/1 are alternative spellings; give one"):
        commit("--build-rank", d0, "--build-rank", d1, "--build-rank0", d0)
    with pytest.raises(SystemExit, match=r"--build-rank given 1 time\(s\) for a world of 2: one Build dir per rank, in rank order"):
        commit("--build-rank", d0)


def test_a_manifest_is_refused_unless_every_rank_s_build_names_its_program_and_its_world_is_the_engine_s(commit, tmp_path):
    d0, d1 = _build(tmp_path, 0, A), _build(tmp_path, 1, B)
    builds = ("--build-rank0", d0, "--build-rank1", d1)
    manifest = _manifest(tmp_path, {"0": A, "1": B})
    with pytest.raises(SystemExit) as no_build:
        commit("--required-manifest", manifest, "--build-rank0", d0)
    assert str(no_build.value) == ("fail-closed: --required-manifest needs --build-rank1 (this rank's Build result.json) so the manifest's Program "
                                   "for rank 1 (bbbbbbbbbbbbbbbb) is checked against the Build, not against itself")
    (Path(d1) / "result.json").write_text("{not json")
    with pytest.raises(SystemExit, match="fail-closed: --required-manifest needs --build-rank1 "):       # an unreadable Build vouches for nothing
        commit("--required-manifest", manifest, *builds)
    _build(tmp_path, 1, C)
    with pytest.raises(SystemExit) as other:
        commit("--required-manifest", manifest, *builds)
    assert str(other.value) == "fail-closed: --build-rank1 Program cccccccccccccccc != manifest Program for rank 1 bbbbbbbbbbbbbbbb"
    _build(tmp_path, 1, B)
    with pytest.raises(SystemExit) as other_world:
        commit("--required-manifest", _manifest(tmp_path, {"0": A, "1": B}, ranks=4), *builds)
    assert str(other_world.value) == "fail-closed: manifest ranks=4 != tp=2"


def test_under_a_manifest_every_rank_is_made_with_its_programs_of_record_and_the_replay_s_config(commit, tmp_path, monkeypatch):
    """Rank 0's Build is a one-request row's; rank 1's composed a workload Program the manifest names, so its components are of
    record with it.  The ring's slots are the committers' reading of the staging variables."""
    d0, d1 = _build(tmp_path, 0, A), _build(tmp_path, 1, B)
    composed = Path(d1) / "build_workload"
    composed.mkdir()
    (composed / "workload_program.json").write_text(json.dumps({"workload_digest": C, "requests": [
        {"component": {"specialization": f'Request{{program_digest="{A}"}}'}}, {"component": {"specialization": f'Request{{program_digest=\\"{B}\\"}}'}},
        {"component": None}]}))
    manifest = _manifest(tmp_path, {"0": A, "1": C})
    monkeypatch.setenv("VERITY_WINDOW_MB", "64")
    monkeypatch.setenv("VERITY_WINDOW_SLOTS", "3")
    llm, out = commit("--required-manifest", manifest, "--build-rank0", d0, "--build-rank1", d1, "--replay-k", "5", "--replay-ranks-at-once", "1",
                      "--replay-target", "layers.0@2")
    assert llm.made["required_manifest"] == manifest and llm.made["program_digest"] == {"0": A, "1": C}
    replay = llm.made["replay"]
    assert replay["programs"] == {"0": d0, "1": d1} and replay["of_record"] == {"0": [A], "1": sorted([A, B, C])}
    assert replay["requests"] == [
        {"request_id": "r0", "arrive_step": 0, "prompt_len": 3, "max_tokens": 4, "prompt_token_ids": [1, 2, 3], "seed": 3},
        {"request_id": "r1", "arrive_step": 1, "prompt_len": 2, "max_tokens": 2, "prompt_token_ids": [4, 5], "seed": 7}]
    assert {k: replay[k] for k in ("per_stratum", "max_evaluate", "seed", "workers", "challenge", "targets", "ring_slots", "k_per_rank",
                                   "ranks_at_once", "slot_dir")} == {
        "per_stratum": 2, "max_evaluate": 2000, "seed": None, "workers": None, "challenge": None, "targets": ["layers.0@2"], "ring_slots": 3,
        "k_per_rank": {"0": 3, "1": 2}, "ranks_at_once": 1, "slot_dir": str(out)}
    versions = _versions(out)
    assert versions["build_program_digest_kind"] == {"0": "request Program", "1": "workload (GP-01)"}
    assert versions["sampled_replay"]["of_record"] == replay["of_record"] and versions["sampled_replay"]["k_per_rank"] == {"0": 3, "1": 2}
    assert versions["weights_pin"]["required"] is True and versions["required_manifest"] == manifest


def test_only_a_composition_the_manifest_names_puts_its_components_of_record(commit, tmp_path, capsys):
    """Each rank's Build dir holds its own workload Program, which the manifest names.  Beside rank 0's is a composition of another
    workload Program, whose components are not of record; beside rank 1's is one nobody can read, which the Commit says and
    goes on with the manifest's Program alone."""
    d0, d1 = _build(tmp_path, 0, A), _build(tmp_path, 1, B)
    for d, own in ((d0, A), (d1, B)):
        (Path(d) / "workload_program.json").write_text(json.dumps({"workload_digest": own}))
        (Path(d) / "build_workload").mkdir()
    (Path(d0) / "build_workload" / "workload_program.json").write_text(json.dumps({"workload_digest": C, "requests": [
        {"component": {"specialization": f'Request{{program_digest="{B}"}}'}}]}))
    (Path(d1) / "build_workload" / "workload_program.json").write_text("not a record")
    llm, _ = commit("--required-manifest", _manifest(tmp_path, {"0": A, "1": B}), "--build-rank0", d0, "--build-rank1", d1)
    assert llm.made["replay"]["of_record"] == {"0": [A], "1": [B]}
    said = [line for line in capsys.readouterr().out.splitlines() if "unreadable" in line]
    assert len(said) == 1 and said[0].startswith("[tp2commit] rank 1: GP-01 workload_program unreadable for the replay's of-record set: ")


@pytest.mark.parametrize("env,slots", [({}, None), ({"VERITY_RETAIN": "host"}, 8), ({"VERITY_RETAIN": "host", "VERITY_WINDOW_MB": "0"}, None),
                                       ({"VERITY_RETAIN": "device", "VERITY_WINDOW_MB": "16", "VERITY_WINDOW_SLOTS": "2"}, 2)],
                         ids=["unset", "retained on the host", "on the host with no window", "on the device with a window of two slots"])
def test_the_replay_s_ring_slots_are_the_staging_variables_as_the_ranks_committers_read_them(commit, tmp_path, monkeypatch, env, slots):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    llm, _ = commit("--required-manifest", _manifest(tmp_path, {"0": A, "1": B}), "--build-rank0", _build(tmp_path, 0, A), "--build-rank1",
                    _build(tmp_path, 1, B))
    assert llm.made["replay"]["ring_slots"] == slots


def test_no_sampled_replay_makes_the_ranks_with_no_replay_config(commit, tmp_path):
    llm, out = commit("--required-manifest", _manifest(tmp_path, {"0": A, "1": B}), "--build-rank0", _build(tmp_path, 0, A), "--build-rank1",
                      _build(tmp_path, 1, B), "--no-sampled-replay", "--no-weights-pin")
    assert llm.made["replay"] is None and "sampled_replay" not in _versions(out)
    assert _versions(out)["weights_pin"]["required"] is False


def test_a_rank_s_program_digest_is_read_from_its_build_by_the_kind_of_row(tmp_path):
    assert TPCM.build_program_digest(None) == (None, None) and TPCM.build_program_digest("") == (None, None)
    one = Path(_build(tmp_path, 0, A))
    assert TPCM.build_program_digest(str(one)) == (A, "request Program")
    parts = [B, C]
    for i, digest in enumerate(parts):
        (one / f"build_request_LP{i}_T4").mkdir()
        (one / f"build_request_LP{i}_T4" / "result.json").write_text(json.dumps({"program": {"digest": digest}}))
    assert TPCM.build_program_digest(str(one)) == (components_fold_digest(parts), "components-fold of 2 request Programs")
    (one / "build_workload").mkdir()
    (one / "build_workload" / "workload_program.json").write_text(json.dumps({"workload_digest": C}))
    assert TPCM.build_program_digest(str(one)) == (C, "workload (GP-01)")
    (one / "workload_program.json").write_text(json.dumps({"workload_digest": B}))                # the Build dir's own goes first
    assert TPCM.build_program_digest(str(one)) == (B, "workload (GP-01)")
    (one / "workload_program.json").write_text(json.dumps({"workload_digest": None}))             # a compose that did not run: the components
    assert TPCM.build_program_digest(str(one)) == (components_fold_digest(parts), "components-fold of 2 request Programs")


@pytest.mark.parametrize("record,kind", [(None, "request Program"), ("{not json", "request Program"), ("[]", "request Program"),
                                         ('{"program": {}}', "request Program"), ('{"program": null}', "request Program")],
                         ids=["absent", "not JSON", "a list", "no digest", "no Program"])
def test_an_unreadable_build_record_answers_no_digest_and_the_kind_it_was_read_as(tmp_path, caplog, record, kind):
    """Each time the driver's log says which dir, what it was read as and the error; with no Build dir there is nothing to say."""
    d = tmp_path / "build"
    d.mkdir()
    if record is not None:
        (d / "result.json").write_text(record)
    assert TPCM.build_program_digest(str(d)) == (None, kind)
    (d / "workload_program.json").write_text("{not json")
    assert TPCM.build_program_digest(str(d)) == (None, None)
    (d / "workload_program.json").unlink()
    (d / "build_request_LP0_T2").mkdir()
    (d / "build_request_LP0_T2" / "result.json").write_text('{"program": {}}')
    assert TPCM.build_program_digest(str(d)) == (None, "components-fold of 1 request Programs")
    assert TPCM.build_program_digest(None) == (None, None)
    said = [r.getMessage() for r in caplog.records if r.name == TPCM.__name__ and r.levelno == logging.WARNING]
    start = f"[tp2commit] the Build's record under {d} is unreadable (read as "
    assert [m[len(start):].split("): ")[0] for m in said if m.startswith(start)] == [
        kind, "a workload Program", "components-fold of 1 request Programs"]
    assert len(said) == 3 and "JSONDecodeError" in said[1] and said[2].endswith("KeyError('digest')")
