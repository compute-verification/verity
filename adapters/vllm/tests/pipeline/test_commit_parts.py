"""`verity_vllm.pipeline.commit`'s small parts that `test_commit_main` stands in for or never reaches: the source's commit
(`git_head`), the device's memory in use (`nvidia_smi_used_mib`), an engine row's request (`_rid_of`) and the staging's
environment (`staging_environ`) where it refuses, excludes families, traces the footprint or drops a flush point; the Programs'
forecast (`c2_programs_forecast`) where an instance file can't be read; the serving loop (`run_workload`) where a request
arrives late, the engine has nothing to serve or a request is let go unfinished, over `test_commit_main`'s stand-in engine;
the hooks-only ablation (`make_committer`'s ``hooks_only``); the check that the engine is left as it was (`assert_pristine`)
where a runner's method or a module's forward is still wrapped, or the sampler keeps a hook; and the kept bytes' digests
(`tensor_digests`): a whole FA2 row's planes, a row of zeros, a later pair's differences and the step dumped.  The two
commands each reader runs, and `commit`'s clock, are stood in.  CPU only."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from tests.pipeline.test_commit_main import TOKENS, WORKLOAD, Clock
from tests.pipeline.test_commit_main import engine as served
from verity_vllm import settings
from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.commit.committer import native_collect as NC
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import cli, commit

#: The environment names `staging_environ` writes: absent at a test's start, and taken back at its end.
ENVIRON = ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED", "VERITY_LEARN_HOST_BUDGET_MB",
           "VERITY_RETAIN_EXCLUDE", "VERITY_FOOTPRINT")


def runs(monkeypatch: pytest.MonkeyPatch, outcome: str | Exception) -> list[list[str]]:
    """`subprocess.run` prints ``outcome`` (raises it if it is an exception); the commands it was asked for."""
    asked: list[list[str]] = []

    def run(cmd: list[str], **_: Any) -> SimpleNamespace:
        asked.append(cmd)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(stdout=outcome)
    monkeypatch.setattr(subprocess, "run", run)
    return asked


def test_the_source_s_commit_is_the_export_s_when_the_directory_holds_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    asked = runs(monkeypatch, "f" * 40 + "\n")
    (tmp_path / "EXPORT.json").write_text(json.dumps({"commit": "a" * 40}))
    assert commit.git_head() == "a" * 40 and asked == []


@pytest.mark.parametrize("export, printed, head", [
    (None, "f" * 40 + "\n", "f" * 40),
    ("{not json", "f" * 40 + "\n", "f" * 40),
    (None, "", None),
    (None, FileNotFoundError("git"), None),
])
def test_else_it_is_git_s_head_and_none_where_git_names_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export: str | None,
                                                              printed: str | Exception, head: str | None) -> None:
    monkeypatch.chdir(tmp_path)
    if export is not None:
        (tmp_path / "EXPORT.json").write_text(export)
    asked = runs(monkeypatch, printed)
    assert commit.git_head() == head and asked == [["git", "rev-parse", "HEAD"]]


@pytest.mark.parametrize("printed, mib", [("1234\n5678\n", 1234), ("", None), ("[N/A]\n", None), (FileNotFoundError("nvidia-smi"), None)])
def test_the_device_s_memory_in_use_is_the_first_device_s_and_none_where_it_cannot_be_read(
        monkeypatch: pytest.MonkeyPatch, printed: str | Exception, mib: int | None) -> None:
    asked = runs(monkeypatch, printed)
    assert commit.nvidia_smi_used_mib() == mib
    assert asked == [["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]]


@pytest.mark.parametrize("rid, named", [("abc", "r1"), ("abc-0123abcd", "r1"), ("abc-0123ABCD", "abc-0123ABCD"),
                                        ("xyz-0123abcd", "xyz-0123abcd"), ("def-89abcdef", "r3"), (7, "r7")])
def test_an_engine_row_s_request_is_the_workload_s_by_its_id_or_the_id_vllm_suffixed_else_the_raw_id(rid: Any, named: str) -> None:
    order = {"abc": "r1", "def-89abcdef": "r3", "7": "r7"}
    assert commit._rid_of(order, rid) == named


@pytest.fixture
def environ(monkeypatch: pytest.MonkeyPatch) -> Callable[[], dict[str, str | None]]:
    """The names `staging_environ` writes, absent now; what each holds when called."""
    for name in ENVIRON:
        monkeypatch.delenv(name, raising=False)
    return lambda: {name: settings.raw(name) for name in ENVIRON}


def staged(tmp_path: Path, **resolved: Any) -> commit.Options:
    """The options of a Commit with its staging resolved: bounded, retained on the host and opened after release, unless given."""
    args = cli.parse("commit", ["--workload", str(tmp_path / "workload.json"), "--out", str(tmp_path / "out")], {})
    return dataclasses.replace(args, **{"window_mb": 256, "window_slots": None, "bounded_staging": True, "retain": "host"} | resolved)


@pytest.mark.parametrize("unmet", [{"bounded_staging": False}, {"retain": "device"}, {"retain": "none"}, {"open_after_release": False}])
def test_families_are_excluded_only_from_bounded_staging_retained_on_the_host_and_opened_after_release(
        tmp_path: Path, environ: Callable[[], dict[str, str | None]], unmet: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match=r"^--retain-exclude needs --bounded-staging --retain host --open-after-release "):
        commit.staging_environ(staged(tmp_path, retain_exclude="mlp_up", **unmet))
    assert environ()["VERITY_RETAIN_EXCLUDE"] is None


def test_the_window_s_slots_the_excluded_families_and_the_footprint_s_file_go_to_the_environment(
        tmp_path: Path, environ: Callable[[], dict[str, str | None]]) -> None:
    footprint = tmp_path / "a" / "b" / "footprint.jsonl"
    commit.staging_environ(staged(tmp_path, retain_exclude="mlp_up,mlp_down", footprint=str(footprint), learn_budget_mb=512,
                                window_slots=4))
    assert environ() == {"VERITY_RETAIN": "host", "VERITY_WINDOW_MB": "256", "VERITY_WINDOW_SLOTS": "4", "VERITY_STAGING_BOUNDED": "1",
                         "VERITY_LEARN_HOST_BUDGET_MB": "512", "VERITY_RETAIN_EXCLUDE": "mlp_up,mlp_down",
                         "VERITY_FOOTPRINT": str(footprint)}
    assert footprint.parent.is_dir() and not footprint.exists()


def test_a_late_read_drops_its_leaf_from_the_gather_flush_points_and_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        environ: Callable[[], dict[str, str | None]]) -> None:
    before = NC.GATHER_FLUSH_LEAVES
    leaf = sorted(before)[0]
    monkeypatch.setattr(NC, "GATHER_FLUSH_LEAVES", before)
    commit.staging_environ(staged(tmp_path, late_read=leaf))
    left = NC.GATHER_FLUSH_LEAVES
    assert left == before - {leaf} and leaf in before
    assert capsys.readouterr().out == (f"[delta] LATE-READ NEGATIVE TEST: gather flush points {sorted(left)} (dropped {leaf!r}); "
                                       "roots must NOT match arm 2\n")
    assert environ()["VERITY_RETAIN_EXCLUDE"] is None


def test_an_instance_file_that_cannot_be_read_is_named_and_counts_no_rows(tmp_path: Path) -> None:
    good, bad = tmp_path / "build_request_a" / "instances.json.gz", tmp_path / "build_request_b" / "instances.json.gz"
    text = json.dumps({"rows": [{"i": 0}, {"i": 1}]}).encode()
    for f in (good, bad):
        f.parent.mkdir()
    with gzip.open(good, "wb") as fh:
        fh.write(text)
    bad.write_bytes(text)
    with pytest.raises(gzip.BadGzipFile) as raised, gzip.open(bad, "rb") as fh:
        fh.read()
    fc = commit.c2_programs_forecast(str(tmp_path), None, factor=2.0)
    assert (fc["rows"], fc["json_bytes"], fc["bytes"], fc["why"]) == (2, len(text), 2 * len(text), None)
    assert fc["files"] == [{"file": str(good), "rows": 2, "json_bytes": len(text), "gz_bytes": good.stat().st_size},
                           {"file": str(bad), "error": repr(raised.value)[:200]}]


def serve(monkeypatch: pytest.MonkeyPatch, requests: list[dict[str, Any]], finishing: set[str] | None = None) -> dict[str, Any]:
    """`run_workload` over `test_commit_main`'s stand-in engine, each step a second of `commit`'s clock from 100 s, and a
    request's sampling its `max_tokens` alone.  With ``finishing``, a request outside it leaves the engine without an output
    that says it finished."""
    clock = Clock()
    clock.take(100.0)
    monkeypatch.setattr(commit, "time", clock)
    monkeypatch.setattr(va, "sampling_params", lambda req, sampling: SimpleNamespace(max_tokens=int(req["max_tokens"])))
    eng = served(clock)
    core = eng.llm.llm_engine
    if finishing is not None:
        stepped = core.step
        monkeypatch.setattr(core, "step", lambda: [SimpleNamespace(**{**vars(o), "finished": o.finished and o.request_id in finishing})
                                                   for o in stepped()])
    return commit.run_workload(eng, requests, {})


@pytest.mark.parametrize("arrive, tokens_out", [(1, [1, 2, 2]), (3, [1, 1, 1, 1, 1]), (7, [1, 1, 1, 1, 1])])
def test_a_request_joins_at_its_arrival_step_or_at_once_when_the_engine_has_nothing_to_serve(
        monkeypatch: pytest.MonkeyPatch, arrive: int, tokens_out: list[int]) -> None:
    """A request is admitted once the engine has stepped `arrive_step` times; an engine with nothing left to serve takes the
    next request at once, without a step."""
    first, second = WORKLOAD["requests"]
    ran = serve(monkeypatch, [first, {**second, "arrive_step": arrive}])
    assert (ran["tokens"], ran["step_tokens_out"], ran["iterations"]) == (TOKENS, tokens_out, len(tokens_out))
    assert ran["step_wall_s"] == [1.0] * len(tokens_out) and (ran["wall_s"], ran["t_last_step"]) == (len(tokens_out), 100 + len(tokens_out))


def test_a_request_the_engine_lets_go_unfinished_is_refused_by_the_count_of_such(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match=r"^1 requests did not finish$"):
        serve(monkeypatch, WORKLOAD["requests"], finishing={"a"})


def test_the_hooks_only_ablation_counts_the_device_tensors_its_hooks_see_and_commits_nothing(
        monkeypatch: pytest.MonkeyPatch, environ: Callable[[], dict[str, str | None]]) -> None:
    """``hooks_only``, an ablation and no commitment, staged directly: its hooks count each device tensor that holds anything
    and copy nothing, and a step waits for its event and records zero roots, no leaves, no bytes and no layout."""
    clock = Clock()
    clock.take(100.0)
    monkeypatch.setattr(commit, "time", clock)
    com = commit.make_committer("hooks_only", None, cast(Any, SimpleNamespace(hash_threads=1, chunk=3072)), "r0")

    def tensor(device: str, n: int) -> SimpleNamespace:
        return SimpleNamespace(device=SimpleNamespace(type=device), numel=lambda: (clock.take(0.25), n)[1])

    waited: list[float] = []
    job = SimpleNamespace(step=3, metas=[], event=SimpleNamespace(synchronize=lambda: waited.append(clock.now)), t_step_end=99.0,
                          backlog=2)
    for name, device, n in [("a", "cuda", 4), ("b", "cuda", 0), ("c", "cpu", 4), ("d", "cuda", 1)]:
        com._acquire(name, tensor(device, n), job)
    com._commit_step(job)
    zero = b"\0" * commit.SR.ROOT_BYTES
    assert (com.name, com.chunk) == ("hooks_only_direct+rows-sha512+retain_host", 3072)
    assert (job.metas, waited, com._layouts) == ([None, None], [100.75], {3: []})
    assert com._done == {3: StepCommitment(3, zero, zero, 0, 0, 2, b"\0" * 32, 99.0, 100.75, 2)}
    assert (com.spans.total["acquisition"], com.spans.per_step[3]["acquisition"]) == (0.75, 0.75)


def engine(**runner: Any) -> Any:
    """An engine whose runner holds ``runner`` and whose model has two modules, as `assert_pristine` reads one."""
    return SimpleNamespace(runner=SimpleNamespace(**runner), model=torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU()))


BASELINE: dict[str, Any] = {"runner_had": {}, "hooks": 0, "sampler_hooks": 0}


@pytest.mark.parametrize("nm", ["execute_model", "sample"])
def test_a_runner_method_still_wrapped_is_refused_unless_the_runner_had_it_before(nm: str) -> None:
    eng = engine(**{nm: lambda *a: None})
    with pytest.raises(AssertionError, match=rf"^runner\.{nm} still wrapped$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"runner_had": {nm: True}})


def test_a_module_forward_still_wrapped_is_refused_unless_the_baseline_counts_it() -> None:
    eng = engine()
    commit.assert_pristine(eng, BASELINE)
    eng.model[0].forward = lambda x: x
    with pytest.raises(AssertionError, match=r"^wrapped module\.forward present: 1 != baseline 0$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"wrapped_forwards": 1})


def test_a_hook_left_on_the_sampler_is_refused_unless_the_baseline_counts_it() -> None:
    sampler = torch.nn.Identity()
    eng = engine(sampler=sampler)
    commit.assert_pristine(eng, BASELINE)
    sampler.register_forward_hook(lambda *a: None)
    with pytest.raises(AssertionError, match=r"^sampler hooks present$"):
        commit.assert_pristine(eng, BASELINE)
    commit.assert_pristine(eng, BASELINE | {"sampler_hooks": 1})


FA2 = "fa2.m1/L0/g1x1x1"
#: A whole-buffer FA2 row's six planes and their bytes, at one head block, one row and one key block.
PLANES = (("S", 512), ("P", 512), ("RP", 256), ("ROW", 32), ("O", 256), ("FIN", 16))
FA2_BYTES = sum(nb for _, nb in PLANES)


def kept(*tensors: tuple[str, bytes]) -> SimpleNamespace:
    """A committer that kept ``tensors`` back to back at step 0, and nothing at step 1."""
    data = b"".join(b for _, b in tensors)
    layout, off = [], 0
    for name, b in tensors:
        layout.append(SimpleNamespace(name=name, dev_off=off, nbytes=len(b), dtype="uint8", shape=(len(b),)))
        off += len(b)
    return SimpleNamespace(_gpu_blocks={0: (torch.frombuffer(bytearray(data), dtype=torch.uint8), len(data)), 1: (None, 0)},
                           _layouts={0: layout})


def b2(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., int]:
    """`tensor_digests` into ``tmp_path`` on `commit`'s clock held at 0, with no step dumped unless the test sets one."""
    monkeypatch.setattr(commit, "time", SimpleNamespace(perf_counter=lambda: 0.0))
    monkeypatch.delenv("VERITY_DUMP_STEP", raising=False)
    return lambda com, pair, prev: commit.tensor_digests(com, tmp_path, pair, prev)


def test_each_kept_tensor_and_each_plane_of_a_whole_fa2_row_is_digested_and_a_row_of_zeros_is_named(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], digests: Callable[..., int]) -> None:
    up, fa2 = bytes(range(8)), bytes(FA2_BYTES)
    prev: dict[str, Any] = {}
    assert digests(kept(("mlp.up/L0", up), (FA2, fa2)), 0, prev) == 1
    planes, off = {}, 0
    for pl, nb in PLANES:
        planes[f"{FA2}/{pl}"] = b2(fa2[off:off + nb])
        off += nb
    step = {"mlp.up/L0": b2(up), FA2: b2(fa2), **planes}
    assert json.loads((tmp_path / "tensor_digests_pair0.json").read_text()) == {"0": step}
    assert prev == {"pair": 0, "digests": {0: step}}
    assert capsys.readouterr().out == f"[delta] FA2 ROWS BOUND AS ZEROS (pair 0): 1 e.g. ['0:{FA2}']\n"


def test_a_later_pair_names_the_steps_and_the_families_whose_digests_differ_from_the_pair_before(
        capsys: pytest.CaptureFixture[str], digests: Callable[..., int]) -> None:
    prev: dict[str, Any] = {}
    digests(kept(("mlp.up/L0", bytes(8)), (FA2, bytes(FA2_BYTES))), 0, prev)
    capsys.readouterr()
    assert digests(kept(("mlp.up/L0", b"\1" * 8), (FA2, b"\1" * 512 + bytes(FA2_BYTES - 512))), 1, prev) == 0
    assert capsys.readouterr().out == (
        "[delta] TENSOR DIGESTS pair 1 vs pair 0: 1 steps differ [0]; differing tensors by family {'up': 1, 'fa2.m1': 1, 'fa2.m1/S': 1}; "
        f"steps whose digest MULTISET is equal (pure permutation) 0; 0.0s e.g. mlp.up/L0, {FA2}, {FA2}/S\n")
    assert prev["pair"] == 1


def test_a_step_whose_tensors_only_changed_places_is_counted_as_a_permutation(
        capsys: pytest.CaptureFixture[str], digests: Callable[..., int]) -> None:
    prev: dict[str, Any] = {}
    digests(kept(("mlp.up/L0", b"\1" * 8), ("mlp.down/L0", b"\2" * 8)), 0, prev)
    digests(kept(("mlp.up/L0", b"\2" * 8), ("mlp.down/L0", b"\1" * 8)), 1, prev)
    assert capsys.readouterr().out == (
        "[delta] TENSOR DIGESTS pair 1 vs pair 0: 1 steps differ [0]; differing tensors by family {'up': 1, 'down': 1}; "
        "steps whose digest MULTISET is equal (pure permutation) 1; 0.0s e.g. mlp.up/L0, mlp.down/L0\n")


def test_the_step_asked_for_dumps_its_fa2_l0_row_s_bytes_and_its_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         digests: Callable[..., int]) -> None:
    monkeypatch.setenv("VERITY_DUMP_STEP", "0")
    fa2 = bytes(range(256)) * (FA2_BYTES // 256) + bytes(FA2_BYTES % 256)
    digests(kept(("mlp.up/L0", bytes(8)), (FA2, fa2)), 3, {})
    assert (tmp_path / "dump_step0_pair3_fa2.m1_L0_g1x1x1.bin").read_bytes() == fa2
    assert json.loads((tmp_path / "dump_step0_pair3_layout.json").read_text()) == [["mlp.up/L0", "uint8", [8], 8, 0],
                                                                                   [FA2, "uint8", [FA2_BYTES], FA2_BYTES, 8]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dump_step0_pair3_fa2.m1_L0_g1x1x1.bin", "dump_step0_pair3_layout.json",
                                                          "tensor_digests_pair3.json"]
