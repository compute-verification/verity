"""`served_match.main`, the served Match's command, on a CPU.

A GPU and vLLM run the command for real.  Here vLLM is the tree's stand-in (`tests.fake_vllm`), and the engine, the source
(`ServedCompiledSource`), the committer (`NativeCollectCommitter`) and the serve (`run_requests_arrivals`) are stand-ins
that write what the command asks of them into one log.  What the command reads off them is stood in too, each a function
with tests of its own: `facts_of`, `served_weights`, `build_here` and `check_committed` (`test_served_match_facts.py`,
`test_served_match_build.py`), the source's `linkage`, and the record of the served switches (`engine.served.record`).
The clock is a stand-in that moves a second and a half at each reading.  So these cases show what the command does
around them: what it asks and in which order, the files it writes, the lines it prints and its exit code.  That a
device's serve commits values the Build's Program replays is what a GPU says.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests import fake_vllm
from verity_vllm import config
from verity_vllm.acquire import served_artifact as SA
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.acquire.sources import served_compiled_source
from verity_vllm.commit.committer import native_collect
from verity_vllm.config import UsageError
from verity_vllm.engine import arrivals
from verity_vllm.engine import env as engine_env
from verity_vllm.engine import served as SW
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import served_match as sm

WORKLOAD = {"requests": [{"prompt_token_ids": [5, 6, 7], "max_tokens": 2, "arrive_step": 0},
                         {"prompt_token_ids": [8], "max_tokens": 4, "arrive_step": 1}],
            "engine": {"args": {"seed": 3, "gpu_memory_utilization": 0.9}}}
SCHEDULE = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 2},
            {"request_id": "r1", "arrive": 1, "prompt_tokens": 1, "max_tokens": 4}]
STEPS = 5
PROMPTS = {"r0": [5, 6, 7], "r1": [8]}
#: what the engine says it runs with: one value JSON has no form for, and more text than a log line shows
EFFECTIVE = {"model": PurePosixPath("/a/model"), "note": "n" * 400}
EFFECTIVE_WRITTEN = {"model": "/a/model", "note": "n" * 400}
FACTS = {"case": "A-CASE", "effective": EFFECTIVE, "device": "NVIDIA H200", "cc": [9, 0], "sms": 132,
         "attention": [{"backend": "FLASH_ATTN", "fa_version": 3}], "max_model_len": 4096, "max_num_batched_tokens": 2048}
RANKS = [{"rank": 0, "switches": {"a-switch": 1}}, {"rank": 1, "switches": {"a-switch": 1}}]
WINNERS = {"k0/a.best_config": {"BLOCK_M": 64}}
ARTIFACT = SA.artifact_of({"sha-a": {"config": {"num_warps": 4}}}, WINNERS)
BUILD = {"program_digest": "p" * 128, "manifest_digest": "m" * 128, "complete": True, "artifact": ARTIFACT, "identities": ["i0", "i1", "i2"]}
REPORT = {"ok": True, "tally": {"gemm": [5, 5]}}
LINKAGE = {"pairs": 4, "unlinked": []}
SUMMARY = {"stats": {"slots": 9}}
ROOT = bytes(range(64))
COMMITTER = {"run_id": "served-match", "arena_bytes": None, "wrap": True, "gather": True, "native_hooks": True, "auto_blocks": True,
             "native_worker": True}


def _weight(name: str) -> Any:
    raise AssertionError(f"the stand-in check reads no weight ({name})")


class _Llm:
    """The engine's `LLM` as far as the command asks it: each rank's reading of the served switches."""

    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log

    def collective_rpc(self, fn: Any) -> Any:
        self.log.append(("collective_rpc", fn))
        return iter(RANKS)


class _Source:
    """`ServedCompiledSource` as the command drives it."""

    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log
        #: the process's compile switches and the winners its Inductor dir held when the hooks went in
        self.at_hooks: dict[str, Any] = {}

    def install_hooks(self) -> None:
        names = ("TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", engine_env.INDUCTOR_DIR_MARK, *engine_env.COMPILE_FRESH)
        self.at_hooks = {name: os.environ.get(name) for name in names}
        self.at_hooks["winners"] = SA.winners(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "")
        self.log.append(("install_hooks",))

    def note_engine(self, model: Any) -> None:
        self.log.append(("note_engine", model))

    def attach(self, com: Any) -> None:
        self.log.append(("attach", com))

    def begin_run(self) -> None:
        self.log.append(("begin_run",))

    def summary(self) -> dict[str, Any]:
        self.log.append(("summary",))
        return SUMMARY


class _Committer:
    """`NativeCollectCommitter` as the command drives it: three engine steps committed."""

    _layouts: dict[int, list[Any]] = {0: [], 1: [], 2: []}

    def __init__(self, log: list[tuple[Any, ...]], retain: str, windowed: bool, extra: dict[str, Any] | None) -> None:
        self.log, self.retain, self.windowed, self.extra = log, retain, windowed, extra

    def size_arena(self, sizes: dict[str, int]) -> tuple[str, dict[str, int]]:
        self.log.append(("size_arena", sizes))
        return ("an arena for", sizes)

    def enable_windows(self, arena: Any) -> None:
        self.log.append(("enable_windows", arena))

    def reset(self) -> None:
        self.log.append(("reset",))

    def install(self) -> None:
        self.log.append(("install",))

    def finalize(self) -> Any:
        self.log.append(("finalize",))
        return SimpleNamespace(run_root=ROOT, extra=self.extra)

    def uninstall(self) -> None:
        self.log.append(("uninstall",))


class _Run:
    """One run of the command among its stand-ins.  `log` is what it asked of them, in order; `main` runs it in `out`, a
    directory under the test's that does not exist yet, given by a relative path."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, facts: dict[str, Any] | None = None,
                 problems: tuple[str, ...] = (), retain: str = "host", windowed: bool = False,
                 build: dict[str, Any] | Exception = BUILD, report: dict[str, Any] = REPORT, extra: dict[str, Any] | None = None) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.out = tmp_path / "runs" / "a"
        self.served = {"ok": not problems, "problems": list(problems), "ranks": RANKS}
        self.build, self.report = build, report
        self.source = _Source(self.log)
        self.com = _Committer(self.log, retain, windowed, extra)
        self.engine = SimpleNamespace(model="the model", llm=_Llm(self.log), effective=EFFECTIVE)
        for name in ("TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", engine_env.INDUCTOR_DIR_MARK, *engine_env.COMPILE_FRESH):
            monkeypatch.setenv(name, "")    # set first, so that the test's end takes out what the command sets
            monkeypatch.delenv(name)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "w.json").write_text(json.dumps(WORKLOAD))
        ticks = itertools.count()
        monkeypatch.setattr(sm, "time", SimpleNamespace(perf_counter=lambda: 100.0 + 1.5 * next(ticks)))
        fake_vllm.install(monkeypatch).set("vllm:SamplingParams", lambda **kw: ("sampling", kw))
        monkeypatch.setattr(served_compiled_source, "ServedCompiledSource", lambda: self.source)
        monkeypatch.setattr(served_compiled_source, "linkage", self._logged("linkage", LINKAGE))
        monkeypatch.setattr(va, "build_engine", self._build_engine)
        monkeypatch.setattr(sm, "facts_of", self._logged("facts_of", dict(FACTS if facts is None else facts)))
        monkeypatch.setattr(SW, "record", self._logged("record", self.served))
        monkeypatch.setattr(native_collect, "NativeCollectCommitter", self._committer)
        monkeypatch.setattr(arrivals, "run_requests_arrivals", self._serve)
        monkeypatch.setattr(sm, "build_here", self._build_here)
        monkeypatch.setattr(sm, "served_weights", self._logged("served_weights", _weight))
        monkeypatch.setattr(sm, "check_committed", self._check)

    def _logged(self, name: str, answer: Any) -> Any:
        def asked(*args: Any) -> Any:
            self.log.append((name, *args))
            return answer
        return asked

    def _build_engine(self, *args: Any, **kw: Any) -> Any:
        self.log.append(("build_engine", args, kw))
        return self.engine

    def _committer(self, eng: Any, **kw: Any) -> _Committer:
        self.log.append(("committer", eng, kw))
        return self.com

    def _serve(self, llm: Any, reqs: list[Any], params: Any) -> tuple[dict[str, Any], str]:
        self.log.append(("serve", llm, [vars(r) for r in reqs], [params(r) for r in reqs]))
        return {"r1": SimpleNamespace(outputs=[SimpleNamespace(token_ids=(13,))]),
                "r0": SimpleNamespace(outputs=[SimpleNamespace(token_ids=(11, 12)), SimpleNamespace(token_ids=(99,))])}, "the serve's timings"

    def _build_here(self, src: Any, facts: dict[str, Any], schedule: list[dict[str, Any]], steps: int, out: str) -> dict[str, Any]:
        self.log.append(("build_here", src, copy.deepcopy(facts), schedule, steps, out))
        if isinstance(self.build, Exception):
            raise self.build
        return self.build

    def _check(self, build: dict[str, Any], com: Any, weight: Any, prompts: dict[str, list[int]], order: dict[str, str], *,
               seed: int | None, rows: int | None, log: Any) -> dict[str, Any]:
        self.log.append(("check", build, com, weight, prompts, order, seed, rows))
        log("a line of the check")
        return self.report

    def main(self, **options: Any) -> int:
        return sm.main(sm.Options(**{"case": "A-CASE", "workload": "w.json", "out": "runs/a", **options}))

    def facts(self, served: bool = False) -> dict[str, Any]:
        """The facts of record as written: the engine's and the served switches', and once the workload was `served` the run's."""
        run = {"run_root": ROOT.hex(), "tokens": {"r1": [13], "r0": [11, 12]}, "linkage": LINKAGE, "source": SUMMARY,
               "errors": (self.com.extra or {}).get("errors")}
        return {**FACTS, "effective": EFFECTIVE_WRITTEN, "served": self.served, **(run if served else {})}

    def read(self, name: str) -> Any:
        return json.loads((self.out / name).read_text())

    def files(self) -> list[str]:
        return sorted(p.name for p in self.out.iterdir() if p.is_file())


def _before_the_committer(run: _Run, **engine: Any) -> list[tuple[Any, ...]]:
    """What a run asks until the served switches are judged."""
    args = {"max_num_seqs": 8, "case": "A-CASE", "execution": "compiled",
            "engine_args": {"seed": 3, "gpu_memory_utilization": 0.3, "kv_cache_memory_bytes": 4 << 30}, **engine}
    return [("install_hooks",), ("build_engine", (config.CHECKPOINTS,), args), ("note_engine", "the model"), ("facts_of", run.engine),
            ("collective_rpc", SW.read_rank), ("record", RANKS, SW.declared({**WORKLOAD, "execution": "compiled"}))]


def _the_serve(run: _Run, batch: int = 8, windowed: bool = False) -> list[tuple[Any, ...]]:
    """What a run asks from the committer on, until the Build."""
    sizes = {"batch": batch, "max_prompt": 3, "max_new": 4, "max_num_batched_tokens": 2048, "max_model_len": 4096}
    requests = [{"request_id": "r0", "prompt_token_ids": [5, 6, 7], "max_tokens": 2, "arrive_step": 0},
                {"request_id": "r1", "prompt_token_ids": [8], "max_tokens": 4, "arrive_step": 1}]
    sampling = [("sampling", {"temperature": 0.0, "max_tokens": n, "ignore_eos": True, "detokenize": False}) for n in (2, 4)]
    return [("committer", run.engine, COMMITTER), ("attach", run.com), ("size_arena", sizes),
            *([("enable_windows", ("an arena for", sizes))] if windowed else []), ("reset",), ("begin_run",), ("install",),
            ("serve", run.engine.llm, requests, sampling), ("finalize",), ("linkage", run.com), ("summary",),
            ("build_here", run.source, {**run.facts(served=True), "effective": EFFECTIVE}, SCHEDULE, STEPS, str(run.out))]


def _the_check(run: _Run, rows: int = 16) -> list[tuple[Any, ...]]:
    return [("served_weights", "the model"), ("check", run.build, run.com, _weight, PROMPTS, {"r0": "r0", "r1": "r1"}, None, rows),
            ("uninstall",)]


def _lines(*said: str) -> str:
    """The command's lines as printed: the clock's reading since its start, a second and a half more at each line."""
    return "".join(f"[served-match {1.5 * (i + 1):7.1f}s] {s}\n" for i, s in enumerate(said))


#: the engine's line shows the first 400 characters of what the engine runs with
ENGINE_LINE = 'engine A-CASE: {"model": "/a/model", "note": "' + "n" * 369
SERVED_LINE = "served 2 requests, 3 steps committed, root 0001020304050607, linkage {'pairs': 4, 'unlinked': []}"
BUILD_LINE = "Build: program pppppppppppppppp manifest mmmmmmmmmmmmmmmm identities 3 complete True"


def test_a_match_that_passes_asks_each_part_in_order_writes_three_files_and_exits_0(monkeypatch, tmp_path, capsys):
    run = _Run(monkeypatch, tmp_path)
    assert run.main() == 0
    assert run.log == [*_before_the_committer(run), *_the_serve(run), *_the_check(run)]
    assert run.files() == ["build.json", "facts.json", "match.json"]
    assert (run.out / "build.json").read_text() == json.dumps(BUILD, indent=1)
    assert (run.out / "facts.json").read_text() == json.dumps(run.facts(served=True), indent=1)
    match = {"ok": True, "build": {"program_digest": "p" * 128, "manifest_digest": "m" * 128, "complete": True, "artifact": ARTIFACT["digest"]},
             "build_of_record": None, "check": REPORT, "facts": run.facts(served=True)}
    assert (run.out / "match.json").read_text() == json.dumps(match, indent=1)
    assert len(ENGINE_LINE) == len("engine A-CASE: ") + 400
    assert capsys.readouterr().out == _lines(ENGINE_LINE, SERVED_LINE, BUILD_LINE, "a line of the check", 'Match PASS: tally {"gemm": [5, 5]}')


def test_the_process_has_its_own_compile_dirs_under_the_out_dir_before_the_hooks_go_in(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.main()
    assert run.source.at_hooks == {"TORCHINDUCTOR_CACHE_DIR": f"{run.out}/inductor", "VLLM_CACHE_ROOT": f"{run.out}/vllm_cache",
                                   engine_env.INDUCTOR_DIR_MARK: "1", **engine_env.COMPILE_FRESH, "winners": {}}


def test_an_out_dir_that_holds_an_earlier_match_is_written_over(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.out.mkdir(parents=True)
    (run.out / "match.json").write_text('{"ok": false}')
    assert run.main() == 0
    assert run.files() == ["build.json", "facts.json", "match.json"]
    assert run.read("match.json")["ok"] is True


@pytest.mark.parametrize(("options", "engine", "batch", "rows"), [
    ({"case": "B-CASE"}, {"case": "B-CASE"}, 8, 16),
    ({"max_num_seqs": 2}, {"max_num_seqs": 2}, 2, 16),
    ({"gpu_memory_utilization": 0.5}, {"engine_args": {"seed": 3, "gpu_memory_utilization": 0.5, "kv_cache_memory_bytes": 4 << 30}}, 8, 16),
    ({"kv_cache_gib": 1}, {"engine_args": {"seed": 3, "gpu_memory_utilization": 0.3, "kv_cache_memory_bytes": 1 << 30}}, 8, 16),
    ({"rows": 4}, {}, 8, 4)])
def test_each_option_reaches_the_engine_the_arena_or_the_check(monkeypatch, tmp_path, options, engine, batch, rows):
    run = _Run(monkeypatch, tmp_path)
    assert run.main(**options) == 0
    assert run.log == [*_before_the_committer(run, **engine), *_the_serve(run, batch), *_the_check(run, rows)]


def test_a_committer_that_commits_in_windows_has_them_sized_by_the_arena_before_it_is_reset(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, windowed=True)
    assert run.main() == 0
    assert run.log == [*_before_the_committer(run), *_the_serve(run, windowed=True), *_the_check(run)]


def test_served_switches_with_a_problem_refuse_the_build_before_a_committer_is_made(monkeypatch, tmp_path, capsys):
    run = _Run(monkeypatch, tmp_path, problems=("rank 0: a problem", "rank 1: another"))
    assert run.main() == 3
    assert run.log == _before_the_committer(run)
    refused = "served switches: rank 0: a problem; rank 1: another"
    assert run.files() == ["match.json"]
    assert (run.out / "match.json").read_text() == json.dumps({"ok": False, "build_refused": refused, "facts": run.facts()}, indent=1)
    assert capsys.readouterr().out == _lines(ENGINE_LINE, f"Build refused: {refused}")


def test_a_device_no_card_is_registered_for_refuses_the_build_before_a_committer_is_made(monkeypatch, tmp_path, capsys):
    facts = {**FACTS, "device": "NVIDIA H100 80GB HBM3"}
    with pytest.raises(Unmodelled) as why:
        sm.device_of(facts)
    run = _Run(monkeypatch, tmp_path, facts=facts)
    assert run.main() == 3
    assert run.log == _before_the_committer(run)
    assert "NVIDIA H100 80GB HBM3" in str(why.value)
    assert run.read("match.json") == {"ok": False, "build_refused": str(why.value), "facts": {**run.facts(), "device": "NVIDIA H100 80GB HBM3"}}
    assert capsys.readouterr().out == _lines(ENGINE_LINE, f"Build refused: {why.value}")


def test_a_committer_that_does_not_retain_on_the_host_is_refused_before_the_source_is_attached(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path, retain="device")
    with pytest.raises(UsageError) as why:
        run.main()
    assert str(why.value) == ("VERITY_RETAIN=device: the check reads the committed values the collector retains on the host "
                              "(export VERITY_RETAIN=host)")
    assert run.log == [*_before_the_committer(run), ("committer", run.engine, COMMITTER)]
    assert run.files() == []


def test_a_build_refused_after_the_serve_is_written_with_the_run_s_facts_and_exits_3(monkeypatch, tmp_path, capsys):
    run = _Run(monkeypatch, tmp_path, build=Unmodelled("no split graph"))
    assert run.main() == 3
    assert run.log == [*_before_the_committer(run), *_the_serve(run)]
    assert run.files() == ["match.json"]
    assert (run.out / "match.json").read_text() == json.dumps({"ok": False, "build_refused": "no split graph", "facts": run.facts(served=True)},
                                                             indent=1)
    assert capsys.readouterr().out == _lines(ENGINE_LINE, SERVED_LINE, "Build refused: no split graph")


@pytest.mark.parametrize(("change", "ok"), [
    ({}, True), ({"extra": {}}, True), ({"extra": {"errors": []}}, True),
    ({"report": {"ok": False, "tally": {"gemm": [4, 5]}}}, False),
    ({"build": {**BUILD, "complete": False}}, False),
    ({"extra": {"errors": ["a value was not committed"]}}, False)])
def test_the_match_passes_when_the_check_does_the_build_is_complete_and_the_run_had_no_error(monkeypatch, tmp_path, capsys, change, ok):
    run = _Run(monkeypatch, tmp_path, **change)
    build, report = change.get("build", BUILD), change.get("report", REPORT)
    assert run.main() == (0 if ok else 1)
    assert run.log == [*_before_the_committer(run), *_the_serve(run), *_the_check(run)]
    match = run.read("match.json")
    assert (match["ok"], match["check"], match["build"]["complete"], match["facts"]) == (ok, report, build["complete"], run.facts(served=True))
    assert run.read("facts.json") == run.facts(served=True) and run.read("build.json") == build
    tally = json.dumps(report["tally"])
    assert capsys.readouterr().out.splitlines()[-1] == f"[served-match     7.5s] Match {'PASS' if ok else 'FAIL'}: tally {tally}"


def _of_record(tmp_path: Path, **over: Any) -> dict[str, Any]:
    """A Build of record in a file, and the option that names it."""
    (tmp_path / "of_record.json").write_text(json.dumps({"program_digest": "p" * 128, "artifact": ARTIFACT, **over}))
    return {"build_of_record": "of_record.json"}


def test_a_build_of_record_seeds_the_inductor_dir_before_the_hooks_and_a_build_that_is_its_passes(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    assert run.main(**_of_record(tmp_path)) == 0
    assert run.source.at_hooks["winners"] == WINNERS
    assert run.log == [*_before_the_committer(run), *_the_serve(run), *_the_check(run)]
    match = run.read("match.json")
    assert match["build_of_record"] == {"program_digest": "p" * 128, "artifact": ARTIFACT["digest"], "artifact_differences": [], "equal": True}
    assert match["ok"] is True


MANY = SA.artifact_of({f"sha-{i:02d}": {"config": {"num_warps": i}} for i in range(24)}, WINNERS)


@pytest.mark.parametrize(("of_record", "build", "differences"), [
    ({"program_digest": "q" * 128}, BUILD, []),
    ({}, {**BUILD, "artifact": SA.artifact_of({"sha-a": {"config": {"num_warps": 2}}}, WINNERS)},
     ["kernels sha-a: pinned {'num_warps': 4}, loaded {'num_warps': 2}"]),
    ({}, {**BUILD, "artifact": MANY},
     [f"kernels sha-{i:02d}: pinned absent, loaded {{'num_warps': {i}}}" for i in range(20)])])
def test_a_build_with_another_program_or_another_artifact_than_the_build_of_record_fails(monkeypatch, tmp_path, capsys, of_record, build,
                                                                                         differences):
    run = _Run(monkeypatch, tmp_path, build=build)
    assert run.main(**_of_record(tmp_path, **of_record)) == 1
    match = run.read("match.json")
    assert match["build_of_record"] == {"program_digest": of_record.get("program_digest", "p" * 128), "artifact": ARTIFACT["digest"],
                                        "artifact_differences": differences, "equal": False}
    assert (match["ok"], match["check"], match["build"]["artifact"]) == (False, REPORT, build["artifact"]["digest"])
    assert capsys.readouterr().out.splitlines()[-1] == '[served-match     7.5s] Match FAIL: tally {"gemm": [5, 5]}'


def test_a_build_of_record_that_pins_no_artifact_is_refused_before_anything_is_asked(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    with pytest.raises(UsageError, match="records no compile artifact"):
        run.main(**_of_record(tmp_path, artifact=None))
    assert run.log == [] and run.files() == []
