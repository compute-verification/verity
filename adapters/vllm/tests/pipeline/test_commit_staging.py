"""`verity_vllm.pipeline.commit`: what `main` does with a committer that has a device staging block, from sizing the block
to registering the weights, and at its instrumented warm-up.

Real: all that `tests.pipeline.test_commit_main` runs for real, and the admissions `main` computes here: the host admission
of the run's scheduled bytes with the planner's resident term (`admission_bound.admission`, `committer_resident`), and for a
committer with bounded staging the stream of the declared schedule, the forecast of the sampled replay's Programs and the
host bound (`native_host.schedule_stream_bound`, `commit.c2_programs_forecast`, `native_host.bounded_host_bound`).  The
fixture keeps what `main` handed each (``handed``) and what each gave before `main` wrote on it (``gave``).

Stood in: that module's stand-ins; the committer, `Staged`, which is its committer with a staging block of round widths
and the staging a test gives it; and the host's memory files, read as a test states them (`HostPath`).

Held: what the committer is asked, in order; the sizing record as `main` completes it, which is the dict the committer gave;
every line `main` prints from the sizing to the weights' registration; the files it writes there; and how it ends: by going
on to its runs, or by exit 3 with no weights registered.  The numbers a test states are the ones its own sizing, limit and
stand-ins give; the planner's and the bound's own terms are read from what those functions gave.
CPU only."""
from __future__ import annotations

import copy
import dataclasses
import inspect
import json
from pathlib import PosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_commit_main as M
from tests.pipeline.test_commit_main import RUN, WARMUP, Committer, Ended, Record, run
from verity_vllm.commit.committer import native_host
from verity_vllm.pipeline import admission_bound, commit

#: `test_commit_main`'s fixture, under its name in this module.
stand_ins = M.stand_ins

MIB, GIB = 1 << 20, 1 << 30
#: The names `main` records the host's limit under hold a word this tree has retired, which stays where it stands and is
#: spelled nowhere new.  They are read from where it stands, the host bound's own parameter: the limit and where it was
#: read, as an admission has them (`LIMIT`, `SOURCE`) and as the sizing has them (`SIZING_LIMIT`, `SIZING_SOURCE`).
LIMIT = next(name for name in inspect.signature(native_host.bounded_host_bound).parameters if name.startswith("host_"))
SOURCE = LIMIT.replace("_bytes", "_source")
SIZING_LIMIT, SIZING_SOURCE = (name.replace("host_", "host_memory_") for name in (LIMIT, SOURCE))
KIND = f"plugin:{__name__}:committer"
#: A staging block's sizing of round widths: a mebibyte a token, a sequence and a step, and no hidden rows.
SIZING: Record = {"per_token_bytes": MIB, "per_seq_bytes": MIB, "per_step_bytes": MIB, "prefill_tokens": 5, "prefill_step_bytes": 8 * MIB,
                  "prefill_hidden_bytes": 0, "decode_step_bytes": 5 * MIB, "decode_hidden_bytes": 0, "step_bound_bytes": 8 * MIB,
                  "block_bytes": 16 * MIB, "prefill_block_bytes": 16 * MIB, "decode_block_bytes": 8 * MIB, "blocks_preallocated": 2,
                  "retained_bytes_bound": 24 * MIB, "device_free_bytes": 1024 * MIB, "gpu_memory_utilization_max": 0.5,
                  "model": {"layers": 1, "hidden_m1": False}}
#: `WORKLOAD`'s envelope as `main` hands it to the sizing.
ENVELOPE: Record = {"batch": 8, "max_prompt": 3, "max_new": 3, "max_num_batched_tokens": None, "max_model_len": None}
RING: Record = {"slots": 4, "window_bytes": 2 * MIB, "largest_tensor_bound": MIB, "in_place_hashing": True}
#: What a bounded committer's run reports of its staging, which `main` reads at the warm-up.
LEARNED: Record = {"bounded_staging": {"learned_only_steps": 3, "learned_only_bytes": 2 * MIB, "transient_peak_bytes": 3 * MIB,
                                       "learn_budget_bytes": 4 * MIB, "plan_mismatches": 0},
                   "native_collect": {"plans_cached": 3}, "pinned_pool_bytes": 5 * MIB}
V2, V1, MEMINFO = "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/proc/meminfo"
MEMTOTAL = "/proc/meminfo MemTotal (no cgroup limit visible)"
GOVERNS = "the admission whose domain is the run's retention mode governs; the other-mode admission is on the record and does not refuse alone"
DOMAIN = "--retain host (the whole run's raw bytes pinned on host); a non-host retention is recorded, never refused"
SIZED_LINE = (f"[delta] arena sizing (X-02, {KIND}): decode block 8 MiB x2 + prefill block 16 MiB (step bound 8 MiB: prefill 5 tok -> 8 MiB "
              "[hidden 0 MiB], decode -> 5 MiB; per-token 1048576 B), retained bound 24 MiB; device free 1024 MiB; gpu_memory_utilization <= 0.5")
#: The scheduled bytes of `WORKLOAD` under `SIZING` with one lag row: steps of 8, 5, 5 and 3 MiB.
ADMISSION_LINE = (f"[delta] HOST ADMISSION ({KIND}): scheduled raw 21 MiB over 4 steps (5 prompt tokens + 5 decode rows, lag 1; peak step 0 = 8 MiB) "
                  "-> predicted host {predicted} MiB vs limit {limit} MiB: {verdict} -- {why}")
RING_LINE = "[delta] windowed staging (P0): 4 x 2 MiB ring + scratch (largest tensor bound 1 MiB); in-place hashing: True"


class Staged(Committer):
    """`Committer` with a device staging block, which `main` sizes before the weights are registered.  ``arena`` False is a
    committer that can size a block and has none; ``retain``, ``bounded``, ``windowed`` and ``exclude`` are its staging as `main`
    reads it; ``sizing`` is what `size_arena` gives, and ``asz`` that same dict, which `main` goes on writing."""

    def __init__(self, eng: SimpleNamespace, run_id: str, *, arena: bool = True, retain: str = "host", bounded: bool = False,
                 windowed: bool = False, exclude: tuple[str, ...] = (), sizing: Record = SIZING, **kw: Any) -> None:
        super().__init__(eng, run_id, **kw)
        self.arena = object() if arena else None
        self.retain, self.bounded, self.windowed, self.retain_exclude = retain, bounded, windowed, frozenset(exclude)
        self.chunk, self.win_bytes, self.learn_budget_bytes, self.pool = 256, 2 * MIB, 4 * MIB, SimpleNamespace(exact=True)
        self.sizing = sizing
        self.sized: list[Record] = []
        self.asz: Any = None
        self.ring_of: Any = None
        self.host_admission: Any = None

    def size_arena(self, envelope: Record, *, hidden_m1: bool) -> Record:
        self.calls.append("size_arena")
        self.sized.append({"envelope": envelope, "hidden_m1": hidden_m1})
        self.asz = copy.deepcopy(self.sizing)
        return self.asz

    def enable_windows(self, asz: Record) -> Record:
        self.calls.append("enable_windows")
        self.ring_of = asz
        return dict(RING)

    def finalize(self) -> Any:
        run = super().finalize()
        return dataclasses.replace(run, extra={**run.extra, **LEARNED}) if self.bounded else run


def committer(eng: SimpleNamespace, run_id: str) -> Staged:
    """``--committer plugin:<this module>:committer``."""
    com = Staged(eng, run_id, **M.MAKE)
    M.MADE.append(com)
    return com


def made() -> list[Staged]:
    """`M.MADE` as what this module's `committer` put there."""
    return cast("list[Staged]", M.MADE)


#: The host's memory files by path; one it does not name is not there.  `host` empties it.
HOST: dict[str, str] = {}


class HostPath(PosixPath):
    """`commit`'s `Path`: a file under ``/sys/fs/cgroup`` or ``/proc`` reads as `HOST` has it, and every other path as itself."""

    def read_text(self, *args: Any, **kw: Any) -> str:
        if not str(self).startswith(("/sys/fs/cgroup/", "/proc/")):
            return super().read_text(*args, **kw)
        if str(self) not in HOST:
            raise FileNotFoundError(str(self))
        return HOST[str(self)]


@pytest.fixture
def host(stand_ins, monkeypatch):
    """`stand_ins` on a host with no memory file, and the admissions' functions kept: ``handed`` is each call's name and what
    `main` handed it, ``gave`` what the last call of each name returned, both as they were at the call."""
    HOST.clear()
    stand_ins.handed, stand_ins.gave = [], {}

    def keep(owner: Any, name: str) -> None:
        real = getattr(owner, name)

        def call(*args: Any, **kw: Any) -> Any:
            stand_ins.handed.append((name, copy.deepcopy(args), copy.deepcopy(kw)))
            out = real(*args, **kw)
            stand_ins.gave[name] = copy.deepcopy(out)
            return out

        monkeypatch.setattr(owner, name, call)

    for owner, name in [(admission_bound, "committer_resident"), (admission_bound, "admission"), (native_host, "schedule_stream_bound"),
                        (commit, "c2_programs_forecast"), (native_host, "bounded_host_bound")]:
        keep(owner, name)
    monkeypatch.setattr(commit, "Path", HostPath)
    return stand_ins


def staged(tmp_path: Any, capsys: Any, *argv: str) -> Ended:
    """`run` with the staged committer and three replay workers, which the planner's resident term would else take from this
    machine's processors."""
    return run(tmp_path, capsys, "--committer", KIND, "--replay-workers", "3", *argv)


def setup_lines(ended: Ended) -> list[str]:
    """What `main` printed after the control's warm-up and before the weights' registration, or to its end."""
    lines = ended.printed[next(i for i, line in enumerate(ended.printed) if line.startswith("[delta] warm-up control")) + 1:]
    return lines[:next((i for i, line in enumerate(lines) if line.startswith("[delta] weight registration")), len(lines))]


def as_json(doc: Any) -> Any:
    return json.loads(json.dumps(doc, default=str))


def of_record(governing: str | None, retain: str = "host", bounded: bool = False) -> Record:
    return {"governing": governing, "retain": retain, "bounded_staging": bounded, "rule": GOVERNS}


def handed(s: SimpleNamespace, name: str) -> tuple[tuple[Any, ...], Record]:
    """The one call of ``name``: what `main` handed it, by position and by keyword."""
    [(args, kw)] = [(a, k) for n, a, k in s.handed if n == name]
    return args, kw


def admission_doc(s: SimpleNamespace, source: str | None, *, governs: bool = True, override: bool = False) -> Record:
    """The host admission as `main` records it: what `admission` gave, with what `main` writes on it."""
    return {**s.gave["admission"], "override": override, SOURCE: source, "domain": DOMAIN, "governs_this_run": governs}


def admission_line(s: SimpleNamespace, limit: int | str) -> str:
    adm = s.gave["admission"]
    return ADMISSION_LINE.format(predicted=adm["predicted_host_bytes"] >> 20, limit=limit, verdict="ADMITTED" if adm["admitted"] else "REFUSED",
                                 why=adm["why"])


def test_a_staging_block_is_sized_and_its_run_admitted_under_the_containers_limit(tmp_path, host, capsys):
    """A committer that keeps a whole run's values on the host, unbounded: the envelope's admission governs, and admits.  The
    sizing's file is written before `main` reads the limit, so it is the sizing as the committer gave it."""
    HOST[V2] = f"{64 * GIB}\n"
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and com.calls == ["size_arena", "register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert com.sized == [{"envelope": ENVELOPE, "hidden_m1": False}]
    limited = {**SIZING, SIZING_SOURCE: V2, SIZING_LIMIT: 64 * GIB,
               "admission_of_record": of_record("f_da_15_host_retention_envelope")}
    assert [name for name, _, _ in host.handed] == ["committer_resident", "admission"]
    (wl, limit), planned = handed(host, "committer_resident")
    assert [(r["request_id"], r["prompt_token_ids"], r["max_tokens"]) for r in wl["requests"]] == [("a", [5, 6, 7], 3), ("b", [9, 2], 2)]
    assert wl["source"] == str(tmp_path / "workload.json") and limit == 64 * GIB and planned == {"pairs": 2, "replay_workers": 3}
    assert handed(host, "admission") == ((limited, wl["requests"], 64 * GIB), {
        "retain": "host", "max_num_batched_tokens": None, "lag": 1, "envelope_bound_bytes": 24 * MIB, "resident": host.gave["committer_resident"]})
    adm = host.gave["admission"]
    assert adm["admitted"] and adm["scheduled_raw_bytes"] == 21 * MIB and adm["predicted_host_bytes"] == adm["pool_bytes"] + adm["resident"]["bytes"]
    assert com.asz == {**limited, "admission": admission_doc(host, V2)}
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2)) and ended.doc(f"arena_sizing_{KIND}.json") == SIZING
    assert setup_lines(ended) == [SIZED_LINE, admission_line(host, 65536)]
    assert [n for n in ended.files() if KIND in n] == [f"admission_{KIND}.json", f"arena_sizing_{KIND}.json"]


@pytest.mark.parametrize(("files", "limit", "source"), [
    ({V2: f"{64 * GIB}\n"}, 64 * GIB, V2),
    ({V2: f"{64 * GIB}\n", V1: f"{32 * GIB}\n"}, 64 * GIB, V2),
    ({V2: "max\n", V1: f"{64 * GIB}\n"}, 64 * GIB, V1),
    ({V1: f"{64 * GIB}\n"}, 64 * GIB, V1),
    ({V2: "max\n", V1: f"{(1 << 63) - 4096}\n", MEMINFO: f"MemFree:  1 kB\nMemTotal:  {64 * GIB >> 10} kB\n"}, 64 * GIB, MEMTOTAL),
    ({MEMINFO: "MemFree:  1 kB\n"}, None, None),
    ({}, None, None),
], ids=["cgroup-v2", "v2-before-v1", "v2-unlimited-then-v1", "v1-alone", "no-cgroup-limit-then-the-machines-total", "a-meminfo-with-no-total",
        "no-file"])
def test_the_limit_is_the_containers_or_else_the_machines_memory_and_its_source_is_named(tmp_path, host, capsys, files, limit, source):
    """The limit is the first cgroup file that holds a number under 2**60, the newer hierarchy's before the older's, else
    `MemTotal`, else none; with none the run is admitted, there being nothing to hold it against."""
    HOST.update(files)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and (com.asz[SIZING_LIMIT], com.asz[SIZING_SOURCE]) == (limit, source)
    assert handed(host, "committer_resident")[0][1] == limit and handed(host, "admission")[0][2] == limit
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, source)) and host.gave["admission"]["admitted"]
    assert setup_lines(ended) == [SIZED_LINE, admission_line(host, 65536 if limit else "none")]
    assert (limit is None) == host.gave["admission"]["why"].startswith("no container memory limit visible")


WARNING_LINE = ("[delta] WARNING host retention bound 30 MiB (raw x1.25 for levels) exceeds the container memory limit 16 MiB: expect a silent "
                "OOM-kill (rc 137) during the instrumented run; use --retain none (openings by replay) or a larger container")
REFUSED_LINE = "[delta] REFUSE: host admission -- {why} (VERITY_ADMIT_OVER_BOUND=1 launches anyway, named on the record)"
OVERRIDDEN_LINE = "[delta] WARNING host admission OVERRIDDEN by VERITY_ADMIT_OVER_BOUND=1 (named on the record): the run may be OOM-killed"


@pytest.mark.parametrize(("limit", "warned"), [(16 * GIB, []), (30 * MIB, []), (16 * MIB, [WARNING_LINE])],
                         ids=["over-by-the-resident", "at-the-envelope", "over-by-the-envelope"])
def test_a_host_retaining_run_predicted_over_the_limit_is_refused_before_the_weights(tmp_path, host, capsys, limit, warned):
    """Exit 3 with the admission on file and nothing registered; a limit under the envelope's bound with its levels, 30 MiB,
    is warned of first, and a limit equal to it is not."""
    HOST[V2] = f"{limit}\n"
    ended = staged(tmp_path, capsys)
    [com] = made()
    why = host.gave["admission"]["why"]
    assert ended.exit == 3 and com.calls == ["size_arena"] and not host.gave["admission"]["admitted"]
    assert [name for name, _ in host.calls] == ["build_engine", "attach_of_record"]
    assert setup_lines(ended) == [SIZED_LINE, *warned, admission_line(host, limit >> 20), REFUSED_LINE.format(why=why)]
    assert ended.files() == [f"admission_{KIND}.json", f"arena_sizing_{KIND}.json", "versions.json"]
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2))


def test_an_override_launches_a_run_over_the_limit_and_is_named_on_its_admission(tmp_path, host, capsys, monkeypatch):
    HOST[V2] = f"{16 * MIB}\n"
    monkeypatch.setenv("VERITY_ADMIT_OVER_BOUND", "1")
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and com.calls == ["size_arena", "register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2, override=True)) and not host.gave["admission"]["admitted"]
    assert setup_lines(ended) == [SIZED_LINE, WARNING_LINE, admission_line(host, 16), OVERRIDDEN_LINE]


def test_a_run_that_keeps_no_values_on_the_host_is_recorded_over_the_limit_and_not_refused(tmp_path, host, capsys):
    """``retain`` other than host: no admission governs, and the envelope's is outside its refusal domain."""
    HOST[V2] = f"{16 * MIB}\n"
    M.MAKE.update(retain="none")
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    adm = host.gave["admission"]
    assert ended.exit is None and com.asz["admission_of_record"] == of_record(None, "none") and handed(host, "admission")[1]["retain"] == "none"
    assert adm["admitted"] and adm["would_exceed_by_bytes"] > 0 and adm["why"].startswith("retain=none: outside the refusal domain")
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2)) and setup_lines(ended) == [SIZED_LINE, admission_line(host, 16)]


NOT_COMPUTED = "admission not computed: KeyError('model')"
NOT_COMPUTED_REFUSED = ("[delta] REFUSE: host admission not computed on a --retain host run: KeyError('model') (VERITY_ADMIT_OVER_BOUND=1 launches "
                        "anyway, named on the record)")
NOT_COMPUTED_WARNED = "[delta] WARNING host admission not computed: KeyError('model') ({how})"


@pytest.mark.parametrize(("retain", "override", "admitted", "refused", "line"), [
    ("host", "", False, " -- refused: an uncomputed admission is not an admission (--retain host)", NOT_COMPUTED_REFUSED),
    ("host", "1", False, "", NOT_COMPUTED_WARNED.format(how="OVERRIDDEN by VERITY_ADMIT_OVER_BOUND=1, named on the record")),
    ("none", "", None, "", NOT_COMPUTED_WARNED.format(how="retain != host: nothing pinned for the run")),
], ids=["host-retaining", "host-retaining-overridden", "nothing-retained"])
def test_an_admission_that_cannot_be_computed_refuses_a_host_retaining_run(tmp_path, host, capsys, monkeypatch, retain, override, admitted,
                                                                             refused, line):
    """A sizing with no model: the schedule's bytes cannot be had.  The error is the admission's record; a run that keeps its
    values on the host exits 3 unless overridden, and any other goes on."""
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(retain=retain, sizing={k: v for k, v in SIZING.items() if k != "model"})
    monkeypatch.setenv("VERITY_ADMIT_OVER_BOUND", override)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert (ended.exit, com.calls[:2]) == ((3, ["size_arena"]) if refused else (None, ["size_arena", "register_weights"]))
    assert "admission" not in host.gave and setup_lines(ended) == [SIZED_LINE, line]
    assert ended.doc(f"admission_{KIND}.json") == com.asz["admission"] == {
        "admitted": admitted, "override": bool(override), SOURCE: V2, "why": NOT_COMPUTED + refused}


@pytest.mark.parametrize(("argv", "make", "sized"), [
    (["--arena-unchecked", "--arena-mb", "64"], {}, False), ([], {"arena": False}, False),
    (["--arena-unchecked"], {}, True), (["--arena-mb", "64"], {}, True),
], ids=["a-given-size-unchecked", "no-block", "unchecked-with-no-size", "a-given-size"])
def test_a_committer_with_no_block_or_a_block_left_unchecked_is_not_sized(tmp_path, host, capsys, argv, make, sized):
    """Not sized is not admitted either: nothing is computed, printed or written before the weights."""
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(make)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys, *argv)
    [com] = made()
    assert ended.exit is None and com.calls == [*(["size_arena"] if sized else []), "register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert bool(host.handed) == bool(setup_lines(ended)) == bool([n for n in ended.files() if KIND in n]) == sized


@pytest.mark.parametrize(("argv", "required", "hidden"), [([], None, False), (["--hidden-so", "tap.so"], None, True),
                                                          (["--hidden-so", "tap.so"], False, False)],
                         ids=["no-tap", "a-tap", "a-tap-the-manifest-does-not-ask-for"])
def test_the_sizing_counts_the_hidden_rows_of_a_tap_the_attachment_did_not_rule_out(tmp_path, host, capsys, monkeypatch, argv, required, hidden):
    HOST[V2] = f"{64 * GIB}\n"
    attach = commit.attach_of_record
    monkeypatch.setattr(commit, "attach_of_record", lambda *a, **kw: SimpleNamespace(**{**vars(attach(*a, **kw)), "hidden_required": required}))
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys, *argv)
    assert ended.exit is None and made()[0].sized == [{"envelope": ENVELOPE, "hidden_m1": hidden}]


def test_the_engines_token_budget_and_context_reach_the_sizing_and_the_budget_the_admission(tmp_path, host, capsys):
    """A prefill of five tokens under a budget of four is two chunks: one more step's bytes than `ADMISSION_LINE` counts."""
    HOST[V2] = f"{64 * GIB}\n"
    host.engine.kwargs.update(max_num_batched_tokens=4, max_model_len=32)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    assert ended.exit is None
    assert made()[0].sized == [{"envelope": {**ENVELOPE, "max_num_batched_tokens": 4, "max_model_len": 32}, "hidden_m1": False}]
    assert handed(host, "admission")[1]["max_num_batched_tokens"] == 4 and host.gave["admission"]["scheduled_raw_bytes"] == 22 * MIB


def test_a_windowed_committer_is_handed_the_sizing_for_its_ring_after_the_admission(tmp_path, host, capsys):
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(windowed=True)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and com.calls[:3] == ["size_arena", "enable_windows", "register_weights"]
    assert com.ring_of is com.asz and "admission" in com.asz and ended.doc(f"window_ring_{KIND}.json") == RING
    assert setup_lines(ended) == [SIZED_LINE, admission_line(host, 65536), RING_LINE]


STREAM_LINE = ("[delta] stream of record (declared schedule, 3 steps, 2 requests): run 16 MiB (hidden 0 MiB), largest step 8 MiB (5 tok); envelope "
               "any-packing bound 24 MiB; raw bound 32 MiB = hidden 0 x 1.05 + other 16 x 2.0 (factors measured on B1)")
#: The bound's line: the engine is `RSS`, the levels a quarter of the raw bound, the learning budget and the exact pool `Staged`'s.
BOUND_LINE = (f"[delta] bounded staging HOST BOUND ({KIND}): {{bound}} MiB = max(phases) + pairs 0 ({{pairs}} pairs) + slack 6144; phases: run 46 = "
              "engine 2 + levels 8 + raw kept 32 (bound 32, excluded hidden 0) + learn 4 + pool_rounding 0 (pool exact) | {phases} MiB; "
              "cgroup {limit}")
PHASES = ("c2 8238 = run + c2_programs 0 + fork_children 8192 (cgroup scope) | replay 58 = run + replay 12 + c2_residual 0 | later_pair_replay 20538 "
          "= replay + pool_carry 20480 -> peak phase later_pair_replay")
#: One pair and no sampled replay: no children, no pool carried into a later pair, and the forecast's line with no forecast.
PHASES_ONE_PAIR = ("c2 46 = run + c2_programs 0 + fork_children 0 (cgroup scope) | replay 58 = run + replay 12 + c2_residual 0 -> peak phase "
                   "openings_after_release")
NO_REPLAY: Record = {"bytes": 0, "rows": 0, "json_bytes": 0, "files": [], "why": "no sampled replay (--replay-per-stratum 0)"}
NO_REPLAY_LINE = ("[delta] C2 PROGRAMS FORECAST: 0 instance rows in 0 files, json 0 MiB -> 0 MiB of Python objects (xNone, None) -- no sampled "
                  "replay (--replay-per-stratum 0)")
FORECAST: Record = {"bytes": 100 * MIB, "rows": 7, "json_bytes": 3 * MIB, "files": ["instances.json.gz"], "factor": 15.0, "calibration": "stand-in"}
FORECAST_LINE = "[delta] C2 PROGRAMS FORECAST: 7 instance rows in 1 files, json 3 MiB -> 100 MiB of Python objects (x15.0, stand-in)"
MARGIN_NO_LIMIT = "[delta] REFUSE (--host-admit): no cgroup memory limit readable to admit against"
MARGIN_OVER = ("[delta] REFUSE (--host-admit 0.1): host peak bound 26682 MiB > 0.90 x cgroup 28672 MiB -- predicted OOM, not launched; dominant "
               "contributor: {text}")
NOT_GOVERNING_LINE = ("[delta] NOTE host-retention envelope admission not met ({why}) -- recorded, not governing: this is a --bounded-staging run "
                      "(the bounded HOST BOUND above governs)")
UNDER_64_GIB, UNDER_28_GIB = "65536 MiB; margin 38853 MiB (59.3 %)", "28672 MiB; margin 1989 MiB (6.9 %)"
UNDER_NO_LIMIT = "None MiB; margin None MiB (None %)"
LEARNED_LINE = (f"[delta] bounded staging warm-up ({KIND}): learned-only steps 3 (2 MiB discarded), transient peak 3 MiB (budget 4 MiB), "
                "plan mismatches 0, plans cached 3, pinned pool 5 MiB")
DECLARED = ("model", "requests", "steps", "horizon", "prefill_steps", "hidden_prefill_bytes", "decode_step_bytes_max")


def bound_kw(**more: Any) -> Record:
    """What `main` hands the host bound for `Staged` under 64 GiB with two pairs, four openings and a sampled replay."""
    return {"engine_rss_bytes": M.RSS[0] * 1024, "retain": "host", "retain_exclude": set(), "c2_programs_bytes": 0, "c2_freed_before_replay": True,
            "fork_children_bytes": commit.FORK_CHILDREN_BYTES, "c2_residual_bytes": 0, "pool_carry_bytes": commit.POOL_CARRY_BYTES,
            "pool_exact": True, "replay_cache": False, "learn_budget_bytes": 4 * MIB, "openings": 4, "chunk_bytes": 256, "window_bytes": 2 * MIB,
            "bounded": True, "open_after_release": True, LIMIT: 64 * GIB, "hidden_factor": 1.05, "other_factor": 2.0,
            "slack_bytes": 6 * GIB, "pairs": 2, **more}


def bound_doc(s: SimpleNamespace, source: str | None, at: float) -> Record:
    """The host bound as `main` records it: what `bounded_host_bound` gave, with the schedule it came from, the limit's source,
    that it governs and when it was printed, ``at`` on `commit`'s clock."""
    stream = s.gave["schedule_stream_bound"]
    return {**s.gave["bounded_host_bound"], "envelope_retained_bytes_bound": 24 * MIB, "schedule_run_bytes": 16 * MIB, "schedule_hidden_bytes": 0,
            "schedule_largest_step": stream["largest_step"], "prefill_tokens": 5, "per_token_bytes": MIB,
            "declared_schedule": {k: stream[k] for k in DECLARED}, "cgroup_path": source, "governs_this_run": True,
            "admission_of_record": of_record("bounded_host_admission", bounded=True), "admission_utc": M.UTC, "admission_monotonic_s": at}


def dominant_line(s: SimpleNamespace) -> str:
    return f"[delta] bounded staging DOMINANT CONTRIBUTOR ({KIND}): {s.gave['bounded_host_bound']['dominant_contributor']['text']}"


def test_a_bounded_committers_host_bound_is_computed_before_its_weights_and_governs_its_run(tmp_path, host, capsys):
    """The stream of the declared schedule, the forecast of the replay's Programs and the bound, each handed the one before;
    the bound goes on the committer, on file and into the sizing's file, which is written again with the limit and the bound.
    The envelope's admission is computed after it and does not govern.  The warm-up's learned plans are on file, and the
    verdict holds the process's peak against this bound."""
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(bounded=True)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and com.calls == ["size_arena", "register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert [name for name, _, _ in host.handed] == ["schedule_stream_bound", "c2_programs_forecast", "bounded_host_bound", "committer_resident",
                                                    "admission"]
    limited = {**SIZING, SIZING_SOURCE: V2, SIZING_LIMIT: 64 * GIB,
               "admission_of_record": of_record("bounded_host_admission", bounded=True)}
    wl = handed(host, "committer_resident")[0][0]
    stream, forecast = host.gave["schedule_stream_bound"], host.gave["c2_programs_forecast"]
    assert handed(host, "schedule_stream_bound") == ((limited, wl["requests"], wl["sampling"]), {}) and stream["run_bytes"] == 16 * MIB
    assert handed(host, "c2_programs_forecast") == ((None, None), {}) and forecast["bytes"] == 0
    assert handed(host, "bounded_host_bound") == ((stream,), bound_kw())
    bound = bound_doc(host, V2, 3.0)
    bounded = {**limited, "schedule_stream_bound": stream, "c2_programs_forecast": forecast, "bounded_host_admission": bound}
    assert com.host_admission == bound and ended.doc(f"host_admission_{KIND}.json") == as_json(bound)
    assert ended.doc(f"arena_sizing_{KIND}.json") == as_json(bounded)
    assert handed(host, "admission")[0] == (bounded, wl["requests"], 64 * GIB)
    assert com.asz == {**bounded, "admission": admission_doc(host, V2, governs=False)}
    assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2, governs=False))
    said = setup_lines(ended)
    assert said[1].startswith("[delta] C2 PROGRAMS FORECAST: 0 instance rows in 0 files, json 0 MiB -> 0 MiB of Python objects (x")
    assert said[1].endswith(f" -- {forecast['why']}") and bound["host_peak_bound_bytes"] >> 20 == 26682
    assert [said[0], *said[2:]] == [SIZED_LINE, STREAM_LINE, BOUND_LINE.format(bound=26682, pairs=2, phases=PHASES, limit=UNDER_64_GIB),
                                    dominant_line(host), admission_line(host, 65536)]
    assert ended.doc(f"bounded_warmup_{KIND}.json") == {
        **LEARNED["bounded_staging"], "pinned_pool_bytes": 5 * MIB, "plans_cached": 3, "host_mem_kb": {"rss": M.RSS[0], "hwm": M.RSS[1]},
        "cgroup_memory_current": {"current": None, "peak": None, "max": 64 * GIB}, "errors": []}
    assert LEARNED_LINE in ended.printed and ended.printed[ended.printed.index(LEARNED_LINE) + 1].startswith("[delta] warm-up instrumented")
    peak = ended.doc("verdict.json")["host_peak"]
    assert (peak["admission"], peak["host_peak_bound_bytes"], peak["governs_this_run"], peak["printed_before_run"]) == (
        f"host_admission_{KIND}.json", bound["host_peak_bound_bytes"], True, True)
    assert [n for n in ended.files() if KIND in n] == [f"admission_{KIND}.json", f"arena_sizing_{KIND}.json", f"bounded_warmup_{KIND}.json",
                                                       f"host_admission_{KIND}.json"]


def test_a_bounded_run_of_one_pair_with_no_sampled_replay_counts_no_forecast_no_children_and_no_carried_pool(tmp_path, host, capsys):
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(bounded=True)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys, "--pairs", "1", "--replay-per-stratum", "0", "--replay-cache")
    [com] = made()
    assert ended.exit is None and [name for name, _, _ in host.handed] == ["schedule_stream_bound", "bounded_host_bound", "committer_resident",
                                                                           "admission"]
    assert handed(host, "bounded_host_bound")[1] == bound_kw(fork_children_bytes=0, pool_carry_bytes=0, replay_cache=True, pairs=1)
    assert handed(host, "committer_resident")[1] == {"pairs": 1, "replay_workers": 3}
    assert com.asz["c2_programs_forecast"] == NO_REPLAY
    assert setup_lines(ended)[1:4] == [NO_REPLAY_LINE, STREAM_LINE, BOUND_LINE.format(bound=6202, pairs=1, phases=PHASES_ONE_PAIR,
                                                                                      limit="65536 MiB; margin 59333 MiB (90.5 %)")]


def test_the_forecast_of_the_replays_programs_is_a_term_of_the_bound_and_a_share_of_it_stays_after_the_replay(tmp_path, host, capsys, monkeypatch):
    """A forecast of 100 MiB: the bound is handed it and `C2_RESIDUAL_FRACTION` of it, and the forecast's line is its record's."""
    HOST[V2] = f"{64 * GIB}\n"
    M.MAKE.update(bounded=True)
    monkeypatch.setattr(commit, "c2_programs_forecast", lambda root, one: dict(FORECAST))
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    assert ended.exit is None and made()[0].asz["c2_programs_forecast"] == FORECAST
    assert handed(host, "bounded_host_bound")[1] == bound_kw(c2_programs_bytes=100 * MIB, c2_residual_bytes=45 * MIB)
    assert setup_lines(ended)[1] == FORECAST_LINE


def test_a_bounded_run_over_the_limit_with_no_margin_asked_is_launched_and_not_warned_of_the_envelope(tmp_path, host, capsys):
    """The envelope's warning is a host-retaining unbounded run's.  A bounded run under a limit below that bound gets its own
    bound, over the limit and on file, and the note that the envelope's admission is not met; nothing refuses it."""
    HOST[V2] = f"{16 * MIB}\n"
    M.MAKE.update(bounded=True)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys)
    [com] = made()
    said = setup_lines(ended)
    assert ended.exit is None and com.calls[:2] == ["size_arena", "register_weights"] and com.host_admission["margin_bytes"] < 0
    assert WARNING_LINE not in said and said[0] == SIZED_LINE and said[2] == STREAM_LINE
    assert said[5:] == [admission_line(host, 16), NOT_GOVERNING_LINE.format(why=host.gave["admission"]["why"][:160])]


@pytest.mark.parametrize(("limit", "margin", "exclude", "refused"), [
    (None, "0.1", (), MARGIN_NO_LIMIT), (28 * GIB, "0.1", ("fa3_hidden_m1",), MARGIN_OVER), (28 * GIB, "0.05", (), None),
], ids=["no-limit", "over-the-margin", "within-the-margin"])
def test_a_bounded_run_to_be_admitted_by_a_margin_is_refused_over_it_or_with_no_limit(tmp_path, host, capsys, limit, margin, exclude, refused):
    """``--host-admit``: exit 3 after the bound is on file and before the envelope's admission.  Within the margin the run is
    launched though the envelope's admission, which counts the whole run on the host, is not met: that is noted, and recorded
    as not governing."""
    if limit:
        HOST[V2] = f"{limit}\n"
    M.MAKE.update(bounded=True, exclude=exclude)
    host.verdicts = [(True, None)]
    ended = staged(tmp_path, capsys, "--host-admit", margin)
    [com] = made()
    source = V2 if limit else None
    bound = bound_doc(host, source, 3.0)
    assert handed(host, "bounded_host_bound")[1] == bound_kw(**{LIMIT: limit}, retain_exclude=set(exclude))
    assert com.host_admission == bound and ended.doc(f"host_admission_{KIND}.json") == as_json(bound)
    assert ended.doc(f"arena_sizing_{KIND}.json")["bounded_host_admission"] == as_json(bound)
    said = setup_lines(ended)
    assert said[3] == BOUND_LINE.format(bound=26682, pairs=2, phases=PHASES, limit=UNDER_28_GIB if limit else UNDER_NO_LIMIT)
    assert said[4] == dominant_line(host)
    if refused:
        assert ended.exit == 3 and com.calls == ["size_arena"] and "admission" not in host.gave
        assert said[5:] == [refused.format(text=bound["dominant_contributor"]["text"])]
        assert ended.files() == [f"arena_sizing_{KIND}.json", f"host_admission_{KIND}.json", "versions.json"]
    else:
        why = host.gave["admission"]["why"]
        assert ended.exit is None and com.calls[:2] == ["size_arena", "register_weights"] and not host.gave["admission"]["admitted"]
        assert ended.doc(f"admission_{KIND}.json") == as_json(admission_doc(host, V2, governs=False))
        assert said[5:] == [admission_line(host, 28672), NOT_GOVERNING_LINE.format(why=why[:160])]
