"""`verity_vllm.pipeline.commit`: `main` whole on the CPU, from its options to the way it ends.

Real: the options (`cli.parse`), the workload record's reading, the stop rule of record, the serving loop (`run_workload`) over
a stand-in engine whose model is a real module, the plugin committer's making (`make_committer`), the run's coins, the
challenge and its openings, the rows, the summary and the verdict's writing.  The committer is a stand-in with a forward hook
on that model while it is installed; it commits what the hook saw by the host committer's offline path
(`NativeHostCommitter.commit_offline`), so its roots and openings are real ones, and reports measurements of its own choosing.

Stood in, each because it needs a GPU, vLLM, the run store or a Program of record: the source identity precheck, the engine's
build and its version facts, vLLM's sampling parameters, the two `torch.cuda` calls that raise with no device, the sources'
attachment (`commit_attach.attach_of_record` refuses a Commit with no required-value manifest), the binding record and, where
a test says so, a run's verdict.  This machine's memory and the command line are stood in too, and `commit`'s clock is one the
stand-ins move (`Clock`), so every number in a row is one the test states.

Held: the files `main` writes, `verdict.json`, `runs.jsonl`, `steps.jsonl` and `summary.json` whole (a run's root and its coins
against what the committer and `commit_coins` gave), what each stand-in was handed, the last line printed, and how `main`
ends: by returning, by exit 3, or by a refusal's sentence.  CPU only."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit.step_rows import program_field
from verity_vllm.engine import served as SV
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import cli, commit, source_identity
from verity_vllm.pipeline import commit_coins as CC

Record = dict[str, Any]

VOCAB = 16
#: The stand-in engine's Program digest as the roots bind it; `commit` reads a served Build's through the same function.
PROGRAM = program_field(hashlib.sha512(b"the stand-in engine's Program").hexdigest())
PLUGIN = f"plugin:{__name__}:committer"
PLUGIN_2 = f"plugin:{__name__}:committer_2"
#: Two requests of unequal `max_tokens`: three engine steps a run, the last with one request.
WORKLOAD: Record = {"schema": "verity-vllm/workload/v2", "model": "org/model", "variants": {}, "engine": {},
                    "requests": [{"request_id": "a", "prompt_token_ids": [5, 6, 7], "max_tokens": 3},
                                 {"request_id": "b", "prompt_token_ids": [9, 2], "max_tokens": 2}]}
WORKLOAD_SHA256 = hashlib.sha256(json.dumps(WORKLOAD, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
#: What the stand-in engine serves for `WORKLOAD`: after token t comes (5 t + 3) mod `VOCAB`.
TOKENS = {"a": [6, 1, 8], "b": [13, 4]}
TOKENS_DIGEST = hashlib.sha256(json.dumps(TOKENS, separators=(",", ":")).encode()).hexdigest()
#: A run's value bytes by step: sixteen float32 scores a live request.
STEP_BYTES = (128, 128, 64)
CLASSES = ("input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights")
CLASSES_SOURCE = "default for FANone (REL-02 interim fallback: counts+classes, not the Program/query manifest)"
SOURCE_IDENTITY: Record = {"packages": {"verity_vllm": {"git_sha": "0" * 40}}, "requested_root": "/tree"}
SERVED: Record = {"checked": "the stand-in engine"}
BINDING: Record = {"rules_version": "stand-in", "map_entries_n": 3, "map_digest": "m" * 64, "population": {"n": 3, "digest": "p" * 64},
                   "coverage": {"ok": True, "bound_n": 3, "expected_n": 3, "missing_n": 0, "extra_n": 0, "missing": []},
                   "openings_population": {"n": 0, "verified": 0, "opened": 0, "unbound_n": 0, "late_step_in_sample": True},
                   "context": {"ok": True, "why": None}}
#: The stop rule of record as `versions.json` keeps it, and as the binding record is handed it.
STOP: Record = {"eos": None, "lag": 1, "declared_lag": 1, "engine_lag": 1, "async_scheduling": True, "notes": [], "eos_source": None}
STOP_RULE: Record = {"eos": None, "lag": 1, "declared_lag": 1, "engine_lag": 1, "async_scheduling": True, "notes": [],
                     "srule": {"eos_token_ids": None, "source": None}}
#: A run's spans as the stand-in committer reports them.
SPANS_TOTAL: Record = {"acquisition": 0.125, "movement": 0.25, "hashing": 1.5, "root_finalisation": 0.75, "opening_generation": 0.0, "wait": 0.5}
STEP_SPANS: Record = {"hashing": 0.5, "root_finalisation": 0.25}
RSS = (2048, 4096)
UTC = "1970-01-01T00:00:00Z"
ARGV = ["verity-vllm", "commit"]
#: The environment names `main` writes on these paths (`settings.put`, `settings.setdefault`, `staging_environ`) and the ones it
#: reads there: absent at a test's start, and taken back at its end.
ENVIRON = ("VERITY_WEIGHTS_HASH", "VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED",
           "VERITY_COLLECT_WATCHDOG_MARKER", "VERITY_FAULT", "VERITY_TRACEDUMP", "VERITY_SWITCH")
#: A committer's calls: the instrumented warm-up, and a measured run with its openings in the run.
WARMUP = ["install", "finalize", "uninstall", "reset"]
RUN = ["reset", "install", "finalize", "uninstall", "open 4", "coverage_declaration"]
PASS_LINE = "[delta] COMMIT PASS (fail-closed X-03): {n} instrumented runs, manifest == bound, 0 errors, openings verified, tokens equal"
FAIL_LINE = "[delta] COMMIT FAIL: {failed}/{n} instrumented runs failed the fail-closed check; first: pair {pair} {arm}: {why}"
VALUE_CORRESPONDENCE: Record = {
    "method": "offline-suite (tests/commit/test_binding_negatives.py: late-read, relabel)", "result": None, "components": {}, "why": None,
    "note": "coverage + openings cannot see a late read or a relabel (brief §4); live evidence = the Match oracle of this attempt / "
            "an independent pinned root / replay"}
#: `summary.md` of the paired Commit: the control's steps take a second and the instrumented arm's a quarter more.
TABLE = """\
### Δ_commit — org/model (B0), workload `workload.json`, committer `{committer}`, 2 interleaved pairs [measured]

| metric | control (mean) | instrumented (mean) | Δ (mean over pairs) | 95% CI (bootstrap over pairs) |
|---|---:|---:|---:|---:|
| end-to-end wall (ms) | 3000.00 | 3750.00 | **+25.00 %** (+750.00 ms) | [+25.00 %, +25.00 %] |
| decode step wall (ms, mean) | 1000.000 | 1250.000 | +25.00 % | [+25.00 %, +25.00 %] |
| prefill step wall (ms) | 1000.000 | 1250.000 | +25.00 % | [+25.00 %, +25.00 %] |
| generated tokens/s | 1.7 | 1.3 | -20.00 % | |
| peak GPU alloc (GB) | 0.000 | 0.000 | | |
| host RSS after run (MB) | 2 | 2 | | |

tokens identical across all runs/arms: **True** (BI=1)

| component (instrumented, per run, mean) | ms | bytes | note |
|---|---:|---:|---|
| acquisition | 125.00 |  | host wall of the component (hashing/root on the worker thread, overlapped) |
| movement | 250.00 |  | host wall of the component (hashing/root on the worker thread, overlapped) |
| hashing | 1500.00 | 320 | host wall of the component (hashing/root on the worker thread, overlapped) |
| root_finalisation | 750.00 |  | host wall of the component (hashing/root on the worker thread, overlapped) |
| opening_generation | 0.00 |  | host wall of the component (hashing/root on the worker thread, overlapped) |
| wait | 500.00 |  | host wall of the component (hashing/root on the worker thread, overlapped) |

root-ready delay (last step end → run root): mean 1500.00 ms, max 1500.00 ms; backlog max 0 steps; bytes bound per run 320 in 3 leaves \
(3 tensors, 3 steps); persistent 1,000 B; staging (pinned, in use) 2,000 B; values retained 320 B; openings 4/4 verified (4096 B mean); \
weight registration 0.25 s for 1,024 B (once, excluded)

steady state — mean hashing time per decode step (ms) by step index: 1:500.00, 2:500.00
growth second half / first half: +0.0%"""


class Clock:
    """`commit`'s `time`, which only the stand-ins move: an engine step takes a second, a hooked forward a quarter more, a
    run's finalize a half, a call for openings a quarter and each opening's check an eighth.  The date is the epoch's."""

    def __init__(self) -> None:
        self.now = 0.0

    def take(self, seconds: float) -> None:
        self.now += seconds

    def perf_counter(self) -> float:
        return self.now

    time = monotonic = perf_counter

    def gmtime(self) -> float:
        return self.now

    def strftime(self, fmt: str, t: float) -> str:
        return UTC


class Model(torch.nn.Module):
    """The served model: one table, whose column for a request's last token scores its next one."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(VOCAB, VOCAB, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            for t in range(VOCAB):
                self.head.weight[(5 * t + 3) % VOCAB, t] = 1.0

    def forward(self, last: torch.Tensor) -> torch.Tensor:
        return self.head(torch.nn.functional.one_hot(last, VOCAB).float())


class Core:
    """`llm.llm_engine`: a step gives every live request one token, the model's argmax for its last one, and ends a request
    at its `max_tokens`.  It has no `engine_core` and no `model_executor`, so `arrivals.ForwardCounter` counts every step a
    forward."""

    def __init__(self, model: Model, clock: Clock) -> None:
        self.model, self.clock = model, clock
        self.live: dict[str, tuple[list[int], list[int], int]] = {}
        self.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True), compilation_config=SimpleNamespace(mode="NONE", cudagraph_mode="NONE"),
            model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace(vocab_size=VOCAB)))

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def add_request(self, request_id: str, prompt: Record, params: Any) -> None:
        self.live[request_id] = (list(prompt["prompt_token_ids"]), [], params.max_tokens)

    def step(self) -> list[SimpleNamespace]:
        ids = list(self.live)
        scores = self.model(torch.tensor([(self.live[i][1] or self.live[i][0])[-1] for i in ids]))
        self.clock.take(1.0)
        outs = []
        for i, token in zip(ids, scores.argmax(-1).tolist()):
            _, tokens, most = self.live[i]
            tokens.append(int(token))
            done = len(tokens) >= most
            outs.append(SimpleNamespace(request_id=i, finished=done,
                                        outputs=[SimpleNamespace(token_ids=list(tokens), finish_reason="length" if done else None)]))
            if done:
                del self.live[i]
        return outs


def engine(clock: Clock) -> SimpleNamespace:
    """`build_engine`'s `Engine` as `main` reads it: eager, no prefix cache, a runner with nothing wrapped."""
    model = Model()
    return SimpleNamespace(build_s=0.5, case="B0", env_flags={"VLLM_BATCH_INVARIANT": "1"}, cp={"repo": "org/model", "local_path": None},
                           kwargs={"enforce_eager": True}, model=model, runner=SimpleNamespace(), clock=clock,
                           llm=SimpleNamespace(llm_engine=Core(model, clock)))


def weights_root(model: Model) -> str:
    return hashlib.sha512(model.head.weight.detach().numpy().tobytes()).hexdigest()


class Committer:
    """A committer's face as `main` reads it on a Commit with no arena, no bounded staging and no release: installed, a
    forward hook on the engine's model, each call one step's value; finalized, those steps committed by the host committer's
    offline path, which also opens and verifies.  ``leave`` keeps the hook at `uninstall`; ``forge`` flips a byte of the
    first opening's value; ``declare`` is the classes its coverage declaration names."""

    def __init__(self, eng: SimpleNamespace, run_id: str, *, name: str = "stand_in", leave: bool = False, forge: bool = False,
                 declare: tuple[str, ...] = ("instance_outputs_native",)) -> None:
        self.eng, self.run_id, self.name, self.leave, self.forge, self.declare = eng, run_id, name, leave, forge, declare
        self.host = NativeHostCommitter(None, run_id=run_id, hash_threads=1, program_digest=PROGRAM)
        self.retain = self.host.retain
        self.steps: list[list[tuple[str, bytes]]] = []
        self.calls: list[str] = []
        self.roots: list[str] = []
        self.hook: Any = None
        self.weights: Record | None = None

    def register_weights(self) -> Record:
        self.calls.append("register_weights")
        self.weights = {"root": weights_root(self.eng.model), "bytes": 4 * VOCAB * VOCAB, "wall_s": 0.25}
        return self.weights

    def _hooked(self, _module: Any, _args: Any, out: torch.Tensor) -> None:
        self.steps.append([("head.out", out.detach().numpy().tobytes())])
        self.eng.clock.take(0.25)

    def install(self) -> None:
        self.calls.append("install")
        self.hook = self.eng.model.register_forward_hook(self._hooked)

    def uninstall(self) -> None:
        self.calls.append("uninstall")
        if not self.leave:
            self.hook.remove()

    def reset(self) -> None:
        self.calls.append("reset")
        self.steps = []

    def finalize(self) -> Any:
        self.calls.append("finalize")
        run = self.host.commit_offline(self.steps)
        self.eng.clock.take(0.5)
        self.roots.append(run.run_root.hex())
        steps = [dataclasses.replace(s, t_step_end=10.0 * s.step, t_root_ready=10.0 * s.step + 2.0, backlog_at_step_end=0, **
                                     {"hashing_s": STEP_SPANS["hashing"], "root_finalisation_s": STEP_SPANS["root_finalisation"]})
                 for s in run.steps]
        spans = {"total_s": dict(SPANS_TOTAL), "calls": {"hashing": len(steps)}, "bytes": {"hashing": sum(s.n_bytes for s in steps)},
                 "per_step": {s.step: dict(STEP_SPANS) for s in steps}}
        extra = {"expected_tensors": len(steps), "committed_tensors": len(steps), "errors": [], "errors_n": 0,
                 "values_retained_bytes": sum(s.n_bytes for s in steps)}
        return dataclasses.replace(run, steps=steps, spans=spans, root_ready_delay_s=1.5, t_finalize_s=0.75, persistent_bytes=1000,
                                   temporary_bytes_peak=2000, weight_registration=self.weights, extra=extra)

    def open(self, sample: list[tuple[int, int]]) -> list[Any]:
        self.calls.append(f"open {len(sample)}")
        self.eng.clock.take(0.25)
        openings = self.host.open(sample)
        if self.forge:
            openings[0] = dataclasses.replace(openings[0], value=bytes([openings[0].value[0] ^ 1]) + openings[0].value[1:])
        return openings

    def verify(self, run: Any, opening: Any) -> bool:
        self.eng.clock.take(0.125)
        return self.host.verify(run, opening)

    def coverage_declaration(self, run: Any) -> Record:
        self.calls.append("coverage_declaration")
        return declaration(self.name, self.declare)

    def close(self) -> None:
        self.calls.append("close")


#: The committers the plugin factories made, in order; `stand_ins` empties it.
MADE: list[Committer] = []
#: What the next committers are made with (`Committer`'s keywords); `stand_ins` empties it.
MAKE: Record = {}


def committer(eng: SimpleNamespace, run_id: str) -> Committer:
    """``--committer plugin:<this module>:committer``."""
    MADE.append(Committer(eng, run_id, **MAKE))
    return MADE[-1]


def committer_2(eng: SimpleNamespace, run_id: str) -> Committer:
    """A second arm's (``--arms``), under its own name."""
    MADE.append(Committer(eng, run_id, name="stand_in_2", **MAKE))
    return MADE[-1]


@pytest.fixture
def stand_ins(monkeypatch, tmp_path):
    """The stand-ins of the module docstring.  ``calls`` keeps what `main` handed each: the engine's build, each committer's
    attachment, each binding record, each run's verdict.  ``verdicts`` is what the runs' verdicts return in order, the last
    one for every later run, or None for the real `commit_verdict`, whose answers ``answers`` keeps."""
    for name in ENVIRON:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    MADE.clear()
    MAKE.clear()
    clock = Clock()
    s = SimpleNamespace(calls=[], verdicts=None, answers=[], clock=clock, engine=engine(clock), environ=set(os.environ))
    real_verdict = commit.commit_verdict

    def build_engine(checkpoints, **kw):
        s.calls.append(("build_engine", {"checkpoints": checkpoints, **kw}))
        return s.engine

    def attach_of_record(com, eng, args, manifest, out, **kw):
        assert eng is s.engine and manifest is None
        s.calls.append(("attach_of_record", {"com": com, **kw}))
        return SimpleNamespace(required_classes=kw["required_classes"], moe_attached=False, hidden_required=None)

    def binding_record(com, rc, reqs, sampling, args, pair, **kw):
        s.calls.append(("binding_record", {"com": com, "pair": pair, **kw}))
        return copy.deepcopy(BINDING)

    def commit_verdict(record, tokens_equal, **kw):
        s.calls.append(("commit_verdict", {"tokens_equal": tokens_equal, **kw}))
        s.answers.append(real_verdict(record, tokens_equal, **kw) if s.verdicts is None else s.verdicts[min(len(s.answers), len(s.verdicts) - 1)])
        return s.answers[-1]

    for owner, name, value in [(source_identity, "precheck", lambda command, out=None: copy.deepcopy(SOURCE_IDENTITY)),
                               (va, "build_engine", build_engine), (va, "versions_doc", lambda eng: {"vllm": "stand-in"}),
                               (va, "sampling_params", lambda req, sampling: SimpleNamespace(max_tokens=int(req["max_tokens"]))),
                               (SV, "require", lambda llm, wl, target, out: dict(SERVED)),
                               (torch.cuda, "synchronize", lambda: None), (torch.cuda, "reset_peak_memory_stats", lambda: None),
                               (commit, "git_head", lambda: None), (commit, "nvidia_smi_used_mib", lambda: None), (commit, "rss_kb", lambda: RSS),
                               (commit, "attach_of_record", attach_of_record), (commit, "binding_record", binding_record),
                               (commit, "commit_verdict", commit_verdict), (commit, "time", clock), (sys, "argv", list(ARGV))]:
        monkeypatch.setattr(owner, name, value)
    return s


@dataclasses.dataclass
class Ended:
    """How a Commit ended: ``exit`` is None when `main` returned, else what it exited with."""
    exit: Any
    out: Path
    printed: list[str]

    def files(self) -> list[str]:
        return sorted(p.name for p in self.out.iterdir())

    def doc(self, name: str) -> Any:
        return json.loads((self.out / name).read_text())

    def rows(self, name: str = "runs.jsonl") -> list[Record]:
        return [json.loads(line) for line in (self.out / name).read_text().splitlines()]


def run(tmp_path: Path, capsys: Any, *argv: str) -> Ended:
    """`main` on `WORKLOAD` with two pairs, one warm-up, four openings a run, the values kept on the host and the plugin
    committer; ``argv`` after those, so it overrides them."""
    (tmp_path / "workload.json").write_text(json.dumps(WORKLOAD))
    args = cli.parse("commit", ["--workload", str(tmp_path / "workload.json"), "--out", str(tmp_path / "out"), "--pairs", "2", "--warmup", "1",
                                "--openings", "4", "--retain", "host", "--committer", PLUGIN, *argv], {})
    code = None
    try:
        assert commit.main(args) is None
    except SystemExit as e:
        assert e.code is not None
        code = e.code
    return Ended(code, tmp_path / "out", capsys.readouterr().out.splitlines())


def declaration(name: str, classes: tuple[str, ...] = ("instance_outputs_native",), run: Record | None = None) -> Record:
    doc: Record = {"committer": name, "classes": [{"class": c, "how": "bound-as-leaf", "tensors": 3, "bytes": 320} for c in classes]}
    return doc if run is None else {**doc, "run": run}


def run_of(tmp_path: Path) -> Record:
    return {"workload": str(tmp_path / "workload.json"), "case": "B0", "model": "org/model"}


def control_row(tmp_path: Path, pair: int, order: int, *, step_s: float = 1.0, **more: Any) -> Record:
    """A run's row less its `commit` record; a control run's steps take a second."""
    return {"schema": "verity/commit-integ/delta/v1", "label": "", "workload": str(tmp_path / "workload.json"), "workload_sha256": WORKLOAD_SHA256,
            "case": "B0", "model": "org/model", "pair": pair, "order_in_pair": order, "arm": "control", "committer": None, "committer_name": None,
            "wall_s": 3 * step_s, "iterations": 3, "engine_steps": 3, "generated_tokens": 5, "prompt_tokens": 5,
            "generated_tokens_per_s": 5 / (3 * step_s), "total_tokens_per_s": 10 / (3 * step_s), "step_wall_s": [step_s] * 3,
            "step_wall_prefill_s": step_s, "step_wall_decode_mean_s": step_s, "tokens_equal_ref": True, "tokens_digest": TOKENS_DIGEST,
            "gpu_mem": {"torch_max_allocated_gb": 0.0, "torch_reserved_gb": 0.0, "nvidia_smi_used_mib": None},
            "host_mem_kb": {"rss_before": RSS[0], "rss_after": RSS[0], "hwm": RSS[1]}, "utc": UTC, **more}


def commit_record(root: str, weights: str, *, classes: tuple[str, ...] = ("instance_outputs_native",), ok: bool = True,
                  why: str | None = None, **openings: Any) -> Record:
    """An instrumented row's `commit` record less its coins: four openings made and checked in the run."""
    return {"run_root": root, "steps": 3, "leaves": 3, "bytes_bound": 320, "tensors": 3, "spans_total_s": SPANS_TOTAL, "spans_calls": {"hashing": 3},
            "spans_bytes": {"hashing": 320}, "spans_throughput": None, "root_ready_delay_s": 1.5, "finalize_s": 0.75,
            "t_ready_minus_last_step_s": 0.5, "persistent_bytes": 1000, "temporary_bytes_peak": 2000,
            "extra": {"expected_tensors": 3, "committed_tensors": 3, "errors": [], "errors_n": 0, "values_retained_bytes": 320},
            "openings": {"n": 4, "verified": 4, "generate_s": 0.25, "verify_s": 0.5, "skipped": None, "deferred_replay_n": 0, "challenge_n": 4,
                         "deferred_positions": [], "deferred_digest": None, "bytes_mean": 4096.0, "failed": [], **openings},
            "weight_registration": {"root": weights, "bytes": 1024, "wall_s": 0.25},
            "classes": {c: {"how": "bound-as-leaf", "tensors": 3, "bytes": 320} for c in classes},
            "expected_tensors": 3, "committed_tensors": 3, "errors_n": 0, "binding": BINDING, "rules_version": "stand-in",
            "value_correspondence": VALUE_CORRESPONDENCE, "cublas_route_pin": None, "commit_pass": ok, "commit_fail_reason": why}


def instrumented_row(tmp_path: Path, pair: int, order: int, root: str, *, arm: str = "instrumented", kind: str = PLUGIN, name: str = "stand_in",
                     weights: str | None = None, **record: Any) -> Record:
    """An instrumented run's row: its steps take a quarter more than the control's."""
    return control_row(tmp_path, pair, order, step_s=1.25, arm=arm, committer=kind, committer_name=name, root_equal_expected=None,
                       commit=commit_record(root, weights_root(Model()) if weights is None else weights, **record))


def less_coins(rows: list[Record]) -> list[Record]:
    """`rows` without each instrumented run's coins, which are this process's own flips after the run's root."""
    for row in rows:
        if "commit" in row:
            coins = row["commit"]["extra"].pop("coins")
            assert coins["source"] == CC.IN_PROCESS and coins["root"] == row["commit"]["run_root"] and list(coins["keys"]) == list(CC.FAMILIES)
            assert all(len(bytes.fromhex(key)) == 64 for key in coins["keys"].values())
    return rows


def step_rows(pair: int, arm: str) -> list[Record]:
    if arm == "control":
        return [{"pair": pair, "arm": arm, "step": i, "wall_s": 1.0} for i in range(3)]
    return [{"pair": pair, "arm": arm, "step": i, "wall_s": 1.25, "n_leaves": 1, "n_bytes": n, "n_tensors": 1, "hashing_s": 0.5,
             "root_finalisation_s": 0.25, "root_ready_after_step_end_s": 2.0, "backlog_at_step_end": 0, "spans": STEP_SPANS}
            for i, n in enumerate(STEP_BYTES)]


def verdict_doc(roots: list[str], *, failed: int = 0, why: str | None = None, **more: Any) -> Record:
    return {"commit_pass": not failed, "runs_instrumented": len(roots), "runs_failed": failed, "first_fail_reason": why,
            "required_classes": list(CLASSES), "required_classes_source": CLASSES_SOURCE, "required_manifest": None,
            "moe_source": {"mode": "auto", "attached": False}, "negatives": None, "run_roots": roots, **more}


def versions_doc(ended: Ended, only_arm: str | None = None) -> Record:
    """`versions.json` as `main` writes it here; the compile settings are the engine's own record, held only where `main`
    writes into it."""
    compilation = ended.doc("versions.json")["compilation"]
    assert compilation["compiled_taps"] == "off" and compilation["enforce_eager"] is True
    return {"vllm": "stand-in", "served": SERVED, "git_head": None, "source_identity": SOURCE_IDENTITY, "argv": ARGV,
            "engine_kwargs": {"enforce_eager": True, "async_scheduling": True}, "enforce_eager": True, "compilation": compilation,
            "only_arm": only_arm, "weights_hash": "host", "stop_rule_of_record": STOP}


def paired_summary(tmp_path: Path, ended: Ended, kind: str = PLUGIN, name: str = "stand_in", *, reasons: tuple[str, ...] = ()) -> Record:
    table = TABLE.format(committer=kind)
    return {"schema": "verity/commit-integ/delta/v1", "workload": str(tmp_path / "workload.json"), "committer": kind, "pairs": 2,
            "delta_wall": [0.25, 0.25], "delta_wall_mean": 0.25, "delta_wall_ci95": [0.25, 0.25], "delta_decode": [0.25, 0.25],
            "delta_decode_ci95": [0.25, 0.25], "delta_prefill": [0.25, 0.25], "delta_prefill_ci95": [0.25, 0.25], "abs_wall_ms": [750.0, 750.0],
            "spans_mean_s": SPANS_TOTAL, "root_ready_delay_s": [1.5, 1.5], "steady_hashing_by_step_s": {"1": 0.5, "2": 0.5}, "steady_growth": 0.0,
            "backlog_max": 0, "tokens_equal": True, "versions": versions_doc(ended), "declaration": declaration(name, run=run_of(tmp_path)),
            "table_md": table, "commit_pass": not reasons, "commit_fail_reasons": list(reasons), "expected_tensors": [3, 3],
            "committed_tensors": [3, 3], "utc": UTC}


def test_a_paired_commit_that_passes(tmp_path, stand_ins, capsys):
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys)
    [com] = MADE
    roots = com.roots[1:]
    assert ended.exit is None and ended.printed[-1] == PASS_LINE.format(n=2) and ended.printed[-2] == f"[delta] wrote {ended.out}"
    assert ended.files() == ["declaration_stand_in.json", "runs.jsonl", "steps.jsonl", "summary.json", "summary.md", "verdict.json", "versions.json"]
    assert len(set(com.roots)) == 3 and com.calls == ["register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert ended.doc("verdict.json") == verdict_doc(roots)
    assert less_coins(ended.rows()) == [control_row(tmp_path, 0, 0), instrumented_row(tmp_path, 0, 1, roots[0]),
                                        instrumented_row(tmp_path, 1, 0, roots[1]), control_row(tmp_path, 1, 1)]
    assert ended.rows("steps.jsonl") == [*step_rows(0, "control"), *step_rows(0, "instrumented"), *step_rows(1, "instrumented"),
                                         *step_rows(1, "control")]
    assert ended.doc("versions.json") == versions_doc(ended)
    assert ended.doc("summary.json") == paired_summary(tmp_path, ended)
    assert (ended.out / "summary.md").read_text() == TABLE.format(committer=PLUGIN) + "\n"
    assert ended.doc("declaration_stand_in.json") == declaration("stand_in", run=run_of(tmp_path))
    said = declaration("stand_in", run=run_of(tmp_path))
    assert stand_ins.calls == [
        ("build_engine", {"checkpoints": "manifests/checkpoints.json", "max_num_seqs": 8, "case": "B0", "engine_args": {}, "target": None,
                          "execution": None}),
        ("attach_of_record", {"com": com, "kind": PLUGIN, "required_classes": CLASSES, "required_classes_cli": False, "hidden_family": None,
                              "fa_version": None, "cb_plan": None, "served": None}),
        ("binding_record", {"com": com, "pair": 0, "served_tokens": TOKENS, "stop": STOP_RULE, "finish_reasons": {"a": "length", "b": "length"}}),
        ("commit_verdict", {"tokens_equal": True, "min_openings": 4, "declaration": said, "required_classes": CLASSES}),
        ("binding_record", {"com": com, "pair": 1, "served_tokens": TOKENS, "stop": STOP_RULE, "finish_reasons": {"a": "length", "b": "length"}}),
        ("commit_verdict", {"tokens_equal": True, "min_openings": 4, "declaration": said, "required_classes": CLASSES})]
    assert {name for name in os.environ if name not in stand_ins.environ} <= set(ENVIRON), "main wrote a name the fixture does not take back"


def test_only_the_control_arm(tmp_path, stand_ins, capsys):
    ended = run(tmp_path, capsys, "--only-arm", "control")
    assert ended.exit is None and ended.printed[-1] == f"[delta] wrote {ended.out} (control arm only)"
    assert MADE == [] and [name for name, _ in stand_ins.calls] == ["build_engine"]
    assert ended.files() == ["runs.jsonl", "steps.jsonl", "summary.json", "summary.md", "verdict.json", "versions.json"]
    versions = versions_doc(ended, "control")
    assert ended.doc("versions.json") == versions
    assert ended.doc("verdict.json") == {"only_arm": "control", "runs": 2, "compilation": versions["compilation"]}
    assert ended.rows() == [control_row(tmp_path, 0, 0), control_row(tmp_path, 1, 0)]
    assert ended.rows("steps.jsonl") == [*step_rows(0, "control"), *step_rows(1, "control")]
    table = "only-arm control: 2 runs, wall [3.0, 3.0]"
    assert ended.doc("summary.json") == {"schema": "verity/commit-integ/delta/v1/only-arm", "only_arm": "control", "runs": 2, "declaration": None,
                                         "wall_s": [3.0, 3.0], "versions": versions, "table_md": table, "commit_pass": None, "utc": UTC}
    assert (ended.out / "summary.md").read_text() == table + "\n"


def test_only_the_instrumented_arm(tmp_path, stand_ins, capsys):
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--only-arm", "instrumented")
    [com] = MADE
    roots = com.roots[1:]
    assert ended.exit is None and ended.printed[-1] == PASS_LINE.format(n=2)
    assert com.calls == ["register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert ended.doc("verdict.json") == verdict_doc(roots)
    assert less_coins(ended.rows()) == [instrumented_row(tmp_path, 0, 0, roots[0]), instrumented_row(tmp_path, 1, 0, roots[1])]
    table = "only-arm instrumented: 2 runs, wall [3.75, 3.75]"
    assert ended.doc("summary.json") == {"schema": "verity/commit-integ/delta/v1/only-arm", "only_arm": "instrumented", "runs": 2,
                                         "declaration": declaration("stand_in", run=run_of(tmp_path)), "wall_s": [3.75, 3.75],
                                         "versions": versions_doc(ended, "instrumented"), "table_md": table, "commit_pass": True, "utc": UTC}


def test_two_arms_against_one_control(tmp_path, stand_ins, capsys):
    """``--arms``: each round runs the control and every arm, the order turned by one a round; a row's arm is its committer's
    kind.  Two commitments to equal values have unequal roots, each hiding under its own keys."""
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--arms", f"{PLUGIN},{PLUGIN_2}")
    one, two = MADE
    assert ended.exit is None and ended.printed[-1] == PASS_LINE.format(n=4)
    assert [c.calls for c in MADE] == [["register_weights", *WARMUP, *RUN, *RUN, "close"]] * 2
    assert [kw["kind"] for name, kw in stand_ins.calls if name == "attach_of_record"] == [PLUGIN, PLUGIN_2]
    assert ended.files() == ["declaration_stand_in.json", "declaration_stand_in_2.json", "runs.jsonl", "steps.jsonl", "summary.json", "summary.md",
                             "verdict.json", "versions.json"]
    assert ended.doc("verdict.json") == verdict_doc([one.roots[1], two.roots[1], one.roots[2], two.roots[2]])
    first = {"arm": PLUGIN, "kind": PLUGIN}
    second = {"arm": PLUGIN_2, "kind": PLUGIN_2, "name": "stand_in_2"}
    assert less_coins(ended.rows()) == [
        control_row(tmp_path, 0, 0), instrumented_row(tmp_path, 0, 1, one.roots[1], **first), instrumented_row(tmp_path, 0, 2, two.roots[1], **second),
        instrumented_row(tmp_path, 1, 0, one.roots[2], **first), instrumented_row(tmp_path, 1, 1, two.roots[2], **second), control_row(tmp_path, 1, 2)]
    summary = ended.doc("summary.json")
    assert summary.pop("table_md") + "\n" == (ended.out / "summary.md").read_text()
    assert summary == {"schema": "verity/commit-integ/delta/v1/multi-arm", "arms": [PLUGIN, PLUGIN_2], "rounds": 2,
                       "per_arm": {PLUGIN: paired_summary(tmp_path, ended), PLUGIN_2: paired_summary(tmp_path, ended, PLUGIN_2, "stand_in_2")},
                       "roots_by_round": {"0": {PLUGIN: one.roots[1], PLUGIN_2: two.roots[1]}, "1": {PLUGIN: one.roots[2], PLUGIN_2: two.roots[2]}},
                       "roots_equal_across_arms": False, "versions": versions_doc(ended), "utc": UTC}


@pytest.mark.parametrize(("make", "argv", "why", "openings"), [
    ({}, (), "required class 'input_token_ids' not declared", {}),
    ({"declare": CLASSES, "forge": True}, (), "openings verified 3 != checked 4", {"verified": 3}),
    ({"declare": CLASSES}, ("--retain", "none"), "openings skipped (retain=none: values by replay (openings_after_release))",
     {"n": 0, "verified": 0, "generate_s": 0.0, "verify_s": 0.0, "bytes_mean": None, "skipped": "retain=none: values by replay (openings_after_release)"}),
    ({"declare": CLASSES}, (), "C2 of record not established", {}),
], ids=["a class not declared", "a forged opening", "no openings in the run", "no C2"])
def test_a_commit_whose_runs_fail_their_verdict_exits_3(tmp_path, stand_ins, capsys, make, argv, why, openings):
    """The real `commit_verdict`, which no run of this engine passes: every file is still written, each run's reason is its
    row's, and the first one is the verdict's and the last line's."""
    MAKE.update(make)
    ended = run(tmp_path, capsys, *argv)
    [com] = MADE
    roots = com.roots[1:]
    assert stand_ins.answers[0][0] is False and stand_ins.answers[0] == stand_ins.answers[1]
    reason = stand_ins.answers[0][1]
    assert ended.exit == 3 and reason.startswith(why)
    assert ended.printed[-1] == FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=reason)
    assert ended.files() == ["declaration_stand_in.json", "runs.jsonl", "steps.jsonl", "summary.json", "summary.md", "verdict.json", "versions.json"]
    assert ended.doc("verdict.json") == verdict_doc(roots, failed=2, why=reason)
    assert ("open 4" in com.calls) == (argv != ("--retain", "none"))
    rows = less_coins(ended.rows())
    failed = [row["commit"]["openings"].pop("failed") for row in rows if "commit" in row]
    assert [[sorted(f) for f in fs] for fs in failed] == [[["identity", "index", "step"]] * bool(make.get("forge"))] * 2
    classes = make.get("declare", ("instance_outputs_native",))
    record = {"classes": classes, "ok": False, "why": reason, **openings}
    expected = [control_row(tmp_path, 0, 0), instrumented_row(tmp_path, 0, 1, roots[0], **record), instrumented_row(tmp_path, 1, 0, roots[1], **record),
                control_row(tmp_path, 1, 1)]
    for row in expected:
        if "commit" in row:
            del row["commit"]["openings"]["failed"]
    assert rows == expected
    summary = ended.doc("summary.json")
    assert (summary["commit_pass"], summary["commit_fail_reasons"]) == (False, [reason, reason])


def test_the_first_run_that_fails_is_the_one_named(tmp_path, stand_ins, capsys):
    stand_ins.verdicts = [(True, None), (False, "the second run's reason")]
    ended = run(tmp_path, capsys)
    roots = MADE[0].roots[1:]
    assert ended.exit == 3
    assert ended.printed[-1] == FAIL_LINE.format(failed=1, n=2, pair=1, arm="instrumented", why="the second run's reason")
    assert ended.doc("verdict.json") == verdict_doc(roots, failed=1, why="the second run's reason")
    assert less_coins(ended.rows())[1:3] == [instrumented_row(tmp_path, 0, 1, roots[0]),
                                             instrumented_row(tmp_path, 1, 0, roots[1], ok=False, why="the second run's reason")]


def test_a_committer_that_leaves_its_hook_stops_the_commit(tmp_path, stand_ins, capsys):
    """`assert_pristine` after the instrumented warm-up: no measured run starts on an engine that still carries a hook."""
    MAKE["leave"] = True
    with pytest.raises(AssertionError, match="forward hooks present: 1 != baseline 0"):
        run(tmp_path, capsys)
    assert MADE[0].calls == ["register_weights", *WARMUP]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["versions.json"]


@pytest.mark.parametrize(("argv", "why"), [
    (("--compiled-taps", "served"), "--compiled-taps served is the treatment process (one committer) and needs --required-manifest"),
    (("--compiled-taps", "served", "--only-arm", "control"), "--compiled-taps served is the treatment process (one committer)"),
    (("--compiled-taps", "served", "--arms", "a,b"), "--compiled-taps served is the treatment process (one committer)"),
    (("--challenger-gate", "gate"), "--challenger-gate and --challenger-peer go together"),
    (("--challenger-peer", "peer"), "--challenger-gate and --challenger-peer go together"),
])
def test_a_refusal_before_the_engine(tmp_path, stand_ins, capsys, argv, why):
    ended = run(tmp_path, capsys, *argv)
    assert isinstance(ended.exit, str) and ended.exit.startswith(why)
    assert stand_ins.calls == [] and MADE == [] and ended.printed == ["[delta] source identity OK: verity_vllm=000000000 tree=/tree"]


def test_a_flipped_weight_is_registered_and_named(tmp_path, stand_ins, capsys):
    """``--flip-weight``: the element changes before the weights' registration, and the verdict names the negative."""
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--flip-weight", "head.weight:0")
    [com] = MADE
    flipped = weights_root(stand_ins.engine.model)
    assert flipped != weights_root(Model()) and float(stand_ins.engine.model.head.weight[0, 0]) == 1.0
    assert ended.exit is None and ended.doc("verdict.json") == verdict_doc(com.roots[1:], negatives={"flip_weight": {
        "param": "head.weight", "index": 0, "before": 0.0, "after": 1.0, "numel": 256, "expect": "weights pin FAIL naming the field; Commit FAIL rc 3"}})
    assert less_coins(ended.rows())[1] == instrumented_row(tmp_path, 0, 1, com.roots[1], weights=flipped)


@pytest.mark.parametrize(("flip", "why"), [
    ("head.weight", "--flip-weight <served param name>:<flat index>[:<value>]"),
    ("nope:0", "--flip-weight names no served parameter 'nope'"),
    ("head.weight:0:0.0", "--flip-weight did not change the element (0.0 -> 0.0)"),
])
def test_a_flip_that_changes_nothing_is_refused(tmp_path, stand_ins, capsys, flip, why):
    ended = run(tmp_path, capsys, "--flip-weight", flip)
    assert isinstance(ended.exit, str) and ended.exit.startswith(why)
    assert [name for name, _ in stand_ins.calls] == ["build_engine", "attach_of_record"] and MADE[0].calls == []


def test_a_root_other_than_the_expected_one(tmp_path, stand_ins, capsys):
    """``--expect-root``: each instrumented row says whether its root is the expected one, and so does its value
    correspondence."""
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--expect-root", "ab" * 64)
    rows = [row for row in ended.rows() if row["arm"] == "instrumented"]
    assert [row["root_equal_expected"] for row in rows] == [False, False]
    assert [row["commit"]["value_correspondence"]["result"] for row in rows] == [False, False]
    assert [f"[delta] ROOT MISMATCH: run root {root} != expected {'ab' * 64}" in ended.printed for root in MADE[0].roots[1:]] == [True, True]


NOT_RUN = "oracle compare not run: "


def oracle_compares(ended: Ended) -> list[Record]:
    """Each instrumented row's oracle-compare record; a control row has no commit record, so none."""
    assert [row.get("commit") for row in ended.rows() if row["arm"] == "control"] == [None, None]
    return [row["commit"]["oracle_compare"] for row in ended.rows() if row["arm"] == "instrumented"]


def test_an_oracle_that_cannot_be_read_is_recorded_as_not_checked(tmp_path, stand_ins, capsys):
    """``--match-dir`` on a directory that holds no Match record: each instrumented row records an oracle compare that did not
    run, under the comparison's own schema id and with no result, and its value correspondence says the compare is partial."""
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--match-dir", str(tmp_path / "no_match"))
    missing = "FileNotFoundError(2, 'No such file or directory')"
    assert oracle_compares(ended) == [{"schema": "verity-sweep/oracle-compare/v1", "method": "oracle-compare", "result": None,
                                       "why": NOT_RUN + missing, "compared": 0, "mismatch_n": 0, "error": missing}] * 2
    for row in ended.rows():
        if row["arm"] == "instrumented":
            correspondence = row["commit"]["value_correspondence"]
            assert correspondence["method"].startswith("oracle-compare(partial) + ")
            assert correspondence["oracle_provenance"] is None
            assert correspondence["components"]["oracle_compare"] == {
                **dict.fromkeys(("result", "grade", "equal", "steps_compared", "steps_not_snapshotted", "complete", "attribution",
                                 "workload_check", "value_source", "provenance")),
                "why": NOT_RUN + missing, "compared": 0, "mismatch_n": 0, "reference": "live Match oracle of this attempt (--match-dir)"}


def test_an_oracle_compare_that_cannot_be_imported_is_recorded_under_the_same_schema(tmp_path, stand_ins, capsys, monkeypatch):
    """The comparison's module is imported where it is first needed.  Where that import fails the record still carries the
    comparison's schema id, which is the id the module itself writes."""
    from verity_vllm import check
    from verity_vllm.check import oracle_compare

    assert oracle_compare.SCHEMA == "verity-sweep/oracle-compare/v1"
    monkeypatch.delattr(check, "oracle_compare")
    monkeypatch.setitem(sys.modules, "verity_vllm.check.oracle_compare", None)
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--match-dir", str(tmp_path / "no_match"))
    records = oracle_compares(ended)
    assert [(r["schema"], r["method"], r["result"], r["compared"], r["mismatch_n"]) for r in records] == [
        ("verity-sweep/oracle-compare/v1", "oracle-compare", None, 0, 0)] * 2
    assert all(r["why"] == NOT_RUN + r["error"] and r["error"].startswith(("ImportError(", "ModuleNotFoundError(")) for r in records)
