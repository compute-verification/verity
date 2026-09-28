"""The driver under `verity-vllm row`: one row's evidence dir, its logs and telemetry, and the stage processes.

The tree that runs is the tree this package was imported from (``config.ROOT``), never the environment: a caller whose
``VERITY_TREE`` names another tree is refused by name.  Stages are ``verity-vllm`` subcommands run as child processes
under the configured interpreter, with the tree as cwd and the environment the CLI was started with plus the pins below.
Evidence: ``<sweep-dir>/<row>/`` -- ``stages.txt`` (one verdict line per stage), ``row.log``, ``timeline.jsonl``.

A stage runs under GNU ``timeout`` in its own process group; a TERM/INT/HUP the row receives is forwarded to it (and
through ``timeout`` to the stage's workers), the row waits for the stage to be gone, records the cancel and exits
128+signal.  Crash-only: a stage that fails is a verdict line and an exit code, never a retry.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT, RunEnv

CLI = ("-m", "verity_vllm.pipeline.cli")
SIGNUM = {"TERM": 15, "INT": 2, "HUP": 1}
_NUM = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")


class Stop(Exception):
    """The row stops here with exit code ``rc`` (its verdict line and log are already written)."""

    def __init__(self, rc: int):
        super().__init__(rc)
        self.rc = rc


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp() -> str:
    ns = time.time_ns()
    return f"{ns // 10**9}.{ns % 10**9:09d}"


def _read_json(path: str | Path) -> Any:
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


class Row:
    """One row's run: ``row`` id, checkpoint ``role``, HF ``repo`` and ``rev``, the requested ``stages``.

    ``proc`` names the driver in the timeline (``row_pod`` for a single-GPU row, ``tp_stage`` for a tensor-parallel one).
    """

    def __init__(self, *, row: str, role: str, repo: str, rev: str, stages: str, env: RunEnv, proc: str):
        self.row, self.role, self.repo, self.rev, self.stages, self.env, self.proc = row, role, repo, rev, stages, env, proc
        self.tree = os.path.realpath(ROOT)
        self.py = env.py
        self.py312 = env.py312 or env.py
        self.d = f"{env.sweep_dir}/{row}"
        self.wl = f"workloads/{row}.json"
        self.timeline = env.timeline or f"{self.d}/timeline.jsonl"
        self.child_env: dict[str, str] = {}
        self._running: set[int] = set()
        self.signalled: str | None = None

    # ------------------------------------------------------------------------------------------------ setup
    def setup(self, pins: dict[str, str]) -> None:
        """Refuse a mixed tree, find the core and the runner beside it, and fix the stages' environment."""
        if self.env.verity_tree and os.path.realpath(self.env.verity_tree) != self.tree:
            print(f"PRECHECK_FAIL_SOURCE_IDENTITY: {self.proc} at {self.tree} but $VERITY_TREE={self.env.verity_tree} names another tree",
                  file=sys.stderr)
            raise Stop(3)
        core, research = Path(self.tree, "../../packages/verity/src"), Path(self.tree, "../../tools/research/src")
        if not core.is_dir():
            print(f"PRECHECK_FAIL_SOURCE_IDENTITY: no packages/verity/src beside {self.tree} (not a shipped Verity tree)", file=sys.stderr)
            raise Stop(3)
        if not research.is_dir():
            print(f"PRECHECK_FAIL_SOURCE_IDENTITY: no tools/research/src beside {self.tree}", file=sys.stderr)
            raise Stop(3)
        core_s, research_s = os.path.realpath(core), os.path.realpath(research)
        env = dict(self.env.environ)
        env.update({"VERITY_TREE": self.tree, "VERITY_CORE": core_s, "VERITY_RESEARCH": research_s,
                    "PYTHONPATH": f".:{core_s}:{research_s}", "HF_HOME": self.env.hf_home, "HF_HUB_OFFLINE": "1",
                    "VLLM_BATCH_INVARIANT": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": self.env.cuda_home})
        env.update(pins)
        self.child_env = env
        if not os.access(self.py, os.X_OK):
            print(f"no python at {self.py} (run verity_vllm/ops/pod_bootstrap.sh first)", file=sys.stderr)
            raise Stop(3)
        env["PATH"] = f"{os.path.dirname(self.py)}:{self.env.cuda_bin}:{self.env.environ.get('PATH', '')}"
        env["VERITY_TIMELINE"] = self.timeline

    def path(self, rel: str) -> str:
        """``rel`` under the tree (the stages' cwd)."""
        return os.path.join(self.tree, rel)

    def export_commit(self) -> str:
        """The shipped tree's commit (``EXPORT.json``), "" when absent."""
        doc = _read_json(self.path("EXPORT.json"))
        return (doc.get("commit") or "") if isinstance(doc, dict) else ""

    # ------------------------------------------------------------------------------------------------ records
    def log(self, msg: str) -> None:
        line = f"{utc()} [row {self.row}] {msg}"
        print(line, flush=True)
        with open(f"{self.d}/row.log", "a") as f:
            f.write(line + "\n")

    def verdict(self, stage: str, status: str, detail: str, *, quiet: bool = False) -> None:
        line = f"{stage} {status} {utc()} {detail}"
        if not quiet:
            print(line, flush=True)
        with open(f"{self.d}/stages.txt", "a") as f:
            f.write(line + "\n")

    def tl(self, stage: str, ev: str, *kv: tuple[str, Any]) -> None:
        """Append one ``verity-sweep/timeline/v1`` event: values that read as numbers / true / false / null unquoted."""
        parts = []
        for k, v in kv:
            v = str(v)
            if _NUM.match(v) or v in ("true", "false", "null"):
                parts.append(f',"{k}":{v}')
            else:
                parts.append(f',"{k}":"{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"')
        line = (f'{{"t":{_stamp()},"utc":"{utc()}","schema":"verity-sweep/timeline/v1","proc":"{self.proc}","pid":{os.getpid()},'
                f'"stage":"{stage}","ev":"{ev}","hot":false{"".join(parts)}}}\n')
        try:
            with open(self.timeline, "a") as f:
                f.write(line)
        except OSError:
            pass

    def research_result(self) -> None:
        """The research runner's result envelope, from the stage verdicts (only under a research run)."""
        if self.env.research_run_dir and os.path.isdir(self.d):
            with open(f"{self.d}/row.log", "a") as f:
                subprocess.run(self.cli("research-result", self.d, self.row, self.role, self.stages), stdout=f, stderr=subprocess.STDOUT,
                               cwd=self.tree, env=self.child_env or dict(self.env.environ))

    # ------------------------------------------------------------------------------------------------ processes
    def cli(self, *args: str, py: str | None = None) -> list[str]:
        """The command line of ``verity-vllm ARGS...`` under ``py`` (default: the stages' interpreter)."""
        return [py or self.py, *CLI, *args]

    def output(self, cmd: list[str], *, stderr: int | None = subprocess.STDOUT) -> tuple[int, str]:
        """Run ``cmd`` to completion in the tree -> (rc, stdout[+stderr] without trailing newlines)."""
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True, cwd=self.tree, env=self.child_env)
        return r.returncode, r.stdout.rstrip("\n")

    def install_signal_forwarding(self) -> None:
        for name in SIGNUM:
            signal.signal(getattr(signal, f"SIG{name}"), self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        name = next(k for k, v in SIGNUM.items() if v == signum)
        self.signalled = name
        for pid in list(self._running):
            self.log(f"SIG{name} received: forwarding to the running stage (timeout pid {pid} -> its process group)")
            try:
                os.kill(pid, signum)
            except OSError:
                pass

    def run_to(self, to: int | str, lf: str, cmd: list[str]) -> int:
        """Run one stage command under ``timeout TO`` with its output in ``lf`` (first line: ``$ <command>``) -> rc."""
        with open(lf, "w") as f:
            f.write("$ " + " ".join(cmd) + "\n")
        with open(lf, "a") as f:
            p = subprocess.Popen(["timeout", str(to), *cmd], stdout=f, stderr=subprocess.STDOUT, cwd=self.tree, env=self.child_env)
            self._running.add(p.pid)
            try:
                rc = p.wait()
            finally:
                self._running.discard(p.pid)
        if rc < 0:
            rc = 128 - rc
        if rc == 124:
            with open(lf, "a") as f:
                f.write(f"TIMEOUT after {to}s\n")
        if self.signalled:
            with open(lf, "a") as f:
                f.write(f"CANCELLED by SIG{self.signalled} (stage rc={rc})\n")
            if threading.current_thread() is threading.main_thread():
                self.cancelled(rc)
        return rc

    def cancelled(self, rc: int) -> None:
        """Stop the row after a forwarded signal: the stage has ended; record the cancel and exit 128+signal."""
        self.log(f"row cancelled by SIG{self.signalled}; stage ended rc={rc} -> stop")
        self.tl("row", "cancelled", ("signal", self.signalled), ("rc", rc))
        self.tl("row", "end", ("ok", "false"), ("stopped_at", "cancel"))
        raise Stop(128 + SIGNUM.get(self.signalled or "TERM", 15))

    def run_tl(self, stage: str, to: int | str, lf: str, cmd: list[str], *extra: tuple[str, Any]) -> int:
        """``run_to`` inside a timeline span named ``stage`` (rc on the end event)."""
        self.tl(stage, "start", *extra)
        rc = self.run_to(to, lf, cmd)
        self.tl(stage, "end", ("rc", rc), *extra)
        return rc

    def gpu_exclusive(self, why: str) -> None:
        """One engine per pod: ask the pod's hot Commit workers to stop before a stage starts its own engine (an idle
        worker exits, a busy one is waited for, never killed); the wait is on the timeline."""
        root = self.env.hot_root
        if not any(Path(root).glob("*/worker.json")):
            return
        t0 = time.time()
        _, out = self.output(self.cli("hot", "stop", "--root", root))
        out = ("".join(ln + " " for ln in out.split("\n")[-3:]) if out else "")[:300]
        s = round(time.time() - t0, 3)
        if '"stopped": []' not in out and out:
            self.log(f"hot worker(s) stopped before {why} (one engine per pod, {s}s): {out}")
        self.tl("hot.gpu_exclusive", "mark", ("why", why), ("s", s), ("out", out))

    # ------------------------------------------------------------------------------------------------ prechecks
    def release_line(self, fallback: str) -> tuple[str, str]:
        """The release line of the tree whose code runs (``release-json line``) and the record it came from."""
        rc, out = self.output(self.cli("release-json", "line", self.tree), stderr=subprocess.DEVNULL)
        if rc != 0:
            out = f"UNRELEASED|commit=unknown tree={self.tree}{fallback}\t-"
        line, _, src = out.partition("\t")
        return line, src or out

    def source_identity(self, pys: list[str], out_of: Any) -> int:
        """Every package resolves from THIS tree under the stages' interpreters (``source-identity``) -> 0 | 3."""
        sha = self.export_commit()
        rc = 0
        for py in pys:
            if not os.access(py, os.X_OK):
                continue
            cmd = self.cli("source-identity", "--tree", self.tree, *(["--sha", sha] if sha else []), "--stage", "precheck",
                           "--out", out_of(py), py=py)
            with open(f"{self.d}/row.log", "a") as f:
                if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=self.tree, env=self.child_env).returncode != 0:
                    rc = 3
        return rc

    def target_family(self, *extra: str, out: str) -> tuple[int, str]:
        """The target-family preflight (``target-family``): the row id's device class, the declared target and the device
        (or the Build that came out) are one family.  Its line goes to row.log."""
        rc, line = self.output(self.cli("target-family", "--row", self.row, "--workload", self.wl, "--stages", self.stages, *extra,
                                        "--out", out))
        with open(f"{self.d}/row.log", "a") as f:
            f.write(line + "\n")
        return rc, line

    def timeline_report(self) -> None:
        subprocess.run(self.cli("timeline", "report", self.timeline, "--json", f"{self.d}/timeline_report.json", "--md",
                                f"{self.d}/timeline_report.md", "--title", self.row),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=self.tree, env=self.child_env)


def stage_on(stages: str, name: str) -> bool:
    return f",{name}," in f",{stages},"


def copy_first(dst: str, *candidates: str) -> None:
    for src in candidates:
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            return
