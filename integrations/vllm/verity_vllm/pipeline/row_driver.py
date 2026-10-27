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

import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT, TARGETS, RowConfig, RunEnv, row_execution, row_path, row_target
from verity_vllm.engine import served
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.program import prefix_plan as PP

CLI = ("-m", "verity_vllm.pipeline.cli")
SIGNUM = {"TERM": 15, "INT": 2, "HUP": 1}
SHARED_JIT = ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT")
_NUM = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
_TREE_ID = re.compile(r"[0-9a-f]{16}")
_TRITON_KERNEL = re.compile(rb"^[ \t]*@triton\.(?:jit|autotune)\b", re.M)


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


def jit_key(py: str, cuda_home: str, row: str, tree: str) -> str | None:
    """What a shared JIT cache is partitioned by (`jit_cache`), hashed: the ``triton``, ``torch`` and ``vllm`` installed for the
    interpreter ``py`` (each dist-info ``RECORD``), the toolkit's ``CUDA_VERSION``, the compute capability of ``row``'s device class
    and the tree's own Triton kernels (each ``verity_vllm`` module that defines one, by path and content), so that a new toolchain or
    an edited kernel starts a cache of its own.  None when any of them can't be read."""
    try:
        parts: list[list[str]] = []
        venv = glob.escape(os.path.dirname(os.path.dirname(py)))
        for name in ("triton", "torch", "vllm"):
            rec = glob.glob(os.path.join(venv, "lib", "python*", "site-packages", f"{name}-*.dist-info", "RECORD"))
            if len(rec) != 1:
                return None
            parts.append([name, hashlib.sha256(Path(rec[0]).read_bytes()).hexdigest()])
        ver = re.search(rb"^#define CUDA_VERSION (\d+)", Path(cuda_home, "include", "cuda.h").read_bytes(), re.M)
        cc = TARGETS.get(row_target(row) or "")
        if not ver or not cc:
            return None
        parts += [["cuda", ver.group(1).decode()], ["cc", f"{cc[0]}.{cc[1]}"]]
        for p in sorted(Path(tree, "verity_vllm").rglob("*.py")):
            b = p.read_bytes()
            if _TRITON_KERNEL.search(b):
                parts.append([p.relative_to(tree).as_posix(), hashlib.sha256(b).hexdigest()])
    except OSError:
        return None
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def jit_cache(env: dict[str, str], tree: str | None = None, key: Callable[[], str | None] | None = None) -> None:
    """The template gives a ``*__bi-eager`` row JIT caches on the host named for its job tree (``TRITON_CACHE_DIR`` and
    ``VLLM_CACHE_ROOT`` ending in the id of ``job_tree.sh``'s copy, ``<id>/integrations/vllm`` = ``tree``), so each sync of a tree
    started them empty and each model's first Commit on it compiled its kernels again under its GPU lease.  The caches named for
    ``tree`` move to ``k-<key()>`` beside them, one per `jit_key`, shared by every tree of that key: Triton's own key covers each
    kernel's source, dependencies, signature, options and backend, vLLM's modelinfos are keyed by their module's hash, and both write
    atomically.  Any other directory is kept, as is every directory when ``key()`` is None.

    The CUDA driver's JIT cache follows the Triton cache, on the host, not in the pod's home: vLLM's FA2 extension has no code for the
    RTX PRO 6000 (sm_120), so each pod's first launch of every attention kernel has the driver compile its PTX on the CPU, about a
    minute of a held GPU, unless the compiled kernels outlive the pod."""
    tree_id = os.path.basename(os.path.dirname(os.path.dirname(tree))) if tree else ""
    named = [v for v in SHARED_JIT if _TREE_ID.fullmatch(tree_id) and os.path.basename(env.get(v) or "") == tree_id]
    shared = key() if named and key else None
    for v in named if shared else ():
        env[v] = os.path.join(os.path.dirname(env[v]), f"k-{shared}")
    triton = env.get("TRITON_CACHE_DIR")
    if not triton or env.get("CUDA_CACHE_PATH"):
        return
    path = os.path.join(os.path.dirname(os.path.dirname(triton)), "cuda", os.path.basename(triton))
    os.makedirs(path, exist_ok=True)
    env.update({"CUDA_CACHE_PATH": path, "CUDA_CACHE_MAXSIZE": str(4 << 30)})


def _links(d: str, targets: dict[str, str]) -> bool:
    try:
        return sorted(os.listdir(d)) == sorted(targets) and all(os.readlink(os.path.join(d, n)) == t for n, t in targets.items())
    except OSError:
        return False


def cublas_names(env: dict[str, str], py: str, root: str | None = None) -> str | None:
    """Put first on ``env``'s ``LD_LIBRARY_PATH`` a directory whose ``libcublas.so`` and ``libcublasLt.so`` are links to the cuBLAS of
    ``py``'s venv (``site-packages/nvidia/*/lib/<name>.so.N``, the library torch loads), and return it.  vLLM's flashinfer autotune
    (its cache's metadata, `flashinfer.autotuner._get_cublas_version`) loads cuBLAS by the unversioned name, which a wheel doesn't
    ship, so without it ld.so.cache gives a node's system CUDA's and the stage maps two cuBLAS: the route pin
    (`check.replay.driver.cublas_versions`) reads exactly one.  The loader reads ``LD_LIBRARY_PATH`` once, when a process starts, so
    this is the stages' environment, which every rank worker vLLM spawns inherits.  The directory is ``root`` (default: the temp dir)
    named for the links' targets, so every stage on one venv has the same environment.  None, and ``env`` as it was, when the venv
    has no cuBLAS wheel; ValueError when it has more than one library of a name."""
    site = os.path.join(glob.escape(os.path.dirname(os.path.dirname(py))), "lib", "python*", "site-packages", "nvidia", "*", "lib")
    found = {f"{n}.so": sorted({os.path.realpath(p) for p in glob.glob(os.path.join(site, f"{n}.so.*"))}) for n in ("libcublas", "libcublasLt")}
    if not any(found.values()):
        return None
    if any(len(f) != 1 for f in found.values()):
        raise ValueError(f"not one cuBLAS of each name in {py}'s venv: {found}")
    targets = {n: f[0] for n, f in found.items()}
    d = os.path.join(root or tempfile.gettempdir(), "verity-cublas-names-" + hashlib.sha256(json.dumps(targets).encode()).hexdigest()[:16])
    if not _links(d, targets):
        tmp = tempfile.mkdtemp(prefix=".verity-cublas-names-", dir=os.path.dirname(d))
        for n, t in targets.items():
            os.symlink(t, os.path.join(tmp, n))
        try:
            os.rename(tmp, d)
        except OSError:
            shutil.rmtree(tmp)
        if not _links(d, targets):
            raise ValueError(f"{d} holds other than the links {targets}")
    rest = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != d]
    env["LD_LIBRARY_PATH"] = ":".join([d, *rest])
    return d


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
        self.wl = row_path(row)
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
        core, research = Path(self.tree, "../.."), Path(self.tree, "../../tools/research/src")
        if not core.is_dir():
            print(f"PRECHECK_FAIL_SOURCE_IDENTITY: no Verity tree above {self.tree} (not a shipped Verity tree)", file=sys.stderr)
            raise Stop(3)
        if not research.is_dir():
            print(f"PRECHECK_FAIL_SOURCE_IDENTITY: no tools/research/src beside {self.tree}", file=sys.stderr)
            raise Stop(3)
        core_s, research_s = os.path.realpath(core), os.path.realpath(research)
        env = dict(self.env.environ)
        env.update({"VERITY_TREE": self.tree, "VERITY_CORE": core_s, "VERITY_RESEARCH": research_s,
                    "PYTHONPATH": f".:{core_s}:{research_s}", "HF_HOME": self.env.hf_home, "HF_HUB_OFFLINE": "1",
                    **served.execution_env(self.execution()), "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": self.env.cuda_home})
        env.update(pins)
        jit_cache(env, self.tree, lambda: jit_key(self.py, env["CUDA_HOME"], self.row, self.tree) if self.bi_eager() else None)
        if GL.leased(env):
            GL.hide_devices(env)
        try:
            cublas_names(env, self.py)
        except (OSError, ValueError) as e:
            print(f"PRECHECK_FAIL_CUBLAS_NAMES: {e}", file=sys.stderr)
            raise Stop(3)
        self.child_env = env
        if not os.access(self.py, os.X_OK):
            print(f"no python at {self.py} (run verity_vllm/ops/pod_bootstrap.sh first)", file=sys.stderr)
            raise Stop(3)
        env["PATH"] = f"{os.path.dirname(self.py)}:{self.env.cuda_bin}:{self.env.environ.get('PATH', '')}"
        env["VERITY_TIMELINE"] = self.timeline

    def path(self, rel: str) -> str:
        """``rel`` under the tree (the stages' cwd)."""
        return os.path.join(self.tree, rel)

    def config(self) -> RowConfig:
        """The row's config, its file in the tree that runs."""
        return RowConfig.load(self.path(self.wl))

    def refuse_unserved_engine(self) -> None:
        """Refuse a tensor-parallel row whose file declares prefix caching: its capture, Match and Commit engines run with it off and a
        rank Program (`vllm_meta.with_peers`) takes no cached prefix, so the row's record would name a serve it never ran.  A
        single-GPU row serves it: a warm request's Program reads its cached rows as its `prefix` input (`program.prefix_plan`)."""
        if self.config().engine.args.get("enable_prefix_caching") is True:
            self.verdict("precheck", "FAIL", "engine.args.enable_prefix_caching is true on a tensor-parallel row: its stages serve with prefix "
                                             "caching off and a rank Program takes no cached prefix, so the record would name a serve it never ran")
            raise Stop(3)

    def execution(self) -> str | None:
        """The row's execution label: its workload's ``execution`` (or ``sweep.execution``), else its id's last token; None without
        a workload file."""
        doc = _read_json(self.path(self.wl))
        if not isinstance(doc, dict):
            return None
        sw = doc.get("sweep") if isinstance(doc.get("sweep"), dict) else {}
        return doc.get("execution") or sw.get("execution") or row_execution(self.row)

    def bi_eager(self) -> bool:
        """The row's execution is ``bi-eager``, the only one the template gives JIT caches.  A compiled row records cubins its Triton
        cache holds (`code_identity.loaded_code_objects`), so it never shares one."""
        return self.execution() == "bi-eager"

    def export_commit(self) -> str:
        """The shipped tree's commit (``EXPORT.json``), "" when absent."""
        doc = _read_json(self.path("EXPORT.json"))
        return (doc.get("commit") or "") if isinstance(doc, dict) else ""

    def serving_rows_request(self, partition: str) -> str:
        """The Commit's serving-rows request (`pipeline.serving_rows`, `pipeline.tp.serving_rows`): its directory inside the research
        run dir when there is one, so the run's custody preserves it, the verifier's partition file, and the run id and source the
        registration names (its domains bind the run root)."""
        rd = self.env.research_run_dir
        d = os.path.join(rd, "serving-rows", self.row) if rd else f"{self.d}/serving-rows"
        os.makedirs(d, exist_ok=True)
        req = os.path.join(d, "request.json")
        with open(req, "w") as f:
            json.dump({"dir": d, "partition": partition, "run": self.env.environ.get("RESEARCH_RUN_ID") or self.row,
                       "committer": {"source": self.env.environ.get("RESEARCH_SOURCE_SHA")}}, f)
        return req

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

    @property
    def leased(self) -> bool:
        """The item leases its GPU (`gpu_lease`): its pod has none of its own."""
        return GL.leased(self.child_env)

    def run_gpu(self, stage: str, to: int | str, lf: str, cmd: list[str], *, lease_dir: str | None = None, kept: str | None = None) -> int:
        """``run_tl`` for the stage that needs the GPU: under a lease when the item is leased, else as any stage.  ``lease_dir``: where
        the lease's markers and ``gpu_lease.json`` go (default the row dir, the Commit's); ``kept``: why a process that never releases
        holds its lease to its exit (`gpu_lease.run`)."""
        if not self.leased:
            return self.run_tl(stage, to, lf, cmd)
        d = lease_dir or self.d
        os.makedirs(d, exist_ok=True)
        rc, rec = GL.run(d, lf, cmd, int(to), self.child_env, lambda t, argv: self.run_tl(stage, t, lf, argv), kept=kept)
        self.log(f"GPU lease ({GL.RECORD}): waited {rec['waited_min']} min, held {rec['held_min']} min, released {rec['released']} after "
                 f"{rec['released_at']}{' (kept: ' + rec['kept'] + ')' if rec.get('kept') else ''}, rc={rc}")
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
        rc, line = self.output(self.cli("target-family", "--row", self.row, "--workload", self.wl, "--stages", self.stages,
                                        "--match-phase", self.cm.match_phase, *extra, "--out", out))
        with open(f"{self.d}/row.log", "a") as f:
            f.write(line + "\n")
        return rc, line

    def launch_args(self, lp: int, t: int, C: int = 0) -> list[str] | None:
        """A B >= 2 request Program is built under its launch context (``launch-context``; `C` a warm request's cached rows); None =
        the context did not read, which FAILS that shape's Build (it is not "no context")."""
        if self.B <= 1:
            return []
        name = PP.shape_name(PP.shape(lp, t, C))
        err = f"{self.d}/launch_context_{name}.err"
        with open(err, "w") as f:
            r = subprocess.run(self.cli("launch-context", self.wl, str(lp), str(t), *([str(C)] if C else [])), stdout=subprocess.PIPE,
                               stderr=f, text=True, cwd=self.tree, env=self.child_env)
        if r.returncode != 0:
            self.log(f"Build FAIL shape {name}: launch_context rc={r.returncode} -- {open(err, errors='replace').read().replace(chr(10), ' ')[-300:]}")
            return None
        os.remove(err)
        lc = r.stdout.rstrip("\n")
        return ["--launch-context", *lc.split("\n")] if lc else []

    def call_boundaries(self, to: int | str, cmd: list[str] | None = None) -> None:
        """After a PASS Build: the row stops (exit 21, no Match, no Commit) unless every `call_boundaries` identity of its manifest has
        a source the Commit attaches (`cmd`, else `gate call-boundaries`); the verdict line names the first identity without one."""
        lf = f"{self.d}/call_boundaries.log"
        rc = self.run_tl("build.call_boundaries", to, lf, cmd or self.cli("gate", "call-boundaries", self.d, "--out", f"{self.d}/call_boundaries.json"))
        with open(lf, errors="replace") as f:
            line = (f.read().splitlines() or [""])[-1][:600]
        if rc == 0:
            self.verdict("build-call-boundaries", "PASS", line)
            return
        self.verdict("build-call-boundaries", "FAIL", f"rc={rc} {line}")
        self.log("call boundaries without a source the Commit attaches -> stop (see call_boundaries.json)")
        self.tl("row", "end", ("ok", "false"), ("stopped_at", "call_boundaries"))
        raise Stop(21)

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
