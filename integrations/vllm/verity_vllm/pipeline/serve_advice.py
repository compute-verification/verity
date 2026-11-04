"""serve-openai's advice (``--advice``): the registered compiler B registered once, before the first Commit period, and each
window's advice committed by the server, its public line out before any coin of the window.

* **Registration.** Before its engine loop runs, the server writes ``advice_registration.json`` (`register`, public): B as
  the verifier recomputes it from ``serve.json`` (``verity-vllm predict compiler``: compile_steps capped at ℓ_a,
  ``--advice-cap``), the advice's format (`window_advice.SCHEMA`), and the deployment's registration log
  (`verity.core.protocols.registration`, under ``--advice-composite``), committed: B's information entry
  (`advice.builder_item`) and the day's declaration after it, B's digest as day ``--advice-day``'s program, so the day's
  one program is B and every window's circuit is B's output on its advice (`declared_log`). ``--advice-log`` extends the
  deployment's log so far, which registers B at most once and declares each day at most once (`duplicate-information`,
  `duplicate-declaration`): another builder, or a day it already declares, is refused. A serve dir that holds a
  registration is refused, and so is a second commit of its windows' advice by `verity-vllm advice`: a run registers B once.
* **Commit.** Each closed Commit period's advice (`advice.from_trace`, a window carrying what the one before left) is
  committed as `advice.write_served` commits it, under ``advice/w<k>/`` (``registration.json``, ``private/``), once the window
  is whole: its close and its ``windows.jsonl`` line, every step it scheduled recorded, and the Commit record of every
  request it ran, which the server writes at the close of the window the request finishes in. A window waits on its slowest
  request, so on at most that request's token cap, and the windows are committed in order. Each gets a line of
  ``advice.jsonl`` (`LINES`, public, mirrored): the compiler's digest, the root of the advice's hiding commitment and ℓ_a,
  the registered width, the length the advice is padded to before it is committed and compiled (`advice_windows.pad`), so
  no row count or public field says the advice's own length, which only the prover's ``private/`` knows; a window the advice
  refuses (one whose encoding is past ℓ_a among them, `advice.within_cap`) is a line naming it refused, its reason in
  ``advice/windows.json``, and the committer goes on to the next window, which opens on nothing carried; one with no step
  a line naming it empty, so every closed window has its line.
* **Coins.** A draw of a window through ``--draws-dir`` before its line is out is refused, and the window's release with it
  (`serve_release.DrawDir`): no coin of a window is drawn before its advice is committed. At the stop the windows left are
  committed (`stop`), a request with no Commit record refused as `write_served` refuses it, before ``summary.json``'s
  ``final``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.core.protocols import registration as REG
from verity_vllm.config import UsageError, option
from verity_vllm.engine import step_trace as ST
from verity_vllm.records import served_advice as RSA

from . import advice as ADV
from . import advice_windows as AW
from . import window_advice as WA

REGISTRATION = "advice_registration.json"
LINES = "advice.jsonl"
COMPOSITES = ("onsite", "remote")


@dataclass(frozen=True, kw_only=True)
class AdviceOptions:
    """serve-openai's ``--advice`` group (`refusal`, `ServedAdvice`)."""
    advice: bool = option("--advice", action="boolean_optional", default=False,
                          help="register B (the served compiler, capped at --advice-cap) before the first window and commit each "
                               "Commit period's advice as the server serves, its public line in advice.jsonl before any draw of "
                               "the window (serve_advice)")
    advice_cap: int = option("--advice-cap", type=int, default=AW.CAP_BITS, metavar="BITS",
                             help="ℓ_a, the registered width: the length in bits every window's advice is padded to before it is "
                                  "committed and compiled, which B's digest covers; a window past it is refused. The least "
                                  "that pads every window within the server's limits is advice_windows.least_cap of "
                                  "window_advice.max_bits (default 2^23)")
    advice_composite: str = option("--advice-composite", choices=COMPOSITES, default="onsite",
                                   help="the deployment whose registration log B's entry opens (default onsite)")
    advice_day: int = option("--advice-day", type=int, default=0, metavar="DAY",
                             help="the day this run serves, by its index among the monitoring period's days: the log declares "
                                  "B as its one program (default 0)")
    advice_log: str | None = option("--advice-log", default=None, metavar="JSON",
                                    help="the deployment's registration log so far (a run's advice_registration.json, or its "
                                         "committed log), which this run's extends; refused if it registers another builder "
                                         "or already declares --advice-day (default: the log opens here)")
    advice_weights: str | None = option("--advice-weights", default=None, metavar="JSON",
                                        help="the committer's weights registration (prove_path.py weights-registration --out), "
                                             "whose roots each window's registration records once it exists (advice."
                                             "registered_weights; default: none recorded)")
    advice_tau: float = option("--advice-tau", type=float, default=0.1,
                               help="the egress bucket's width in seconds, for each output's release bucket in the advice")


def refusal(opts: Any) -> str | None:
    """Why ``--advice`` can't hold (None: it can): it reads the Commit's windows, ℓ_a is a whole number of bits, a multiple
    of 8 and at least 8, the day is a natural number, and a log it extends is the deployment's, checks and doesn't declare
    the day yet (B's own refusals, which need serve.json, are `declared_log`'s)."""
    a = opts.advising
    if not a.advice:
        return None
    if not opts.commit:
        return "--advice commits each Commit period's advice: it needs the Commit (drop --no-commit)"
    if isinstance(a.advice_cap, bool) or a.advice_cap < 8 or a.advice_cap % 8:
        return f"--advice-cap {a.advice_cap}: ℓ_a is a whole number of bits, a multiple of 8 and at least 8"
    if isinstance(a.advice_day, bool) or a.advice_day < 0:
        return f"--advice-day {a.advice_day}: a day is its index among the monitoring period's days, 0 or more"
    if a.advice_log is not None:
        if not Path(a.advice_log).is_file():
            return f"--advice-log {a.advice_log}: no such file"
        try:
            theirs, entries = REG.log_of(log_doc(json.loads(Path(a.advice_log).read_text())))
            view = REG.check(REG.Auditor(), theirs, entries).view()
        except (KeyError, TypeError, ValueError, REG.Refused) as e:
            return f"--advice-log {a.advice_log}: not a registration log that checks ({type(e).__name__}: {e})"
        if theirs != a.advice_composite:
            return f"--advice-log {a.advice_log}: the registration log is {theirs}'s, not {a.advice_composite}'s"
        if view.declared(a.advice_day) is not None:
            return f"--advice-log {a.advice_log}: it declares day {a.advice_day} already, and a day has one program"
    if (Path(opts.out) / REGISTRATION).exists():
        return (f"{Path(opts.out) / REGISTRATION} exists: a run registers B once, so a server doesn't serve into another's serve "
                "dir (give a fresh --out)")
    return None


def log_doc(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    """The committed registration log in ``doc``: a run's ``advice_registration.json`` (`records.served_advice`, its log) or
    the log."""
    if isinstance(doc, Mapping) and doc.get("schema") == RSA.SCHEMA:
        return RSA.ServedAdviceRegistration.of(doc, REGISTRATION).log
    return doc


def declared_log(composite: str, compiler: AW.Capped, day: int, log: Mapping[str, Any] | None = None) -> list[REG.Entry]:
    """The deployment's registration log for a run of day ``day`` under B: ``log``, the log so far as committed (None: it
    opens here), then B's information entry unless the log registers B already, then the day's declaration, whose program
    is B's digest: the day's one program is B, and each window's circuit is B's output on the window's advice. ValueError
    naming why when ``log`` is another deployment's or doesn't check, registers another builder, or declares the day."""
    entries: list[REG.Entry] = []
    if log is not None:
        theirs, entries = REG.log_of(log)
        if theirs != composite:
            raise ValueError(f"the registration log is {theirs}'s, not {composite}'s")
    try:
        builder = REG.check(REG.Auditor(), composite, entries).view().builder
    except REG.Refused as e:
        raise ValueError(f"the registration log so far doesn't check: {e}") from None
    item = ADV.builder_item(compiler)
    if builder not in (None, item):
        raise ValueError("the registration log registers another builder than B, and a deployment registers its builder once")
    full = REG.chain(composite, [*entries, *([] if builder else [REG.Information((item,))]),
                                 REG.Declaration(int(day), compiler.digest)])
    try:
        REG.check(REG.Auditor(), composite, full)
    except REG.Refused as e:
        raise ValueError(f"the registration log refuses day {day}'s declaration of B ({e.name}): a day has one program, "
                         "declared once") from None
    return full


def compiler_doc(serve: Mapping[str, Any], cap_bits: int) -> dict[str, Any]:
    """B of the run whose ``serve.json`` is ``serve``, by the verifier's formula (`served_compiler.compiler_doc`), from
    ``verity-vllm predict compiler`` in a process of its own: the predictor imports the serving pipeline, never the reverse.
    ValueError naming the formula's refusal."""
    r = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "predict", "compiler", "--serve-json", "-",
                        "--cap-bits", str(int(cap_bits))], input=json.dumps(serve, default=str), capture_output=True, text=True)
    if r.returncode:
        raise ValueError(f"verity-vllm predict compiler: exit {r.returncode}: {(r.stderr.strip().splitlines() or [''])[-1]}")
    return json.loads(r.stdout)


def register(out: Path, serve: Mapping[str, Any], cap_bits: int, composite: str = "onsite", *, day: int = 0,
             log: Mapping[str, Any] | None = None, t: float | None = None) -> tuple[AW.Capped, dict[str, Any]]:
    """B registered for the run whose ``serve.json`` is ``serve``, serving day ``day``: ``out/advice_registration.json``,
    written once (an existing one is a `UsageError`), at ``t`` (unix seconds, default now), its log `declared_log` of
    ``log``. -> B and the document."""
    doc = compiler_doc(serve, cap_bits)
    compiler = ADV.registered_compiler(doc["inner"]["name"], doc["inner"]["digest"], cap_bits)
    if compiler.digest != doc["digest"]:
        raise ValueError(f"B's digest {compiler.digest[:16]}… is not the formula's {doc['digest'][:16]}…")
    entries = declared_log(composite, compiler, day, log)
    body = RSA.ServedAdviceRegistration(
        compiler=doc, advice=WA.SCHEMA, cap_bits=int(cap_bits), row_bits=AW.ROW_BITS, day=int(day), composite=composite,
        head=REG.head(composite, entries), log=json.loads(REG.committed(composite, entries)),
        t=time.time() if t is None else float(t)).dump()
    try:
        with open(Path(out) / REGISTRATION, "x") as fh:
            fh.write(json.dumps(body, indent=1) + "\n")
    except FileExistsError:
        raise UsageError(f"{Path(out) / REGISTRATION} exists: B is registered once a run") from None
    return compiler, body


def public_line(window: int, reg: Mapping[str, Any], t: float) -> dict[str, Any]:
    """Window ``window``'s line of `LINES`: its registration's binding (the advice committed padded to ℓ_a, `advice.commit`, so
    its length is ℓ_a) and the rows ℓ_a fixes, the window's Commit and when the line was written (unix seconds)."""
    b = reg["binding"]
    return {"window": int(window), "compiler": dict(b["compiler"]), "advice": dict(b["advice"]), "rows": int(reg["rows"]),
            "commit": reg.get("commit"), "t": t}


class ServedAdvice:
    """The server's committer of its windows' advice (module docstring): `CommitWindows` hands it each window's line
    (`closed`) and each finished request's Commit record (`recorded`); a thread of its own commits every window that is whole,
    in order; `stop` commits the rest. `committed` is what the draw gate reads."""

    def __init__(self, out: Path, trace: Any, *, weights: str | None = None, tau: float = 0.1, T: int = 600,
                 wall: Callable[[], float] = time.time, poll_s: float = 1.0) -> None:
        self.out, self.trace, self.weights, self.tau, self.T, self.wall = Path(out), trace, weights, float(tau), int(T), wall
        self.poll_s = float(poll_s)
        self.compiler: AW.Capped | None = None
        self.serve: Mapping[str, Any] | None = None
        self.index = ADV.TraceIndex()
        self.io: dict[str, dict[str, Any]] = {}
        self.commits: dict[int, dict[str, Any]] = {}
        self.carry: tuple[ADV.Left, ...] = ()
        self.gone: int | None = None
        self.done: set[int] = set()                     # every window settled: committed, refused or empty
        self.lines: dict[int, dict[str, Any]] = {}
        self.refused: dict[int, str] = {}
        self.reports: dict[int, dict[str, Any]] = {}
        self.error: str | None = None
        self._closed: set[int] = set()                   # the index's closed windows as of its last extend, for `summary`
        self._lock = threading.Lock()                    # io, commits and the settled windows, which the sealer's threads add to
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def register(self, serve: Mapping[str, Any], cap_bits: int, composite: str, day: int = 0,
                 log: str | None = None) -> dict[str, Any]:
        """B registered (`register`) from the run's ``serve.json`` for day ``day``, extending the log in the file ``log``
        (`log_doc`), and the committer started under it (`start`)."""
        self.serve = serve
        so_far = log_doc(json.loads(Path(log).read_text())) if log else None
        compiler, body = register(self.out, serve, cap_bits, composite, day=day, log=so_far, t=self.wall())
        self.start(compiler)
        return body

    def start(self, compiler: AW.Capped) -> ServedAdvice:
        """The committer's thread, committing under ``compiler``."""
        self.compiler = compiler
        self._thread = threading.Thread(target=self._run, name="served-advice", daemon=True)
        self._thread.start()
        return self

    def describe(self) -> dict[str, Any]:
        c = self.compiler
        return {"registration": REGISTRATION, "lines": LINES, "cap_bits": c.cap_bits if c else None,
                "compiler": {"name": c.name, "digest": c.digest} if c else None, "tau": self.tau, "weights": self.weights}

    # -- what CommitWindows hands it
    def recorded(self, rec: Mapping[str, Any]) -> None:
        """A finished request's line of ``requests.jsonl``."""
        rid, entry = ADV.io_entry(dict(rec))
        with self._lock:
            self.io[rid] = entry
        self._wake.set()

    def closed(self, rec: Mapping[str, Any]) -> None:
        """A closed window's line of ``windows.jsonl``: its Commit."""
        with self._lock:
            self.commits[int(rec["window"])] = {"run_id": rec.get("run_id"), "run_root": rec.get("run_root")}
        self._wake.set()

    def wake(self) -> None:
        """The trace has grown (a close mark): the committer looks again."""
        self._wake.set()

    def committed(self, window: int) -> bool:
        """Whether window ``window``'s advice line is out (refused or committed): what a draw of it waits for."""
        with self._lock:
            return int(window) in self.lines

    # -- the committer
    def _run(self) -> None:
        while True:
            self._wake.wait(self.poll_s)                # also unwoken: async scheduling records a last step after the close
            self._wake.clear()
            if self._stopping:
                return
            self._poll(final=False)

    def stop(self) -> dict[str, Any]:
        """The server has stopped with every Commit record written: the windows left committed (a request with no record
        refused), the index ``advice/windows.json`` written; the summary."""
        if self._thread is not None:
            self._stopping = True
            self._wake.set()
            self._thread.join()
            self._poll(final=True)
        return self.summary()

    def _poll(self, final: bool) -> None:
        try:
            self._commit_ready(final)
        except Exception as e:  # noqa: BLE001  (a committer that fails is the run's visible failure; the server goes on)
            self.error = self.error or f"{type(e).__name__}: {e}"[:500]

    def _commit_ready(self, final: bool) -> None:
        recs = self.trace.records
        self.index.extend(recs[self.index.n:len(recs)])
        idx = self.index
        with self._lock:
            self._closed = set(idx.closed)
        for w in sorted(set(idx.closed) - self.done):
            if not final and not idx.complete(w):
                return
            sub = idx.window(w)
            if sub is None:
                self._settle(w, {"window": w, "empty": True, "t": self.wall()})
                continue
            ids = {r["request"] for r in sub if r["kind"] == "arrive"}
            with self._lock:
                commit = self.commits.get(w)
                io = {i: q for i, q in self.io.items() if i in ids}
            if not final and (commit is None or ids - set(io)):
                return
            try:
                b = ADV.within_cap(self.compiler, ADV.from_trace(sub, io, tau=self.tau, carry=self.carry))
            except ValueError as e:
                why = str(e) + (f" (window {self.gone} before it is refused, so it opens on nothing carried)"
                                if self.gone is not None else "")
                self.carry, self.gone = (), w
                self._settle(w, {"window": w, "refused": True, "t": self.wall()}, refused=why)
                continue
            self.carry, self.gone = b.left, None
            rep = ADV._written(self.out, self.out / "advice" / f"w{w}", b, sub, self.out / ST.SERVED,
                               self.out / "requests.jsonl", T=self.T, compiler=self.compiler,
                               extra={"window": w, "commit": commit}, registered_roots=self._roots())
            self._settle(w, public_line(w, rep["registration"], self.wall()), report=rep)

    def _roots(self) -> dict[str, str] | None:
        if self.weights is None or not Path(self.weights).exists():
            return None
        return ADV.registered_weights(json.loads(Path(self.weights).read_text()), dict(self.serve or {}))

    def _settle(self, w: int, line: dict[str, Any], *, refused: str | None = None, report: dict[str, Any] | None = None) -> None:
        with open(self.out / LINES, "a") as fh:
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        with self._lock:
            self.done.add(w)
            self.lines[w] = line
            if refused is not None:
                self.refused[w] = refused
            if report is not None:
                self.reports[w] = report
        self._write_index()

    def _write_index(self) -> None:
        """``advice/windows.json``, as `advice.write_served` writes it."""
        with self._lock:
            reports, refused = dict(self.reports), dict(self.refused)
        index = {"schema": ADV.SERVED, "run": str(self.out), "trace": str(self.out / ST.SERVED), "T": self.T,
                 "windows": {str(w): {"registration": f"w{w}/registration.json",
                                      "root": r["registration"]["binding"]["advice"]["root"],
                                      "bits": r["registration"]["binding"]["advice"]["bits"],
                                      "commit": r["registration"]["commit"]} for w, r in sorted(reports.items())},
                 "refused": {str(w): why for w, why in sorted(refused.items())},
                 "open": sorted(set(self.index.opened) - set(self.index.closed))}
        d = self.out / "advice"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "windows.json.tmp"
        tmp.write_text(json.dumps(index, indent=1) + "\n")
        tmp.replace(d / "windows.json")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            lines, refused, waiting = dict(self.lines), dict(self.refused), sorted(self._closed - self.done)
        return {"committed": sorted(w for w, x in lines.items() if w not in refused and not x.get("empty")),
                "empty": sorted(w for w, x in lines.items() if x.get("empty")), "refused": sorted(refused), "waiting": waiting,
                "error": self.error, "passed": self.error is None and not waiting}
