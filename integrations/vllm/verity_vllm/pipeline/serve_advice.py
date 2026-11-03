"""serve-openai's advice (``--advice``): the registered compiler B registered once, before the first Commit window, and each
window's advice committed by the server, its public line out before any coin of the window.

* **Registration.** Before its engine loop runs, the server writes ``advice_registration.json`` (`register`, public): B as
  the verifier recomputes it from ``serve.json`` (`served_compiler.compiler_doc`: compile_steps capped at ℓ_a,
  ``--advice-cap``), the advice's format (`window_advice.SCHEMA`) and B's information entry in the deployment's registration
  log (`advice.builder_item`, `verity.core.protocols.registration`, under ``--advice-composite``), committed. A serve dir
  that holds one is refused, and so is a second commit of its windows' advice by `verity-vllm advice`: a run registers B once.
* **Commit.** Each closed Commit window's advice (`advice.from_trace`, a window carrying what the one before left) is
  committed as `advice.write_served` commits it, under ``advice/w<k>/`` (``registration.json``, ``private/``), once the window
  is whole: its close and its ``windows.jsonl`` line, every step it scheduled recorded, and the Commit record of every
  request it ran, which the server writes at the close of the window the request finishes in. A window waits on its slowest
  request, so on at most that request's token cap, and the windows are committed in order. Each gets a line of
  ``advice.jsonl`` (`LINES`, public, mirrored): the compiler's digest, the root of the advice's hiding commitment and ℓ_a,
  the registered width, where the advice's own length stays a value (``registration.json`` holds it, in the serve dir); a
  window the advice refuses is a line naming it refused, its reason in ``advice/windows.json``.
* **Coins.** A draw of a window through ``--draws-dir`` before its line is out is refused, and the window's release with it
  (`serve_release.DrawDir`): no coin of a window is drawn before its advice is committed. At the stop the windows left are
  committed (`stop`), a request with no Commit record refused as `write_served` refuses it, before ``summary.json``'s
  ``final``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.core.protocols import registration as REG
from verity_vllm.config import UsageError, option
from verity_vllm.engine import step_trace as ST

from . import advice as ADV
from . import advice_windows as AW
from . import served_compiler as SC
from . import window_advice as WA

REGISTRATION = "advice_registration.json"
LINES = "advice.jsonl"
SCHEMA = "verity-vllm/served-advice-registration/v1"
COMPOSITES = ("onsite", "remote")


@dataclass(frozen=True, kw_only=True)
class AdviceOptions:
    """serve-openai's ``--advice`` group (`refusal`, `ServedAdvice`)."""
    advice: bool = option("--advice", action="boolean_optional", default=False,
                          help="register B (the served compiler, capped at --advice-cap) before the first window and commit each "
                               "Commit window's advice as the server serves, its public line in advice.jsonl before any draw of "
                               "the window (serve_advice)")
    advice_cap: int = option("--advice-cap", type=int, default=AW.CAP_BITS, metavar="BITS",
                             help="ℓ_a, the registered width: the cap on a window's advice in bits, which B's digest covers and "
                                  "each window's public line states in place of the advice's own length (default 2^23)")
    advice_composite: str = option("--advice-composite", choices=COMPOSITES, default="onsite",
                                   help="the deployment whose registration log B's entry opens (default onsite)")
    advice_weights: str | None = option("--advice-weights", default=None, metavar="JSON",
                                        help="the committer's weights registration (prove_path.py weights-registration --out), "
                                             "whose roots each window's registration records once it exists (advice."
                                             "registered_weights; default: none recorded)")
    advice_tau: float = option("--advice-tau", type=float, default=0.1,
                               help="the egress bucket's width in seconds, for each output's release bucket in the advice")


def refusal(opts: Any) -> str | None:
    """Why ``--advice`` can't hold (None: it can): it reads the Commit's windows, and ℓ_a is a whole number of bits, at least 8."""
    a = opts.advising
    if not a.advice:
        return None
    if not opts.commit:
        return "--advice commits each Commit window's advice: it needs the Commit (drop --no-commit)"
    if isinstance(a.advice_cap, bool) or a.advice_cap < 8:
        return f"--advice-cap {a.advice_cap}: ℓ_a is a whole number of bits, at least 8"
    if (Path(opts.out) / REGISTRATION).exists():
        return (f"{Path(opts.out) / REGISTRATION} exists: a run registers B once, so a server doesn't serve into another's serve "
                "dir (give a fresh --out)")
    return None


def register(out: Path, serve: Mapping[str, Any], cap_bits: int, composite: str = "onsite") -> tuple[AW.Capped, dict[str, Any]]:
    """B registered for the run whose ``serve.json`` is ``serve``: ``out/advice_registration.json``, written once (an existing one
    is a `UsageError`). -> B and the document."""
    doc = SC.compiler_doc(serve, cap_bits=cap_bits)
    compiler = ADV.registered_compiler(doc["inner"]["name"], doc["inner"]["digest"], cap_bits)
    if compiler.digest != doc["digest"]:
        raise ValueError(f"B's digest {compiler.digest[:16]}… is not the formula's {doc['digest'][:16]}…")
    log = REG.chain(composite, [REG.Information((ADV.builder_item(compiler),))])
    body = {"schema": SCHEMA, "compiler": doc, "advice": WA.SCHEMA, "cap_bits": int(cap_bits), "row_bits": AW.ROW_BITS,
            "registration": {"composite": composite, "head": REG.head(composite, log),
                             "log": json.loads(REG.committed(composite, log))}}
    try:
        with open(Path(out) / REGISTRATION, "x") as fh:
            fh.write(json.dumps(body, indent=1) + "\n")
    except FileExistsError:
        raise UsageError(f"{Path(out) / REGISTRATION} exists: B is registered once a run") from None
    return compiler, body


def public_line(window: int, reg: Mapping[str, Any], cap_bits: int, t: float) -> dict[str, Any]:
    """Window ``window``'s line of `LINES`: its registration's binding with the advice's length at ℓ_a, the rows ℓ_a fixes, the
    window's Commit and when the line was written (unix seconds)."""
    b = reg["binding"]
    return {"window": int(window), "compiler": dict(b["compiler"]), "advice": {"root": b["advice"]["root"], "bits": int(cap_bits)},
            "rows": -(-int(cap_bits) // AW.ROW_BITS), "commit": reg.get("commit"), "t": t}


class ServedAdvice:
    """The server's committer of its windows' advice (module docstring): `CommitWindows` hands it each window's line
    (`closed`) and each finished request's Commit record (`recorded`); a thread of its own commits every window that is whole,
    in order; `stop` commits the rest. `committed` is what the draw gate reads."""

    def __init__(self, out: Path, trace: Any, *, weights: str | None = None, tau: float = 0.1, T: int = 600,
                 wall: Callable[[], float] = time.time) -> None:
        self.out, self.trace, self.weights, self.tau, self.T, self.wall = Path(out), trace, weights, float(tau), int(T), wall
        self.compiler: AW.Capped | None = None
        self.serve: Mapping[str, Any] | None = None
        self.index = ADV.TraceIndex()
        self.io: dict[str, dict[str, Any]] = {}
        self.commits: dict[int, dict[str, Any]] = {}
        self.carry: tuple[ADV.Left, ...] = ()
        self.gone: int | None = None
        self.done: set[int] = set()                     # every window settled: committed, refused or with no step
        self.lines: dict[int, dict[str, Any]] = {}
        self.refused: dict[int, str] = {}
        self.reports: dict[int, dict[str, Any]] = {}
        self.error: str | None = None
        self._closed: set[int] = set()                   # the index's closed windows as of its last extend, for `summary`
        self._lock = threading.Lock()                    # io, commits and the settled windows, which the sealer's threads add to
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def register(self, serve: Mapping[str, Any], cap_bits: int, composite: str) -> dict[str, Any]:
        """B registered (`register`) from the run's ``serve.json``, and the committer started under it (`start`)."""
        self.serve = serve
        compiler, body = register(self.out, serve, cap_bits, composite)
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

    def committed(self, window: int) -> bool:
        """Whether window ``window``'s advice line is out (refused or committed): what a draw of it waits for."""
        with self._lock:
            return int(window) in self.lines

    # -- the committer
    def _run(self) -> None:
        while True:
            self._wake.wait()
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
                with self._lock:
                    self.done.add(w)
                continue
            ids = {r["request"] for r in sub if r["kind"] == "arrive"}
            with self._lock:
                commit = self.commits.get(w)
                io = {i: q for i, q in self.io.items() if i in ids}
            if not final and (commit is None or ids - set(io)):
                return
            try:
                b = ADV.from_trace(sub, io, tau=self.tau, carry=self.carry)
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
            self._settle(w, public_line(w, rep["registration"], self.compiler.cap_bits, self.wall()), report=rep)

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
        return {"committed": sorted(w for w in lines if w not in refused), "refused": sorted(refused), "waiting": waiting,
                "error": self.error, "passed": self.error is None and not waiting}
