"""serve-openai's tie of each PoUW H1 window to the audited window circuit (``--h1-window-commit CMD``): at the window's
close, before its sample coin, H1 names as its Commit record the audited circuit's registration R, which reuses H1's tables
from the window's registration W, so one record is H1's Commit record and the audited window's commitment.

* **When.** On the window's release thread (`serve_h1.H1Windows`' ``closer``), inside `pouw_h1.H1Run.finish` once H1 has
  closed the window and its tables are fixed (the pouw setting ``commit_record``, `hook`): first the window's seal line
  (`sealed`, which `serve_commit.CommitWindows` calls once the seal is written, or with why there is none), then the advice
  trace's lines flushed (``flush``) and the window's prover state written (``kept``).
* **W.** The window's registration (`pouw_h1_commit.registration`), a ``verity/registration/v2`` record of its served
  tables' entries, chained after the log's last record (``previous``; None for the first), written to
  ``h1/w<k>/registration.json``.
* **R.** ``CMD`` runs as ``bash -c CMD _ SERVE_DIR K W_FILE R_FILE PROVER_DIR`` (``PROVER_DIR`` the window's
  ``--h1-prover`` directory), its output in ``h1/w<k>/window-commit.log``: V3c's window-commit of window K from its sealed
  row dir (``SERVE_DIR/windows/w<K>``), its advice, the weights registration and H1's prover state, which writes to R_FILE
  (``h1/w<k>/record.json``) R, a registration chained right after W that registers each of the window's A tables and
  input-ids tables as its outputs from W (`pouw_h1_commit.reused`).
* **Then.** `pouw_h1_commit.bind` holds R to W and binds it into calls_root, and only then come draw_root and the sample
  coin. The log (``h1/log.jsonl``) names W once written and R once bound, so the next window's W chains after R.
* **Refused, never retried.** A window whose seal failed or never came (the server stopped first), whose release ran on the
  engine loop (which seals it after), whose prover state wasn't written, or whose command failed or wrote no R binds no
  record: no sample coin, the window's visible refusal (`pouw_h1_commit.named`). The command bounds its own time; the
  window's gate sessions bound the wait (each reserve's minutes, the gate's default 120).
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from verity.protocols.spot.verifier.one_stage import registration as REG
from verity_vllm.protocols.pouw.h1 import commit as CT

#: the deployment's registration log under the serve dir's ``h1/``: one line a record, W at its writing and R once bound
LOG = "log.jsonl"
#: each window's files under ``h1/w<k>/``: W, R as the command writes it, and the command's output
W_FILE, R_FILE, COMMAND_LOG = "registration.json", "record.json", "window-commit.log"


def run_command(argv: list[str], log: IO[str]) -> int:
    """``argv`` run to its end, its stdout and stderr into ``log``: its exit code."""
    return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False).returncode


class WindowTie:
    """The tie of each H1 window under serve-openai's ``out``: ``command`` (``--h1-window-commit``) run a window by
    ``runner`` (`run_command`), after ``flush`` (the advice trace's `StepTrace.drain`) when given. ``last`` is the digest of
    the log's last record, which the next W chains after."""

    def __init__(self, command: str, out: Path, *, flush: Callable[[], Any] | None = None,
                 runner: Callable[[list[str], IO[str]], int] = run_command) -> None:
        self.command, self.out, self.flush, self.runner = command, Path(out), flush, runner
        self.log = self.out / "h1" / LOG
        self.last: str | None = None
        if self.log.exists():
            got = [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]
            self.last = got[-1]["digest"] if got else None
        self._cv = threading.Condition()
        self._seals: dict[int, dict[str, Any]] = {}
        self._refused: dict[int, str] = {}
        self._named: dict[int, tuple[Path, str]] = {}
        self._stopped = False

    def describe(self) -> dict[str, Any]:
        return {"log": f"h1/{LOG}", "window": f"h1/w<k>/{W_FILE}", "record": f"h1/w<k>/{R_FILE}"}

    def sealed(self, window: int, line: Mapping[str, Any]) -> None:
        """Window ``window``'s seal line, or why it has none (``error``): its release, waiting in `hook`, goes on."""
        with self._cv:
            self._seals[window] = dict(line)
            self._cv.notify_all()

    def refuse(self, window: int, why: str) -> None:
        """Window ``window`` can't wait for its seal (its release runs on the engine loop): its hook refuses, naming why."""
        with self._cv:
            self._refused[window] = why

    def stop(self) -> None:
        """No seal comes any more (the server's sealer has closed): a window still waiting for one is refused."""
        with self._cv:
            self._stopped = True
            self._cv.notify_all()

    def seal_of(self, window: int) -> dict[str, Any]:
        """Window ``window``'s seal line, once it is in. ValueError when the window can't wait for it, has none, or the
        server stopped before it came."""
        with self._cv:
            if window in self._refused:
                raise ValueError(f"window {window} can't wait for its seal: {self._refused.pop(window)}")
            while window not in self._seals and not self._stopped:
                self._cv.wait()
            line = self._seals.pop(window, None)
        if line is None:
            raise ValueError(f"window {window} has no seal: the server stopped before it was sealed")
        if "error" in line:
            raise ValueError(f"window {window}'s seal failed, so window-commit has no row dir: {line['error']}"[:500])
        return line

    def hook(self, window: int, prover: Path | None, kept: Callable[[], Any] | None) -> CT.CommitRecord:
        """Window ``window``'s pouw setting ``commit_record`` (`pouw_h1_commit.CommitRecord`): its prover directory
        ``prover`` and ``kept``, which waits until its prover state is written."""
        return lambda run: self.record(window, run, prover, kept)

    def record(self, window: int, run: Any, prover: Path | None, kept: Callable[[], Any] | None
               ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Window ``window``'s W and R (the module's order): W written and logged, then ``command`` run on it. ValueError
        (OSError for its prover state) when the window has none."""
        self.seal_of(window)
        if self.flush is not None:
            self.flush()
        if kept is not None:
            kept()
        d = self.out / "h1" / f"w{window}"
        d.mkdir(parents=True, exist_ok=True)
        w, wf, rf = CT.registration(run, self.last), d / W_FILE, d / R_FILE
        wf.write_text(json.dumps(w, indent=1) + "\n")
        self.last = self._logged(window, "registration", wf, REG.digest(w))
        rf.unlink(missing_ok=True)
        argv = ["bash", "-c", self.command, "_", str(self.out), str(window), str(wf), str(rf), str(prover or "")]
        with open(d / COMMAND_LOG, "w") as log:
            rc = self.runner(argv, log)
        if rc != 0:     # its output may name the window's values: the reason, which windows.jsonl carries, only points to it
            raise ValueError(f"--h1-window-commit exited {rc}: its output is in the serve dir's h1/w{window}/{COMMAND_LOG}")
        try:
            r = json.loads(rf.read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"--h1-window-commit wrote no record R to h1/w{window}/{R_FILE}: {e}") from None
        with self._cv:
            self._named[window] = (rf, REG.digest(r) if isinstance(r, Mapping) else "")
        return w, r

    def bound(self, window: int, run: Any) -> None:
        """After window ``window``'s close: R logged as the log's last record once `pouw_h1_commit.bind` bound it."""
        with self._cv:
            named = self._named.pop(window, None)
        record = run.details.get("commit_record")
        if named is not None and isinstance(record, Mapping) and record.get("named"):
            self.last = self._logged(window, "record", *named)

    def _logged(self, window: int, kind: str, path: Path, digest: str) -> str:
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"window": window, "kind": kind, "file": str(path.relative_to(self.log.parent)),
                                 "digest": digest}) + "\n")
        return digest


__all__ = ["COMMAND_LOG", "LOG", "R_FILE", "W_FILE", "WindowTie", "run_command"]
