"""The harness kit: the four calls benchmarks' runners share, each written once.

- `lease(gpus, minutes)`: the `gpu-lease` argv a job's command follows, with its bound.
- `host_stamp()`: where a number was measured, as one dict: the card, each GPU's UUID and clocks, the driver, CUDA, the node
  and the commit. With no GPU it says so: `card` is ``"none"``, beside `why`.
- `paired(arm, stock, confidence=0.95)`: an arm's ratio over stock from matched repetitions, its interval and `n`.
- `writer(out).measurement(name, value, unit, ...)`: appends one measurement to the run's result: under a research run
  (`RESEARCH_RUN_DIR` set) `$RESEARCH_RUN_DIR/result.json`, where `research` reads it, and outside one `out/result.json`.
  `writer(out).validation(status, checker, detail, evidence=())` writes the run's validation there, once.

**A result** is `RESULT_SCHEMA`, the form `research` reads a run's `result.json` in: `run_id`, `measurements` and the run's
`validation` (`status`, `checker`, `detail` and `evidence`, the files it stands on, as paths relative to the result's
directory: under a research run, the run's). Each
measurement is the form a reader of numbers takes, `value`, `unit`, `commit` and `run`, with its `name`, its 95% interval
(`ci95`, ``[low, high]``), its `target`, the `verdict` of its check against that target, the `inputs` it was computed from
(other measurements' names, run or `art:` ids) and the `host` stamp it was written under; a measurement with no value
(`value` None) may say why (`why`, held only where given). The writer appends: every
writer of a research run, whatever its `out`, appends to the run's one result and keeps what the run already wrote there.
A result holds one run's measurements, one a name, and one validation, so a rerun writes a fresh directory.

**A measurement is a number, and its verdict is its target check.** Nothing under `benchmarks/` writes a verdict of record or
a seal: a verdict of record is its protocol's verifier's, and a seal its Commit's. So the writer refuses a value that isn't
a number, a verdict with no target, and a verdict in a verifier's or a seal's words (`OF_RECORD`); a target check says
where the value stands against its target (``"within target"``). **A validation is the run's own gates**, never a
protocol's security verdict: its `status` is the schema's ``"passed"`` or ``"failed"`` and nothing else, which is the
whole guard, and its `detail` says in the gates' own words what passed or failed.

Standard library and `verity_bench.settings` only, so a kernel's runner imports it without vLLM or torch.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import math
import numbers
import os
import platform
import re
import shutil
import statistics
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_bench import settings

RESULT_SCHEMA = "research/result/v0.1"
RESULT = "result.json"
#: what the stamp asks nvidia-smi of each GPU, in this order
QUERY = ("name", "uuid", "driver_version", "clocks.sm", "clocks.mem")
#: a query that hasn't answered by then is taken as failed, so a sick GPU can't hold a finished job's lease
QUERY_TIMEOUT_S = 60
#: the words of a verdict of record (a verifier's decision, a stage's PASS or FAIL) and of a seal, each in every form a
#: verdict takes (the verb's and its noun's), which no measurement's verdict uses
OF_RECORD = frozenset("accept accepts accepted accepting acceptance reject rejects rejected rejecting rejection "
                      "pass passes passed passing fail fails failed failing failure seal seals sealed sealing".split())
#: the statuses of a validation the writer writes: whether the run's own gates held (the schema's third, ``not_run``, is a
#: result with no validation)
VALIDATION_STATUSES = ("passed", "failed")


# ---- the lease ---------------------------------------------------------------------------------------------------------------


def lease(gpus: int, minutes: float) -> list[str]:
    """The `gpu-lease` argv a job's command follows: the wait for `gpus` GPUs of the node, held for the job's bound of
    `minutes` (gpu-lease's `--max-min`, in whole minutes). The job takes the lease itself, since `research run --gpus` is
    refused on a node whose GPUs gpu-lease doesn't hand out."""

    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1:
        raise ValueError(f"a lease holds one GPU or more, not {gpus!r}")
    if not (isinstance(minutes, numbers.Real) and math.isfinite(minutes) and minutes > 0):
        raise ValueError(f"a lease needs its bound in minutes, not {minutes!r}")
    return ["gpu-lease", str(gpus), "--wait", "--max-min", str(math.ceil(minutes)), "--"]


# ---- the host stamp ----------------------------------------------------------------------------------------------------------


def _run(argv: Sequence[str]) -> tuple[int, str]:
    """A query's exit and what it printed (its stderr when it failed); 127 and why when it couldn't run or didn't answer."""

    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=QUERY_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as ex:
        return 127, f"{type(ex).__name__}: {ex}"
    return r.returncode, r.stdout if r.returncode == 0 else r.stderr or r.stdout


def _commit() -> str | None:
    """The commit of the tree the job runs: the one `research run` shipped, else this checkout's HEAD; None where neither
    says."""

    if sha := settings.RESEARCH_SOURCE_SHA:
        return sha
    rc, out = _run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"])
    return out.strip() or None if rc == 0 else None


def _mhz(cell: str) -> int | None:
    return int(cell) if cell.isdigit() else None


def _gpus() -> tuple[list[dict], str | None]:
    """Each GPU's row of `QUERY` (the lease's GPUs where the job holds one, else every GPU of the host), or none and why."""

    if shutil.which("nvidia-smi") is None:
        return [], "no nvidia-smi on PATH"
    leased = settings.GPU_LEASE_UUID
    rc, out = _run(["nvidia-smi", *(["-i", leased] if leased else []), f"--query-gpu={','.join(QUERY)}",
                    "--format=csv,noheader,nounits"])
    if rc:
        return [], f"nvidia-smi exited {rc}: {' '.join(out.split())[:200]}"
    rows = [row for row in csv.reader(out.splitlines(), skipinitialspace=True) if row]
    if bad := [row for row in rows if len(row) != len(QUERY)]:
        return [], f"nvidia-smi answered {', '.join(bad[0])!r}, not {', '.join(QUERY)}"
    if not rows:
        return [], "nvidia-smi lists no GPU"
    return [dict(zip(QUERY, (cell.strip() for cell in row), strict=True)) for row in rows], None


def host_stamp() -> dict:
    """Where a number was measured, as one dict a result carries: `card` (the GPUs' name), `gpus` (each one's `uuid` and its
    SM and memory clocks as nvidia-smi reads them then, `sm_mhz` and `mem_mhz`), `driver`, `cuda` (the newest CUDA the driver
    serves, from nvidia-smi's header; a build's toolkit is its build record's), `node` and `commit`. The GPUs are the job's
    lease's (`GPU_LEASE_UUID`), else every GPU of the host. With no GPU, `card` is ``"none"``, `gpus` is empty and `why`
    says how it knows: a host with none stamps like any other, and a GPU job's stamp can't pass for one."""

    gpus, why = _gpus()
    cuda = None
    if gpus:
        rc, out = _run(["nvidia-smi"])
        found = re.search(r"CUDA Version:\s*([0-9.]+)", out) if rc == 0 else None
        cuda = found.group(1) if found else None
    stamp: dict = {
        "card": ", ".join(dict.fromkeys(g["name"] for g in gpus)) or "none",
        "gpus": [{"uuid": g["uuid"], "sm_mhz": _mhz(g["clocks.sm"]), "mem_mhz": _mhz(g["clocks.mem"])} for g in gpus],
        "driver": ", ".join(dict.fromkeys(g["driver_version"] for g in gpus)) or None,
        "cuda": cuda, "node": platform.node(), "commit": _commit()}
    if why:
        stamp["why"] = why
    return stamp


# ---- the paired ratio --------------------------------------------------------------------------------------------------------


def _order_statistics(n: int, confidence: float) -> tuple[int, int] | None:
    """The 1-based order statistics (l, u) whose span covers the median with at least `confidence`: the largest l with
    P(Binomial(n, 1/2) < l) at most (1 - confidence) / 2, and u = n + 1 - l (the 6th and 15th of 20 at 0.95); None when n
    reps can't bound the median at `confidence`."""

    tail, l, cdf = (1 - confidence) / 2, 0, 0.0
    while l < n:
        nxt = cdf + math.comb(n, l) / 2**n
        if nxt > tail:
            break
        cdf, l = nxt, l + 1
    return (l, n + 1 - l) if l else None


def paired(arm: Sequence[float], stock: Sequence[float], confidence: float = 0.95) -> dict:
    """An arm's cost over stock's from matched repetitions (rep i of the arm beside rep i of stock, each one cost): `ratio`,
    the median of the reps' ratios; `gamma`, the ratio less one, the arm's overhead; `n`; the sorted `ratios`; the
    overhead's interval at `confidence` from the ratios' order statistics, exact sign-test bounds on the median, under the
    key of its level (`ci95` at 0.95), None when `n` reps can't bound the median at it; and `half_width`, the interval's
    half-width as a share of the median ratio, None with the interval and where that ratio is 0 (most of the arm's reps
    cost nothing), which nothing is a share of."""

    if not 0 < confidence < 1:
        raise ValueError(f"a confidence is between 0 and 1, not {confidence!r}")
    if len(arm) != len(stock) or not arm:
        raise ValueError(f"paired reps go one for one: {len(arm)} arm reps beside {len(stock)} stock reps")
    if any(s <= 0 for s in stock):
        raise ValueError("a stock rep cost nothing: a ratio needs every stock rep above zero")
    ratios = sorted(a / b for a, b in zip(arm, stock, strict=True))
    median = statistics.median(ratios)
    ci = f"ci{confidence * 100:g}"
    out: dict = {"gamma": median - 1, "ratio": median, "n": len(ratios), "ratios": ratios, ci: None, "half_width": None}
    if (bounds := _order_statistics(len(ratios), confidence)) is None:
        return out
    lo, hi = bounds
    out[ci] = [ratios[lo - 1] - 1, ratios[hi - 1] - 1]
    if median:
        out["half_width"] = max(median - ratios[lo - 1], ratios[hi - 1] - median) / median
    return out


# ---- the writer --------------------------------------------------------------------------------------------------------------


def _number(what: str, x: Any) -> int | float:
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x):
        raise ValueError(f"a measurement's {what} is a finite number, not {x!r}: a verdict of record or a seal is never a "
                         "measurement, and nothing under benchmarks/ writes one")
    return int(x) if isinstance(x, numbers.Integral) else float(x)


def _target_check(verdict: Any, target: Any) -> str | None:
    """A measurement's verdict, or ValueError where it isn't the check of its value against its target."""

    if verdict is None:
        return None
    if not isinstance(verdict, str) or not verdict.strip():
        raise ValueError(f"a measurement's verdict says in words where its value stands against its target, not {verdict!r}")
    if words := sorted(set(re.findall(r"[a-z]+", verdict.casefold())) & OF_RECORD):
        raise ValueError(f"the verdict {verdict!r} reads as a verdict of record or a seal ({', '.join(words)}), which "
                         "nothing under benchmarks/ writes: say where the value stands against its target (\"within "
                         "target\")")
    if target is None:
        raise ValueError(f"the verdict {verdict!r} has no target: a measurement's verdict is its check against `target`, and "
                         "a verdict of record is its verifier's to write")
    return verdict


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """One appender at a time in `directory`: two processes of a job each keep their measurement."""

    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Writer:
    """`writer(out)`: the one writer of `out/result.json`, for the run `run` under the host stamp `stamp`. `out` is the
    directory the result is in: under a research run the run's own, whatever `writer` was given."""

    out: Path
    run: str
    stamp: Mapping

    def measurement(self, name: str, value: float | None, unit: str, interval: Sequence[float] | None = None,
                    target: float | None = None, verdict: str | None = None, inputs: Iterable[str] = (),
                    why: str | None = None) -> dict:
        """Appends one measurement to `out/result.json` and returns it: `value` in `unit` (None where the benchmark has no
        finite value to give, and then `why`, in words, where given), its 95% `interval` (low, high), which the result
        holds as `ci95`, the key `paired` gives it under, its `target`, the `verdict` of its check against that target, and
        the `inputs` it was computed from, with this run, its commit and the host stamp. It refuses what isn't a
        measurement (a value that isn't a number, a verdict with no target or in a verdict of record's or a seal's words,
        a `why` beside a value), a name the result already holds, and a result of another run."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"a measurement has a name, not {name!r}")
        if not isinstance(unit, str):
            raise ValueError(f"{name}'s unit is a string, not {unit!r}")
        if why is not None and (value is not None or not isinstance(why, str) or not why.strip()):
            raise ValueError(f"{name}'s why says in words why it has no value, beside a value of None, not {why!r} beside "
                             f"{value!r}")
        inputs = [inputs] if isinstance(inputs, (str, bytes)) else list(inputs)
        if not all(isinstance(i, str) and i for i in inputs):
            raise ValueError(f"{name}'s inputs are the names of what it was computed from, each a string, not {inputs!r}")
        if interval is not None:
            try:
                low, high = interval
            except (TypeError, ValueError):
                raise ValueError(f"{name}'s interval is (low, high), not {interval!r}") from None
            if (low := _number("interval", low)) > (high := _number("interval", high)):
                raise ValueError(f"{name}'s interval is (low, high), not {interval!r}")
            interval = [low, high]
        entry = {"name": name, "value": None if value is None else _number("value", value), "unit": unit,
                 "commit": self.stamp["commit"], "run": self.run, "ci95": interval,
                 "target": None if target is None else _number("target", target),
                 "verdict": _target_check(verdict, target), "inputs": inputs, "host": dict(self.stamp)}
        if why is not None:
            entry["why"] = why
        with self._open() as (path, doc):
            if any(m.get("name") == name for m in doc["measurements"]):
                raise ValueError(f"{path} already holds a measurement named {name}: a result holds one a name (a rerun writes "
                                 "a fresh directory)")
            doc["measurements"].append(entry)
        return entry

    def validation(self, status: str, checker: str, detail: str, evidence: Sequence[str] = ()) -> dict:
        """Writes the run's validation into `out/result.json` and returns it: `status`, ``"passed"`` or ``"failed"``, whether
        the run's own gates held; the `checker` that says so, by name; its `detail` in words; and the `evidence`, the files
        it stands on, as paths relative to `out`, in a list. It refuses a second validation in one result, any other status,
        a checker or a detail that isn't a string, evidence that isn't a list of relative paths inside `out`, and a result
        of another run. The status is the whole guard against a protocol's security verdict, which is its verifier's to
        write: a detail says what passed or failed in the gates' own words, and the writer takes any."""

        if not isinstance(status, str) or status not in VALIDATION_STATUSES:
            raise ValueError(f"a validation's status is {' or '.join(map(repr, VALIDATION_STATUSES))}, whether the run's own "
                             f"gates held, not {status!r}: a verdict of record or a seal is never a run's validation, and "
                             "nothing under benchmarks/ writes one")
        for what, text in (("checker", checker), ("detail", detail)):
            if not isinstance(text, str):
                raise ValueError(f"a validation's {what} is a string, not {text!r}")
        if not isinstance(evidence, (list, tuple)):
            raise ValueError(f"a validation's evidence is a list of paths relative to {self.out}, not {evidence!r}")
        for rel in evidence:
            if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError(f"a validation's evidence is a list of paths relative to {self.out} and inside it, each a "
                                 f"string, not {rel!r}")
        block = {"status": status, "checker": checker, "detail": detail, "evidence": list(evidence)}
        with self._open() as (path, doc):
            if doc.get("validation") is not None:
                raise ValueError(f"{path} already holds a validation: a result holds one, the run's own (a rerun writes a "
                                 "fresh directory)")
            doc["validation"] = block
        return block

    @contextlib.contextmanager
    def _open(self) -> Iterator[tuple[Path, dict]]:
        """`out/result.json`'s path and this run's result to change, under `out`'s lock: what the caller leaves in the
        result replaces the file whole, and a caller that raises leaves the file as it was."""

        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / RESULT
        with _locked(self.out):
            doc = self._read(path)
            yield path, doc
            tmp = path.with_name(f"{RESULT}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)

    def _read(self, path: Path) -> dict:
        """The result at `path` to append to: this run's, or a new one where there is none."""

        if not path.exists():
            return {"schema": RESULT_SCHEMA, "run_id": self.run, "measurements": []}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as ex:
            raise ValueError(f"{path} isn't JSON ({ex}): the writer appends only to a {RESULT_SCHEMA} result") from ex
        if not isinstance(doc, dict) or doc.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"{path} isn't a {RESULT_SCHEMA} result: the writer appends only to one")
        if doc.get("run_id") != self.run:
            raise ValueError(f"{path} holds run {doc.get('run_id')}'s measurements, and this run is {self.run}: a run writes "
                             "its own directory")
        if not isinstance(doc.setdefault("measurements", []), list):
            raise ValueError(f"{path}'s measurements aren't a list: the writer appends only to a {RESULT_SCHEMA} result")
        return doc


def writer(out: str | os.PathLike) -> Writer:
    """The writer of this run's result (`RESEARCH_RUN_ID`, ``local`` outside one), stamped with the host now. Under a
    research run (`RESEARCH_RUN_DIR` set) the result is `$RESEARCH_RUN_DIR/result.json`, the one file `research` reads a
    run's result from, whatever `out` is, so its evidence paths are relative to the run's directory; outside one it is
    `out/result.json`."""

    where = settings.RESEARCH_RUN_DIR if settings.is_set("RESEARCH_RUN_DIR") else out
    return Writer(Path(where), settings.RESEARCH_RUN_ID, host_stamp())
