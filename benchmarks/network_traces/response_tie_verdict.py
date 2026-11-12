"""The response tie's verdict as a run's record: what the response tie's verifier
(`verity.protocols.certifier.verifier.response_tie_service`) decided on one class of one execution phase, written
where a run keeps it, for whatever reads the verdicts of a served class from the store.

A record is one JSON object: `run`, the run that produced the verdict; `verifier`, the module that decided it and the
commit it ran at; `verdict`, the tie's own (`TieVerdict.to_json`: its head and its record, `accepted`, `complete` and
the refusal's codes); and `proved_by`, null: the verifier is Python, with no Lean proof behind it, so a reader takes
the verdict as run and never as proved. A refused or incomplete tie is written like an accepted one: the record says
what the run produced, and its reader decides what counts.

`write` appends the record as one line of `FILE` in a directory (a run's `out/`), one line a tie. The run and the
commit are the caller's, or `research run`'s environment's (`RUN`, `COMMIT`): with neither the record is refused,
since one that names no run or no commit says nothing a reader can hold. The record's name in the store is `RECORD`,
which a run takes by label, never from this file: `research data label RUN run_of_record RECORD --by LANE`."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from verity.protocols.certifier.verifier import response_tie_service as TS

#: the record's name in the store, a run's `run_of_record` label
RECORD = "exfiltration-glm53-stop-verdicts"
#: the file of a run's directory that holds its ties' records, one JSON object a line
FILE = "response-tie-verdicts.jsonl"
#: the environment's names of the run and of the commit its tree is at, as `research run` sets them
RUN, COMMIT = "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA"
KEYS = ("run", "verifier", "verdict", "proved_by")


def record(verdict: TS.TieVerdict, *, run: str | None = None, commit: str | None = None,
           environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The record of `verdict`: the run and the commit given, or else the environment's (`environ`, the process's by
    default). ValueError where `verdict` is no tie's verdict, or the run or the commit is neither given nor set."""
    if type(verdict) is not TS.TieVerdict:
        raise ValueError(f"a record is of a response tie's verdict (response_tie_service.TieVerdict), not "
                         f"{type(verdict).__name__}")
    environ = os.environ if environ is None else environ
    run, commit = run or environ.get(RUN), commit or environ.get(COMMIT)
    if not (isinstance(run, str) and run):
        raise ValueError(f"the record names the run that produced the verdict: pass run=, or run under `research run`, "
                         f"which sets {RUN}")
    if not (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError(f"the record names the commit the verifier ran at, 40 hex digits: pass commit=, or run under "
                         f"`research run --source`, which sets {COMMIT}; not {commit!r}")
    return {"run": run, "verifier": {"module": TS.__name__, "commit": commit}, "verdict": verdict.to_json(),
            "proved_by": None}


def write(out: Path | str, verdict: TS.TieVerdict, *, run: str | None = None, commit: str | None = None,
          environ: Mapping[str, str] | None = None) -> Path:
    """`record` of `verdict`, appended as one line of `FILE` in the directory `out`, which is made where it is
    missing: the file's path."""
    line = json.dumps(record(verdict, run=run, commit=commit, environ=environ), sort_keys=True, separators=(",", ":"))
    path = Path(out) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def read(out: Path | str) -> list[dict[str, Any]]:
    """The records of `FILE` in the directory `out`, in the order written. ValueError for a line that is no record."""
    records = [json.loads(line) for line in (Path(out) / FILE).read_text(encoding="utf-8").splitlines()]
    for r in records:
        if not isinstance(r, dict) or tuple(sorted(r)) != tuple(sorted(KEYS)):
            raise ValueError(f"a record holds {KEYS}, not {r!r}")
    return records


__all__ = ["COMMIT", "FILE", "KEYS", "RECORD", "RUN", "read", "record", "write"]
