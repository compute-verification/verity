"""The response tie's verdict as a run's record: what the response tie's verifier
(`verity.protocols.certifier.verifier.response_tie_service`) decided on one class of one execution phase, written
where a run keeps it, for whatever reads the verdicts of a served class from the store.

A record is one JSON object: `run`, the run that produced the verdict; `verifier`, the module that decided it and the
commit it ran at; `verdict`, the tie's own (`TieVerdict.to_json`: its `format`, the `link`, `window` and `class` it
is on, its `head` and its record, `accepted`, `complete` and the refusal's codes); and `proved_by`, null: the
verifier is Python, with no Lean proof behind it, so a reader takes the verdict as run and never as proved. A refused
or incomplete tie is written like an accepted one: the record says what the run produced, and its reader decides what
counts.

`write` appends the record as one line of `FILE` in a directory (a run's `out/`), and keeps the registration the
verdict was decided under as one line of `REGISTRATIONS` beside it, as it was handed (`connection_rows.registration`'s
document), once however many classes' verdicts read it: the record holds only the head's digest, and a reader that
needs the head's counts reads them there, under the network certifier's key. A verdict that names no head was
handed a document that doesn't read, which is not kept. A tie has one verdict: a second for the same class of the
same execution phase of a link is refused, since nothing retries a tie.

The run and the commit are the caller's, or `research run`'s environment's (`RUN`, `COMMIT`): with neither the record
is refused, since one that names no run or no commit says nothing a reader can hold. The record's name in the store
is `RECORD`, which a run takes by label, never from this file: `research data label RUN run_of_record RECORD --by
LANE`."""
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
#: the file beside it that holds the registrations the verdicts were decided under, one a line
REGISTRATIONS = "registrations.jsonl"
#: the environment's names of the run and of the commit its tree is at, as `research run` sets them
RUN, COMMIT = "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA"
KEYS = ("run", "verifier", "verdict", "proved_by")


def _line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


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


def write(out: Path | str, verdict: TS.TieVerdict, registration: Mapping[str, Any], *, run: str | None = None,
          commit: str | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """`record` of `verdict`, appended as one line of `FILE` in the directory `out`, which is made where it is
    missing, with `registration`, the document the verdict was decided under, kept in `REGISTRATIONS` where the
    verdict names its head and the file doesn't hold it yet: `FILE`'s path. ValueError, with nothing written, where
    the record is refused, where `out` already holds a verdict of the same class of the same execution phase of the
    link, or where a verdict that names a head comes with no JSON object."""
    new = record(verdict, run=run, commit=commit, environ=environ)
    out = Path(out)
    tie = (verdict.link, verdict.window, verdict.cls)
    if any((v["link"], v["window"], v["class"]) == tie for v in (r["verdict"] for r in read(out))):
        raise ValueError(f"{FILE} already holds the verdict of class {verdict.cls} of execution phase {verdict.window} "
                         f"of {verdict.link!r}: a tie has one verdict, and nothing retries it")
    kept = None
    if verdict.head is not None:
        if not isinstance(registration, Mapping):
            raise ValueError("a verdict that names its head is kept with the registration it was decided under, the "
                             f"JSON object connection_rows.registration writes, not {type(registration).__name__}")
        kept = _line(dict(registration))
    out.mkdir(parents=True, exist_ok=True)
    if kept is not None and kept not in _lines(out / REGISTRATIONS):
        with (out / REGISTRATIONS).open("a", encoding="utf-8") as f:
            f.write(kept + "\n")
    with (out / FILE).open("a", encoding="utf-8") as f:
        f.write(_line(new) + "\n")
    return out / FILE


def read(out: Path | str) -> list[dict[str, Any]]:
    """The records of `FILE` in the directory `out`, in the order written, none where it has no such file. ValueError
    for a line that is no record."""
    records = [json.loads(line) for line in _lines(Path(out) / FILE)]
    for r in records:
        if not isinstance(r, dict) or tuple(sorted(r)) != tuple(sorted(KEYS)) or not isinstance(r["verdict"], dict) \
                or not {"link", "window", "class"} <= set(r["verdict"]):
            raise ValueError(f"a record holds {KEYS}, its verdict a tie's, not {r!r}")
    return records


def registrations(out: Path | str) -> list[dict[str, Any]]:
    """The registrations kept in `REGISTRATIONS` in the directory `out`, in the order kept. They are documents as
    handed: a reader holds each to the network certifier's key itself (`connection_rows.check_registration`)."""
    return [json.loads(line) for line in _lines(Path(out) / REGISTRATIONS)]


__all__ = ["COMMIT", "FILE", "KEYS", "RECORD", "REGISTRATIONS", "RUN", "read", "record", "registrations", "write"]
