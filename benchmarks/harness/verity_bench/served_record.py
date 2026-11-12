"""The reader of a served two-stage run's record: what `verity bench exfiltration --served-run` takes
(`exfiltration.served_share`), read from the files the run wrote. It computes no share. A field the files don't hold is
None in the record and is named, with the file read for it and what that file holds instead (`read`'s second value),
so nothing stands in for a time or a count of GPUs no run wrote.

    python -m verity_bench.served_record --second DIR [--run ID] [--serving META] [--stock META] [--replay FILE]
                                         [--seconds busy_gpu_s|session_s|prove_total_s]
                                         [--serving-cost C --serving-cost-source S]

* **The audit's parts** are read from the second stage's summary (`benchmarks/one_stage/two_stage.py` ``second``,
  under its ``--out``): ``summary.json``, or ``second.json``, the same object as a served run publishes it, its
  verifier-side fields alone (`summary_file`).

  - ``prove``, one entry a session: ``audit.timing``, by member, holds each session's seconds (`SECONDS`).
    ``busy_gpu_s`` is the GPU-seconds the session's cards were busy, summed over them: the entry is that over the
    session's ``gpus``, on those GPUs. It is taken wherever every session holds it. ``session_s`` is the audit's own
    clock on the session: on the audit's proof server, from the statement's submission to both of its records, which
    counts the session's wait behind the others of its prover; where a session is a process pair of its own, around
    that pair's run. It leaves out no time the session's prover ran, so it is at most high, and it is taken where the
    busy seconds are not written. ``timing.prove_total_s`` is the prover's own timer summed over the session's
    tables, each from the start of that table's proving to its proof: it leaves out the witness, the commitment's
    round trips and the wait for the challenger's coins, and is taken only where asked. `reading` says which a
    record took. The GPUs a session held are its ``gpus``. The audit's proof server proves on the CPU unless it is
    given devices, and the command takes an entry only on a whole count of GPUs, 1 or more: a session proved on the
    CPU has no entry in this record's form.
  - ``staging``: ``audit.staging``'s ``before_draw_s`` and ``drawn_s``, with its ``gpus``. The one-stage audit times
    both in its own summary, and the two-stage summary keeps neither.
  - ``replay`` and ``interior``: ``replay`` and ``interior_commit``, each ``{"seconds", "gpus"}``. The two-stage audit
    reads no clock over either. The replay of the sampled spans is a run of its own on the serving node, and its
    ``summary.json`` (``--replay``) holds its seconds (``replay_wall_s``) and, where written, its ``gpus``.

* **The window's serving**, and stock vLLM's on the same load, are not in the audit's files: each is a load's
  ``meta.json`` of `verity bench serve_tax` (``--serving``, ``--stock``), its ``wall_s`` and its ``gpu.gpus``.
  Without ``--stock``, a serving cost and its source (``--serving-cost``, ``--serving-cost-source``) go in the
  record as given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: the record's fields, as `exfiltration.served_share` reads them
FIELDS = ("run", "prove", "staging", "replay", "interior", "serving", "stock")
#: a session's seconds in ``audit.timing``: the GPU-seconds its cards were busy, the audit's clock on the whole session,
#: or the prover's own timer over its tables
BUSY = "busy_gpu_s"
SECONDS = (BUSY, "session_s", "prove_total_s")
SUMMARY = "summary.json"
#: the summary as a served run publishes it: its verifier-side fields alone
PUBLISHED = "second.json"


def _entry(seconds: Any, gpus: Any) -> dict:
    return {"seconds": seconds, "gpus": gpus}


def _load(serving: Path | None, field: str, absent: list[dict]) -> dict | None:
    """A load's entry from `verity bench serve_tax`'s ``meta.json``: the seconds it was served over and the GPUs its
    engine held."""

    if serving is None:
        absent.append({"field": field, "file": None, "why": "no `meta.json` of the load is given: the audit's files don't "
                       "hold the serve"})
        return None
    meta = json.loads(serving.read_text())
    entry = _entry(meta.get("wall_s"), (meta.get("gpu") or {}).get("gpus"))
    absent += [{"field": f"{field}.{k}", "file": str(serving), "why": why} for k, why in (
        ("seconds", "the load's `meta.json` holds no `wall_s`"),
        ("gpus", "the load's `meta.json` holds no `gpu.gpus`: it was served without NVML's bindings")) if entry[k] is None]
    return entry


def summary_file(second: Path) -> Path:
    """A second stage's summary: ``second`` itself where it is a file, else its `SUMMARY` or its `PUBLISHED`."""

    second = Path(second)
    if second.is_file():
        return second
    for name in (SUMMARY, PUBLISHED):
        if (second / name).is_file():
            return second / name
    raise ValueError(f"{second} holds no `{SUMMARY}` and no `{PUBLISHED}`: it is not a second stage's output")


def reading(timing: Any, seconds: str | None = None) -> str:
    """Which of a session's seconds a record takes of ``timing`` (``audit.timing``): ``seconds`` where one is asked
    for; else the busy seconds of the session's cards where every session holds them and its GPUs; else the audit's
    clock on the session."""

    if seconds is not None:
        if seconds not in SECONDS:
            raise ValueError(f"a session's seconds are one of {', '.join(SECONDS)}, not {seconds!r}")
        return seconds
    sessions = list(timing.values()) if isinstance(timing, Mapping) else []
    return BUSY if sessions and all(None not in (s.get(BUSY), s.get("gpus")) for s in sessions) else "session_s"


def _seconds(session: Mapping, took: str) -> Any:
    if took == BUSY:
        busy, gpus = session.get(BUSY), session.get("gpus")
        return None if None in (busy, gpus) or not gpus else busy / gpus
    return session.get(took) if took == "session_s" else (session.get("timing") or {}).get(took)


def read(second: Path, *, run: str | None = None, serving: Path | None = None, stock: Path | None = None,
         seconds: str | None = None, replay: Path | None = None, serving_cost: float | None = None,
         serving_cost_source: str | None = None) -> tuple[dict, list[dict]]:
    """The record of the served two-stage run whose second stage wrote ``second``, and each field it lacks: its name
    as `exfiltration.served_share` names it, the file read for it, and why the file doesn't hold it."""

    file = summary_file(second)
    summary = json.loads(file.read_text())
    audit = summary.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"{file} holds no `audit`: its second stage stopped before it proved its sample")
    absent: list[dict] = []

    def lacks(field: str, why: str) -> None:
        absent.append({"field": field, "file": str(file), "why": why})

    if run is None:
        absent.append({"field": "run", "file": None, "why": "the run's id is not in its files: give it (--run)"})

    timing = audit.get("timing")
    prove = None
    if not isinstance(timing, Mapping) or not timing:
        lacks("prove", "`audit.timing` holds no session")
    else:
        took = reading(timing, seconds)
        prove = [_entry(_seconds(s, took), s.get("gpus")) for _, s in sorted(timing.items())]
        if any(e["seconds"] is None for e in prove):
            lacks("prove.seconds", f"`audit.timing` holds no `{took if took != SECONDS[2] else 'timing.' + took}` for a "
                  "session")
        if any(e["gpus"] is None for e in prove):
            lacks("prove.gpus", "`audit.timing` holds each session's seconds and not the GPUs it held (`gpus`)")

    staged = audit.get("staging")
    staging = None
    if not isinstance(staged, Mapping):
        lacks("staging", "`audit.staging` is absent: the one-stage audit times its staging and the two-stage summary "
              "keeps `audit.timing` alone")
    else:
        took = [staged.get(k) for k in ("before_draw_s", "drawn_s")]
        staging = [_entry(None if None in took else sum(took), staged.get("gpus"))]
        if None in took:
            lacks("staging.seconds", "`audit.staging` holds no `before_draw_s` or no `drawn_s`")
        if staged.get("gpus") is None:
            lacks("staging.gpus", "`audit.staging` holds its seconds and not the GPUs it held (`gpus`)")

    parts = {}
    for field, key in (("replay", "replay"), ("interior", "interior_commit")):
        held = summary.get(key)
        parts[field] = None
        if field == "replay" and not isinstance(held, Mapping) and replay is not None:
            own = json.loads(Path(replay).read_text())
            parts[field] = [_entry(own.get("replay_wall_s"), own.get("gpus"))]
            absent += [{"field": f"replay.{k}", "file": str(replay), "why": why} for k, why in (
                ("seconds", "the replay's `summary.json` holds no `replay_wall_s`"),
                ("gpus", "the replay's `summary.json` holds its seconds (`replay_wall_s`) and not the GPUs its engine "
                         "held (`gpus`)")) if parts[field][0][k] is None]
        elif not isinstance(held, Mapping):
            lacks(field, f"`{key}` is absent: the two-stage audit reads no clock over it")
        else:
            parts[field] = [_entry(held.get("seconds"), held.get("gpus"))]
            for k in ("seconds", "gpus"):
                if held.get(k) is None:
                    lacks(f"{field}.{k}", f"`{key}` holds no `{k}`")

    record = {"run": run, "prove": prove, "staging": staging, **parts,
              "serving": _load(serving, "serving", absent), "stock": _load(stock, "stock", absent)}
    if record["stock"] is None and serving_cost is not None:
        record |= {"serving_cost": serving_cost, "serving_cost_source": serving_cost_source}
    return record, absent


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m verity_bench.served_record",
        description="A served two-stage run's record as `verity bench exfiltration --served-run` reads it, on stdout as "
                    "one JSON object; each field the run's files lack is None there and is named on stderr.")
    ap.add_argument("--second", type=Path, required=True, metavar="DIR",
                    help="the second stage's output (its --out), or its `summary.json` or published `second.json`")
    ap.add_argument("--run", help="the served run's id")
    ap.add_argument("--serving", type=Path, metavar="META", help="the served load's `meta.json` (`verity bench serve_tax`)")
    ap.add_argument("--stock", type=Path, metavar="META", help="stock vLLM's `meta.json` on the same load")
    ap.add_argument("--replay", type=Path, metavar="FILE", help="the `summary.json` of the replay of the sampled spans")
    ap.add_argument("--seconds", choices=SECONDS,
                    help="a session's seconds: the busy seconds of its cards, the audit's clock on the whole session or "
                         "the prover's own timer over its tables (by default the first where every session holds it, "
                         "else the second)")
    ap.add_argument("--serving-cost", type=float, metavar="C",
                    help="without --stock: the serving cost of the served arm over stock, a fraction")
    ap.add_argument("--serving-cost-source", metavar="S", help="the run and load the serving cost is of")
    a = ap.parse_args(argv)
    try:
        record, absent = read(a.second, run=a.run, serving=a.serving, stock=a.stock, seconds=a.seconds, replay=a.replay,
                              serving_cost=a.serving_cost, serving_cost_source=a.serving_cost_source)
    except (ValueError, OSError) as e:
        print(f"served_record: {e}", file=sys.stderr)
        return 2
    for gap in absent:
        print(f"served_record: the record lacks `{gap['field']}`: {gap['why']}" + (f" ({gap['file']})" if gap["file"] else ""),
              file=sys.stderr)
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
