"""The reader of a served two-stage run's record: what `verity bench exfiltration --served-run` takes, read from the
files the run's runs published. It computes no share and decides no verdict. A field the files don't hold is None in
the record and is named, with the file read for it and what that file holds instead (`read`'s second value), so
nothing stands in for a time, a count of GPUs or a verdict no run wrote.

    python -m verity_bench.served_record --second DIR [--first DIR] [--serve DIR] [--replay DIR] [--run ID]
                                         [--serving META] [--stock META]
                                         [--serving-cost C --serving-cost-source S] > record.json
    verity bench exfiltration --served-run record.json

A served run is four runs, and each ``DIR`` is one's directory as `research fetch --all RUN` leaves it, or as its run
record is fetched from the store: its ``job.json`` beside its ``out/`` (a run's ``out/``, or the file itself, is
taken too, and then the run's id and commit are not read). Only the second stage is required.

* The **second stage** publishes ``second.json``, its summary's verifier-side fields (a stand-in run, which serves
  nothing, writes ``summary.json`` whole): `SECOND`.
* The **first stage** publishes ``stage1.json`` (the class's commitment and layout, the rule, the registration, the
  sample) and ``check.txt`` (the check of its record and its coins): `STAGE1`, `CHECK`.
* The **serve** publishes each class's ``commit.json`` (`COMMITS`). Its load (`verity bench serve_tax`'s
  ``meta.json``: its ``wall_s`` and its ``gpu.gpus``) stays on the serving node; where it can be read it is
  ``--serving``, and stock vLLM's on the same load ``--stock``. Without ``--stock``, a serving cost and its source
  (``--serving-cost``, ``--serving-cost-source``) go in the record as given.
* The **replay** of the sampled spans publishes its own seconds (``replay_wall_s``) and, where written, the GPUs its
  engine held (``gpus``): `REPLAY`.

The record is `exfiltration.served_share`'s object, with what the audit showed beside it:

* ``prove``, one entry a session, from ``audit.timing`` by member: its seconds and its ``gpus``. Each session's
  `SOURCE` says which seconds are its proving's (`SOURCES`): ``busy``, the proof server's own clock on the session's
  prover (``busy_s``), or ``session``, the audit's clock on the whole session (``session_s``), which counts the
  session's wait behind the others of its prover and so cannot read low. The record's `SOURCE` is the sessions' own
  where they agree, ``session`` where some session's is, and None where a session names none: the seconds are then
  ``session_s``, and the field is named as absent.
* ``staging``: ``audit.staging``'s ``before_draw_s`` and ``drawn_s``, with its ``gpus``. ``replay`` and
  ``interior``: the summary's ``replay`` and ``interior_commit``, each ``{"seconds", "gpus"}``; the replay's else from
  its own run's file. A count of 0 GPUs is kept as written: that part started no process on a card.
* ``run`` and ``commit``, the second stage's run and the commit it ran; ``runs``, each of the four runs' id and
  commit (`run_of`).
* ``audit``, what the audit showed (`shown`): the sessions' verdicts counted by kind, the driver's checks, the
  stream's outcome and its codes, the layout the serve committed, what the interior registered, the first stage's
  sample and check, what holds the runs' files to each other (`ties`), the sessions' seconds as the files hold
  them, and the cards the proof server's provers ran on (``audit.proof_server.busy``, each prover's ``device``, its
  ``sessions`` and its ``seconds``; a session's own ``device``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: the record's fields, as `exfiltration.served_share` reads them
FIELDS = ("run", "prove", "staging", "replay", "interior", "serving", "stock")
#: which seconds are a session's proving's, as the driver writes it in ``audit.timing`` and as the record passes it on
SOURCE = "prove_seconds_source"
#: a session's seconds by source: the proof server's own clock on its prover, or the audit's clock on the whole session
SOURCES = {"busy": "busy_s", "session": "session_s"}
#: where a run's directory holds each file, the first that exists
SECOND = ("out/second.json", "out/second/summary.json", "second/summary.json", "second.json", "summary.json")
STAGE1 = ("out/stage1/*/stage1.json", "out/first/*/stage1.json", "stage1/*/stage1.json", "first/*/stage1.json",
          "stage1.json")
CHECK = ("out/check.txt", "check.txt")
COMMITS = ("out/boundary/*/*/commit.json", "boundary/*/*/commit.json", "commit.json")
REPLAY = ("out/replay.json", "out/summary.json", "replay.json", "summary.json")
MACHINE = "machine.txt"
#: the layout's fields the record keeps, the committed values by name
LAYOUT = ("check", "replay", "per_prompt", "per_output", "inline", "prompt", "outputs", "spans", "window", "stop", "ids",
          "tables")
#: the stream's outcome's lists of codes; ``interior`` names the lists that hold one
CODES = ("interior", "replay_sample", "statements")
_RUN = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _entry(seconds: Any, gpus: Any) -> dict:
    return {"seconds": seconds, "gpus": gpus}


def _json(path: Path | None) -> Any:
    return None if path is None else json.loads(path.read_text())


def _find(directory: Path | None, patterns: Sequence[str], what: str) -> Path | None:
    """A run's file: ``directory`` itself where it is a file, else the first of ``patterns`` it holds."""

    if directory is None:
        return None
    directory = Path(directory)
    if directory.is_file():
        return directory
    for pattern in patterns:
        if found := sorted(directory.glob(pattern)):
            return found[0]
    raise ValueError(f"{directory} holds none of {', '.join(patterns)}: it is not {what}")


def run_of(directory: Path | None) -> dict | None:
    """A run's id, the commit it ran and the machine it ran on, from its own record (``job.json``), else from the
    directory's name and the commit its ``machine.txt`` holds; ``why`` says what names no commit. None for no
    directory."""

    if directory is None:
        return None
    directory = Path(directory)
    out = {"run": directory.name if _RUN.match(directory.name) else None, "commit": None, "dirty": False, "why": None,
           "machine": None}
    job = directory / "job.json"
    if job.is_file():
        doc = json.loads(job.read_text())
        run, source = doc.get("run") or {}, ((doc.get("store") or {}).get("conditions") or {}).get("source") or {}
        out["run"], out["dirty"] = run.get("id") or out["run"], bool(source.get("dirty"))
        out["machine"] = (run.get("remote") or {}).get("machine")
        for commit in (source.get("commit"), (run.get("remote") or {}).get("source_sha"),
                       (run.get("cwd_clone") or {}).get("commit"), (doc.get("computation") or {}).get("tree_sha")):
            if isinstance(commit, str) and _COMMIT.match(commit):
                out["commit"] = commit
                break
    machine = next((p for p in (directory / "out" / MACHINE, directory / MACHINE) if p.is_file()), None)
    if out["commit"] is None and machine is not None:
        out["commit"] = next((word for word in machine.read_text().split() if _COMMIT.match(word)), None)
    if out["commit"] is None:
        out["why"] = ("its record names no commit" if job.is_file() else
                      "its directory holds no `job.json`" if directory.is_dir() else "a file is given, not the run's directory")
    return out


def check_of(path: Path | None) -> dict | None:
    """The first stage's ``check.txt``: its JSON object, then what its last line says."""

    if path is None:
        return None
    text = path.read_text()
    end = text.rindex("}") + 1
    return json.loads(text[:end]) | {"said": text[end:].strip() or None}


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


def _count(values: Sequence[Any]) -> dict:
    return {"accepted": sum(1 for v in values if v is True), "of": len(values)}


def ties(summary: Mapping, stage1: Mapping | None, check: Mapping | None, published: Any) -> list[dict]:
    """What holds the runs' files to each other, each with whether it does (None: a file it reads is not given).
    ``published`` is the serve's ``commit.json`` for the first stage's class: the object, False where the serve
    published none equal to it, None where no serve is given."""

    first = summary.get("first_stage") or {}
    registration = None if stage1 is None else stage1.get("registration")
    sampled = None if stage1 is None else (stage1.get("draw") or {}).get("units")
    short = None if check is None else check.get("registration")

    def same(*values: Any) -> bool | None:
        return None if any(v is None for v in values) else all(v == values[0] for v in values)

    return [{"what": "the first stage's registration is the one the second stage read",
             "holds": same(first.get("registration"), registration)},
            {"what": "both stages name the same sampled spans", "holds": same(first.get("units"), sampled)},
            {"what": "the first stage's check is of that registration",
             "holds": None if None in (short, registration) else registration.startswith(short)},
            {"what": "the first stage's class is the serve's published one, field for field",
             "holds": None if stage1 is None or published is None else published is not False}]


def shown(summary: Mapping, stage1: Mapping | None, check: Mapping | None, published: Any, machine: Path | None) -> dict:
    """What the audit showed, as its files hold it: counts and codes, no verdict of the reader's own."""

    audit = summary.get("audit") or {}
    inner, timing = audit.get("audit") or {}, audit.get("timing") or {}
    sessions = list(timing.values())
    by_kind: dict[str, list] = {}
    for name, v in (inner.get("verdicts") or {}).items():
        by_kind.setdefault(name.split("/", 1)[0], []).append(v if isinstance(v, bool) else (v or {}).get("accepted"))
    refused = [v if isinstance(v, bool) else bool((v or {}).get("refused")) for v in (audit.get("negatives") or {}).values()]
    round_two = summary.get("round_two")
    checks = {"checks_before_draw": _count(list((audit.get("checks_before_draw") or {}).values())),
              "pins": _count(list((summary.get("pins") or {}).values())), "negatives": _count(refused)}
    checks |= {k: _count([inner[k]]) for k in ("registered_reads", "complete") if k in inner}
    if isinstance(round_two, Mapping):
        checks["round_two"] = _count([round_two.get("ok")]) | {"codes": list(round_two.get("codes") or [])}
    outcome = summary.get("outcome")
    if isinstance(outcome, Mapping):
        codes = [c for k in CODES for c in (outcome.get(k) or []) if c not in CODES]
        outcome = {"accepted": outcome.get("accepted") is True, "codes": list(dict.fromkeys(codes)),
                   "refused": outcome.get("refused")}

    def total(read: Any) -> float | None:
        values = [read(s) for s in sessions]
        return None if not values or any(v is None for v in values) else float(sum(values))

    layout = (stage1 or {}).get("commit") or (published if isinstance(published, Mapping) else None)
    interior = summary.get("interior") or {}
    draw = (stage1 or {}).get("draw") or {}
    server = audit.get("proof_server") or {}
    cards = {str(p["device"]): {"sessions": p.get("sessions"), "seconds": p.get("seconds")}
             for p in (server.get("busy") or {}).values() if isinstance(p, Mapping) and p.get("device") is not None}
    for s in sessions:
        if s.get("device") is not None:
            cards.setdefault(str(s["device"]), {"sessions": None, "seconds": None})
    cores = None
    if machine is not None and (words := machine.read_text().split()) and words[0].isdigit():
        cores = int(words[0])
    return {
        "ties": ties(summary, stage1, check, published),
        "first_stage": None if stage1 is None else {
            "registration": stage1.get("registration"), "rule": (stage1.get("rule") or {}).get("replay"),
            "sampled": draw.get("units"), "of": draw.get("population"),
            "check": None if check is None else {k: check.get(k) for k in ("error", "session_error", "coins_reason", "said")}},
        "layout": None if layout is None else {k: layout[k] for k in LAYOUT if k in layout}
        | {"values": sorted(layout.get("values") or {})},
        "interior": {"registered": list(interior.get("registered") or []), "links": len(interior.get("links") or []),
                     "returns": len(interior.get("returns") or []), "population": interior.get("population"),
                     "unlinked": summary.get("unlinked")},
        "verdicts": {kind: _count(values) for kind, values in sorted(by_kind.items())},
        "checks": checks, "outcome": outcome, "held": summary.get("held"), "refused": summary.get("refused"),
        "one_stage_record": {"accepted": summary.get("accepted"), "record": summary.get("record")},
        "sessions": len(sessions), "proof_blocks": len((summary.get("second_stage") or {}).get("draw") or []),
        "seconds": {"session_s": total(lambda s: s.get("session_s")), "busy_s": total(lambda s: s.get("busy_s")),
                    "prove_total_s": total(lambda s: (s.get("timing") or {}).get("prove_total_s")),
                    "lean_verify_s": total(lambda s: s.get("lean_verify_s")),
                    "verify_s": total(lambda s: s.get("verify_s"))},
        "sizes": sorted({(s.get("timing") or {}).get("k_log") for s in sessions} - {None}),
        "cards": cards, "cores": cores,
    }


def read(second: Path, *, first: Path | None = None, serve: Path | None = None, replay: Path | None = None,
         run: str | None = None, serving: Path | None = None, stock: Path | None = None,
         serving_cost: float | None = None, serving_cost_source: str | None = None) -> tuple[dict, list[dict]]:
    """The record of the served two-stage run whose second stage is ``second``, and each field it lacks: its name
    as `exfiltration.served_share` names it, the file read for it, and why the file doesn't hold it."""

    file = _find(second, SECOND, "a second stage's run")
    summary = json.loads(file.read_text())
    audit = summary.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"{file} holds no `audit`: its second stage stopped before it proved its sample")
    absent: list[dict] = []

    def lacks(field: str, why: str, where: Path | None = file) -> None:
        absent.append({"field": field, "file": None if where is None else str(where), "why": why})

    runs = {name: run_of(d if d is not None and Path(d).is_dir() else None)
            for name, d in (("serve", serve), ("first", first), ("replay", replay), ("second", second))}
    mine = runs["second"] or {}
    run = run or mine.get("run")
    if run is None:
        lacks("run", "the run's id is not in its files: give the run's directory, or its id (--run)", None)

    timing = audit.get("timing")
    prove = source = None
    if not isinstance(timing, Mapping) or not timing:
        lacks("prove", "`audit.timing` holds no session")
    else:
        sessions = [s for _, s in sorted(timing.items())]
        named = [s.get(SOURCE) for s in sessions]
        if unknown := sorted({str(n) for n in named if n is not None and n not in SOURCES}):
            raise ValueError(f"{file}: a session's `{SOURCE}` is {', '.join(unknown)}, not one of {', '.join(SOURCES)}")
        if None in named:
            lacks(SOURCE, f"`audit.timing` holds no `{SOURCE}` for a session: its seconds are read as `session_s`, and "
                  "nothing says whether they are the proof server's own")
        else:
            source = "session" if "session" in named else "busy"
        prove = [_entry(s.get(SOURCES[n or "session"]), s.get("gpus")) for s, n in zip(sessions, named)]
        if any(e["seconds"] is None for e in prove):
            lacks("prove.seconds", "`audit.timing` holds no `" + "` or no `".join(
                sorted({SOURCES[n or "session"] for n in named})) + "` for a session")
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

    own = _find(replay, REPLAY, "a replay's run")
    parts = {}
    for field, key in (("replay", "replay"), ("interior", "interior_commit")):
        held = summary.get(key)
        parts[field] = None
        if isinstance(held, Mapping):
            parts[field] = [_entry(held.get("seconds"), held.get("gpus"))]
            for k in ("seconds", "gpus"):
                if held.get(k) is None:
                    lacks(f"{field}.{k}", f"`{key}` holds no `{k}`")
        elif field == "replay" and own is not None:
            its = json.loads(own.read_text())
            parts[field] = [_entry(its.get("replay_wall_s"), its.get("gpus"))]
            for k, why in (("seconds", "the replay's own file holds no `replay_wall_s`"),
                           ("gpus", "the replay's own file holds its seconds (`replay_wall_s`) and not the GPUs its "
                                    "engine held (`gpus`)")):
                if parts[field][0][k] is None:
                    lacks(f"replay.{k}", why, own)
        else:
            lacks(field, f"`{key}` is absent: the two-stage audit reads no clock over it")

    stage1 = _json(_find(first, STAGE1, "a first stage's run"))
    check = None
    if first is not None and Path(first).is_dir():
        check = check_of(next((p for pattern in CHECK for p in sorted(Path(first).glob(pattern))), None))
    published = None
    if serve is not None:
        commits = ([Path(serve)] if Path(serve).is_file() else
                   [p for pattern in COMMITS for p in sorted(Path(serve).glob(pattern))])
        if not commits:
            raise ValueError(f"{serve} holds none of {', '.join(COMMITS)}: it is not a serve's run")
        wanted = (stage1 or {}).get("commit")
        published = next((c for c in map(_json, commits) if wanted is None or c == wanted), False)
    machine = next((p for p in (file.parent / MACHINE, file.parent.parent / MACHINE) if p.is_file()), None)

    record = {"run": run, "commit": mine.get("commit"), "runs": runs, "prove": prove, SOURCE: source, "staging": staging,
              **parts, "serving": _load(serving, "serving", absent), "stock": _load(stock, "stock", absent)}
    if record["stock"] is None and serving_cost is not None:
        record |= {"serving_cost": serving_cost, "serving_cost_source": serving_cost_source}
    record["audit"] = shown(summary, stage1, check, published, machine)
    return record, absent


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m verity_bench.served_record",
        description="A served two-stage run's record as `verity bench exfiltration --served-run` reads it, on stdout as "
                    "one JSON object; each field the run's files lack is None there and is named on stderr.")
    ap.add_argument("--second", type=Path, required=True, metavar="DIR",
                    help="the second stage's run directory (or its `out/`, its `second.json` or its `summary.json`)")
    ap.add_argument("--first", type=Path, metavar="DIR", help="the first stage's run directory (or its `stage1.json`)")
    ap.add_argument("--serve", type=Path, metavar="DIR", help="the serve's run directory (or the class's `commit.json`)")
    ap.add_argument("--replay", type=Path, metavar="DIR",
                    help="the run directory of the replay of the sampled spans (or its own `replay.json`)")
    ap.add_argument("--run", help="the served run's id, where the second stage's directory does not name it")
    ap.add_argument("--serving", type=Path, metavar="META", help="the served load's `meta.json` (`verity bench serve_tax`)")
    ap.add_argument("--stock", type=Path, metavar="META", help="stock vLLM's `meta.json` on the same load")
    ap.add_argument("--serving-cost", type=float, metavar="C",
                    help="without --stock: the serving cost of the served arm over stock, a fraction")
    ap.add_argument("--serving-cost-source", metavar="S", help="the run and load the serving cost is of")
    a = ap.parse_args(argv)
    try:
        record, absent = read(a.second, first=a.first, serve=a.serve, replay=a.replay, run=a.run, serving=a.serving,
                              stock=a.stock, serving_cost=a.serving_cost, serving_cost_source=a.serving_cost_source)
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
