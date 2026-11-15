"""`verity graph`: the check's task graph of this checkout, one task run, and the serial reference executor.

    verity graph [--json [--packed]] [--family F ...] [--root DIR]
                                  the graph (graph/v2: --json prints its canonical bytes, --packed those of it packed,
                                  `graph.pack`); with --family, those families' tasks and every task they read
    verity graph unpack [FILE]    the canonical bytes of the graph/v2 a packed listing in FILE (default stdin) packs
    verity graph run ID --work DIR [--graph FILE]
                                  run task ID here in work directory DIR (its reads already there) and print its result/v1;
                                  exit 0 pass, 1 fail, 3 error; with FILE, a listing of this checkout (`verity graph --json`,
                                  packed or not) or ID's part of it (`graph.part`), no lister runs
    verity graph check --local [--family F ...] [--work DIR] [--retries N] [--events FILE]
                                  every task (or those families' and what they read) serially, in order, uncached, by the
                                  reference executor (`local.py`), its events/v1 lines (purpose adhoc) appended to FILE
                                  (default DIR/events.jsonl) and each task's last result/v1 in DIR/results.json; exit 0
                                  when every one passed
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import ROOT, contract, graph, local, runners

EXIT = {"pass": 0, "fail": 1, "error": 3}


def _families(fs: list[str] | None) -> set[str] | None:
    return set(fs) if fs else None


def _list(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity graph", description="The check's task graph of this checkout (graph/v2).")
    ap.add_argument("--json", action="store_true", help="print the graph/v2 document's canonical bytes")
    ap.add_argument("--packed", action="store_true", help="with --json, the document packed (graph.pack), which `verity graph "
                    "unpack` makes graph/v2 again")
    ap.add_argument("--family", action="append", metavar="F", help="only family F's tasks and every task they read")
    ap.add_argument("--root", type=Path, default=ROOT, help="list the checkout at DIR (default: this one)")
    a = ap.parse_args(args)
    if a.packed and not a.json:
        ap.error("--packed packs the --json document")
    doc = graph.listing(a.root.resolve(), _families(a.family))
    if a.json:
        sys.stdout.buffer.write(contract.canonical(graph.pack(doc) if a.packed else doc) + b"\n")
        return 0
    for t in doc["tasks"]:
        print(f"{t['id']:72} {t['env']:5} {t['key'][7:23] if t['key'] else 'reads'}  {t['about']}")
    print(f"{len(doc['tasks'])} tasks, {len(doc['files'])} files, commit {doc['commit'] or 'none (the tree is dirty)'}",
          file=sys.stderr)
    return 0


def _unpack(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity graph unpack", description="Print the graph/v2 a packed listing packs.")
    ap.add_argument("file", nargs="?", type=Path, metavar="FILE", help="the packed listing (default: stdin)")
    a = ap.parse_args(args)
    try:
        doc = json.loads(a.file.read_bytes() if a.file else sys.stdin.buffer.read())
    except (OSError, ValueError) as e:
        raise graph.GraphError(f"{a.file or 'stdin'}: {e}") from e
    sys.stdout.buffer.write(contract.canonical(graph.unpack(doc)) + b"\n")
    return 0


def find(task_id: str, listed: Path | None = None) -> tuple[dict, dict]:
    """(task ``task_id``, the part of this checkout's graph that running it needs: `graph.task_graph`, or from the graph/v2
    file ``listed``, `graph.given`)."""
    doc = graph.task_graph(ROOT, task_id) if listed is None else graph.given(ROOT, listed, task_id)
    return next(t for t in doc["tasks"] if t["id"] == task_id), doc


def _run(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity graph run", description="Run one task here and print its result/v1.")
    ap.add_argument("id", metavar="ID")
    ap.add_argument("--work", type=Path, required=True, help="its work directory, holding the artifacts it reads")
    ap.add_argument("--graph", type=Path, metavar="FILE", help="a listing of this checkout (`verity graph --json`) or ID's part "
                    "of it (`graph.part`), so that no lister runs")
    a = ap.parse_args(args)
    task, doc = find(a.id, a.graph)
    res = runners.execute(task, doc, a.work.resolve())
    sys.stdout.buffer.write(contract.canonical(res) + b"\n")
    return EXIT[res["status"]]


def _check(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity graph check", description="Run the graph serially with the reference executor.")
    ap.add_argument("--local", action="store_true", required=True, help="here, serially and uncached (the only executor here)")
    ap.add_argument("--family", action="append", metavar="F", help="only family F's tasks and every task they read")
    ap.add_argument("--work", type=Path, help="the directory for work directories and artifacts (default: a new temporary one)")
    ap.add_argument("--retries", type=int, default=1, help="how often an error is retried (default %(default)s); a fail never is")
    ap.add_argument("--events", type=Path, metavar="FILE", help="append events/v1 lines here (default: DIR/events.jsonl)")
    a = ap.parse_args(args)
    work = (a.work or Path(tempfile.mkdtemp(prefix="verity-graph-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    events = a.events or work / "events.jsonl"
    with open(events, "a") as ev:
        out = local.check(ROOT, _families(a.family), work, retries=a.retries, events=ev)
    (work / "results.json").write_bytes(contract.canonical({**out["results"], **out["errors"]}) + b"\n")
    counts = {s: sum(r["status"] == s for r in out["results"].values()) for s in ("pass", "fail")}
    print(json.dumps({"ok": out["ok"], **counts, "error": len(out["errors"]), "cancelled": len(out["cancelled"]),
                      "work": str(work), "events": str(events)}), flush=True)
    return 0 if out["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    try:
        if args[:1] == ["run"]:
            return _run(args[1:])
        if args[:1] == ["check"]:
            return _check(args[1:])
        if args[:1] == ["unpack"]:
            return _unpack(args[1:])
        return _list(args)
    except graph.GraphError as e:
        print(f"verity graph: {e}", file=sys.stderr)
        return 2
