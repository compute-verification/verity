"""circuit-binding: one catalog binding's targets (the task's var ``binding``), `circuit-check --all --catalog-units --binding
NAME --slice`, from that binding's slice and the part every binding's shares, which the circuit-catalog task cut from the
records it built (vars ``slice`` and ``shared``), as a shard of check's circuit-check runs from the records step's: no
binding is built, no other binding's record read and no other target keyed.  Each target is reused while everything its own
check read is unchanged (`circuit_check.cache`), so the cache is the machine's, check's own (`cache`), not the work
directory's: a rerun checks only the targets whose reads changed.  circuit-check keys its records and results by the
``VERITY_*`` and ``CIRCUIT_CHECK_*`` environment, which this passes on as it is.

A pass is circuit-check's exit 0; a fail its exit 1 (a target's failure), its refusal of a name no catalog binding has
(`NO_BINDING`) or the catalog task's marker that it made no slice of that name (`absent`); anything else is an error: another
refusal (a slice not of this tree), a kill, no report.  `detail`: the binding's targets, how many passed, failed and were
reused, and the failures' names, as the report's summary names them (``failure_names``, `circuit_check.checks.summarize`'s).
Its one output is every failure of its Definition targets, each by its check, code and Definition (`FIELDS`).  `checker`,
`pack` and `unpack` are the circuit-catalog and circuit-coverage runners' too."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call

#: `circuit_check.cli.NO_BINDING`, word for word (`tests/test_circuit_binding.py`)
NO_BINDING = "the catalog has no binding of that name"
REPORT = "report.json"
#: the tarball's directory of records, as check's records step tars it (`check.py`'s `cc_records`)
TOP = "cache"
#: the failures `detail` names, at most
NAMED = 100
#: what the ``failures`` output keeps of a failure
FIELDS = ("check", "code", "definition")


def cache() -> Path:
    """circuit-check's cache on this machine: check's own where check keeps it unless ``CHECK_CACHE`` moves it, as on every node
    (`tools/verity/check/common.py`'s ``cache_dir() / "circuit-check"``).  ``CHECK_CACHE`` is check's registry's to read
    (`tests/lint/test_repo_env_reads.py`), so a machine that sets it keeps these tasks' records apart from check's."""
    return Path.home() / ".cache" / "verity-check" / "circuit-check"


def checker(task: dict, work: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, dict | None]:
    """`circuit-check ARGS` from the root on the task's cores and memory in ``env`` (this process's when None), its report in
    the work directory: (its exit, the report, None when it wrote none)."""
    out = work / REPORT
    out.unlink(missing_ok=True)
    r = task["resources"]
    rc = call([sys.executable, "-m", "circuit_check", *args, "-q", "--jobs", str(max(1, int(r["cores"]))),
               "--mem-gb", str(max(1, int(r["mem_gb"]))), "--out", str(out)], work, ROOT, env)
    try:
        return rc, json.loads(out.read_text())
    except (OSError, ValueError):
        return rc, None


def pack(records: Path, to: Path) -> int:
    """The records under ``records`` tarred at ``to`` under `TOP`, as check's records step tars its cache, the bytes a function
    of the records' paths and bytes alone (in order, no times, owners or modes); how many."""
    files = sorted(p for p in records.rglob("*.json") if p.is_file())
    to.parent.mkdir(parents=True, exist_ok=True)
    with open(to, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as t:
        for p in files:
            info = tarfile.TarInfo(f"{TOP}/{p.relative_to(records).as_posix()}")
            info.size, info.mode = p.stat().st_size, 0o644
            with open(p, "rb") as f:
                t.addfile(info, f)
    return len(files)


def unpack(tar: Path, into: Path) -> int:
    """The records of a `pack` tarball into the cache ``into``, each by rename, as check unpacks the records step's
    (`check.py`'s `cc_records_in`): a circuit-check reading the cache meanwhile sees each whole or not at all.  They are
    unpacked in a directory inside the cache as its path resolves, so no rename crosses a filesystem when that path is a link
    or mount onto another disk; circuit-check lists records only in their key's directory (`circuit_check.cache._dir`), never
    there.  A record already there is left as it is: its file is named by its bytes' digest (`circuit_check.cache.store`).
    How many."""
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    d = Path(tempfile.mkdtemp(prefix=".circuit-records-", dir=root))
    try:
        with tarfile.open(tar) as t:
            t.extractall(d, filter="data")
        n = 0
        for p in sorted((d / TOP).rglob("*.json")):
            if not (to := root / p.relative_to(d / TOP)).exists():
                to.parent.mkdir(parents=True, exist_ok=True)
                os.replace(p, to)
            n += 1
        return n
    finally:
        shutil.rmtree(d, ignore_errors=True)


def absent(name: str) -> dict:
    """What the circuit-catalog task writes in the place of binding ``name``'s slice when circuit-check made none."""
    return {"binding": name, "absent": NO_BINDING}


#: the line circuit-check's ``--binding`` starts a refusal with (exit 2), which says why
REFUSED = "circuit-check: refused: "


def run(task: dict, work: Path) -> dict:
    name, c = task["vars"]["binding"], cache()
    piece, shared = work / task["vars"]["slice"], work / task["vars"]["shared"]
    if piece.stat().st_size < 4096 and json.loads(piece.read_text()) == absent(name):
        return {"status": "fail", "detail": {"error": f"circuit-check's catalog has no binding {name}"}}
    rc, rep = checker(task, work, "--all", "--catalog-units", "--cache", str(c), "--binding", name,
                      "--slice", str(piece), str(shared))
    said = (work / LOG).read_text(errors="replace")
    if rc == 2 and NO_BINDING in said:
        return {"status": "fail", "detail": {"error": f"circuit-check's catalog has no binding {name}"}}
    if rc not in (0, 1) or rep is None:
        why = next((ln.removeprefix(REFUSED) for ln in said.splitlines() if ln.startswith(REFUSED)), None) if rc == 2 else None
        return {"status": "error", "detail": {"error": f"circuit-check --binding exited {rc}"
                                              + ("" if rep is not None else " and wrote no report")
                                              + (f": {why[:500]}" if why else "")}}
    facts = sorted({tuple(f[k] for k in FIELDS) for t in rep["targets"] if t["kind"] == "definition"
                    for f in t.get("failures", [])})
    out = work / task["outputs"][0]["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"binding": name, "failures": [dict(zip(FIELDS, f)) for f in facts]}, sort_keys=True) + "\n")
    named = set(rep["summary"]["failure_names"])
    failed = sorted({t["target"] for t in rep["targets"] if any(f.get("name") in named for f in t.get("failures", []))})
    detail = {"targets": len(rep["targets"]), "passed": len(rep["targets"]) - len(failed), "failed": len(failed),
              "reused": (rep.get("cache") or {}).get("hits", 0)}
    if named:
        detail["failures"] = sorted(named)[:NAMED]
    return {"status": "fail" if rc else "pass", "detail": detail}
