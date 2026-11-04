"""circuit-known: each `circuit_check.known` entry still matches a failure of the catalog's Definitions (`known.stale`, as
`circuit-check --all` and `--combine` hold it), over the failures every circuit-binding task wrote (its output ``failures``, all
in the directory at var ``failures``): no target is checked again.  A binding task's report is its targets' alone, unjoined, so
none of them sees an entry no failure matches any more, which would keep tolerating its target's next failure; here it is a
fail, each such entry named in `detail`'s ``stale``.  `detail` counts the bindings and failures read too.

A reader of a task that didn't pass is cancelled, so this runs once every binding passes.  The failures `by_calls.joined` adds
to a by-calls root (`JOINED`) are no binding task's, since each callee is a target of its own and fails its own task: an entry
for one is left to `circuit-check --all`."""
from __future__ import annotations

import json
from pathlib import Path

from circuit_check import known

#: the check of the failures only `by_calls.joined` makes (`by_calls.CHECK`, `tests/test_circuit_binding.py`)
JOINED = "by-calls/"


def run(task: dict, work: Path) -> dict:
    files = sorted((work / task["vars"]["failures"]).glob("*.json"))
    fs = [f for p in files for f in json.loads(p.read_text())["failures"]]
    stale = [k for k in known.stale(fs) if not k.startswith(JOINED)]
    return {"status": "fail" if stale else "pass", "detail": {"bindings": len(files), "failures": len(fs), "stale": stale}}
