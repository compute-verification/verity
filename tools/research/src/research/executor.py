"""The task executor's tasks/v1 side: each task's verdict read by its key.

`tasks/v1` (`verity tasks list --json`, `tools/check/tasks.py`) is the graph schema.  A graph's keys hash the lister's
environment, so the executor lists a check on a worker, puts it in the store (`tasks-graph/v1`, `TASKS_GRAPH_KIND`) and labels
`commit:<sha> tasks_graph art:<id>` (`vocab.TASKS_GRAPH`).

A task's verdict comes from the `verdict` labels on `task:<key>` (`task_target`), written by the executor from its run of `verity
tasks run ID`: pass or fail by that run's result, or `ERROR` when no run gave one, which is pending (not run); a step that
skipped gets no label and stays pending.  A fail stands until an error label is written after it (a false fail relabelled), so
a rerun that passes never turns a fail green; past that, the newest of pass and error is the key's (`fold`).  A task with no key
of its own (tasks/v1's `unkeyed`) gets one of its check alone (`keyed`), so no verdict of it is reused across checks."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable


PASS, FAIL, PENDING = "pass", "fail", "pending"
#: the label of a task whose runs all ended without a result (the machine lost, the run refused or cancelled): no verdict of
#: the tree, so `fold` reads it as pending
ERROR = "error"
#: the store kind of a tasks/v1 graph the executor listed
TASKS_GRAPH_KIND = "tasks-graph/v1"


def task_target(key: str) -> str:
    """The store target of a task's key: `task:<128 hex>`, its key without a `sha512:`."""
    return f"task:{key.removeprefix('sha512:')}"


def check_id(graph: dict) -> str:
    """The check a graph is of: its kind and commit (`after-landing:SHA`)."""
    return f"{graph.get('kind') or 'after-landing'}:{graph['commit']}"


def _digest(doc) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def keyed(graph: dict) -> dict:
    """``graph`` with each task's `key` the one its verdict is labelled under: its own, or for an unkeyed task, SHA-512 over its
    check and id (its `unkeyed` reason kept), which no other check shares."""
    check = check_id(graph)
    return {**graph, "tasks": [t if t.get("key") else {**t, "key": _digest({"check": check, "id": t["id"]})}
                               for t in graph["tasks"]]}


def fold(values: Iterable[str]) -> str:
    """A key's verdict from its labels' values, oldest first: fail while a fail has no `ERROR` after it, else pass when the
    newest is pass, else pending (no label, or `ERROR` newest)."""
    vs = list(values)
    if FAIL in vs and ERROR not in vs[len(vs) - vs[::-1].index(FAIL):]:
        return FAIL
    return PASS if vs and vs[-1] == PASS else PENDING

