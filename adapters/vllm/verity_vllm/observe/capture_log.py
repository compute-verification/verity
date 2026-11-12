"""The capture log as Match reads it: the RunHeader, each engine step's StepStart rows, and the request each prompt parameter is."""

from __future__ import annotations

import gzip
import json
import re

_TAG_RE = re.compile(r'^\{"\$e":\s*"(\w+)"')


def read_steps(log_path: str) -> tuple[dict, dict[int, dict]]:
    """RunHeader (as dict) and per engine step {rows: [(req, pos)], logits_rows: [(req, pos)], logits_indices: [int]}."""
    header = None
    steps: dict[int, dict] = {}
    opener = gzip.open if log_path.endswith(".gz") else open
    with opener(log_path, "rt") as f:
        for line in f:
            m = _TAG_RE.match(line)  # the writer puts "$e" first (`events.to_json`); decode only the two record kinds we need
            tag = m.group(1) if m else None
            if tag == "StepStart":
                d = json.loads(line)
                s = int(d["step"])
                if s in steps:
                    raise ValueError(f"log has two StepStart records for step {s}")
                steps[s] = {"rows": [(r, int(p)) for r, p in d["rows"]], "logits_rows": [(r, int(p)) for r, p in d["logits_rows"]],
                            "logits_indices": [int(x) for x in d.get("logits_indices") or []]}
            elif tag == "RunHeader":
                if header is not None:
                    raise ValueError("log has two RunHeader records")
                header = json.loads(line)
    if header is None:
        raise ValueError(f"{log_path}: no RunHeader")
    return header, steps


def request_params(header: dict) -> tuple[list[dict], dict[str, int], dict[str, str]]:
    """(requests in header order = prompt parameter order, request_id -> prompt param index, output id -> parent id).
    Mirrors `fold._header`: a forked child (`workload.children`) reads its PARENT's prompt parameter."""
    reqs = list((header.get("workload") or {}).get("requests") or [])
    if not reqs:
        raise ValueError("RunHeader.workload has no requests")
    idx = {r["request_id"]: i for i, r in enumerate(reqs)}
    parent = {r["request_id"]: r["request_id"] for r in reqs}
    for c in (header.get("workload") or {}).get("children") or []:
        if c.get("parent") not in idx:
            raise ValueError(f"child {c.get('request_id')!r} names unknown parent {c.get('parent')!r}")
        idx[c["request_id"]] = idx[c["parent"]]
        parent[c["request_id"]] = c["parent"]
    return reqs, idx, parent
