"""The observed record GM-01 reads: the capture log per engine step, and every request reference renamed to the Program's `r<k>`."""
from __future__ import annotations

import collections
import gzip
import json
import re

_TAG_RE = re.compile(r'^\{"\$e":\s*"(\w+)"')


def read_record(log_path: str) -> dict:
    """RunHeader + per engine step: rows, input tokens, logits rows, sampled tokens; StepLinks."""
    header, steps, samples, links = None, {}, {}, []
    sampler_batch: dict[int, int] = {}   # engine step -> rows of the sampler's logits tensor (SampleEnd.logits_view shape[0])

    def step_start(d: dict) -> None:
        s = int(d["step"])
        if s in steps:
            raise ValueError(f"two StepStart records for step {s}")
        steps[s] = {"rows": [(r, int(p)) for r, p in d["rows"]], "input_token_ids": [int(x) for x in d.get("input_token_ids") or []],
                    "logits_rows": [(r, int(p)) for r, p in d["logits_rows"]]}

    def sample_end(d: dict) -> None:
        samples[int(d["step"])] = [int(x) for x in d["sampled_token_ids"]]
        lv = (d.get("logits_view") or {}).get("$v") if isinstance(d.get("logits_view"), dict) else None
        if isinstance(lv, list) and len(lv) > 2 and isinstance(lv[2], list) and lv[2]:
            sampler_batch[int(d["step"])] = int(lv[2][0])

    def step_link(d: dict) -> None:
        links.append({k: d.get(k) for k in ("step", "row", "request_id", "src_step", "relation", "value")})

    def run_header(d: dict) -> None:
        nonlocal header
        header = d

    read = {"StepStart": step_start, "SampleEnd": sample_end, "StepLink": step_link, "RunHeader": run_header}
    opener = gzip.open if log_path.endswith(".gz") else open
    with opener(log_path, "rt") as f:
        for line in f:
            m = _TAG_RE.match(line)
            on = read.get(m.group(1)) if m else None
            if on is not None:
                on(json.loads(line))
    if header is None:
        raise ValueError(f"{log_path}: no RunHeader")
    return {"header": header, "steps": steps, "samples": samples, "links": links, "sampler_batch": sampler_batch}


def request_aliases(gp: dict) -> dict[str, str]:
    """record request id (the manifest UUID vLLM served under, or an already-canonical id) -> the Program's `r<k>`."""
    alias: dict[str, str] = {}
    for r in gp["requests"]:
        rid = str(r["request_id"])
        alias[rid] = rid
        w = r.get("workload_request_id")
        if w is not None:
            alias[str(w)] = rid
    return alias


def canonicalize_record(rec: dict, tokens: dict | None, oracle_rows: list[dict] | None, alias: dict[str, str]) -> dict:
    """Rename every request reference of the record (RunHeader.workload, StepStart rows / logits rows, StepLinks, tokens.json, fold
    instance meta) to the canonical `r<k>` through `alias`, IN PLACE. Ids the table does not know are left as they are and returned
    (G1/G2 then report them as undeclared). The original id is kept as `workload_request_id` on header/tokens entries."""
    unknown: collections.Counter = collections.Counter()

    def m(x):
        if x is None:
            return None
        y = alias.get(str(x))
        if y is None:
            unknown[str(x)] += 1
            return str(x)
        return y

    for st in rec["steps"].values():
        st["rows"] = [(m(r), p) for r, p in st["rows"]]
        st["logits_rows"] = [(m(r), p) for r, p in st["logits_rows"]]
    for lk in rec["links"]:
        lk["request_id"] = m(lk.get("request_id"))
    wl = rec["header"].get("workload") or {}
    for r in wl.get("requests") or []:
        r["workload_request_id"] = r.get("request_id")
        r["request_id"] = m(r.get("request_id"))
    for c in wl.get("children") or []:
        c["request_id"], c["parent"] = m(c.get("request_id")), m(c.get("parent"))
    for r in (tokens or {}).get("requests", []) if tokens else []:
        r["workload_request_id"] = r.get("request_id")
        r["request_id"] = m(r.get("request_id"))
    for o in oracle_rows or []:
        if o.get("request") is not None:
            o["request"] = m(o["request"])
    return {"renamed": len(alias), "unknown_ids": dict(unknown)}
