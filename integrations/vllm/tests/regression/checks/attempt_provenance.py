"""(T0) Attempt provenance: a re-run of a reference row, published as `research` Attempts (Phase 5a), cites the SAME row configuration
the fixture row has -- the key params of `vllm.build` / `vllm.match` / `vllm.commit` (integrations/vllm/verity_vllm/research_tools.py):
model id + revision, dtype, hw family, tp, batch, input / output lengths, arrival mix, sampling, execution mode, the verification query
id -- and the stages chain: the Match Attempt names the Build artifact it consumed, the Commit names both, and those are the artifacts
the earlier Attempts published.

Expected = lifted from the fixture (`fixtures.toml` row key + `tp` / `batch`, the in-repo workload manifest's model / tokenizer_revision,
the query id default).  Actual = read from the candidate Attempts' manifests (`params` restricted to `key_params`).  Applies only in
candidate mode with attempt ids (`VERITY_REGRESSION_CANDIDATE=r2026…,…`); a directory candidate has no Attempts to cite.
"""
from __future__ import annotations

import json

from .. import resolver

NAME = "attempt_provenance"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("workloads/<row>.json (in-repo)", "the candidate Attempt manifests (research data show <run-id> --json)")

DEFAULT_QUERY = "Q_module_body_v1"
ROW_FIELDS = ("model_tag", "dtype", "hw", "tp", "batch", "input", "output", "arrivals", "sampling", "exec")
CONFIG_KEYS = ("row", "model", "revision", "query", *ROW_FIELDS)
STAGE_OUTPUT = {"vllm.build": "build", "vllm.match": "match", "vllm.commit": "verdict"}


def row_config(row: str) -> dict:
    """The row id's fields (the same grammar research_tools.row_config parses; duplicated here so the harness stays importable without it)."""
    parts = row.split("__")
    out: dict = {k: None for k in ROW_FIELDS}
    if len(parts) != len(ROW_FIELDS):
        out["model_tag"] = parts[0] if parts else None
        return out
    for k, v in zip(ROW_FIELDS, parts):
        pfx = {"tp": "tp", "batch": "b", "input": "i", "output": "o"}.get(k)
        out[k] = int(v[len(pfx):]) if pfx and v.startswith(pfx) and v[len(pfx):].isdigit() else v
    return out


def applies(row: resolver.Row) -> str | None:
    if not resolver.candidate_attempt_ids():
        return "no candidate Attempt ids (VERITY_REGRESSION_CANDIDATE is unset or a directory)"
    if not row.candidate_attempts():
        return "no candidate Attempt names this row"
    return None


def expected(row: resolver.Row) -> dict:
    wl = json.load(open(row.workload()))
    cfg = row_config(row.key)
    out = {"row": row.key, "model": wl.get("model"), "revision": wl.get("tokenizer_revision"), "query": DEFAULT_QUERY, **cfg}
    if row.spec.get("tp") is not None:
        out["tp"] = int(row.spec["tp"])
    if row.spec.get("batch") is not None:
        out["batch"] = int(row.spec["batch"])
    return {"config": out, "chain_consistent": True}


def actual(row: resolver.Row) -> dict:
    docs = row.candidate_attempts()
    stages: dict = {}
    configs = []
    problems: list[str] = []
    for d in docs:
        tool = (d.get("tool") or {}).get("name")
        params = d.get("params") or {}
        keys = d.get("key_params") or []
        key = {k: params.get(k) for k in keys if k in params}
        cfg = {k: key.get(k) for k in CONFIG_KEYS}
        configs.append(cfg)
        stage = tool.split(".", 1)[1] if isinstance(tool, str) and tool.startswith("vllm.") else None
        stages[stage or d["id"]] = {"attempt": d["id"], "tool": tool, "inputs": d.get("inputs") or {},
                                   "artifact": (d.get("outputs") or {}).get(STAGE_OUTPUT.get(tool, ""))}
    # one configuration across the stages (the first attempt's; the others must agree)
    cfg0 = configs[0] if configs else {k: None for k in CONFIG_KEYS}
    for i, c in enumerate(configs[1:], 1):
        for k in CONFIG_KEYS:
            if c.get(k) != cfg0.get(k):
                problems.append(f"attempt {docs[i]['id']} {k}={c.get(k)!r} != {docs[0]['id']} {cfg0.get(k)!r}")
    # the chain: Match cites the Build's artifact; Commit cites both
    b, m, c = stages.get("build"), stages.get("match"), stages.get("commit")
    if m and b and m["inputs"].get("build") != b["artifact"]:
        problems.append(f"match cites build={m['inputs'].get('build')} but the Build attempt published {b['artifact']}")
    if c and b and c["inputs"].get("build") != b["artifact"]:
        problems.append(f"commit cites build={c['inputs'].get('build')} but the Build attempt published {b['artifact']}")
    if c and m and c["inputs"].get("match") != m["artifact"]:
        problems.append(f"commit cites match={c['inputs'].get('match')} but the Match attempt published {m['artifact']}")
    if m and not m["inputs"].get("build"):
        problems.append("the Match attempt cites no Build input")
    if c and (not c["inputs"].get("build") or not c["inputs"].get("match")):
        problems.append("the Commit attempt does not cite both a Build and a Match input")
    return {"config": cfg0, "chain_consistent": not problems, "chain_problems": problems, "stages": stages, "source": "candidate-attempts"}
