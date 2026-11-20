"""The benchmark result schema and identity-strict comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from verity_bench import kit

SCHEMA = "verity-bench/vllm-overhead/v1"
#: each metric's unit, as a point's result records it (`record`)
UNITS = {
    "time_per_output_token_s": "s",
    "time_to_first_token_s": "s",
    "tokens_per_s": "tokens/s",
    "peak_gpu_memory_bytes": "bytes",
    "decode_time_per_token_s": "s",
    "inter_token_latency_p50_s": "s",
    "inter_token_latency_p99_s": "s",
}


def validate(doc: Mapping[str, Any]) -> None:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"benchmark result schema must be {SCHEMA!r}")
    for key in ("identity", "baseline", "sweep", "gates", "execution", "points"):
        if key not in doc:
            raise ValueError(f"benchmark result has no {key!r}")
    arm_ids = [a.get("id") for a in doc["identity"].get("arms", [])]
    if len(set(arm_ids)) != len(arm_ids) or doc["baseline"] not in arm_ids:
        raise ValueError("benchmark result arm ids must be unique and include its baseline")
    if any(not g.get("passed", False) for g in doc["gates"]) and doc["points"]:
        raise ValueError("a failed benchmark gate must have no timed points")
    sweep = [p["id"] for p in doc["sweep"]]
    have = {(g.get("kind"), g.get("arm"), g.get("point")) for g in doc["gates"]}
    for arm in doc["identity"]["arms"]:
        need = [("arm", arm["id"], None)]
        if arm.get("control"):
            need += [("kv-pool", arm["id"], None), *(("tokens", arm["id"], p) for p in sweep)]
            need += [("timed-tokens", arm["id"], p) for p in sweep] if doc["points"] else []
        if missing := [n for n in need if n not in have]:
            raise ValueError(f"benchmark result lacks gates (kind, arm, point) {missing}")
    if doc["points"]:
        _check_points(doc, sweep)


def _check_points(doc: Mapping[str, Any], sweep: list[str]) -> None:
    """The points must be exactly what the runner's aggregation makes of their own reps, every rep of every arm."""

    from types import SimpleNamespace

    from verity_bench.vllm.presets import SweepPoint
    from verity_bench.vllm.runner import _aggregate

    if [p["point"]["id"] for p in doc["points"]] != sweep:
        raise ValueError("benchmark result points must be its sweep's points, in order")
    arms = tuple(SimpleNamespace(id=a["id"]) for a in doc["identity"]["arms"])
    reps = list(range(doc["execution"]["reps"]))
    samples = {}
    for row in doc["points"]:
        for arm in arms:
            got = row["arms"][arm.id]["reps"]
            if sorted(r["rep"] for r in got) != reps:
                raise ValueError(f"{arm.id} at {row['point']['id']} has reps {[r['rep'] for r in got]}, not {reps}")
            samples[(arm.id, row["point"]["id"])] = got
    points = tuple(SweepPoint(*(p["point"][k] for k in ("batch", "prompt_tokens", "traffic", "output_tokens")))
                   for p in doc["points"])
    again = json.dumps(_aggregate(arms, points, doc["baseline"], samples), sort_keys=True)
    if again != json.dumps(doc["points"], sort_keys=True):
        raise ValueError("benchmark result points differ from the aggregate of their own reps")


def write(doc: Mapping[str, Any], path: str | Path) -> None:
    validate(doc)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def measurements(doc: Mapping[str, Any], evidence: str) -> list[dict[str, Any]]:
    """The kit writer's arguments for each number of `doc`'s timed points, each measured from `evidence`: an arm's mean
    of a metric with its 95% interval, named ``ARM.METRIC``, and for each arm but the baseline its paired ratio over the
    baseline's reps, ``ARM.METRIC.ratio``, with no value where a baseline rep measured zero; each the sweep's point it
    was timed at as its `member`. A failed gate leaves no timed point, so it leaves no measurement."""

    out: list[dict[str, Any]] = []
    for point in doc["points"]:
        at = {"point": point["point"]["id"]}
        for arm, held in point["arms"].items():
            for metric, m in held["metrics"].items():
                out.append({"name": f"{arm}.{metric}", "value": m["mean"], "unit": UNITS[metric],
                            "interval": m.get("ci95"), "why": None if m["mean"] is not None else "no rep measured it",
                            "inputs": [evidence], "member": at["point"], "extra": {**at, "n": m["n"]}})
                if arm == doc["baseline"]:
                    continue
                r = m["ratio_to_stock"]
                out.append({"name": f"{arm}.{metric}.ratio", "value": r["mean"], "unit": "ratio", "interval": r.get("ci95"),
                            "why": None if r["mean"] is not None else "a baseline rep measured zero",
                            "inputs": [evidence], "member": at["point"],
                            "extra": {**at, "n": r["n"], "baseline": doc["baseline"]}})
    return out


def record(doc: Mapping[str, Any], writer: kit.Writer, evidence: str) -> None:
    """Appends `doc`'s `measurements` to the run's result through the kit's `writer`."""

    for m in measurements(doc, evidence):
        writer.measurement(**m, mark="measured")


