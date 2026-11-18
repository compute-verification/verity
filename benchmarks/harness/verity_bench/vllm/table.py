"""Markdown ratio tables for the shared vLLM benchmark."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

METRIC_LABELS = {
    "time_per_output_token_s": "time / output token",
    "time_to_first_token_s": "time to first token",
    "tokens_per_s": "output tokens / second",
    "peak_gpu_memory_bytes": "peak GPU memory",
    "decode_time_per_token_s": "decode time / token",
    "inter_token_latency_p50_s": "per-token latency p50",
    "inter_token_latency_p99_s": "per-token latency p99",
}


def _ratio(metric: Mapping[str, Any]) -> str:
    r = metric.get("ratio_to_stock")
    if not r or r.get("mean") is None:
        return "—"
    mean = r["mean"]
    ci = r.get("ci95")
    return f"{mean:.3f}× (n={r['n']})" if ci is None else f"{mean:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}]"


def render(doc: Mapping[str, Any]) -> str:
    """Render headline time per output token first, then the rest of the sweep."""

    arms = [a["id"] for a in doc["identity"]["arms"]]
    points = sorted(doc["points"], key=lambda p: (not p["point"].get("headline", False), p["point"]["id"]))
    metrics = tuple(METRIC_LABELS)
    rows: list[tuple[dict, str]] = []
    for point in points:
        rows.extend((point, m) for m in metrics if all(m in point["arms"][arm]["metrics"] for arm in arms))
    rows.sort(key=lambda x: (
        not (x[0]["point"].get("headline", False) and x[1] == "time_per_output_token_s"),
        not x[0]["point"].get("headline", False),
        x[0]["point"]["id"],
        metrics.index(x[1]),
    ))
    ungated = [a["id"] for a in doc["identity"]["arms"] if not a.get("control") and a["id"] != doc["baseline"]]
    lines = [
        f"Baseline: `{doc['baseline']}`. Ratios are arm / baseline; lower is better for time and memory.",
        *([f"No control checks the tokens of: {', '.join(f'`{a}`' for a in ungated)}."] if ungated else []),
        "",
        "| point | metric | " + " | ".join(arms) + " |",
        "|---|---|" + "|".join("---:" for _ in arms) + "|",
    ]
    for point, metric in rows:
        cells = [_ratio(point["arms"][arm]["metrics"][metric]) for arm in arms]
        headline = " **headline**" if point["point"].get("headline") and metric == "time_per_output_token_s" else ""
        lines.append(f"| {point['point']['id']}{headline} | {METRIC_LABELS[metric]} | " + " | ".join(cells) + " |")
    if not rows:
        lines.append("| no timed points | gates failed | " + " | ".join("—" for _ in arms) + " |")
    return "\n".join(lines + _serve_lines(doc, arms)) + "\n"


def _cell(value: Any) -> str:
    return "unset" if value is None else str(value)


def _serve_lines(doc: Mapping[str, Any], arms: list[str]) -> list[str]:
    """Each arm's settings that differ from `vllm serve`'s defaults (`serve_defaults.differences`), where its engine
    reported them."""

    serve = doc.get("vllm_serve")
    if not serve:
        return []
    ctx = serve["context"]
    lines = ["", f"Where each arm differs from `vllm serve MODEL`'s defaults at vLLM {serve['vllm']}, on {ctx['card']} "
                 "and this checkpoint:"]
    for arm in arms:
        if (got := serve["arms"].get(arm)) is None:
            continue
        lines += ["", f"`{arm}`" + (f": {got['note']}." if got.get("note") else ""), "",
                  "| setting | this arm | `vllm serve` | | where vLLM sets it |", "|---|---|---|---|---|"]
        lines += [f"| {r['setting']} | {_cell(r['arm'])} | {_cell(r['vllm_serve'])} | {r['whose']} | {r['where']} |"
                  for r in got["differences"]]
    return lines
