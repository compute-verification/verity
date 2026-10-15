"""Markdown ratio tables for the shared vLLM benchmark."""

from __future__ import annotations

from typing import Any, Mapping

METRIC_LABELS = {
    "time_per_output_token_s": "time / output token",
    "time_to_first_token_s": "time to first token",
    "tokens_per_s": "output tokens / second",
    "peak_gpu_memory_bytes": "peak GPU memory",
}


def _ratio(metric: Mapping[str, Any]) -> str:
    r = metric.get("ratio_to_stock")
    if not r:
        return "—"
    mean = r["mean"]
    ci = r.get("ci95")
    return f"{mean:.3f}×" if ci is None else f"{mean:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}]"


def render(doc: Mapping[str, Any]) -> str:
    """Render headline time per output token first, then the rest of the sweep."""

    arms = [a["id"] for a in doc["identity"]["arms"]]
    points = sorted(doc["points"], key=lambda p: (not p["point"].get("headline", False), p["point"]["id"]))
    metrics = tuple(METRIC_LABELS)
    rows: list[tuple[dict, str]] = []
    for point in points:
        order = metrics if point["point"].get("headline") else metrics
        rows.extend((point, metric) for metric in order)
    rows.sort(key=lambda x: (
        not (x[0]["point"].get("headline", False) and x[1] == "time_per_output_token_s"),
        not x[0]["point"].get("headline", False),
        x[0]["point"]["id"],
        metrics.index(x[1]),
    ))
    lines = [
        f"Baseline: `{doc['baseline']}`. Ratios are arm / baseline; lower is better for time and memory.",
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
    return "\n".join(lines) + "\n"
