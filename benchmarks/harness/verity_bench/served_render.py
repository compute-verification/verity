"""The served table (`served_table`) as Markdown: PoUW's live rows, each protocol's served rows with their ratios over
stock, the conversion factors, the breakdowns beside a protocol's run of record, and the runs passed over."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity_bench.breakdown import AUDIT, AUDIT_SPANS, DECODE_ROWS, PLAIN, PLAIN_SPANS, ms_cell
from verity_bench.served_table import CAVEAT_KEYS, LIVE_CAVEAT_KEYS, PER_DECODE, PER_OUTPUT


def _ratio(summary: Mapping[str, Any] | None) -> str:
    if summary is None or summary.get("mean") is None:
        return "—"
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}×" if ci is None else f"{summary['mean']:.3f}× ± {(ci[1] - ci[0]) / 2:.3f}"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _caveat(caveats: Sequence[Mapping[str, Any]], keys: Sequence[str] = CAVEAT_KEYS) -> str:
    newest = {label["key"]: label["value"] for label in caveats}
    return _cell(" / ".join(str(newest[key]) for key in keys if key in newest))


def _art(art: str | None) -> str:
    return f"`{art[:16]}`" if art else "—"


def _sources(row: Mapping[str, Any]) -> list[str]:
    if row["run"] is None:
        return ["—", "no passing run", "—", ""]
    record = f" (record {_art(row['record'])})" if row.get("record") else ""
    return [str(row["reps"]), row["run"], _art(row["art"]) + record, _caveat(row["caveats"])]


def _breakdown(points: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Where the bandwidth audit's time goes, by span (`verity-bench breakdown`)",
        "",
        ("Each cell is ms an output token, a difference and not a ratio: the mean over the point's paired reps ± its "
         f"95% half-width (Student t). `{AUDIT_SPANS}` − `{PLAIN_SPANS}` is split into rows that sum to it: the "
         "top-up's contention from the arms with the top-up off, every other row a part of the spans arms with the "
         f"top-up off. `{AUDIT}` − `{PLAIN}` (spans off) is the audit's cost; the spans' marks add their own time to "
         "the spans arms. Each spans arm's step by part: `verity-bench breakdown`."),
    ]
    for protocol in dict.fromkeys(point["protocol"] for point in points):
        group = [point for point in points if point["protocol"] == protocol]
        rows = [{row["row"]: row for row in point["rows"]} for point in group]
        lines += ["", f"{protocol}:", "",
                  "| row | " + " | ".join(f"{point['point']}, ms an output token | share" for point in group) + " |",
                  "|---|" + "---:|---:|" * len(group),
                  f"| `{AUDIT}` − `{PLAIN}` (credited lines, spans off) | "
                  + " | ".join(f"{ms_cell(point['credited'])} | " for point in group) + " |",
                  f"| **`{AUDIT_SPANS}` − `{PLAIN_SPANS}`** | "
                  + " | ".join(f"**{ms_cell(point['total'])}** | 100%" for point in group) + " |"]
        for name in rows[0]:
            part = name.removeprefix("decode ")
            label = f"{name}: {DECODE_ROWS[part]}" if part in DECODE_ROWS else name
            cells = []
            for by_name in rows:
                row = by_name.get(name)
                share = "—" if row is None or row["share"] is None else f"{row['share']:.0%}"
                cells.append(("—" if row is None else ms_cell(row["ms_per_token"])) + f" | {share}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("| sum of the rows | " + " | ".join(f"{point['sum']:.3f} | " for point in group) + " |")
        lines += [""] + [f"- {point['point']}: `{point['spec']}`, {point['reps']} reps, run {point['run']}, art "
                         f"{_art(point['art'])}" + (f" (record {_art(point['record'])})" if point["record"] else "")
                         + (f"; {caveat}" if (caveat := _caveat(point["caveats"])) else "") for point in group]
    return lines


def render(table: Mapping[str, Any]) -> str:
    rows, factors = table["rows"], table["factors"]
    lines = [
        "# Served overhead on Llama-3.1-8B",
        "",
        "## PoUW: run of record (live rows)",
        "",
        ("Each row is one in-process vLLM run on one GPU: Llama-3.1-8B int8 (the verity-int8 pin), every engine on "
         "vLLM's defaults (torch.compile, CUDA graphs), prefix caching off, decode at batch B with max_num_batched_tokens "
         "T. A decode step is (t(N2) − t(N1)) / (N2 − N1) "
         "over two greedy generate calls of N1 = 32 and N2 = 160 output tokens per request, so prefill and per-call "
         "overhead cancel; it returns one token for each of the B requests. The ratio is the PoUW arm's median decode "
         "step over its control's (the int8-exact engine without PoUW, same run, same GPU), per step, not per token: "
         "it is not comparable with the per-token ratios over stock FP8 below. A row stands only when its run's gates "
         "all passed: the device gate on poisoned (0xA5) buffers, the arm's tokens equal to the int8-exact engine's at "
         "the gate and at every timed rep, and a passing live window at the run's end on off-node coins."),
        "",
        ("| device | machine | B | T | PoUW decode step, ms | control | control's step, ms | PoUW / control, per decode "
         "step | reps | coins | run | run record | caveat |"),
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in table["live"]:
        lines.append("| " + " | ".join([
            row["device"], str(row["machine"]), str(row["batch"]), str(row["t"]), f"{row['step_ms']:.3f}",
            f"`{row['control']}`", f"{row['control_ms']:.3f}", f"{row['ratio']:.3f}×", str(row["reps"]), row["coins"],
            row["run"], _art(row["record"]), _caveat(row["caveats"], LIVE_CAVEAT_KEYS)]) + " |")
    lines += ["", "PoUW arms and their gates' form:", ""]
    lines += [f"- {row['device']} (`{row['run']}`): `{row['arm']}`; {_cell(row['form'])}" for row in table["live"]]
    lines += [
        "",
        "## Served rows, one GPU each (`vllm_bench_point`)",
        "",
        ("The device is the GPU the run's engines saw (the result's `identity.gpu`); a ratio's arm and stock ran on it in "
         "the same run. "
         f"Ratios are a protocol's arm over `{table['served_stock']}` (stock FP8 as vLLM serves it), paired by rep: mean "
         "± 95% half-width (Student t), or the mean alone at one rep; lower is better. A row is its protocol's run of "
         "record (its runs, or an art's, labelled `run_of_record`) or, without one, the newest passing run of its spec and point in "
         "the evidence store; a result labelled `superseded_by` is skipped. The caveat is the newest `label` and `note` "
         "on the result and its run."),
        "",
        ("| protocol | device | spec | point | / served stock, per output token | / served stock, per decode token "
         "| / eager stock, per output token | reps | run | art | caveat |"),
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        eager = _ratio(row["over_eager"][PER_OUTPUT]) if row["over_eager"] else "—"
        served = row["over_served"] or {}
        lines.append("| " + " | ".join([row["protocol"], row["device"] or "—", f"`{row['spec']}`", row["point"],
                                        _ratio(served.get(PER_OUTPUT)), _ratio(served.get(PER_DECODE)), eager,
                                        *_sources(row)]) + " |")
    lines += ["", "Arms:", ""]
    seen = set()
    for row in rows:
        if row["protocol"] in seen:
            continue
        seen.add(row["protocol"])
        own = (f"; over eager stock, `{row['own_arm']}` / `{row['eager_stock']}`" if row["eager_stock"]
               else "; no eager stock in its runs")
        lines.append(f"- {row['protocol']}: `{row['arm']}` / `{row['served_stock']}`{own}.")
    lines += [
        "",
        f"## Conversion factors: `{table['served_stock']}` / batch-invariant eager `stock-fp8`",
        "",
        "| spec | point | per output token | per decode token | reps | run | art | caveat |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in factors:
        factor = row["factor"] or {}
        lines.append("| " + " | ".join([f"`{row['spec']}`", row["point"], _ratio(factor.get(PER_OUTPUT)),
                                        _ratio(factor.get(PER_DECODE)), *_sources(row)]) + " |")
    if table.get("breakdowns"):
        lines += _breakdown(table["breakdowns"])
    if table["skipped"]:
        lines += ["", "Passed over (a run that can't stand):", ""]
        lines += [f"- `{s['spec']}` {s['point'] or ''}: {s['run']} ({_art(s['art'])}): {_cell(s['why'])}"
                  for s in table["skipped"]]
    return "\n".join(lines) + "\n"
