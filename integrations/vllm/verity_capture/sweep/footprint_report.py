"""R13 P0 (C-COLLECTOR-FOOTPRINT): render the `VERITY_FOOTPRINT` timeline (JSONL from `native_host._Footprint`) into a memory-timeline JSON
and a short markdown table -- the design input for the streaming absorb/release work.

    python -m verity_capture.sweep.footprint_report <timeline.jsonl> [--out out/gen/sweep/evidence/p0/footprint_profile] [--label ...]

Per step (serving-thread begin/end + worker absorb): tokens, arena block taken, bytes staged (by value family), when the step ended, when
the tree absorbed it (leaves + levels done), what was retained afterwards (device copy / host copy / levels), the arena pool size, the
device retained total and torch allocated / device free at each event.  Peak resident and its composition come from the last event
before the peak of `cuda.allocated`."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MiB = 1 << 20


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_runs(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The timeline file holds every run of the process (warm-up, replay, pairs); a `begin_step` of step 0 starts a new run."""
    runs: list[list[dict[str, Any]]] = []
    for e in events:
        if e["event"] == "begin_step" and e.get("step") == 0 or not runs:
            runs.append([])
        runs[-1].append(e)
    return runs


def summarise(events: list[dict[str, Any]], run: int = -1) -> dict[str, Any]:
    runs = split_runs(events)
    events = runs[run] if runs else []
    steps: dict[int, dict[str, Any]] = defaultdict(dict)
    peak = {"allocated": 0, "event": None}
    families_total: Counter = Counter()
    for e in events:
        c = e.get("cuda") or {}
        if c.get("allocated", 0) > peak["allocated"]:
            peak = {"allocated": c["allocated"], "event": e}
        st = e.get("step")
        if st is None:
            continue
        s = steps[st]
        kind = e["event"]
        if kind == "begin_step":
            s.update(step=st, ntok=e.get("ntok"), nreq=e.get("nreq"), t_begin=e["t"], arena_block_bytes=e.get("arena_block_bytes"),
                     arena_total_at_begin=e.get("arena_total_bytes"), arena_free_blocks_at_begin=len(e.get("arena_free_blocks") or []),
                     retained_device_at_begin=e.get("retained_device_total"), allocated_at_begin=c.get("allocated"), device_free_at_begin=c.get("device_free"))
        elif kind == "end_step":
            s.update(t_end=e["t"], staged_bytes=e.get("staged_bytes", e.get("used")), used=e.get("used"), backlog_at_end=e.get("backlog"), tensors=e.get("tensors"),
                     allocated_at_end=c.get("allocated"))
        elif kind == "absorbed":
            fam = {k: v["bytes"] for k, v in (e.get("families") or {}).items()}
            for k, v in fam.items():
                families_total[k] += v
            s.update(t_absorbed=e["t"], padded=e.get("padded"), levels_bytes=e.get("levels_bytes"), levels_on_device=e.get("levels_device"),
                     keep_bytes=e.get("keep_bytes"), keep_on_device=e.get("keep_device"), retain=e.get("retain"),
                     retained_device_after=e.get("retained_device_total"), families=fam, allocated_after_absorb=c.get("allocated"),
                     device_free_after_absorb=c.get("device_free"))
    rows = [steps[k] for k in sorted(steps)]
    for s in rows:
        if "t_end" in s and "t_begin" in s:
            s["step_wall_s"] = round(s["t_end"] - s["t_begin"], 6)
        if "t_absorbed" in s and "t_end" in s:
            s["absorb_latency_s"] = round(s["t_absorbed"] - s["t_end"], 6)     # lifetime of the staged block after the step ended
    pe = peak["event"] or {}
    pc = pe.get("cuda") or {}
    return {"steps": rows, "n_steps": len(rows), "run_index": run, "runs_in_file": len(runs), "families_total_bytes": dict(families_total),
            "peak": {"allocated": peak["allocated"], "reserved": pc.get("reserved"), "device_free": pc.get("device_free"), "device_total": pc.get("device_total"),
                     "at_event": pe.get("event"), "at_step": pe.get("step"), "t": pe.get("t"),
                     "retained_device_total": pe.get("retained_device_total"), "arena_total_bytes": pe.get("arena_total_bytes")},
            "release_events": [e for e in events if e["event"] == "release_staging"]}


def table(summary: dict[str, Any], label: str, max_rows: int = 12) -> str:
    rows = summary["steps"]
    big = sorted(rows, key=lambda s: -(s.get("staged_bytes") or 0))[:3]
    keep = {s["step"] for s in big} | {s["step"] for s in rows[:3]} | {s["step"] for s in rows[-2:]}
    out = [f"### {label}", "",
           "| step | tok | req | arena block | staged | of which fa2 hidden | gemm | norm+rope+act+attn | levels | kept copy | where | retained on device after | absorb latency | torch alloc after | device free after |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    shown = 0
    for s in rows:
        if s["step"] not in keep and shown >= max_rows:
            continue
        fam = s.get("families") or {}
        gemm = sum(v for k, v in fam.items() if k.startswith("gemm"))
        other = sum(v for k, v in fam.items() if k in ("norm", "rope", "act", "attention_out", "embedding", "logits", "sampled_ids", "input_ids", "other"))
        where = "device" if s.get("keep_on_device") else ("host" if s.get("keep_on_device") is False else "none")
        out.append(f"| {s['step']} | {s.get('ntok')} | {s.get('nreq')} | {(s.get('arena_block_bytes') or 0) / MiB:.0f} MiB | {(s.get('staged_bytes') or 0) / MiB:.1f} MiB | "
                   f"{fam.get('fa2_hidden_m1', 0) / MiB:.1f} MiB | {gemm / MiB:.1f} MiB | {other / MiB:.1f} MiB | {(s.get('levels_bytes') or 0) / MiB:.1f} MiB | "
                   f"{(s.get('keep_bytes') or 0) / MiB:.1f} MiB | {where} | {(s.get('retained_device_after') or 0) / MiB:.0f} MiB | "
                   f"{(s.get('absorb_latency_s') or 0) * 1e3:.1f} ms | {(s.get('allocated_after_absorb') or 0) / MiB:.0f} MiB | {(s.get('device_free_after_absorb') or 0) / MiB:.0f} MiB |")
        shown += 1
    p = summary["peak"]
    out += ["", f"run {summary['run_index']} of {summary['runs_in_file']} in the file; " + f"peak torch-allocated **{p['allocated'] / MiB:.0f} MiB** at `{p['at_event']}` step {p['at_step']} (device free {(p['device_free'] or 0) / MiB:.0f} MiB of "
            f"{(p['device_total'] or 0) / MiB:.0f}); retained on device there {(p['retained_device_total'] or 0) / MiB:.0f} MiB, arena pool {(p['arena_total_bytes'] or 0) / MiB:.0f} MiB.",
            "", "bytes staged per value family over the run: " + ", ".join(f"{k} {v / MiB:.1f} MiB" for k, v in sorted(summary["families_total_bytes"].items(), key=lambda kv: -kv[1]))]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--out", default=None, help="write <out>.json and <out>.md (default: next to the timeline)")
    ap.add_argument("--label", default="")
    ap.add_argument("--append", action="store_true", help="append the table to <out>.md instead of overwriting")
    ap.add_argument("--run", type=int, default=-1, help="which run of the file to summarise (0 = warm-up, -1 = last)")
    a = ap.parse_args()
    ev = load(a.timeline)
    s = summarise(ev, a.run)
    label = a.label or Path(a.timeline).stem
    base = Path(a.out) if a.out else Path(a.timeline).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    j = base.with_suffix(".json")
    prev = json.loads(j.read_text()) if (a.append and j.exists()) else {}
    prev[label] = s
    j.write_text(json.dumps(prev, indent=1) + "\n")
    md = base.with_suffix(".md")
    txt = table(s, label)
    if a.append and md.exists():
        md.write_text(md.read_text().rstrip() + "\n\n" + txt + "\n")
    else:
        md.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
