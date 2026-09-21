"""`research compare BASELINE CANDIDATE`: what differs between two runs, and the measured ratios where both have a number.

Descriptive only.  It shows source, tool, machine, fingerprint (expected and observed, key by key), environment, execution,
instrumentation, outcome, and measurement differences; it does not decide which differences invalidate the comparison.  That is the
workload's contract (a changed ELF may be the point of the experiment; a changed relation is not).  Ratios are labeled
candidate / baseline.  One pair of runs is one measured ratio, not a confidence interval; repetitions inside the workload's own
files are not flattened here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import record
from . import runs as R


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _env_view(job: dict[str, Any] | None) -> dict[str, Any]:
    e = (job or {}).get("environment") or {}
    gpu = e.get("gpu") or {}
    return {"hostname": e.get("hostname"), "host_class": e.get("host_class"), "kernel": e.get("kernel"), "python": e.get("python"),
            "cpu_count": e.get("cpu_count"), "host_memory_total_bytes": e.get("host_memory_total_bytes"),
            "cgroup_effective_memory_max_bytes": _get(e, "cgroup", "effective_memory_max_bytes"),
            "gpu_devices": [d.get("name") for d in gpu.get("devices") or []] if gpu.get("available") else "none",
            "gpu_driver": next((d.get("driver") for d in gpu.get("devices") or []), None),
            "cuda_visible_devices": gpu.get("cuda_visible_devices")}


def _dict_diff(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[tuple[str, Any, Any]]:
    a = a or {}
    b = b or {}
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k, "<absent>"), b.get(k, "<absent>")))
    return out


def compare(base_dir: Path, cand_dir: Path) -> dict[str, Any]:
    sb, sc = R.summarize(base_dir), R.summarize(cand_dir)
    jb, jc = record.read_job(base_dir), record.read_job(cand_dir)
    sections: dict[str, list[tuple[str, Any, Any]]] = {}
    sections["identity"] = _dict_diff(
        {"tree_sha": sb["tree_sha"], "workload_digest": sb["workload_digest"], "tool_sha": _get(sb["tool"], "research_git_sha"),
         "tool_snapshot": _get(sb["tool"], "research_snapshot"), "classifier": _get(sb["tool"], "classifier_version"),
         "machine": _get(sb["remote"], "machine"), "pod_id": _get(sb["remote"], "pod_id"), "project": sb["project"], "stage": sb["stage"]},
        {"tree_sha": sc["tree_sha"], "workload_digest": sc["workload_digest"], "tool_sha": _get(sc["tool"], "research_git_sha"),
         "tool_snapshot": _get(sc["tool"], "research_snapshot"), "classifier": _get(sc["tool"], "classifier_version"),
         "machine": _get(sc["remote"], "machine"), "pod_id": _get(sc["remote"], "pod_id"), "project": sc["project"], "stage": sc["stage"]})
    sections["fingerprint_expected"] = _dict_diff(sb["fingerprint_expected"], sc["fingerprint_expected"])
    sections["fingerprint_observed"] = _dict_diff(sb["fingerprint_observed"], sc["fingerprint_observed"])
    sections["environment"] = _dict_diff(_env_view(jb), _env_view(jc))
    sections["execution"] = _dict_diff(
        {"command": sb["command"], "timeout_s": _get(jb, "execution", "timeout_s"), "inputs": [(i.get("name"), i.get("sha256")) for i in _get(jb, "computation", "inputs") or []],
         "profile": _get(jb, "computation", "profile"), "policy_sha256": _get(jb, "execution", "policy_sha256")},
        {"command": sc["command"], "timeout_s": _get(jc, "execution", "timeout_s"), "inputs": [(i.get("name"), i.get("sha256")) for i in _get(jc, "computation", "inputs") or []],
         "profile": _get(jc, "computation", "profile"), "policy_sha256": _get(jc, "execution", "policy_sha256")})
    sections["instrumentation"] = _dict_diff(
        {"instruments": _get(jb, "telemetry", "instruments"), "cadence_s": _get(jb, "telemetry", "cadence_s"), "job_schema": sb["schema"]},
        {"instruments": _get(jc, "telemetry", "instruments"), "cadence_s": _get(jc, "telemetry", "cadence_s"), "job_schema": sc["schema"]})
    outcomes = {
        "baseline": {"state": sb["state"], "rc": sb["rc"], "class": sb["class"], "result": sb["result"], "validation": sb["validation"], "preserved": bool(sb["preserved"]), "wall_s": sb["wall_s"]},
        "candidate": {"state": sc["state"], "rc": sc["rc"], "class": sc["class"], "result": sc["result"], "validation": sc["validation"], "preserved": bool(sc["preserved"]), "wall_s": sc["wall_s"]},
    }
    mb = {m["name"]: m for m in sb["measurements"] or []}
    mc = {m["name"]: m for m in sc["measurements"] or []}
    measurements = []
    for name in sorted(set(mb) | set(mc)):
        b, c = mb.get(name), mc.get(name)
        row: dict[str, Any] = {"name": name, "baseline": b and b["value"], "candidate": c and c["value"],
                               "unit": (b or c)["unit"], "unit_mismatch": bool(b and c and b["unit"] != c["unit"])}
        if b and c and not row["unit_mismatch"] and isinstance(b["value"], (int, float)) and isinstance(c["value"], (int, float)):
            row["delta_candidate_minus_baseline"] = c["value"] - b["value"]
            row["ratio_candidate_over_baseline"] = (c["value"] / b["value"]) if b["value"] else None
        measurements.append(row)
    caveats = []
    if not (sb["result_valid"] and sc["result_valid"]):
        caveats.append("at least one run has no valid result.json; measurements above come only from valid results")
    if sb["validation"] != "passed" or sc["validation"] != "passed":
        caveats.append("at least one run did not report validation passed; a speedup on an unvalidated run is not a result")
    if sections["fingerprint_observed"]:
        caveats.append("observed fingerprints differ; whether that is the experiment or an invalid comparison is the workload contract's call")
    if sections["environment"]:
        caveats.append("environments differ; the experiment decides whether the machines are comparable")
    if any(m.get("ratio_candidate_over_baseline") is not None for m in measurements):
        caveats.append("one pair of runs gives a measured ratio, not a confidence interval")
    return {"baseline": {"id": sb["run_id"], "dir": sb["dir"]}, "candidate": {"id": sc["run_id"], "dir": sc["dir"]},
            "differences": {k: [{"field": f, "baseline": a, "candidate": b} for f, a, b in v] for k, v in sections.items()},
            "outcomes": outcomes, "measurements": measurements, "caveats": caveats}


def render(rep: dict[str, Any]) -> str:
    lines = [f"baseline   {rep['baseline']['id']}", f"candidate  {rep['candidate']['id']}", ""]
    any_diff = False
    for sec, rows in rep["differences"].items():
        if not rows:
            continue
        any_diff = True
        lines.append(f"[{sec}]")
        for r in rows:
            lines.append(f"  {r['field']:<34} {_short(r['baseline'])}  ->  {_short(r['candidate'])}")
    if not any_diff:
        lines.append("[identity/fingerprint/environment/execution/instrumentation] no recorded differences")
    lines.append("")
    lines.append("[outcomes]")
    for who in ("baseline", "candidate"):
        o = rep["outcomes"][who]
        lines.append(f"  {who:<10} {o['state']} rc={o['rc']} class={o['class']} result={o['result']} validation={o['validation']} preserved={o['preserved']} wall_s={o['wall_s']}")
    if rep["measurements"]:
        lines.append("")
        lines.append("[measurements]                          baseline        candidate       candidate/baseline")
        for m in rep["measurements"]:
            ratio = m.get("ratio_candidate_over_baseline")
            lines.append(f"  {m['name']:<36} {_num(m['baseline']):>14}  {_num(m['candidate']):>14}  {('x%.4f' % ratio) if ratio is not None else '-':>10}  {m['unit']}"
                         + ("  UNIT MISMATCH" if m["unit_mismatch"] else ""))
    if rep["caveats"]:
        lines.append("")
        lines.append("[caveats]")
        lines += [f"  - {c}" for c in rep["caveats"]]
    return "\n".join(lines)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _num(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.6g}" if isinstance(v, float) else str(v)
