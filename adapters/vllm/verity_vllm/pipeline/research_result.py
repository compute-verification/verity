"""The sweep row's `result.json` for the `research` runner (schema research/result/v0.1, tools/research/src/research/result.py).

    verity-vllm research-result <evidence-dir> <row-id> <role> <stages> [--run-dir DIR] [--run-id ID]

`row_pod.sh` calls this on EXIT when the runner set `RESEARCH_RUN_DIR`: the run record then states, separately from the process
exit, whether the workload's own checker (the per-stage verdicts of `stages.txt`) reported PASS for every requested stage, what
actually ran (row, role, checkpoint, tree identity, Program digest), and which small records were copied beside it under
`evidence/` (the multi-GB evidence itself stays in the sweep dir / on the volume; its absolute path is in the fingerprint).
Stdlib only; never raises out of `main` (a broken result file would hide the row's own verdicts).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from verity_vllm import settings
from verity_vllm.config import option

#: none of them private (`research_outputs.private`): the run publishes them as its run_files
RECORDS = ("stages.txt", "row.log", "source_identity.json", "admission.json", "verdict.json", "manifest_verify.log",
           "build_request/digest.txt", "call_boundaries.json", "commit/versions.json", "match/run.json", "match_compare.json",
           "match_decomp.json", "global_match.json")
MAX_COPY = 8 << 20


def verdicts(stages_txt: Path) -> dict[str, tuple[str, str]]:
    """stage -> (verdict, rest) from the LAST line per stage of stages.txt (`<stage> <PASS|FAIL|...> <utc> <detail>`)."""
    out: dict[str, tuple[str, str]] = {}
    if not stages_txt.exists():
        return out
    for line in stages_txt.read_text(errors="replace").splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


def build_result(d: Path, row: str, role: str, stages: list[str], run_id: str | None, run_dir: Path | None) -> dict:
    v = verdicts(d / "stages.txt")
    requested = [s for s in stages if s in ("build", "match", "commit")]
    status = "not_run"
    if requested and all(s in v for s in requested):
        status = "passed" if all(v[s][0] == "PASS" for s in requested) else "failed"
    elif any(s in v and v[s][0] != "PASS" for s in requested) or v.get("precheck", ("PASS", ""))[0] != "PASS":
        status = "failed"
    detail = "; ".join(f"{s} {v[s][0]}" for s in v)
    fp: dict = {"row": row, "role": role, "stages": requested, "evidence_dir": str(d)}
    sid = d / "source_identity.json"
    if sid.exists():
        try:
            doc = json.loads(sid.read_text())
            fp["source_identity"] = {k: doc.get(k) for k in ("requested_root", "requested_sha", "ok", "sys_path0") if k in doc}
            ids = doc.get("identities") or {}
            fp["tree_sha"] = next((i.get("git_sha") for i in ids.values() if isinstance(i, dict) and i.get("git_sha")), None)
        except (OSError, ValueError):
            fp["source_identity"] = "unreadable"
    m = re.search(r"digest[= ]([0-9a-f]{16,64})", v.get("build", ("", ""))[1])
    if m:
        fp["program_digest"] = m.group(1)
    meas = []
    for s in requested:
        m = re.search(r"wall=(\d+)s", v.get(s, ("", ""))[1])
        if m:
            meas.append({"name": f"{s}_wall", "value": int(m.group(1)), "unit": "s"})
    files: list[str] = []
    if run_dir is not None:
        ev = run_dir / "evidence"
        ev.mkdir(parents=True, exist_ok=True)
        for rel in RECORDS:
            src = d / rel
            if src.is_file() and src.stat().st_size <= MAX_COPY:
                dst = ev / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                files.append(f"evidence/{rel}")
    return {
        "schema": "research/result/v0.1",
        "run_id": run_id or settings.RESEARCH_RUN_ID or "",
        "workload_fingerprint": fp,
        "validation": {"status": status, "checker": "verity_vllm.ops.row_pod stages.txt", "detail": detail or "no stages.txt",
                       "evidence": [f for f in files if f.endswith("stages.txt")]},
        "measurements": meas,
        "measurement_files": [f for f in files if f.endswith((".json", ".txt"))],
        "artifacts": files,
    }


@dataclass(frozen=True, kw_only=True)
class Options:
    evidence_dir: str = option("evidence_dir")
    row: str = option("row")
    role: str = option("role")
    stages: str = option("stages", help="comma list, as passed to row_pod.sh")
    run_dir: str | None = option("--run-dir", env="RESEARCH_RUN_DIR", default=None)
    run_id: str | None = option("--run-id", env="RESEARCH_RUN_ID", default=None)


def main(a: Options) -> int:
    if not a.run_dir:
        print("research_result: RESEARCH_RUN_DIR not set; nothing written", file=sys.stderr)
        return 0
    try:
        run_dir = Path(a.run_dir)
        doc = build_result(Path(a.evidence_dir), a.row, a.role, a.stages.split(","), a.run_id, run_dir)
        tmp = run_dir / "result.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        os.replace(tmp, run_dir / "result.json")
        print(f"research_result: {run_dir / 'result.json'} validation={doc['validation']['status']}")
    except Exception as e:  # noqa: BLE001 -- never mask the row's own verdicts
        print(f"research_result: FAILED to write result.json: {type(e).__name__}: {e}", file=sys.stderr)
    return 0
