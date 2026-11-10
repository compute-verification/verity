"""`verity-vllm epoch`: the re-baseline epoch's row job (`ops/epoch_row.sh`, `epoch_store.sh`, `epoch_failfast.sh`) asks these; each
prints one line or document the script reads.

    verity-vllm epoch job RUN_DIR --custody-dir DIR [--store-reserve-s N] [--job-deadline UTC]   -> "RUN_END JOB_END KEY_EXPIRES"
    verity-vllm epoch resolve ROW                                                                 -> "ROLE REPO REV WORLD"
    verity-vllm epoch row-state ROW_DIR [--header OUT]                                           -> "FAMILIES OUTCOME MANIFEST_VERIFY"
    verity-vllm epoch word-check ROW ROW_DIR --out JSON [--world W] [--build-word-check V] [--build-max-gates V] [--allowed-max-gates V]
    verity-vllm epoch digests ROW_DIR... [--out JSON] [--markdown MD]
    verity-vllm epoch store-meta ROW_DIR --kind K --role R --rownum N --run-id ID --epoch-sha SHA [--gpu G] [--pod P] [--epoch E]
    verity-vllm epoch custody-env CUSTODY_DIR                                                     -> export lines of the run's key
    verity-vllm epoch device [--expect-sms N] [--expect-cc CC]                                    -> the device, once torch and vLLM load

`job`: the run's timeout end (status.json's `running` start + timeout_s, or control/timeout_s after `tele set-timeout`), the job's end
(that less the store reserve, or the job deadline if earlier) and the custody key's expiry, epoch seconds / UTC, "-" for unknown.
`word-check` is the strict word check before a Commit, from the Build's own manifest step: the manifest states the query of record with a
partition; manifest.log has one `Q_word_v1{X=16,W=32,...}: N calls -> U units over G gates` line per component (the workload's requests x
the TP world) and no traceback or query-rule violation; the Build ran with the record's word check (its `VERITY_WORD_CHECK` unset or
16/32) and no `VERITY_QWORD_MAX_GATES` but the one a GO allows for the row -- the script passes the job's values.  Exit 0 PASS, 1 FAIL.
`digests` is `verity-vllm/rebaseline-digests/v1`: per rank the Programs, then the manifest, partition digests, run roots and verdict.
`device` exits 3 when the device is not EXPECT_SMS SMs at compute capability EXPECT_CC.
"""
from __future__ import annotations

import calendar
import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT, option, row_file

WORD_LINE = re.compile(r"Q_word_v1\{X=16,W=32[^}]*\}: (\d+) calls -> (\d+) units over (\d+) gates")
ROLE_RANGE = re.compile(r"([A-Z]+)(\d+)-(?:[A-Z]+)?(\d+)")
ROLE = re.compile(r"([A-Z]+)(\d+)")
KEY_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def _doc(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _epoch(utc: str | None) -> int | None:
    if not utc:
        return None
    return calendar.timegm(time.strptime(utc[:19].rstrip("Z"), "%Y-%m-%dT%H:%M:%S" if len(utc.rstrip("Z")) > 16 else "%Y-%m-%dT%H:%M"))


def run_end(run_dir: str | Path) -> int | None:
    """When the run's timeout ends (epoch seconds), or None."""
    d = Path(run_dir)
    runs = [t for t in _doc(d / "status.json").get("transitions") or [] if t.get("state") == "running"]
    if not runs or not runs[-1].get("started_utc"):
        return None
    timeout = _doc(d / "control" / "timeout_s").get("timeout_s", runs[-1].get("timeout_s"))
    return int(_epoch(runs[-1]["started_utc"]) + float(timeout)) if timeout else None


def job(run_dir: str | Path, custody_dir: str | Path, store_reserve_s: int, job_deadline: str | None) -> tuple[int | None, int | None, str | None]:
    end = run_end(run_dir)
    jend = _epoch(job_deadline)
    if end is not None and (jend is None or jend > end - store_reserve_s):
        jend = end - store_reserve_s
    return end, jend, _doc(Path(custody_dir) / "cred.json").get("expires_utc")


def has_role(checkpoint: dict[str, Any], role: str) -> bool:
    """A checkpoint entry serves `role`: its `role`, or one of its comma-separated roles, is `role` or a range holding it (`B1-B4`)."""
    r = ROLE.fullmatch(role)
    for tok in (t.strip() for t in str(checkpoint.get("role") or "").split(",")):
        m = ROLE_RANGE.fullmatch(tok)
        if tok == role or (m and r and m.group(1) == r.group(1) and int(m.group(2)) <= int(r.group(2)) <= int(m.group(3))):
            return True
    return False


def resolve(row: str) -> tuple[str, str, str, int]:
    """The row's checkpoint role, repo, revision and TP world (its workload and `manifests/checkpoints.json`)."""
    wl = _doc(row_file(row))
    role = wl["case"]
    cps = _doc(ROOT / "manifests" / "checkpoints.json")["checkpoints"]
    cp = next((c for c in cps if c.get("role") == role), None) or next(c for c in cps if has_role(c, role) and c.get("local_path"))
    tp = int((wl.get("sweep") or {}).get("tp") or wl.get("tp") or (2 if "__tp2__" in row else 1))
    return role, cp["repo"], cp["revision"], tp


def row_state(row_dir: str | Path, header: str | None) -> tuple[list[str], str | None, bool | None]:
    """The manifest's required families, the Commit verdict's outcome and manifest-verify's `ok`; the manifest's header to `header`."""
    d = Path(row_dir)
    man = _doc(d / "manifest.json")
    if man and header:
        keys = ("schema", "manifest_digest", "program_digest", "complete", "populations", "query")
        Path(header).write_text(json.dumps({k: man.get(k) for k in keys}, indent=1))
    fams = list((man.get("query") or {}).get("required_families") or [])
    return fams, _doc(d / "verdict.json").get("outcome"), _doc(d / "commit" / "manifest_verify.json").get("ok")


def word_check(row: str, row_dir: str | Path, world: int, build_word_check: str, build_max_gates: str, allowed_max_gates: str) -> dict[str, Any]:
    from verity_vllm.query.required import QUERY_OF_RECORD
    d = Path(row_dir)
    problems: list[str] = []
    man = _doc(d / "manifest.json")
    q = man.get("query") or {}
    if q.get("query_id") != QUERY_OF_RECORD or not QUERY_OF_RECORD.startswith("Q_word_v1{X=16,W=32"):
        problems.append(f"manifest query {q.get('query_id')} is not the query of record {QUERY_OF_RECORD}")
    if not q.get("partition"):
        problems.append("the manifest names no partition (query.partition)")
    log = (d / "manifest.log").read_text(errors="replace") if (d / "manifest.log").is_file() else ""
    lines = WORD_LINE.findall(log)
    wl = _doc(row_file(row))
    batch = int((wl.get("sweep") or {}).get("batch") or len(wl.get("requests") or []) or 1)
    need = batch * max(1, world)
    if len(lines) < need:
        problems.append(f"manifest.log has {len(lines)} Q_word lines, fewer than the {need} components (batch {batch} x world {world})")
    if re.search(r"Traceback|QueryRuleViolation|RecomputeViolation", log):
        problems.append("manifest.log shows a traceback or a query-rule violation")
    if build_word_check not in ("", "16/32"):
        problems.append(f"VERITY_WORD_CHECK={build_word_check} in the Build's environment")
    if build_max_gates and build_max_gates != allowed_max_gates:
        problems.append(f"VERITY_QWORD_MAX_GATES={build_max_gates} in the Build's environment (allowed for this row: {allowed_max_gates or None})")
    calls, units, gates = (sum(int(x[i]) for x in lines) for i in range(3))
    return {"row": row, "check": "fast", "query_of_record": QUERY_OF_RECORD, "query_id": q.get("query_id"), "partition": bool(q.get("partition")),
            "manifest_digest": man.get("manifest_digest"), "word_lines": len(lines), "components": need, "calls": calls, "units": units,
            "gates": gates, "env": {"VERITY_WORD_CHECK": build_word_check or None, "VERITY_QWORD_MAX_GATES": build_max_gates or None},
            "problems": problems, "ok": not problems}


def _partition_digests(part: Any) -> list[str]:
    """Every partition digest a manifest's `query.partition` names: one `{object, digest}`, or `by_component` / `by_rank` of them."""
    if not isinstance(part, dict):
        return []
    if "object" in part:
        return [part["digest"]]
    return sorted(dg for k in ("by_component", "by_rank") for v in (part.get(k) or {}).values() for dg in _partition_digests(v))


def build_programs(d: Path) -> dict[str, dict]:
    """Per rank, the Program digests of a row's Build (`build_summary.json`: a TP row's per rank under `ranks`, `row_records.tp_build_summary`)."""
    s = _doc(d / "build_summary.json")
    per_rank = {k.removeprefix("rank"): v for k, v in (s.get("ranks") or {}).items()} if isinstance(s.get("ranks"), dict) else {"0": s} if s else {}
    return {r: {"step": (v.get("step") or v.get("build_step") or {}).get("program_digest"), "workload": (v.get("workload") or {}).get("workload_digest"),
                "requests": {k: x.get("program_digest") for k, x in sorted(v.items()) if (k == "request" or k.startswith("build_request")) and isinstance(x, dict)}}
            for r, v in sorted(per_rank.items(), key=lambda kv: int(kv[0]))}


def row_digests(d: Path) -> dict[str, Any]:
    """One row's digests from its run directory."""
    ranks = build_programs(d)
    man, vd = _doc(d / "manifest.json"), _doc(d / "verdict.json")
    q = man.get("query") or {}
    return {"row": d.name, "programs": ranks, "manifest_digest": man.get("manifest_digest"), "program_digest": man.get("program_digest"),
            "query_id": q.get("query_id"), "partition_digests": _partition_digests(q.get("partition")),
            "run_roots": list(vd.get("run_roots") or []), "verdict": vd.get("outcome")}


def digests_markdown(rows: list[dict[str, Any]]) -> str:
    short = lambda x: (x or "-")[:16]                                        # noqa: E731
    md = ["| row | verdict | step (rank 0) | workload (rank 0) | requests | manifest | partitions | run roots |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p0 = r["programs"].get("0") or {}
        md.append(f"| {r['row']} | {r['verdict'] or '-'} | {short(p0.get('step'))} | {short(p0.get('workload'))} | {len(p0.get('requests') or {})} "
                  f"| {short(r['manifest_digest'])} | {len(r['partition_digests'])} | {', '.join(x[:16] for x in r['run_roots']) or '-'} |")
    return "\n".join(md) + "\n"


def store_meta(row_dir: str | Path, *, kind: str, role: str, rownum: int, run_id: str, epoch_sha: str, gpu: str, pod: str, epoch: str) -> dict[str, Any]:
    """A stored tree's meta (serving-view's record-build layout), from the Build's own summaries."""
    d = Path(row_dir)
    progs, wds = {}, {}
    for r, p in build_programs(d).items():
        progs.update({f"rank{r}/{k}": dg for k, dg in [("step", p["step"]), *p["requests"].items()] if dg})
        if p["workload"]:
            wds[f"rank{r}"] = p["workload"]
    m, v = _doc(d / "manifest.json"), _doc(d / "verdict.json")
    return {"name": f"epoch-{kind}/{d.name}", "role": role, "row": rownum, "row_id": d.name, "epoch": epoch, "program_digests": progs,
            "workload_digests": wds, "manifest_digest": m.get("manifest_digest"), "query_id": (m.get("query") or {}).get("query_id"),
            "verdict": v.get("outcome"), "build": {"gpu": gpu or None, "machine": pod or None, "run": run_id, "source": epoch_sha, "stage": kind}}


def custody_exports(custody_dir: str | Path) -> list[str]:
    """`export` lines of the run's custody key (cred.json beside its store.toml), for the store's `research data` calls."""
    cred = _doc(Path(custody_dir) / "cred.json")
    return [f"export {k}={shlex.quote(cred[k])}" for k in KEY_VARS if cred.get(k)]


def device(expect_sms: int | None, expect_cc: str) -> tuple[dict[str, Any], bool]:
    """The CUDA device once torch initialises CUDA and vLLM imports (raises until they do), and whether it is the expected one."""
    import torch
    torch.cuda.init()
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False after init")
    import vllm  # noqa: F401 -- the fail-fast waits for vLLM to import
    p = torch.cuda.get_device_properties(0)
    dev = {"name": p.name, "sms": p.multi_processor_count, "cc": f"{p.major}.{p.minor}", "count": torch.cuda.device_count()}
    return dev, expect_sms is None or (dev["sms"] == expect_sms and dev["cc"] == expect_cc)


# ------------------------------------------------------------------------------------------------------------------------------ the CLI
@dataclass(frozen=True, kw_only=True)
class JobOptions:
    """The run's timeout end, the job's end and the custody key's expiry."""

    run_dir: str = option("run_dir")
    custody_dir: str = option("--custody-dir")
    store_reserve_s: int = option("--store-reserve-s", type=int, default=2400)
    job_deadline: str | None = option("--job-deadline", default=None)
    cmd: str = field(default="job", init=False)


@dataclass(frozen=True, kw_only=True)
class ResolveOptions:
    """A row's checkpoint role, repo, revision and TP world."""

    row: str = option("row")
    cmd: str = field(default="resolve", init=False)


@dataclass(frozen=True, kw_only=True)
class RowStateOptions:
    """A row dir's required families, Commit outcome and manifest-verify result."""

    row_dir: str = option("row_dir")
    header: str | None = option("--header", default=None, help="write the manifest's header here")
    cmd: str = field(default="row-state", init=False)


@dataclass(frozen=True, kw_only=True)
class WordCheckOptions:
    """The strict word check before a row's Commit, from the Build's own manifest step."""

    row: str = option("row")
    row_dir: str = option("row_dir")
    out: str = option("--out")
    world: int = option("--world", type=int, default=1)
    build_word_check: str = option("--build-word-check", default="", help="the job's VERITY_WORD_CHECK ('' = unset)")
    build_max_gates: str = option("--build-max-gates", default="", help="the job's VERITY_QWORD_MAX_GATES ('' = unset)")
    allowed_max_gates: str = option("--allowed-max-gates", default="", help="the one VERITY_QWORD_MAX_GATES a GO allows for the row")
    cmd: str = field(default="word-check", init=False)


@dataclass(frozen=True, kw_only=True)
class DigestsOptions:
    """Row digests (verity-vllm/rebaseline-digests/v1)."""

    row_dirs: list[str] = option("row_dirs", nargs="+")
    out: str | None = option("--out", default=None)
    markdown: str | None = option("--markdown", default=None)
    cmd: str = field(default="digests", init=False)


@dataclass(frozen=True, kw_only=True)
class StoreMetaOptions:
    """A stored tree's meta document."""

    row_dir: str = option("row_dir")
    kind: str = option("--kind")
    role: str = option("--role")
    rownum: int = option("--rownum", type=int)
    run_id: str = option("--run-id")
    epoch_sha: str = option("--epoch-sha")
    gpu: str = option("--gpu", default="")
    pod: str = option("--pod", default="")
    epoch: str = option("--epoch", default="vllm-rebaseline-epoch")
    cmd: str = field(default="store-meta", init=False)


@dataclass(frozen=True, kw_only=True)
class CustodyEnvOptions:
    """Export lines of the run's custody key."""

    custody_dir: str = option("custody_dir")
    cmd: str = field(default="custody-env", init=False)


@dataclass(frozen=True, kw_only=True)
class DeviceOptions:
    """The CUDA device, once torch and vLLM load."""

    expect_sms: int | None = option("--expect-sms", type=int, default=None)
    expect_cc: str = option("--expect-cc", default="8.9")
    cmd: str = field(default="device", init=False)


COMMANDS = {"job": JobOptions, "resolve": ResolveOptions, "row-state": RowStateOptions, "word-check": WordCheckOptions,
            "digests": DigestsOptions, "store-meta": StoreMetaOptions, "custody-env": CustodyEnvOptions, "device": DeviceOptions}


def main(a: Any) -> int:
    dash = lambda x: "-" if x is None or x == "" else str(x)                 # noqa: E731
    if a.cmd == "job":
        print(" ".join(map(dash, job(a.run_dir, a.custody_dir, a.store_reserve_s, a.job_deadline))))
    elif a.cmd == "resolve":
        print(" ".join(map(str, resolve(a.row))))
    elif a.cmd == "row-state":
        fams, outcome, mv = row_state(a.row_dir, a.header)
        print(dash(",".join(fams)), dash(outcome), dash(mv))
    elif a.cmd == "word-check":
        rep = word_check(a.row, a.row_dir, a.world, a.build_word_check, a.build_max_gates, a.allowed_max_gates)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(("PASS" if rep["ok"] else "FAIL") + f" fast: {rep['word_lines']}/{rep['components']} word lines, {rep['calls']} calls -> "
              f"{rep['units']} units over {rep['gates']} gates, manifest {str(rep['manifest_digest'])[:16]}"
              + ("" if rep["ok"] else ": " + "; ".join(rep["problems"])))
        return 0 if rep["ok"] else 1
    elif a.cmd == "digests":
        rows = [row_digests(Path(d)) for d in a.row_dirs]
        text = json.dumps({"schema": "verity-vllm/rebaseline-digests/v1", "rows": rows}, indent=1) + "\n"
        if a.out:
            Path(a.out).write_text(text)
        md = digests_markdown(rows)
        if a.markdown:
            Path(a.markdown).write_text(md)
        print(md if a.out else text, end="")
        return 0 if all(r["programs"] and r["manifest_digest"] for r in rows) else 1
    elif a.cmd == "store-meta":
        print(json.dumps(store_meta(a.row_dir, kind=a.kind, role=a.role, rownum=a.rownum, run_id=a.run_id, epoch_sha=a.epoch_sha,
                                    gpu=a.gpu, pod=a.pod, epoch=a.epoch)))
    elif a.cmd == "custody-env":
        print("\n".join(custody_exports(a.custody_dir)))
    elif a.cmd == "device":
        dev, ok = device(a.expect_sms, a.expect_cc)
        print(json.dumps(dev))
        return 0 if ok else 3
    return 0
