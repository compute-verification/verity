"""Target-family preflight.

A workload id names its DEVICE CLASS in its third token (`<model>__<dtype>__<device>__tp<k>__...`, e.g. `__l40s__`).  The
workload's Build target family, the pod device the GPU stages run on, and the Build that comes out (build_summary.json
step.target) must all be that class: otherwise a Hopper-target Program (cc 9.0, FA3, Gemm_v2) built on an H100 host for an
`__l40s__` workload has its Match compare an L40S capture against it.  The check is a PREFLIGHT: it runs before Build and
before any engine, and fails by name `PRECHECK_FAIL_TARGET_FAMILY`.

Family = compute capability (major, minor), the same reading the finalizer applies (l40s 8.9 / a100 8.0 / h100 9.0 /
b200 10.0).  Three facts are compared with the workload's family:
  declared  -- the manifest's `target.compute_capability` when the workload declares one, else the id's device class when its SM
               count is fixed (`declared_profile`; Build gets it as --target, and a Build-only stage then needs no device)
  device    -- the pod GPU's compute capability (nvidia-smi compute_cap; torch as fallback) when a GPU stage runs here or the Build
               target is the device (no declared target); a device-target Build under an empty CUDA_VISIBLE_DEVICES (a leased
               workload, `pipeline.gpu_lease`) is refused, since it would see no device and derive for the default target.  The
               Match's CPU phase (`--match-phase cpu`: the check, the derives and the fold over GPU stages already captured) is no
               GPU stage
  built     -- build_summary.json step.target.compute_capability, checked again AFTER Build (--build-summary)
A WAIVER (`--waiver <reason>`, row_pod.sh: $TARGET_FAMILY_WAIVER) turns a would-be failure into `WAIVED <reason>` -- recorded in
stages.txt and in the record, never silent; the release canary sets it because its `__l40s__` base workloads also run on the Hopper
canary host.  A workload of record never carries a waiver (the finalizer reads the line).

Torch-free; stdlib only.  Record schema: verity-sweep/target-family/v1.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from verity_vllm import settings
from verity_vllm.config import TARGET_NUM_SMS, TARGETS, RowEngine, option, workload_arg

FAIL_CLASS = "PRECHECK_FAIL_TARGET_FAMILY"
SCHEMA = "verity-sweep/target-family/v1"

# device token in a row id -> compute capability family: the target table of `config.py`
DEVICE_FAMILY: dict[str, tuple[int, int]] = TARGETS

_ROW_RE = re.compile(r"^[^_]+(?:_[^_]+)*__[^_]+__([a-z0-9]+)__tp\d+__")


def device_token_of_row(row_id: str) -> str | None:
    """`smollm2-135m__bf16__l40s__tp1__b1__...` -> `l40s`; None when the id does not carry the grammar."""
    parts = row_id.split("__")
    if len(parts) >= 4 and parts[3].startswith("tp"):
        return parts[2]
    m = _ROW_RE.match(row_id)
    return m.group(1) if m else None


def row_family(row_id: str) -> tuple[str | None, tuple[int, int] | None]:
    tok = device_token_of_row(row_id)
    return tok, DEVICE_FAMILY.get(tok) if tok else None


def _cc(v: Any) -> tuple[int, int] | None:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            a, b = v.strip().split(".")
            return int(a), int(b)
        except ValueError:
            return None
    try:
        a, b = v
        return int(a), int(b)
    except (TypeError, ValueError):
        return None


def declared_profile(row_id: str | None, workload: dict) -> dict:
    """The Build's declared TargetProfile block: the workload's `engine.target`, laid over the row's device-class profile (capability +
    `config.TARGET_NUM_SMS`) when the workload names no capability and the class's SM count is fixed; {} when neither declares one."""
    t = dict(RowEngine.of(workload).target or {})
    tok, fam = row_family(row_id) if row_id else (None, None)
    if _cc(t.get("compute_capability")) is None and fam and tok in TARGET_NUM_SMS:
        t.pop("compute_capability", None)
        t = {"compute_capability": list(fam), "num_sms": TARGET_NUM_SMS[tok], **t}
    return t


def declared_target(workload: dict, row_id: str | None = None) -> tuple[int, int] | None:
    return _cc(declared_profile(row_id, workload).get("compute_capability"))


def device_cc() -> tuple[tuple[int, int] | None, str]:
    """(cc, source) of the GPU this process would use; (None, reason) when no device is visible."""
    vis = settings.CUDA_VISIBLE_DEVICES
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        rows = [r.strip() for r in out.stdout.splitlines() if r.strip()]
        if out.returncode == 0 and rows:
            if vis not in (None, ""):
                want = vis.split(",")[0].strip()
                rows = [r for r in rows if r.split(",")[0].strip() == want] or rows
            cc = _cc(rows[0].split(",")[1])
            if cc:
                return cc, "nvidia-smi compute_cap"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch  # noqa: WPS433 -- fallback only

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0)), "torch.cuda.get_device_capability"
        return None, "torch.cuda.is_available() False"
    except Exception as e:  # noqa: BLE001
        return None, f"no device: {type(e).__name__}"


def built_target(build_summary: dict) -> tuple[int, int] | None:
    step = build_summary.get("step") or {}
    t = step.get("target") or {}
    cc = _cc(t.get("compute_capability"))
    if cc is None:
        app = step.get("applicability") or {}
        fam = (app.get("tensor_core_family") or app.get("gemm_target") or "").lower()
        if fam == "hopper":
            return (9, 0)
    return cc


def fmt(cc: tuple[int, int] | None) -> str:
    return "?" if cc is None else f"{cc[0]}.{cc[1]}"


def check(row_id: str, workload: dict | None, stages: list[str], *, want_device: bool = True, build_summary: dict | None = None,
          waiver: str | None = None, devices: str | None = None, match_phase: str = "all") -> dict:
    tok, fam = row_family(row_id)
    rec: dict[str, Any] = {"schema": SCHEMA, "row": row_id, "device_token": tok, "row_family": list(fam) if fam else None,
                           "stages": stages, "match_phase": match_phase, "declared": None, "device": None, "device_source": None,
                           "built": None, "mismatches": [], "ok": False, "fail_class": None, "waiver": waiver or None, "verdict": None}
    if fam is None:
        rec["mismatches"].append({"fact": "row", "value": tok, "why": f"unknown device class {tok!r} in the row id (known: {sorted(DEVICE_FAMILY)})"})
    if workload is not None:
        d = declared_target(workload, row_id)
        rec["declared"] = list(d) if d else None
        if d and fam and d != fam:
            rec["mismatches"].append({"fact": "declared", "value": fmt(d), "why": f"manifest target.compute_capability {fmt(d)} != row family {fmt(fam)}"})
    gpu_stage = any(s in ("commit", "bench") or (s == "match" and match_phase != "cpu") for s in stages)
    build_on_device = "build" in stages and (workload is None or declared_target(workload, row_id) is None)
    if want_device and build_on_device and devices == "":
        rec["mismatches"].append({"fact": "device", "value": None, "why": "the Build's target is this host's GPU, which CUDA_VISIBLE_DEVICES "
                                  "hides from it (a leased row): declare engine.target's compute_capability and num_sms, or the device "
                                  "class's SM count in config.TARGET_NUM_SMS"})
        build_on_device = False
    if want_device and (gpu_stage or build_on_device):
        dc, src = device_cc()
        rec["device"] = list(dc) if dc else None
        rec["device_source"] = src
        if dc is None:
            rec["mismatches"].append({"fact": "device", "value": None, "why": f"no GPU visible for a GPU stage / device-target Build ({src})"})
        elif fam and dc != fam:
            rec["mismatches"].append({"fact": "device", "value": fmt(dc), "why": f"pod device {fmt(dc)} != row family {fmt(fam)} (stages {','.join(stages)})"})
    if build_summary is not None:
        b = built_target(build_summary)
        rec["built"] = list(b) if b else None
        if b is None:
            rec["mismatches"].append({"fact": "built", "value": None, "why": "build_summary.json step.target.compute_capability absent"})
        elif fam and b != fam:
            rec["mismatches"].append({"fact": "built", "value": fmt(b), "why": f"Build target {fmt(b)} != row family {fmt(fam)}"})
    if not rec["mismatches"]:
        rec["ok"] = True
        rec["verdict"] = "PASS"
    elif waiver:
        rec["ok"] = True
        rec["verdict"] = "WAIVED"
        rec["fail_class"] = None
        rec["would_fail_class"] = FAIL_CLASS
    else:
        rec["verdict"] = "FAIL"
        rec["fail_class"] = FAIL_CLASS
    return rec


def summary_line(rec: dict) -> str:
    fam = fmt(tuple(rec["row_family"]) if rec["row_family"] else None)
    facts = f"row={rec['device_token']}({fam}) declared={fmt(tuple(rec['declared']) if rec['declared'] else None)} " \
            f"device={fmt(tuple(rec['device']) if rec['device'] else None)} built={fmt(tuple(rec['built']) if rec['built'] else None)}"
    if rec["verdict"] == "PASS":
        return f"TARGET-FAMILY-OK {facts}"
    why = "; ".join(m["why"] for m in rec["mismatches"])
    if rec["verdict"] == "WAIVED":
        return f"TARGET-FAMILY-WAIVED ({rec['waiver']}) {facts} -- {why}"
    return f"{FAIL_CLASS}: {why} [{facts}]"


@dataclass(frozen=True, kw_only=True)
class Options:
    row: str = option("--row", help="row id (device class in its third token)")
    workload: str | None = option("--workload", type=workload_arg, default=None, help="workload manifest (declared target block)")
    stages: str = option("--stages", default="build,match,commit")
    build_summary: str | None = option("--build-summary", default=None,
                                       help="check the Build that came out (step.target) as well")
    no_device: bool = option("--no-device", default=False, action="store_true",
                             help="do not read the pod GPU (CPU-only host doing a declared-target Build)")
    waiver: str | None = option("--waiver", env="TARGET_FAMILY_WAIVER", default=None,
                                help="named waiver (canary); recorded, never silent")
    devices: str | None = option("--devices", env="CUDA_VISIBLE_DEVICES", env_empty=True, default=None,
                                 help="the GPUs the workload's stages see (empty under a GPU lease: a Build then has no device to target)")
    match_phase: str = option("--match-phase", default="all", choices=("all", "gpu", "cpu"),
                              help="the Match's phase (`verity-vllm row --match-phase`): cpu runs no GPU stage, so its match reads no device")
    out: str | None = option("--out", default=None, help="write the record here")


def main(a: Options) -> int:
    wl = json.load(open(a.workload)) if a.workload and os.path.exists(a.workload) else None
    bs = json.load(open(a.build_summary)) if a.build_summary and os.path.exists(a.build_summary) else None
    stages = [s for s in a.stages.replace(" ", "").split(",") if s]
    rec = check(a.row, wl, stages, want_device=not a.no_device, build_summary=bs, waiver=a.waiver, devices=a.devices,
                match_phase=a.match_phase)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, a.out)
    line = summary_line(rec)
    if rec["verdict"] == "FAIL":
        print(line, file=sys.stderr)
        return 3
    print(line)
    return 0
