"""The Nebius server's bootstrap check: is this the machine we pay for, and is its weight cache fast enough?

    research run --on vy-nebius-1 --project verity --source . --cwd source -- \
        python3 tools/research/src/research/pods/nebius/check.py

Checks, each PASS or FAIL on one line (exit 1 on any FAIL; with RESEARCH_RUN_DIR, result.json carries every number):
    driver       NVIDIA driver >= 575 (the image's cuda13.0 preset ships 580.x)
    gpus         8 GPUs, each compute capability 12.0 with 188 SMs (the CUDA driver API, through ctypes: no torch)
    ram          MemTotal >= 97% of 1,744 GiB (the guest kernel keeps a little)
    vcpu         192 online CPUs
    workspace    /workspace is the data disk (its own mount, >= 4.8 TiB)
    disk_read    sequential direct-I/O read of a file fio lays out on /workspace >= 1,500 MiB/s (the disk type's bound is 2 GiB/s)
    lease        vy-lease.service is active, so the VM stops itself when its lease runs out
    gpu_lease    gpu-lease answers `status`
Stdlib only.  VY_CHECK_SKIP_FIO=1 skips the disk read (a quick re-check).
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

WANT = {"driver_major": 575, "gpus": 8, "cc": (12, 0), "sms": 188, "ram_gib": 1744, "ram_fraction": 0.97, "vcpu": 192,
        "workspace_tib": 4.8, "read_mib_s": 1500}
CU_ATTR_SM_COUNT, CU_ATTR_CC_MAJOR, CU_ATTR_CC_MINOR = 16, 75, 76

results: list[dict] = []


def check(name: str, ok: bool, value, want: str, unit: str = "") -> None:
    results.append({"name": name, "ok": bool(ok), "value": value, "want": want, "unit": unit})
    print(f"{'PASS' if ok else 'FAIL'} {name:<10} {value}{(' ' + unit) if unit else ''} (want {want})", flush=True)


def run(argv: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def driver() -> None:
    try:
        v = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).stdout.split()
    except OSError as e:
        return check("driver", False, f"no nvidia-smi ({e})", f">= {WANT['driver_major']}")
    major = int(v[0].split(".")[0]) if v else 0
    check("driver", major >= WANT["driver_major"], v[0] if v else "none", f">= {WANT['driver_major']}")


def gpus() -> None:
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return check("gpus", False, f"no libcuda ({e})", f"{WANT['gpus']} x cc 12.0, 188 SMs")
    n = ctypes.c_int()
    if cu.cuInit(0) != 0 or cu.cuDeviceGetCount(ctypes.byref(n)) != 0:
        return check("gpus", False, "cuInit failed", f"{WANT['gpus']} GPUs")
    seen = []
    for i in range(n.value):
        dev, val = ctypes.c_int(), ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(dev), i)
        attrs = []
        for a in (CU_ATTR_CC_MAJOR, CU_ATTR_CC_MINOR, CU_ATTR_SM_COUNT):
            cu.cuDeviceGetAttribute(ctypes.byref(val), a, dev)
            attrs.append(val.value)
        seen.append({"index": i, "cc": f"{attrs[0]}.{attrs[1]}", "sms": attrs[2]})
    good = [g for g in seen if g["cc"] == "%d.%d" % WANT["cc"] and g["sms"] == WANT["sms"]]
    check("gpus", len(seen) == WANT["gpus"] and len(good) == len(seen),
          f"{len(seen)} GPUs: " + ", ".join(sorted({f"cc {g['cc']} / {g['sms']} SMs" for g in seen})), f"{WANT['gpus']} x cc 12.0 / 188 SMs")


def ram() -> None:
    kib = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    gib = kib / 2**20
    check("ram", gib >= WANT["ram_gib"] * WANT["ram_fraction"], round(gib, 1), f">= {WANT['ram_gib'] * WANT['ram_fraction']:.0f}", "GiB")


def vcpu() -> None:
    n = os.cpu_count() or 0
    check("vcpu", n == WANT["vcpu"], n, str(WANT["vcpu"]))


def workspace() -> None:
    mounted = os.path.ismount("/workspace")
    tib = shutil.disk_usage("/workspace").total / 2**40 if mounted else 0.0
    check("workspace", mounted and tib >= WANT["workspace_tib"], f"{'mounted' if mounted else 'not a mount'}, {tib:.2f} TiB",
          f"its own mount, >= {WANT['workspace_tib']} TiB")


def disk_read() -> None:
    if os.environ.get("VY_CHECK_SKIP_FIO") == "1":
        return
    if not shutil.which("fio"):
        return check("disk_read", False, "no fio", f">= {WANT['read_mib_s']}", "MiB/s")
    f = Path("/workspace/cache/.vy-check-fio")   # /workspace itself is root's; runs are the research user
    try:
        r = run(["fio", "--name=seqread", f"--filename={f}", "--size=32G", "--rw=read", "--bs=1M", "--direct=1", "--ioengine=libaio",
                 "--iodepth=32", "--numjobs=1", "--runtime=30", "--time_based", "--output-format=json"], timeout=1800)
        mib_s = json.loads(r.stdout)["jobs"][0]["read"]["bw"] / 1024 if r.returncode == 0 else 0.0
    except (ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as e:
        return check("disk_read", False, f"fio failed ({type(e).__name__})", f">= {WANT['read_mib_s']}", "MiB/s")
    finally:
        f.unlink(missing_ok=True)
    check("disk_read", mib_s >= WANT["read_mib_s"], round(mib_s), f">= {WANT['read_mib_s']}", "MiB/s")


def lease() -> None:
    active = run(["systemctl", "is-active", "vy-lease.service"]).stdout.strip()
    line = ""
    d = os.environ.get("LEASE_DIR") or str(Path.home() / ".research")
    if Path(d, "lease").is_file():
        line = Path(d, "lease").read_text().strip()
    check("lease", active == "active", f"{active}; {line or 'no lease file'}", "active")


def gpu_lease() -> None:
    try:
        r = run(["gpu-lease", "status"])
        ok, value = r.returncode == 0, f"{len(r.stdout.splitlines())} GPUs listed"
    except OSError as e:
        ok, value = False, f"not installed ({e})"
    check("gpu_lease", ok, value, "installed")


def main() -> int:
    for f in (driver, gpus, ram, vcpu, workspace, disk_read, lease, gpu_lease):
        f()
    failed = [r["name"] for r in results if not r["ok"]]
    if rd := os.environ.get("RESEARCH_RUN_DIR"):
        doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
               "workload_fingerprint": {"check": "nebius-bootstrap/v1", "host": socket.gethostname(), "want": {k: str(v) for k, v in WANT.items()}},
               "validation": {"status": "failed" if failed else "passed", "checker": "nebius-bootstrap/v1",
                              "detail": "failed: " + ", ".join(failed) if failed else "all checks passed"},
               "measurements": [{"name": r["name"], "value": r["value"], "unit": r["unit"] or "-"} for r in results]}
        tmp = Path(rd, "result.json.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, Path(rd, "result.json"))
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
