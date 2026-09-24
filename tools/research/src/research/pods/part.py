"""Reference-part gate: is the provisioned board the datasheet part of its family, or a same-named variant?

A provider "RTX 4090" reporting 48 GiB is a re-memoried AD102 board, not the part Table 2's native peak names
(`tables.device_variant` rejects its results).  `~/.research/machines.toml` carries the expected VRAM per `hw.family`:

    [families.rtx4090]
    vram_mb = 24564          # what nvidia-smi memory.total reports on the reference board (hw.vram_mb of the store's attempts)

    research pods check-part <machine|pod-id>     # name, VRAM, host CPU, load, verdict `reference` / `variant (+N %)`
    research pods create ... [--require-reference-part]   # the same verdict once sshd answers; non-zero on a variant

The tolerance (12 %) is `tables.device_variant`'s: driver-reported totals differ from the datasheet by a few percent, never by half.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .. import remote
from ..env import sku_family

TOLERANCE = 0.12
PROBE = ("sh", "-c", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; echo @@; nproc; echo @@; "
         "grep -m1 'model name' /proc/cpuinfo; echo @@; cat /proc/loadavg; echo @@; hostname")


def load_families(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or remote.machines_path()
    return tomllib.loads(p.read_text()).get("families", {}) if p.exists() else {}


def verdict(sku: str | None, vram_mb: int | None, families: dict[str, dict[str, Any]], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """`reference` when memory.total is within `tolerance` of the family's vram_mb; `variant (+N %)` otherwise; `unknown` without a fact."""
    fam = sku_family(sku, 1 if sku else None)
    ref = (families.get(fam) or {}).get("vram_mb") if fam else None
    out = {"sku": sku, "vram_mb": vram_mb, "family": fam, "reference_vram_mb": ref, "reference": None, "verdict": "unknown"}
    if ref and vram_mb:
        pct = (vram_mb / ref - 1.0) * 100.0
        out["reference"] = abs(pct) <= tolerance * 100.0
        out["verdict"] = "reference" if out["reference"] else f"variant ({pct:+.0f} %)"
    elif fam and not ref:
        out["verdict"] = f"unknown (no [families.{fam}] vram_mb in {remote.machines_path()})"
    return out


def parse_probe(text: str) -> dict[str, Any]:
    gpu, nproc, model, loadavg, host = ([s.strip() for s in text.split("@@")] + [""] * 5)[:5]
    rows = [[c.strip() for c in line.split(",")] for line in gpu.splitlines() if line.strip()]
    sku = rows[0][0] if rows and rows[0][0] else None
    try:
        vram = int(float(rows[0][1])) if rows and len(rows[0]) > 1 else None
    except ValueError:
        vram = None
    cpu = (model.split(":", 1)[1].strip() if ":" in model else model) or None
    load = loadavg.split()[0] if loadavg.split() else None
    return {"sku": sku, "count": len(rows), "vram_mb": vram, "cpu": f"{cpu} x{nproc}" if cpu and nproc else cpu, "hostname": host or None,
            "load": float(load) if load else None}


def check_pod(pod_id: str, families: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Probe a RUNNING pod over ssh (read-only) and judge it.  Raises `remote.Refused` when the pod is not reachable."""
    from . import runpod

    try:
        pod = runpod.get(pod_id)
    except runpod.PodError as e:                     # an HTTP error (404: terminated), no API config, no network
        raise remote.Refused(f"pod {pod_id}: {e}") from None
    tgt = runpod.ssh_target(pod) if pod and (pod.get("desiredStatus") or "").upper() == "RUNNING" else None
    if tgt is None:
        raise remote.Refused(f"pod {pod_id} is {pod.get('desiredStatus') if pod else 'not found'} or has no ssh port yet")
    r = remote.ssh_run(remote.ssh_argv(*tgt), list(PROBE), timeout=60)
    if r.returncode != 0:
        raise remote.Refused(f"probe failed on {pod_id}: {r.stderr.decode(errors='replace')[-300:]}")
    facts = parse_probe(r.stdout.decode(errors="replace"))
    fams = load_families() if families is None else families
    return {"pod_id": pod_id, "pod_name": pod.get("name"), "cost_per_hr": pod.get("costPerHr"), **facts,
            **{k: v for k, v in verdict(facts["sku"], facts["vram_mb"], fams).items() if k not in facts}}


def check_machine(name: str) -> dict[str, Any]:
    """`name` is a machines.toml entry or a raw pod id."""
    m = remote.load_machines().get(name)
    return {"machine": name if m else None, **check_pod(m["pod_id"] if m else name)}


def render(rep: dict[str, Any]) -> str:
    return (f"{rep.get('machine') or rep['pod_id']}  {rep['sku']} x{rep['count']}  {rep['vram_mb']} MiB (reference {rep['reference_vram_mb']} MiB)"
            f"  host {rep['cpu']}  load {rep['load']}  {rep['hostname']}  ->  {rep['verdict']}")


def main(argv: list[str]) -> int:
    """`research pods check-part <machine|pod-id> [--json]` | `research pods create <runpod create flags> [--require-reference-part]`
    | `research pods ssh|sync ...` (connect.py) | `research pods health <pod>` (health.py) | any other verb goes to
    `python -m research.pods.runpod` unchanged.  Exit 3 = not the reference part (check-part, create) or DEGRADED (health)."""
    import argparse
    import sys

    from . import runpod

    if argv[:1] == ["ssh"]:
        from .connect import main_ssh

        return main_ssh(argv[1:])
    if argv[:1] == ["sync"]:
        from .connect import main_sync

        return main_sync(argv[1:])
    if argv[:1] == ["check-part"]:
        ap = argparse.ArgumentParser(prog="research pods check-part")
        ap.add_argument("target", metavar="MACHINE|POD_ID")
        ap.add_argument("--json", action="store_true")
        ns = ap.parse_args(argv[1:])
        try:
            rep = check_machine(ns.target)
        except remote.Refused as e:
            print(f"research pods check-part: {e}", file=sys.stderr)
            return 1
        print(json.dumps(rep, indent=2, sort_keys=True) if ns.json else render(rep))
        return 0 if rep.get("reference") else 3
    if argv[:1] == ["create"]:
        ap = argparse.ArgumentParser(prog="research pods create")
        runpod.create_args(ap)
        ap.add_argument("--require-reference-part", action="store_true", help="exit 3 (pod left running) unless the board is the reference part")
        ns = ap.parse_args(argv[1:])
        pod = runpod.create(ns)
        runpod.wait_ssh(pod["id"])
        rep = check_pod(pod["id"])
        print(render(rep))
        if ns.require_reference_part and not rep.get("reference"):
            print(f"research pods create: not the reference part; `research pods terminate {pod['id']}` releases it", file=sys.stderr)
            return 3
        return 0
    if argv[:1] in (["unpreserved"], ["drain"]) or (argv[:1] == ["terminate"] and "--require-preserved" in argv):
        from .drain import main as drain_main                 # teardown gated on `preserved` (drain.py)

        return drain_main(argv)
    if argv[:1] == ["guard"]:
        from .guard import main as guard_main                 # the fleet budget guard (guard.py)

        return guard_main(argv[1:])
    if argv[:1] == ["health"]:
        from .health import main as health_main               # fit to measure on? quota, GPU state, microbenchmark (health.py)

        return health_main(argv[1:])
    sys.argv = ["research pods", *argv]
    runpod.main()
    return 0
