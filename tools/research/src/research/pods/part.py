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
from typing import Any, Callable

from .. import remote
from ..env import sku_family

TOLERANCE = 0.12
PROBE = ("sh", "-c", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits; echo @@; nproc; echo @@; "
         "grep -m1 'model name' /proc/cpuinfo; echo @@; cat /proc/loadavg; echo @@; hostname")


def load_families(path: Path | None = None) -> dict[str, dict[str, Any]]:
    if path is None:
        from .registry import families

        return families(remote.machines_path())
    return tomllib.loads(path.read_text()).get("families", {}) if path.exists() else {}


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


#: RunPod's replies that mean "no machine of that spec right now": the only create failure a caller may answer with another spec.
NO_STOCK = ("no instances currently available", "no longer any instances available")
EXIT_EXISTS, EXIT_NO_STOCK, EXIT_FAILED, EXIT_UNBUDGETED = 4, 5, 2, 6


def _named(name: str) -> list[dict]:
    """Live (not terminated) pods whose RunPod name is ``<name>-veritor-campaign`` (`runpod.create`'s naming)."""
    from . import runpod

    full = f"{name}-veritor-campaign"
    return [p for p in (runpod._request("GET", "/pods") or []) if p.get("name") == full and p.get("desiredStatus") != "TERMINATED"]


def create_once(ns: Any) -> tuple[dict | None, int]:
    """At most one pod per --name, whatever the create call says (retry loops made duplicates three times, 2026-09-26; red-team-
    flock-2's pod-create.sh): refuse when a live pod of the name exists (exit 4, its id printed; ``--allow-duplicate`` overrides);
    after a failed create, a pod that appeared under the name anyway is the pod; only an explicit no-stock reply is exit 5 (try
    another spec), any other failure exit 2 (do not retry blindly).  Returns (pod, 0) or (None, exit code)."""
    import sys

    from . import runpod

    before = {p["id"] for p in _named(ns.name)}
    if before and not getattr(ns, "allow_duplicate", False):
        print(f"research pods create: {ns.name} already has live pod(s) {', '.join(sorted(before))}; not creating another "
              f"(register or terminate it, or --allow-duplicate)", file=sys.stderr)
        return None, EXIT_EXISTS
    try:
        return runpod.create(ns), 0
    except runpod.PodError as e:
        appeared = [p for p in _named(ns.name) if p["id"] not in before]
        if appeared:
            print(f"research pods create: the create call failed ({e}) but pod {appeared[0]['id']} exists under {ns.name}; using it",
                  file=sys.stderr)
            return appeared[0], 0
        if any(x in str(e).lower() for x in NO_STOCK):
            print(f"research pods create: NO-STOCK {e}", file=sys.stderr)
            return None, EXIT_NO_STOCK
        print(f"research pods create: FAILED (no pod under {ns.name}): {e}", file=sys.stderr)
        return None, EXIT_FAILED


def create_leased(ns: Any, *, gate: Callable[[str, float], tuple[str | None, str | None]] | None = None,
                  wait_ssh: Callable[[str], tuple[str, int]] | None = None, arm: Callable[..., dict] | None = None,
                  terminate: Callable[[str], None] | None = None, clock: Callable[[], float] | None = None) -> tuple[dict | None, int]:
    """`research pods create`'s lifecycle: the budget gate, the create call with the lease in the pod's boot (research.pods.lease),
    then, once sshd answers, the lease and the idle guard armed over ssh.  A pod that cannot be armed is terminated: no pod leaves
    here unbounded.  Returns (pod with its `lease` report, 0) or (None, exit code)."""
    import sys
    import time

    from . import budgets, lease, runpod

    name = f"{ns.name}-veritor-campaign"
    why, line = (gate or budgets.check_create)(name, ns.max_hours)
    if why:
        print(f"research pods create: REFUSED {ns.name}: {why}", file=sys.stderr)
        return None, EXIT_UNBUDGETED
    print(f"research pods create: {ns.name} is under the budget line {line!r}", file=sys.stderr)
    ns.lease_expires = int((clock or time.time)() + ns.max_hours * 3600)
    pod, rc = create_once(ns)
    if pod is None:
        return None, rc
    kill = terminate or runpod.terminate
    try:
        host, port = (wait_ssh or runpod.wait_ssh)(pod["id"])
        from .. import remote

        pod["lease"] = (arm or lease.arm)(remote.ssh_argv(host, port), int(ns.max_hours * 3600), first_idle_min=ns.first_idle_min,
                                          idle_min=ns.idle_min)
    except Exception as e:  # noqa: BLE001  (no sshd, a failed ssh, a failed arm: the pod must not outlive this call unbounded)
        print(f"research pods create: pod {pod['id']} could not be armed ({type(e).__name__}: {e}); terminating it", file=sys.stderr)
        try:
            kill(pod["id"])
        except Exception as e2:  # noqa: BLE001
            print(f"research pods create: TERMINATE FAILED for {pod['id']} ({e2}): its boot lease and the budgets guard still bound it; "
                  f"`research pods terminate {pod['id']}`", file=sys.stderr)
        return None, EXIT_FAILED
    rep = pod["lease"]
    idle = (f"idle guard pid {rep.get('guard_pid')}{'' if rep.get('guard_alive') else ' NOT RUNNING'}: terminates after {ns.first_idle_min:g} "
            f"idle min before a first run, {ns.idle_min:g} after") if ns.idle_min > 0 else "no idle guard (--idle-min 0)"
    print(f"research pods create: pod {pod['id']} leased until {rep.get('expires_utc')} (lease loop pid {rep.get('lease_pid')}, {idle}); "
          f"`research pods extend {ns.name} --hours H` extends it", file=sys.stderr)
    return pod, 0


def main(argv: list[str]) -> int:
    """`research pods check-part <machine|pod-id> [--json]` | `research pods create <runpod create flags> --max-hours H [...]`
    | `research pods extend|lease TARGET` | `research pods ssh|sync ...` (connect.py) | any other
    verb goes to `python -m research.pods.runpod` unchanged.  Exit 3 = not the reference part (check-part, create);
    create's exit 6 = no budget line lets it (research.pods.budgets.gate)."""
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
    if argv[:1] in (["register"], ["unregister"], ["registry"]):
        from .registry import main as registry_main           # one file per machine (registry.py)

        return registry_main(argv)
    if argv[:1] in (["extend"], ["lease"]):
        from . import lease
        from .connect import ssh_base

        ap = argparse.ArgumentParser(prog=f"research pods {argv[0]}", description="the pod's lease (research.pods.lease)")
        ap.add_argument("target", metavar="TARGET", help="machine name, pod id or pod name")
        if argv[0] == "extend":
            ap.add_argument("--hours", type=float, required=True, help="the lease ends no sooner than now + H (it is never shortened)")
            ap.add_argument("--by", default=None, help="who extends it, recorded in the lease line (default $RESEARCH_LANE, else user@host)")
        ns = ap.parse_args(argv[1:])
        import os
        import socket

        by = (getattr(ns, "by", None) or os.environ.get("RESEARCH_LANE") or f"{os.environ.get('USER', '?')}@{socket.gethostname()}").replace(" ", "_")
        cmd = lease.remote_argv("extend", str(int(ns.hours * 3600)), by) if argv[0] == "extend" else lease.remote_argv("show")
        try:
            r = remote.ssh_run(ssh_base(ns.target), cmd, timeout=90)
        except (remote.Refused, OSError) as e:
            print(f"research pods {argv[0]}: {e}", file=sys.stderr)
            return 1
        out = r.stdout.decode(errors="replace").strip()
        if r.returncode != 0:
            print(f"research pods {argv[0]}: {ns.target}: {(out + r.stderr.decode(errors='replace')).strip()[-300:] or f'rc {r.returncode}'} "
                  f"(a pod created before leases has none)", file=sys.stderr)
            return 1
        print(f"{ns.target}: lease {out}")
        return 0
    if argv[:1] == ["create"]:
        ap = argparse.ArgumentParser(prog="research pods create")
        runpod.create_args(ap)
        ap.add_argument("--max-hours", type=float, required=True,
                        help="the pod's lease: it terminates itself this long after creation unless extended (research pods extend, "
                             "research run --on); at most the budget line's max_pod_hours")
        ap.add_argument("--boot-lease", action="store_true",
                        help="also run the lease loop from the pod's start command, before sshd (RunPod images only; research.pods.lease)")
        ap.add_argument("--first-idle-min", type=float, default=20.0, help="the idle guard's limit before the pod's first run (default 20)")
        ap.add_argument("--idle-min", type=float, default=90.0,
                        help="the idle guard's limit after it (default 90); 0 arms no idle guard (a pool pod kept up idle: its lease and "
                             "budget line still bound it)")
        ap.add_argument("--require-reference-part", action="store_true", help="exit 3 (pod left running) unless the board is the reference part")
        ap.add_argument("--register", action="store_true", help="register the pod under --name once sshd answers (research pods register)")
        ap.add_argument("--project", default=None, help="--register: the project the machine is bound to")
        ap.add_argument("--guard", default=None, metavar="N|true", help="--register: opt the pod into the idle guard")
        ap.add_argument("--allow-duplicate", action="store_true", help="create even when a live pod of this --name exists")
        ns = ap.parse_args(argv[1:])
        if ns.register and not ns.project:
            ap.error("--register needs --project")
        if ns.max_hours <= 0:
            ap.error("--max-hours must be positive")
        pod, rc = create_leased(ns)
        if pod is None:
            return rc
        if ns.register:
            from .registry import main as registry_main

            rc = registry_main(["register", ns.name, "--pod-id", pod["id"], "--project", ns.project,
                                *(["--guard", ns.guard] if ns.guard else [])])
            if rc:
                print(f"research pods create: pod {pod['id']} is up but not registered", file=sys.stderr)
                return rc
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
    sys.argv = ["research pods", *argv]
    runpod.main()
    return 0
