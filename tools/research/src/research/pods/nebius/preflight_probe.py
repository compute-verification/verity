"""Check's preflight on each node, hourly, so a drifted toolchain is found before a lane's check is refused (network-accounting,
infra's 6th user interview: on 30 Sep node 1's preflight refused #326's check, because uv had drifted and pod_setup.sh's links to
elan, lake and cargo were gone).

Each pass updates a clean checkout of origin/main and, for each machine, launches what `check.py --record --on` launches, with
`true` for the check: `research run --on M --preflight check --send <the Lean bundles' URLs> -- true`. Its preflight needs the URLs
wherever a check holds the node's one warm copy of the Lean dependencies (2 Oct: refused without them, passed with them); the
runner deletes them before taking the run's record (`store.custody.CREDENTIALS`), since `true` leaves them in its inputs. The
preflight refusing it
(PreflightFailed) is an incident: one post in #agent-alerts naming the machine and the failure, and a "cleared" reply in that
thread once the preflight passes there again. Infra is the only reader, so each also goes in infra's inbox thread, which wakes it.
Any other refusal (the machine held exclusively, ssh down) is only logged. A pass that passes leaves one Attempt (campaign
`infra-probe`). It runs on the control pod, as the watchers' `preflight-probe` loop (infra/control/watchers.toml).

Nodes' drift from main is the monitors relay's `drift` monitor's (`research.monitors.deploy_drift`), not this probe's.

  python3 preflight_probe.py --tree CHECKOUT [--machines vy-nebius-1 vy-nebius-2] [--every 3600] [--state PATH] [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

REFUSED = "(PreflightFailed): "
CAMPAIGN = "infra-probe"


def verdict(rc: int, out: str) -> tuple[str, str]:
    """What one probe run's output says: ("ok", run id) when it launched, ("failed", the preflight's failure) when the preflight
    refused it, else ("error", its last line)."""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    for ln in lines:
        if REFUSED in ln:
            return "failed", ln.split(REFUSED, 1)[1].strip()
    launched = next((ln for ln in lines if " launched on " in ln), None)
    if rc == 0 and launched:
        return "ok", launched.split("run ", 1)[1].split()[0]
    return "error", lines[-1] if lines else f"exit {rc}, no output"


def step(state: dict, results: dict[str, tuple[str, str]], post: Callable[[str, "str | None"], "str | None"]) -> None:
    """One post per machine whose preflight fails, and a "cleared" reply once it passes; an error changes nothing."""
    for m, (v, detail) in sorted(results.items()):
        if v == "failed" and m not in state["open"]:
            text = (f"*Check's preflight fails on `{m}`*, so a check recorded there is refused: {detail[:600]} "
                    f"Probe: `research run --on {m} --preflight check $(python tools/verity/check/check.py --lean-deps-files) -- true`.")
            state["open"][m] = {"thread": post(text, None), "since": detail[:200]}
        elif v == "ok" and m in state["open"]:
            post(f"*Cleared:* check's preflight passes on `{m}` again (run {detail}).", state["open"][m]["thread"])
            del state["open"][m]


def probe(tree: Path, machines: list[str], dry: bool) -> dict[str, tuple[str, str]]:
    """Update `tree` to origin/main, then one probe run per machine."""
    git = ["git", "-C", str(tree)]
    subprocess.run([*git, "fetch", "-q", "origin", "main"], check=True, timeout=300)
    subprocess.run([*git, "checkout", "-q", "--detach", "origin/main"], check=True, timeout=120)
    env = {**os.environ, "PYTHONPATH": str(tree / "tools" / "research" / "src")}
    send = subprocess.run([sys.executable, "tools/verity/check/check.py", "--lean-deps-files"], cwd=tree, env=env, capture_output=True,
                          text=True, timeout=300).stdout.split()
    out = {}
    for m in machines:
        argv = [sys.executable, "-m", "research", "run", "--on", m, "--project", "verity", "--source", ".", "--cwd", "source",
                "--campaign", CAMPAIGN, "--preflight", "check", *send, "--", "true"]
        if dry:
            print("would run:", " ".join(argv[1:]), flush=True)
            continue
        try:
            r = subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
            out[m] = verdict(r.returncode, r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            out[m] = ("error", "research run took over 30 minutes")
    return out


def slack_post(dry: bool) -> Callable[[str, "str | None"], "str | None"]:
    from research import slack

    reg = slack.load_registry(slack.registry_path())
    api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=dry)
    alerts, front = slack.channel_id_for(reg, "agent-alerts"), slack.channel_id_for(reg, slack.FRONT_DOOR)
    inbox = (slack.Routes().get_route("infra") or {}).get("inbox_ts")

    def post(text: str, thread: str | None) -> str | None:
        ts = slack._post(api, alerts, "infra", text, thread)
        slack._post(api, front, "infra", f"from preflight-probe: {text}", inbox)
        return ts
    return post


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="preflight_probe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, type=Path, help="a clean checkout of verity, kept at origin/main (detached)")
    ap.add_argument("--machines", nargs="+", default=["vy-nebius-1", "vy-nebius-2"])
    ap.add_argument("--every", type=float, default=3600.0)
    ap.add_argument("--state", type=Path, default=Path.home() / ".research" / "preflight-probe.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the runs instead (and keep no state)")
    ns = ap.parse_args(argv)
    state = json.loads(ns.state.read_text()) if ns.state.exists() else {"open": {}}
    post = slack_post(ns.dry_run)
    while True:
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        try:
            results = probe(ns.tree, ns.machines, ns.dry_run)
            step(state, results, post)
        except Exception as e:  # noqa: BLE001  (the next pass tries again)
            results = {}
            print(f"{stamp} pass failed: {type(e).__name__}: {e}", flush=True)
        for m, (v, detail) in sorted(results.items()):
            print(f"{stamp} {m}: {v} {detail[:300]}", flush=True)
        if not ns.dry_run:
            ns.state.parent.mkdir(parents=True, exist_ok=True)
            ns.state.write_text(json.dumps(state))
        if ns.once:
            return 0
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main())
