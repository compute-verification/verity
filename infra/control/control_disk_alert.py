"""Infra's relay for the control pod's root disk (vy-control-verity's container overlay; when the old pod's 10 GB one filled on 1 Oct,
the spend guard's state writes failed from 16:12Z). Every 5 minutes, read `df -P /` on the pod over ssh. One alert in #agent-alerts when usage
reaches 90%, another at 95%, one note when it falls back under 85%, and one alert when the pod can't be read for 30 minutes.

The same pass reads the steward's notes clone, whose sync (and the budgets guard's read of budgets.toml) stalled silently from
9 Oct to 13 Oct on a detached HEAD and a stale index.lock. One alert when the clone's HEAD is detached, its index.lock is older than
10 minutes, or a change has waited unpushed for an hour (a local commit or an uncommitted file), and one note when it clears.
The pod holds no Slack credential, so this runs on infra's VM and stops when that VM is suspended.

usage (tmux control-disk-alert, research importable, $SLACK_BOT_TOKEN set, $RESEARCH_SLACK_REGISTRY when not run from a checkout):
  python3 infra/control/control_disk_alert.py [--once] [--dry-run]"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research import slack
from research.pods import budgets, connect

SEEN = Path.home() / "control-disk/seen.json"
LEVELS = (90, 95)
CLEAR = 85
UNREAD_S = 30 * 60
BIG = "du -xslm /tmp /var /usr /root 2>/dev/null"
NOTES = "/workspace/steward/research-notes"
LOCK_S, UNPUSHED_S = 10 * 60, 60 * 60
# now, branch (- when detached), index.lock mtime (0: none), oldest unpushed commit's time (0: none), oldest uncommitted file's mtime
# (0: none; the steward's own state file is never committed)
SYNC = (f"cd {NOTES} && b=$(git symbolic-ref -q --short HEAD || echo -) && "
        "l=$(stat -c %Y .git/index.lock 2>/dev/null || echo 0) && "
        "u=$([ \"$b\" = - ] || git log --format=%ct \"origin/$b..HEAD\" | tail -1) && "
        "d=$(git status --porcelain=v1 -z --untracked-files=all | tr '\\0' '\\n' | cut -c4- | grep -vx steward-state.json "
        "| xargs -r -d '\\n' stat -c %Y 2>/dev/null | sort -n | head -1) && "
        "echo \"$(date +%s) $b $l ${u:-0} ${d:-0}\"")
DRY = "--dry-run" in sys.argv
ALERTS = slack.channel_id_for(slack.load_registry(slack.registry_path()), "agent-alerts")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def post(text: str) -> bool:
    """A top-level post in #agent-alerts (`msg send` posts top-level only in #agent-coordination)."""
    try:
        ts = slack._post(slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=DRY), ALERTS, "infra", text)
    except Exception as e:  # noqa: BLE001
        log(f"post failed: {type(e).__name__}: {e}")
        return False
    log(f"posted {ts}: {text[:200]}")
    return True


def pod(cmd: str) -> str:
    # resolved on every read: a RunPod reset gives the pod a new ssh port
    return subprocess.run([*connect.ssh_base(budgets.GUARD_HOST), cmd], capture_output=True, text=True, timeout=120, check=True).stdout


def sync_stall(line: str) -> list[str]:
    """Why the steward's notes sync is stalled, from one line of SYNC's output; empty when it isn't."""
    now, branch, lock, unpushed, dirty = line.split()
    now, lock, unpushed, dirty = int(now), int(lock), int(unpushed), int(dirty)
    why = []
    if branch == "-":
        why.append("HEAD is detached (an aborted rebase leaves it so): check out `main`")
    if lock and now - lock > LOCK_S:
        why.append(f"`.git/index.lock` is {(now - lock) / 3600:.1f} h old: remove it once no git process runs")
    if unpushed and now - unpushed > UNPUSHED_S:
        why.append(f"a local commit has waited {(now - unpushed) / 3600:.1f} h unpushed")
    if dirty and now - dirty > UNPUSHED_S:
        why.append(f"an uncommitted change has waited {(now - dirty) / 3600:.1f} h")
    return why


seen = json.loads(SEEN.read_text()) if SEEN.exists() else {"level": 0, "last_read": time.time(), "unread": False}
while True:
    try:
        why = sync_stall(pod(SYNC).strip().splitlines()[-1])
        # two reads in a row: a file mirrored with an old mtime is committed by the steward's next pass
        seen["sync_bad"] = seen.get("sync_bad", 0) + 1 if why else 0
        if seen["sync_bad"] >= 2 and not seen.get("sync_stall"):
            seen["sync_stall"] = post(f"*Steward notes sync stalled* on `vy-control-verity` (`{NOTES}`): " + "; ".join(why) +
                                      ". The budgets guard reads budgets.toml through this clone, and `pods create` refuses while that "
                                      "read fails. Log: `/workspace/steward/watch.log`.")
        elif not why and seen.get("sync_stall"):
            seen["sync_stall"] = not post("*Steward notes sync flowing again* on `vy-control-verity`.")
        log("steward sync: " + ("; ".join(why) or "ok"))
    except Exception as e:  # noqa: BLE001
        log(f"steward sync read failed: {type(e).__name__}: {e}")
    try:
        size, used, avail, pct = (int(x.rstrip("%")) for x in pod("df -Pk / | tail -1").split()[1:5])
        seen["last_read"] = time.time()
        if seen["unread"]:
            seen["unread"] = not post(f"*Control pod readable again:* `vy-control-verity` root disk at {pct}%.")
        log(f"{pct}% used {used / 2**20:.2f} GiB of {size / 2**20:.2f} GiB, {avail / 2**20:.2f} GiB free")
        level = max((lv for lv in LEVELS if pct >= lv), default=0)
        if level > seen["level"]:
            big = ", ".join(f"`{p}` {int(m) / 1024:.2f} GiB" for m, p in (ln.split("\t") for ln in pod(BIG).splitlines()))
            if post(f"*Control pod disk at {pct}%:* `vy-control-verity`'s root (a {size / 2**20:.0f} GiB overlay) has "
                    f"{avail / 2**20:.2f} GiB free. When it filled on 1 Oct, the spend guard's state writes failed. Largest: {big}. "
                    "ssh: `research pods ssh vy-control-verity` (key `RUNPOD_SSH_KEY_B64`)."):
                seen["level"] = level
        elif pct < CLEAR and seen["level"]:
            if post(f"*Control pod disk back to {pct}%* (under {CLEAR}%): `vy-control-verity` has {avail / 2**20:.2f} GiB free."):
                seen["level"] = 0
    except Exception as e:  # noqa: BLE001
        log(f"read failed: {type(e).__name__}: {e}")
        late = time.time() - seen["last_read"]
        if late > UNREAD_S and not seen["unread"]:
            seen["unread"] = post(f"*Control pod unreadable for {late / 60:.0f} minutes:* infra's disk relay can't run `df` on "
                                  f"`vy-control-verity` (`{type(e).__name__}`), so its disk isn't watched.")
    if not DRY:
        SEEN.write_text(json.dumps(seen))
    if "--once" in sys.argv:
        break
    time.sleep(300)
