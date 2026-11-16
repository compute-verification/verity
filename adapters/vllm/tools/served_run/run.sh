#!/usr/bin/env bash
# `bash adapters/vllm/tools/served_run/run.sh [window|compose|setup] [--dry-run] [--development] [--tie] [FLAG...]`: one of
# the vLLM served run's runs, its script adapters/vllm/tools/served_run/RUN.sh, started by `research run` on the node
# placement.toml places it on, with that node's paths and the run's own variables as its environment, and for a run on
# the served node the model the served run's workload names (placement.toml's `row`, or `tied_row` with --tie: the
# integration's file adapters/vllm/workloads/<row>.json, fetched by its art id and checked against that directory's
# SHA256SUMS: `model`, `case` and `engine.revision`):
#   window   the default, `CARD=N CORES=A-B END=UTC run.sh window`: one execution phase served on card CARD and cores CORES
#            through benchmarks/one_stage/served_verdicts.sh (window.sh), under the row's Build of record (held to its
#            traced Build, placement.toml's builds), SPOT on it and each subprotocol's piece where it has one, and the
#            compose hand-off packed into the run's out/handoff/. With --tie, the tied window (placement.toml's
#            [run.window.tie]). It exits 0 when it did what it should, its record pending (composed off the node), 1 for
#            a rejection or a failed step, 75 where the card is busy or the row too short, with nothing run
#   compose  `HANDOFF=DIR run.sh compose`, from a checkout of the window's commit: that window's record composed off the
#            served node (compose.sh) from its hand-off, DIR holding the window run's out/handoff/compose.tar and its .sha256
#   setup    `CARD=N CORES=A-B END=UTC run.sh setup [--tie]`, once per row and node: what the window reads (setup.sh)
# The window takes the gate's coins, the auditor's challenger's, with main's first-parent line read from origin's main at
# launch, so it starts only from a commit on that line; --development takes its development coins instead, and the run
# says so. Further FLAGs go to `research run` (`--env WC_TIMEOUT=45m`, `--timeout 200m`); --dry-run prints the command and
# starts nothing. It returns once the run starts, 0, or 1 where it can't start it: `research status RUN` checks on it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../../.."
RUN=window
case ${1:-} in window|compose|setup) RUN=$1; shift;; esac
DRY=0 DEV=0 TIE=0
while :; do
  case ${1:-} in --dry-run) DRY=1; shift;; --development) DEV=1; shift;; --tie) TIE=1; shift;; *) break;; esac
done
mapfile -t ARGV < <(python3 - "$HERE" "$RUN" "$DEV" "$TIE" "$@" <<'EOF'
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

here, run, dev, tie = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
script = f"adapters/vllm/tools/served_run/{run}.sh"
placement = tomllib.loads((here / "placement.toml").read_text())
r = placement["run"][run]
node = placement["node"][r["node"]]
if dev and "development" not in r.get("coins", {}):
    sys.exit(f"run.sh: the {run} run takes no coins, so --development chooses nothing: drop it")
if tie and run == "compose":
    sys.exit("run.sh: the compose run composes the hand-off it is given, tied or not: drop --tie")
coins = r.get("coins", {}).get("development" if dev else "gate", {})
given = {k: v for f, a in zip(sys.argv[5:], sys.argv[6:]) if f == "--env" for k, _, v in [a.partition("=")]}
held = {**{k: v for k, v in os.environ.items() if v}, **{k: v for k, v in given.items() if v}}


def fill(value: str, where: str) -> str:
    """``value`` with each ${NAME} the launch's environment or a FLAG's --env gives; with one unset nothing starts."""
    if unset := [name for name in re.findall(r"\$\{(\w+)\}", value) if name not in held]:
        sys.exit(f"run.sh: {unset[0]} is unset: placement.toml's {where} names it, so set it where the run is launched, "
                 f"or give --env {unset[0]}=VALUE")
    return re.sub(r"\$\{(\w+)\}", lambda m: held[m[1]], value)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


placed = {**{k: v for k, v in node.items() if not isinstance(v, dict)}, **r.get("env", {}),
          **(r.get("tie", {}).get("env", {}) if tie else {}), **coins.get("env", {})}
env = {k: fill(str(v), k) for k, v in placed.items() if k not in given}  # a FLAG's value is the run's
send = [fill(s, f"{run} run's send") for s in r.get("send", [])]
for name in send:
    if not Path(name).is_file():
        sys.exit(f"run.sh: the {run} run sends {name}, which is not a file")
head = git("rev-parse", "HEAD").stdout.strip()
if run == "compose":  # composed at the window's commit, which its hand-off names
    with tarfile.open(send[0]) as t, t.extractfile("private/compose/compose.env") as f:
        served = dict(line.split("=", 1) for line in f.read().decode().splitlines() if "=" in line).get("COMPOSE_TREE", "")
    if served != head:
        sys.exit(f"run.sh: the window was served at {served or 'no commit'} and this checkout is at {head}: compose "
                 f"at the window's commit, from a checkout of it (git worktree add DIR {served})")
if coins.get("main_line"):  # the line the gate's session pins on, origin's main as of now
    subprocess.run(["git", "fetch", "-q", "origin", "main"], check=True, stdout=sys.stderr)
    line = git("rev-list", "--first-parent", "origin/main").stdout
    if head not in line.split():
        sys.exit(f"run.sh: {head} is not on origin/main's first-parent line, and the gate gives coins only to a commit on "
                 "that line (main.txt): land it first, or take development coins with --development")
    (main := Path(tempfile.mkdtemp(prefix="served-run-")) / "main.txt").write_text(line)
    send.append(str(main))
if "LLAMA_ROOT" in node:  # a run on the served node: the workload's model, and its Build of record's traced dir
    row, rows = placement["tied_row" if tie else "row"], here.parents[1] / "workloads"
    sums = {name.lstrip("*"): h for h, name in (line.split() for line in (rows / "SHA256SUMS").read_text().splitlines() if line.strip())}
    if (want := sums.get(f"{row}.json")) is None:
        sys.exit(f"run.sh: placement.toml's row {row} is not a workload adapters/vllm/workloads/SHA256SUMS lists")
    f = rows / f"{row}.json"
    if not f.is_file():
        subprocess.run(["research", "data", "fetch-fixtures", str(f)], check=True, stdout=sys.stderr)
    if (got := hashlib.sha256(f.read_bytes()).hexdigest()) != want:
        sys.exit(f"run.sh: {f}: SHA-256 {got} is not the {want} SHA256SUMS lists; `research data fetch-fixtures {f}` restores it")
    doc = json.loads(f.read_text())
    env |= {"REPO": doc["model"], "REV": doc["engine"]["revision"], "ROLE": doc["case"], "ROW_ID": row}
    if (built := node.get("builds", {}).get(row)) is not None:
        env["ROW"] = f"{node['LLAMA_ROOT']}/{built}/{row}"
    elif run == "window":
        sys.exit(f"run.sh: placement.toml's builds name no traced Build of {row} on {r['node']}: run "
                 f"`run.sh setup{' --tie' if tie else ''}` and name its run id there")
    taps = ["adapters/vllm/verity_vllm/acquire/fa3_tap_src", "adapters/vllm/verity_vllm/acquire/norm_tap_src",
            "adapters/vllm/verity_vllm/ops/pod_fa3_tap.sh", "adapters/vllm/verity_vllm/ops/pod_norm_tap.sh"]
    if run == "window" and (d := git("diff", "--name-only", node["TAPS_AT"], "HEAD", "--", *taps)).stdout + d.stderr:
        sys.exit(f"run.sh: the taps were built at {node['TAPS_AT'][:12]} and this commit differs there "
                 f"({' '.join((d.stdout or d.stderr).split()[:4])}): rebuild them (`run.sh setup` with --env STEPS=taps) "
                 "and set placement.toml's TAPS_AT to that run's commit")
    if r.get("send_workload"):  # the fetched fixture, which the run's clone of the tree doesn't hold
        send.append(os.path.relpath(f))
wrap = [fill(w, f"{run} run's wrap") for w in r.get("wrap", [])]
print("\n".join(["research", "run", "--on", r["node"], "--project", "verity", "--source", ".", *r["research"],
                 *(x for name in send for x in ("--send", name)),
                 *(x for k, v in env.items() for x in ("--env", f"{k}={v}")), "--", *wrap, "bash", script]))
EOF
)
[ "${#ARGV[@]}" -gt 0 ] || exit 1
for i in "${!ARGV[@]}"; do [ "${ARGV[$i]}" = -- ] && break; done
CMD=("${ARGV[@]:0:$i}" "$@" "${ARGV[@]:$i}")
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
