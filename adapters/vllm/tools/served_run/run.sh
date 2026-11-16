#!/usr/bin/env bash
# Starts one of the served run's runs where placement.toml places it (the README's Commands).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../../.."
RUN=window
case ${1:-} in window|compose|setup|copy|audit|settings) RUN=$1; shift;; esac
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
r = placement["run"]["window" if run == "settings" else run]
if dev and "development" not in r.get("coins", {}):
    sys.exit(f"run.sh: the {run} run takes no coins, so --development chooses nothing: drop it")
if tie and "tie" not in r:
    sys.exit(f"run.sh: the {run} run has no tied form: drop --tie")
coins = {} if run == "settings" else r.get("coins", {}).get("development" if dev else "gate", {})
given = {k: v for f, a in zip(sys.argv[5:], sys.argv[6:]) if f == "--env" for k, _, v in [a.partition("=")]}
held = {**{k: v for k, v in os.environ.items() if v}, **{k: v for k, v in given.items() if v}}
PARAMETER = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")
ON_NODE = {"HOME", *(k for n in placement["node"].values() for k in n.get("env_files", {}))}


def unset(value: str) -> list[str]:
    return [m[1] for m in PARAMETER.finditer(value) if m[1] not in held and m[1] not in ON_NODE and m[2] is None]


def fill(value: str, where: str) -> str:
    """Exits naming the first ${NAME} with no default that neither the environment nor a FLAG's --env sets."""
    if names := unset(value):
        sys.exit(f"run.sh: {names[0]} is unset: placement.toml's {where} names it, so set it where the run is launched, "
                 f"or give --env {names[0]}=VALUE")
    return PARAMETER.sub(lambda m: m[0] if m[1] in ON_NODE else held.get(m[1], m[2] or ""), value)


def workload(row: str) -> tuple[Path, dict]:
    rows = here.parents[1] / "workloads"
    sums = {name.lstrip("*"): h for h, name in (line.split() for line in (rows / "SHA256SUMS").read_text().splitlines() if line.strip())}
    if (want := sums.get(f"{row}.json")) is None:
        sys.exit(f"run.sh: placement.toml's row {row} is not a workload adapters/vllm/workloads/SHA256SUMS lists")
    f = rows / f"{row}.json"
    if not f.is_file():
        subprocess.run(["research", "data", "fetch-fixtures", str(f)], check=True, stdout=sys.stderr)
    if (got := hashlib.sha256(f.read_bytes()).hexdigest()) != want:
        sys.exit(f"run.sh: {f}: SHA-256 {got} is not the {want} SHA256SUMS lists; `research data fetch-fixtures {f}` restores it")
    return f, json.loads(f.read_text())


if run == "settings":
    model = sys.argv[5] if len(sys.argv) > 5 else ""
    here_first = sorted(("row", "tied_row"), key=lambda k: not (here.parents[1] / "workloads" / f"{placement[k]}.json").is_file())
    if (k := next((k for k in here_first if workload(placement[k])[1]["model"] == model), None)) is None:
        sys.exit(f"run.sh: settings takes the model of a row placement.toml places ({placement['row']}, "
                 f"{placement['tied_row']}), not {model!r}")
    tie = k == "tied_row"


on = fill(r["node"], f"{run} run's node")
node = placement["node"].get(on, {})


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


placed = {**{k: v for k, v in node.items() if not isinstance(v, dict)}, **r.get("env", {}),
          **r.get("tie" if tie else "plain", {}).get("env", {}), **coins.get("env", {})}
# PRODUCT is the configuration's, and a launch parameter with no default the launch's
env = {k: fill(str(v), k) for k, v in placed.items()
       if k not in given and not (run == "settings" and (k == "PRODUCT" or unset(str(v))))}
send = [fill(s, f"{run} run's send") for s in r.get("send", [])]
for name in send:
    if not Path(name).is_file():
        sys.exit(f"run.sh: the {run} run sends {name}, which is not a file")
head = git("rev-parse", "HEAD").stdout.strip()
if run == "compose":
    with tarfile.open(send[0]) as t, t.extractfile("private/compose/compose.env") as f:
        served = dict(line.split("=", 1) for line in f.read().decode().splitlines() if "=" in line).get("COMPOSE_TREE", "")
    if served != head:
        sys.exit(f"run.sh: the window was served at {served or 'no commit'} and this checkout is at {head}: compose "
                 f"at the window's commit, from a checkout of it (git worktree add DIR {served})")
if coins.get("main_line"):
    subprocess.run(["git", "fetch", "-q", "origin", "main"], check=True, stdout=sys.stderr)
    line = git("rev-list", "--first-parent", "origin/main").stdout
    if head not in line.split():
        sys.exit(f"run.sh: {head} is not on origin/main's first-parent line, and the gate gives coins only to a commit on "
                 "that line (main.txt): land it first, or take development coins with --development")
    (main := Path(tempfile.mkdtemp(prefix="served-run-")) / "main.txt").write_text(line)
    send.append(str(main))
if r.get("send_workload"):
    row = placement["tied_row" if tie else "row"]
    f, doc = workload(row)
    env |= {"REPO": doc["model"], "REV": doc["engine"]["revision"], "ROLE": doc["case"], "ROW_ID": row}
    if (built := node.get("builds", {}).get(row)) is not None:
        env["ROW"] = f"{node['LLAMA_ROOT']}/{built}/{row}"
    elif run in ("window", "settings"):
        sys.exit(f"run.sh: placement.toml's builds name no traced Build of {row} on {on}: run "
                 f"`run.sh setup{' --tie' if tie else ''}` and name its run id there")
    taps = ["adapters/vllm/verity_vllm/acquire/fa3_tap_src", "adapters/vllm/verity_vllm/acquire/norm_tap_src",
            "adapters/vllm/verity_vllm/ops/pod_fa3_tap.sh", "adapters/vllm/verity_vllm/ops/pod_norm_tap.sh"]
    if run == "window" and (d := git("diff", "--name-only", node["TAPS_AT"], "HEAD", "--", *taps)).stdout + d.stderr:
        sys.exit(f"run.sh: the taps were built at {node['TAPS_AT'][:12]} and this commit differs there "
                 f"({' '.join((d.stdout or d.stderr).split()[:4])}): rebuild them (`run.sh setup` with --env STEPS=taps) "
                 "and, once its exactness test passes, set placement.toml's TAPS_AT to that run's commit")
    send.append(os.path.relpath(f))
if run == "settings" and "--" in sys.argv[6:]:
    files = {}
    for var, path in node["env_files"].items():
        if not (p := Path(path)).is_file() or p.stat().st_mode & 0o777 != 0o600:
            sys.exit(f"run.sh: {path}, the mode-600 env file that sets {var} on this node, is missing or has another mode")
        files |= {k.removeprefix("export ").strip(): v.strip().strip("'\"") for k, _, v in
                  (line.partition("=") for line in p.read_text().splitlines() if "=" in line and not line.startswith("#"))}
        if not files.get(var):
            sys.exit(f"run.sh: {path} sets no {var}")
    known, named = {**os.environ, **files}, re.compile(r"\$\{(\w+)\}")
    env = {k: named.sub(lambda m: known.get(m[1], m[0]), v) for k, v in {**env, **os.environ}.items()
           if k not in os.environ or named.search(v)} | {v: files[v] for v in node["env_files"]}
if run == "settings":
    print("\n".join(f"{k}={v}" for k, v in env.items()))
    sys.exit(0)
wrap = [fill(w, f"{run} run's wrap") for w in r.get("wrap", [])]
print("\n".join(["research", "run", "--on", on, "--project", "verity", "--source", ".",
                 *(fill(x, f"{run} run's research") for x in r["research"]),
                 *(x for name in send for x in ("--send", name)),
                 *(x for k, v in env.items() for x in ("--env", f"{k}={v}")), "--", *wrap, "bash", script]))
EOF
)
[ "${#ARGV[@]}" -gt 0 ] || exit 1
if [ "$RUN" = settings ]; then
  while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
  [ $# -gt 1 ] || { printf '%s\n' "${ARGV[@]}"; exit 0; }
  shift
  for kv in "${ARGV[@]}"; do export "${kv?}"; done
  log() { echo "run.sh settings: $*" >&2; }
  . adapters/vllm/tools/served_run/workload_env.sh || exit 2
  mkdir -p "$SERVE_ROOT" && chmod 700 "$SERVE_ROOT"
  "$@"
  exit
fi
for i in "${!ARGV[@]}"; do [ "${ARGV[$i]}" = -- ] && break; done
CMD=("${ARGV[@]:0:$i}" "$@" "${ARGV[@]:$i}")
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
