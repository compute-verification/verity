#!/usr/bin/env bash
# `bash adapters/vllm/tools/workload/run.sh [window|commit|serve] [--dry-run] [--development] [FLAG...]`: one of the vLLM workload's runs,
# started by `research run` on the node placement.toml places it on, with that node's paths and the run's own variables as
# its environment, and the model the workload names (placement.toml's `row`, the integration's file adapters/vllm/workloads/<row>.json, fetched by
# its art id and checked against that directory's SHA256SUMS: `model`, `case` and `engine.revision`):
#   window   the default: one sampling block of the served execution blocks through every subprotocol,
#            benchmarks/one_stage/served_verdicts.sh: the endpoint committing under the workload's Build of record
#            (`predict row`, held to the traced workload dir placement.toml's COMMIT_RUN names), its requests
#            (serve_load.py --varied), every closed sampling block sealed, the sampling
#            block's SPOT (Flock --zk behind the Lean ZK server, and the Lean verifier's verdict), PoUW, the
#            bandwidth audit, PoUS and the network attester where each has its piece, and one record of every verdict.
#            The run's exit (`research status RUN`) is the record's: 0 only for the accept of a compiled Lean composite
#            of record (composed off the served node), 2 for pending, which means no composite of record yet and is
#            what a window run on the served node ends with, not a crash, and 1 for a rejection, a refused composite or
#            a failed step
#   commit   once per workload: its Build, Match and Commit with its sealed replay bundle (commit.sh), a workload dir under
#            LLAMA_ROOT (placement.toml's COMMIT_RUN names the one `window` and `serve` use)
#   serve    the endpoint alone, serving until UNTIL (serve.sh)
# A run with a coins table (the window's) takes the gate's coins, the auditor's challenger's, sending the files its
# `send` names from the tree's root; --development takes its development coins instead, and the run says so.
# Further FLAGs go to `research run` (`--env WC_TIMEOUT=45m`); --dry-run prints the command and
# starts nothing. It returns once the run starts, 0, or 1 where it can't start it: `research status RUN` checks on it.
# A placement value's ${NAME} is an address, which this environment holds and no file does (the window run's H1_GATE
# names VY_CHALLENGER_1_HOST, the challenger host's): with NAME unset nothing starts, unless a FLAG sets that value.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../../../.."
RUN=window
case ${1:-} in window|commit|serve) RUN=$1; shift;; esac
DRY=0 DEV=0
while :; do
  case ${1:-} in --dry-run) DRY=1; shift;; --development) DEV=1; shift;; *) break;; esac
done
mapfile -t ARGV < <(python3 - "$HERE" "$RUN" "$DEV" "$@" <<'EOF'
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

here, run, dev = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
script = {"window": "benchmarks/one_stage/served_verdicts.sh", "commit": "adapters/vllm/tools/workload/commit.sh", "serve": "adapters/vllm/tools/workload/serve.sh"}[run]
placement = tomllib.loads((here / "placement.toml").read_text())
r = placement["run"][run]
node = placement["node"][r["node"]]
if dev and "development" not in r.get("coins", {}):
    sys.exit(f"run.sh: the {run} run takes no coins, so --development chooses nothing: drop it")
coins = r.get("coins", {}).get("development" if dev else "gate", {})
for name in coins.get("send", []):
    if not Path(name).is_file():
        sys.exit(f"run.sh: the {run} run takes the gate's coins, the auditor's challenger's, and sends {name}, which is not at "
                 f"the tree's root: put it there (the deployment's challenger-key/v1 record, labelled `challenger-key current`), "
                 f"or choose development coins with --development")
placed = {**{k: v for k, v in node.items() if not isinstance(v, dict)}, **(node.get("serving", {}) if run != "commit" else {}),
          **r.get("env", {}), **coins.get("env", {})}
flags = {a.split("=", 1)[0] for f, a in zip(sys.argv[4:], sys.argv[5:]) if f == "--env"}
for key, value in list(placed.items()):
    unset = [name for name in re.findall(r"\$\{(\w+)\}", str(value)) if not os.environ.get(name)]
    if unset and key in flags:
        del placed[key]  # the FLAG's value is the run's
    elif unset:
        sys.exit(f"run.sh: {unset[0]} is unset: placement.toml's {key} names it, so set it where the run is launched, "
                 f"or give --env {key}=VALUE")
    else:
        placed[key] = re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m[1]], str(value))
row, rows = placement["row"], here.parents[1] / "workloads"
sums = {name.lstrip("*"): h for h, name in (line.split() for line in (rows / "SHA256SUMS").read_text().splitlines() if line.strip())}
if (want := sums.get(f"{row}.json")) is None:
    sys.exit(f"run.sh: placement.toml's row {row} is not a workload adapters/vllm/workloads/SHA256SUMS lists")
f = rows / f"{row}.json"
if not f.is_file():
    subprocess.run(["research", "data", "fetch-fixtures", str(f)], check=True, stdout=sys.stderr)
if (got := hashlib.sha256(f.read_bytes()).hexdigest()) != want:
    sys.exit(f"run.sh: {f}: SHA-256 {got} is not the {want} SHA256SUMS lists; `research data fetch-fixtures {f}` restores it")
doc = json.loads(f.read_text())
env = {**placed, "REPO": doc["model"], "REV": doc["engine"]["revision"], "ROLE": doc["case"],
       "ROW": row if run == "commit" else f"{node['LLAMA_ROOT']}/{node['COMMIT_RUN']}/{row}"}
print("\n".join(["research", "run", "--on", r["node"], "--project", "verity", "--source", ".", *r["research"],
                 *(x for name in coins.get("send", []) for x in ("--send", name)),
                 *(x for k, v in env.items() for x in ("--env", f"{k}={v}")), "--", *r.get("wrap", []), *coins.get("wrap", []),
                 "bash", script]))
EOF
)
[ "${#ARGV[@]}" -gt 0 ] || exit 1
for i in "${!ARGV[@]}"; do [ "${ARGV[$i]}" = -- ] && break; done
CMD=("${ARGV[@]:0:$i}" "$@" "${ARGV[@]:$i}")
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
