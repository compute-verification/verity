#!/usr/bin/env bash
# `verity example vllm [window|commit|serve] [--dry-run] [FLAG...]`: one of the vLLM example's runs, started by `research run` on the
# node placement.toml places it on, with that node's paths and the run's own variables as its environment, and the model the
# example's row names (its one file in workloads/: `model`, `case` and `engine.revision`; bf16, served as pinned, `--precision pin`):
#   window   the default: one served window through every subprotocol, benchmarks/one_stage/served_verdicts.sh: the endpoint
#            committing under the row's Commit of record, its requests (serve_load.py --varied), every closed window sealed, the
#            window's sampled proofs (C-Flock --zk behind the Lean firewall, and the Lean verifier's verdict), PoUW, the bandwidth
#            audit, PoUS and the network certifier where each has its piece, and one record of every verdict
#   commit   once per row: the row's Commit of record with its sealed replay bundle (commit.sh), the row dir the endpoint commits
#            under (placement.toml's COMMIT_RUN names the one `window` and `serve` use)
#   serve    the endpoint alone, serving until UNTIL (serve.sh)
# Further FLAGs go to `research run` (`--env CLASSES='4,32,4 record'`, `--env PP_CHALLENGER=...`); --dry-run prints the command and
# starts nothing. It returns once the run starts: `research status RUN` checks on it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../.."
RUN=window
case ${1:-} in window|commit|serve) RUN=$1; shift;; esac
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
mapfile -t ARGV < <(python3 - "$HERE" "$RUN" <<'EOF'
import json
import sys
import tomllib
from pathlib import Path

here, run = Path(sys.argv[1]), sys.argv[2]
script = {"window": "benchmarks/one_stage/served_verdicts.sh", "commit": "examples/vllm/commit.sh", "serve": "examples/vllm/serve.sh"}[run]
rows = sorted((here / "workloads").glob("*.json"))
if len(rows) != 1:
    sys.exit(f"run.sh: examples/vllm/workloads/ holds {len(rows)} row files; the example's row is its one")
row, doc = rows[0].stem, json.loads(rows[0].read_text())
placement = tomllib.loads((here / "placement.toml").read_text())
r = placement["run"][run]
node = placement["node"][r["node"]]
env = {**{k: v for k, v in node.items() if not isinstance(v, dict)}, **(node.get("serving", {}) if run != "commit" else {}),
       **r.get("env", {}), "REPO": doc["model"], "REV": doc["engine"]["revision"], "ROLE": doc["case"], "PRECISION": "pin",
       "ROW": row if run == "commit" else f"{node['LLAMA_ROOT']}/{node['COMMIT_RUN']}/{row}"}
print("\n".join(["research", "run", "--on", r["node"], "--project", "verity", "--source", ".", *r["research"],
                 *(x for k, v in env.items() for x in ("--env", f"{k}={v}")), "--", *r.get("wrap", []), "bash", script]))
EOF
)
[ "${#ARGV[@]}" -gt 0 ] || exit 2
for i in "${!ARGV[@]}"; do [ "${ARGV[$i]}" = -- ] && break; done
CMD=("${ARGV[@]:0:$i}" "$@" "${ARGV[@]:$i}")
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
