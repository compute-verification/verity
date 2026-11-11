#!/usr/bin/env bash
# The network certifier's id form in front of a vLLM server that answers stock HTTP (vLLM's own server under the certified
# tick: corpus.py's --admit tick, serve_tax's tick arm), from the tree's root:
#   UPSTREAM=HOST:PORT [LISTEN=127.0.0.1:18746] [CELL_BYTES=4096] [LOG=FILE] [PY=python] \
#     bash benchmarks/network_e2e/traces/ids.sh [CMD ARG...]
# It starts the id front (verity-vllm ids-front: adapters/vllm/verity_vllm/pipeline/ids_front.py) on LISTEN before the
# server at UPSTREAM, for a network certifier whose cells are CELL_BYTES bytes (an id event is CELL_BYTES - 9), with its
# record of each id request in LOG (default $RESEARCH_RUN_DIR/out/front.jsonl under research run, else none). PY is an
# interpreter with numpy: the node's vLLM venv's python, or "uv run python" in a synced tree.
# With CMD, once the front listens it runs CMD ARG..., the network certifier's side through LISTEN on this host (a process
# beside the server, not a device of the auditor's), and exits with CMD's status; the front is stopped when the job ends,
# however it ends. Without CMD the front serves until SIGTERM, for a network certifier that runs elsewhere and reaches
# LISTEN.
set -uo pipefail
UPSTREAM=${UPSTREAM:?UPSTREAM: HOST:PORT of the server}
LISTEN=${LISTEN:-127.0.0.1:18746}; CELL_BYTES=${CELL_BYTES:-4096}
LOG=${LOG:-${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/out/front.jsonl}}
read -ra py <<<"${PY:-python}"
export PYTHONPATH=$PWD:$PWD/adapters/vllm${PYTHONPATH:+:$PYTHONPATH}
front=("${py[@]}" -m verity_vllm.pipeline.cli ids-front --upstream "$UPSTREAM" --listen "$LISTEN" --id-payload "$((CELL_BYTES - 9))")
[ -z "$LOG" ] || { mkdir -p "$(dirname "$LOG")"; front+=(--log "$LOG"); }
[ $# -gt 0 ] || exec "${front[@]}"
exec 3< <(exec "${front[@]}")
pid=$!
trap 'kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null' EXIT
trap 'exit 143' TERM INT HUP
IFS= read -r listening <&3 || { echo "ids: the front ended before it listened"; exit 1; }
echo "ids: the front $listening before $UPSTREAM at $(date -u +%H:%M:%SZ)"
"$@"
