#!/usr/bin/env bash
# MEM-67 3a.2: one streamed wipe of one GPU's memory, as one Attempt. wipe_period's leg on this node: the prover keeps the
# fill in device memory (wipe_store.Cuda, the CUDA driver through ctypes: no toolkit, no build), the memory challenger
# streams it natively (wipe_stream.c, built and gated on the challenger's host) and times its spot reads (--reads), and
# the Lean (wipe-protocol, verity/wipe/v0) and the Python judge the signed session and must agree. cwd = a clone (the
# Lean builds into it):
#
#   research run --on NODE --project verity --source . --cwd clone --timeout 90m -- \
#     bash benchmarks/pous/streamed_wipe/demo.sh [--dry-run] [OPTIONS]
#
#   --dry-run            a 64-segment (2 GiB) fill with the memory challenger on this node: every step of the hour's
#                        run but the size and the placement (trusted-nebius-host unmet)
#   --segments N         the fill in 32 MiB segments (default: the device's free memory less --reserve-gib; 64 dry)
#   --reserve-gib G      device memory left out of the fill (default 1); the whole device is declared, so what the fill
#                        doesn't reach counts in full in the published foreign_bits (R)
#   --reads K            the timed spot reads, which the certificate is computed at (default 2152; 5231 is the H200's
#                        StreamedWipeMeets141GB5, r = 59): a challenger here is started with it; one on its own host
#                        chooses its own (its --challenger-command's wipe_party --reads K), and the leg refuses a
#                        session at another count
#   --challenger HOST --challenger-address H:P --challenger-command 'ARGV'
#                        the memory challenger on a host of its own (all three; infra's wiring of wipe_party's side
#                        there, e.g. 'python -m verity.core.protocols.pous.wipe_party --device D --listen 0.0.0.0:P
#                        --reads 5231 --streamer build'); without them it runs on this node, the premise unmet
#   --simulated          the memory challenger here, named the stand-in for one on its own host (placement
#                        `simulated`, the premise assumed; its key in this run's scratch all the same)
#   --foreign-gb G|fM    the cheat: the prover keeps G GB (10^9 bytes) of prior content in place of the fill's first
#                        labels, or fM, one label past the foreign_bits published at the reads; a cheat's run
#                        expects reject
#   --expect accept|reject  the verdict that exits 0 (default accept; reject with --foreign-gb)
#   --egress-bits N      the network certifier's egress count over the session, which a challenger here signs
#                        (default: UNCOUNTED, the bound vacuous); a challenger on its own host signs the count its
#                        --challenger-command starts it with
#   --link NAME=TO@VIA   a link leaving this node besides the control link and the challenger's, which the record names
#                        uncounted (repeatable); the control link (`control`: the default route's NIC, which ssh and
#                        research run reach the node by) is always named, and the challenger's from the connection made
#   --threads T          the native streamer's threads when the challenger runs here (default: every CPU)
#   --store cuda|host    host: the CPU stand-in, host memory for the device (no GPU)
#   --gpu NAME           refuse a GPU whose name lacks NAME (default H200; any: take any)
#   --node NAME          the prover's node, the wiped server's id (default: hostname)
#   --limit-ns N         the deadline a read (default 20 ms)
#   --no-lean            judge in Python alone; judge out/wipe.case.json with wipe-protocol later on a CPU node
#
# Under gpu-lease when the node has it (outside the lease: uv sync, the Lean build, the streamer's build check).
# Outputs under $OUT (a run's out/; else ./out): gpu.txt, env.txt, lake-build.log, period.json, wipe.json,
# wipe.case.json, summary.json. summary.json's t0 and t1 are the session's window on the challenger's clock (the prover
# connected, the session signed), and its `uncertified` what the bound doesn't cover (the placement's links and
# undeclared tiers, an uncounted egress). Scratch (the challenger's key and session log, the built streamer) is a
# mktemp -d, removed on exit: no key leaves the node. Exit 0 on the expected verdict, the Lean and the Python agreeing.
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
ROOT=$(cd "$(dirname "$SELF")/../../.." && pwd)
cd "$ROOT"
export O=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$O"

if [ "${1:-}" = leased ]; then
  [ -n "${GPU_LEASE_UUID:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
else
  DRY=0 SEGMENTS= RESERVE=1 CH= CH_ADDR= CH_CMD= THREADS= STORE=cuda GPU=H200 NODE=$(hostname) LIMIT=20000000 LEAN=1
  SIMULATED=0 FOREIGN=0 EXPECT= EGRESS= LINKS= READS=
  ARGS=("$@")
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY=1 ;;
      --simulated) SIMULATED=1 ;;
      --egress-bits) EGRESS=$2; shift ;;
      --reads) READS=$2; shift ;;
      --link) LINKS="$LINKS$2"$'\n'; shift ;;
      --foreign-gb) FOREIGN=$2; shift ;;
      --expect) EXPECT=$2; shift ;;
      --segments) SEGMENTS=$2; shift ;;
      --reserve-gib) RESERVE=$2; shift ;;
      --challenger) CH=$2; shift ;;
      --challenger-address) CH_ADDR=$2; shift ;;
      --challenger-command) CH_CMD=$2; shift ;;
      --threads) THREADS=$2; shift ;;
      --store) STORE=$2; shift ;;
      --gpu) GPU=$2; shift ;;
      --node) NODE=$2; shift ;;
      --limit-ns) LIMIT=$2; shift ;;
      --no-lean) LEAN=0 ;;
      *) echo "demo.sh: unknown option $1 (see the header)"; exit 2 ;;
    esac
    shift
  done
  if [ -n "$CH$CH_ADDR$CH_CMD" ] && [ -z "$CH" -o -z "$CH_ADDR" -o -z "$CH_CMD" ]; then
    echo "demo.sh: --challenger, --challenger-address and --challenger-command go together"; exit 2
  fi
  [ "$SIMULATED" = 1 ] && [ -n "$CH" ] && { echo "demo.sh: --simulated runs the challenger here, not on --challenger"; exit 2; }
  [ -n "$EGRESS" ] && [ -n "$CH" ] && {
    echo "demo.sh: --egress-bits reaches only a challenger here; the one on $CH signs the count its --challenger-command starts it with"; exit 2; }
  read -r CTRL_DEV CTRL_SRC < <(ip -o route get 1.1.1.1 2>/dev/null | sed -nE 's/.* dev ([^ ]+).* src ([^ ]+).*/\1 \2/p') || true
  CTRL_MBPS=$(cat "/sys/class/net/${CTRL_DEV:-none}/speed" 2>/dev/null || echo)
  [ -n "$EXPECT" ] || { [ "$FOREIGN" = 0 ] && EXPECT=accept || EXPECT=reject; }
  [ "$DRY" = 1 ] && [ -z "$SEGMENTS" ] && SEGMENTS=64
  S=$(mktemp -d)
  trap 'rm -rf "$S"' EXIT
  { echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"; echo "args: ${ARGS[*]:-}"; date -u +%FT%TZ; hostname
    lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"; free -g; df -h /tmp; uptime
    ip -br link 2>/dev/null; for i in /sys/class/net/*; do echo "$(basename "$i") $(cat "$i/speed" 2>/dev/null || echo -) Mb/s"; done
  } > "$O/env.txt" 2>&1
  uv sync -q --all-packages --extra torch-cpu
  [ -n "$CH" ] || cc -O3 -pthread verity/core/protocols/pous/wipe_stream.c -o "$S/wipe_stream-check"
  TOOLS={}
  if [ "$LEAN" = 1 ]; then
    L=verity/core/devices/memory_challenger/lean
    export PATH="$HOME/.elan/bin:$PATH"
    command -v lake >/dev/null 2>&1 || bash tools/verity/lean/setup.sh "$L" > "$O/lake-setup.log" 2>&1
    (cd "$L" && lake build wipe-protocol) > "$O/lake-build.log" 2>&1 \
      || { tail -30 "$O/lake-build.log"; echo "REFUSED: wipe-protocol doesn't build"; exit 3; }
    TOOLS="{\"wipe-protocol\": \"$ROOT/$L/.lake/build/bin/wipe-protocol\"}"
  fi
  mkdir -p "$S/periods/p0"
  uv run --no-sync python - "$S/periods/p0/period.json" "$TOOLS" "$STORE" "${SEGMENTS:-null}" "$RESERVE" "${THREADS:-null}" \
    "$LIMIT" "$CH" "$CH_ADDR" "$CH_CMD" "$SIMULATED" "$FOREIGN" "${EGRESS:-null}" "${READS:-null}" "${CTRL_DEV:-}" \
    "${CTRL_SRC:-}" "$CTRL_MBPS" "$LINKS" <<'EOF'
import json, shlex, sys
from fractions import Fraction
(path, tools, store, segments, reserve, threads, limit, ch, addr, cmd, simulated, foreign, egress, reads, dev, src, mbps,
 specs) = sys.argv[1:]
links = [] if not dev else [{"name": "control", "to": "outside", "via": f"{dev} {src}: the node's own NIC (ssh, research run)",
                             "mbps": int(mbps) if mbps.strip().isdigit() and int(mbps) > 0 else None}]
for spec in filter(None, specs.splitlines()):
    name, _, rest = spec.partition("=")
    to, _, via = rest.partition("@")
    if not (name and to and via):
        sys.exit(f"demo.sh: --link is NAME=TO@VIA, not {spec!r}")
    links.append({"name": name, "to": to, "via": via})
doc = {"tools": json.loads(tools),
       "wipe": {"store": store, "segments": json.loads(segments), "reserve_bytes": int(reserve) << 30,
                "streamer": "native", "threads": json.loads(threads), "limit_ns": int(limit),
                "foreign_bytes": foreign if foreign == "fM" else int(Fraction(foreign) * 10 ** 9),
                "placement": "simulated" if simulated == "1" else None, "egress_bits": json.loads(egress),
                "reads": json.loads(reads), "links": links}}
if ch:
    doc |= {"challenger_host": ch, "challenger": {"wipe": {"command": shlex.split(cmd), "address": addr}}}
open(path, "w").write(json.dumps(doc, indent=1, sort_keys=True))
EOF
  cp "$S/periods/p0/period.json" "$O/period.json"
  export DEMO_PERIOD=$S/periods/p0 DEMO_NODE=$NODE DEMO_STORE=$STORE DEMO_GPU=$GPU DEMO_EXPECT=$EXPECT
  if [ "$STORE" = cuda ] && [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease >/dev/null 2>&1; then
    GPU_LEASE_WHO=memory-accounting gpu-lease 1 --wait --max-min 60 -- bash "$SELF" leased
  else
    bash "$SELF" leased
  fi
  exit $?
fi

if [ "$DEMO_STORE" = cuda ]; then
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  nvidia-smi -q -d MEMORY,CLOCK,PERFORMANCE >> "$O/gpu.txt" 2>&1 || true
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} | head -1)
  case "$DEMO_GPU:$NAME" in any:*|*:*"$DEMO_GPU"*) ;; *) echo "REFUSED: the GPU is '$NAME', not an $DEMO_GPU"; exit 3 ;; esac
fi
rc=0
uv run --no-sync python -m verity.core.protocols.pous.wipe_period --period "$DEMO_PERIOD" --out "$O" --node "$DEMO_NODE" \
  --expect "$DEMO_EXPECT" > "$O/summary.json" || rc=$?
cat "$O/summary.json"
exit $rc
