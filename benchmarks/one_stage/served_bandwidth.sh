#!/usr/bin/env bash
# The bandwidth audit's piece of `served_verdicts.sh`, run in the tree's root as `bash -c CMD _ SERVE_DIR WINDOW DIR`:
#
#   BANDWIDTH_CMD='bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'
#
# One remote bandwidth session over the node's registered store, PoUS's P2 C* (BW_STORE, mem69-v2's encoded store,
# only ever read), each pass one engine step's weight reads from it through kernels' staging decode on the card, its
# receipts sketched from the rows the decode reads (`benchmarks/pous/p2_v1/bandwidth.py session`), under a gpu-lease of
# its own (BW_LEASE_MIN minutes, default 40: each pass reads the whole store from disk, about 140 GB at mem69-v2, which
# takes the H200 node about 10 minutes, and the gpu-lease ends the session at its limit). The verifier is the auditor's,
# never a process on this node: the `p2_party --listen` that the auditor host's gate starts for BW_PARTY (the ssh command
# its key is forced to), reached at BW_PARTY_HOST; it holds its own registration, which the store's must be
# (BW_REGISTRATION, the tracked one). The
# session's record is DIR/session, and its answer, held to the verifier's own record of it, is DIR/answer.json in the
# core interface's form (`served_bandwidth.py answer`): the credited reads q_bytes at eps and δ, and γ against
# BW_R_BYTES_PER_S (default an H200's specified 4.8 TB/s). The session runs after the window was served, beside it, and
# is not interleaved with the engine's reads, since serving is not hooked: the answer says so (an accept in its profile,
# which has no reason), and DIR/served.json names the window, the session's times and its card.
#
# The layout is the store's (`p2_v1/store.py layout`), held to the store's setup.json; the lib is kernels' build
# (`benchmarks/torch_gpu/p2_stage/build.sh` at this commit, its source.txt and MANIFEST.sha256 kept in DIR), or BW_LIB, a
# built lib/ directory. Both are written to scratch under DIR, which is removed: DIR keeps only records. A toy on the
# CPU: BW_STORE, BW_LAYOUT, BW_REGISTRATION and BW_NODE name it, BW_DEVICE=cpu, BW_REF_DECODE=1 and BW_LEASE_MIN=0
# (no lease, no lib). Exits 0 iff the session's answer is an accept; DIR/rc.txt has each step's exit code.
set -uo pipefail
SERVE=$1 K=$2 O=$3
mkdir -p "$O"
P2=benchmarks/pous/p2_v1
PY=${PY:-/workspace/cp/venv312/bin/python}
STORE=${BW_STORE:-/workspace/research/pous-store/mem69-v2}
NODE=${BW_NODE:-vy-dc-h200-2}
REG=${BW_REGISTRATION:-$P2/bandwidth-mem69-v2.json}
PARTY=${BW_PARTY:-ssh -i $HOME/.ssh/vy-auditor2-bw -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 research@10.96.0.36 bw_party}
PARTY_HOST=${BW_PARTY_HOST:-10.96.0.36}
WINDOWS=${BW_WINDOWS:-2} PASSES=${BW_PASSES:-1} SAMPLES=${BW_SAMPLES:-1024} WINDOW_S=${BW_WINDOW_S:-1}
DEADLINE_S=${BW_DEADLINE_S:-1} DEVICE=${BW_DEVICE:-cuda} LEASE_MIN=${BW_LEASE_MIN:-40}
R=${BW_R_BYTES_PER_S:-4800000000000}
S=$O/scratch
rm -rf "$S"; mkdir -p "$S"
trap 'rm -rf "$S"' EXIT
export PYTHONPATH=$PWD:$PWD/benchmarks/pous${PYTHONPATH:+:$PYTHONPATH}
: > "$O/rc.txt"
step() { echo "$1 rc $2 at $(date -u +%H:%M:%SZ)" >> "$O/rc.txt"; }

LAYOUT=${BW_LAYOUT:-}
if [ -z "$LAYOUT" ]; then
  LAYOUT=$S/layout.json
  "$PY" "$P2/store.py" layout --out "$LAYOUT" > "$O/layout.log" 2>&1 \
    && "$PY" - "$STORE/setup.json" "$LAYOUT" >> "$O/layout.log" 2>&1 <<'EOF'
import hashlib, json, sys
want, got = json.load(open(sys.argv[1]))["layout_sha512"], hashlib.sha512(open(sys.argv[2], "rb").read()).hexdigest()
sys.exit(0 if want == got else f"the layout's sha512 {got[:16]} is not the store's {want[:16]}")
EOF
  step layout $?
fi
LIB=${BW_LIB:-unused}
if [ "$DEVICE" != cpu ] && [ -z "${BW_LIB:-}" ]; then
  bash benchmarks/torch_gpu/p2_stage/build.sh "$S/lib" > "$O/build.log" 2>&1 \
    && (cd "$S/lib" && sha256sum -c --quiet MANIFEST.sha256)
  step lib $?
  cp "$S/lib/source.txt" "$S/lib/toolchain.txt" "$S/lib/MANIFEST.sha256" "$O/" 2>/dev/null
  LIB=$S/lib/lib
fi
grep -qE " rc [1-9]" "$O/rc.txt" && { cat "$O/rc.txt"; tail -20 "$O/layout.log" "$O/build.log" 2>/dev/null; }

args=(session --store "$STORE" --layout "$LAYOUT" --lib "$LIB" --node "$NODE" --registration "$REG"
      --party "$PARTY" --party-host "$PARTY_HOST" --windows "$WINDOWS" --passes "$PASSES" --samples "$SAMPLES"
      --window-s "$WINDOW_S" --deadline-s "$DEADLINE_S" --device "$DEVICE" --out "$O/session" --expect accept)
[ "${BW_REF_DECODE:-0}" = 1 ] && args+=(--ref-decode)
[ -n "${BW_WORKERS:-}" ] && args+=(--workers "$BW_WORKERS")
[ "${BW_CHEAT:-}" = cache ] && args+=(--cache)
export PY P2 O
run() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$O/started.txt"
  nvidia-smi --query-gpu=index,uuid,name,memory.total ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} --format=csv,noheader \
    > "$O/gpu.txt" 2>/dev/null
  "$PY" "$P2/bandwidth.py" "$@" > "$O/session.log" 2> "$O/session.err"
  local rc=$?
  date -u +%Y-%m-%dT%H:%M:%SZ > "$O/ended.txt"
  return $rc
}
if grep -qE " rc [1-9]" "$O/rc.txt"; then
  :
elif [ "$LEASE_MIN" = 0 ]; then
  run "${args[@]}"; step session $?
else
  export -f run
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-bandwidth-piece} env -u GPU_LEASE_TIMED gpu-lease 1 --wait --max-min "$LEASE_MIN" \
    -- bash -c 'run "$@"' _ "${args[@]}"
  step session $?
fi
tail -n 1 "$O/session.log" 2>/dev/null
"$PY" benchmarks/one_stage/served_bandwidth.py answer --session "$O/session/session.json" --serve-dir "$SERVE" \
  --window "$K" --out "$O" --r-bytes-per-s "$R" --started "$(cat "$O/started.txt" 2>/dev/null)" \
  --ended "$(cat "$O/ended.txt" 2>/dev/null)" --gpu "$O/gpu.txt"
rc=$?
step answer $rc
exit $rc
