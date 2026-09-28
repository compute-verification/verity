#!/usr/bin/env bash
# An A-route-a Table 2 cell (bench.cell's driver drivers/a_route_a.py): route (a) (A-GKR on the GPU + Flock's keyed-BLAKE3 rows
# on the host CPU + the sigma link) swept over batch sizes, every timed session live against a verifier on a separate pod of the
# same datacenter.  Runs from the shipped tree (research run --source . --cwd source); KEY=VALUE arguments become the environment.
#
#   ROLE=verifier  REL=<rel>+blake3 [SET=<set tarball under inputs/>] [POINTS="1024 2048 4096"] [RUNS=5] [PORT=7200]
#                  [ECHO_PORT=7201] [FIRST_SIZE_TIMEOUT=5400] [PER_SIZE_TIMEOUT=1800] [OPERATOR=<lane>]
#       flock b684b12 + both Flock patches + this tree's live crate (release) and the flock-link selftest at 8 VUs of the set's K;
#       then an echo on ECHO_PORT for the prover's RTT probe, and size by size in POINTS order (capped at the set), the verifier's own
#       statement of that size (leaf digests and prime
#       commitment from the set: tools/cell.py statement --verifier-side) and one `flock-link cell-serve` on PORT serving RUNS
#       sessions (it serves the prime coins too).  A size that ends short of RUNS sessions ends the run (the prover stopped).
#   ROLE=prover    REL=<rel>+blake3 VERIFIER=host:port RTT_TARGET=host:echoport [SET=...] [POINTS=...] [RUNS=5] [GATE=1]
#                  [LOOPBACK=1] [RTT_WAIT=1800] [OPERATOR=<lane>]
#       pod_bootstrap.sh, flock-link and verity-gkr-verify built from this tree, their tests; with GATE=1 the correctness gate at
#       the first size (loopback verifier: honest accepted and admitted by tools/cell_gate.py except non_producer, a stale prime
#       state and a Fiat-Shamir prime prover refused, the statement claimed as another relation refused); the RTT probe (it waits
#       up to RTT_WAIT s for the verifier's echo); then per
#       size in POINTS (capped at the set): the statement (the serving commit timed), one warm-up session against a verifier on
#       127.0.0.1 (the loopback probe), RUNS timed sessions against VERIFIER; the sweep stops when two doublings add < 2 % or a size
#       fails.  result.json = the plateau (tools/cell.py finish), outputs.json = the other sizes.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); RD=${RESEARCH_RUN_DIR:?}; W=/workspace; O=$RD/out; mkdir -p $O $W/bin
command -v python3 >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 >/dev/null; }
ROLE=${ROLE:-prover}; REL=${REL:?}; RUNS=${RUNS:-5}
say() { echo "##### $(date -u +%H:%M:%SZ) $*"; }
# the cgroup's CPU quota, rounded up: v2 cpu.max, else v1 cfs_quota_us / cfs_period_us (US-NC-1 L40S pods: 13.6 cores of 128)
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
if [ -r /sys/fs/cgroup/cpu.max ]; then read -r Q PER < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1; fi
NT=$(( Q > 0 ? (Q + PER - 1) / PER : $(nproc) )); [ "$NT" -lt 1 ] && NT=1; export RAYON_NUM_THREADS=$NT
export PYTHONPATH="$REPO:$REPO/benchmarks/numerical/python:$REPO/tools/research/src:$REPO:$REPO/archive/gkr"
export PATH="$HOME/.cargo/bin:$PATH"
F=$W/flock-link/flock; B=$W/bin/flock-link-live; VB=$W/bin/verity-gkr-verify-live

build_flock() {
  bash verity/ml/flock/pod/00-setup-cpu.sh > $RD/flock-setup.log 2>&1
  source $HOME/.cargo/env
  # the live crate on main needs both Flock patches (verity/ml/flock/pod/20-pure.sh's recipe)
  ( cd $F && git checkout -q b684b12 && git checkout -q -- . && git clean -qfd crates && git apply $REPO/verity/ml/flock/flock-link-b684b12.patch \
    && git apply $REPO/verity/ml/flock/flock-gpu-link-b684b12.patch && cp -r $REPO/verity/ml/flock/live crates/flock-live \
    && { grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml; } \
    && cargo build --release -p flock-live --bin flock-link -j $NT 2>&1 | grep -E '^error|-->|Finished' | head -20 ) || return 1
  cp ${CARGO_TARGET_DIR:-$F/target}/release/flock-link $B || return 1
}

# the sizes of the sweep: POINTS, capped at the input set's size (a set is not recycled)
SET_DIR=""; CAP=""
if [ -n "${SET:-}" ]; then
  S=$SET; [ -e "$S" ] || S=$RD/inputs/$SET
  if [ -f "$S" ]; then mkdir -p $RD/set && tar -xf "$S" -C $RD/set && S=$(dirname "$(find $RD/set -name manifest.json | head -1)"); fi
  SET_DIR=$S
  CAP=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]+'/manifest.json'))['n'])" "$S") || { echo "no input set at $S"; exit 2; }
  K=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]+'/manifest.json'))['relation']['statics']['K'])" "$S")
else
  K=1536
fi
SIZES=""
for n in ${POINTS:-1024 2048 4096}; do [ -z "$CAP" ] || [ "$n" -le "$CAP" ] && SIZES="$SIZES $n"; done
SIZES=${SIZES# }
say "$ROLE: $REL K=$K sizes [$SIZES]${CAP:+ (set of $CAP)} runs $RUNS, $NT vCPU of $(nproc), DC ${RUNPOD_DC_ID:-?}"

if [ "$ROLE" = verifier ]; then
  PORT=${PORT:-7200}
  say "setup: flock-link, python (numpy, blake3)"
  build_flock || { tail -20 $RD/flock-setup.log; echo "BUILD FAILED"; exit 1; }
  PY=$W/venv-v/bin/python
  if ! $PY -c "import numpy, blake3" 2>/dev/null; then
    command -v uv >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"; uv venv $W/venv-v --python 3.12 >/dev/null 2>&1; uv pip install --python $PY numpy blake3 2>&1 | tail -1
  fi
  sha256sum $B | tee $O/binary.sha256; lscpu | grep 'Model name'
  $B selftest --vus 8 --k $K > $O/selftest-8.txt 2>&1
  echo "flock-link selftest (8 VUs, K=$K): $(grep -c '"pass":true' $O/selftest-8.txt) passed, $(grep -c '"pass":false' $O/selftest-8.txt) failed; $(grep SELFTEST $O/selftest-8.txt | cut -c1-200)"
  [ -n "$SET_DIR" ] || { echo "the verifier builds its own statement from an input set (SET=); the frozen set needs its bench-instances build"; exit 2; }
  # an echo on ECHO_PORT for the prover's open-connection RTT probe (a TCP connect over RunPod's global network costs ~4x a round
  # trip).  It starts once setup is done: the prover's probe waits for it, so its sessions (a 60 s connect retry) find the verifier up
  python3 -c "
import socket, threading
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', ${ECHO_PORT:-7201})); s.listen(16)
def echo(c):
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with c:
        while (d := c.recv(256)):
            c.sendall(d)
while True:
    c, _ = s.accept(); threading.Thread(target=echo, args=(c,), daemon=True).start()
" > $O/echo.log 2>&1 &
  ECHO_PID=$!
  first=1
  for n in $SIZES; do
    # each size's statement just before serving it: the prover builds the same statement and runs a warm-up before its first
    # remote session of that size, so the verifier is up first
    V=$W/vstmt/k$K-c$n
    $PY archive/gkr/tools/cell.py statement --cell $V --input-set $SET_DIR --vus $n --verifier-side | tail -1
    mkdir -p $O/statements; cp $V/cell.json $O/statements/cell-$n.json; sha256sum $V/leaf_digests.bin >> $O/statements/digests.sha256
    PC=$(python3 -c "import json;print(json.load(open('$V/cell.json'))['prime_commitment'])")
    # the first size waits for the prover's setup and gate too
    T=$([ $first = 1 ] && echo ${FIRST_SIZE_TIMEOUT:-5400} || echo ${PER_SIZE_TIMEOUT:-1800}); first=0
    say "serve $n VUs on :$PORT ($RUNS sessions, at most $T s)"
    timeout $T $B cell-serve --listen 0.0.0.0:$PORT --out $RD/sessions-$n --vus $n --k $K --digests $V/leaf_digests.bin \
      --prime-commitment $PC --operator ${OPERATOR:-agkr-route-a} --sessions $RUNS > $O/serve-$n.log 2>&1
    got=$(cat $RD/sessions-$n/index.jsonl 2>/dev/null | wc -l)
    echo "$n: $got sessions, accepted $(grep -c '"accepted":true' $RD/sessions-$n/index.jsonl 2>/dev/null)"
    [ "$got" -ge "$RUNS" ] || { say "fewer than $RUNS sessions at $n VUs: the prover stopped its sweep"; break; }
  done
  python3 - "$RD" <<'EOF'
import json, sys
from pathlib import Path
rd = Path(sys.argv[1])
files = [str(p.relative_to(rd)) for p in sorted(rd.glob("sessions-*/**/*")) if p.is_file()]
files += [str(p.relative_to(rd)) for p in sorted((rd / "out").rglob("*")) if p.is_file()]
(rd / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": [
    {"name": "verifier-records", "kind": "run-files/v1", "files": files,
     "meta": {"label": "A-route-a live verifier: every session's record, the Flock proofs kept beside it, the statements it served"}}]}))
print(f"outputs.json: {len(files)} files")
EOF
  kill $ECHO_PID 2>/dev/null
  rm -rf $RD/set
  exit 0
fi

# ---- prover -----------------------------------------------------------------------------------------------------------------
VERIFIER=${VERIFIER:?}; RTT_TARGET=${RTT_TARGET:?}
say "bootstrap"
VERIFIER= SRC=$REPO BENCH_INSTANCES=$([ -z "$SET_DIR" ] && echo 1) HEALTH=0 bash archive/direct/ligero/pod_bootstrap.sh > $RD/bootstrap.log 2>&1
echo "bootstrap rc=$?"; tail -n 4 $RD/bootstrap.log
source $W/env.sh; cd "$REPO"
export PYTHONPATH="$PYTHONPATH:$REPO/archive/gkr" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_MMAP_MAX_=0 \
  MALLOC_TRIM_THRESHOLD_=1000000000000 RAYON_NUM_THREADS=$NT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee $O/gpu.txt; lscpu | grep 'Model name'
say "build flock-link + verity-gkr-verify"
build_flock || { tail -20 $RD/flock-setup.log; echo "BUILD FAILED"; exit 1; }
( cd archive/gkr/verifier && cargo build --release -j $NT 2>&1 | grep -E '^error|Finished' | head -3 \
  && cp ${CARGO_TARGET_DIR:-target}/release/verity-gkr-verify $VB && cargo test --release -j $NT 2>&1 | grep -E 'test result|FAILED' ) | tee $O/verifier-build.txt
[ -x $VB ] || { echo "verity-gkr-verify build failed"; exit 1; }
sha256sum $B $VB | tee $O/binaries.sha256
$PY -m pytest -q archive/gkr/tests 2>&1 | tail -3 | tee $O/pytest.txt
INST=(); [ -n "$SET_DIR" ] && INST=(--input-set $SET_DIR) || INST=(--instances $W/bench-instances/v1)
pc() { $PY -c "import json;print(json.load(open('$1/cell.json'))['prime_commitment'])"; }
loopback() {  # $1 = cell dir, $2 = vus, $3 = out dir: a verifier on 127.0.0.1:7300 (SP)
  local t
  $B cell-serve --listen 127.0.0.1:7300 --out $3 --vus $2 --k $K --digests $1/leaf_digests.bin --prime-commitment $(pc $1) \
    --operator loopback-selftest > $3.log 2>&1 &
  SP=$!; for t in $(seq 240); do grep -q SERVING $3.log 2>/dev/null && break; sleep 0.5; done
}
stop_loopback() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }
statement() {  # $1 = vus -> $W/cell/k$K-c$1, built in this run (its serving commit is this run's commit.seconds)
  local C=$W/cell/k$K-c$1
  rm -rf $C
  $PY archive/gkr/tools/cell.py statement --cell $C "${INST[@]}" --vus $1 --flock $B > $O/statement-$1.log 2>&1 || return 1
  mkdir -p $O/statement-$1; cp -r $C/statement/. $O/statement-$1/; cp $C/cell.json $C/leaf_digests.bin $O/statement-$1/
}

if [ "${GATE:-1}" = 1 ]; then
  n=${SIZES%% *}; C=$W/cell/k$K-c$n; G=$O/gate; mkdir -p $G
  say "gate at $n VUs (loopback verifier)"
  statement $n || { tail $O/statement-$n.log; echo "GATE: statement failed"; exit 4; }
  loopback $C $n $G/sessions
  timeout 3600 $PY archive/gkr/tools/cell.py prove --cell $C --flock $B --verifier 127.0.0.1:7300 --sessions 1 --warmup 0 --rust $VB \
    --threads $NT > $G/honest.txt 2>&1; echo "honest rc=$?" | tee -a $G/rc.txt
  mkdir -p $G/honest; cp $C/s0/proof.bin $C/s0/times.json $C/s0/link.txt $C/s0/prime_coins.txt $G/honest/ 2>/dev/null
  for neg in stale fs; do
    NC=$W/cell/k$K-neg-$neg; rm -rf $NC; mkdir -p $NC; cp -r $C/statement $C/cell.json $C/leaf_digests.bin $C/operands.bin $C/y.npy $NC/
    X=$([ $neg = stale ] && echo "--neg-stale-round 5" || echo "--prime-coins fs")
    timeout 1800 $PY archive/gkr/tools/cell.py prove --cell $NC --flock $B --verifier 127.0.0.1:7300 --sessions 1 --warmup 0 --rust $VB \
      --threads $NT $X > $G/neg-$neg.txt 2>&1; echo "neg-$neg rc=$? (must be non-zero)" | tee -a $G/rc.txt; tail -2 $G/neg-$neg.txt
  done
  sleep 2; stop_loopback
  # the verifier's records in session order: honest, stale, fs (refused at Hello).  The gate reads the record, never the prover's view:
  # the stale prover's own replay is honest by construction, the record's replay must reject it
  mapfile -t SES < <(ls -d $G/sessions/l* | sort)
  for pair in "honest:${SES[0]:-}:$G/honest/proof.bin" "stale:${SES[1]:-}:$W/cell/k$K-neg-stale/s0/proof.bin"; do
    IFS=: read -r name ses proof <<<"$pair"
    [ -n "$ses" ] || { echo "gate $name: no session record" | tee -a $G/rc.txt; continue; }
    $PY archive/gkr/tools/cell_gate.py --session $ses --sigma $C/statement/sigma.txt --producer ${OPERATOR:-agkr-route-a} \
      --producer loopback-selftest --out $G/admit-$name --rust $VB --statement $C/statement --proof $proof --vus $n --relation $REL \
      --threads $NT --flock $B --digests $C/leaf_digests.bin --prime-commitment $(pc $C) > $G/admit-$name.txt 2>&1
    $PY -c "import json;d=json.load(open('$G/admit-$name/gate.json'));print('gate $name:', {k: v['ok'] for k, v in d['checks'].items()}, d['checks'].get('prime', {}).get('detail', '')[:160])" | tee -a $G/rc.txt
  done
  OTHER=$([ "$REL" = bf16-ampere+blake3 ] && echo bf16-ampere-k2048+blake3 || echo bf16-ampere+blake3)
  $VB verify --dir $G/admit-honest/statement --proof $G/honest/proof.bin --vus $n --relation $OTHER --require-live-coins --json $G/wrong-relation.json \
    > /dev/null 2>&1; echo "claimed as $OTHER: rc=$? $($PY -c "import json;print(json.load(open('$G/wrong-relation.json'))['error'])")" | tee -a $G/rc.txt
fi

say "RTT probe to $RTT_TARGET"
# RTT_TARGET: the verifier's echo (ECHO_PORT) gives the round trip on one open connection, the way the session's rounds travel; else
# (e.g. its sshd) the median of 20 TCP connects, a failed one retried (a first connect over global networking can time out).  The
# echo is up once the verifier's setup is done, so a refused or timed-out connect is retried for RTT_WAIT s (default 1800)
read -r RTT RTT_KIND < <($PY -c "
import socket, statistics as S, sys, time
h, p = '$RTT_TARGET'.rsplit(':', 1); p = int(p); MAGIC = b'verity-rtt-probe\\n'
def open_rtt():
    c = socket.create_connection((h, p), timeout=5); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); c.settimeout(3)
    with c:
        c.sendall(MAGIC); got = b''
        while len(got) < len(MAGIC) and (d := c.recv(len(MAGIC) - len(got))):
            got += d
        if got != MAGIC:
            return None
        t = []
        for _ in range(50):
            a = time.perf_counter(); c.sendall(b'p' * 8); r = b''
            while len(r) < 8:
                r += c.recv(8 - len(r))
            t.append((time.perf_counter() - a) * 1e3)
        return t
t = None; end = time.monotonic() + ${RTT_WAIT:-1800}
while True:
    try:
        t = open_rtt()
        break
    except OSError:
        if time.monotonic() > end:
            break
        time.sleep(2)
if t:
    print(round(S.median(t), 3), 'open'); sys.exit()
t = []
for _ in range(40):
    try:
        a = time.perf_counter(); c = socket.create_connection((h, p), timeout=5); t.append((time.perf_counter() - a) * 1e3); c.close()
    except OSError:
        pass
    time.sleep(0.05)
    if len(t) == 20:
        break
print(round(S.median(t), 3), 'connect') if len(t) == 20 else sys.exit('only %d of 40 connects' % len(t))
")
[ -n "${RTT:-}" ] || { echo "RTT probe failed"; exit 3; }
if [ "$RTT_KIND" = open ]; then
  RTT_METHOD="median of 50 ping-pongs of 8 bytes on one open TCP connection (TCP_NODELAY) from the prover pod to the verifier pod's echo ($RTT_TARGET), measured in this run before the sweep"
else
  RTT_METHOD="median of 20 TCP connects from the prover pod to the verifier pod ($RTT_TARGET), measured in this run before the sweep"
fi
echo "RTT $RTT ms ($RTT_KIND)" | tee $O/rtt.txt
i=0; TPS=()
for n in $SIZES; do
  P=$O/points/p$i-$n; mkdir -p $P; C=$W/cell/k$K-c$n
  status() { $PY -c "import json,sys;json.dump({'vus': $n, 'rc': int(sys.argv[1]), 'why': sys.argv[2], 'verifier': '$VERIFIER'}, open('$P/status.json','w'))" "$1" "$2"; }
  say "size $n: statement"
  statement $n || { status 5 "statement: $(tail -1 $O/statement-$n.log | cut -c1-200)"; break; }
  loopback $C $n $P/loopback-sessions
  say "size $n: 1 warm-up (loopback) + $RUNS sessions against $VERIFIER"
  $PY archive/gkr/tools/cell.py prove --cell $C --flock $B --verifier $VERIFIER --warmup-verifier 127.0.0.1:7300 --sessions $RUNS --warmup 1 \
    --rust $VB --threads $NT --result $P/result.json --rtt-ms $RTT --rtt-method "$RTT_METHOD" > $P/prove.txt 2>&1
  rc=$?; stop_loopback
  grep -E '^session|^result|CELL-PROVE|Error|error' $P/prove.txt | tail -8 | cut -c1-300
  why=$(grep -E 'OutOfMemory|out of memory|rejected|Error|exited' $P/prove.txt | tail -1 | cut -c1-200)
  [ $rc -eq 0 ] && why=ok || why=${why:-"exit $rc without an error line (prove.txt)"}
  status $rc "$why"
  cp $C/s0/times.json $P/loopback-times.json 2>/dev/null; cp $C/contention.json $P/ 2>/dev/null
  mkdir -p $O/cell-$n; for s in $C/s*; do d=$O/cell-$n/$(basename $s); mkdir -p $d; cp $s/*.json $s/*.txt $s/*.jsonl $s/*.stderr $s/proof.bin $d/ 2>/dev/null; done
  [ $rc -eq 0 ] || break
  TPS+=($($PY -c "import json;m={x['name']:x['value'] for x in json.load(open('$P/result.json'))['measurements']};print($n/m['t.total'])"))
  echo "size $n: ${TPS[-1]} VU/s"
  k=${#TPS[@]}; if [ $k -ge 3 ] && $PY -c "import sys;sys.exit(0 if ${TPS[-1]} < 1.02*${TPS[-3]} else 1)"; then say "two doublings added < 2 %"; break; fi
  i=$((i + 1))
done
say "the cell's result"
$PY archive/gkr/tools/cell.py finish --points $O/points/p* --verifier $VERIFIER --operator "${OPERATOR:-}" --out $RD/result.json \
  --others $RD/outputs.json | tail -n 25
rc=$?
rm -rf $RD/set          # the registered input set (SET_ART), not a result of this run
exit $rc
