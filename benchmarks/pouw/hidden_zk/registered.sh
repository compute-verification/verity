#!/usr/bin/env bash
# Registered GEMM-coordinate sessions under C-Flock --zk on CPUs, the prover column's check (F6): per K in HZ_K, generate N
# instances of the sm80 bf16 gemm-coordinate template, stage them with `verity_flock.circuit` (timed apart: the template
# shape's staging cost), and prove one loopback --zk session with job.sh's proving build at HZ_THREADS threads. Run from the
# source tree's root (`research run --cwd source --declared-output 'out/*'`). Results: out/k<K>/{generate,stage,serve,
# prove}.time, prove.out, stage.out, registered.json, and out/registered.json.
# A block is at most 2^27 bits (Stmt.InRange), so K = 32768 (k_log 27) is the largest stageable GEMM coordinate.
# env: HZ_K="16384 32768"  HZ_N=16  HZ_MEM_GB  HZ_RUNS=2  HZ_WARM=1  HZ_THREADS=32  HZ_CACHE  FLOCK_PROVE / HZ_PYTHON (prebuilt)
set -uo pipefail
export HZ_CONFIGS=registered FLOCK_VERIFY=${FLOCK_VERIFY:-/bin/true}
source benchmarks/pouw/hidden_zk/job.sh
setup
export RAYON_NUM_THREADS=${HZ_THREADS:-32}
SEM=sm80.mma.m16n8k16.bf16
for k in ${HZ_K:-16384 32768}; do
  D=$W/k$k O=$OUT/k$k
  mkdir -p "$D" "$O"
  log "k$k: generate $N instances"
  $TIME "$O/generate.time" $PY -m verity_numerical.bench.generate --template gemm-coordinate --param K=$k --param semantics=$SEM \
    --n "$N" --seed 1 --out "$D/is" > "$O/generate.out" 2>&1 || { log "k$k: generate FAILED: $(tail -2 "$O/generate.out")"; continue; }
  log "k$k: stage"
  $TIME "$O/stage.time" bash -c "$(vlimit $STAGE_GB) exec $PY -m verity_flock.circuit \
    --input-set $D/is/sets/gemm-coordinate-ampere-bf16-k$k --n $N --out $D" > "$O/stage.out" 2> "$O/stage.err" \
    || { log "k$k: stage FAILED: $(tail -2 "$O/stage.err")"; rm -rf "$D"; continue; }
  grep '^{' "$O/stage.out" | tail -1 | $PY -c "import json, sys; print(json.load(sys.stdin)['pin'])" > "$D/pin.txt"
  ln -s "inst-$N.bin" "$D/inst.bin" && ln -s "pub-$N.bin" "$D/pub.bin"
  grep -m1 '^META ' "$D/circuit.txt" | $PY -c "import json, sys; m = json.loads(sys.stdin.read()[5:]); \
print(json.dumps({k: m.get(k) for k in ('k_log', 'unit_log', 'per_vu', 'vus_per_block')}))" > "$O/meta.json"
  stat -L -c %s "$D/circuit.txt" > "$O/circuit.bytes"
  log "k$k: staged $(cat "$O/stage.time"), $(cat "$O/circuit.bytes") B, $(cat "$O/meta.json")"
  log "k$k: --zk session ($WARM warm + $RUNS timed, $RAYON_NUM_THREADS threads)"
  session "$D" "$O" || log "k$k: --zk session FAILED: $(tail -2 "$O/prove.err")"
  $PY -c "import json, sys; sys.path.insert(0, '$J'); sys.argv[1:] = ['$OUT']; import summarize as S; from pathlib import Path; o = Path('$O'); \
t = lambda n: json.loads((o / n).read_text()) if (o / n).exists() else None; \
s = {'K': $k, 'n': $N, 'threads': $RAYON_NUM_THREADS, 'circuit_bytes': int((o / 'circuit.bytes').read_text()), **t('meta.json'), \
'generate': t('generate.time'), 'stage': t('stage.time'), 'zk': S.session(o)}; \
(o / 'registered.json').write_text(json.dumps(s, indent=1)); print(json.dumps(s))" | tee -a "$OUT/job.log"
  rm -rf "$D"
done
$PY -c "import json; from pathlib import Path; o = Path('$OUT'); \
json.dump([json.loads(p.read_text()) for p in sorted(o.glob('k*/registered.json'))], open(o / 'registered.json', 'w'), indent=1)"
log "registered: done"
