#!/usr/bin/env bash
# Uncut GEMM-coordinate sessions under C-Flock --zk on CPUs, beside registered.sh's cut ones: per K in HZ_K, generate N
# instances of the sm80 bf16 gemm-coordinate template, stage them with uncut_stage.py (the whole coordinate one unit: one
# input group, no tail stage, no wired accumulator, the form ProvedScope covers), prove one loopback --zk session with
# job.sh's proving build at HZ_THREADS threads (the Rust verifier's SESSION verdict, its proved-scope mirror included), then
# verify the last session with the Lean `verify --zk`. Run from the source tree's root (`research run --cwd source
# --declared-output 'out/*'`). Results: out/k<K>/{generate,stage,serve,prove,lean}.time, stage.out (the unit's rows, nnz,
# k_log), prove.out, serve.log, lean.out, uncut.json, and out/uncut.json.
# env: HZ_K="2048 16384"  HZ_N=16  HZ_MEM_GB  HZ_RUNS=2  HZ_WARM=1  HZ_THREADS=32  HZ_CACHE  FLOCK_PROVE / FLOCK_VERIFY / HZ_PYTHON
set -uo pipefail
export HZ_CONFIGS=uncut
source benchmarks/pouw/hidden_zk/job.sh
setup
export RAYON_NUM_THREADS=${HZ_THREADS:-32}
SEM=sm80.mma.m16n8k16.bf16
for k in ${HZ_K:-2048 16384}; do
  D=$W/k$k O=$OUT/k$k
  mkdir -p "$D" "$O"
  log "k$k: generate $N instances"
  $TIME "$O/generate.time" $PY -m verity_numerical.bench.generate --template gemm-coordinate --param K=$k --param semantics=$SEM \
    --n "$N" --seed 1 --out "$D/is" > "$O/generate.out" 2>&1 || { log "k$k: generate FAILED: $(tail -2 "$O/generate.out")"; continue; }
  log "k$k: stage (uncut)"
  $TIME "$O/stage.time" bash -c "$(vlimit $STAGE_GB) exec $PY $J/uncut_stage.py $D/is/sets/gemm-coordinate-ampere-bf16-k$k $N $D" \
    > "$O/stage.out" 2> "$O/stage.err" || { log "k$k: stage FAILED: $(tail -2 "$O/stage.err")"; rm -rf "$D"; continue; }
  grep '^{' "$O/stage.out" | tail -1 | $PY -c "import json, sys; print(json.load(sys.stdin)['pin'])" > "$D/pin.txt"
  ln -s "inst-$N.bin" "$D/inst.bin" && ln -s "pub-$N.bin" "$D/pub.bin"
  log "k$k: staged $(cat "$O/stage.time"), $(grep '^{' "$O/stage.out" | tail -1 | cut -c1-600)"
  log "k$k: --zk session ($WARM warm + $RUNS timed, $RAYON_NUM_THREADS threads)"
  session "$D" "$O" || log "k$k: --zk session FAILED: $(tail -2 "$O/prove.err")"
  log "k$k: rust verdicts: $(grep '^SESSION' "$O/serve.log" | $PY -c "import json, sys; print([(j['accepted'], j['why']) for j in (json.loads(l[8:]) for l in sys.stdin)])")"
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tail -1)
  if [ -n "$s" ]; then
    $TIME "$O/lean.time" bash -c "$(vlimit $STAGE_GB) exec $LEAN verify --zk --circuit $D/circuit.txt --statement verity/flock-circuit \
      --public $D/pub.bin --session $s" > "$O/lean.out" 2> "$O/lean.err"
    log "k$k: lean verify: $(cat "$O/lean.time"): $(grep '^VERDICT' "$O/lean.out" | cut -c1-300) $(tail -1 "$O/lean.err" | cut -c1-200)"
  fi
  $PY -c "import json, sys; sys.path.insert(0, '$J'); sys.argv[1:] = ['$OUT']; import summarize as S; from pathlib import Path; o = Path('$O'); \
t = lambda n: json.loads((o / n).read_text()) if (o / n).exists() else None; \
st = [json.loads(l) for l in (o / 'stage.out').read_text().splitlines() if l.startswith('{')]; \
v = [json.loads(l[8:]) for l in (o / 'lean.out').read_text().splitlines() if l.startswith('VERDICT ')] if (o / 'lean.out').exists() else []; \
r = [json.loads(l[8:]) for l in (o / 'serve.log').read_text().splitlines() if l.startswith('SESSION')] if (o / 'serve.log').exists() else []; \
s = {'K': $k, 'n': $N, 'threads': $RAYON_NUM_THREADS, 'unit': st[-1] if st else None, 'generate': t('generate.time'), \
'stage': t('stage.time'), 'zk': S.session(o), 'rust_sessions': [(j['accepted'], j['why']) for j in r], \
'lean': {'time': t('lean.time'), 'verdict': v[0] if v else None}}; \
(o / 'uncut.json').write_text(json.dumps(s, indent=1)); print(json.dumps(s))" | tee -a "$OUT/job.log"
  rm -rf "$D"
done
$PY -c "import json; from pathlib import Path; o = Path('$OUT'); \
json.dump([json.loads(p.read_text()) for p in sorted(o.glob('k*/uncut.json'))], open(o / 'uncut.json', 'w'), indent=1)"
log "uncut: done"
