#!/usr/bin/env bash
# One served H1 leaf proved as its 32 lanes and its combine (h1zy's PouwH1LaneZy_v1 and PouwH1Combine_v1, pouw.leaf_lanes),
# one C-Flock --zk statement each, verified by the Lean verifier against the verifier's own files (run.py's docstring has
# the protocol and the negatives).
#   research run --on vy-nebius-cpu-1 --pool proofs --project verity --campaign proofs-24h --source . --cwd source \
#     --mem-gb G --declared-output 'out/*' --env H1L_MEM_GB=G -- bash benchmarks/pouw/h1_lanes/job.sh
#   1. hidden_zk/job.sh's builds (flock-circuit's proving build, the Lean flock-verify, a venv) and PoUW's pouw-leaf-own-window
#   2. run.py prepare: the statements' Programs, the leaf's registered values and partials, every statement staged (WORK/plan.tsv)
#   3. one loopback --zk session per statement, H1L_PAR at a time, each drawing its one instance
#   4. the plan's Lean checks, each with its statement's own Program, partition, public file and registered reads, and
#      pouw-leaf-own-window's check of those reads, placed from the window's descriptor and round log, against the
#      registration and produced record the verifier holds, and of each session's draw against the draw policy (every unit)
#   5. run.py summarize: out/summary.json
# env: H1L_MEM_GB (the job's --mem-gb)  H1L_K (448; h1zy.combine takes K >= 384)  H1L_JOBS (8: prepare's stagers)  H1L_PAR (3: sessions at a time)
#      H1L_LPAR (8: Lean checks at a time)  H1_TILE (v1 or v3)  HZ_CACHE: hidden_zk's
set -uo pipefail
[ -z "${H1_TMPL:-}" ] || { echo "H1_TMPL: the prover proves only the flat statement; run.py's stager alone writes the templated one" >&2; exit 2; }
export HZ_CONFIGS=h1-lanes HZ_MEM_GB=${H1L_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk:$SRC/benchmarks/pouw/h1_prove
H=$SRC/benchmarks/pouw/h1_lanes
K=${H1L_K:-448} PAR=${H1L_PAR:-3} LPAR=${H1L_LPAR:-8}
WK=$W/work
mkdir -p "$OUT/lean"
setup
build_own() {       # build_own DEST: DEST/verity/core/protocols/pouw/lean/check/.lake/build/bin/pouw-leaf-own-window
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  local D=$1
  rm -rf "$D.tmp" && mkdir -p "$D.tmp" \
    && (cd "$SRC" && tar --exclude=.lake --exclude=__pycache__ -cf - verity/core) | tar -xf - -C "$D.tmp" \
    && (cd "$D.tmp/verity/core/protocols/pouw/lean/check" && lake build pouw-leaf-own-window) && rm -rf "$D" && mv "$D.tmp" "$D"
}
OWN=$(cached leafownwin "$(tree_key verity/core)" build_own)/verity/core/protocols/pouw/lean/check/.lake/build/bin/pouw-leaf-own-window
[ -x "$OWN" ] || { log "pouw-leaf-own-window build FAILED: $(tail -20 "$OUT/build-leafownwin.log")"; exit 1; }
sha256sum "$OWN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
log "h1-lanes: K=$K tile=${H1_TILE:-v1} sessions $PAR at a time, Lean $LPAR at a time"

$TIME "$OUT/prepare.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $H/run.py prepare --work $WK --k $K --jobs ${H1L_JOBS:-8} \
  --name h1-lanes-${RESEARCH_RUN_ID:-local}" > "$OUT/prepare.out" 2> "$OUT/prepare.err"
rc=$?
tail -4 "$OUT/prepare.out" | cut -c1-2000 | tee -a "$OUT/job.log"
cp "$WK/plan.json" "$WK/prepare.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -5 "$OUT/prepare.err")"; exit 1; }
mkdir -p "$OUT/verifier" && cp "$WK"/verifier/* "$OUT/verifier/"

# ---- proofs: one --zk session per statement ------------------------------------------------------------------------------
# A lane's serve outgrows its prove (15 GB was short at K = 384, its prove 7 GB), so a session's share is split evenly.
share=$(( (PROVE_GB + SERVE_GB) / PAR / 2 ))
N=1 PROVE_GB=$share SERVE_GB=$share
export RAYON_NUM_THREADS=$(( TH / PAR > 0 ? TH / PAR : 1 ))
while IFS=$'\t' read -r kind tag D; do
  [ "$kind" = S ] || continue
  mkdir -p "$OUT/$tag"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  ( session "$D" "$OUT/$tag" || log "prove $tag FAILED: $(tail -2 "$OUT/$tag/prove.err")" ) &
done < "$WK/plan.tsv"
wait
log "proved: $(ls -d "$WK"/st/*/sessions/*/ 2>/dev/null | wc -l) sessions"

# ---- Lean: the plan's checks -----------------------------------------------------------------------------------------------
cap=$(( STAGE_GB / LPAR ))
while IFS=$'\t' read -r kind tag D pub reg prog part _expect regrec prodrec desc rounds unit stmt; do
  [ "$kind" = C ] || continue
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; continue; }
  "$OWN" "$desc" "$rounds" "$regrec" "$prodrec" "$reg" "$unit" "$stmt" $s > "$OUT/lean/$tag.own" 2>&1
  while [ "$(jobs -rp | wc -l)" -ge "$LPAR" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement verity/flock-circuit --public $pub --partition $part --program $prog --registered $reg --session $s" \
    > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
done < "$WK/plan.tsv"
wait

"$PY" "$H/run.py" summarize --work "$WK" --out "$OUT" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
log "h1-lanes: done (exit $rc)"
exit "$rc"
