#!/usr/bin/env bash
# One served H1 leaf proved as its 32 lanes and its combine (h1zy's PouwH1LaneZy_v1 and PouwH1Combine_v1, pouw.leaf_lanes),
# one C-Flock --zk statement each, verified by the Lean verifier against the verifier's own files (run.py's docstring has
# the protocol and the negatives).
#   research run --on MACHINE --pool proofs --project verity --source . --cwd source \
#     --mem-gb G --declared-output 'out/*' --env H1L_MEM_GB=G -- bash benchmarks/pouw/h1_lanes/job.sh
#   1. hidden_zk/job.sh's builds (flock-circuit's proving build, the Lean flock-verify, a venv), PoUW's pouw-leaf-own-window
#      and, under H1_TMPL=structural, the prover's flock-tmpl-unit (verity/ml/flock/tests/lean/TmplUnit.lean, which
#      unrolls the unit and so is no part of the verifier), built beside flock-verify's sources
#   2. run.py prepare: the statements' Programs, the leaf's registered values and partials, every statement staged (WORK/plan.tsv),
#      each circuit flat, or under H1_TMPL=structural the templated statement of its structural lowering (h1unit.layout),
#      whose unit rows flock-tmpl-unit then places (circuits/<kind>/unit.placed, the prover's --tmpl-unit)
#   3. one loopback --zk session per statement, H1L_PAR at a time, each drawing its one instance
#   4. the plan's Lean checks, each with its statement's own Program, partition, public file and registered reads, and
#      pouw-leaf-own-window's check of those reads, placed from the window's descriptor and round log, against the
#      registration and produced record the verifier holds, and of each session's draw against the draw policy (every unit)
#   5. run.py summarize: out/summary.json (a templated check refused only for its form after every other check,
#      run.py's TMPL_WHY, is labelled run.py's TEMPLATED, never ACCEPT)
# env: H1L_MEM_GB (the job's --mem-gb)  H1L_K (448; h1zy.combine takes K >= 384)  H1L_JOBS (8: prepare's stagers)  H1L_PAR (3: sessions at a time)
#      H1L_LPAR (8: Lean checks at a time)  H1_TILE (v1 or v3)  H1_TMPL (h1unit.layout's: unset or flat the flat statement,
#      structural the templated one at iv/hybrid/4)  HZ_CACHE: hidden_zk's
#      H1L_J10R=1: j10r.py's J10-R leaf in place of run.py's (H1L_K 4608 by default, its lanes in 3 chunks; H1L_RC its
#      R x C, 16x16), with no pouw-leaf-own-window check, which states v1's statements only
set -uo pipefail
export HZ_CONFIGS=h1-lanes HZ_MEM_GB=${H1L_MEM_GB:?} HZ_RUNS=1 HZ_WARM=0
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk:$SRC/benchmarks/pouw/h1_prove
H=$SRC/benchmarks/pouw/h1_lanes
RUN=$H/run.py K=${H1L_K:-448} PAR=${H1L_PAR:-3} LPAR=${H1L_LPAR:-8}
[ -n "${H1L_J10R:-}" ] && RUN=$H/j10r.py K=${H1L_K:-4608}
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
OWN=:
if [ -z "${H1L_J10R:-}" ]; then
  OWN=$(cached leafownwin "$(tree_key verity/core)" build_own)/verity/core/protocols/pouw/lean/check/.lake/build/bin/pouw-leaf-own-window
  [ -x "$OWN" ] || { log "pouw-leaf-own-window build FAILED: $(tail -20 "$OUT/build-leafownwin.log")"; exit 1; }
  sha256sum "$OWN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi
build_tmpl_unit() { # build_tmpl_unit DEST: DEST/.lake/build/bin/flock-tmpl-unit
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  local D=$1
  rm -rf "$D.tmp" && mkdir -p "$D.tmp" && (cd "$SRC/verity/core/service/flock" && tar --exclude=.lake -cf - .) | tar -xf - -C "$D.tmp" \
    && cp "$SRC/verity/ml/flock/tests/lean/TmplUnit.lean" "$D.tmp/" \
    && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> "$D.tmp/lakefile.toml" \
    && (cd "$D.tmp" && lake build flock-tmpl-unit) && rm -rf "$D" && mv "$D.tmp" "$D"
}
TU= TUP=
if [ "${H1_TMPL:-flat}" != flat ]; then
  cached tmplunit "$(tree_key verity/core/service/flock verity/ml/flock/tests/lean/TmplUnit.lean)" build_tmpl_unit > "$W/tu.dir" &
  TUP=$!
fi
log "h1-lanes: $(basename "$RUN") K=$K tile=${H1_TILE:-v1} form ${H1_TMPL:-flat} sessions $PAR at a time, Lean $LPAR at a time"

$TIME "$OUT/prepare.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $RUN prepare --work $WK --k $K --jobs ${H1L_JOBS:-8} \
  --name h1-lanes-${RESEARCH_RUN_ID:-local}" > "$OUT/prepare.out" 2> "$OUT/prepare.err"
rc=$?
tail -4 "$OUT/prepare.out" | cut -c1-2000 | tee -a "$OUT/job.log"
cp "$WK/plan.json" "$WK/prepare.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -5 "$OUT/prepare.err")"; exit 1; }
mkdir -p "$OUT/verifier" && cp "$WK"/verifier/* "$OUT/verifier/"
if [ -n "$TUP" ]; then
  wait "$TUP"
  TU=$(cat "$W/tu.dir")/.lake/build/bin/flock-tmpl-unit
  [ -x "$TU" ] || { log "flock-tmpl-unit build FAILED: $(tail -20 "$OUT/build-tmplunit.log")"; exit 1; }
  sha256sum "$TU" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi
for t in "$WK"/circuits/*/unit.tmpl; do   # each templated circuit's unit rows at the positions the verifier folds them at
  [ -e "$t" ] || continue
  c=$(basename "$(dirname "$t")")
  [ -n "$TU" ] || { log "circuit $c is templated, and H1_TMPL=${H1_TMPL:-} built no flock-tmpl-unit"; exit 1; }
  $TIME "$OUT/tmpl-unit-$c.time" bash -c "$(vlimit "$STAGE_GB") exec $TU $t --layout $(cat "${t%.tmpl}.layout") \
    --out ${t%.tmpl}.placed" > "$OUT/tmpl-unit-$c.txt" 2>&1 || { log "flock-tmpl-unit $c FAILED: $(tail -3 "$OUT/tmpl-unit-$c.txt")"; exit 1; }
  log "tmpl-unit $c: $(grep '^TMPL ' "$OUT/tmpl-unit-$c.txt" | cut -c1-300)"
done

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
while IFS=$'\t' read -r kind tag D pub reg prog part _expect regrec prodrec desc rounds unit stmt fstmt; do
  [ "$kind" = C ] || continue
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; continue; }
  "$OWN" "$desc" "$rounds" "$regrec" "$prodrec" "$reg" "$unit" "$stmt" $s > "$OUT/lean/$tag.own" 2>&1
  while [ "$(jobs -rp | wc -l)" -ge "$LPAR" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement $fstmt --public $pub --partition $part --program $prog --registered $reg --session $s" \
    > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
done < "$WK/plan.tsv"
wait

"$PY" "$RUN" summarize --work "$WK" --out "$OUT" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
log "h1-lanes: done (exit $rc)"
exit "$rc"
