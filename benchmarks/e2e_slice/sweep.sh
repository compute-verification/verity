#!/usr/bin/env bash
# The checkpoint spacing's proof cost: one fake-engine window (window.sh ENGINE=fake, real C-Flock proofs) a configuration,
# one after another, each under $OUT/<fold>-s<S>. CONFIGS ("duo:1 duo:4 duo:8 duo:16 duo:32"), ROUNDS (3), SHAPE (the
# prover's --m --n --k, "--m 256 --n 256 --k 4096": o_proj's k, so the statement is the served one), THREADS (C-Flock's
# prove threads, 48).
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source -- bash benchmarks/e2e_slice/sweep.sh
set -uo pipefail
R=${OUT:?}
SHAPE=${SHAPE:---m 256 --n 256 --k 4096}
K=$(echo "$SHAPE" | sed -E 's/.*--k ([0-9]+).*/\1/')
for cfg in ${CONFIGS:-duo:1 duo:4 duo:8 duo:16 duo:32}; do
  f=${cfg%%:*} s=${cfg##*:}
  mkdir -p "$R/$f-s$s"
  OUT=$R/$f-s$s ENGINE=fake WINDOW=$s FOLD=$f ROUNDS=${ROUNDS:-3} CALLS=1 CADENCE=${CADENCE:-2} \
    PROVER_ARGS="$SHAPE --prove-threads ${THREADS:-48} --workers 8" CHALLENGER_ARGS="--k $K" \
    bash benchmarks/e2e_slice/window.sh > "$R/$f-s$s.log" 2>&1
  echo "$cfg rc $?" >> "$R/rc.txt"
done
cat "$R/rc.txt"
