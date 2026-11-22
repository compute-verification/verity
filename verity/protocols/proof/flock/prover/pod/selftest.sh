# Sourced by a pod job that runs a prover's selftest, whose binary exits 0 whatever its cases say.
# `selftest_run LABEL OUT ERR CMD...` runs CMD (stdout to OUT, stderr to ERR), prints its SELFTEST line and each failed case,
# and records LABEL as failed unless CMD exited 0 and its SELFTEST line says all_pass true (under `--only CASE`, which prints
# no SELFTEST line: its one NEG line says pass true); `selftest_end` exits 1 naming each.
SELFTEST_FAILED=
selftest_run() {
  local label=$1 out=$2 err=$3 rc failed; shift 3
  "$@" > "$out" 2> "$err"; rc=$?
  grep -h '^SELFTEST' "$out"; grep -h '"pass":false' "$out" | cut -c1-300
  failed=$(grep '"pass":false' "$out" | grep -o '"case":"[^"]*"' | cut -d'"' -f4 | paste -sd, -)
  if [ "$rc" != 0 ]; then SELFTEST_FAILED+=" $label (exit $rc)"
  elif [[ " $* " == *" --only "* ]]; then
    grep -q '^NEG.*"pass":true' "$out" && [ -z "$failed" ] || SELFTEST_FAILED+=" $label (case ${failed:-without a passing NEG line})"
  elif ! grep -q '^SELFTEST' "$out"; then SELFTEST_FAILED+=" $label (no SELFTEST line)"
  elif ! grep -q '^SELFTEST.*"all_pass":true' "$out"; then SELFTEST_FAILED+=" $label (not all passed: ${failed:-no case named})"
  fi
  return 0
}
selftest_end() {
  [ -z "$SELFTEST_FAILED" ] || { echo "SELFTEST FAILED:$SELFTEST_FAILED"; exit 1; }
}
