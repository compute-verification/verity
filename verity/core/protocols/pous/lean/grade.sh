#!/usr/bin/env bash
# Grade one submission against one guarantee:
#   bash grade.sh SUBMISSION.lean Pous.Guarantees.NAME
# The submission declares `theorem solution : Pous.Guarantees.NAME`. One JSON line on stdout; exit 0 on PASS, 1 on FAIL,
# 2 if the trusted layer, the grader or the sandbox is broken or changed during grading, or the machine lacks
# unprivileged user namespaces (`unshare --user`), which the sandbox needs.
#   1. The trusted sources match TRUSTED.sha256, and the trusted layer and the grader build, before any submission code
#      runs. The hashes of the grader binary, the toolchain's `lean`, the trusted sources and every `.olean` on the
#      grader's LEAN_PATH (the security packages, Mathlib and their other dependencies, the toolchain's core) are recorded.
#      The trusted layer is built in `Security/Proofs`, and the grader here; TRUSTED.sha256's paths are relative
#      to this directory.
#   2. The submission may import only the trusted layer (`Definitions.Pous`, `Definitions.Pous.Assumptions`,
#      `Definitions.Pous.Guarantees`, in `Security`, and the lemmas' statements `Proofs.Pous.Statements`, which proves
#      nothing), the proved `Proofs.Pous.Sanity` and `Proofs.Pous.Accounting.Numbers` (in `Security/Proofs`), and
#      `Mathlib.*`; the grader's trusted environment is exactly those imports.
#   3. It compiles in a throwaway directory, inside a user, mount, PID and network namespace where every other mount is
#      read-only. Compile-time IO (`#eval`, `run_cmd`) can write nothing outside that directory, and every process it
#      starts dies with the namespace.
#   4. The recorded hashes and TRUSTED.sha256 are checked again. Then the grader, with its allowed axioms and pinned
#      targets compiled in, loads the submission's `.olean` (data only, no code runs), replays it through the kernel,
#      matches `solution` against the guarantee and audits its axioms.
# The compile and the grader each run under POUS_GRADE_TIMEOUT seconds (default 3600) and POUS_GRADE_MEMORY_MB
# (default 16384): Lean's resident-memory limit, which the kernel and the interpreter check (`lean -M`), and twice that
# in address space for everything else. Lean maps far more than it touches: the largest honest grades (P3) peak at
# 5.3 GB resident and 17.9 GB mapped.
set -uo pipefail
root=$(cd "$(dirname "$0")" && pwd)
proofs=$(cd "$root/../../../../../Security/Proofs" && pwd)
cd "$root"
export PATH="$HOME/.elan/bin:$PATH"
file=${1:?usage: grade.sh SUBMISSION.lean Pous.Guarantees.NAME}
target=${2:?usage: grade.sh SUBMISSION.lean Pous.Guarantees.NAME}
fail() { printf '{"file":"%s","target":"%s","verdict":"FAIL","reason":"%s","axioms":[]}\n' "$file" "$target" "$1"; exit 1; }
fatal() { echo "grade.sh: $1" >&2; exit 2; }
[ -f "$file" ] || fail "file not found"
unshare --user --map-root-user --pid --fork --mount --net true 2> /dev/null \
  || fatal "unprivileged user namespaces are unavailable (unshare --user failed); the sandbox needs them"

# 1. the trusted tree, before any submission code runs
sha256sum --quiet -c TRUSTED.sha256 >&2 || fatal "trusted files differ from TRUSTED.sha256"
(cd "$proofs" && lake build Definitions.Pous Definitions.Pous.Assumptions Definitions.Pous.Guarantees Proofs.Pous.Statements \
  Proofs.Pous.Sanity Proofs.Pous.Accounting.Numbers) >&2 || fatal "the trusted layer does not build"
lake build pous-grader >&2 || fatal "the grader does not build"
grader=$root/.lake/build/bin/pous-grader
lean=$(elan which lean) || fatal "no lean for $(cat lean-toolchain)"
lean_path=$(cd "$proofs" && lake env printenv LEAN_PATH) || fatal "no LEAN_PATH"
state() {
  { sha256sum "$grader" "$lean" $(cut -c67- TRUSTED.sha256)
    for d in $(echo "$lean_path" | tr ':' ' '); do [ -d "$d" ] && find "$d" -name '*.olean*' -type f; done | sort -u \
      | xargs -r -P "$(nproc)" -n 500 sha256sum | sort -k 2; } | sha256sum
}
before=$(state)

work=$(mktemp -d); out=$(mktemp -d); probe=$(mktemp -d)
trap 'rm -rf "$work" "$out" "$probe"' EXIT
cp "$file" "$work/PousSubmission.lean" || fatal "cannot read $file"

# 2. imports
imports=$(sed -n 's/^[[:space:]]*import[[:space:]]\{1,\}//p' "$work/PousSubmission.lean" | tr -s ' \t' '\n' | sed '/^$/d')
[ -n "$imports" ] || fail "no import"
for m in $imports; do
  case "$m" in
    Definitions.Pous|Definitions.Pous.Assumptions|Definitions.Pous.Guarantees|Proofs.Pous.Statements|Mathlib|Mathlib.*) ;;
    Proofs.Pous.Sanity|Proofs.Pous.Accounting.Numbers) ;;
    *) fail "import $m: a submission may import only the trusted layer, Sanity, Numbers and Mathlib modules" ;;
  esac
done

# 3. the sandboxed compile, after a dry run of the sandbox itself
sandbox='
w=$1; root=$2; lean=$3; export LEAN_PATH=$4; mode=$5; mem=$6  # $7: the security packages
mount --bind "$w" "$w" || exit 97
for mp in $(awk "{print \$2}" /proc/self/mounts | sort -u); do
  case "$mp" in "$w"|/proc|/proc/*|/dev|/dev/*|/sys|/sys/*) ;; *) mount -o remount,bind,ro "$mp" 2> /dev/null ;; esac
done
for probe in "$root" "$7" "$HOME" "$(dirname "$lean")" /tmp; do
  touch "$probe/.pous-sandbox-probe" 2> /dev/null && { echo "sandbox: $probe is writable" >&2; exit 97; }
done
[ "$mode" = probe ] && exit 0
ulimit -v $((2 * mem * 1024))
cd "$w" && exec "$lean" -M "$mem" -o PousSubmission.olean PousSubmission.lean
'
mem=${POUS_GRADE_MEMORY_MB:-16384}
run_sandbox() {
  timeout "${POUS_GRADE_TIMEOUT:-3600}" unshare --user --map-root-user --pid --fork --kill-child --mount --mount-proc \
    --net bash -c "$sandbox" _ "$1" "$root" "$lean" "$lean_path" "$2" "$mem" "$(dirname "$proofs")" >&2
}
run_sandbox "$probe" probe
case $? in
  0) ;;
  97) fatal "the sandbox could not make the trusted tree read-only" ;;
  *) fatal "the sandbox does not start (user, mount, PID or network namespace unavailable)" ;;
esac
run_sandbox "$work" compile
rc=$?
[ "$rc" = 97 ] && fatal "the sandbox could not make the trusted tree read-only"
[ "$rc" = 0 ] && [ -f "$work/PousSubmission.olean" ] || fail "does not compile"
cp "$work/PousSubmission.olean" "$out/PousSubmission.olean"

# 4. nothing trusted changed; grade
[ "$(state)" = "$before" ] || fatal "the grader, the toolchain or a trusted .olean changed while the submission built"
sha256sum --quiet -c TRUSTED.sha256 >&2 || fatal "trusted files changed while the submission built"
LEAN_PATH="$out:$lean_path" POUS_GRADE_MEMORY_MB=$mem timeout "${POUS_GRADE_TIMEOUT:-3600}" \
  prlimit --as=$((2 * mem * 1024 * 1024)) "$grader" "$file" "$target" "$(echo $imports | tr ' ' ',')"
