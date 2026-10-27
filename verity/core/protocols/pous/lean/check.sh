#!/usr/bin/env bash
# Acceptance check: the grader and the security analysis (`Security`, where PoUS's trusted layer and proofs are)
# build and pass Verity's Lean audit (`tools/verity/lean/check.py --fresh`), and the grader passes the positive control
# and rejects each negative control for the stated reason. Prints ALL CHECKS PASSED.
set -uo pipefail
cd "$(dirname "$0")"
security=$(cd ../../../../../Security && pwd)
export PATH="$HOME/.elan/bin:$PATH"
fail() { echo "CHECK FAILED: $1" >&2; exit 1; }

echo "== the packages: Verity's Lean audit (its controls, every declaration, pins, kernel replay, leanchecker --fresh)"
audit_py=${POUS_LEAN_AUDIT:-$PWD/../../../../../tools/verity/lean/check.py}
[ -f "$audit_py" ] || fail "no Verity tools/verity/lean/check.py at $audit_py (set POUS_LEAN_AUDIT to another copy)"
lake build > /dev/null || fail "lake build"
alog=$(mktemp)
python3 "$audit_py" --build --fresh . "$security" > "$alog" 2>&1 || { tail -40 "$alog" >&2; rm -f "$alog"; fail "check.py"; }
grep -E '^(audit: controls|AUDIT )' "$alog" | sed 's/^/   /' | cut -c1-160
rm -f "$alog"

echo "== grader controls"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
expect() { # file target exit substring
  local out code
  out=$(bash grade.sh "$1" "$2" 2>/dev/null); code=$?
  echo "   $(basename "$1") -> exit $code: $out"
  [ "$code" = "$3" ] || fail "$1: exit $code, expected $3"
  echo "$out" | grep -Fq "$4" || fail "$1: output lacks '$4'"
}

expect submissions/reference.lean Pous.Guarantees.KFloor 0 '"verdict":"PASS"'

cat > "$tmp/sorry.lean" <<'EOF'
import Proofs.Pous.Statements
theorem solution : Pous.Guarantees.Theorem1ii := sorry
EOF
expect "$tmp/sorry.lean" Pous.Guarantees.Theorem1ii 1 'sorryAx'

cat > "$tmp/rogue_axiom.lean" <<'EOF'
import Proofs.Pous.Statements
axiom cheat : False
theorem solution : Pous.Guarantees.Theorem1ii := cheat.elim
EOF
expect "$tmp/rogue_axiom.lean" Pous.Guarantees.Theorem1ii 1 'unregistered axioms: [cheat]'

cat > "$tmp/wrong_statement.lean" <<'EOF'
import Proofs.Pous.Statements
theorem solution : Pous.Params.ρ = 18 / 19 := rfl
EOF
expect "$tmp/wrong_statement.lean" Pous.Guarantees.KFloor 1 'does not have the pinned type'

expect submissions/reference.lean Pous.Params.ρ 1 'is not a pinned statement'

# The d = 12 band pin submitted as the d = 5 one. The statements differ only in a literal, so deciding definitional
# equality would keep the kernel unfolding; the grader compares syntactically. The timeout turns a regression into a
# failure instead of a hang.
cat > "$tmp/near_miss.lean" <<'EOF'
import Definitions.Pous.Guarantees
theorem solution : Pous.Guarantees.BandChainMeets64D12 := sorry
EOF
POUS_GRADE_TIMEOUT=120 expect "$tmp/near_miss.lean" Pous.Guarantees.BandChainMeets64D5 1 'does not have the pinned type'

# `native_decide` rests on an auxiliary axiom per use (`solution._native.native_decide.ax_…`).
cat > "$tmp/native.lean" <<'EOF'
import Proofs.Pous.Statements
theorem solution : Pous.Guarantees.Theorem1ii := by
  have : (2 : ℕ) + 2 = 4 := by native_decide
  exact sorry
EOF
expect "$tmp/native.lean" Pous.Guarantees.Theorem1ii 1 'native_decide.ax'

cat > "$tmp/import.lean" <<'EOF'
import Proofs.Pous.Statements
import Proofs.Pous
theorem solution : Pous.Guarantees.Theorem1ii := Pous.SecurityProofs.Theorem1ii
EOF
expect "$tmp/import.lean" Pous.Guarantees.Theorem1ii 1 'may import only'

# Adds `solution` without kernel checking; the option name is assembled so the textual filter misses it.
cat > "$tmp/kernel_bypass.lean" <<'EOF'
import Proofs.Pous.Statements
open Lean Elab Command in
run_cmd liftCoreM do
  let opt : Name := .str (.str .anonymous ("deb" ++ "ug")) ("skipKernel" ++ "TC")
  withOptions (·.setBool opt true) do
    addDecl (.thmDecl { name := `solution, levelParams := [], type := .const ``Pous.Guarantees.Theorem1ii [],
                        value := .const ``True.intro [] })
EOF
expect "$tmp/kernel_bypass.lean" Pous.Guarantees.Theorem1ii 1 'the kernel rejects the submission'

# The statement red team's F1 exploits: compile-time IO that widens the allowlist, or swaps the grader binary for one
# that prints PASS, each with a background process that retries after the build. Errors are swallowed so the files
# compile and reach the grader.
grader_hash=$(sha256sum .lake/build/bin/pous-grader)
cat > "$tmp/rt_registry_tamper.lean" <<'EOF'
import Proofs.Pous.Statements
#eval show IO Unit from do
  for p in ["AXIOMS", "ROOT/AXIOMS", "ROOT/Grader/Registry.lean", "ROOT/TRUSTED.sha256"] do
    try
      let orig ← IO.FS.readFile p
      IO.FS.writeFile p (orig ++ "sorryAx\n")
    catch _ => pure ()
  try IO.FS.writeFile "ROOT/AXIOMS" "propext\nClassical.choice\nQuot.sound\nsorryAx\n" catch _ => pure ()
  let _ ← IO.Process.spawn { cmd := "sh", args := #["-c",
    "(sleep 3; echo sorryAx >> ROOT/Grader/Registry.lean; touch ROOT/AXIOMS) > /dev/null 2>&1 &"] }
theorem solution : Pous.Guarantees.Theorem1ii := sorry
EOF
cat > "$tmp/rt_grader_swap.lean" <<'EOF'
import Proofs.Pous.Statements
#eval show IO Unit from do
  let g := "ROOT/.lake/build/bin/pous-grader"
  try IO.FS.rename g (g ++ ".orig") catch _ => pure ()
  try IO.FS.writeFile g "#!/bin/sh\necho PASS\n" catch _ => pure ()
  let _ ← IO.Process.spawn { cmd := "sh", args := #["-c",
    "(sleep 3; cp /bin/true ROOT/.lake/build/bin/pous-grader) > /dev/null 2>&1 &"] }
theorem solution : (1 : Nat) = 1 := rfl
EOF
sed -i "s#ROOT#$PWD#g" "$tmp/rt_registry_tamper.lean" "$tmp/rt_grader_swap.lean"
expect "$tmp/rt_registry_tamper.lean" Pous.Guarantees.Theorem1ii 1 'unregistered axioms: [sorryAx]'
expect "$tmp/rt_grader_swap.lean" Pous.Guarantees.Theorem1ii 1 'does not have the pinned type'
sleep 4
sha256sum --quiet -c TRUSTED.sha256 || fail "a control changed a trusted file"
[ "$(sha256sum .lake/build/bin/pous-grader)" = "$grader_hash" ] || fail "a control changed the grader binary"
[ ! -e AXIOMS ] && [ ! -e .lake/build/bin/pous-grader.orig ] || fail "a control wrote into the trusted tree"
echo "   the trusted tree is unchanged"

# A machine without unprivileged user namespaces is an environment error (exit 2), not a failed submission.
fake=$(mktemp -d)
printf '#!/bin/sh\nexit 1\n' > "$fake/unshare"; chmod +x "$fake/unshare"
err=$(PATH="$fake:$PATH" bash grade.sh submissions/reference.lean Pous.Guarantees.KFloor 2>&1 > /dev/null); code=$?
rm -rf "$fake"
[ "$code" = 2 ] && echo "$err" | grep -q 'user namespaces are unavailable' \
  || fail "missing user namespaces gave exit $code, not the environment error"
echo "   no user namespaces -> exit 2 (environment error)"

echo "ALL CHECKS PASSED"
