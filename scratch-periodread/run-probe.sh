set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
(cd verity/protocols/spot/verifier/lean && mkdir -p .lake && flock .lake/verity-build.lock lake build Spot) > "$OUT/spot-build.log" 2>&1
rc=$?; echo "SPOT-BUILD exit $rc"; grep -n 'error' "$OUT/spot-build.log" | head -40; tail -2 "$OUT/spot-build.log"
[ $rc -eq 0 ] || exit 4
bash tools/verity/lean/setup.sh security/Proofs > "$OUT/setup.log" 2>&1
rc=$?; echo "SETUP exit $rc"; tail -3 "$OUT/setup.log"
[ $rc -eq 0 ] || exit 5
cd security/Proofs && mkdir -p .lake
flock .lake/verity-build.lock lake build Proofs.Properties.Pouw.LiveSalt.Ledger Proofs.Properties.Challenger.Window > "$OUT/proofs-build.log" 2>&1
rc=$?; echo "PROOFS-BUILD exit $rc"; grep -n 'error' -A 12 "$OUT/proofs-build.log" | head -80; tail -3 "$OUT/proofs-build.log"
[ $rc -eq 0 ] || exit 6
for t in "Properties.Spot.Period Properties.Spot.PeriodLedger" "+Properties.Spot.Period +Properties.Spot.PeriodLedger"; do
  flock .lake/verity-build.lock lake build $t > "$OUT/statements-build.log" 2>&1
  rc=$?; echo "STATEMENTS-BUILD ($t) exit $rc"
  [ $rc -eq 0 ] && break
  grep -n 'error\|warning' -A 14 "$OUT/statements-build.log" | head -200
done
tail -3 "$OUT/statements-build.log"
[ $rc -eq 0 ] || exit 7
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
cat > "$d/Statements.lean" <<'LEAN'
import Properties.Spot.PeriodLedger
#check @Properties.Spot.Period.LedgerMeans
#check @Properties.Spot.Period.AcceptMeans
#check @Properties.Spot.Period.CommittedOnce
#check @Properties.Spot.Period.AuditedAt
#check @Properties.Spot.Period.LedgerLinesOfDevice
#check @Properties.Spot.Period.LedgerOfDevice
#print axioms Properties.Spot.Period.LedgerMeans
#print axioms Properties.Spot.Period.AcceptMeans
#print axioms Properties.Spot.Period.CommittedOnce
#print axioms Properties.Spot.Period.LedgerLinesOfDevice
#print axioms Properties.Spot.Period.LedgerOfDevice
LEAN
lake env lean "$d/Statements.lean" > "$OUT/statements.log" 2>&1
rc=$?; echo "STATEMENTS exit $rc"; head -40 "$OUT/statements.log"
cat > "$d/Probe.lean" <<'LEAN'
__SCRATCH__
LEAN
lake env lean "$d/Probe.lean" > "$OUT/probe.log" 2>&1
echo "PROBE exit $?"; wc -l "$OUT/probe.log"
exit $rc
