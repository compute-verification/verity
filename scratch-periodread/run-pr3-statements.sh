set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
(cd verity/protocols/spot/verifier/lean && mkdir -p .lake && flock .lake/verity-build.lock lake build Spot) > "$OUT/spot-build.log" 2>&1
rc=$?; echo "SPOT-BUILD exit $rc"; grep -n 'error' "$OUT/spot-build.log" | head -40; tail -3 "$OUT/spot-build.log"
[ $rc -eq 0 ] || exit 4
bash tools/verity/lean/setup.sh security/Properties > "$OUT/setup.log" 2>&1
rc=$?; echo "SETUP exit $rc"; tail -4 "$OUT/setup.log"
[ $rc -eq 0 ] || exit 5
(cd security/Properties && mkdir -p .lake && flock .lake/verity-build.lock lake build Properties.Spot.Period) > "$OUT/properties-build.log" 2>&1
rc=$?; echo "PROPERTIES-BUILD exit $rc"; grep -n 'error\|warning' "$OUT/properties-build.log" | head -60; tail -4 "$OUT/properties-build.log"
[ $rc -eq 0 ] || exit 6
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
cat > "$d/Probe.lean" <<'LEAN'
import Properties.Spot.Period
#check @Properties.Spot.Period.LedgerMeans
#check @Properties.Spot.Period.AcceptMeans
#check @Properties.Spot.Period.CommittedOnce
#check @Properties.Spot.Period.AuditedAt
#print axioms Properties.Spot.Period.LedgerMeans
#print axioms Properties.Spot.Period.AcceptMeans
#print axioms Properties.Spot.Period.CommittedOnce
LEAN
(cd security/Properties && lake env lean "$d/Probe.lean") > "$OUT/probe.log" 2>&1
rc=$?; echo "PROBE exit $rc"; head -60 "$OUT/probe.log"
exit $rc
