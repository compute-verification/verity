set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
echo "HEAD_AT $(git rev-parse HEAD 2>/dev/null)"
python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Spot/Period.lean > "$OUT/try-period.log" 2>&1
rc=$?; echo "TRY-PERIOD exit $rc"; head -c 120000 "$OUT/try-period.log"
python3 tools/verity/lean/lean_changed.py --build security/Properties/Spot/PeriodLedger.lean security/Proofs/Properties/Challenger/Window.lean > "$OUT/build-ledger.log" 2>&1
rc2=$?; echo "BUILD-LEDGER exit $rc2"; tail -c 8000 "$OUT/build-ledger.log"
if [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; then
  python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Spot/PeriodLedger.lean > "$OUT/try-ledger.log" 2>&1
  rc3=$?; echo "TRY-LEDGER exit $rc3"; head -c 60000 "$OUT/try-ledger.log"
fi
exit 0
