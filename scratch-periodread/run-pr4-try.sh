set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
echo "HEAD_AT $(git rev-parse HEAD 2>/dev/null)"
python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Spot/Period.lean > "$OUT/try-period.log" 2>&1
rc=$?; echo "TRY-PERIOD exit $rc"; grep -n -E "error|^lean-try:" "$OUT/try-period.log" | cut -c1-400 | tail -40
python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Challenger/Lines.lean > "$OUT/try-lines.log" 2>&1
rc2=$?; echo "TRY-LINES exit $rc2"; grep -n -E "error|^lean-try:" "$OUT/try-lines.log" | cut -c1-400 | tail -40
if [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; then
  python3 tools/verity/lean/lean_changed.py --try security/Proofs/Properties/Spot/PeriodLedger.lean > "$OUT/try-ledger.log" 2>&1
  rc3=$?; echo "TRY-LEDGER exit $rc3"; grep -n -E "error|^lean-try:" "$OUT/try-ledger.log" | cut -c1-400 | tail -40
fi
exit 0
