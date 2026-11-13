set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
(cd verity/protocols/spot/verifier/lean && mkdir -p .lake && flock .lake/verity-build.lock lake build Spot sampled-proofs-period sampled-proofs-difftest) > "$OUT/lean-build.log" 2>&1
rc=$?; echo "LEAN-BUILD exit $rc"; grep -n 'error' -A 12 "$OUT/lean-build.log" | head -80; tail -3 "$OUT/lean-build.log"
[ $rc -eq 0 ] || exit 4
E=verity/protocols/spot/verifier/lean/.lake/build/bin/sampled-proofs-period
echo "ID: $($E id)"
uv run -q pytest verity/protocols/spot/verifier/lean/tests/test_period_difftest.py verity/protocols/spot/verifier/lean/tests/test_period_audit_difftest.py -q -rs --durations=5 2>&1 | tee "$OUT/period-difftests.txt" | tail -40
echo "PYTEST exit ${PIPESTATUS[0]}"
exit ${PIPESTATUS[0]}
