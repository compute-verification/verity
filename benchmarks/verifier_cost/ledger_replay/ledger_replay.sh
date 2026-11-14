#!/usr/bin/env bash
# One replay of the auditor's ledger against its sessions: ledger_replay.py writes, per S, a ledger of S sessions as the
# challenger's writer signs one, with what the `challenger_ledger` piece's reference answers on it, and times the
# reference's replay; ledger-replay-time (Main.lean, built in a scratch Lake package against the checkout's verifier,
# main's cached build of it and of the packages it requires) times the compiled replay, `Spot.Challenger.checked`, the
# piece around it, and the replay's parts. A size whose period's close is past a ledger line's cap is refused by both,
# and its row says so; PERIOD=N puts the sessions in periods of N, which goes past it. Run:
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --pool research --cwd source \
#     --mem-wait 1800 -- bash benchmarks/verifier_cost/ledger_replay/ledger_replay.sh 10 30 100 300 1000
# PERIOD (a period's sessions; default: one period), REPEAT (timings a phase, the least kept; default 3) and LMS (the
# key's W,H; default: the deployed) are read from the environment. $OUT gets inputs.jsonl (one row per S: entries,
# bytes, the longest line, the reference's seconds and its signature checks), ledger_replay.jsonl (one row per S and
# phase: microseconds, peak RSS), table.jsonl (ledger_replay.py --table: one row per S, the split and the fit),
# cpu.txt, setup.log and build.log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: ledger_replay.sh S..." >&2; exit 2; }
P=$PWD/verity/protocols/proofs/spot/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
uv sync -q --all-packages --extra torch-cpu
bash tools/verity/lean/setup.sh "$P" > "$O/setup.log" 2>&1 || { tail -40 "$O/setup.log" >&2; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
# the packages the verifier requires by path, each restored as main built it, so that lake unpacks them
python3 tools/verity/lean/cache.py get verity/protocols/proofs/flock/verifier verity/protocols/interface \
  verity/commitments/lms/lean verity/circuits/difftest verity/circuits/base verity/circuits/lean \
  verity/commitments/lean >> "$O/setup.log" 2>&1 || echo "ledger_replay.sh: a cached build didn't restore; lake compiles it"
mkdir -p "$S/time"
cp benchmarks/verifier_cost/ledger_replay/Main.lean "$P/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "ledger_replay_time"
defaultTargets = ["ledger-replay-time"]

[[require]]
name = "core_spot"
path = "$P"

[[lean_exe]]
name = "ledger-replay-time"
root = "Main"
EOF
(cd "$S/time" && lake build ledger-replay-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
lscpu | grep -E '^(Model name|CPU\(s\)|CPU max MHz)' > "$O/cpu.txt" || true
uv run python benchmarks/verifier_cost/ledger_replay/ledger_replay.py "$S/in" "$@" --period "${PERIOD:-0}" \
  ${LMS:+--lms "$LMS"} | tee "$O/inputs.jsonl"
for n in "$@"; do
  "$S/time/.lake/build/bin/ledger-replay-time" "$n" "$S/in/ledger-$n.json" "$S/in/expect-$n.json" "${REPEAT:-3}" \
    | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/ledger_replay.jsonl"
  rm -f "$S/in/ledger-$n.json"
done
uv run python benchmarks/verifier_cost/ledger_replay/ledger_replay.py --table "$O" | tee "$O/table.jsonl"
