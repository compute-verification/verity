#!/usr/bin/env bash
# Poor-man's sampling profile of the dietmc prover (perf is unavailable in the
# vy-cpu container: perf_event_paranoid=4, no CAP_PERFMON).  Runs one
# prover/verifier pair on an exported (relation, instance, witness) triple and
# samples the prover's stack with gdb every SAMPLE_S seconds; the leaf frames
# and the "coarse buckets" (sVOLE / online / parse) are histogrammed.
#
#   profile.sh <dir-with-relation.txt,instance.txt,witness.txt> <field> [samples] [threads]
set -euo pipefail
DIR=$1; FIELD=${2:-F61p}; N=${3:-300}; THREADS=${4:-1}
DIETMC=${DIETMC:-/workspace/target-swanky/release/dietmc}
OUT=${OUT:-$DIR/profile}
mkdir -p "$OUT"
cat > "$OUT/dmc.toml" <<EOF
verifier = false
prover = true
instance = "$DIR/instance.txt"
witness = "$DIR/witness.txt"
relation = "$DIR/relation.txt"
lpn = "medium"
threads = $THREADS
connection_addr = "127.0.0.1:${PORT:-7311}"
EOF
sed -e 's/^verifier = false/verifier = true/' -e 's/^prover = true/prover = false/' "$OUT/dmc.toml" > "$OUT/dmc_v.toml"
"$DIETMC" -c "$OUT/dmc_v.toml" > "$OUT/verifier.log" 2>&1 &
VPID=$!
sleep 0.3
"$DIETMC" -c "$OUT/dmc.toml" > "$OUT/prover.log" 2>&1 &
PPID_=$!
: > "$OUT/samples.txt"
i=0
while kill -0 "$PPID_" 2>/dev/null && [ "$i" -lt "$N" ]; do
  gdb -q -batch -p "$PPID_" -ex "thread apply all bt 40" 2>/dev/null \
    | sed -n 's/^#[0-9]* *\(0x[0-9a-f]* in \)\{0,1\}\([^(]*\).*/\2/p' \
    | sed 's/ *$//' | paste -sd'|' >> "$OUT/samples.txt" || true
  echo >> "$OUT/samples.txt"
  i=$((i+1))
  sleep "${SAMPLE_S:-0.1}"
done
wait "$PPID_" || true
wait "$VPID" || true
echo "samples: $(grep -c . "$OUT/samples.txt")"
python3 - "$OUT/samples.txt" <<'EOF'
import collections, re, sys
buckets = collections.Counter(); leaves = collections.Counter(); n = 0
RULES = [
    ("svole: LPN/spsVOLE/GGM (svole_wykw)", r"svole_wykw|spsvole|ggm|lpn|LpnParams"),
    ("svole: OT extension / base OT (kos, chou_orlandi, copee)", r"ot_alsz_kos|kos|chou_orlandi|copee|base_svole"),
    ("online: mult/zero checks (homcom/backend)", r"homcom|check_zero|quicksilver|MacProver|assert_zero|mult_check|prove_mul|input_private|input_public"),
    ("parse/eval: SIEVE-IR text (sieve_ir_parser / circuit_ir)", r"sieve_ir|text_parser|circuit_ir|parse|Reader|BufRead|read_line|tokeniz"),
    ("crypto primitives (AES / Blake3 / Ristretto)", r"aes|Aes|blake3|curve25519|ristretto|sha"),
    ("field arithmetic (scuttlebutt / F61p)", r"scuttlebutt|F61p|FiniteField|mul_assign|reduce"),
    ("channel I/O (std::net / write / read / flush)", r"std::net|TcpStream|std::io|write_all|read_exact|flush|Channel|syscall|send|recv"),
    ("alloc / memcpy / Vec", r"alloc|memcpy|memmove|realloc|RawVec|Vec"),
]
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    n += 1
    frames = [f for f in line.split("|") if f]
    leaves[frames[0][:70] if frames else "?"] += 1
    b = "other"
    for name, pat in RULES:
        if any(re.search(pat, f) for f in frames):
            b = name; break
    buckets[b] += 1
print(f"{n} samples")
print("\n== buckets (first matching frame anywhere in the stack, in priority order)")
for k, v in buckets.most_common(): print(f"  {100*v/n:5.1f}%  {k}")
print("\n== leaf frames (top 25)")
for k, v in leaves.most_common(25): print(f"  {100*v/n:5.1f}%  {k}")
EOF
