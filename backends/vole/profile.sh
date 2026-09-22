#!/usr/bin/env bash
# Instruction-count profile of the dietmc prover with valgrind --tool=callgrind.
# perf (perf_event_paranoid=4) and ptrace attach (no CAP_SYS_PTRACE) are both unavailable in the vy-cpu
# container, so this is the available profiler: exact instruction counts per function (Ir), not time.
# Instructions are a fair proxy here because the prover is single-threaded scalar integer code; the one
# place they mislead is blocking I/O, which costs no instructions -- the wall-time split in GPU.md s1 comes
# from dietmc's timers, this profile attributes the *compute* within each phase.
#
#   profile.sh <dir-with-relation.txt,instance.txt,witness.txt> [field] [threads]
set -euo pipefail
DIR=$1; FIELD=${2:-F61p}; THREADS=${3:-1}
DIETMC=${DIETMC:-/workspace/target-swanky/release/dietmc}
OUT=${OUT:-$DIR/profile}
PORT=${PORT:-7311}
mkdir -p "$OUT"
printf "lpn = 'medium'\nthreads = %s\nno_batching = false\n" "$THREADS" > "$OUT/dmc.toml"
BASE=("$DIETMC" --text --relation "$DIR/relation.txt" --instance "$DIR/instance.txt" --config "$OUT/dmc.toml" --connection-addr "127.0.0.1:$PORT")
RUST_LOG=info "${BASE[@]}" > "$OUT/verifier.log" 2>&1 &
VPID=$!
sleep 0.5
RUST_LOG=info valgrind --tool=callgrind --callgrind-out-file="$OUT/callgrind.out" --compress-strings=no --compress-pos=no \
  "${BASE[@]}" --witness "$DIR/witness.txt" > "$OUT/prover.log" 2>&1 || true
wait "$VPID" || true
callgrind_annotate --inclusive=no "$OUT/callgrind.out" > "$OUT/annotate_self.txt" 2>/dev/null || true
callgrind_annotate --inclusive=yes "$OUT/callgrind.out" > "$OUT/annotate_incl.txt" 2>/dev/null || true
python3 - "$OUT/annotate_self.txt" <<'EOF'
import re, sys, collections
RULES = [
    ("sVOLE: LPN expansion / spsVOLE / GGM (svole_wykw)", r"svole_wykw|spsvole|ggm|lpn"),
    ("sVOLE: OT extension, base OT, COPEe", r"ot_alsz_kos|kos|chou_orlandi|copee|base_svole|ocelot"),
    ("online: QuickSilver mult/zero checks, commits (homcom/backend)", r"homcom|check_zero|quicksilver|Mac|assert_zero|mult_check|prove_mul|input_private|input_public|backend|DietMacAndCheese|dmc"),
    ("parse: SIEVE-IR text (sieve_ir_parser / circuit_ir / plaintext eval)", r"sieve_ir|text_parser|circuit_ir|parse|Reader|BufRead|read_line|token|Number|number|from_str|Plaintext|plaintext"),
    ("crypto primitives: AES / Blake3 / Ristretto", r"aes|Aes|blake3|curve25519|ristretto|sha"),
    ("field arithmetic (scuttlebutt / swanky_field / F61p)", r"scuttlebutt|swanky_field|F61p|FiniteField|PrimeFiniteField|reduce"),
    ("channel I/O (std::net / std::io / syscalls)", r"std::net|TcpStream|std::io|write_all|read_exact|flush|Channel|syscall|__libc|memcpy|memmove|memset"),
    ("allocator", r"alloc|malloc|free|realloc|RawVec"),
]
total = None; buckets = collections.Counter(); rows = []
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\s*([\d,]+)\s+\(\s*([\d.]+)%\)\s+(.*)", line)
    if not m: 
        t = re.match(r"\s*([\d,]+)\s+\(100\.0%\)\s+PROGRAM TOTALS", line)
        if t: total = int(t.group(1).replace(",", ""))
        continue
    ir = int(m.group(1).replace(",", "")); fn = m.group(3).strip()
    if "PROGRAM TOTALS" in fn: total = ir; continue
    rows.append((ir, fn))
    b = "other"
    for name, pat in RULES:
        if re.search(pat, fn): b = name; break
    buckets[b] += ir
if total is None: total = sum(ir for ir, _ in rows) or 1
print(f"PROGRAM TOTAL Ir = {total:,}")
print("\n== buckets by self instructions (first matching rule on the function name)")
for k, v in buckets.most_common(): print(f"  {100*v/total:5.1f}%  {k}")
print("\n== top 30 functions by self Ir")
for ir, fn in rows[:30]: print(f"  {100*ir/total:5.1f}%  {fn[:150]}")
EOF
