# leg B, one wave on node A: stage WAVE's units in /dev/shm (RAM_WORK=1, at most 600 GB, stopped under 500 GB of the node's
# MemAvailable), the form mirror over what staged, then prove it on CARDS; every file of the wave removed at any exit.
# env: WAVE (N) STAGE_CORES PROVE_CORES CARDS STAGE_NOT_BEFORE STAGE_KILL_AT PROVE_NOT_BEFORE PROVE_KILL_AT (UTC epoch s)
set -u
IN=$RESEARCH_RUN_DIR/inputs; OUT=$RESEARCH_RUN_DIR/out; mkdir -p "$OUT"
FW=/dev/shm/proofs-legB-w$WAVE-${RESEARCH_RUN_ID:-$$}; DW=/workspace/tmp/proofs-legB-w$WAVE-exec-${RESEARCH_RUN_ID:-$$}
trap 'rm -rf "$FW" "$DW"; echo "$(date -u +%H:%M:%SZ) removed $FW and $DW" >> "$OUT/steps.txt"' EXIT
trap 'exit 143' TERM INT
U=$IN/wave$WAVE.units
w=$(( STAGE_NOT_BEFORE - $(date +%s) )); [ "$w" -le 0 ] || { echo "$(date -u +%H:%M:%SZ) sleeping $w s, holding nothing" >> "$OUT/steps.txt"; sleep "$w"; }
uv sync -q --all-packages --extra torch-cpu || exit 3
export RAM_WORK=1 FLOCK_WORK=$FW DISK_WORK=$DW DEV=gpu SM=90 MEM_FLOOR_GB=500 WIN_DISK_GB=600 WIN_FREE_GB=300 \
  WIN_MEM_GB=${WIN_MEM_GB:-1000}
STEPS=stage UNITS=$U KILL_AT=$STAGE_KILL_AT taskset -c "$STAGE_CORES" nice -n 19 bash benchmarks/served_units/stage_prove.sh
rc=$?; [ -d "$FW/data" ] || exit "$rc"
uv run --no-sync python benchmarks/served_units/ks_form.py "$FW" > "$OUT/ks_form.tsv" 2> "$OUT/ks_form.err"
echo "$(date -u +%H:%M:%SZ) ks_form exit $? rows $(($(wc -l < "$OUT/ks_form.tsv") - 1))" >> "$OUT/steps.txt"
uv run --no-sync python "$IN/heads.py" "$FW" > "$OUT/heads.tsv" 2> "$OUT/heads.err"
mkdir -p "$OUT/stage"; for d in "$FW"/data/*/; do cp "$d/stage.json" "$OUT/stage/$(basename "$d").json" 2>/dev/null; done
if [ ! -s "$FW/.staged" ]; then
  # a cut or failed statement leaves no .staged: prove what did stage, under a units file of those alone
  P=$OUT/proved.units; : > "$P"
  awk -F'\t' 'NR > 1 && $5 == "staged" { print $1 "@" }' "$OUT/stage.tsv" | while read -r u; do grep -F "$u" "$U"; done >> "$P"
  n=$(grep -vc '^#' "$P"); [ "$n" -gt 0 ] || { echo "nothing staged to prove" >> "$OUT/steps.txt"; exit 6; }
  echo "$(git rev-parse HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ) units-sha256 $(sha256sum < "$P" | cut -c1-16) staged $n" > "$FW/.staged"
  echo "$(date -u +%H:%M:%SZ) proving the $n of $(grep -vc '^#' "$U") statements that staged (out/proved.units)" >> "$OUT/steps.txt"
  U=$P
fi
awk -F'\t' 'NR > 1 && $6 != "accepted" { print "form mirror refuses " $2 }' "$OUT/ks_form.tsv" >> "$OUT/steps.txt"
STEPS=prove UNITS=$U NOT_BEFORE=$PROVE_NOT_BEFORE KILL_AT=$PROVE_KILL_AT taskset -c "$PROVE_CORES" bash benchmarks/served_units/stage_prove.sh
