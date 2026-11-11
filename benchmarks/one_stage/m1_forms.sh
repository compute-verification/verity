#!/usr/bin/env bash
# M1 on one Definition per form, side by side: each case one unit built, staged, proved under --zk and verified by the Lean
# verifier (m1_unit.py), the verify call timed, on a CPU machine, on hidden_zk's builds (its job.sh, sourced: Flock's
# flock-circuit, the Lean verifier and the prover's flock-tmpl-unit, cached under $MF_CACHE):
#   MF_CACHE=/workspace/research/cache/pouw-hidden-zk bash benchmarks/one_stage/m1_forms.sh
#   (a job run from the source tree, 8 CPUs, 128 GB, keeping out/*)
# A case is "TAG ARGS" (';'-separated in MF_CASES): ARGS are m1_unit.py's, {TU} the flock-tmpl-unit binary; one unit a
# statement, or the --n a case gives.  The default cases are an attention head's first key block and its end at D = 64,
# BN = 192 on Hopper: the word Definitions, whose table reads are inline (`AttnBlockFirst_v1` at 1 and 2 keys,
# `AttnHeadEnd_v1`), and their Boolean versions, which read no table (`AttnBlockFirst_v3` at 1, 35 and 192 keys and
# `AttnHeadEnd_v2`), each flat and structural, and two 35-key blocks in one statement in each form.  The 192-key block
# flat is the case that needs the memory: without it (MF_CASES) 64 GB do.
# Prints "M1F {json}" a case, kept in $OUT/forms.jsonl: circuit_bytes and program_bytes (the files Lean reads); ands and
# unit_rows; the unit's block: unit_cells, the cells its layout takes of its slot of 2^unit_log (a template's with its
# copies aligned), block_bits, where the block's last range ends, of the 2^k_log it is padded to, vus_per_block, and
# block_share, a unit's share of Flock's largest block (2^27 bits);
# build_s and build_rss_gb; prove_s, the prove call's wall; and the Lean verify call's lean_s, lean_rss_gb and lean
# (accepted, or its refusal).  Records: $OUT/<tag>/m1.log.
# env: MF_CASES  MF_MEM_GB=128 (the job's --mem-gb)  MF_CACHE
set -uo pipefail
FIRST='AttnBlockFirst_v%d{D=64,NVIS=%d,BN=192,DOT={"fn":"HopperBF16WgmmaDot16_v%d"}}'
END='AttnHeadEnd_v%d{D=64,INV={"fn":"Fa3InvSum_v%d"}}'
TMPL='--structural iv/hybrid/4 --tmpl-unit-bin {TU}'
if [ -z "${MF_CASES+x}" ]; then
  MF_CASES="word-1 --spec $(printf "$FIRST" 1 1 1);word-2 --spec $(printf "$FIRST" 1 2 1)"
  MF_CASES+=";bits-1 --spec $(printf "$FIRST" 3 1 2);bits-1-tmpl --spec $(printf "$FIRST" 3 1 2) $TMPL"
  MF_CASES+=";bits-35 --spec $(printf "$FIRST" 3 35 2);bits-35-tmpl --spec $(printf "$FIRST" 3 35 2) $TMPL"
  MF_CASES+=";bits-192 --spec $(printf "$FIRST" 3 192 2);bits-192-tmpl --spec $(printf "$FIRST" 3 192 2) $TMPL"
  MF_CASES+=";word-end --spec $(printf "$END" 1 1);bits-end --spec $(printf "$END" 2 2);bits-end-tmpl --spec $(printf "$END" 2 2) $TMPL"
  MF_CASES+=";bits-35-n2 --spec $(printf "$FIRST" 3 35 2) --n 2;bits-35-tmpl-n2 --spec $(printf "$FIRST" 3 35 2) $TMPL --n 2"
fi
export HZ_CONFIGS=m1-forms HZ_MEM_GB=${MF_MEM_GB:-128}
[ -n "${MF_CACHE:-}" ] && export HZ_CACHE=$MF_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
TU=$(cached tmplunit "$(tree_key verity/protocols/flock/verifier verity/protocols/flock/prover/tests/lean/TmplUnit.lean)" build_tmpl_unit)/.lake/build/bin/flock-tmpl-unit
[ -x "$TU" ] || { log "flock-tmpl-unit build FAILED: $(tail -20 "$OUT/build-tmplunit.log")"; exit 1; }
# the Lean verifier with its verify call under timed.py (seconds, peak RSS); its other commands pass through
cat > "$W/lean-timed" <<EOF
#!/bin/sh
[ "\$1" = verify ] || exec "$LEAN" "\$@"
exec "$PY" "$J/timed.py" "\$MF_LEAN_TIME" "$LEAN" "\$@"
EOF
chmod +x "$W/lean-timed"
sha256sum "$TU" | tee -a "$OUT/binaries.sha256"
export RUST_MIN_STACK=$((64 << 20))
rc=0
IFS=';' read -ra FORMS <<< "$MF_CASES"
for form in "${FORMS[@]}"; do
  read -ra args <<< "${form//\{TU\}/$TU}"
  tag=${args[0]}
  mkdir -p "$OUT/$tag"
  log "m1 $tag: ${args[*]:1}"
  ( ulimit -v $(( STAGE_GB * 1024 * 1024 )); export MF_LEAN_TIME="$W/$tag.lean.json"
    "$PY" benchmarks/one_stage/m1_unit.py --out "$W/$tag" --bin "$BIN" --lean "$W/lean-timed" --n 1 "${args[@]:1}" ) > "$OUT/$tag/m1.log" 2>&1
  r=$?
  [ $r = 0 ] || { log "m1 $tag: rc $r"; tail -30 "$OUT/$tag/m1.log" | cut -c1-2000 | tee -a "$OUT/job.log"; rc=1; }
  (cd "$W/$tag" && find . -type f \( -name '*.json' -o -name '*.log' \) -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/$tag/")
  line=$("$PY" - "$tag" "$OUT/$tag" "$W/$tag.lean.json" <<'PYEOF'
import json, os, sys
tag, out, timed = sys.argv[1:]
ph = {}
for line in open(f"{out}/m1.log"):
    if line.startswith("M1U "):
        j = json.loads(line[4:])
        ph[j["phase"]] = j
b, p, v = (ph.get(k, {}) for k in ("built", "proved", "verified"))
t = json.load(open(timed)) if os.path.exists(timed) else {}
prog = f"{out}/stage/program.json"
print(json.dumps({
    "case": tag, "circuit_bytes": b.get("circuit_bytes"), "program_bytes": os.path.getsize(prog) if os.path.exists(prog) else None,
    "ands": b.get("ands"), "unit_rows": b.get("unit_rows"), "unit_cells": b.get("unit_cells"), "unit_log": b.get("unit_log"),
    "block_bits": b.get("block_bits"),
    "k_log": b.get("k_log"), "vus_per_block": b.get("vus_per_block"),
    "block_share": 2.0 ** (b["k_log"] - 27) / b["vus_per_block"] if "k_log" in b else None, "build_s": b.get("seconds"), "build_rss_gb": b.get("rss_gb"), "prove_s": p.get("seconds"), "lean_s": t.get("wall_s"),
    "lean_rss_gb": round(t["maxrss_kb"] / 2**20, 2) if t else None,
    "lean": "accepted" if v.get("accepted") else v.get("why") or "no verdict"}))
PYEOF
  )
  echo "$line" >> "$OUT/forms.jsonl"
  log "M1F $line"
  rm -rf "$W/$tag" "$W/$tag.lean.json"
done
log "m1-forms: done, rc $rc"
exit "$rc"
