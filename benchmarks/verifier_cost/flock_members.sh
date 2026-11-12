#!/usr/bin/env bash
# The Lean verifier's seconds a member of a run before its first check, one process a member against one kept process
# (flock_members.py, whose guard holds each kept member's verdict to verify's alone), on statements staged, proved and
# verified as one_stage/m1_unit.py does it, each bound to the program.json and partition.json it writes, on hidden_zk's
# builds (its job.sh, sourced: Flock's CPU prover and this tree's Lean verifier, cached under $FM_CACHE). From the
# source tree:
#   research run --as proofs --on auto --cpus 16 --mem-gb 64 --max-min 150 --cwd source -- \
#     env FM_MEM_GB=64 bash benchmarks/verifier_cost/flock_members.sh
# Per "NAME<TAB>SPEC<TAB>ARGS" line of the file FM_SPECS (ARGS: more m1_unit.py arguments, or none; default: SPECS below,
# an expert GEMM at two widths, whose programs are about one size and whose circuits are not), $OUT/NAME gets m1.log,
# the member's verify call (verify.argv), members.json and each process's output and stage lines (flock_members.py's
# docstring). A spec whose session the verifier refuses is not timed, and the job exits 1.
# env: FM_MEM_GB (the job's --mem-gb)  FM_SPECS  FM_MEMBERS=3  FM_ROUNDS=2  FM_CACHE  FM_PORT=7611  FM_CORE
#      MIMALLOC_PURGE_DELAY (the allocator setting flock_members.py's processes run under; unset: the verifier's own,
#      proofs.py's VERIFIER_ENV)
set -uo pipefail
SPECS=$(cat <<'EOF'
expert-gemm-small	MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=384}"},K=768,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v2"}}
expert-gemm	MoeExpertFp8ReadGemm_v1{READ={"fn":"EmbeddingRead_v1{V=256,H=3072}"},K=6144,N=1,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v2"}}
EOF
)
[ -n "${FM_SPECS:-}" ] && SPECS=$(cat "$FM_SPECS")
export HZ_CONFIGS=flock-members HZ_MEM_GB=${FM_MEM_GB:?}
[ -n "${FM_CACHE:-}" ] && export HZ_CACHE=$FM_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
uv sync -q --all-packages --extra torch-cpu > "$OUT/sync.log" 2>&1 || { tail -20 "$OUT/sync.log"; exit 2; }
PYM=$SRC/.venv/bin/python
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
# the verifier m1_unit.py runs, its verify call recorded an argument a line
RECORDED=$W/verify-recorded
cat > "$RECORDED" <<EOF
#!/usr/bin/env bash
[ "\$1" = verify ] && printf '%s\n' "\$@" > "\${FM_ARGV:?}"
exec "$LEAN" "\$@"
EOF
chmod +x "$RECORDED"
rc=0
while IFS=$'\t' read -r name spec more; do
  [ -n "$name" ] || continue
  O=$OUT/$name
  mkdir -p "$O"
  export FM_ARGV=$O/verify.argv
  log "$name: $spec $more; load $(cut -d' ' -f1-3 /proc/loadavg)"
  # shellcheck disable=SC2086
  "$PYM" benchmarks/one_stage/m1_unit.py --spec "$spec" --out "$W/$name" --bin "$BIN" --lean "$RECORDED" \
    --port "${FM_PORT:-7611}" --n 1 $more < /dev/null > "$O/m1.log" 2>&1
  r=$?
  grep '^M1U ' "$O/m1.log" | cut -c1-700 | tee -a "$OUT/job.log"
  log "$name: files: $(cd "$W/$name" 2> /dev/null && find . -maxdepth 2 -type f \( -name 'program.json' -o -name 'partition.json' -o -name 'circuit*' -o -name 'pub*' \) -printf '%p %s; ')"
  if [ $r = 0 ] && [ -s "$FM_ARGV" ]; then
    "$PYM" benchmarks/verifier_cost/flock_members.py --lean "$LEAN" --argv "$FM_ARGV" --out "$O" \
      --members "${FM_MEMBERS:-3}" --rounds "${FM_ROUNDS:-2}" ${FM_CORE:+--core "$FM_CORE"} < /dev/null 2>&1 | tee -a "$OUT/job.log"
    m=${PIPESTATUS[0]}
    [ "$m" = 0 ] || { log "$name: flock_members.py exited $m"; rc=1; }
    sed -i "s#$W#W#g" "$FM_ARGV"
  else
    log "$name: m1_unit.py exited $r: $(tail -4 "$O/m1.log" | cut -c1-300 | tr '\n' '|')"
    rc=1
  fi
  rm -rf "${W:?}/$name"
done <<< "$SPECS"
log "flock-members: done, rc $rc"
exit "$rc"
