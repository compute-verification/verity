#!/usr/bin/env bash
# The CPU half of the NciCheck.Desc impact study: the recorded computations that gave NCI verdicts, rerun with the
# binaries lean.sh built (sent as nci-check-TREE and nci-desc-fuzz-TREE, TREE one of pre, main, fix), from the
# checkout's root on vy-nebius-cpu-3/4/5:
#
#   research run --on vy-nebius-cpu-N --project verity --source . --cwd clone --send nci-check-main ... -- \
#     bash benchmarks/verifier_cost/desc_impact/cpu.sh MODE TREE...
#
# MODE, each TREE in parallel into $OUT/MODE-TREE unless said:
#   serve    run.py measure on Serve_v1 (one IU per Call; --policy, and --policy --unrolled up to 4e7 gates)
#   glm      nci_glm.py measure --case registered --case rank --case slice (sent: glm53-b1-pins.json,
#            glm53-b1-rank<r>.descriptor.json.gz)
#   node     nci_glm.py's node case (the eight ranks joined) at A alone (node_a.py; sent as for glm)
#   regnode  nci_glm.py measure --case registered-node --node-points full: the check of record (sent:
#            glm53-b1-node.descriptor.json.gz, .trusted.json, .provenance.json, .registration.json)
#   step     the step node of r20261024-003544-5439: its m4_desc.py instance, then nci-check --policy (sent: m4_desc.py,
#            step-node.descriptor.json.gz, step-node.json), every TREE in turn
#   shared   shared.py on #4066's difftest data (sent: instances, expanded, python .jsonl.gz), every TREE at once
#   fuzz     the falsification search (fuzz/gen): the hand family, the difftest vectors of the fix tree (its 12
#            shared ones among them), random, copies, big and chain, each TREE's nci-desc-fuzz in turn on the same
#            Programs (FUZZ_RANDOM FUZZ_COPIES FUZZ_BIG FUZZ_CHAIN FUZZ_EXTRA FUZZ_SEED override the sizes)
set -uo pipefail
O=${OUT:?}
IN=${RESEARCH_RUN_DIR:?}/inputs
MODE=$1
shift
TREES="$*"
H=$(pwd)
I=benchmarks/verifier_cost/desc_impact
N=verity/core/protocols/network/nci/lean
F=verity/core/protocols/network/nci/fuzz
MAIN=14acc70d97
NODE_DIGEST=21cad21ecadbd3d7db3c923441246e27c26665d206760a63f79512f4141e04f158555fdbae1a08cea5abb084958f57589aa51bdb4f312eb5c6a024cd419a18a5
d=$(mktemp -d)
trap 'git worktree remove --force "$d/fix" 2>/dev/null; rm -rf "$d"' EXIT
git log --oneline -1 > "$O/head.txt"
for t in $TREES; do
  for e in nci-check nci-desc-fuzz; do
    [ -e "$IN/$e-$t" ] || continue
    cp "$IN/$e-$t" "$d/$e-$t" && chmod +x "$d/$e-$t"
    sha256sum "$d/$e-$t" >> "$O/head.txt"
  done
done
case $MODE in serve|glm|node|regnode|fuzz)
  uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -40 "$O/sync.log"; exit 1; } ;;
esac
PY=$H/.venv/bin/python
RANKS=(--pins "$IN/glm53-b1-pins.json" --rank "$IN"/glm53-b1-rank{0..7}.descriptor.json.gz)

one() { # TREE: MODE's computation with TREE's nci-check
  local t=$1 o=$O/$MODE-$1 x=$d/nci-check-$1
  mkdir -p "$o" "$d/s-$t"
  case $MODE in
    serve) "$PY" benchmarks/verifier_cost/run.py measure --exe "$x" --out "$o" --unrolled-max 40000000 ;;
    glm) "$PY" benchmarks/verifier_cost/nci_glm.py measure --exe "$x" --out "$o" --scratch "$d/s-$t" "${RANKS[@]}" \
           --case registered --case rank --case slice --jobs 4 ;;
    node) "$PY" "$I/node_a.py" measure --exe "$x" --out "$o" --scratch "$d/s-$t" "${RANKS[@]}" --case node --jobs 1 ;;
    regnode) "$PY" benchmarks/verifier_cost/nci_glm.py measure --exe "$x" --out "$o" --scratch "$d/s-$t" \
               --case registered-node --node "$IN/glm53-b1-node.descriptor.json.gz" --node-digest "$NODE_DIGEST" \
               --node-trusted "$IN/glm53-b1-node.trusted.json" --node-provenance "$IN/glm53-b1-node.provenance.json" \
               --node-registration "$IN/glm53-b1-node.registration.json" --node-points full --jobs 1 ;;
  esac > "$o/measure.log" 2>&1
  local rc=$?
  echo "$MODE $t rc=$rc"
  tail -6 "$o/measure.log"
  return $rc
}

rc=0
case $MODE in
  serve|glm|node|regnode)
    pids=()
    for t in $TREES; do one "$t" > "$O/$MODE-$t.txt" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    cat "$O"/$MODE-*.txt ;;
  shared)
    for n in instances expanded python; do gunzip -c "$IN/$n.jsonl.gz" > "$d/$n.jsonl"; done
    args=()
    for t in $TREES; do args+=(--exe "$t" "$d/nci-check-$t"); done
    "$PY" "$I/shared.py" "$d" "${args[@]}" --out "$O/shared.json" | tee "$O/shared.txt" || rc=1 ;;
  step)
    python3 "$IN/m4_desc.py" "$IN/step-node.descriptor.json.gz" "$IN/step-node.json" "$d/step.json" | tee "$O/step-instance.json" || exit 1
    for t in $TREES; do
      "$d/nci-check-$t" --policy "$d/step.json" > "$O/step-$t.json" || rc=1
      echo "$t: $(cat "$O/step-$t.json")"
    done ;;
  fuzz)
    G=$d/fix/$F/gen
    git worktree add -q "$d/fix" $MAIN && git -C "$d/fix" apply "$H/$I/fix-4066.patch" && cp -r "$I/fuzz" "$d/fix/$F" || exit 1
    S=$d/programs
    mkdir -p "$S"
    SEED=${FUZZ_SEED:-6000000}
    jobs=$(nproc); (( jobs > 32 )) && jobs=32
    shards() { # family count mutate seed
      local fam=$1 n=$2 mut=$3 seed=$4 per i
      (( n > 0 )) || return 0
      per=$(( (n + jobs - 1) / jobs ))
      for ((i = 0; i * per < n; i++)); do
        local cnt=$per
        (( i * per + per > n )) && cnt=$(( n - i * per ))
        python3 "$G/programs.py" "$fam" --seed $(( seed + i * per )) --count "$cnt" --mutate "$mut" --out "$S/$fam.$i.jsonl" &
      done
      wait
      cat "$S/$fam".*.jsonl > "$S/$fam.jsonl" && rm -f "$S/$fam".*.jsonl
    }
    python3 "$G/programs.py" hand --out "$S/hand.jsonl"
    "$PY" "$G/vectors.py" --out "$S/vectors.jsonl" --extra "${FUZZ_EXTRA:-2000}" > "$O/vectors.log" 2>&1 || { tail -20 "$O/vectors.log"; exit 1; }
    shards random "${FUZZ_RANDOM:-300000}" 0.3 "$SEED"
    shards copies "${FUZZ_COPIES:-100000}" 0.2 "$SEED"
    shards big "${FUZZ_BIG:-30000}" 0.3 9000000
    shards chain "${FUZZ_CHAIN:-20000}" 0.2 "$SEED"
    wc -l "$S"/*.jsonl | tee "$O/programs.txt"
    for t in $TREES; do
      mkdir -p "$O/fuzz-$t"
      for f in hand vectors random copies big chain; do
        [ -s "$S/$f.jsonl" ] || continue
        s=$SECONDS
        "$d/nci-desc-fuzz-$t" "$S/$f.jsonl" "$d/$t-$f.out.jsonl" > /dev/null || rc=1
        echo "$t $f: $(wc -l < "$S/$f.jsonl") programs, $((SECONDS - s)) s"
        grep '"program"' "$d/$t-$f.out.jsonl" | gzip > "$O/fuzz-$t/$f.failures.jsonl.gz"
      done
      python3 "$G/summarize.py" "$d/$t"-*.out.jsonl --json "$O/fuzz-$t/summary.json" > "$O/fuzz-$t/summary.txt" 2>&1 || rc=1
      rm -f "$d/$t"-*.out.jsonl
    done ;;
  *) echo "cpu.sh: unknown MODE $MODE"; exit 2 ;;
esac
exit $rc
