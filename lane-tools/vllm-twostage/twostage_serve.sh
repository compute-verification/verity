#!/usr/bin/env bash
# vllm-twostage (VLLM-250): one served window of class G under the two-stage boundary (#8045's C, Tue's BOUNDARY_ROWS
# line), on node A under research run, from the shipped tree's root (a tree holding #8045): a plan of N requests, each the
# first P GLM-5.3 ids of a C4 document holding at least P, decoding O greedy tokens past EOS, served by serve_tax's
# boundary-rows arm (record, prefix caching off) in one burst after the 20 s warm-up, so window w1 holds them all.
# Tue's (c) at 16:00Z.
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd source --send twostage_serve.sh [--send fi_smoke.py] \
#     [--env C4=PATH] [--env END_UTC=ISO [--env NEED_MIN=20] [--env AFTER=RUN]] -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/twostage_serve.sh'
# AFTER: a run on this node whose end (its status.json's state done, failed or cancelled) this waits for before touching
# the GPUs. END_UTC: serve only while NEED_MIN minutes remain before it, else exit 75 having served nothing.
# The engine runs FlashInfer (run.sh's FLASHINFER=1, LDCONFIG=1) under CUDA_HOME=SERVE_CUDA (default /usr/local/cuda-12.9),
# named explicitly: run.sh otherwise takes the newest toolkit, and node A's /usr/local/cuda (13.3) lacks the headers
# FlashInfer's JIT compiles against (cublasLt.h, curand_kernel.h, nvrtc.h), which this checks first. A sent fi_smoke.py
# runs before the engine (run.sh), so FlashInfer's FP8 GEMMs compile outside the served traffic.
# The load stays on the node (Daniel's 1 Nov ruling, VLLM-230): the plan, serve_tax's whole run (its arm's meta, client
# records, step trace, server log and boundary sink, each class's index.json, index-salt.bin and salts.bin, the table and
# engine_env.txt) and its log go to the private dir PRIV=SERVE_ROOT/RUN (default /workspace/research/vllm-twostage,
# outside the run dir, 0700, kept KEEP long by `research keep`), which OUT/private.txt names; serve_tax's run.sh writes
# under RESEARCH_RUN_DIR/out, which this points at PRIV. Custody publishes the run dir: OUT gets only each class's
# commit.json at OUT/boundary/w<window>/p<P>-o<O>/, and stdout only commit.json's public fields (the window, the class,
# its spans N and replay block: the Spec's public population and window) and whether every finished request committed.
# (b)'s replay reads the classes' private files at PRIV/out/boundary-rows/server/boundary/w<window>/p<P>-o<O>/.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
N=${N:-8}; P=${P:-256}; O=${O:-256}
PRIV=${SERVE_ROOT:-/workspace/research/vllm-twostage}/${RESEARCH_RUN_ID:-$(basename "$D")}
case $PRIV in "$D"|"$D"/*) echo "twostage-serve: the private dir $PRIV is in the run dir, which custody publishes"; exit 2 ;; esac
(umask 077; mkdir -p "$PRIV") || { echo "twostage-serve: no private dir $PRIV"; exit 2; }
echo "$PRIV" > "$OUT/private.txt"
left() { echo $(( $(date -u -d "$END_UTC" +%s) - $(date +%s) )); }
if [ -n "${AFTER:-}" ]; then
  st=/workspace/research/runs/$AFTER/status.json; absent=0
  echo "twostage-serve: waiting for $AFTER to end, from $(date -u +%T)"
  while :; do
    s=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state"))' "$st" 2>/dev/null || echo absent)
    case $s in done|failed|cancelled) echo "twostage-serve: $AFTER is $s at $(date -u +%T)"; break ;; esac
    [ "$s" = absent ] && absent=$((absent + 1))
    [ "$absent" -ge 9 ] && { echo "twostage-serve: $AFTER has no status.json on this node after 3 min; going on"; break; }
    if [ -n "${END_UTC:-}" ] && [ "$(left)" -lt $((${NEED_MIN:-20} * 60)) ]; then
      echo "twostage-serve: $AFTER still $s at $(date -u +%T), under ${NEED_MIN:-20} min before $END_UTC: not serving"; exit 75
    fi
    sleep 20
  done
fi
if [ -n "${END_UTC:-}" ] && [ "$(left)" -lt $((${NEED_MIN:-20} * 60)) ]; then
  echo "twostage-serve: $(( $(left) / 60 )) min left before $END_UTC, under ${NEED_MIN:-20}: not serving"; exit 75
fi
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312
PY=${VENV:-$venv}/bin/python
REV=aca966e4e02791568aa6a4ced368624b3d897f42
if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$C4" ] || { echo "twostage-serve: no C4 shard (--send it, or C4=PATH)"; exit 2; }
grep -q "def await_commits" benchmarks/serve_tax/serve_tax.py || { echo "twostage-serve: this tree lacks #8045"; exit 2; }
CUDA=${SERVE_CUDA:-/usr/local/cuda-12.9}
for h in cublasLt.h curand_kernel.h nvrtc.h; do
  [ -f "$CUDA/include/$h" ] || { echo "twostage-serve: $CUDA has no include/$h, which FlashInfer's JIT needs"; exit 2; }
done
if [ -f "$D/inputs/fi_smoke.py" ]; then (umask 077; mkdir -p "$PRIV/inputs") && cp "$D/inputs/fi_smoke.py" "$PRIV/inputs/"; fi
PYTHONPATH=$PWD/tools/research/src python3 -m research keep "$PRIV" --owner @vllm --expires "${KEEP:-7d}" --priority high \
  --why "VLLM-250 (c): the served window's load and each class's index and salts; the replay and the second stage read them" \
  > "$PRIV/keep.log" 2>&1 || echo "twostage-serve: research keep refused (the private dir's keep.log); serving anyway"
echo "twostage-serve: serving from $(date -u +%T), tree $(git rev-parse --short HEAD 2>/dev/null || echo shipped); plan and run in $PRIV"

"$PY" - "$C4" "$CK" "$PRIV/plan-c.jsonl" "$N" "$P" "$O" > "$PRIV/plan.log" 2>&1 <<'EOF' || { echo "twostage-serve: no plan (the private dir's plan.log)"; exit 1; }
import json, random, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/serve_tax")
import serve_tax as S
c4, ck, out, n, p, o = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), *map(int, sys.argv[4:])
k = 400 if p <= 4096 else 400 + n * p // 100
docs = S.load_docs(Path(c4), ck, k)
rows = [d[:p] for d in docs if len(d) >= p][:n]
if len(rows) < n and p > 4096:  # a long prompt: the shard's documents back to back
    flat = [t for d in docs for t in d]
    rows = [flat[i * p:(i + 1) * p] for i in range(n) if (i + 1) * p <= len(flat)]
if len(rows) < n:
    raise SystemExit(f"twostage-serve: the shard's first {k} documents give {len(rows)} prompts of {p} ids, not {n}")
gaps = random.Random("gap_draw/0")  # prod_plan's convention: a unit-rate gap per row, from a stream of its own
out.write_text("".join(json.dumps({"i": i, "prompt_ids": r, "max_tokens": o, "gap_draw": gaps.expovariate(1.0),
                                   "sampling": {"temperature": 0.0, "ignore_eos": True}}, separators=(",", ":")) + "\n"
                       for i, r in enumerate(rows)))
# serve_tax's own reads of a PLAN_FILE row before any request is sent, so a missing key fails here and not after the load
plan = S.plan_of(out, Path(c4), ck, n, 0, 0)
C = S.corpus()
C.traffic.arrivals(plan, 1000.0)
for r in plan:
    C.request_body("served", r, f"plan-{r['i']:05d}", "greedy")
print(f"twostage-serve: plan of {n} requests, {p} prompt ids and {o} out each, in {out}")
EOF

# CANON=1: the canonical DSA top-k (`dsa_topk`'s vLLM plugin), which a row past 2,048 keys needs to pass the circuit's
# top-k; rows of at most 2,048 keys are canonical without it. serve_tax's server_env keeps the caller's PYTHONPATH.
if [ -n "${CANON:-}" ]; then
  "$PY" benchmarks/dsa_topk/dsa_topk.py site "$PRIV/topk" > /dev/null \
    && PYTHONPATH=$PRIV/topk "$PY" -c 'from importlib.metadata import entry_points; import dsa_topk
assert any(e.name == "verity_dsa_topk" for e in entry_points(group="vllm.general_plugins"))' \
    || { echo "twostage-serve: CANON=1 but dsa_topk's site doesn't import"; exit 2; }
  export PYTHONPATH=$PRIV/topk${PYTHONPATH:+:$PYTHONPATH} VERITY_DSA_TOPK=canonical VERITY_DSA_TOPK_LOG=$PRIV/topk-log
fi
echo "twostage-serve: canonical top-k $([ "${VERITY_DSA_TOPK:-}" = canonical ] && echo on || echo off); max model len ${MAX_MODEL_LEN:-4096 (serve_tax default)}"

C4=$C4 RESEARCH_RUN_DIR=$PRIV ARMS=boundary-rows MODEL_ID=glm53 RATES=burst=1000 REQUESTS=$N REPS=1 WINDOW_S=3600 \
  PLAN_FILE=$PRIV/plan-c.jsonl FLASHINFER=1 CUDA_HOME=$CUDA LDCONFIG=1 \
  BOUNDARY_ROWS='{"check":"TokenSelect_v4{V=154880}","inline":true,"per_prompt":0,"per_output":1,"drained":true}' \
  bash benchmarks/serve_tax/run.sh > "$PRIV/serve.log" 2>&1
rc=$?
echo "twostage-serve: serve_tax exit $rc at $(date -u +%T) (its log: the private dir's serve.log)"
B=$PRIV/out/boundary-rows/server/boundary
for c in "$B"/w*/p*-o*; do
  [ -f "$c/commit.json" ] || continue
  k=${c#"$B"/}; mkdir -p "$OUT/boundary/$k" && cp "$c/commit.json" "$OUT/boundary/$k/"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[2])); print("COMMIT", sys.argv[1], {k: d.get(k) for k in ("format","window","prompt","outputs","spans","check","inline","replay")})' \
    "boundary/$k" "$c/commit.json"
done
python3 -c 'import json,sys; print("BOUNDARY-COMMITS complete:", (json.load(open(sys.argv[1])).get("boundary_commits") or {}).get("complete"))' \
  "$PRIV/out/boundary-rows/meta.json" 2>/dev/null || echo "BOUNDARY-COMMITS: no meta.json"
# Each boundary window's advice (a sent advice_c.py, on the CPU): its trace slice, requests.jsonl and `verity-vllm advice`
# output stay in PRIV/advice/w<k>/, and OUT/advice/w<k>/registration.json carries only the binding the first stage's
# --advice reads. The cut refuses a request outside its boundary window's steps; stdout gets only advice_c.py's own lines.
if [ $rc = 0 ] && [ -f "$D/inputs/advice_c.py" ]; then
  (umask 077; CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/integrations/vllm:$PWD:$PWD/tools/research/src "$PY" \
    "$D/inputs/advice_c.py" "$PRIV/out/boundary-rows" "$PRIV/advice" "$OUT" --controls > "$PRIV/advice.log" 2>&1); arc=$?
  grep "^advice-c:" "$PRIV/advice.log" || true
  if [ $arc != 0 ] && ! grep -q "^advice-c: \(the cut does not\|no Commit record\|.* NOT REFUSED\|.* does not round-trip\)" "$PRIV/advice.log"; then
    echo "twostage-serve: advice failed in $(grep -oE '^[A-Za-z_.]+(Error|Exception|Exit|Interrupt)\b' "$PRIV/advice.log" | tail -n 1), through:"
    grep -E '^  File "' "$PRIV/advice.log" | tail -n 4 | sed -E 's/^  File "([^"]*)", line ([0-9]+), in (.*)$/  \1:\2 \3/'
  fi
  echo "twostage-serve: advice exit $arc (its log: the private dir's advice.log)"
fi
keep=(! -path ./private.txt ! -path "./boundary/*/commit.json" ! -path "./advice/*/registration.json")
stray=$(cd "$OUT" && find . -type f "${keep[@]}" | wc -l)
if [ "$stray" -gt 0 ]; then
  mkdir -p "$PRIV/stray" && (cd "$OUT" && find . -type f "${keep[@]}" -print0 \
    | xargs -0 -r cp --parents -t "$PRIV/stray" && find . -type f "${keep[@]}" -delete)
  echo "twostage-serve: moved $stray other file(s) from OUT to the private dir's stray/"
fi
exit $rc
