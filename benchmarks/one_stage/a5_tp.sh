#!/usr/bin/env bash
# One-stage audit, run A5, over a TP Commit's windows, one per rank (a5.py --tp-commit), on a CPU machine: each rank's served
# window through registration, the verifier's Lean draw, M0's prover and the Lean verifier of record. M0's flock-circuit and
# Lean's flock-verify are built from this tree, in scratch. A5 takes a TP window as served only under a SHA-512 TP run root
# (tp.serving_rows.source: sha512): a TP Commit's ranks, the stand-in's too, commit under rows-sha512 under the SHA-512 TP run
# root (tp.run_root, verity/tp-run-root/v2); an old record's SHA-256 v1 root is refused before the draw.
#   GLM-5.3 TP8: A5_TP=DIR holds the TP Commit's record (commit/runs.jsonl, commit/versions.json) and its windows (serving-rows/
#     rank<r>/), A5_BUILD=DIR each rank's Programs of record (rank<r>/); the workload (integrations/vllm/workloads/$ROW.json) and
#     the partition file (A5_PARTITION, default glm53_silu_mul_v2_tp8.json) are this tree's. A5_WINDOWS=bundles: the Commit wrote no
#     window and deferred its ranks' replay to sealed bundles (A5_TP/commit/replay_bundle_p0_rank<r>, as Commit 38b2), so each
#     rank's stand-in window is made here first from its bundle (`verity-vllm tp-serving-rows`, up to A5_JOBS ranks at once);
#     A5_CHECKPOINT=DIR opens the registered tensors the bundles don't carry (a MoE partition's expert stacks) from the checkpoint
#     of record, without its shard sha256 (each tensor read is held to the committed weights root).
#   The stand-in (no A5_TP): the CPU TP Commit tests/pipeline/served_tp, world 2, once per STANDIN entry I:LAW (its SiluMul_v2{I}
#     rows, its partition's law); with A5_WINDOWS=bundles it is served_tp --deferred and its windows come from its bundles.
#     The STANDIN entry sparse-mla is a5_sparse_mla.py: test_serving_rows_splits' window (one SparseMlaSplit row's split parts
#     and combines, each combine slot wired to its part), run as the entry row-tables is.
#     The STANDIN entry row-tables is a5_row_tables.py: test_row_tables' window (ScaledMmS8Coordinate chunks whose x rows link
#     its row tables) as rank 0 of a one-rank step, on a local challenger's coins (it needs A5_CHALLENGER for --main).
#     The STANDIN entries warden and warden:KIND are a5_warden.py: transfer-gate chunks whose ports open the bucket rows of
#     one connection's cells under the warden's signed roots, beside a drawn rest, on a warden run rehearsed here first (warden_serve.py
#     rehearse --fake, about a minute, its prompts up to 2048 tokens so that seed 0's traffic has a connection admitted in its
#     link-window with a5_warden.py's 16 ingress cells and D·r + 16 egress cells; only its public deployment.json, cells.jsonl
#     and check.json are kept), on a local challenger's coins (A5_CHALLENGER=local) or the verifier's own (A5_COINS=verifier);
#     warden:altered reads one cell a bit off, and warden:pad, warden:pad-named and warden:swapped make the egress chunk's
#     statement for another units' wiring (--made-for); each is to be refused by Lean's registered reads. A5_WARDEN_RUN=DIR
#     audits that warden run instead of rehearsing one (e.g. warden_replay.py's, a served window's traffic through the
#     warden), A5_WARDEN_UNITS=N each transfer-gate chunk's units (a5_warden.py --units, default 16).
#     The STANDIN entries tp-gates and tp-gates:KIND are a5_tp_gates.py: both ranks of a synthetic Llama TP2 step of two
#     requests, each rank's AllReduce2 chunks reading both ranks' registered partials of each request (layer 0's embedding,
#     o_proj and down_proj at H = 2048), beside a drawn rest, on the verifier's own coins (A5_COINS=verifier); tp-gates:altered
#     moves a bit through rank 1's received partial, and tp-gates:swapped, tp-gates:other-collective, tp-gates:other-request
#     and tp-gates:own make that port's statement for another read (--made-for); each is to be refused by Lean's registered
#     reads on rank 1, rank 0 accepted.
#   A5_FLOCK_WORK=DIR keeps M0's build there across runs (default: scratch); A5_RANKS=0,3 audits those ranks only; A5_JOBS=N
#   audits up to N ranks at once (a GLM-5.3 rank, SiluMul_v2 at step 16: its window ~16 min and ~11.7 GB, its audit ~10 GB;
#   the MoE w2 coordinates of one layer, glm53_moe_w2_tp8_step16.json: its window ~4 min and ~13.1 GB, then ~2.5 min of the
#   verifier's window, ~5 min staging its 768 chunks' public files and ~2.5 min to the draw (a drawn chunk's prover file is
#   staged after it), then each drawn chunk's session ~8 min and ~32 GB, which a loaded machine
#   stretched past an hour; a5's --session-timeout, A5_SESSION_TIMEOUT, is each prover's limit and each compose's).
#   bash benchmarks/one_stage/a5_tp.sh
#   (a job run from a clone of the tree, 64 GB)
# Kept in $OUT: a5's records and logs, the public files, the sessions (their proofs and records), the binaries' sha256 and the
# circuits' pins; never a prover's file (inst-*, m*-inst-*, *-private.bin: the rows and their salts) nor a circuit (its pin is).
# Stored: each session's circuit.txt files and lookup tables, by SHA-512 (sha512/<hex>, the Lean verifier's --archive layout), as one
#   fixture/v1 tree with the run's custody key (integrations/vllm/verity_vllm/ops/store_trees.sh), the run labelled `circuits` = its
#   art id and $OUT/store/NAME.sha512 naming each file's key, so the run can be verified again from the store. A session whose
#   tree isn't stored and preserved fails the run.
# A5_DRY=DIR: none of the builds, windows or audits: each session dir DIR/NAME a run's scratch held (a5's --out) is kept and
#   stored as a run's is.
# A5_CHALLENGER=local|gate:USER@HOST: each rank draws by its partition file's rule (stratified:K among them) from the auditor's
#   challenger's coins through its own service.Stream, the proof system's path of record, its outcome held to the Lean's run;
#   one session on the TP run root (a5.py --challenger, a5_coins.py): local, a development session on this
#   machine; gate:USER@HOST, the challenger host through infra's gate, with A5_PEER (this machine in its /etc/vy/peers),
#   A5_CHALLENGER_KEY (the challenger's key record) and A5_GATE_KEY (the lane's gate key). The step's ranks draw from the one
#   session, so A5_JOBS covers every audited rank. It opens once every rank has passed its checks before the draw, with
#   coins sized from every rank's draw, and not at all if one hasn't. The session's files, the ledger and the host's pins log are kept in
#   $OUT/<name>/challenger/. The verifier's own copy of main's first-parent history (a5.py --main) is A5_MAIN, else this
#   clone's fetch of origin/main ($OUT/main.txt, as benchmarks/remote_session/auditor.sh writes it).
# A5_COINS=verifier (instead of A5_CHALLENGER): each rank draws by its partition file's rule through its own service.Stream
#   on the verifier's own live coins (a5.py --coins verifier: the Lean's IO.getRandomBytes, read once the stream received
#   the rank's registration), no challenger session, so A5_JOBS takes any number of ranks. Its record says source: verifier.
# A5_COMPOSE_CACHE=DIR: this machine's composed circuits (a5.py --compose-cache, compose_circuit.py --cache), so that the ranks
#   audited here, in this run or another of the same tree, compose each chunked template once, and every later rank takes the
#   kept circuit with its own partition and program (GLM-5.3's all-families draw composes ~95 min a rank without it).
# A5_ADVICE=workload (with A5_WINDOWS=bundles, the stand-in or A5_TP): advice windows. The advice A (I1, the schedule of the
#   fixed-length workload the Commit served, `advice_windows.advice_of_workload`) and A' (A less its last request) are written
#   in scratch, each rank's window is committed from its bundle over the stand-in compiler's Programs for A, and for A', its
#   registration binding the advice's hiding root (`verity-vllm advice-windows`), and a5 derives its own from the record, the
#   advice's opening and the compiler, never the workload (a5.py --advice-opening; --other-advice-opening and --other-windows
#   its negatives). The advice and its openings are the prover's: never kept.
#   Another TP Commit than GLM-5.3's: A5_ROW its row (the workload integrations/vllm/workloads/$A5_ROW.json), A5_NAME the
#   audit's name (default glm), A5_SETS the SiluMul_v2 widths its partition reads (default "256 1536"), e.g. LLAMA32_1B TP2:
#   A5_ROW=llama32-1b__bf16__rtxpro6000__tp2__b2__i256__o32__mixed__greedy__bi-eager A5_PARTITION=benchmarks/one_stage/
#   llama32_1b_silu_mul_v2_tp2.json A5_SETS=4096.  A rank's proved population must keep C-Flock's m = k_log + log2(units) <= 35:
#   a SiluMul_v2{I=4096} unit is 2^25 bits, so that partition's scope keeps layer 0's decode steps 1..7 (8 units a rank, not 4,736).
#   An A5_SETS entry with a / is a subcircuit spec of its own (rope-head/d64/bf16), any other a SiluMul_v2 width.
# A5_TRANSFER_GATES="T ..." (with A5_ADVICE): the templates the stand-in compiler lists as holding transfer gates, every unit of
#   theirs proved (selection All) beside the rest drawn by the partition's work:K (advice_windows: `verity-vllm advice-windows
#   --transfer-gate`, a5.py --transfer-gate and its transfer-gate negatives). The stand-in's partition holds them before its
#   SiluMul_v2{I} (served_tp --template), e.g. STANDIN=256:work:2 A5_TRANSFER_GATES='RoPEHead_v1{D=64}'
#   A5_SETS='256 rope-head/d64/bf16': the toy's 32 RoPE heads a rank proved whole beside 2 of its 8 SiluMul rows.
set -uxo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p $O
PY=${PY:-$REPO/.venv/bin/python}; VV=${VV:-$REPO/.venv/bin/verity-vllm}
if [ -n "${A5_DRY:-}" ]; then W=$A5_DRY; else W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; fi

STORE_RC=0
store() {  # NAME: $W/NAME's circuit.txt files and lookup tables by SHA-512, one tree through store_trees.sh, the run labelled with it
  local t=$W/store/$1 f h rc
  rm -rf $t; mkdir -p $t/sha512 $O/store
  (cd $W/$1 && find . -type f \( -name circuit.txt -o -path '*/tables/*.u32' \) ! -path '*/archive/*' -print0 | sort -z) \
    | while IFS= read -r -d '' f; do
        h=$(sha512sum "$W/$1/$f" | cut -d' ' -f1); [ -e $t/sha512/$h ] || ln "$W/$1/$f" $t/sha512/$h; echo "$h ${f#./}"
      done > $O/store/$1.sha512
  [ -s $O/store/$1.sha512 ] || { echo "STORED $1 none: no circuit under $W/$1"; rm -rf $t; return 0; }
  $PY -c 'import json, sys; n, run, src, keys = sys.argv[1:5]
print(json.dumps({"name": f"a5-circuits/{n}", "run": run, "source": src, "layout": "sha512/<hex>, the Lean verifier --archive",
                  "files": dict(reversed(x.split(" ", 1)) for x in open(keys).read().splitlines())}, sort_keys=True))' \
    $1 "${RESEARCH_RUN_ID:-}" "$(git rev-parse HEAD)" $O/store/$1.sha512 > $O/store/meta-$1.json
  EV=$O/store VV=$VV LABEL_KEY=circuits LABEL_BY=proofs bash $REPO/integrations/vllm/verity_vllm/ops/store_trees.sh $W/store $1 > $O/store/$1.log 2>&1
  rc=$?
  grep -E '^(STORED|PRESERVED-CHECK|LABEL|no custody)' $O/store/$1.log
  rm -rf $t; rmdir $W/store 2>/dev/null
  return $rc
}
keep() {  # NAME: $W/NAME's records into $O/NAME (never the MUFU tables a circuit with lookups reads: ir_lower.TABLES pins them), and
          # its circuits and tables into the store
  mkdir -p $O/$1
  (cd $W/$1 && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' ! -path './*/archive/*' \
    ! -path '*/tables/*.u32' -print0 \
    | xargs -0 -r -I{} cp --parents {} $O/$1/)
  store $1 || STORE_RC=1
}
if [ -n "${A5_DRY:-}" ]; then
  for d in $W/*/; do n=$(basename $d); [ $n = store ] || keep $n; done
  exit $STORE_RC
fi

ROW=${A5_ROW:-glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager}
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock HEAD:benchmarks/numerical; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }

(cd verity/core/service/flock && lake build flock-verify flock-firewall) > $O/build-lean.log 2>&1 &
LP=$!
mkdir -p $O/m0-build
FW=${A5_FLOCK_WORK:-$W}   # a kept build dir: 20-gpu-link.sh re-applies this tree's patches and cargo rebuilds what changed
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > $O/m0-build/build.log 2>&1 \
  || { tail -40 $O/m0-build/build.log; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $O/build-lean.log; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify; FWL=${LEAN%/*}/flock-firewall
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > $O/build-sampled-proofs.log 2>&1 || { tail -40 $O/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN $FWL | tee $O/binaries.sha256

# the verifier's lowering: a synthetic input set per template (its values are never read)
for I in ${A5_SETS:-256 1536}; do
  case $I in */*) SUB=$I D=${I//\//_};; *) SUB=silu-mul-v2/i$I/bf16 D=i$I;; esac
  $PY -m verity_numerical.bench.generate --subcircuit $SUB --n 4 --seed 1 --out $W/sets/$D > $O/sets-$D.log 2>&1 || exit 9
done
TG=(); TPL=()
for t in ${A5_TRANSFER_GATES:-}; do TG+=(--transfer-gate "$t"); TPL+=(--template "$t"); done
[ ${#TG[@]} -eq 0 ] || [ -n "${A5_ADVICE:-}" ] || { echo "A5_TRANSFER_GATES lists the advice compiler's transfer gates: set A5_ADVICE=workload"; exit 3; }
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
MAIN=${A5_MAIN:-}
if [ -n "${A5_CHALLENGER:-}" ] && [ -z "$MAIN" ]; then
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  # a run's clone (--cwd clone) has the shipped tree as its origin, which holds main only as its own origin/main (the node's
  # bare repository's main, fed by the launcher's origin/main)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > $O/fetch-main.log 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> $O/fetch-main.log 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > $O/main.txt || { tail -5 $O/fetch-main.log; exit 4; }
  MAIN=$O/main.txt; echo "main $(head -1 $MAIN), $(grep -c . $MAIN) first-parent commits" | tee -a $O/fetch-main.log
fi

TIME=$([ -x /usr/bin/time ] && echo "/usr/bin/time -v")
audit() {  # NAME TP_COMMIT BUILD WORKLOAD WINDOWS PARTITION [A5 ARGS...]: WORKLOAD empty under --advice
  mkdir -p $O/$1
  $TIME $PY benchmarks/one_stage/a5.py --tp-commit $2 --build-root $3 ${4:+--workload $4} --windows $5 --partition $6 "${@:7}" \
    ${A5_RANKS:+--ranks $A5_RANKS} ${A5_JOBS:+--jobs $A5_JOBS} ${A5_SESSION_TIMEOUT:+--session-timeout $A5_SESSION_TIMEOUT} --template-sets $W/sets --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --out $W/$1 \
    ${A5_CHALLENGER:+--challenger $A5_CHALLENGER} ${A5_COINS:+--coins $A5_COINS} ${A5_PEER:+--peer $A5_PEER} ${A5_CHALLENGER_KEY:+--challenger-key $A5_CHALLENGER_KEY} ${A5_GATE_KEY:+--gate-key $A5_GATE_KEY} \
    ${MAIN:+--main $MAIN} ${A5_COMPOSE_CACHE:+--compose-cache $A5_COMPOSE_CACHE} > $O/$1/a5.log 2> $O/$1/a5.err
  local rc=$?
  grep '^A5' $O/$1/a5.log | cut -c1-4000; tail -25 $O/$1/a5.err
  keep $1
  return $rc
}

public() {  # WINDOWS DEST: serving's public files of every rank
  for d in $1/rank*; do
    mkdir -p $2/$(basename $d) && cp $d/{registration,window,partition,index}.json $d/m*-public.bin $2/$(basename $d)/
  done
  cp $1/tp.json $2/ 2>/dev/null
}

bundle_windows() {  # COMMIT BUILD PARTITION OUT LOGS: each rank's stand-in window from its sealed bundle
  local ranks=${A5_RANKS:-$($PY -c "from verity_vllm.pipeline.tp import serving_rows as S; print(','.join(map(str, range(S.record('$1')['world']))))")}
  mkdir -p $5
  echo ${ranks//,/ } | tr ' ' '\n' | xargs -P ${A5_JOBS:-1} -I{} sh -c \
    "$TIME $REPO/.venv/bin/verity-vllm tp-serving-rows $1 --build-root $2 --partition $3 --out $4 --ranks {} \
      ${A5_CHECKPOINT:+--checkpoint-dir $A5_CHECKPOINT --no-verify-shards} > $5/tp-serving-rows-{}.log 2>&1" \
    || { tail -n 20 $5/tp-serving-rows-*.log; return 1; }
  grep -h '^\[tp-serving-rows\]' $5/tp-serving-rows-*.log
}

advice_windows() {  # COMMIT BUILD PARTITION OUT ADVICE LOGS [ARGS...]: each rank's stand-in window from its sealed bundle over the
                    # stand-in compiler's Programs for the advice, its opening written to ADVICE.opening.json
  local log=$6/advice-windows-$(basename $4).log
  mkdir -p $6
  $TIME $REPO/.venv/bin/verity-vllm advice-windows $1 --build-root $2 --partition $3 --out $4 --advice $5 --opening ${5%.bin}.opening.json \
    ${A5_RANKS:+--ranks $A5_RANKS} "${@:7}" > $log 2>&1 || { tail -20 $log; return 1; }
  grep -h '^\[advice-windows\]' $log
}

rc=0
PART=${A5_PARTITION:-benchmarks/one_stage/glm53_silu_mul_v2_tp8.json}
advice_pair() {  # COMMIT BUILD PARTITION WORKLOAD WINDOWS LOGS: windows under the workload's advice A into WINDOWS and under A'
                 # (A less its last request) into other-windows beside it, the advice and openings there too; sets ADV, a5's
                 # advice arguments
  [ "$A5_ADVICE" = workload ] && [ "${A5_WINDOWS:-served}" = bundles ] \
    || { echo "A5_ADVICE=workload commits each rank's window from its bundle: set A5_WINDOWS=bundles"; exit 3; }
  local d=$(dirname $5) last=$($PY -c "import json, sys; print('r%d' % (len(json.load(open(sys.argv[1]))['requests']) - 1))" $4)
  advice_windows $1 $2 $3 $5 $d/advice.bin $6 --write-from-workload $4 "${TG[@]}" || exit 8
  advice_windows $1 $2 $3 $d/other-windows $d/other-advice.bin $6 --write-from-workload $4 --drop $last "${TG[@]}" || exit 8
  ADV=(--advice-opening $d/advice.opening.json --other-advice-opening $d/other-advice.opening.json --other-windows $d/other-windows "${TG[@]}")
  public $d/other-windows $6/other-window
}

if [ -n "${A5_TP:-}" ]; then
  N=${A5_NAME:-glm}; WIN=$A5_TP/serving-rows; WL=integrations/vllm/workloads/$ROW.json; ADV=()
  $REPO/.venv/bin/verity-vllm workload-file $WL > /dev/null || exit 8
  if [ -n "${A5_ADVICE:-}" ]; then
    WIN=$W/serving-rows
    advice_pair $A5_TP/commit ${A5_BUILD:-$A5_TP/build} $PART $WL $WIN $O/$N; WL=
  elif [ "${A5_WINDOWS:-served}" = bundles ]; then
    WIN=$W/serving-rows
    bundle_windows $A5_TP/commit ${A5_BUILD:-$A5_TP/build} $PART $WIN $O/$N || exit 8
  fi
  public $WIN $O/$N/window
  audit $N $A5_TP/commit ${A5_BUILD:-$A5_TP/build} "$WL" $WIN $PART "${ADV[@]}" || rc=1
else
  for e in ${STANDIN:-256:subset:4 1536:subset:4}; do   # a rank's stand-in window holds 8 units; a draw of all leaves no negative
    if [ "$e" = row-tables ] || [ "$e" = sparse-mla ]; then
      [ -n "$MAIN" ] || { echo "STANDIN $e draws from a challenger's coins: set A5_CHALLENGER=local"; exit 3; }
      mkdir -p $O/$e
      $TIME $PY benchmarks/one_stage/a5_${e//-/_}.py --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --main $MAIN \
        ${A5_SESSION_TIMEOUT:+--session-timeout $A5_SESSION_TIMEOUT} ${A5_COMPOSE_CACHE:+--compose-cache $A5_COMPOSE_CACHE} \
        --out $W/$e > $O/$e/a5.log 2> $O/$e/a5.err || rc=1
      grep '^A5' $O/$e/a5.log | cut -c1-4000; tail -25 $O/$e/a5.err
      keep $e
      continue
    fi
    if [ "${e%%:*}" = warden ]; then
      case $e in warden) REFUSE=();; warden:altered) REFUSE=(--altered);; warden:pad|warden:pad-named|warden:swapped) REFUSE=(--made-for ${e#warden:});;
        *) echo "STANDIN $e: warden, warden:altered, warden:pad, warden:pad-named or warden:swapped"; exit 3;; esac
      if [ -n "${A5_COINS:-}" ]; then DRAW=(--coins $A5_COINS)
      elif [ "${A5_CHALLENGER:-}" = local ]; then DRAW=(--challenger local --main $MAIN)
      else echo "STANDIN $e draws through its stream: set A5_CHALLENGER=local or A5_COINS=verifier"; exit 3; fi
      N=${e/:/-}; mkdir -p $O/$N
      if [ -n "${A5_WARDEN_RUN:-}" ] && [ ! -d $W/warden-run ]; then
        mkdir -p $O/warden-run && cp $A5_WARDEN_RUN/{deployment.json,cells.jsonl,check.json} $O/warden-run/ || exit 8
        ln -s $A5_WARDEN_RUN $W/warden-run
      fi
      if [ ! -d $W/warden-run ]; then
        (cd benchmarks/network_traces && $PY warden_serve.py rehearse --fake --out $W/warden-run --requests 6 --rate 1 --tail 5 \
          --max-tokens 64 --prompt-max 2048 --start-in 2) > $O/warden-rehearse.log 2>&1; r=$?
        mkdir -p $O/warden-run && cp $W/warden-run/{deployment.json,cells.jsonl,check.json} $O/warden-run/
        [ $r = 0 ] || { tail -20 $O/warden-rehearse.log; $PY -c 'import json, sys; print(*json.load(open(sys.argv[1])).get("problems", []), sep="\n")' \
          $O/warden-run/check.json; exit 8; }
      fi
      $TIME $PY benchmarks/one_stage/a5_warden.py --run $W/warden-run --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL \
        "${DRAW[@]}" "${REFUSE[@]}" ${A5_SESSION_TIMEOUT:+--session-timeout $A5_SESSION_TIMEOUT} ${A5_WARDEN_UNITS:+--units $A5_WARDEN_UNITS} \
        --out $W/$N > $O/$N/a5.log 2> $O/$N/a5.err || rc=1
      grep '^A5' $O/$N/a5.log | cut -c1-4000; tail -25 $O/$N/a5.err
      keep $N
      continue
    fi
    if [ "${e%%:*}" = tp-gates ]; then
      case $e in tp-gates) REFUSE=();; tp-gates:altered) REFUSE=(--altered);;
        tp-gates:swapped|tp-gates:other-collective|tp-gates:other-request|tp-gates:own) REFUSE=(--made-for ${e#tp-gates:});;
        *) echo "STANDIN $e: tp-gates, tp-gates:altered, tp-gates:swapped, tp-gates:other-collective, tp-gates:other-request" \
             "or tp-gates:own"; exit 3;; esac
      [ "${A5_COINS:-}" = verifier ] || { echo "STANDIN $e draws both ranks on the verifier's own coins: set A5_COINS=verifier"; exit 3; }
      N=${e/:/-}; mkdir -p $O/$N
      $TIME $PY benchmarks/one_stage/a5_tp_gates.py --m0-pythonpath $M0PATH --bin $BIN --lean $LEAN --firewall $FWL --coins verifier \
        "${REFUSE[@]}" ${A5_SESSION_TIMEOUT:+--session-timeout $A5_SESSION_TIMEOUT} --compose-cache ${A5_COMPOSE_CACHE:-$W/compose}/tp-gates \
        --out $W/$N > $O/$N/a5.log 2> $O/$N/a5.err || rc=1
      grep '^A5' $O/$N/a5.log | cut -c1-4000; tail -25 $O/$N/a5.err
      keep $N
      continue
    fi
    I=${e%%:*}; LAW=${e#*:}; S=$W/served-i$I
    DEFERRED=$([ "${A5_WINDOWS:-served}" = bundles ] && echo --deferred)
    (cd integrations/vllm && $PY -m tests.pipeline.served_tp $S --world 2 --I $I --law $LAW $DEFERRED "${TPL[@]}") \
      > $O/standin-i$I.json 2> $O/standin-i$I.err || { tail -20 $O/standin-i$I.err; exit 8; }
    WL=$S/workload.json; ADV=()
    if [ -n "${A5_ADVICE:-}" ]; then
      advice_pair $S/commit $S/build $S/partition.json $WL $S/serving-rows $O/standin-i$I; WL=
    elif [ -n "$DEFERRED" ]; then
      bundle_windows $S/commit $S/build $S/partition.json $S/serving-rows $O/standin-i$I || exit 8
    fi
    # the record's files, never a sealed bundle (its store holds the prover's values)
    public $S/serving-rows $O/standin-i$I/window && mkdir -p $O/standin-i$I/commit && cp $S/workload.json $S/partition.json $O/standin-i$I/ \
      && find $S/commit -maxdepth 1 -type f -exec cp {} $O/standin-i$I/commit/ \;
    audit standin-i$I $S/commit $S/build "$WL" $S/serving-rows $S/partition.json "${ADV[@]}" || rc=1
  done
fi
[ $STORE_RC = 0 ] || { echo "a session's circuits weren't stored and preserved: $O/store/"; rc=1; }
exit $rc
