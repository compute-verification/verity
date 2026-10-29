#!/usr/bin/env bash
# The coverage campaign's pod-side runner.  One configuration = one cell of the coverage table:
# GPU phase (capture / control / census / nonint) then the CPU phase (resolve … tier A k=3 … gates + card) of
# verity_vllm.pipeline.match, all on the pod's container disk (/workspace: /vol can be over quota, and the laptop stays a thin
# client), then a SMALL bundle (json reports, cards, logs, tokens -- no snapshots, no instances.jsonl) at
# /workspace/cp/cov/bundles/<run>.small.tgz for the laptop to pull.  The frozen candidate (program.json / instances.jsonl /
# accesses.jsonl.gz) stays in the run dir on the pod (and is copied to /vol/cp/cov/<run>/ when the volume takes it).
#
#   cp verity_vllm/ops/cov_pod.sh /workspace/cp/ && COV_REPO=<tree>/integrations/vllm nohup bash /workspace/cp/cov_pod.sh run <case> <workload-name> [driver args…] > … &
#   bash /workspace/cp/cov_pod.sh gpu <case> <workload-name> [driver args…]   # GPU phase only; COV_RUN_SUFFIX appends to the run name
#   bash /workspace/cp/cov_pod.sh cpu <run-name> [driver args…]     # CPU phase again over an existing capture
#   bash /workspace/cp/cov_pod.sh regate <run-name> [--census]      # cheap check stages (+ census GPU stage) + gates again over an existing capture
#   bash /workspace/cp/cov_pod.sh holdouts <run-name> <replicate-run>...  # replicate runs as G7 holdouts (holdouts.json) + gates again
#   bash /workspace/cp/cov_pod.sh bundle <run-name>                 # small bundle only
#   bash /workspace/cp/cov_pod.sh queue <case> <name1> <name2> …    # run several workloads of one case serially (GPU), CPU phases in
#                                                                   # the background (at most COV_CPU_PAR at a time, default 2)
# <workload-name> is the stem of workloads/workload_<name>.json (e.g. cov_b0_b8-mixed_c256); <run-name> = <case>-<name>.
# Env: COV_K (tier A k, default 3), COV_PAR (replay stages in parallel, default 8), COV_SNAP_MB (snapshot cap in MiB for B8 / C1024
# runs, default 256), COV_CPU_PAR (concurrent CPU phases in `queue`, default 2), COV_EXTRA (extra driver args for every run).
set -uo pipefail
# COV_REPO: the integrations/vllm directory to run (the campaign runner points it at a per-attempt code snapshot); default: the tree
# this script sits in when run in place.  A copy outside any tree has no default.
SELF_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd -P)
[ -f "$SELF_TREE/verity_vllm/__init__.py" ] || SELF_TREE=
cd "${COV_REPO:-${SELF_TREE:?cov_pod.sh is not inside a tree: set COV_REPO to the integrations/vllm directory to run}}" || exit 1
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
PY312=/workspace/venv-py312/bin/python; PY311=/workspace/venv-cu129/bin/python
BASE=/workspace/cp/cov; OUT=$BASE/runs; CARDS=$BASE/cards; LOGS=$BASE/logs; BUNDLES=$BASE/bundles
mkdir -p "$OUT" "$CARDS" "$LOGS" "$BUNDLES"
K=${COV_K:-3}; PAR=${COV_PAR:-8}; SNAP_MB=${COV_SNAP_MB:-256}; SNAP_MB_MOE=${COV_SNAP_MB_MOE:-24}; CPU_PAR=${COV_CPU_PAR:-2}
# COV_VOL: the durable copy of the small bundle.  Default = the campaign's canonical /vol/cp/cov; a rerun of gates / holdouts over a copy
# of a canonical record sets COV_VOL=/vol/cp/<name>, so a recheck of a copy never publishes over the canonical bundle.
VOL=${COV_VOL:-/vol/cp/cov}
say() { echo "[cov_pod] $(date -u +%FT%TZ) $*"; }
drv() { "${PY312%/*}/verity-vllm" match --python-gpu $PY311 --python-cpu $PY312 --cards-dir "$CARDS" "$@"; }

ckpt_dir() {  # <case> -> the seeded HF snapshot dir of the role (checkpoints.json local_path)
  $PY312 - "$1" <<'EOF'
import json, sys, os, glob
case = sys.argv[1]
doc = json.load(open("manifests/checkpoints.json"))
for c in doc["checkpoints"]:
    if c.get("role") == case or (case == "B1" and str(c.get("role", "")).startswith("B1")) or (case == "B0" and c.get("role") == "B0"):
        lp = c.get("local_path")
        if lp and os.path.isdir(lp):
            print(lp); sys.exit(0)
        repo = c["repo"].replace("/", "--")
        cands = sorted(glob.glob(f"/workspace/hf/hub/models--{repo}/snapshots/*"))
        if cands:
            print(cands[-1]); sys.exit(0)
sys.exit(f"no local snapshot for case {case}")
EOF
}

snap_args() {  # <workload-name>: raise the snapshot cap for batched / long prefills (the 32 MiB default blinds MLP rows > ~936 tokens)
  # MoE (OLMoE, top-8): the observer copies every event's tensors into pinned host memory until the step flushes (~30 MiB of
  # copies per prefill token for OLMoE: B1 C256 155 tokens = 4.55 GiB in step 0; B8-mixed step 14, 363 tokens = 11.6 GiB), and
  # the fused-MoE intermediates [T*TOPK, 2I] / [T, TOPK, H] (~62% of it) are re-snapshotted ~23x per layer by aten view churn.
  # B8-static's ~1300-token step 0 is ~40 GiB of copies (x1.5 pinned-allocator rounding + ~8 GB baseline) = the pod's 62 GB
  # cgroup: SIGKILLed twice (once at a flat 48 MiB cap, which those ~42 MB buffers pass).  MoE runs therefore keep the full
  # cap at Triton launch events (each expert-GEMM output is copied once, at the launch whose snapshot tier A reads) and apply
  # the lower COV_SNAP_MB_MOE cap (24 MiB) at dispatcher-call events (--snapshot-max-bytes-op): cache2 [T*TOPK, I] = 21 MB and
  # qkv [T, 6144] = 16 MB at T=1300 still pass; only the ~42 MB MoE buffers and their views are dropped there.
  case "$1" in
    *olmoe*b8-*|*olmoe*c1024*) echo "--snapshot-max-bytes $((SNAP_MB << 20)) --snapshot-max-bytes-op $((SNAP_MB_MOE << 20))" ;;
    *b8-*|*c1024*) echo "--snapshot-max-bytes $((SNAP_MB << 20))" ;;
    *) echo "" ;;
  esac
}

bundle() {  # <run-name>: json reports + cards + logs + tokens (no values/, traces/, program.json, instances.jsonl, accesses)
  # wiring.json (≈ 5 KB, G5's operand-read evidence) is in the bundle: without it a G7 holdouts regate over a compact primary re-derives
  # G5 as `skipped: wiring.json missing`, and the recorded G5 pass is not reproducible from the retained artefacts.
  local n=$1
  [ -d "$OUT/$n" ] || { say "bundle $n: no run dir"; return 1; }
  (cd "$OUT" && tar czf "$BUNDLES/$n.small.tgz" \
      --exclude="$n/capture/values" --exclude="$n/census/traces" --exclude="$n/nonint/*/values" --exclude="$n/control/values" \
      --exclude="$n/program.json" --exclude="$n/instances.jsonl" --exclude="$n/accesses.jsonl.gz" --exclude="$n/capture/*.npy" \
      --exclude="$n/*/cos_sin_cache.npy" --exclude="$n/capture/snapshots.json" "$n") \
    && say "bundle: $BUNDLES/$n.small.tgz ($(du -h "$BUNDLES/$n.small.tgz" | cut -f1))" || say "bundle $n FAILED"
  # The /vol copy is optional spill space, but a failed copy is an infrastructure WARNING on record, never silent.  Over quota (EDQUOT)
  # the volume accepts the cp and drops the bytes (0-byte file, cp rc=0), so the copy is verified by size before it counts; a short copy
  # is removed and recorded like a failed one.
  local verr
  if verr=$( (mkdir -p "$VOL" && cp "$BUNDLES/$n.small.tgz" "$VOL/.$n.small.tgz.part" \
              && [ "$(stat -c %s "$VOL/.$n.small.tgz.part" 2>/dev/null || echo -1)" = "$(stat -c %s "$BUNDLES/$n.small.tgz")" ] \
              && mv "$VOL/.$n.small.tgz.part" "$VOL/$n.small.tgz") 2>&1 ); then
    say "  (+ copy on $VOL, $(stat -c %s "$VOL/$n.small.tgz") bytes verified)"
  else
    [ -n "$verr" ] || verr="short copy: /vol accepted the file but kept $(stat -c %s "$VOL/.$n.small.tgz.part" 2>/dev/null || echo 0) of $(stat -c %s "$BUNDLES/$n.small.tgz") bytes (quota)"
    rm -f "$VOL/.$n.small.tgz.part" 2>/dev/null
    say "  WARNING VOL_COPY_FAILED: $(printf '%s' "$verr" | tail -1 | cut -c1-160) -- the small bundle exists on this worker only"
    printf '{"utc":"%s","run":"%s","infra":"VOL_COPY_FAILED","error":%s}\n' "$(date -u +%FT%TZ)" "$n" "$(printf '%s' "$verr" | tail -1 | $PY312 -c 'import json,sys;print(json.dumps(sys.stdin.read().strip()))')" >> "$OUT/$n/infra_warnings.jsonl"
  fi
}

inherit_quarantine() {  # <run-name>: a run dir without quarantine.json inherits the CASE's registered per-Definition admission record
  # G4 needs `quarantine.json` whenever the run resolves to quarantine Definitions (Pythia: GeluErf_v1, LayerNormAten_v1).  That record
  # is per-Definition (difftest check + verify_lane rule 2a), so a replicate or a run on another worker would otherwise be rejected at G4
  # for a missing file, not for numerics.  The admitted record is tracked at verity_vllm/engine/profiles/quarantine/<CASE>.json (with
  # `admitted_by`); it is copied in here, stamped `inherited`, and G4 still re-runs tier-a replay on THIS run's own instances.  Cases
  # without a record (no quarantine specs) skip.
  local n=$1 rec="$PWD/verity_vllm/engine/profiles/quarantine/${1%%-*}.json"   # cwd = the code snapshot (cd above)
  [ -f "$OUT/$n/quarantine.json" ] && return 0
  [ -f "$rec" ] || return 0
  $PY312 - "$rec" "$OUT/$n/quarantine.json" "$n" <<'EOF' && say "quarantine.json inherited for $n from $(basename "$rec") (per-Definition admission record)"
import json, sys, time, hashlib
q = json.load(open(sys.argv[1]))
q["inherited"] = {"from": sys.argv[1].split("/verity_vllm/observe/")[-1], "record_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
                  "into_run": sys.argv[3], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "note": "admission record of the case's quarantine Definitions (not produced by this run); tier-a replay below is this run's own"}
json.dump(q, open(sys.argv[2], "w"), indent=1)
EOF
}

cpu_phase() {  # <run-name> [driver args…]
  local n=$1; shift
  inherit_quarantine "$n"
  say "cpu $n start (k=$K, parallel=$PAR) $*"
  drv --run-dir "$OUT/$n" --phase cpu --parallel "$PAR" --k "$K" --seed 0 ${COV_EXTRA:-} "$@" > "$LOGS/$n.cpu.log" 2>&1
  local rc=$?
  say "cpu $n rc=$rc  $(grep -m1 -E 'VERDICT|verdict' "$OUT/$n/CARD.md" 2>/dev/null | cut -c1-120)"
  bundle "$n"
  return $rc
}

gpu_phase() {  # <case> <workload-name> [driver args…] -> run name on stdout
  local case=$1 wl=$2; shift 2
  local n="$case-$wl${COV_RUN_SUFFIX:-}" ck; ck=$(ckpt_dir "$case") || { say "gpu $n: $ck"; return 1; }   # COV_RUN_SUFFIX: e.g. ".ovC" for a repeat capture of the same cell
  say "gpu $n start (ckpt $ck) $(snap_args "$wl") $*"
  drv --case "$case" --workload "workloads/workload_$wl.json" --out "$OUT/$n" --checkpoint-dir "$ck" --phase gpu \
      --max-num-seqs 8 $(snap_args "$wl") ${COV_EXTRA:-} "$@" > "$LOGS/$n.gpu.log" 2>&1
  local rc=$?
  say "gpu $n rc=$rc  $(grep -m1 '\[realized\]' "$LOGS/$n.gpu.log" | cut -c1-200)"
  du -sh "$OUT/$n" 2>/dev/null | sed "s/^/[cov_pod]   run dir /"
  return $rc
}

regate() {  # <run-name> [--census]: re-run the cheap check stages (+ the census GPU stage with --census) over an existing capture, then gates
  local n=$1; shift
  [ -f "$OUT/$n/run.json" ] || { say "regate $n: no run.json"; return 1; }
  inherit_quarantine "$n"
  if [ "${1:-}" = "--census" ]; then
    say "regate $n: census (GPU) again"
    # VERITY_CENSUS_GPU_UTIL: let the census share the GPU with a queue's engine (the census re-run must not wait for the queue)
    VERITY_CENSUS_GPU_UTIL=${COV_REGATE_GPU_UTIL:-0.12} drv --run-dir "$OUT/$n" --phase gpu --skip capture,control,nonint \
        > "$LOGS/$n.regate-census.log" 2>&1 || say "regate $n: census rc=$?"
  fi
  say "regate $n: census_reconcile + golden + gates"
  drv --run-dir "$OUT/$n" --phase cpu --skip resolve,nonint_compare,compare,fold,fold_2,fold_other,root_policy,wiring,replay \
      > "$LOGS/$n.regate.log" 2>&1
  local rc=$?
  say "regate $n rc=$rc  $($PY312 -c "import json;g=json.load(open('$OUT/$n/gates.json'));print(g['verdict'], {k:v.get('status') for k,v in g['gates'].items()})" 2>/dev/null)"
  bundle "$n"
  return $rc
}

holdouts() {  # <primary-run> <holdout-run>...: the replicate runs' gates.json assembled into the primary's holdouts.json (G7), then gates again.
  # Campaign holdouts = the campaign's replicate workloads (same seeded length draws, other prompts / request ids; generated apart from
  # the runs they hold out).  They cover fresh token values, not fresh lengths -- the card says so (holdouts.json lists the runs).
  local n=$1; shift
  local hd="$OUT/$n/holdouts"; mkdir -p "$hd"
  for h in "$@"; do
    [ -f "$OUT/$h/gates.json" ] || { say "holdouts $n: $h has no gates.json"; return 1; }
    ln -sfn "$OUT/$h" "$hd/$h"
  done
  "${PY312%/*}/verity-vllm" holdout --assemble "$hd" --out "$OUT/$n/holdouts.json" > "$LOGS/$n.holdouts.log" 2>&1 || say "holdouts $n: assemble rc=$?"
  drv --run-dir "$OUT/$n" --phase cpu --skip resolve,census_reconcile,nonint_compare,compare,fold,fold_2,fold_other,root_policy,golden,wiring,replay \
      >> "$LOGS/$n.holdouts.log" 2>&1
  local rc=$?
  say "holdouts $n ($*) rc=$rc  $($PY312 -c "import json;g=json.load(open('$OUT/$n/gates.json'));print(g['verdict'], {k:v.get('status') for k,v in g['gates'].items()})" 2>/dev/null)"
  bundle "$n"
  return $rc
}

MODE=${1:-run}; shift || true
case "$MODE" in
  run)    case_=$1; wl=$2; shift 2; gpu_phase "$case_" "$wl" "$@" && cpu_phase "$case_-$wl${COV_RUN_SUFFIX:-}" ;;
  gpu)    case_=$1; wl=$2; shift 2; gpu_phase "$case_" "$wl" "$@" ;;             # GPU phase only (capture / control / census / nonint)
  cpu)    n=$1; shift; cpu_phase "$n" "$@" ;;
  regate) n=$1; shift; regate "$n" "$@" ;;
  holdouts) n=$1; shift; holdouts "$n" "$@" ;;
  bundle) bundle "$1" ;;
  queue)  case_=$1; shift
          for wl in "$@"; do
            gpu_phase "$case_" "$wl" || { say "queue: gpu $wl failed, continuing"; continue; }
            while [ "$(jobs -rp | wc -l)" -ge "$CPU_PAR" ]; do sleep 20; done
            ( cpu_phase "$case_-$wl" ) &
          done
          say "queue: all gpu phases done; waiting for the cpu phases"; wait; say "queue: ALL DONE" ;;
  *) echo "usage: cov_pod.sh run|cpu|bundle|queue …" >&2; exit 2 ;;
esac
