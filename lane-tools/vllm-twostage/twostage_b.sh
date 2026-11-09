#!/usr/bin/env bash
# vllm-twostage-int (VLLM-250): Tue's (b), the replay side's GPU shape-only replay of (c)'s served window, on node A under
# research run with --cwd source, from a tree holding main, #8111, #8124 and #8385 (replay.py --capture and run.sh's
# REPLAY_CAPTURE, sampler_capture.piece_rows, the pieces' rows named after their parameters as the second stage stages
# them). It reads (c)'s private dir (twostage_serve.sh's PRIV, SERVE_ROOT/RUN_C, which
# must be 0700 and this user's), holds the first stage's stage1.json (twostage_first.sh's OUT/stage1/<class>/stage1.json,
# sent) to the class's commit.json there, and that commit.json to the class's private files (boundary_rows.reopen); then
# serve_tax's run.sh replays the load (REPLAY_FROM=the boundary-rows arm) in (c)'s engine environment (its engine_env.txt's
# FLASHINFER, VLLM_USE_FLASHINFER_SAMPLER, CUDA_HOME and MAX_MODEL_LEN) at TP 8, capturing the sampled spans' logits rows
# and ids at the sampler, and capture_pieces_gpu.py holds each captured span's token-select pieces on one leased GPU to
# the host's.
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd source --timeout 3h50m \
#     --send twostage_b.sh --send capture_pieces_gpu.py --send stage1.json --env RUN_C=RUN \
#     [--env END_UTC=ISO [--env NEED_MIN=45]] [--env AFTER=RUN] [--env LOAD=burst] \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/twostage_b.sh'
# A rehearsal with no first stage sends no stage1.json and gives --env UNITS=0,1: the sample is those spans of (c)'s one
# class (its commit.json in the private dir), not the challenger's coins, and summary.json says so ("sample"). Where (c)
# committed several classes (a warm-up window w0 and the burst w1), --env CLASS=w1/p256-o256 names the one.
# Everything prover-only stays on the node, in the private dir PRIV=REPLAY_ROOT/RUN (default
# /workspace/research/vllm-twostage-int, outside the run dir, 0700, kept KEEP long by `research keep`), which
# OUT/private.txt names: the sampled units' engine ids, the replay (its capture: the spans' logits rows and ids; its
# tokens, steps, replay.json and engine_env.txt), the pieces checks and every log. OUT gets only summary.json: per sampled
# unit (the first stage's public sample) whether it was captured, whether its replayed ids are the served ones and whether
# its pieces on the GPU equal the host's; whether the whole replay was bit exact and ran in (c)'s environment; the steps'
# exit codes and times. This job draws no salt: the interior registration commits the capture after it is copied to the
# proving node (twostage_copy.sh), under that node's own private dir.
# AFTER: a run on this node whose end this waits for before touching the GPUs. END_UTC: start only while NEED_MIN minutes
# remain before it, else exit 75 having replayed nothing. Stub tests: PIECES_DEVICE=cpu, KEEP=none (no `research keep`).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; I=$D/inputs; mkdir -p "$OUT"
RUN_C=${RUN_C:?RUN_C: the served window's run, twostage_serve.sh's}
PRIV=${REPLAY_ROOT:-/workspace/research/vllm-twostage-int}/${RESEARCH_RUN_ID:-$(basename "$D")}
case $PRIV in "$D"|"$D"/*) echo "twostage-b: the private dir $PRIV is in the run dir, which custody publishes"; exit 2 ;; esac
(umask 077; mkdir -p "$PRIV") || { echo "twostage-b: no private dir $PRIV"; exit 2; }
chmod 700 "$PRIV"; echo "$PRIV" > "$OUT/private.txt"
rc_inputs=-1 rc_replay=-1 rc_pieces=-1
summary() {
  python3 - "$OUT/summary.json" "$PRIV" "$rc_inputs" "$rc_replay" "$rc_pieces" <<'EOF'
import json, sys
from pathlib import Path
out, priv, rcs = Path(sys.argv[1]), Path(sys.argv[2]), dict(zip(("inputs", "replay", "pieces"), map(int, sys.argv[3:6])))
def load(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
units, rep, env = load(priv / "units.json") or {}, load(priv / "out/replay-compiled/replay.json") or {}, load(priv / "env.json") or {}
cap = rep.get("capture") or {}
per = []
for u, rid in sorted((int(k), v) for k, v in (units.get("units") or {}).items()):
    pc = load(priv / "pieces" / f"{u}.json") or {}
    per.append({"unit": u, "captured": rid in cap, "as_served": (cap.get(rid) or {}).get("as_served"),
                "pieces_equal": pc.get("equal"), "host_pieces_s": pc.get("host_pieces_s"),
                "device_piece_rows_s": pc.get("device_piece_rows_s")})
s = {"format": "vllm-twostage-int/b/v0", "class": units.get("class"), "sample": units.get("sample"),
     "units": [p["unit"] for p in per], "per_unit": per,
     "bit_exact": rep.get("bit_exact"), "env_as_served": env.get("as_served"), "env_differing": env.get("differing"),
     "fi_cache_as_after_serve": env.get("fi_cache_as_after_serve"),
     "replay_wall_s": rep.get("replay_wall_s"), "engine_ready_s": rep.get("engine_ready_s"), "rc": rcs}
out.write_text(json.dumps(s, indent=1) + "\n")
print("SUMMARY", json.dumps({k: s[k] for k in ("class", "sample", "units", "bit_exact", "env_as_served", "rc")}),
      "as_served", [p["as_served"] for p in per], "pieces_equal", [p["pieces_equal"] for p in per])
EOF
}
finish() {
  local rc=$1
  summary
  stray=$(find "$OUT" -type f ! -path "$OUT/private.txt" ! -path "$OUT/summary.json" | wc -l)
  if [ "$stray" -gt 0 ]; then
    mkdir -p "$PRIV/stray" && (cd "$OUT" && find . -type f ! -path ./private.txt ! -path ./summary.json -print0 \
      | xargs -0 -r cp --parents -t "$PRIV/stray" && find . -type f ! -path ./private.txt ! -path ./summary.json -delete)
    echo "twostage-b: moved $stray other file(s) from OUT to the private dir's stray/"
  fi
  echo "twostage-b: done $(date -u +%T), exit $rc"
  exit "$rc"
}
left() { echo $(( $(date -u -d "$END_UTC" +%s) - $(date +%s) )); }
if [ -n "${AFTER:-}" ]; then
  st=/workspace/research/runs/$AFTER/status.json; absent=0
  echo "twostage-b: waiting for $AFTER to end, from $(date -u +%T)"
  while :; do
    s=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state"))' "$st" 2>/dev/null || echo absent)
    case $s in done|failed|cancelled) echo "twostage-b: $AFTER is $s at $(date -u +%T)"; break ;; esac
    [ "$s" = absent ] && absent=$((absent + 1))
    [ "$absent" -ge 9 ] && { echo "twostage-b: $AFTER has no status.json on this node after 3 min; going on"; break; }
    if [ -n "${END_UTC:-}" ] && [ "$(left)" -lt $((${NEED_MIN:-45} * 60)) ]; then
      echo "twostage-b: $AFTER still $s at $(date -u +%T), under ${NEED_MIN:-45} min before $END_UTC: not replaying"; finish 75
    fi
    sleep 20
  done
fi
if [ -n "${END_UTC:-}" ] && [ "$(left)" -lt $((${NEED_MIN:-45} * 60)) ]; then
  echo "twostage-b: $(( $(left) / 60 )) min left before $END_UTC, under ${NEED_MIN:-45}: not replaying"; finish 75
fi
PRIV_C=${SERVE_ROOT:-/workspace/research/vllm-twostage}/$RUN_C
[ -d "$PRIV_C" ] || { echo "twostage-b: no private dir $PRIV_C for $RUN_C"; finish 2; }
m=$(stat -c '%a %U' "$PRIV_C"); [ "$m" = "700 $(id -un)" ] || { echo "twostage-b: $PRIV_C is '$m', not '700 $(id -un)'"; finish 2; }
ARM=$PRIV_C/out/boundary-rows LOAD=${LOAD:-burst}
{ [ -f "$I/stage1.json" ] || [ -n "${UNITS:-}" ]; } && [ -f "$I/capture_pieces_gpu.py" ] \
  || { echo "twostage-b: --send stage1.json (or --env UNITS=... for a rehearsal) and capture_pieces_gpu.py"; finish 2; }
[ -f "$ARM/$LOAD/rep-1/meta.json" ] || { echo "twostage-b: (c) served no load $LOAD (no boundary-rows/$LOAD/rep-1)"; finish 2; }
[ -f "$PRIV_C/out/engine_env.txt" ] || { echo "twostage-b: (c) recorded no engine_env.txt"; finish 2; }
grep -q 'REPLAY_CAPTURE' benchmarks/serve_tax/run.sh && grep -q 'def piece_rows' integrations/vllm/verity_vllm/engine/sampler_capture.py \
  && grep -q '^def named' integrations/vllm/verity_vllm/pipeline/interior_rows.py \
  || { echo "twostage-b: this tree lacks run.sh's REPLAY_CAPTURE, sampler_capture.piece_rows or interior_rows.named (#8111, #8124, #8385)"; finish 2; }
[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312
PY=${VENV:-$venv}/bin/python
export PYTHONPATH=$PWD:$PWD/integrations/vllm${PYTHONPATH:+:$PYTHONPATH}
[ "${KEEP:-7d}" = none ] || PYTHONPATH=$PWD/tools/research/src python3 -m research keep "$PRIV" --owner @vllm \
  --expires "${KEEP:-7d}" --priority high \
  --why "VLLM-250 (b): the replay's capture of the sampled spans; the interior registration and the second stage read it" \
  > "$PRIV/keep.log" 2>&1 || echo "twostage-b: research keep refused (the private dir's keep.log); replaying anyway"
echo "twostage-b: from $(date -u +%T), tree $(git rev-parse --short HEAD 2>/dev/null || echo shipped), (c) $RUN_C"

"$PY" - "$I/stage1.json" "$ARM/server/boundary" "$PRIV/units.json" "${UNITS:-}" "${CLASS:-}" > "$PRIV/inputs.log" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm.pipeline import boundary_rows as BR
root, out, rehearsal, named = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
if Path(sys.argv[1]).exists():
    s1, sample = json.loads(Path(sys.argv[1]).read_text()), "stage1"
else:
    classes = sorted(p for p in root.glob("w*/p*-o*") if (p / BR.COMMIT).exists())
    if named:
        classes = [p for p in classes if str(p.relative_to(root)) == named]
    if len(classes) != 1:
        raise SystemExit(f"a rehearsal's UNITS name spans of one class (CLASS={named!r}); (c) committed "
                         f"{sorted(str(p.relative_to(root)) for p in root.glob('w*/p*-o*') if (p / BR.COMMIT).exists())}")
    s1 = {"commit": json.loads((classes[0] / BR.COMMIT).read_text()), "units": [int(u) for u in rehearsal.split(",")]}
    sample = "rehearsal: UNITS, not the challenger's coins"
if s1.get("error"):
    raise SystemExit(f"the first stage failed: {s1['error']}")
c = s1["commit"]
k = f"w{c['window']}/p{c['prompt']}-o{c['outputs']}"
d = root / k
if json.loads((d / BR.COMMIT).read_text()) != c:
    raise SystemExit(f"stage1.json's commit is not {k}'s commit.json")
BR.reopen(d, BR.check_template(c["check"]))
spans = json.loads((d / BR.INDEX).read_bytes())["spans"]
units = [int(u) for u in s1["units"]]
if not units or not all(0 <= u < len(spans) for u in units):
    raise SystemExit(f"a sample {units} of the class's {len(spans)} spans")
out.write_text(json.dumps({"class": k, "dir": str(d), "units": {u: spans[u]["engine_id"] for u in units}, "sample": sample}))
print(f"twostage-b: {k} reopens to its commit.json; sample {units} of {len(spans)} spans")
EOF
rc_inputs=$?
[ $rc_inputs = 0 ] || { echo "twostage-b: the inputs don't hold (the private dir's inputs.log)"; finish 1; }
tail -n 1 "$PRIV/inputs.log"
rids=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["units"].values()))' "$PRIV/units.json")

python3 - "$PRIV_C/out/engine_env.txt" "$PRIV/mirror.sh" <<'EOF'
import shlex, sys
keys = ("FLASHINFER", "VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME", "MAX_MODEL_LEN")
got = dict(x.split("=", 1) for x in open(sys.argv[1]).read().splitlines() if "=" in x and x.split("=", 1)[0] in keys)
open(sys.argv[2], "w").write("".join(f"unset {k}\n" if got.get(k, "unset") == "unset" else f"export {k}={shlex.quote(got[k])}\n"
                                     for k in keys))
EOF
. "$PRIV/mirror.sh"
# the canonical DSA top-k as (c) served it (twostage_serve.sh's CANON=1 leaves its site at PRIV_C/topk), or CANON=1
if [ -n "${CANON:-}" ] || [ -d "$PRIV_C/topk" ]; then
  "$PY" benchmarks/dsa_topk/dsa_topk.py site "$PRIV/topk" > /dev/null || { echo "twostage-b: no dsa_topk site"; finish 2; }
  export PYTHONPATH=$PRIV/topk${PYTHONPATH:+:$PYTHONPATH} VERITY_DSA_TOPK=canonical VERITY_DSA_TOPK_LOG=$PRIV/topk-log
fi
echo "twostage-b: (c)'s engine environment: FLASHINFER=${FLASHINFER:-unset} CUDA_HOME=${CUDA_HOME:-unset}; canonical top-k $([ "${VERITY_DSA_TOPK:-}" = canonical ] && echo on || echo off)"
echo "twostage-b: replay of $LOAD from $(date -u +%T)"
RESEARCH_RUN_DIR=$PRIV MODEL_ID=glm53 REPLAY=$LOAD REPLAY_FROM=$ARM REPLAY_MODES=compiled REPLAY_CAPTURE=$rids \
  LDCONFIG=${LDCONFIG:-1} GPU_LEASE_WHO=twostage-b bash benchmarks/serve_tax/run.sh > "$PRIV/replay.log" 2>&1
rc_replay=$?
echo "twostage-b: replay exit $rc_replay at $(date -u +%T) (its log: the private dir's replay.log)"
python3 - "$PRIV_C/out/engine_env.txt" "$PRIV/out/engine_env.txt" "$PRIV/env.json" <<'EOF'
import json, sys
KEYS = ("FLASHINFER=", "VLLM_USE_FLASHINFER_SAMPLER=", "CUDA_HOME=", "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=", "/", "Build ",
        "has_flashinfer=")
def env(p):
    try:
        lines = open(p).read().splitlines()
    except OSError:
        return None, None
    caches = [x.split(": ", 1)[1] for x in lines if x.startswith(("before the arms: ", "after "))]
    return [x for x in lines if x.startswith(KEYS)], caches
(a, ca), (b, cb) = env(sys.argv[1]), env(sys.argv[2])
json.dump({"as_served": None if b is None else a == b, "differing": None if b is None else sorted(set(a) ^ set(b)),
           "fi_cache_as_after_serve": None if not (ca and cb) else ca[-1] == cb[0]}, open(sys.argv[3], "w"))
EOF
CAP=$PRIV/out/replay-compiled/capture
if [ $rc_replay = 0 ] && [ -d "$CAP" ]; then
  mkdir -p "$PRIV/pieces"; : > "$PRIV/pieces/failed.txt"; rc_pieces=0
  pieces() {
    python3 -c 'import json,sys; [print(u, r) for u, r in json.load(open(sys.argv[1]))["units"].items()]' "$PRIV/units.json" \
      | while read -r u rid; do
          # the engine id is prover-only and a process listing shows a command line: it goes in a 0600 file, never on argv
          (umask 077; printf '%s\n' "$rid" > "$PRIV/pieces/$u.rid")
          DEVICE=${PIECES_DEVICE:-cuda:0} "$PY" "$I/capture_pieces_gpu.py" "$CAP" "@$PRIV/pieces/$u.rid" > "$PRIV/pieces/$u.json" 2> "$PRIV/pieces/$u.log" \
            || echo "$u" >> "$PRIV/pieces/failed.txt"
        done
  }
  if command -v gpu-lease > /dev/null; then
    export -f pieces; export PRIV CAP PY I PIECES_DEVICE
    GPU_LEASE_WHO=twostage-b gpu-lease 1 --wait --max-min 30 -- bash -c pieces > "$PRIV/pieces/lease.log" 2>&1 || rc_pieces=1
  else
    pieces
  fi
  [ -s "$PRIV/pieces/failed.txt" ] && rc_pieces=1
  echo "twostage-b: pieces exit $rc_pieces at $(date -u +%T); units unequal or failed: [$(paste -sd, "$PRIV/pieces/failed.txt")]"
  [ $rc_pieces = 0 ] && rm -rf "$CAP/ranks"
fi
rc=0; for r in $rc_inputs $rc_replay $rc_pieces; do [ "$r" = 0 ] || rc=1; done
du -sh "$PRIV" 2>/dev/null | cut -f1 | sed 's/^/twostage-b: private dir /'
finish $rc
