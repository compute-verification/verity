#!/usr/bin/env bash
# OVERHEADDEF's hooks-match run on vy-spot-glm53 (GLM-5.3 FP8 TP8, 8x H200), under research run --cwd clone from
# cursor/proofs-replaycov-glm-643a (benchmarks/replay_ladder/run.sh and benchmarks/serve_tax/replay.py), with these sent:
# serve_logprobs.patch (applied to the clone: the ladder's --serve-logprobs, SERVE_LOGPROBS in run.sh, client.jsonl's
# logprobs), dsa_topk.py and dsa_topk_kernels.py (cursor/proofs-topk-7adf), capture.py, stage.py and spot_job.sh
# (cursor/proofs-overheaddef-1d19), spot_tp8.sh, gpu-lease and sitecustomize.py (the TP8 shims), and this file.
#   [HOLD=/workspace/research/holds/glm53-sessions] [DEADLINE=UNIX_TIME] [SERVED_OUT=RUN/out] [OLD=DIR] \
#     bash "$RESEARCH_RUN_DIR/inputs/spot_canonical.sh"
# SERVED_OUT skips step 1 and replays against an earlier run's serve (its record arm, OUT/served and engine env; REC=ARM_DIR,
# LOAD=NAME and ENGINE_ENV_FILE=FILE name another layout's, such as a serve_tax arm's, whose replay is then held to the arm's
# client tokens, and to its logprobs only where SERVED_OUT/served.json says they were written); OLD is the
# capture removed first when /workspace has under NEED_G (250) GB free.
# 0. HOLD=FILE: vllm's session job (glm53_sessions.sh) waits while FILE exists. It is written first, then the job waits up to
#    IDLE_MIN (40) minutes for every GPU to be under 1 GiB used, and FILE is removed when this script exits.
# 1. serve (no capture plugin): run.sh glm53's pinned arm as fe57 served it (fe57's plan and C4 shard, MAX_MODEL_LEN 40960,
#    FLASHINFER=1, CUDA graphs, SPAN_S 120), no replays, VERITY_DSA_TOPK=canonical in every engine process
#    (OUT/topk-serve), recording only what two-stage serving keeps: the client's tokens and each sampled token's logprob
#    (client.jsonl), serve_tax's step trace (the record arm's), and vllm's corpus recorder (vllm_trace_plugin, OUT/sched/raw;
#    both servers write there, each under its engine core's pid). A window whose servers rewrote no long top-k row is
#    refused (exit 4).
# 2. OUT/served: the window's served tokens and logprobs by engine request id (replay.py --against's format).
# 3. replay with every hook on: spot_job.sh on that record arm from an empty engine, with capture.py (the taps, the graph-off
#    check at each captured step and the commitment) and replay.py --logprobs --against OUT/served; its first-stage coin
#    draws a small, a big and a long-prefill step (past 2,048 keys), kept whole while they fit the disk (KEEP_TENSORS=all);
#    then the match gate on one instance of every call of every type at each step (GATE_EVERY=1, OUT/gate-step<s>), and the
#    drawn blocks' staging (OUT/values-step<s>) until DEADLINE.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; IN=$D/inputs; mkdir -p "$OUT"
FE57=${FE57:-/workspace/research/runs/r20261107-090959-fe57}
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
SENT="serve_logprobs.patch dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh gpu-lease sitecustomize.py spot_canonical.sh"
for f in $SENT; do [ -f "$IN/$f" ] || { echo "canonical: --send $f"; exit 2; }; done
[ -f benchmarks/replay_ladder/run.sh ] && [ -f benchmarks/serve_tax/replay.py ] || { echo "canonical: run from the replay ladder's tree (--cwd clone)"; exit 2; }
git apply "$IN/serve_logprobs.patch" || { echo "canonical: serve_logprobs.patch does not apply (--cwd clone of f095ae769f)"; exit 2; }
PLAN=${PLAN:-$FE57/out/plan/plan.jsonl}; C4=${C4:-$(ls "$FE57"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$PLAN" ] && [ -f "$C4" ] || { echo "canonical: no plan ($PLAN) or C4 shard ($C4)"; exit 2; }
{ echo "tree $(git rev-parse HEAD 2>/dev/null) + serve_logprobs.patch"; (cd "$IN" && sha256sum $SENT); sha256sum "$PLAN" "$C4"; } | tee "$OUT/inputs.sha256"
T=$(mktemp -d); trap 'rm -rf "$T"; [ -n "${HOLD:-}" ] && rm -f "$HOLD" && echo "canonical: removed $HOLD"' EXIT
trap 'exit 143' TERM INT HUP
if [ -n "${HOLD:-}" ]; then
  mkdir -p "$(dirname "$HOLD")" && echo "OVERHEADDEF slot (proofs, $(basename "$D"), hooks-match serve and replay): sessions wait" > "$HOLD" || exit 2
  echo "canonical: wrote $HOLD $(date -u +%FT%TZ)"
  for i in $(seq 0 $((${IDLE_MIN:-40} * 2))); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)
    [ "$busy" = 0 ] && break
    [ "$i" = $((${IDLE_MIN:-40} * 2)) ] && { echo "canonical: $busy GPUs still in use after ${IDLE_MIN:-40} min"; nvidia-smi; exit 5; }
    sleep 30
  done
  echo "canonical: GPUs idle $(date -u +%FT%TZ)"
fi
# the whole-step capture wants room: this lane's earlier capture (its staged values are in its runs' records) goes first
OLD=${OLD:-/workspace/spot-capture/r20261107-103343-2364}
free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ "${free:-0}" -lt "${NEED_G:-250}" ] && [ -d "$OLD" ]; then echo "canonical: ${free}G free, removing $OLD"; rm -rf "$OLD"; fi
df -BG /workspace | tee "$OUT/disk_before.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpus_before.csv"
"$PY" "$IN/dsa_topk.py" site "$T/topk" >/dev/null || exit 2
mkdir -p "$T/trace/verity_trace_serve-0.0.dist-info" "$OUT/sched/raw"
cp benchmarks/network_e2e/traces/vllm_trace_plugin.py "$T/trace/"
printf 'Metadata-Version: 2.1\nName: verity-trace-serve\nVersion: 0.0\n' > "$T/trace/verity_trace_serve-0.0.dist-info/METADATA"
printf '[vllm.general_plugins]\nverity_trace = vllm_trace_plugin:register\n' > "$T/trace/verity_trace_serve-0.0.dist-info/entry_points.txt"
unset VLLM_PLUGINS VERITY_TRACE_DIR
export PYTHONPATH="$T/topk${PYTHONPATH:+:$PYTHONPATH}" VERITY_DSA_TOPK=canonical
live() {  # the long top-k rows the plugin rewrote, summed over the processes that logged under $1
  "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
logs = sorted(Path(sys.argv[1]).glob("dsa_topk.*.json"))
recs = [json.loads(p.read_text()) for p in logs]
per = [sum(r["rows_long"] for r in rec["rows"].values()) for rec in recs]
print(json.dumps({"dir": sys.argv[1], "processes": len(per), "processes_rewriting": sum(n > 0 for n in per), "rows_long": sum(per)}))
sys.exit(0 if sum(per) > 0 else 1)
EOF
}
if [ -n "${SERVED_OUT:-}" ]; then
  R=${REC:-$SERVED_OUT/pinned/glm53/record}; EE=${ENGINE_ENV_FILE:-$SERVED_OUT/serve_engine_env.txt}
  [ -f "$R/meta.json" ] && [ -f "$R/${LOAD:-saturated}/rep-1/client.jsonl" ] && [ -f "$EE" ] \
    || { echo "canonical: no serve at $SERVED_OUT (record arm $R with load ${LOAD:-saturated}, engine env $EE)"; exit 3; }
  echo "== the serve is $R's load ${LOAD:-saturated} $(date -u +%FT%TZ)"
  cp "$EE" "$OUT/serve_engine_env.txt" || exit 2
  if [ -f "$SERVED_OUT/served.json" ]; then cp -r "$SERVED_OUT/served" "$OUT/served" && cp "$SERVED_OUT/served.json" "$OUT/" || exit 2; fi
  cp "$SERVED_OUT/topk-serve.json" "$OUT/" 2>/dev/null
else
echo "== serve $(date -u +%FT%TZ)"
PYTHONPATH="$T/trace:$PYTHONPATH" VERITY_TRACE_DIR=$OUT/sched/raw SERVE_LOGPROBS=1 \
  VERITY_DSA_TOPK_LOG=$OUT/topk-serve PLAN=$PLAN C4=$C4 MAX_MODEL_LEN=40960 FLASHINFER=1 CUDA_HOME=/usr/local/cuda-12.9 \
  ARMS=pinned RBLOCK0_ARMS=pinned REPLAYS=0 FRESH=0 ALONE=0 LOGPROBS=1 SPAN_S=120 VENV=$VENV \
  bash benchmarks/replay_ladder/run.sh glm53
echo "canonical: run.sh rc=$? $(date -u +%FT%TZ)"
R=$OUT/pinned/glm53/record
[ -f "$R/meta.json" ] && [ -f "$R/saturated/rep-1/client.jsonl" ] || { echo "canonical: no record arm at $R"; exit 3; }
live "$OUT/topk-serve" | tee "$OUT/topk-serve.json" || { echo "canonical: the servers rewrote no long top-k row"; exit 4; }
ls -la "$OUT/sched/raw" | tail -n 12
cp "$OUT/engine_env.txt" "$OUT/serve_engine_env.txt"
"$PY" - "$(pwd)" "$R" "$OUT/served" <<'EOF' | tee "$OUT/served.json"
import gzip, json, sys
from pathlib import Path
tree, arm, out = sys.argv[1:]
sys.path.insert(0, str(Path(tree) / "benchmarks" / "serve_tax"))
import replay
w = replay.window(Path(arm), "saturated", 1)
client = {json.loads(x)["id"]: json.loads(x) for x in (Path(arm) / "saturated" / "rep-1" / "client.jsonl").read_text().splitlines() if x.strip()}
toks, lps = {}, {}
for r in w["requests"]:
    c = client[r["engine_id"].split("-", 1)[1].rsplit("-", 2)[0]]
    toks[r["engine_id"]] = c["output_ids"]
    if len(c.get("logprobs") or []) == len(c["output_ids"]):
        lps[r["engine_id"]] = c["logprobs"]
Path(out).mkdir(parents=True, exist_ok=True)
replay.write_tokens(Path(out), toks)
if len(lps) == len(toks):
    replay.write_logprobs(Path(out), lps)
print(json.dumps({"requests": len(toks), "tokens": sum(map(len, toks.values())), "requests_with_logprobs": len(lps),
                  "steps": len(w["steps"]), "logprobs_written": len(lps) == len(toks)}))
EOF
fi
echo "== replay with every hook on $(date -u +%FT%TZ)"
AGAINST=; [ -f "$OUT/served.json" ] && AGAINST="--logprobs --against $OUT/served"
VERITY_DSA_TOPK_LOG=$OUT/topk-capture ARM=$R LOAD=${LOAD:-saturated} ENGINE_ENV=$OUT/serve_engine_env.txt HF_HOME=/workspace/hf \
  KEEP_TENSORS=${KEEP_TENSORS:-all} LONG=${LONG:-1} GATE_EVERY=${GATE_EVERY:-1} STAGE=${STAGE:-1} DEADLINE=${DEADLINE:-0} \
  REPLAY_ARGS="$AGAINST" VENV=$VENV bash "$IN/spot_tp8.sh"
rc=$?
live "$OUT/topk-capture" | tee "$OUT/topk-capture.json" || echo "canonical: the capture's replay rewrote no long top-k row"
"$PY" - "$OUT" <<'EOF' | tee "$OUT/hooks_match.json"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rp = json.loads((out / "replay" / "replay.json").read_text()) if (out / "replay" / "replay.json").is_file() else {}
ag = rp.get("against") or {}
res = {"bit_exact_tokens_and_steps": rp.get("bit_exact"), "steps_differing": rp.get("steps_differing"),
       "first_differing_step": rp.get("first_differing_step"), "requests_tokens_differing": rp.get("requests_tokens_differing"),
       "logprobs": {k: (ag.get("logprobs") or {}).get(k) for k in ("requests_logprobs_differing", "first_step")},
       "first_logprob_difference": ((ag.get("logprobs") or {}).get("requests_differing") or [None])[0],
       "checks": {}, "gate": {}}
for m in sorted(out.glob("commit/rank*/step*/meta.json")):
    res["checks"][f"{m.parent.parent.name}/{m.parent.name}"] = json.loads(m.read_text()).get("check")
for g in sorted(out.glob("gate-step*/gate.json")):
    j = json.loads(g.read_text())
    res["gate"][g.parent.name] = {k: {f: v.get(f) for f in ("calls", "calls_gated", "agree", "disagree", "errors", "why") if v.get(f) is not None}
                                  for k, v in j["types"].items()}
print(json.dumps(res, indent=1, default=str))
EOF
df -BG /workspace
echo "canonical: done rc=$rc $(date -u +%FT%TZ)"
exit $rc
