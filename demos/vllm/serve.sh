#!/usr/bin/env bash
# A LLAMA32_1B committing OpenAI endpoint whose Commit windows are sealed as deferred Commits' workload dirs, for PROVE-PATH's served
# windows (`--serve-run DIR --window K`): `verity-vllm serve-openai --scheme rows-sha512 --seal` (the bench workload's Commit of record's
# committer and acquire sources, built from its manifest ROW/manifest.json and taps HIDDEN_SO, NORM_TAP_SO: hm96-sha512 rows in
# frame-v3-sha512 trees) on one leased RTX PRO 6000, a short varied load in bursts (each burst one window, closed at the engine's
# idle), `verity-vllm advice` per closed window under PROVE-PATH's registered compiler for a served run (`prove_path.py compiler
# --serve-run`: the server's, listing its model's transfer gates of record, the one serve_check holds each registration to),
# a registered cap (--cap) and the committer's weights registration (`prove_path.py weights-registration`: each weight
# committed under salts drawn from the committer's key, $D/private/weights.key, which never leaves the node, once its shards
# check against the checkpoint's manifest; each window's registration records its roots, one a weight parameter, so the window
# circuit's weight reads count registered at the pins the verifier holds before the first window's coins, `window-circuit
# --pins weights_registration.json`; without it, which steps.txt says, they count free),
# then each window's request Programs and manifest.json into its workload dir (`tools/advice_compile.py RUN --window K`), and the
# workload dir opened as PROVE-PATH opens it (`deferred_rows.of_record`, `opening.CommittedStore`) with what the workload's Programs
# read looked up (identities.txt). The endpoint serves until UNTIL (UTC HH:MM) or `touch $D/stop`, or until its filesystem has under
# DISK_FLOOR_GB free (default 300, checked before serving and every 10 s). The serve dir's records (requests.jsonl, serve.json,
# windows.jsonl, advice_trace.jsonl, summary.json) are copied into $OUT every 10 s, at the end and on SIGTERM, so custody keeps them.
# VARIED=N (with ADMIT=release-order; VARIED_RUNS=3, VARIED_RATE=4, VARIED_JITTER_MS=400): before READY, VARIED_RUNS runs of N
# varied requests (network-accounting's traffic, `serve_load.py --varied`), each numbered under a fresh release run and
# sent up to the jitter late, then `tools/admit_rows_check.py` over the serve dir: every window's first rows must be in
# admission order, else the endpoint stops (exit 14). After UNTIL the check runs again over everything served.
# `verity example vllm serve` (run.sh) starts it where placement.toml's [run.serve] says, on one leased GPU, with the node's VENV,
# CUDA_HOME, HF_HOME, HIDDEN_SO and NORM_TAP_SO, the run's SERVE_ROOT, and the workload's REPO, REV and ROW (the Commit of record's
# workload dir) as its environment; --env PORT=... and --env UNTIL=... after `serve` change the run's own. Windows are sealed off the engine loop and close at WINDOW_GB committed; the server itself also refuses to start, and later
# stops taking requests, below DISK_FLOOR_GB free on the serve dir's filesystem, and copies its records into $OUT after each
# window and at any stop. SEAL_HOLD_GB (64), SEAL_WARM_GB (0), MAX_NUM_SEQS (8), MAX_NUM_BATCHED_TOKENS (2048) and
# WINDOW_REQUESTS (unset: no bound) are serve-openai's --seal-hold-gb, --seal-warm-gb, --max-num-seqs,
# --max-num-batched-tokens and --window-requests (a window's class slots, e.g. 4 for V3b's class of record).
# PROTOCOL (unset: vllm) is its --protocol: ids streams the scan's id form raw to the network certifier's id requests, for a
# certifier in front of the endpoint, and answers every other request (the smokes, the bursts, VARIED's load, the certifier's
# shim and probes) as vllm does.
# READVICE=DIR (WINDOW=K, default 5) serves nothing: DIR's advice again under that registered compiler, and window K's
# request Programs and manifest.json.
# Needs on the node: vLLM's venv, the checkpoint in HF_HOME and CUDA (the collector's JIT build). The serve dir D (workload dirs, bundles) stays on the node under `research keep`; $OUT keeps the records and logs.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
PORT=${PORT:-18084} UNTIL=${UNTIL:-13:30} ADMIT=${ADMIT:-arrival} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512} BURSTS=${BURSTS:-8} PER_BURST=${PER_BURST:-4} CAP=${CAP:-8388608}
for v in REPO REV ROW HIDDEN_SO NORM_TAP_SO VENV CUDA_HOME HF_HOME SERVE_ROOT; do
  [ -n "${!v:-}" ] || { echo "serve.sh: no $v: start it with \`verity example vllm serve\` (demos/vllm/run.sh)"; exit 2; }
done
TREE=$PWD
ROOT=$SERVE_ROOT
D=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$D" "$ROOT/cache"
export PY=$VENV/bin/python HF_HOME HF_HUB_OFFLINE=1
export CUDA_HOME TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host
export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
# A learning step's transient host copy: a 1,984-token prefill step stages over 16 GiB, past the
# collector's 4096 MiB default, and isn't committed; lease --mem-gb 256 with it.
[ "$MAX_MODEL_LEN" -gt 512 ] && export VERITY_LEARN_HOST_BUDGET_MB=${VERITY_LEARN_HOST_BUDGET_MB:-65536}
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "serve dir $D port $PORT until $UNTIL"; nvidia-smi -L; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 7d --priority high \
  --why "G-A (network accounting): LLAMA32_1B serve-openai windows sealed as rows-sha512 row dirs for PROVE-PATH" | tail -2
cd "$TREE/integrations/vllm"
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }
compiler_args() {   # DIR: `verity-vllm advice`'s --compiler-name/--compiler-digest for the served run DIR (its serve.json)
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$1" --cap-bits "$CAP") \
    | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])"
}
CKPT=$HF_HOME/hub/models--${REPO//\//--}/snapshots/$REV
weights_args() {    # DIR FILE: `verity-vllm advice`'s --weights-of-record for the served run DIR, the committer's registration written to FILE
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$1" --checkpoint-dir "$CKPT" \
    --key "$1/private/weights.key" --out "$2") >> "$OUT/weights_registration.txt" 2>&1 && echo "--weights-of-record $2"
}
no_weights() { echo "NO WEIGHTS REGISTRATION for $1 (weights_registration.txt): no window registers a weight's root, so every weight read counts free" | tee -a "$OUT/steps.txt"; }

if [ -n "${READVICE:-}" ]; then     # a served dir's advice again (every closed window), then WINDOW's Programs and manifest.json
  K=${WINDOW:-5}
  CA=$(compiler_args "$READVICE") && [ -n "$CA" ] || { echo "no registered compiler for $READVICE (prove_path.py compiler)"; exit 16; }
  echo "registered compiler: $CA" | tee "$OUT/compiler.txt"
  WA=$(weights_args "$READVICE" "$OUT/weights_registration.json") || no_weights "$READVICE"
  # shellcheck disable=SC2086
  vv advice "$READVICE" --cap "$CAP" $CA $WA 2>&1 | tee "$OUT/advice.log"
  "$PY" tools/advice_compile.py "$READVICE" --window "$K" > "$OUT/compile_w$K.log" 2>&1
  echo "w$K compile rc=$?" | tee "$OUT/steps.txt"
  cp "$READVICE/advice/w$K/registration.json" "$READVICE/windows/w$K/manifest.json" "$OUT/"
  exit 0
fi

DISK_FLOOR_GB=${DISK_FLOOR_GB:-300}; floor=$((DISK_FLOOR_GB * 1048576))
under_floor() { [ "$(df --output=avail -k "$ROOT" | tail -1)" -lt "$floor" ]; }
keep_records() {
  for f in requests.jsonl serve.json windows.jsonl seals.jsonl pauses.jsonl advice_trace.jsonl summary.json; do
    [ -f "$D/$f" ] && cp "$D/$f" "$OUT/.$f.tmp" 2>/dev/null && mv "$OUT/.$f.tmp" "$OUT/$f"
  done
}
under_floor && { df -h "$ROOT"; echo "DISK FLOOR: under $DISK_FLOOR_GB GB free on $ROOT; not serving"; exit 15; }

# the server's own pid (not a function's subshell), so a stop's SIGTERM reaches its handler: it writes summary.json and
# finishes the pending seals
"$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" --precision pin \
  --run-id "${RESEARCH_RUN_ID:-local}" \
  --scheme rows-sha512 --seal --gpu-tree --window-steps 256 --required-manifest "$ROW/manifest.json" --hidden-so "$HIDDEN_SO" \
  --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 --admit "$ADMIT" --max-model-len "$MAX_MODEL_LEN" \
  --window-gb "${WINDOW_GB:-32}" --disk-floor-gb "$DISK_FLOOR_GB" \
  --seal-hold-gb "${SEAL_HOLD_GB:-64}" --seal-warm-gb "${SEAL_WARM_GB:-0}" ${MAX_NUM_SEQS:+--max-num-seqs "$MAX_NUM_SEQS"} \
  ${WINDOW_REQUESTS:+--window-requests "$WINDOW_REQUESTS"} ${PROTOCOL:+--protocol "$PROTOCOL"} \
  ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
  --copy-records-to "$OUT" > "$OUT/serve.log" 2>&1 &
SRV=$!
echo "$SRV" > "$D/serve.pid"
trap 'keep_records; kill -TERM "$SRV" 2>/dev/null; wait "$SRV"; keep_records; exit 143' TERM INT
( while kill -0 "$SRV" 2>/dev/null; do
    keep_records
    under_floor && { echo "$(date -u +%H:%M:%SZ) DISK FLOOR: under $DISK_FLOOR_GB GB free on $ROOT" | tee -a "$OUT/steps.txt"; touch "$D/stop"; kill -TERM "$SRV"; }
    sleep 10
  done; keep_records ) &
for _ in $(seq 1 600); do
  grep -q "SERVE-OPENAI listening" "$OUT/serve.log" && break
  kill -0 "$SRV" 2>/dev/null || { tail -40 "$OUT/serve.log"; echo "SERVER EXITED before listening"; exit 10; }
  sleep 2
done
grep "SERVE-OPENAI listening" "$OUT/serve.log" || { tail -40 "$OUT/serve.log"; echo "SERVER NOT LISTENING"; kill "$SRV"; exit 11; }
cp "$D/serve.json" "$OUT/" 2>/dev/null
CA=$(compiler_args "$D") && [ -n "$CA" ] || { echo "no registered compiler for $D (prove_path.py compiler)"; kill -TERM "$SRV"; wait "$SRV"; exit 16; }
echo "registered compiler: $CA" | tee "$OUT/compiler.txt"
WA=$(weights_args "$D" "$D/weights_registration.json") || no_weights "$D"
cp "$D/weights_registration.json" "$OUT/" 2>/dev/null

# Each window's advice committed once, at serve time, as its windows.jsonl line lands: `advice --window K` into advice/.w<K>,
# moved to advice/w<K>, and advice/windows.json the union of every window's index (committed and refused), its registration
# recording each weight's registered root (`--weights-of-record`, `prove_path.py weights-registration`).
closed() { "$PY" -c "import json,sys; [print(json.loads(l)['window']) for l in open(sys.argv[1]) if l.strip()]" "$D/windows.jsonl" 2>/dev/null; }
advice_pass() {
  for k in $(closed); do
    [ -f "$D/advice/.w$k/windows.json" ] && continue
    # shellcheck disable=SC2086
    vv advice "$D" --window "$k" --cap "$CAP" $CA $WA --out "$D/advice/.w$k" >> "$OUT/advice.log" 2>&1; rc=$?
    [ -f "$D/advice/.w$k/windows.json" ] || continue        # not yet in the step trace, or pending (rc 3) on a record: next pass
    [ -d "$D/advice/.w$k/w$k" ] && mv "$D/advice/.w$k/w$k" "$D/advice/w$k"
    "$PY" - "$D/advice" "$k" "$rc" <<'EOF' | tee -a "$OUT/advice_windows.jsonl"
import json, sys, time
from pathlib import Path
a, k, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
idx = {"windows": {}, "refused": {}}
for p in sorted(a.glob(".w*/windows.json")):
    d = json.loads(p.read_text())
    idx = {**d, "windows": {**idx["windows"], **d["windows"]}, "refused": {**idx["refused"], **d["refused"]}}
idx.pop("open", None)
(a / "windows.json").write_text(json.dumps(idx, indent=1) + "\n")
w = idx["windows"].get(k)
print(json.dumps({"window": int(k), "at": time.strftime("%H:%M:%SZ", time.gmtime()), "rc": rc,
                  "root": w and w["root"], "bits": w and w["bits"], "refused": idx["refused"].get(k)}))
EOF
  done
}
( while kill -0 "$SRV" 2>/dev/null; do advice_pass; sleep 10; done; advice_pass ) &
ADV=$!

if [ "$ADMIT" = release-order ]; then   # releases 1 then 0 of a fresh release run, a second apart: 0 must reach the engine first
  "$PY" - "$PORT" "$REPO" "$D" <<'EOF2' 2>&1 | tee "$OUT/admit_smoke.txt"
import json, sys, threading, time, urllib.request, uuid
from pathlib import Path
port, model, d = sys.argv[1], sys.argv[2], Path(sys.argv[3])
run = uuid.uuid4().hex[:16]
sent = {}
def post(n):
    body = {"model": model, "user": f"admit-smoke-{n}", "prompt": [128000] + list(range(2000 + 50 * n, 2040 + 50 * n)),
            "max_tokens": 8, "temperature": 0, "top_p": 1.0, "return_token_ids": True}
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                {"content-type": "application/json", "x-verity-release": str(n), "x-verity-release-run": run})
    sent[n] = time.time()
    with urllib.request.urlopen(rq, timeout=120) as r:
        r.read()
t1 = threading.Thread(target=post, args=(1,)); t1.start(); time.sleep(1.0)
t0 = threading.Thread(target=post, args=(0,)); t0.start(); t0.join(); t1.join()
recs = {}
for _ in range(120):
    recs = {r["user"]: r for r in map(json.loads, (d / "requests.jsonl").read_text().splitlines()) if r.get("user", "").startswith("admit-smoke-")} \
        if (d / "requests.jsonl").exists() else {}
    if len(recs) == 2:
        break
    time.sleep(1)
a, b = recs.get("admit-smoke-0", {}), recs.get("admit-smoke-1", {})
ok = bool(a and b) and sent[1] < sent[0] and a["admitted"] < b["admitted"]
print(json.dumps({"run": run, "http_sent": {"release 1": sent[1], "release 0": sent[0]},
                  "release 0": {k: a.get(k) for k in ("release", "admitted", "arrival", "windows")},
                  "release 1": {k: b.get(k) for k in ("release", "admitted", "arrival", "windows")}, "reordered": ok}))
sys.exit(0 if ok else 1)
EOF2
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "ADMIT SMOKE FAILED"; kill -TERM "$SRV"; wait "$SRV"; exit 12; }
fi

if [ "$ADMIT" = release-order ] && "$PY" -c "import pytest" 2>/dev/null; then   # the pinned vLLM's own sort_batch_req_ids
  "$PY" -m pytest -q -p no:cacheprovider tests/pipeline/test_serve_admit.py -k "pinned or refused or keeps or verifying or one_token" \
    > "$OUT/admit_rows_tests.txt" 2>&1; echo "admit-rows tests rc=$?" | tee -a "$OUT/steps.txt"
fi

if [ "$MAX_MODEL_LEN" -gt 512 ]; then  # one request filling the context: MAX_MODEL_LEN - 64 prompt ids, then all 64 new ones, its window sealed
  # (ignore_eos at max_tokens, not min_tokens: that masks the EOS logit, which the Program's sampling doesn't model, so compile_steps refuses it)
  "$PY" - "$PORT" "$REPO" "$D" "$MAX_MODEL_LEN" <<'EOF2' 2>&1 | tee "$OUT/long_smoke.txt"
import json, sys, time, urllib.request
from pathlib import Path
port, model, d, L = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
P = L - 64
body = {"model": model, "user": "long-smoke", "prompt": [128000] + [1000 + (7 * i) % 100000 for i in range(P - 1)],
        "max_tokens": 64, "temperature": 0, "top_p": 1.0, "return_token_ids": True, "ignore_eos": True}
rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
with urllib.request.urlopen(rq, timeout=300) as r:
    resp = json.loads(r.read())
ch = resp["choices"][0]
n_out = len(ch.get("token_ids") or [])
def at(name, w):   # the window's line of windows.jsonl, or of seals.jsonl, where the off-loop sealer records a seal once done
    f = d / name
    return next((x for x in map(json.loads, f.read_text().splitlines()) if x["window"] == w), {}) if f.exists() else {}
rec, line, seal = {}, {}, {}
for _ in range(300):
    if (d / "requests.jsonl").exists():
        rec = next((r for r in map(json.loads, (d / "requests.jsonl").read_text().splitlines()) if r.get("user") == "long-smoke"), {})
    if rec and rec.get("windows"):
        w = rec["windows"][-1]
        line, seal = at("windows.jsonl", w), at("seals.jsonl", w)
        if line and seal:
            break
    time.sleep(1)
ok = n_out == 64 and len(rec.get("prompt_ids") or []) == P and bool(line.get("passed")) and bool(seal.get("row_dir"))
print(json.dumps({"max_model_len": L, "prompt": P, "new": n_out, "finish": ch.get("finish_reason"), "windows": rec.get("windows"),
                  "check": rec.get("check"), "window_passed": line.get("passed"), "sealed": seal.get("row_dir"),
                  "port_layout": seal.get("port_layout"), "ok": ok}))
sys.exit(0 if ok else 1)
EOF2
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "LONG SMOKE FAILED"; kill -TERM "$SRV"; wait "$SRV"; exit 13; }
fi

"$PY" "$TREE/demos/vllm/serve_load.py" --port "$PORT" --model "$REPO" --bursts "$BURSTS" --per-burst "$PER_BURST" --out "$D/load.jsonl" \
  2>&1 | tee "$OUT/load.log"
sleep 5
for _ in $(seq 1 120); do                                   # the load's windows' advice, before they compile
  pending=0; for k in $(closed); do [ -f "$D/advice/.w$k/windows.json" ] || pending=1; done
  [ "$pending" = 0 ] && break; sleep 5
done
cp "$D/windows.jsonl" "$D/requests.jsonl" "$D/advice/windows.json" "$OUT/" 2>/dev/null
WS=$("$PY" -c "import json,sys; print(' '.join(sorted(json.load(open('$D/advice/windows.json'))['windows'], key=int)))" 2>/dev/null)
FIRST=${WINDOW:-5}
for k in $FIRST $WS; do
  [ -d "$D/windows/w$k" ] || continue
  [ -f "$D/windows/w$k/manifest.json" ] && continue
  "$PY" tools/advice_compile.py "$D" --window "$k" > "$OUT/compile_w$k.log" 2>&1
  rc=$?
  if [ "$rc" = 2 ]; then      # compile_steps refused the window's advice: request by request instead, the refusals named
    "$PY" tools/advice_compile.py "$D" --window "$k" --partial > "$OUT/compile_w${k}_partial.log" 2>&1
    rc="2, --partial rc=$?"
  fi
  echo "$(date -u +%H:%M:%SZ) w$k compile rc=$rc" | tee -a "$OUT/steps.txt"
  "$PY" - "$D" "$k" <<'EOF' | tee -a "$OUT/row_dirs.txt"
import json, sys
from pathlib import Path
from verity_vllm.pipeline import deferred_rows as DR
d, k = Path(sys.argv[1]), sys.argv[2]
w = d / "windows" / f"w{k}"
v = json.loads((w / "commit" / "verdict.json").read_text())
m = json.loads((w / "manifest.json").read_text()) if (w / "manifest.json").exists() else {}
sw = DR.of_record(v["replay_deferred"]["bundles"][0]["path"], str(w)).ctx["served_window"]
seal = next((json.loads(x) for x in (d / "seals.jsonl").read_text().splitlines() if json.loads(x).get("window") == int(k)), {}) \
    if (d / "seals.jsonl").exists() else {}
srw = sw.get("served_rows") or {}
print(json.dumps({"window": int(k), "row_dir": str(w), "bundle": str(w / "commit" / "replay_bundle_p0"), "run_root": v.get("run_root"),
                  "commit_scheme": v.get("commit_scheme"), "manifest_sha256": v["replay_deferred"]["bundles"][0]["manifest_sha256"],
                  "registration": str(d / "advice" / f"w{k}" / "registration.json"), "programs": len(m.get("requests") or {}),
                  "builds": sorted(p.name for p in w.glob("build_request*")),
                  "ctx_served_rows": {key: srw.get(key) for key in ("schema", "binding", "root", "count", "n_bits", "statics", "error") if key in srw} or srw,
                  "ctx_step_input_port": sw.get("step_input_port"), "port_layout": seal.get("port_layout")}))
EOF
  "$PY" - "$D" "$k" <<'EOF' | tee -a "$OUT/identities.txt"
import json, sys
from collections import Counter
from pathlib import Path
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.pipeline import deferred_rows as DR
d, k = Path(sys.argv[1]), sys.argv[2]
w = d / "windows" / f"w{k}"
v = json.loads((w / "commit" / "verdict.json").read_text())
r = DR.of_record(v["replay_deferred"]["bundles"][0]["path"], str(w))
bmap = json.loads((w / "commit" / "binding_map_p0.json").read_text())
store = CommittedStore.of_record(bmap, lambda *_a: None, r.manifest)
pairs = sorted({(e["identity"]["request_id"], int(e["identity"]["step"])) for e in bmap["entries"] if e.get("kind") == "leaf"})
reads = {"gemm_out": ("model.layers.0.mlp.gate_up_proj", "out"), "rope_l0_query": ("model.layers.0.self_attn.rotary_emb", "query"),
         "rope_l15_key": ("model.layers.15.self_attn.rotary_emb", "key"), "norm_scale": ("model.layers.3.input_layernorm", "norm_scale"),
         "rmsnorm_residual": ("model.layers.3.post_attention_layernorm", "1")}
have = {name: sum(store.has(q, s, op, m) for q, s in pairs) for name, (op, m) in reads.items()}
fa2 = sum(any(e["identity"]["op_path"].startswith("fa2.m1/") and e["identity"]["request_id"] == q and e["identity"]["step"] == s
              for e in bmap["entries"]) for q, s in pairs[:8])
how = Counter(e.get("observation", "label/slot") for e in bmap["entries"] if e.get("kind") == "leaf")
raw = sorted({e["identity"]["op_path"] + "/" + e["identity"]["output_member"] for e in bmap["entries"]
              if e.get("kind") == "leaf" and e["identity"]["output_member"] in ("0", "1") and not e["identity"]["op_path"].endswith("layernorm")})
first = pairs[0] if pairs else None
print(json.dumps({"window": int(k), "committer": json.loads((w / "commit" / "runs.jsonl").read_text())["commit"]["committer"],
                  "aliases": len((r.manifest or {}).get("op_path_aliases") or {}), "request_steps": len(pairs), "found": have,
                  "fa2_in_first_8": fa2, "embed_tokens_in": dict(Counter(e.get("name") for e in bmap["entries"]
                                                          if (e["identity"]["op_path"], e["identity"]["output_member"]) == ("model.embed_tokens", "in"))), "observation": dict(how), "raw_slot_members": raw[:6],
                  "lookup": None if first is None else {"request": first[0], "step": first[1] + 1,
                                                       "gate_up_proj/out": store.words(first[0], first[1] + 1, *reads["gemm_out"], 0, 1)[1]
                                                       if not store.has(first[0], first[1] + 1, *reads["gemm_out"]) else "bound"}}))
EOF
done

if [ "${VARIED:-0}" -gt 0 ]; then   # network-accounting's varied traffic, each run numbered as the network certifier releases it
  for i in $(seq 1 "${VARIED_RUNS:-3}"); do
    run=$("$PY" -c "import uuid; print(uuid.uuid4().hex)")
    "$PY" "$TREE/demos/vllm/serve_load.py" --port "$PORT" --model "$REPO" --varied "$VARIED" --rate "${VARIED_RATE:-4}" --prompt-range 16:1024 \
      --release "$run" --jitter-ms "${VARIED_JITTER_MS:-400}" --seed "$i" --out "$D/varied_$i.jsonl" > "$OUT/varied_$i.log" 2>&1
    echo "$(date -u +%H:%M:%SZ) varied run $i ($run) rc=$?" | tee -a "$OUT/steps.txt"
  done
  sleep 10
  "$PY" tools/admit_rows_check.py "$D" --json "$OUT/admit_rows.json" 2>&1 | tee "$OUT/admit_rows.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "ADMIT ROWS FAILED"; kill -TERM "$SRV"; wait "$SRV"; exit 14; }
fi
echo "READY $D" | tee -a "$OUT/steps.txt"

end=$(date -u -d "today $UNTIL" +%s); [ "$end" -lt "$(date -u +%s)" ] && end=$((end + 86400))   # UNTIL past midnight
while [ "$(date -u +%s)" -lt "$end" ] && [ ! -e "$D/stop" ] && kill -0 "$SRV" 2>/dev/null; do sleep 30; done
trap - TERM INT
kill -TERM "$SRV" 2>/dev/null; wait "$SRV"; rc=$?
wait "$ADV"
keep_records
"$PY" tools/admit_rows_check.py "$D" --json "$OUT/admit_rows_final.json" > "$OUT/admit_rows_final.txt" 2>&1; tail -1 "$OUT/admit_rows_final.txt"
tail -3 "$OUT/serve.log"
exit $rc
