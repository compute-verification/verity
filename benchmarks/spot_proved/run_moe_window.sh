#!/usr/bin/env bash
# One opened MoE layer of a spot capture cut per expert (`moe_window.py`) and audited from its records (`records_audit.py
# audit`), on a fresh pod from a clone of the tree (`research run --cwd clone`): run_audit.sh's builds and flock-firewall, then
# three arms, each staged with `uv run` and audited under the registration's pins, the circuit and run root the verifier holds
# it to from `moe_window.py --circuit-only`:
#   served  the layer's served values: the audit's negatives and MW_FLIPS proof edits, each of which Lean must refuse
#   forged  the same window, every w13 GEMM unit's output off by one bit (--forge w13): an audit --forged, Lean must refuse it
#   bench   the same shapes, tiles and draw with benchmark values (--bench MW_SEED), its tables self-registered
# then h.json: by template, the served and benchmark sessions' seconds, their ratio, and the card's h (MW_CARD, a spot-card/v1
# file) times that ratio. Every arm's stage.json and audit summary go to $OUT/<arm>/. A relative MW_TABLES, MW_CARD or MW_CAPTURE
# that is not here is a --send file, under $RESEARCH_RUN_DIR/inputs/.
#   MW_CAPTURE=DIR|TAR.GZ MW_STEP=S MW_LAYER=L (MW_TABLES=FILE MW_KEY=FILE | MW_SELF=1) [MW_RANK=0] [MW_TOKENS=1]
#   [MW_LAW=stratified:1] [MW_CAP=2048] [MW_BLOCKS="--block w13=1 --block w2=32"] [MW_ARMS="served forged bench"] [MW_SEED=1]
#   [MW_FLIPS=4] [MW_CARD=FILE] [MW_WARM=2] [MW_CHECKPOINT=DIR] bash benchmarks/spot_proved/run_moe_window.sh
#   MW_MODE=build ...  # the builds only
#   MW_MODE=synth ...  # tests/test_moe_window.write_capture's synthetic capture in place of MW_CAPTURE (its step, layer and
#                      # blocks, two tokens, members of 4): the smoke run
set -uo pipefail
SRC=$(pwd)
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
SP_MODE=build bash benchmarks/spot_proved/run_audit.sh > "$OUT/builds.log" 2>&1 || { tail -40 "$OUT/builds.log"; exit 7; }
export PATH="$HOME/.elan/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export HZ_CONFIGS=spot HZ_MEM_GB=${SP_MEM_GB:-400} HZ_CACHE=${SP_CACHE:-/workspace/spot-cache}
source benchmarks/pouw/hidden_zk/job.sh
setup > /dev/null
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(flock 9; cd "${LEAN%/.lake/*}" && lake build flock-firewall) 9> "$HZ_CACHE/.lock-firewall" > "$OUT/build-firewall.log" 2>&1 \
  || { tail -40 "$OUT/build-firewall.log"; exit 7; }
[ -x "$FC_FIREWALL_LEAN" ] || { log "missing: $FC_FIREWALL_LEAN"; exit 7; }
[ "${MW_MODE:-run}" = build ] && { log "moe-window: builds done"; exit 0; }
uv sync -q --all-packages --extra torch-cpu > "$OUT/uv-sync.log" 2>&1 || { tail -20 "$OUT/uv-sync.log"; exit 7; }
UPY="uv run -q --extra torch-cpu python"

if [ "${MW_MODE:-run}" = synth ]; then
  got=$($UPY - "$W/capture" <<'EOF'
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/spot_proved/tests")
import test_moe_window as T
spec = importlib.util.spec_from_file_location("spot_moe_window", "benchmarks/spot_proved/moe_window.py")
MW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MW)
T.write_capture(Path(sys.argv[1]), MW)
print("SYNTH", sys.argv[1], T.STEP, T.LAYER, " ".join(T.BLOCKS))
EOF
  ) || { log "synth: the capture writer FAILED"; exit 2; }
  read -r _ MW_CAPTURE MW_STEP MW_LAYER MW_BLOCKS <<< "$(grep '^SYNTH ' <<< "$got")"
  MW_SELF=1 MW_TOKENS=${MW_TOKENS:-2} MW_CAP=${MW_CAP:-4}
  unset MW_TABLES
fi
sent() { [ -z "$1" ] || [ -e "$1" ] && echo "$1" || echo "$RESEARCH_RUN_DIR/inputs/$1"; }   # a --send file by its name
MW_TABLES=$(sent "${MW_TABLES:-}") MW_CARD=$(sent "${MW_CARD:-}")
CAP=$(sent "${MW_CAPTURE:?}")
case $CAP in
  *.tar.gz) mkdir -p "$W/cap" && tar -xzf "$CAP" -C "$W/cap" || exit 2
            CAP=$(dirname "$(dirname "$(find "$W/cap" -name commit.json -path "*step${MW_STEP:?}*" | head -1)")") ;;
esac
LAW=${MW_LAW:-stratified:1}
BASE=(--capture "$CAP" --step "${MW_STEP:?}" --layer "${MW_LAYER:?}" --rank "${MW_RANK:-0}" --tokens "${MW_TOKENS:-1}"
      --cap "${MW_CAP:-2048}" --law "$LAW" ${MW_CHECKPOINT:+--checkpoint "$MW_CHECKPOINT"})
if [ -n "${MW_TABLES:-}" ]; then REG=(--tables "$MW_TABLES" --key "${MW_KEY:?}")
elif [ "${MW_SELF:-0}" = 1 ]; then
  read -ra BLK <<< "${MW_BLOCKS:---block w13=1 --block w2=32}"; REG=(--self-register --key "$W/served.key" "${BLK[@]}")
else log "moe-window: give MW_TABLES and MW_KEY (the rank's registration), or MW_SELF=1 (a registration made after serving)"; exit 2; fi
MW=benchmarks/spot_proved/moe_window.py
RA=benchmarks/prove_path/records_audit.py

arm() {   # arm NAME [moe_window flags]: stage NAME's window, then audit it; -> the audit's exit
  local name=$1 o=$OUT/$1 d=$W/$1 held rc; shift
  mkdir -p "$o" "$d"
  local args=("${BASE[@]}" "$@")
  log "moe-window $name: staging"
  $UPY "$MW" "${args[@]}" --out "$d/c" --circuit-only > "$o/circuit.json" 2> "$o/circuit.err" \
    || { log "$name: --circuit-only FAILED: $(tail -1 "$o/circuit.err")"; return 2; }
  $UPY "$MW" "${args[@]}" --out "$d/stage" > "$o/stage.log" 2>&1
  rc=$?
  cp "$d/stage/stage.json" "$o/" 2>/dev/null; cp "$d/stage/tables.json" "$o/" 2>/dev/null
  [ $rc = 0 ] || { log "$name: staging FAILED: $(tail -1 "$o/stage.log")"; return 2; }
  log "moe-window $name: $(grep '^STAGED ' "$o/stage.log" | cut -c1-600)"
  held=$("$PY" -c "import json, sys; c = json.load(open(sys.argv[1])); print(c['circuit'], c['run'])" "$o/circuit.json")
  local flags=()
  case $name in
    served) flags=(--flip-proofs "${MW_FLIPS:-4}") ;;
    forged) flags=(--forged --skip-negatives) ;;
    *) flags=(--skip-negatives) ;;
  esac
  if [ "${MW_WARM:-2}" -gt 0 ]; then
    "$PY" "$RA" warm --window "$d/stage/window" --law "$LAW" --compose-cache "$HZ_CACHE/moe-compose" --jobs "${MW_WARM:-2}" \
      > "$o/warm.log" 2>&1 || log "$name: warm FAILED: $(tail -1 "$o/warm.log")"
    log "moe-window $name: $(grep '^WARM ' "$o/warm.log")"
    flags+=(--compose-cache "$HZ_CACHE/moe-compose")
  fi
  log "moe-window $name: audit ${flags[*]}"
  "$PY" "$RA" audit --window "$d/stage/window" --law "$LAW" --circuit "${held% *}" --run "${held#* }" \
    --registered "$d/stage/tables.json" --m0-pythonpath "$SRC" --bin "$BIN" --lean "$LEAN" --firewall "$FC_FIREWALL_LEAN" \
    --out "$d/audit" "${flags[@]}" > "$o/audit.log" 2>&1
  rc=$?
  for f in summary.json window/audit.json; do [ -e "$d/audit/$f" ] && cp "$d/audit/$f" "$o/$(basename "$f")"; done
  log "moe-window $name: audit exit $rc: $(grep -E '^(RECORDS|PROOF-EDITS) ' "$o/audit.log" | cut -c1-900)"
  return $rc
}

fail=0
for a in ${MW_ARMS:-served forged bench}; do
  case $a in
    served) arm served "${REG[@]}" || fail=1 ;;
    forged) arm forged "${REG[@]}" --forge w13 || fail=1 ;;
    bench) blocks=$("$PY" -c "import json, sys; print(' '.join(f'--block {k}={v}' for k, v in json.load(open(sys.argv[1]))['blocks'].items()))" \
             "$OUT/served/stage.json" 2>/dev/null) || blocks=${MW_BLOCKS:---block w13=1 --block w2=32}
           read -ra BB <<< "$blocks"
           arm bench --bench "${MW_SEED:-1}" --key "$W/bench.key" "${BB[@]}" || fail=1 ;;
  esac
done
"$PY" - "$OUT" "${MW_CARD:-}" <<'EOF' | tee -a "$OUT/job.log"
import json, re, sys
from pathlib import Path

out, card = Path(sys.argv[1]), sys.argv[2]


def sessions(arm):
    f = out / arm / "audit.json"
    got = {}
    for key, s in (json.loads(f.read_text()).get("statement") or {}).items() if f.is_file() else ():
        t = s.get("timing") or {}
        got.setdefault(re.sub(r"/c\d+$", "", key), []).append(
            {"member": key, "session_s": s.get("session_s"), "prove_total_s": t.get("prove_total_s"), "units": t.get("units"),
             "k_log": t.get("k_log"), "lean_verify_s": s.get("lean_verify_s")})
    return got


def total(xs, k):
    v = [x[k] for x in xs]
    return sum(v) if v and None not in v else None


served, bench = sessions("served"), sessions("bench")
rows = {r["block"]: r for r in (json.loads(Path(card).read_text())["value"]["blocks"] if card else [])}
h = {}
for t in sorted(set(served) | set(bench)):
    s, b = served.get(t, []), bench.get(t, [])
    sp, bp = total(s, "prove_total_s"), total(b, "prove_total_s")
    ratio = sp / bp if sp and bp else None
    c = rows.get(t) or {}
    h[t] = {"served": s, "bench": b, "ratio_prove": ratio,
            "ratio_session": (total(s, "session_s") / total(b, "session_s")) if total(s, "session_s") and total(b, "session_s") else None,
            "card": {k: c.get(k) for k in ("overhead", "fixed_seconds", "seconds_per_block", "statement_blocks", "k_log", "macs")} if c else None,
            "served_h": c["overhead"] * ratio if c.get("overhead") and ratio else None}
(out / "h.json").write_text(json.dumps(h, indent=1, sort_keys=True))
for t, x in h.items():
    print("H " + json.dumps({"template": t[:90], "ratio_prove": x["ratio_prove"], "card_h": (x["card"] or {}).get("overhead"),
                             "served_h": x["served_h"]}))
EOF
log "moe-window: done, fail $fail"
exit $fail
