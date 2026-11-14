#!/usr/bin/env bash
# lean_shard.sh: Flock's Lean verifier on the sessions proving runs kept (window.py prove's out/kept), one shard of their
# statements a research run. Launched from a checkout whose `research` reads the store, the proving runs' own tree comes in
# as a checksummed kit, and the kept sessions from the proving runs' custody records:
#   research run --on auto --as LANE --source TREE --cwd source --store-read --mem-gb GB --cpus 16 \
#     --send SHARD.units --env UNITS=SHARD.units --env KIT_ART=art:... --env KIT_SHA256=HEX \
#     --env RECORDS='RUN=art:...;RUN=art:...' -- bash benchmarks/served_units/lean_shard.sh
# KIT_ART: `git archive --prefix=verity-COMMIT12/ COMMIT | gzip -n` in the store; its sha256 must be KIT_SHA256, or nothing runs.
# RECORDS: each proving run's custody record (its .custody's run_record); a statement's out/kept/NAME/ (pub.bin, stage.json,
#   pin.txt, sessions/) is fetched from the record that lists it, and every file is held to the sha256 the record lists.
# UNITS: window.py's units lines (UNIT@N[,N] and its estimates, only the first field read), in order.
#   0. the kit fetched, held to KIT_SHA256 and unpacked; the kept files fetched and held to their records (out/kept.tsv)
#   1. the verifier and flock-read-tmpl built in the kit (the executables the lakefile roots at FlockVerify and
#      FlockReadTmpl), and the stager's venv (numpy, blake3)
#   2. pass 1, each statement in order: restaged by stage.py from this tree (prove_time.sh's stage step), its circuit held
#      to the kept stage.json's circuit_sha512 and pin.txt (a statement that differs is refused and verified no further);
#      program-word.json written from the restaged stage.json's word (the Program of the word Definition, which lean_verify.py records and
#      compares by name, and holds to nothing in the circuit); one kept session verified (`verity bench prove lean-verify
#      --repeat 1`, a tree of that session directory and the statement's files) at the statement its circuit's META names
#      (--statement: a templated statement's, a census read's, is verity/flock-circuit/tmpl)
#   3. pass 2: every other kept session of each statement, restaging it again only if its circuit was dropped for disk
# A refusal is recorded and the shard goes on to the next session. out/lean.tsv: name, session, accepted, label, wall s,
# peak GiB, circuit bytes; out/lean/NAME/SESSION/lean-verify.json each run's record; out/restage.tsv each restage.
# Env: UNITS KIT_ART KIT_SHA256 RECORDS; LEAN_DISK_GB (default 200: circuits kept for pass 2 while their bytes fit); LEAN_MAX
# (only the first so many statements).
set -uo pipefail
R=$RESEARCH_RUN_DIR; IN=$R/inputs; O=$R/out; mkdir -p "$O/lean" "$O/restage"
UNITS=${UNITS:?}; [ -f "$UNITS" ] || UNITS=$IN/$UNITS
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/shard.log"; }
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
export PATH=$HOME/.elan/bin:$HOME/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1
MAIN=$PWD
rs() { (cd "$MAIN" && uv run -q --project tools/research research "$@"); }
log "main $(git -C "$MAIN" rev-parse --short=12 HEAD), $(hostname), $(nproc) cpus, $(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB"

log "0. the kit and the kept files"
kd=$(rs data fetch "${KIT_ART:?}" --to "$S/kitdl" 2>> "$O/fetch.log" | tail -1)
kf=$( { [ -f "$kd" ] && echo "$kd"; } || find "$S/kitdl" -type f -name '*.tar.gz' | head -1)
got=$(sha256sum "$kf" 2>/dev/null | cut -d' ' -f1)
[ -n "$got" ] && [ "$got" = "${KIT_SHA256:?}" ] || { log "KIT REFUSED: $KIT_ART's file ${kf:-absent} has sha256 ${got:-none}, not $KIT_SHA256"; exit 2; }
mkdir -p "$S/kit" && tar -xzf "$kf" -C "$S/kit" && rm -rf "$S/kitdl" || { log "kit not unpacked"; exit 2; }
REPO=$(ls -d "$S"/kit/*/ | head -1); REPO=${REPO%/}; cd "$REPO" || exit 2
log "kit $KIT_ART, sha256 ${got:0:16}, unpacked as $(basename "$REPO")"
# Flock is under verity/protocols/proofs/flock in a kit of main, under verity/protocols/flock in a kit from before that move
F=$REPO/verity/protocols/proofs/flock; [ -d "$F" ] || F=$REPO/verity/protocols/flock; L=$F/verifier
IFS=';' read -r -a recs <<< "${RECORDS:?}"
for rr in "${recs[@]}"; do rs data fetch "${rr#*=}" --list --json > "$S/list-${rr%%=*}.json" 2>> "$O/fetch.log" || { log "record ${rr#*=} (${rr%%=*}) not listed"; exit 2; }; done
printf 'name\trun\trecord\tfiles\tbytes\tvs_record\n' > "$O/kept.tsv"
kept() {   # kept NAME: $S/kept/NAME from the record that lists it, every file held to the record's sha256; 0 or why not
  local name=$1 rr run art
  for rr in "${recs[@]}"; do
    run=${rr%%=*}; art=${rr#*=}
    python3 -c 'import json, sys; sys.exit(0 if any(r["path"].startswith(sys.argv[2]) for r in json.load(open(sys.argv[1]))) else 1)' \
      "$S/list-$run.json" "out/kept/$name/" || continue
    rs data fetch "$art" --to "$S/rec/$run" --path "out/kept/$name" >> "$O/fetch.log" 2>&1 || { log "$name: fetch from $art failed"; return 1; }
    python3 - "$S/list-$run.json" "out/kept/$name/" "$S/rec/$run" "$name" "$run" "$art" >> "$O/kept.tsv" <<'EOF' || { log "$name: KEPT FILES REFUSED: $(tail -1 "$O/kept.tsv")"; return 1; }
import hashlib, json, sys
from pathlib import Path
rows, pre, root, name, run, art = json.load(open(sys.argv[1])), *sys.argv[2:]
bad = [r["path"] for r in rows if r["path"].startswith(pre) and (not (Path(root) / r["path"]).is_file()
       or hashlib.sha256((Path(root) / r["path"]).read_bytes()).hexdigest() != r["sha256"])]
n = [r for r in rows if r["path"].startswith(pre)]
print("\t".join(map(str, (name, run, art, len(n), sum(r["bytes"] for r in n), f"DIFFERS: {bad[:3]}" if bad else "equal"))))
sys.exit(1 if bad else 0)
EOF
    mkdir -p "$S/kept" && mv "$S/rec/$run/out/kept/$name" "$S/kept/$name"; return 0
  done
  return 1
}

log "1. build"
uv sync -q --all-packages --extra torch-cpu > "$O/uv.log" 2>&1 || { tail -20 "$O/uv.log"; exit 3; }
exes=$(uv run python -c 'import sys, tomllib
r = {e.get("root"): e["name"] for e in tomllib.load(open(sys.argv[1], "rb"))["lean_exe"]}
print(r["FlockVerify"], r["FlockReadTmpl"])' "$L/lakefile.toml")
exe=${exes% *}; rt=${exes#* }
t0=$(date +%s); (cd "$L" && lake build "$exe" "$rt") > "$O/build.log" 2>&1 || { tail -40 "$O/build.log"; log "VERIFIER BUILD FAILED"; exit 3; }
EXE=$L/.lake/build/bin/$exe; sha256sum "$EXE" > "$O/verifier.sha256"; log "verifier $exe and $rt built in $(( $(date +%s) - t0 )) s, sha256 $(cut -c1-16 "$O/verifier.sha256")"
export PATH=$L/.lake/build/bin:$PATH
PY=$S/py/bin/python; uv venv -q --python 3.12 "$S/py" && uv pip install -q --python "$PY" numpy blake3 || exit 3
SP=$REPO:$F/prover/python:$REPO/tools/verity/circuit_check/src:$REPO/adapters/vllm

nm() { "$PY" -c 'import importlib.util as u, sys; s = u.spec_from_file_location("w", "benchmarks/served_units/window.py"); m = u.module_from_spec(s); sys.modules["w"] = m; s.loader.exec_module(m); print(m.name(sys.argv[1], sys.argv[2]))' "$1" "$2"; }
restage() {   # restage UNIT N NAME: $S/st/NAME holds the statement and its program-word.json, held to kept/NAME; 0 or why not
  local u=$1 n=$2 name=$3 d=$S/st/$3 k=$S/kept/$3 want got
  [ -s "$d/.ok" ] && return 0
  rm -rf "$d"; mkdir -p "$d" "$O/restage/$name"
  local t=$(date +%s)
  PYTHONPATH=$SP /usr/bin/time -v -o "$O/restage/$name/time.txt" "$PY" benchmarks/served_units/stage.py "$u" "$n" "$d" > "$O/restage/$name/stage.log" 2>&1 \
    || { log "$name: RESTAGE FAILED: $(tail -1 "$O/restage/$name/stage.log" | cut -c1-200)"; rm -rf "$d"; return 1; }
  want=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["circuit_sha512"])' "$k/stage.json")
  got=$(sha512sum "$d/circuit.txt" | cut -d' ' -f1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$(( $(date +%s) - t ))" "$(awk '/Maximum resident/ {printf "%.2f", $NF/1048576}' "$O/restage/$name/time.txt")" \
    "$(stat -c %s "$d/circuit.txt")" "${got:0:16}" "$([ "$got" = "$want" ] && [ "$(head -c 128 "$k/pin.txt")" = "$want" ] && echo equal || echo DIFFERS)" >> "$O/restage.tsv"
  [ "$got" = "$want" ] && [ "$(head -c 128 "$k/pin.txt")" = "$want" ] \
    || { log "$name: REFUSED: the restaged circuit ${got:0:16} is not the kept statement's ${want:0:16}"; rm -rf "$d"; return 1; }
  cmp -s "$d/pub.bin" "$k/pub.bin" || log "$name: the restaged pub.bin differs from the kept one (values are seeded by the staged name); the kept one, which the session was proved on, is verified"
  kw=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["word"])' "$k/stage.json"); rw=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["word"])' "$d/stage.json")
  [ "$kw" = "$rw" ] || log "$name: kept as $kw; restaged as $rw, the same circuit"
  PYTHONPATH=$SP "$PY" - "$d" > "$O/restage/$name/program-word.json.txt" 2>&1 <<'EOF' || { log "$name: program-word.json NOT WRITTEN: $(tail -1 "$O/restage/$name/program-word.json.txt" | cut -c1-200)"; rm -rf "$d"; return 1; }
import hashlib, json, sys
from pathlib import Path
from verity.circuits.codec import descriptor_chunks
from verity.circuits.program import Program
from verity_vllm.pipeline.serving_rows import template
d = Path(sys.argv[1]); word = json.loads((d / "stage.json").read_text())["word"]
h = hashlib.sha512()
with open(d / "program-word.json", "wb") as f:
    for c in descriptor_chunks(Program(template(word).fn)):
        f.write(c); h.update(c)
desc = json.loads((d / "program-word.json").read_text())
assert desc.get("program_fn") == word and "annotations" not in desc, (desc.get("program_fn"), word)
print(json.dumps({"word": word, "sha512": h.hexdigest()}))
EOF
  echo ok > "$d/.ok"
}
check_session() {   # check_session NAME SESSION_DIR
  local name=$1 sd=$2 sid=$(basename "$2") t=$S/t/$1-$(basename "$2") d=$S/st/$1 o=$O/lean/$1/$(basename "$2")
  rm -rf "$t"; mkdir -p "$t" "$o"
  cp -r "$sd" "$t/session"; ln -s "$d/circuit.txt" "$t/circuit.txt"; ln -s "$d/program-word.json" "$t/program-word.json"
  ln -s "$d/stage.json" "$t/stage.json"   # the restaged one, its circuit held to the kept pin
  for f in pub.bin partition.json; do [ -e "$S/kept/$name/$f" ] && ln -s "$S/kept/$name/$f" "$t/$f"; done
  [ -d "$S/kept/$name/tables" ] && cp -r "$S/kept/$name/tables" "$t/tables"   # rglob descends into no linked directory
  local st; st=$("$PY" -c 'import json, sys; f = open(sys.argv[1], "rb"); f.readline(); print(json.loads(f.readline()[5:])["statement"])' "$d/circuit.txt")
  uv run verity bench prove lean-verify --lean "$EXE" --proof "$t" --statement "$st" --repeat 1 --out "$o" > "$o/stdout.txt" 2>&1; local rc=$?
  "$PY" - "$o/lean-verify.json" "$name" "$sid" "$rc" "$(stat -c %s "$d/circuit.txt")" >> "$O/lean.tsv" <<'EOF'
import json, sys
from pathlib import Path
p, name, sid, rc, cb = sys.argv[1:]
r = json.loads(Path(p).read_text()) if Path(p).exists() else {}
run = ((r.get("verifier") or {}).get("runs") or [{}])[0]
why = "" if r.get("accepted") else run.get("why") or (run.get("verdict") or {}).get("why", "")
print("\t".join(map(str, (name, sid, r.get("accepted", f"no record (exit {rc})"), r.get("label") or why, run.get("wall_s", ""),
                          run.get("maxrss_gib", ""), cb))))
EOF
  log "$name $sid: exit $rc, $(tail -1 "$O/lean.tsv" | cut -f3-6)"
  rm -rf "$t"
}
printf 'name\tsession\taccepted\tlabel\twall_s\tpeak_gib\tcircuit_bytes\n' > "$O/lean.tsv"
printf 'name\tseconds\tpeak_gb\tcircuit_bytes\tpin\tvs_kept\n' > "$O/restage.tsv"
mapfile -t stmts < <(grep -v '^#' "$UNITS" | cut -f1 | while IFS= read -r line; do u=${line%@*}; for n in ${line##*@}; do n=${n//,/ }; for m in $n; do echo "$u	$m"; done; done; done)
[ -n "${LEAN_MAX:-}" ] && stmts=("${stmts[@]:0:$LEAN_MAX}")
log "${#stmts[@]} statements"
for pass in 1 2; do
  log "pass $pass"
  for s in "${stmts[@]}"; do
    u=${s%	*}; n=${s##*	}; name=$(nm "$u" "$n")
    [ $pass = 2 ] || [ -d "$S/kept/$name" ] || kept "$name" || { log "$name: in no record of RECORDS, or its files differ"; continue; }
    [ -d "$S/kept/$name/sessions" ] || { [ $pass = 1 ] && log "$name: not kept by the GPU run"; continue; }
    mapfile -t ss < <(ls -d "$S/kept/$name"/sessions/*/ 2>/dev/null | sed 's,/$,,' | while read -r x; do [ -s "$x/session.json" ] && echo "$x"; done | sort)
    [ ${#ss[@]} -gt 0 ] || { [ $pass = 1 ] && log "$name: no kept session"; continue; }
    if [ $pass = 1 ]; then todo=("${ss[-1]}"); else todo=("${ss[@]:0:${#ss[@]}-1}"); fi
    [ ${#todo[@]} -gt 0 ] || continue
    restage "$u" "$n" "$name" || continue
    for sd in "${todo[@]}"; do check_session "$name" "$sd"; done
    used=$(du -s --block-size=1G "$S/st" 2>/dev/null | cut -f1)
    { [ $pass = 2 ] || [ "${used:-0}" -gt "${LEAN_DISK_GB:-200}" ]; } && rm -rf "$S/st/$name"
  done
done
log "done: $(awk -F'\t' 'NR > 1 && $3 == "True"' "$O/lean.tsv" | wc -l) accepted of $(awk 'NR > 1' "$O/lean.tsv" | wc -l) sessions"
