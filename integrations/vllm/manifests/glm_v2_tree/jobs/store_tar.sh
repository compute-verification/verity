#!/usr/bin/env bash
# Row-dir files carried through the evidence store, for nodes that can't reach each other and to keep a row's inputs off a node
# that may go away (a RunPod pod's balance).  LIST is a sha256 list with paths relative to the sweep (match_gpu.sh's
# match.sha256, build_record.sh's rowdir.sha256, match_cpu.sh's lists).  `put` tars the files it names (those EXCLUDE, an
# extended regex, doesn't match) into CHUNK_GB parts, each a preserved artifact written and dropped one at a time, then puts the
# set's parts.json (the list's art: id, the parts' in order, each `<row>.tree` it names) as a vllm-rowdir-parts/v1 artifact and
# labels this run `store_tar.parts.NAME` with it.  `get` takes PARTS as that art: id, as RUN:NAME (waiting up to WAIT_MIN
# minutes for RUN's label), or as a parts.json file; it streams the parts into tar under SW, extracting only the paths ONLY
# matches and EXCLUDE doesn't, checks every extracted file against the list, and writes each `<row>.tree` SW lacks.  A
# `<row>.tree` SW holds at another commit is refused (exit 2): copy_rowdir.sh decides between two trees.  `fail` labels this run
# `store_tar.parts.NAME` with "failed: MSG" instead, so a `get` waiting for that set stops (exit 8) rather than waiting it out.
#   research run --on NODE --mem-gb 16 --timeout 4h --custody-ttl 24h --cwd source --project verity --send match.sha256 -- env LIST=match.sha256 NAME=legs [EXCLUDE=/values/] bash integrations/vllm/manifests/glm_v2_tree/jobs/store_tar.sh put
#   research run --on NODE --mem-gb 16 --timeout 4h --custody-ttl 24h --cwd source --project verity -- env PARTS=RUN:legs [ONLY=REGEX] bash integrations/vllm/manifests/glm_v2_tree/jobs/store_tar.sh get
# env: put: LIST (a path, else $RESEARCH_RUN_DIR/inputs/LIST), NAME, [EXCLUDE], [CHUNK=16G]; get: PARTS, [ONLY], [EXCLUDE],
# [WAIT_MIN=600], [FETCH_MIN_MBPS=16], [FETCH_TRIES=3]; fail: NAME, MSG; put and fail: [LABEL_ALSO: run ids labelled too, for a run
# that replaces one a waiting `get` names]; all: [SW], [PY], [EVICT_FREE_GB: after each part, `research data evict` to this much free].
set -uo pipefail
MODE=${1:?put or get}
TREE=$PWD SW=${SW:-/workspace/cp/circuits/glm-v2/rows}
for p in "${PY:-}" /workspace/cp/circuits/venv312/bin/python /workspace/cp/venv312/bin/python; do
  [ -n "$p" ] && [ -x "$p" ] && { PY=$p; break; }
done
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C (research run --custody-ttl)"; exit 4; }
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/integrations/vllm RESEARCH_STORE_CONFIG=$C/store.toml
eval "$("$PY" -m verity_vllm.pipeline.cli epoch custody-env "$C")"
mkdir -p /workspace/research/scratch && S=$(mktemp -d /workspace/research/scratch/store-tar.XXXXXX) || exit 3
trap 'rm -rf "$S"' EXIT
RB=$S/research
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$RB" && chmod +x "$RB"
art_of() { grep -oE 'art:[0-9a-f]{64}' | tail -1; }
evict() { [ -z "${EVICT_FREE_GB:-}" ] || "$RB" data evict --target-free-gb "$EVICT_FREE_GB" --min-mb 64 >/dev/null 2>&1 || true; }
label() {  # label KEY VALUE on this run (and LABEL_ALSO's), pushed: the custody store has no write-through, and another node's `get` reads the remote
  local t
  for t in "$RESEARCH_RUN_ID" ${LABEL_ALSO:-}; do
    "$RB" data label "$t" "$1" "$2" --by circuits --off-vocab --ref "$RESEARCH_RUN_ID" >/dev/null 2>"$S/label.err" || { cat "$S/label.err"; return 1; }
  done
  "$PY" -c 'import sys; from research.store import labels_sync as LS; from research.store.local import LocalStore
r = LS.push_targets(LocalStore(), sys.argv[1:]); sys.exit("label push: " + "; ".join(r["failed"]) if r["failed"] else 0)' "$RESEARCH_RUN_ID" ${LABEL_ALSO:-}
}

if [ "$MODE" = fail ]; then
  label "store_tar.parts.${NAME:?NAME}" "failed: ${MSG:-}" || exit 5
  echo "$NAME: labelled failed on $RESEARCH_RUN_ID (${MSG:-})"; exit 0
fi

if [ "$MODE" = put ]; then
  : "${LIST:?LIST}" "${NAME:?NAME}"
  [ -f "$LIST" ] || LIST=$RESEARCH_RUN_DIR/inputs/$LIST
  [ -s "$LIST" ] || { echo "no list $LIST"; exit 3; }
  awk -v x="${EXCLUDE:-}" 'substr($0, 67) !~ /\.tree$/ && (x == "" || substr($0, 67) !~ x)' "$LIST" >"$OUT/$NAME.sha256"
  awk 'substr($0, 67) ~ /\.tree$/' "$LIST" >"$S/trees"
  echo "$NAME: $(wc -l <"$OUT/$NAME.sha256") of $(wc -l <"$LIST") files${EXCLUDE:+ (EXCLUDE=$EXCLUDE)}"
  (cd "$SW" && cat "$OUT/$NAME.sha256" "$S/trees" | sha256sum --quiet -c -) || { echo "$SW differs from the list"; exit 3; }
  list_art=$("$RB" data put --kind vllm-rowdir-list/v1 --file "$OUT/$NAME.sha256" --meta "{\"set\":\"$NAME\"}" --preserve | art_of)
  [ -n "$list_art" ] || { echo "list put failed"; exit 5; }
  cut -c67- "$OUT/$NAME.sha256" >"$S/paths"
  : >"$OUT/$NAME.parts.txt"
  put_part() {
    local f=$S/$FILE sz h art
    cat >"$f" || return 1
    sz=$(stat -c %s "$f") h=$(sha256sum "$f" | cut -c1-64)
    art=$("$RB" data put --kind vllm-rowdir-tar/v1 --file "$f" --meta "{\"set\":\"$NAME\",\"part\":\"$FILE\",\"sha256\":\"$h\"}" --preserve | art_of)
    [ -n "$art" ] || { echo "$FILE: put failed"; return 1; }
    echo "$FILE $sz $h $art" | tee -a "$OUT/$NAME.parts.txt"
    rm -f "$f"; evict
  }
  export -f put_part evict art_of; export S RB OUT NAME EVICT_FREE_GB
  (cd "$SW" && tar -cf - -T "$S/paths") | SHELL=/bin/bash split -b "${CHUNK:-16G}" -d -a 3 --filter 'put_part' - "$NAME.part."
  ps=("${PIPESTATUS[@]}")
  [ "${ps[*]}" = "0 0" ] || { echo "$NAME: tar or a part's put failed (${ps[*]})"; exit 5; }
  python3 - "$OUT/$NAME.parts.txt" "$list_art" "$NAME" "$(wc -l <"$OUT/$NAME.sha256")" "$SW" "$S/trees" >"$OUT/$NAME.parts.json" <<'EOF'
import json, os, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
trees = {l[66:].strip(): open(os.path.join(sys.argv[5], l[66:].strip())).read().strip() for l in open(sys.argv[6]) if l.strip()}
print(json.dumps({"schema": "glm-v2/store-tar/v1", "set": sys.argv[3], "list": sys.argv[2], "files": int(sys.argv[4]), "trees": trees,
                  "parts": [{"name": n, "bytes": int(b), "sha256": h, "art": a} for n, b, h, a in rows]}, indent=1))
EOF
  cp "$OUT/$NAME.parts.json" "$OUT/parts.json"
  parts_art=$("$RB" data put --kind vllm-rowdir-parts/v1 --file "$OUT/$NAME.parts.json" --meta "{\"set\":\"$NAME\"}" --ref "list=$list_art" \
              --preserve | art_of)
  [ -n "$parts_art" ] || { echo "$NAME: parts.json put failed"; exit 5; }
  label "store_tar.parts.$NAME" "$parts_art" || { echo "$NAME: labelling $RESEARCH_RUN_ID failed"; exit 5; }
  echo "$NAME: $(wc -l <"$OUT/$NAME.parts.txt") parts, $(awk '{s+=$2} END {printf "%.1f", s/2^30}' "$OUT/$NAME.parts.txt") GiB, list $list_art," \
       "parts $parts_art (label store_tar.parts.$NAME on $RESEARCH_RUN_ID)"
  exit 0
fi

[ "$MODE" = get ] || { echo "MODE $MODE: put or get"; exit 2; }
P=${PARTS:?PARTS: an art: id, RUN:NAME or a parts.json}
if [[ $P =~ ^(r[0-9]{8}-[0-9]{6}-[0-9a-f]+):(.+)$ ]]; then
  run=${BASH_REMATCH[1]} key=store_tar.parts.${BASH_REMATCH[2]} waited=0
  until P=$("$RB" data labels "$run" --key "$key" --json 2>/dev/null |
            python3 -c 'import json, sys; l = json.load(sys.stdin); print(l[-1]["value"] if l else "")' 2>/dev/null) && [ -n "$P" ]; do
    [ "$waited" -lt "${WAIT_MIN:-600}" ] || { echo "no $key label on $run after $waited min"; exit 7; }
    [ $((waited % 30)) != 0 ] || echo "$(date -u +%H:%MZ) waiting for $run's $key"
    sleep 60; waited=$((waited + 1))
  done
  echo "$run's $key: $P (after $waited min)"
  [[ $P == art:* ]] || exit 8
fi
if [[ $P == art:* ]]; then
  "$RB" data fetch "$P" --to "$S/parts.d" >/dev/null || { echo "$P: fetch failed"; exit 5; }
  P=$(find "$S/parts.d" -type f -name '*.json' | head -1)
fi
[ -s "$P" ] || { echo "no parts.json at $P"; exit 3; }
read -r NAME list_art < <(python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["set"], d["list"])' "$P")
"$RB" data fetch "$list_art" --to "$S/list" >/dev/null || { echo "list fetch failed"; exit 5; }
L=$(find "$S/list" -type f | head -1)
awk -v o="${ONLY:-}" -v x="${EXCLUDE:-}" '(o == "" || substr($0, 67) ~ o) && (x == "" || substr($0, 67) !~ x)' "$L" >"$OUT/got-$NAME.sha256"
cut -c67- "$OUT/got-$NAME.sha256" >"$S/paths"
echo "$NAME: extracting $(wc -l <"$OUT/got-$NAME.sha256") of $(wc -l <"$L") files${ONLY:+ (ONLY=$ONLY)}${EXCLUDE:+ (EXCLUDE=$EXCLUDE)} into $SW"
mkdir -p "$SW"
python3 - "$P" "$SW" <<'EOF' || exit 2
import json, os, sys
for t, at in json.load(open(sys.argv[1])).get("trees", {}).items():
    p = os.path.join(sys.argv[2], t)
    have = open(p).read().strip() if os.path.exists(p) else None
    if have is None:
        open(p, "w").write(at + "\n")
        print(f"{t}: {at}")
    elif have != at:
        sys.exit(f"{t}: {sys.argv[2]} holds {have}, the set {at}: copy_rowdir.sh decides between two trees")
EOF
python3 -c 'import json, sys; [print(p["art"], p["sha256"], p["bytes"]) for p in json.load(open(sys.argv[1]))["parts"]]' "$P" >"$S/parts"
while read -r art h b; do
  d=$S/part && rm -rf "$d"
  # the client's own deadline lets a body trickle at 64 KiB/s (a 16 GiB part for days: vy-nebius-cpu-5, 24 Oct); a fresh process
  # gets a fresh connection, so a part slower than FETCH_MIN_MBPS on average is fetched again, up to FETCH_TRIES times
  for try in $(seq "${FETCH_TRIES:-3}"); do
    rm -rf "$d"
    timeout -k 30 $((600 + b / (${FETCH_MIN_MBPS:-16} << 20))) "$RB" data fetch "$art" --to "$d" >/dev/null && break
    echo "$art: fetch try $try of ${FETCH_TRIES:-3} failed or timed out (rc $?)" >&2
    [ "$try" -lt "${FETCH_TRIES:-3}" ] || exit 5
  done
  f=$(find "$d" -type f | head -1)
  [ "$(sha256sum "$f" | cut -c1-64)" = "$h" ] || { echo "$art: sha256 differs" >&2; exit 5; }
  cat "$f"; rm -rf "$d"; evict
done <"$S/parts" | (cd "$SW" && tar -xf - -T "$S/paths")
ps=("${PIPESTATUS[@]}")
[ "${ps[*]}" = "0 0" ] || { echo "$NAME: a part's fetch or tar failed (${ps[*]})"; exit 5; }
(cd "$SW" && sha256sum --quiet -c -) <"$OUT/got-$NAME.sha256" || { echo "$NAME: extracted files differ from the list"; exit 6; }
echo "$NAME: every extracted file equals the list ($(wc -l <"$OUT/got-$NAME.sha256") files)"
