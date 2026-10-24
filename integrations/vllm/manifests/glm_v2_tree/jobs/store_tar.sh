#!/usr/bin/env bash
# Row-dir files carried through the evidence store, for nodes that can't reach each other and to keep a row's inputs off a node
# that may go away (a RunPod pod's balance).  LIST is a sha256 list with paths relative to the sweep (match_gpu.sh's
# match.sha256, build_record.sh's rowdir.sha256, match_cpu.sh's lists).  `put` tars the files it names (those EXCLUDE, an
# extended regex, doesn't match) into CHUNK_GB parts, each a preserved artifact written and dropped one at a time, and writes
# $OUT/parts.json: the list (as an artifact too) and the parts' art: ids in order.  `get` streams those parts into tar under SW,
# extracting only the paths ONLY matches when set, and checks every extracted file against the list.  `<row>.tree` is never
# carried: each node's first job writes its own, and copy_rowdir.sh's rule for two trees applies.
#   research run --on NODE --mem-gb 16 --timeout 4h --custody-ttl 24h --cwd source --project verity --send match.sha256 -- env LIST=match.sha256 NAME=legs [EXCLUDE=/values/] bash integrations/vllm/manifests/glm_v2_tree/jobs/store_tar.sh put
#   research run --on NODE --mem-gb 16 --timeout 4h --custody-ttl 24h --cwd source --project verity --send parts.json -- env [ONLY=REGEX] bash integrations/vllm/manifests/glm_v2_tree/jobs/store_tar.sh get
# env: put: LIST (a path, else $RESEARCH_RUN_DIR/inputs/LIST), NAME, [EXCLUDE], [CHUNK_GB=16], [EVICT_FREE_GB: after each part,
# `research data evict` to this much free]; get: [PARTS=$RESEARCH_RUN_DIR/inputs/parts.json], [ONLY]; both: [SW], [PY].
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

if [ "$MODE" = put ]; then
  : "${LIST:?LIST}" "${NAME:?NAME}"
  [ -f "$LIST" ] || LIST=$RESEARCH_RUN_DIR/inputs/$LIST
  [ -s "$LIST" ] || { echo "no list $LIST"; exit 3; }
  awk -v x="${EXCLUDE:-}" 'substr($0, 67) !~ /\.tree$/ && (x == "" || substr($0, 67) !~ x)' "$LIST" >"$OUT/$NAME.sha256"
  echo "$NAME: $(wc -l <"$OUT/$NAME.sha256") of $(wc -l <"$LIST") files${EXCLUDE:+ (EXCLUDE=$EXCLUDE)}"
  (cd "$SW" && sha256sum --quiet -c -) <"$OUT/$NAME.sha256" || { echo "$SW differs from the list"; exit 3; }
  list_art=$("$RB" data put --kind vllm-rowdir-list/v1 --file "$OUT/$NAME.sha256" --preserve | art_of)
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
  (cd "$SW" && tar -cf - -T "$S/paths") | SHELL=/bin/bash split -b "${CHUNK_GB:-16}G" -d -a 3 --filter 'put_part' - "$NAME.part."
  [ "${PIPESTATUS[0]}" = 0 ] && [ "${PIPESTATUS[1]}" = 0 ] || { echo "$NAME: tar or a part's put failed"; exit 5; }
  python3 - "$OUT/$NAME.parts.txt" "$list_art" "$NAME" "$(wc -l <"$OUT/$NAME.sha256")" >"$OUT/$NAME.parts.json" <<'EOF'
import json, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({"schema": "glm-v2/store-tar/v1", "set": sys.argv[3], "list": sys.argv[2], "files": int(sys.argv[4]),
                  "parts": [{"name": n, "bytes": int(b), "sha256": h, "art": a} for n, b, h, a in rows]}, indent=1))
EOF
  cp "$OUT/$NAME.parts.json" "$OUT/parts.json"
  echo "$NAME: $(wc -l <"$OUT/$NAME.parts.txt") parts, $(awk '{s+=$2} END {printf "%.1f", s/2^30}' "$OUT/$NAME.parts.txt") GiB, list $list_art"
  exit 0
fi

[ "$MODE" = get ] || { echo "MODE $MODE: put or get"; exit 2; }
P=${PARTS:-$RESEARCH_RUN_DIR/inputs/parts.json}
[ -s "$P" ] || { echo "no parts.json at $P"; exit 3; }
list_art=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["list"])' "$P")
"$RB" data fetch "$list_art" --to "$S/list" >/dev/null || { echo "list fetch failed"; exit 5; }
L=$(find "$S/list" -type f | head -1)
awk -v o="${ONLY:-}" 'o == "" || substr($0, 67) ~ o' "$L" >"$OUT/got.sha256"
cut -c67- "$OUT/got.sha256" >"$S/paths"
echo "extracting $(wc -l <"$OUT/got.sha256") of $(wc -l <"$L") files${ONLY:+ (ONLY=$ONLY)} into $SW"
python3 -c 'import json,sys; [print(p["art"], p["sha256"]) for p in json.load(open(sys.argv[1]))["parts"]]' "$P" >"$S/parts"
mkdir -p "$SW"
while read -r art h; do
  d=$S/part && rm -rf "$d"
  "$RB" data fetch "$art" --to "$d" >/dev/null || { echo "$art: fetch failed" >&2; exit 5; }
  f=$(find "$d" -type f | head -1)
  [ "$(sha256sum "$f" | cut -c1-64)" = "$h" ] || { echo "$art: sha256 differs" >&2; exit 5; }
  cat "$f"; rm -rf "$d"; evict
done <"$S/parts" | (cd "$SW" && tar -xf - -T "$S/paths")
[ "${PIPESTATUS[0]}" = 0 ] && [ "${PIPESTATUS[1]}" = 0 ] || { echo "a part's fetch or tar failed"; exit 5; }
(cd "$SW" && sha256sum --quiet -c -) <"$OUT/got.sha256" || { echo "extracted files differ from the list"; exit 6; }
echo "every extracted file equals the list ($(wc -l <"$OUT/got.sha256") files)"
