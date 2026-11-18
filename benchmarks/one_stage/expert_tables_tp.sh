#!/usr/bin/env bash
# The committer's per-expert tables registration of a TP model, one file a rank (`verity-vllm tp-register-tables --served`), on a
# CPU machine that holds the checkpoint: each rank's expert stacks composed from the checkpoint of record for the rank, each
# weight held to a capture's served SHA-512 (its params.json, one a rank), every table committed under fresh salts. The files are
# what the verifier pins (records_audit.py audit --registered) and what the rank's windows commit their tables under.
#   TABLES_CHECKPOINT=DIR the checkpoint, TABLES_MODEL and TABLES_REVISION its checkpoints.json entry, which pins its shards (rank
#     0 hashes every one); TABLES_SERVED=DIR holds params-rank<r>.json; TABLES_WORLD the TP world; TABLES_PRIVATE=DIR the committer's
#     registrations with their salt files beside them (tables-rank<r>.salts/), never under $OUT; TABLES_STACKS=S,.. only those expert stacks (default every
#     one); TABLES_BLOCKS the projections' block rows (default "w13=1 w2=32"); TABLES_WORKERS processes a rank (default 2).
#   Out: $OUT/tables-rank<r>.json (a copy of the registration, without its salts) and its log, every rank at once.
set -euo pipefail
: "${TABLES_CHECKPOINT:?}" "${TABLES_MODEL:?}" "${TABLES_REVISION:?}" "${TABLES_SERVED:?}" "${TABLES_WORLD:?}" "${TABLES_PRIVATE:?}" "${OUT:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$(realpath -m "$TABLES_PRIVATE")/" in "$(realpath -m "$OUT")"/*) echo "TABLES_PRIVATE is under \$OUT: the salts never leave the committer" >&2; exit 2;; esac
blocks=()
for b in ${TABLES_BLOCKS:-w13=1 w2=32}; do blocks+=(--block "$b"); done
stacks=()
[ -z "${TABLES_STACKS:-}" ] || stacks=(--stacks "$TABLES_STACKS")
mkdir -p "$OUT" "$TABLES_PRIVATE"
pids=()
for ((r = 0; r < TABLES_WORLD; r++)); do
  verify=(--no-verify-shards)
  [ "$r" = 0 ] && verify=()
  "$REPO/.venv/bin/verity-vllm" tp-register-tables --served "$TABLES_SERVED/params-rank$r.json" --world "$TABLES_WORLD" --rank "$r" \
    --checkpoint-dir "$TABLES_CHECKPOINT" --checkpoints-manifest "$REPO/adapters/vllm/manifests/checkpoints.json" \
    --model "$TABLES_MODEL" --revision "$TABLES_REVISION" "${blocks[@]}" "${stacks[@]}" \
    --workers "${TABLES_WORKERS:-2}" --out "$TABLES_PRIVATE/tables-rank$r.json" "${verify[@]}" > "$OUT/tables-rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for ((r = 0; r < TABLES_WORLD; r++)); do
  [ ! -f "$TABLES_PRIVATE/tables-rank$r.json" ] || cp "$TABLES_PRIVATE/tables-rank$r.json" "$OUT/"
done
tail -n 1 "$OUT"/tables-rank*.log
(cd "$OUT" && sha256sum tables-rank*.json)
exit "$fail"
