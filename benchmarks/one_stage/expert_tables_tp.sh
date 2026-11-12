#!/usr/bin/env bash
# The committer's per-expert tables registration of a TP model, one file a rank (`verity-vllm tp-register-tables --served`), on a
# CPU machine that holds the checkpoint: each rank's expert stacks composed from the checkpoint of record for the rank, each
# weight held to a capture's served SHA-512 (its params.json, one a rank), every table committed under the committer's key. The
# files are what the verifier pins (records_audit.py audit --registered) and what the rank's windows commit their tables under.
#   TABLES_CHECKPOINT=DIR the checkpoint, TABLES_MODEL and TABLES_REVISION its checkpoints.json entry, which pins its shards (rank
#     0 hashes every one); TABLES_SERVED=DIR holds params-rank<r>.json; TABLES_WORLD the TP world; TABLES_KEY=FILE the committer's
#     32-byte key, minted here when absent (expert_tables.mint_key: mode 0600, with its sidecar FILE.nonces, without which every
#     rank refuses the key), never under $OUT; TABLES_STACKS=S,.. only those expert stacks (default every
#     one); TABLES_BLOCKS the projections' block rows (default "w13=1 w2=32"); TABLES_WORKERS processes a rank (default 2).
#   Out: $OUT/tables-rank<r>.json and its log, every rank at once.
set -euo pipefail
: "${TABLES_CHECKPOINT:?}" "${TABLES_MODEL:?}" "${TABLES_REVISION:?}" "${TABLES_SERVED:?}" "${TABLES_WORLD:?}" "${TABLES_KEY:?}" "${OUT:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$(realpath -m "$TABLES_KEY")" in "$(realpath -m "$OUT")"/*) echo "TABLES_KEY is under \$OUT: the key never leaves the committer" >&2; exit 2;; esac
[ -e "$TABLES_KEY" ] || "$REPO/.venv/bin/python" -c 'import sys; from verity_vllm.pipeline.expert_tables import mint_key; mint_key(sys.argv[1])' "$TABLES_KEY"
blocks=()
for b in ${TABLES_BLOCKS:-w13=1 w2=32}; do blocks+=(--block "$b"); done
stacks=()
[ -z "${TABLES_STACKS:-}" ] || stacks=(--stacks "$TABLES_STACKS")
mkdir -p "$OUT"
pids=()
for ((r = 0; r < TABLES_WORLD; r++)); do
  verify=(--no-verify-shards)
  [ "$r" = 0 ] && verify=()
  "$REPO/.venv/bin/verity-vllm" tp-register-tables --served "$TABLES_SERVED/params-rank$r.json" --world "$TABLES_WORLD" --rank "$r" \
    --checkpoint-dir "$TABLES_CHECKPOINT" --checkpoints-manifest "$REPO/adapters/vllm/manifests/checkpoints.json" \
    --model "$TABLES_MODEL" --revision "$TABLES_REVISION" --key "$TABLES_KEY" "${blocks[@]}" "${stacks[@]}" \
    --workers "${TABLES_WORKERS:-2}" --out "$OUT/tables-rank$r.json" "${verify[@]}" > "$OUT/tables-rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
tail -n 1 "$OUT"/tables-rank*.log
(cd "$OUT" && sha256sum tables-rank*.json)
exit "$fail"
