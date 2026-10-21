# Sourced by the row jobs: `tree_guard ROW SW REF [first]`.  A row is built, served and judged by one code tree: its first job (with
# `first`) writes SW/ROW.tree (REF@SHA), the copies carry it, and a later job runs at that commit or at a descendant whose changes
# since are only pins and this runbook (PINS_ONLY) or a named Commit-only fix (COMMIT_FIXES): of the tree, only the Build
# (`config_record.aot_of`) and publish read a row's pins, so a moved row is re-pinned between its Match GPU legs and its Build of
# record.  Needs TREE (the job's source root).
PINS_ONLY='^integrations/vllm/manifests/(records\.json$|glm_v2_tree/|dsv4_v2_tree/)'
# Commit-only fixes a later job may also run past, each `PATH BLOB_BEFORE BLOB_AFTER`, allowed only when the row's tree holds the
# first blob and the job's the second. a371c5a7f1: the binding map's leaf unit; only the Commit calls `build_binding_map`, and the
# Match's `logical_request` is unchanged. a521e5ed60 and 822660cef7 (#3028): the Commit replays the Match capture's MoE align schedule
# (`moe_align_source.arm_replay`); the capture's values, tokens and witness digests are unchanged and it writes one more file
# (align_schedule.rank<r>.pt), so a capture taken before it is not replayed and the Match GPU legs rerun at it.  Neither the Build
# nor publish reads the code either fix changes.
COMMIT_FIXES='integrations/vllm/verity_vllm/commit/binding.py 986fe71aa64a28158d0eb79f126ad061c9a284de 308391e05e6376641d6f5c1819706cdcfce9b30e
integrations/vllm/tests/commit/test_binding_negatives.py f8e60d391a898a68ad4e80bcf23a244a7ce8ae2b 12ffd26912b4539a47b51d9c2eab8f606214964b
integrations/vllm/verity_vllm/acquire/sources/moe_align_source.py 2ba05001747aac82998eb3667f2ba859dd111f42 db602f84830a3c2626dcfe4879ff5a03434cdc69
integrations/vllm/verity_vllm/acquire/sources/oracle_witness.py ac58f49a9f2fb724924f1761ade2ef1d87002043 e2d32e68304c677d349de84c7da2e86d172614dd
integrations/vllm/verity_vllm/engine/rank_worker.py 1074b8748aead6d8549cdcedb80a9d70f1df139b afd49d6a071670fe1c7e7975eef8de0a2143ee0c
integrations/vllm/tests/acquire/test_dsv4_moe_align_source.py 76e846c696d896e6b538afb4164c320fd3d1d7e2 1bda6a56fd02eff5722f229831bec271af51372a
integrations/vllm/tests/acquire/test_tp_partial_match_oracle.py 5e636ad9ccb081520e1de8ce8cdf3132d722be79 5a13ea1f00f78e4827cfc30547481f281ff24174
integrations/vllm/README.md de929db34f42ba54c84cc0b37d5f75bc59064e7a 26410d49190dc8f96ed8bcdcaa752942545bdf6e'
commit_fix() {  # commit_fix S0 SHA PATH: PATH moved from S0 to SHA by exactly a fix COMMIT_FIXES names
  local p b0 b1
  while read -r p b0 b1; do
    [ "$p" = "$3" ] && [ "$(git -C "$TREE" rev-parse -q --verify "$1:$p")" = "$b0" ] &&
      [ "$(git -C "$TREE" rev-parse -q --verify "$2:$p")" = "$b1" ] && return 0
  done <<<"$COMMIT_FIXES"
  return 1
}
tree_guard() {
  local row=$1 sw=$2 ref=$3 first=${4:-} sha t at s0 moved
  sha=${RESEARCH_SOURCE_SHA:-$(git -C "$TREE" rev-parse HEAD)} t=$sw/$row.tree
  if [ ! -s "$t" ]; then
    [ -n "$first" ] || { echo "no $t: the row's first job (match_gpu.sh or build_record.sh) writes it"; return 2; }
    echo "$ref@$sha" >"$t" && echo "tree: $row starts at $ref@$sha"; return
  fi
  at=$(cat "$t") s0=${at##*@}
  [ "$at" = "$ref@$s0" ] || { echo "$row is at $at, not on $ref"; return 2; }
  [ "$s0" = "$sha" ] && { echo "tree: $row at $at"; return 0; }
  git -C "$TREE" merge-base --is-ancestor "$s0" "$sha" 2>/dev/null ||
    { echo "$row is at $at, and $sha does not descend from it here: a row's jobs run at one tree"; return 2; }
  moved=$(git -C "$TREE" diff --name-only "$s0" "$sha" | grep -vE "$PINS_ONLY" | while read -r p; do commit_fix "$s0" "$sha" "$p" || echo "$p"; done |
    head -5 | tr '\n' ' ')
  [ -z "$moved" ] || { echo "$row is at $at, and $sha changes code since ($moved): a row's jobs run at one tree"; return 2; }
  echo "tree: $row at $at; $sha changes only pins, this runbook and COMMIT_FIXES since"
}
