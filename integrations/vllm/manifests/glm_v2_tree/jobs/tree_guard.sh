# Sourced by the row jobs: `tree_guard ROW SW REF [first]`.  A row is built, served and judged by one code tree: its first job (with
# `first`) writes SW/ROW.tree (REF@SHA), the copies carry it, and a later job runs at that commit or at a descendant whose changes
# since are only pins and this runbook (PINS_ONLY) or a named Commit-only fix (COMMIT_FIXES): of the tree, only the Build
# (`config_record.aot_of`) and publish read a row's pins, so a moved row is re-pinned between its Match GPU legs and its Build of
# record.  Needs TREE (the job's source root).
PINS_ONLY='^integrations/vllm/manifests/(records\.json$|glm_v2_tree/|dsv4_v2_tree/)'
# Commit-only fixes a later job may also run past, each `PATH BLOB_BEFORE BLOB_AFTER`, allowed only when the row's tree holds the
# first blob and the job's the second. a371c5a7f1: the binding map's leaf unit; only the Commit calls `build_binding_map`, and the
# Match's `logical_request` is unchanged. 190ee66678: each request's call-boundary spelling in its own Program order; only the
# Commit builds a `Binder`.
COMMIT_FIXES='integrations/vllm/verity_vllm/commit/binding.py 986fe71aa64a28158d0eb79f126ad061c9a284de 308391e05e6376641d6f5c1819706cdcfce9b30e
integrations/vllm/tests/commit/test_binding_negatives.py f8e60d391a898a68ad4e80bcf23a244a7ce8ae2b 12ffd26912b4539a47b51d9c2eab8f606214964b
integrations/vllm/verity_vllm/query/manifest/bind.py b9d3ec27ec5b7659948f05e246da4b187bdafddd 4797d292d6c7d142f3814230dda1cdb4ceb02320
integrations/vllm/tests/query/test_bind.py 5e61154f8cafa41e702cee9b1ee78141898a9354 8f51c8e7dd88393ae8d4d5410251d0e282617ee4'
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
