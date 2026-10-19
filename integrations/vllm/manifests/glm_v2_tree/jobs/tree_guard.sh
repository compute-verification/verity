# Sourced by the row jobs: `tree_guard ROW SW REF [first]`.  A row is built, served and judged by one code tree: its first job (with
# `first`) writes SW/ROW.tree (REF@SHA), the copies carry it, and a later job runs at that commit or at a descendant whose changes
# since are only pins and this runbook (PINS_ONLY): of the tree, only the Build (`config_record.aot_of`) and publish read a row's
# pins, so a moved row is re-pinned between its Match GPU legs and its Build of record.  Needs TREE (the job's source root).
PINS_ONLY='^integrations/vllm/manifests/(records\.json$|glm_v2_tree/)'
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
  moved=$(git -C "$TREE" diff --name-only "$s0" "$sha" | grep -vE "$PINS_ONLY" | head -5 | tr '\n' ' ')
  [ -z "$moved" ] || { echo "$row is at $at, and $sha changes code since ($moved): a row's jobs run at one tree"; return 2; }
  echo "tree: $row at $at; $sha changes only pins and this runbook since"
}
