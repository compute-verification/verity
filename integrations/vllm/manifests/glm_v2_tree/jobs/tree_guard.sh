# Sourced by the row jobs: `tree_guard ROW SW REF [first]`.  A row is built, served and judged by one code tree: its first job (with
# `first`) writes SW/ROW.tree (REF@SHA), the copies carry it, and a later job runs at that commit or at a descendant whose changes
# since are only pins and this runbook (PINS_ONLY) or a named Commit-only fix (COMMIT_FIXES): of the tree, only the Build
# (`config_record.aot_of`) and publish read a row's pins, so a moved row is re-pinned between its Match GPU legs and its Build of
# record.  Needs TREE (the job's source root).
PINS_ONLY='^integrations/vllm/manifests/(records\.json$|glm_v2_tree/|dsv4_v2_tree/)'
# Commit-only fixes a later job may also run past, each `PATH BLOB_BEFORE BLOB_AFTER` (`-` for a file a tree does not hold), allowed
# only when the row's tree holds the first blob and the job's the second. a371c5a7f1: the binding map's leaf unit; only the Commit
# calls `build_binding_map`, and the Match's `logical_request` is unchanged. 190ee66678: each request's call-boundary spelling in its
# own Program order; only the Commit builds a `Binder`. fb04c6ead0 (#3118): the replay refuses a Definition the Build of record did
# not bind (`check.replay.definitions`); only the Commit's replay and `tp-commit-replay` read it, neither the Build nor publish.
# dac6fa0e36 (#3130): before grading, that replay self-checks each graded family's row kernel against its Definition
# (`definitions.kernel_refusal`, `kernel_registry.check`); `check` is new, and no other function of `kernel_registry` changes.
# 38126ff6d0 (#3163): the replay's row kernel for LlBf16SplitKGemm_v1 (`glm53_rows.ll_splitk_gemm`) and its provenance; only the
# Commit's replay grades by the row kernels, and the Match's one reader of the evaluators (`form_b_families`) took every step already.
# 9096c004bf and fede99cc68 (#3897): the TP fold Match folds each rank under its Build's construction (`match.fold_construction_args`)
# and reuses a rank's legs only over a kept fold; they change how the Match's CPU phase judges the legs, not the capture, the
# Build or publish, and a Match run past them refolds every rank its tree's driver folded without the construction.
COMMIT_FIXES='integrations/vllm/verity_vllm/commit/binding.py 986fe71aa64a28158d0eb79f126ad061c9a284de 308391e05e6376641d6f5c1819706cdcfce9b30e
integrations/vllm/tests/commit/test_binding_negatives.py f8e60d391a898a68ad4e80bcf23a244a7ce8ae2b 12ffd26912b4539a47b51d9c2eab8f606214964b
integrations/vllm/verity_vllm/query/manifest/bind.py b9d3ec27ec5b7659948f05e246da4b187bdafddd 4797d292d6c7d142f3814230dda1cdb4ceb02320
integrations/vllm/tests/query/test_bind.py 5e61154f8cafa41e702cee9b1ee78141898a9354 8f51c8e7dd88393ae8d4d5410251d0e282617ee4
integrations/vllm/tests/check/test_replay_definitions.py - 5ac6073ea2c71562d7b78b97003ebea309c4d250
integrations/vllm/verity_vllm/check/replay/definitions.py - 60c2c5431b7d0f2f2640770bf6acaa02c63907d7
integrations/vllm/verity_vllm/pipeline/c2_replay.py 465d70b4737ac412af1faca32a5a415437f88464 9054b8110e943d268cdf8363fdadd6fe0ba58cea
integrations/vllm/verity_vllm/pipeline/cpu_replay.py e6c427b8a9cfc9d9eaedc8b2205712f5def6eb5d 49ea998c5e309c5b693dac214c4e971d93d028f4
integrations/vllm/verity_vllm/pipeline/deferred_rows.py 0813fc8d218f2a38c72cb25f2a23499626b8d727 fa2df7ad0ef275b5bb69479dd7f5c2f5a1469b29
integrations/vllm/verity_vllm/pipeline/replay_plan.py fb940d0a601d33f38c91f48bdc5ce6aa2963cd7d af48e07500db28e53a0c71cf1336d211d9a282f7
integrations/vllm/verity_vllm/pipeline/tp/rank_replay.py 0d0dd867c71745b446e5d5de2a0b05895cde1d86 dfd9d14bc94f95f8fed97109dced1d67cf2fb4cb
integrations/vllm/verity_vllm/pipeline/tp/serving_rows.py 28253606f628a40d983847178b1c90baa3041bf6 ee0f6a80db17fe58af54fe0cfe1671780a6a0439
integrations/vllm/tests/check/test_replay_definitions.py - 9a27de6c6431bdbd095370b7eedc9b75d14b8066
integrations/vllm/tests/check/test_replay_definitions.py 5ac6073ea2c71562d7b78b97003ebea309c4d250 9a27de6c6431bdbd095370b7eedc9b75d14b8066
integrations/vllm/verity_vllm/check/replay/definitions.py - eaaf3b5433581e410f301cec059d9079d38878b8
integrations/vllm/verity_vllm/check/replay/definitions.py 60c2c5431b7d0f2f2640770bf6acaa02c63907d7 eaaf3b5433581e410f301cec059d9079d38878b8
integrations/vllm/verity_vllm/program/kernels/kernel_registry.py 483bb39df5eae54507e51b087c3aa49425ce813e dab120fb8d11b8a5b50a96ce35e01ae7091facfc
integrations/vllm/tests/program/test_kernel_self_check.py 9ea1481e226f26b17058ed848fbdff97244e5f55 7343c33b1362a3b74258a381dcf7623ff060ba45
integrations/vllm/verity_vllm/program/kernels/glm53_rows.py 943628786f28df845255e0b40161cfac6213041a a79bf320be98ac9e9c1371a4fb33ddd3cfe7a248
integrations/vllm/verity_vllm/program/kernels/rows.py bacf38fa1c32d414c31cc6d5f2f2ef56423ee33b 872ec007c254799ee031c303b0662319a4012527
integrations/vllm/tests/check/test_sampled_replay_glm53_rows.py ef7dccd4633f1055255cdfac200b3362daf242f0 96a5932b0de3e1ad417d8ec2aaccfd229a5d3b1e
integrations/vllm/verity_vllm/pipeline/match.py 6111a51feb24444c6153a3ebe175d5dd84a9e73c a842a38993f6d1b9a28513c3b0473d73cb49b877
integrations/vllm/verity_vllm/pipeline/tp/fold_match.py 75f6f7d34dc3fa1dc339db98f24710c3ddeb23bd c47b17a5cd8be94f986c290086672323ba704419
integrations/vllm/tests/check/test_tp2_d90_fixture.py 277982ec444e46046a20bf1b61cf767d3fb2ec9a e8225dfdcc77d68469ea043f08eacc5a485582da'
_blob() {  # _blob SHA PATH: PATH's blob at SHA, `-` when SHA holds no PATH
  git -C "$TREE" rev-parse -q --verify "$1:$2" || echo -
}
commit_fix() {  # commit_fix S0 SHA PATH: PATH moved from S0 to SHA by exactly a fix COMMIT_FIXES names
  local p b0 b1
  while read -r p b0 b1; do
    [ "$p" = "$3" ] && [ "$(_blob "$1" "$p")" = "$b0" ] && [ "$(_blob "$2" "$p")" = "$b1" ] && return 0
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
