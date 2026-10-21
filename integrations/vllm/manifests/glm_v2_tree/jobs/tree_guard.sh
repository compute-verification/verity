# Sourced by the row jobs: `tree_guard ROW SW REF [first]`.  A row is built, served and judged by one code tree: its first job (with
# `first`) writes SW/ROW.tree (REF@SHA), the copies carry it, and a later job runs at that commit or at a descendant whose changes
# since are only pins and this runbook (PINS_ONLY) or a named Commit-only fix (COMMIT_FIXES): of the tree, only the Build
# (`config_record.aot_of`) and publish read a row's pins, so a moved row is re-pinned between its Match GPU legs and its Build of
# record.  Needs TREE (the job's source root).
PINS_ONLY='^integrations/vllm/manifests/(records\.json$|glm_v2_tree/|dsv4_v2_tree/)'
# Commit-only fixes a later job may also run past, each `PATH BLOB_BEFORE BLOB_AFTER` (`-` for a file a tree does not hold), allowed
# only when the row's tree holds the first blob and the job's the second. a371c5a7f1: the binding map's leaf unit; only the Commit
# calls `build_binding_map`, and the Match's `logical_request` is unchanged. a521e5ed60 and 822660cef7 (#3028): the Commit replays the
# Match capture's MoE align schedule (`moe_align_source.arm_replay`); the capture's values, tokens and witness digests are unchanged
# and it writes one more file (align_schedule.rank<r>.pt), so a capture taken before it is not replayed and the Match GPU legs rerun
# at it.  c74d38bf6f, c9accad0ad, 6fd9f01777 and 18474e99c0 (#3075, #3076, #3077, #3079): replay row kernels (`kernels.dsv4_rows`)
# for the 19 families the deferred replay held at its no-evaluator gate, each equal to its Definition; the replay reads them, and the
# Commit's call-boundary evaluation (`evaluate_batch`) now takes a family's row kernel where it took the reference, the same words.
# 906df5058c (#3093): the deferred replay opens DeepSeek-V4's padded attn_sink and hc_attn_fn_broadcast by their Build roles
# (`derived_weights`), as the weights of record do; only `tp-commit-replay` reads it.  df251e69b6 (#3099): both replay readers serve
# int32 registered tensors (DeepSeek-V4's `ffn.gate.tid2eid`); only the Commit (its replay and serving rows) reads `opening`'s reader.
# Neither the Build nor publish reads the code these fixes change.
COMMIT_FIXES='integrations/vllm/verity_vllm/commit/binding.py 986fe71aa64a28158d0eb79f126ad061c9a284de 308391e05e6376641d6f5c1819706cdcfce9b30e
integrations/vllm/tests/commit/test_binding_negatives.py f8e60d391a898a68ad4e80bcf23a244a7ce8ae2b 12ffd26912b4539a47b51d9c2eab8f606214964b
integrations/vllm/verity_vllm/acquire/sources/moe_align_source.py 2ba05001747aac82998eb3667f2ba859dd111f42 db602f84830a3c2626dcfe4879ff5a03434cdc69
integrations/vllm/verity_vllm/acquire/sources/oracle_witness.py ac58f49a9f2fb724924f1761ade2ef1d87002043 e2d32e68304c677d349de84c7da2e86d172614dd
integrations/vllm/verity_vllm/engine/rank_worker.py 1074b8748aead6d8549cdcedb80a9d70f1df139b afd49d6a071670fe1c7e7975eef8de0a2143ee0c
integrations/vllm/tests/acquire/test_dsv4_moe_align_source.py 76e846c696d896e6b538afb4164c320fd3d1d7e2 1bda6a56fd02eff5722f229831bec271af51372a
integrations/vllm/tests/acquire/test_tp_partial_match_oracle.py 5e636ad9ccb081520e1de8ce8cdf3132d722be79 5a13ea1f00f78e4827cfc30547481f281ff24174
integrations/vllm/README.md de929db34f42ba54c84cc0b37d5f75bc59064e7a 26410d49190dc8f96ed8bcdcaa752942545bdf6e
integrations/vllm/verity_vllm/program/kernels/rows.py bacf38fa1c32d414c31cc6d5f2f2ef56423ee33b 35e07a2aaefdee610695956fb1749da9eac2edc3
integrations/vllm/verity_vllm/program/kernels/dsv4_rows.py - 1d983ac7d616684e5426e22359da78ac58784364
integrations/vllm/verity_vllm/program/kernels/dsv4_moe_twin.py - 75afc0d14ad266da21f413b0dbd8d944405ddfa9
integrations/vllm/verity_vllm/program/kernels/dsv4_indexer_twin.py - 8de04fed7354540fda4d8716f128ced2d274d0c2
integrations/vllm/verity_vllm/program/kernels/dsa_norm_rows.py 840d2e326b8feca9d7866f9ecf0f1f9d9ff366ab 2522ef8d49017db951c7d001403770ff98eb2033
integrations/vllm/tests/check/test_sampled_replay_dsv4_rows.py - 8201f3d3e95f21f913d633f9bd51fa06d5dba1a8
integrations/vllm/tests/program/test_dsv4_derive.py 8c48477289e67dda31ed96def6be0e2ddd705425 2929952c977afb91469f3814fa3122f34b7850ef
integrations/vllm/tests/properties/dsv4_moe_capture_gpu.py e83cf1b06ce92f10e76d253104e2f51d51cce56c fe66dd4bc8ab534ed14170629bb3004c914d131d
integrations/vllm/tests/properties/test_dsv4_moe_capture.py b9a5fbc85bc872b6cae1a6c06ecfe6281896d856 ab6f73f055acbd035206f05827cb1dd77a51a367
integrations/vllm/tests/properties/dsv4_indexer_capture_gpu.py 0ec0019327ac283b35e8879eb43b30c972759631 394ca07704910601573b709f986f3a7983373b3e
integrations/vllm/tests/properties/test_dsv4_indexer_capture.py bd27b0588ec34a9ca987fcb8072a1ad038135cee cea547bb244913a416b40092bfc958ea95930db2
integrations/vllm/verity_vllm/pipeline/cpu_replay.py e6c427b8a9cfc9d9eaedc8b2205712f5def6eb5d 43c9adc9d3ca6e913f5e202f791586a610994388
integrations/vllm/tests/pipeline/test_tp_replay_deferred.py ed4d627b719c936f45c84329ef39bc2de7ba93f0 cc76ecfb9fb3ce9edfc6d9f267f085fa54b2ab35
integrations/vllm/verity_vllm/check/replay/opening.py bea24ee88b59e77f8a828a75627aa7eae4e14c8c a68d9d4d7fd48f360e72feca8e7c5707ea1f4e49
integrations/vllm/tests/pipeline/test_cpu_replay.py 50646716d06ca4b9aee55a57826408a7e849a487 576136b47c834a6b753317d4f4e1cdb426804389
integrations/vllm/tests/check/test_sampled_replay.py 198eb6b84a2ceca9e2a4624b88d938b0a912cd8c 22a76448e97cb695d59a6ba9c78ba9ea315a9a78'
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
