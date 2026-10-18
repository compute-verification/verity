S=de869b7d1f3f09dc5ae0fc098f99c719b1391cf1
git cat-file -t $S 2>&1; git show $S:integrations/vllm/verity_vllm/commit/committer/native_rows.cu 2>&1 | head -2
d=$(mktemp -d); git -C $d init -q; git -C $d fetch -q --depth=1 https://github.com/compute-verification/verity.git cursor/commit-speed-21f3 2>&1 | tail -2; git -C $d show FETCH_HEAD:integrations/vllm/verity_vllm/commit/committer/native_rows.cu 2>&1 | head -2; rm -rf $d
