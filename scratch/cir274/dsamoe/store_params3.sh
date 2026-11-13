R="uv run -q --project tools/research research data sql"
for k in vllm-rowdir-tar/v1 vllm-replay-bundle/v1 vllm-rowdir-list/v1 vllm-rowdir-parts/v1 vllm-row-meta/v1 glm53/tp8-match-record/v1 vllm-match-record/v1 vllm-build/v1 spot-card/v1; do
  echo "--- $k"; $R "select id, length(manifest_json), substr(manifest_json,1,700) from artifacts where kind='$k' limit 3" 2>&1 | tail -4 | cut -c1-900
  $R "select count(*) from artifacts where kind='$k' and (manifest_json like '%params%' or manifest_json like '%step_%')" 2>&1 | tail -1
done
