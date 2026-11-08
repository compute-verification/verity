for run in "$@"; do
  O=/workspace/research/runs/$run/out
  [ -d "$O" ] || continue
  echo "@@RUN $run head $(cat $O/head.txt 2>/dev/null)"
  grep -E "row |bootstrap rc|^vllm |metadata rc|libcuda|neutrality (start|rc)|NEUTRAL|MOVED|^ +rank|records.json" $O/summary.txt | cut -c1-220
  python3 - "$O/neutrality.json" <<'P'
import json,sys
j=json.load(open(sys.argv[1]))
for k,v in j["rows"].items():
    ranks=v["programs"]
    print("VERDICT", k, "build_rc", v["build_rc"], "neutral", v["neutral"], "differences", v["differences"], "ranks", len(ranks),
          "requests/rank", sorted({len(p["requests"]) for p in ranks.values()}),
          "manifest", v["manifest_digest"], "pinned", v["manifest_digest_pinned"])
P
  git -C /workspace/research/scratch/$run/tree diff --stat -- integrations/vllm/manifests/records.json 2>/dev/null | tail -1
  echo "records.json sha $(sha256sum $O/records.json | cut -c1-16)"
done
