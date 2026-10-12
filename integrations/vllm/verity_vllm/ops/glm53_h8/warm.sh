# Every shard read once from the shared filesystem: hashed against the pin, which leaves the 755.7 GB in the page cache for the
# engine loads that follow.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
P=${WARM_JOBS:-48}
{
stamp warm
t0=$(date +%s)
(cd "$CK" && ls *.safetensors | xargs -P "$P" -n 1 sha256sum) > "$O/shards.sha256"
rc=$?
t1=$(date +%s)
echo "[warm] sha256sum rc=$rc wall=$((t1 - t0))s jobs=$P"
"$PY" - "$O/shards.sha256" <<'EOF'
import json, sys
got = {p.strip(): h for h, p in (ln.split(None, 1) for ln in open(sys.argv[1]))}
pin = next(c for c in json.load(open("manifests/checkpoints.json"))["checkpoints"] if c.get("role") == "GLM53")["safetensors_sha256"]
bad = sorted(f for f in pin if got.get(f) != pin[f])
print(f"SHARDS {'OK' if not bad and len(got) == len(pin) == 141 else 'FAIL'} hashed={len(got)} pinned={len(pin)} mismatched={bad[:10]}")
sys.exit(1 if bad or len(got) != len(pin) else 0)
EOF
r=$?
stamp warm-end
free -g | head -2
exit $r
} 2>&1 | tee "$O/warm.log"
exit "${PIPESTATUS[0]}"
