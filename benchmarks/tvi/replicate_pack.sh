# On the node that has the venv: pack /workspace/jobs/tvi/venv, its interpreter and the templates into the evidence store.
# Writes $OUT/put.json (the art: id); replicate_unpack.sh ART on the other node puts them back at the same paths.
set -euo pipefail
S=$(mktemp -d -p /workspace/jobs/tvi .pack.XXXXXX)
trap 'rm -rf "$S"' EXIT
test -e /workspace/jobs/tvi/venv/READY || { echo "no READY: the venv isn't final"; exit 3; }
cd /
time tar -I "zstd -T0 -3" --exclude='__pycache__' --exclude='workspace/jobs/tvi/venv/READY' -cf "$S/tvi-venv.tar.zst" \
  workspace/jobs/tvi/venv workspace/jobs/tvi/templates workspace/jobs/cache/python/cpython-3.12-linux-x86_64-gnu
ls -l "$S/tvi-venv.tar.zst"
sha256sum "$S/tvi-venv.tar.zst" | tee "$OUT/sha256.txt"
time research data put --kind tvi-venv/v1 --file "$S/tvi-venv.tar.zst" --preserve --json \
  --meta '{"what":"training-vs-inference shared venv /workspace/jobs/tvi/venv, its python and templates","unpack":"tar -I zstd -xf FILE -C /"}' | tee "$OUT/put.json"
