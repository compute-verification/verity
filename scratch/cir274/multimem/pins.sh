set -u
echo "== source $(git rev-parse HEAD)"
echo "== stale before"; python3 tools/verity/check/pins.py --stale; echo "stale rc=$?"
echo "== regen"; python3 tools/verity/check/pins.py --regen 2>&1 | tail -30; echo "regen rc=${PIPESTATUS[0]}"
echo "== status after"; git status --porcelain | tee "$OUT/status.txt"
git diff --binary > "$OUT/pins.diff"; wc -c "$OUT/pins.diff"
git ls-files --others --exclude-standard -z | xargs -0 -r tar czf "$OUT/untracked.tgz"
echo "== stale after"; python3 tools/verity/check/pins.py --stale; echo "stale-after rc=$?"
