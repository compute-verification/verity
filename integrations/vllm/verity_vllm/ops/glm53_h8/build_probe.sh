# The TP8 row's CPU Build alone, in a sweep dir of its own: whether this tree derives GLM-5.3's rank Programs (the Commit needs their
# GP-01 workload digests). Its exit code is the Build's. No GPU is visible to it, as in a row (`gpu_lease`), so it can run beside
# a GPU step without opening contexts on its devices.
export SWEEP_DIR=/workspace/cp/sweep-build-probe CUDA_VISIBLE_DEVICES=
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
rm -rf "${SWEEP_DIR:?}/$ROW"
stamp build-probe | tee "$O/build-probe.log"
VV row stage build "$ROW" "$ROLE" "$REPO" "$REV" --gpu-util 0.75 >> "$O/build-probe.log" 2>&1
rc=$?
d=$SWEEP_DIR/$ROW
cp "$d"/row.log "$d"/stages.txt "$d"/target_family.json "$O"/ 2>/dev/null
for f in "$d"/build/rank*/*.log; do
  [ -f "$f" ] || continue
  r=${f#"$d"/build/}
  { echo "== $r ($(cat "${f%.log}.rc" 2>/dev/null || echo no rc))"; tail -25 "$f"; } >> "$O/derive-tails.log"
done
echo "[build-probe] rc=$rc" | tee -a "$O/build-probe.log"
exit $rc
