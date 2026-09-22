#!/bin/bash
# Pull one case x profile-tag bundle's JSON off the pod into fixtures/results/<experiment_id>/ (contract §9).
# Tensors never enter git: the capture (ru/, weights/) stays under /workspace/results on the pod; its
# sha256s, sizes and pod paths are recorded in bundle.json. Files > 5 MB are skipped by rsync and listed
# with sha + pod path instead (§9: "description.json (or its sha and pod path if > 5 MB)").
# usage: collect.sh <CASE> <TAG> [results_root]   (runs on the laptop/VM; needs /tmp/podssh.sh + rsync)
#   VERIFY_DIR=verify-security  selects the verifier output subdir (run_case.sh writes verify-<challenge mode>)
set -eu
CASE=$1; TAG=$2; ROOT=${3:-fixtures/results}
SSH=${PODSSH:-/tmp/podssh.sh}
HOST=${POD_HOST:-root@213.181.111.2}; PORT=${POD_PORT:-48289}; KEY=${POD_KEY:-$HOME/.ssh/vllm-poc-key}
R=/workspace/results/$CASE/$TAG
VDIR=${VERIFY_DIR:-verify}   # an alternative verifier output dir of the same commit (W10: a re-verify keeps the first one aside)

# 1. pod side: experiment id + inventory of the bundle (shas, sizes, paths), written next to it
$SSH "cd $R && ${POD_PY:-/workspace/venv/bin/python} - <<'EOF'
import hashlib, json, os, sys
from pathlib import Path
R = Path('.')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()
def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())
v = json.loads((R / '$VDIR' / 'verify.json').read_text())
inv = {'experiment_id': v['experiment_id'], 'case': v['case'], 'profile_id': v['profile_id'], 'seed': v.get('seed'),
       'challenge': (v.get('challenge') or {}).get('mode', 'diagnostic'), 'verify_dir': '$VDIR',
       'pod': os.uname().nodename, 'bundle_dir': str((R / 'commit').resolve()), 'files': {}, 'dirs': {}}
for name in ['commit/manifest.json', 'commit/description.json', 'commit/required.json', 'commit/commitments.json',
             'commit/replay.json', 'ref1/replay.json', 'ref2/replay.json', 'compare.json', 'production/manifest.json',
             'production/production.jsonl', 'production/timings.json']:
    p = R / name
    if p.exists():
        inv['files'][name] = {'sha256': sha(p), 'bytes': p.stat().st_size, 'pod_path': str(p.resolve())}
for d in ['commit/ru', 'commit/weights', 'ref1/ru', 'ref2/ru', '$VDIR/negatives']:
    p = R / d
    if p.exists():
        inv['dirs'][d] = {'bytes': du(p), 'files': sum(1 for f in p.rglob('*') if f.is_file()), 'pod_path': str(p.resolve())}
(R / 'bundle.json').write_text(json.dumps(inv, indent=1) + '\n')
print(v['experiment_id'])
EOF" > /tmp/collect_exp_id.txt
EXP=$(tail -1 /tmp/collect_exp_id.txt | tr -d '\r')
[ -n "$EXP" ] || { echo "no experiment id"; exit 1; }
DEST=$ROOT/$EXP
mkdir -p "$DEST"
echo "collecting $CASE/$TAG -> $DEST"

RS="rsync -az --max-size=5m -e \"ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR\""
# 2. the verifier's outputs (tables, challenge, verify.json, negative reproducers) -> experiment root
eval $RS "$HOST:$R/$VDIR/" "$DEST/"
# 3. bundle JSON (tensors excluded), reference-run reports, production record
eval $RS --include='*.json' --include='*.log' --exclude='*' "$HOST:$R/commit/" "$DEST/bundle/"
for m in ref1 ref2; do eval $RS --include='*.json' --include='*.log' --exclude='*' "$HOST:$R/$m/" "$DEST/$m/"; done
eval $RS "$HOST:$R/production/" "$DEST/production/"
eval $RS "$HOST:$R/compare.json" "$HOST:$R/bundle.json" "$HOST:$R/pipeline.log" "$DEST/"
# 4. §9 names at the experiment root: manifest.json, description.json, commitments.json, log.txt
cp "$DEST/bundle/manifest.json" "$DEST/manifest.json"
[ -f "$DEST/bundle/description.json" ] && cp "$DEST/bundle/description.json" "$DEST/description.json"
[ -f "$DEST/bundle/commitments.json" ] && cp "$DEST/bundle/commitments.json" "$DEST/commitments.json"
{
  echo "# $EXP  ($CASE/$TAG)"; echo; cat "$DEST/pipeline.log"; echo
  # VERIFY_LOG (W10): the verifier log's name when it is not verify (e.g. a re-verify's own log)
  VLOG=${VERIFY_LOG:-verify}
  for f in production ref1 ref2 commit compare $VLOG $([ "$VDIR" != verify ] && [ "$VDIR" != "$VLOG" ] && echo "$VDIR"); do
    echo "## /workspace/logs/$CASE-$TAG-$f.log (tail)"
    $SSH "tail -40 /workspace/logs/$CASE-$TAG-$f.log 2>/dev/null | cut -c1-400" 2>/dev/null | grep -v setlocale || true
    echo
  done
} > "$DEST/log.txt"
find "$DEST" -type f -size +5M -print -delete
du -sh "$DEST"; echo "$EXP"
