#!/usr/bin/env bash
# Pod-side dead-man's switch (integrator, 2026-09-12; D34 recipe, saved after the night program).
#
#   zk/campaign/deadman.sh <tag> <own-prefix> <cap-usd> <deadline-utc-ISO> [pod-max-hours]   e.g.  2026-09-13-curve cv- 10 2026-09-13T23:00:00Z 4
#
# [2026-09-13, D55] The controller reads its cap from /root/dm/CAP at every poll (phase gates: raise it with
#   ssh ... 'echo 30 > /root/dm/CAP'), terminates any owned pod older than <pod-max-hours> (default 4), and ships pod_guard.sh so job pods
# can copy it from here.  Verify it is alive from anywhere: ssh ... 'tail -2 /root/.runpod/budget-<tag>-podside.log'.
#
# Creates the smallest CPU pod `<prefix>int-deadman`, copies budget_cap.py + runpod.py + the RunPod config there, and starts two
# laptop-independent loops under nohup: (1) budget_cap.py run --own-prefix <prefix> (counts + terminates only <prefix>* pods at the cap);
# (2) a hard deadline that terminates every <prefix>* pod at <deadline> (itself last).  Prints the pod id.  Nothing here depends on the
# laptop after it returns.
set -euo pipefail
TAG=${1:?tag}; PREFIX=${2:?own-prefix}; CAP=${3:?cap}; DEADLINE=${4:?deadline ISO utc}; PODMAX=${5:-4}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); PY="$ROOT/.venv/bin/python"
NAME="${PREFIX}int-deadman"
POD_JSON=$("$PY" "$HERE/runpod.py" create --name "$NAME" --cpu cpu3c --vcpu 1 --disk 10 ${DEADMAN_DC:+--data-center "$DEADMAN_DC"})
POD_ID=$(printf '%s' "$POD_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "deadman pod $POD_ID ($NAME)"
SSH_LINE=$("$PY" "$HERE/runpod.py" wait-ssh "$POD_ID" | tail -1)   # prints: ssh ... -p <port> ... root@<host>
PORT=$(printf '%s\n' "$SSH_LINE" | sed -E 's/.* -p ([0-9]+) .*/\1/'); HOST=$(printf '%s\n' "$SSH_LINE" | sed -E 's/.*root@([^ ]+).*/\1/')
echo "ssh root@$HOST -p $PORT"
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -p "$PORT" "root@$HOST")
SCP=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$HOME/.runpod/ssh/runpodctl-ssh-key" -P "$PORT")
"${SSH[@]}" 'mkdir -p /root/.runpod /root/dm'
"${SCP[@]}" "$HERE/budget_cap.py" "$HERE/runpod.py" "$HERE/pod_guard.sh" "root@$HOST:/root/dm/"
"${SSH[@]}" "echo $CAP > /root/dm/CAP"
"${SCP[@]}" "$HOME/.runpod/config.toml" "root@$HOST:/root/.runpod/config.toml"
DEADLINE_EPOCH=$(date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$DEADLINE" +%s 2>/dev/null || date -u -d "$DEADLINE" +%s)
"${SSH[@]}" "cat > /root/dm/deadline.sh <<'EOF'
#!/usr/bin/env bash
# terminate every owned pod at the deadline, self last
PREFIX=\$1; DL=\$2; ME=\$3
while [ \$(date +%s) -lt \$DL ]; do sleep 30; done
echo \"\$(date -u +%FT%TZ) DEADLINE reached\" >> /root/dm/deadline.log
python3 - \"\$PREFIX\" \"\$ME\" <<'PY' >> /root/dm/deadline.log 2>&1
import sys; sys.path.insert(0, '/root/dm'); import runpod
prefix, me = sys.argv[1], sys.argv[2]
pods = runpod._request('GET', '/pods') or []
mine = [p for p in pods if str(p.get('name','')).startswith(prefix) and (p.get('desiredStatus') or '').upper() not in ('EXITED','TERMINATED')]
for p in sorted(mine, key=lambda p: p['id'] == me):
    try:
        runpod._request('DELETE', f\"/pods/{p['id']}\"); print('TERMINATED', p['id'], p.get('name'))
    except SystemExit as e:
        print('terminate failed', p['id'], e)
PY
EOF
chmod +x /root/dm/deadline.sh
cd /root/dm && (setsid nohup python3 budget_cap.py run --tag ${TAG}-podside --cap $CAP --cap-file /root/dm/CAP --pod-max-hours $PODMAX --rate-max 30 --own-prefix $PREFIX --allow-multiple > /root/dm/budget.out 2>&1 &)
cd /root/dm && (setsid nohup /root/dm/deadline.sh $PREFIX $DEADLINE_EPOCH $POD_ID > /root/dm/deadline.out 2>&1 &)
sleep 3; pgrep -fl 'budget_cap.py run|deadline.sh' | cut -c1-120; tail -2 /root/.runpod/budget-${TAG}-podside.log"
echo "$POD_ID" > "$HOME/.runpod/deadman-${TAG}.pod"
echo "ssh root@$HOST -p $PORT" > "$HOME/.runpod/deadman-${TAG}.ssh"
echo "armed: cap \$$CAP (file /root/dm/CAP), pod-max ${PODMAX} h, deadline $DEADLINE, owns ${PREFIX}*  (pod $POD_ID; ssh line in ~/.runpod/deadman-${TAG}.ssh)"
