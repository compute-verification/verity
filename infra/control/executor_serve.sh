#!/bin/sh
# The task executor's serve (INF-410): `executor.py serve --label` on /root/executor/queue.db, the queue drive's
# `research executor enqueue` fills (drive.sh's RESEARCH_EXECUTOR_ENQUEUE), restarted 30 s after it exits. Its slots, NAME NODE
# POOL CORES MEM_GB a line, are /data/executor/slots.txt, read at each start. No serve while /data/executor/stopped exists:
# `executor_serve.sh stop` makes it and ends the serve, `start` removes it. Every start puts the tasks the last serve had leased
# back in the queue (`requeue_leased`), so two serves on one queue run each such task twice: this lock and `stop` keep it one.
# Code: /root/executor on the container disk (infra/executor.py, and the verity checkout whose tools/research it runs); a new
# container gets executor.py from /data/executor/infra. Its runs' sender is RESEARCH_AGENT_URL from /data/executor/agent-url,
# executor@vy-control-verity, which agent watch's quiet_senders leaves unrung (drive reads each task's verdict).
# Log: /data/executor/serve.log; /data/executor/alive is touched each minute a serve runs, or the loop is stopped (cloud watch).
D=/data/executor
LOG=$D/serve.log
X=/root/executor
mkdir -p $D
case "$1" in
  stop)
    touch $D/stopped
    [ -s $D/serve.pid ] && kill "$(cat $D/serve.pid)" 2> /dev/null
    echo "$(date -u +%FT%TZ) stopped" >> "$LOG"; exit 0 ;;
  start) rm -f $D/stopped; echo "$(date -u +%FT%TZ) started" >> "$LOG"; exit 0 ;;
  status) [ -f $D/stopped ] && echo stopped || echo "serving (pid $(cat $D/serve.pid 2> /dev/null))"; tail -3 "$LOG"; exit 0 ;;
esac
exec 9> /root/executor-serve.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) another copy holds /root/executor-serve.lock" >> "$LOG"; exit 0; }
while :; do
  if [ -f $D/stopped ]; then touch $D/alive; sleep 60; continue; fi
  if [ ! -f $X/infra/executor.py ]; then
    mkdir -p $X/infra && cp $D/infra/executor.py $X/infra/ \
      || { echo "$(date -u +%FT%TZ) no $X/infra/executor.py and no copy in $D/infra" >> "$LOG"; sleep 300; continue; }
  fi
  (
    set -a; . /root/.config/verity/r2.env; set +a
    export PATH=/data/drive/bin:/root/.local/bin:$PATH RESEARCH_LANE=infra RESEARCH_NOTES=/data/steward/research-notes \
      RESEARCH_STORE=/root/drive-store PYTHONPATH=$X/verity/tools/research/src LC_ALL=C.UTF-8 \
      RESEARCH_AGENT_URL="$(cat $D/agent-url 2> /dev/null)"
    cd $X && python3.12 infra/executor.py --db queue.db serve --slots $D/slots.txt --label &
    pid=$!
    echo $pid > $D/serve.pid
    while kill -0 $pid 2> /dev/null; do touch $D/alive; sleep 60; done
    wait $pid
  ) >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) serve exited rc=$?" >> "$LOG"
  rm -f $D/serve.pid
  sleep 30
done
