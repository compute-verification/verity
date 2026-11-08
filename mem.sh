free -g | head -2; cat /sys/fs/cgroup/user.slice/user-1001.slice/memory.max 2>/dev/null
for f in /workspace/ramlock/*.json; do python3 -c "
import json,sys;j=json.load(open('$f'));print('$f'.split('/')[-1], {k:j[k] for k in j if k in ('gb','mem_gb','run','run_id','pid','owner','since','at_utc')})" 2>/dev/null; done
