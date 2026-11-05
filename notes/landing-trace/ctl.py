import sys, subprocess
sys.path.insert(0,'/agent/repos/verity/tools/research/src')
from research.pods import budgets
from research.pods.connect import ssh_base
b = ssh_base(budgets.GUARD_HOST)
r = subprocess.run([*b, sys.argv[1]], capture_output=True, timeout=int(sys.argv[2]) if len(sys.argv)>2 else 120)
sys.stdout.buffer.write(r.stdout); sys.stderr.buffer.write(r.stderr[-2000:])
sys.exit(r.returncode)
