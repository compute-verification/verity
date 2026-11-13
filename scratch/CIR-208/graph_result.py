"""graph_result.py FILE: the status, detail and undeclared reads of a `verity graph run` result."""
import json, sys
try:
    r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
    print({k: r.get(k) for k in ("status", "detail", "undeclared_reads")})
except Exception as e:
    print("no result:", type(e).__name__, e)
