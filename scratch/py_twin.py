# scratch (lean/WIRED): the Python twin's verdicts on one case
import gzip, json, sys, time
sys.path.insert(0, sys.argv[1])
import served_record_case as SR
t0 = time.time()
got = SR.verdicts(json.loads(gzip.open(sys.argv[2], "rt").read()))
print("PY_TWIN check:", json.dumps(got["check"], sort_keys=True)[:900], flush=True)
print("PY_TWIN count:", json.dumps(got["count"], sort_keys=True)[:300], f"in {time.time() - t0:.0f} s", flush=True)
