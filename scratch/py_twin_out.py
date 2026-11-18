# scratch (lean/WIRED): the twin with `_out_rows` keeping an `out` row its layout states (a probe of how far the twin gets
# past the `out`-is-0-words refusal, not a cure), the refusal under `wires` caught where it is raised
import gzip, json, os, sys, time
sys.path.insert(0, sys.argv[1])
import verity.commitments.periods.interface as IF
import served_record_case as SR
inner_out = IF._out_rows
IF._out_rows = lambda lay: inner_out(lay) | {k: n for k, n in lay.outs if k == "out"}
inner_cw = IF.check_wires
def cw(*a, **k):
    try:
        return inner_cw(*a, **k)
    except Exception as e:
        print(f"WIRES_CAUSE check_wires raised {type(e).__name__}: {str(e)[:1500]}", flush=True); raise
IF.check_wires = cw
t0 = time.time()
got = SR.verdicts(json.loads(gzip.open(sys.argv[2], "rt").read()))
print("PY_TWIN_OUT check:", json.dumps(got["check"], sort_keys=True)[:900], flush=True)
print("PY_TWIN_OUT count:", json.dumps(got["count"], sort_keys=True)[:300], f"in {time.time() - t0:.0f} s", flush=True)
