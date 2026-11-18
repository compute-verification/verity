# scratch (lean/WIRED): the exception under the twin's `wires` refusal
import gzip, json, sys, traceback
sys.path.insert(0, sys.argv[1])
import inference_only_record_reference as R
import served_record_case as SR

inner = R.wires_of


def wires_of(*a, **k):
    try:
        return inner(*a, **k)
    except R.Malformed as e:
        c = e.__cause__
        print(f"WIRES_CAUSE {type(c).__name__}: {str(c)[:1500]}")
        for fr in traceback.extract_tb(c.__traceback__)[-4:]:
            print(f"   at {fr.filename.split('/')[-1]}:{fr.lineno} {fr.name}: {fr.line}")
        sys.stdout.flush()
        raise


R.wires_of = wires_of
case = json.loads(gzip.open(sys.argv[2], "rt").read())
got = SR.verdicts(case)
print("WIRES_DIAG check:", json.dumps(got["check"], sort_keys=True)[:600], flush=True)
