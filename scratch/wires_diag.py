# scratch (lean/WIRED): the refusal under the twin's `wires`, caught where the shared modules raise it
import gzip, json, os, sys, traceback
sys.path.insert(0, sys.argv[1])
import verity.commitments.periods.interface as IF
import verity.commitments.periods.edges as E
import served_record_case as SR

seen = {"wires": 0}


def wrap(owner, name, label, refusal_key=None):
    inner = getattr(owner, name)

    def f(*a, **k):
        try:
            got = inner(*a, **k)
        except Exception as e:
            print(f"WIRES_CAUSE {label} raised {type(e).__name__}: {str(e)[:1500]}")
            if label == "Wire.of":
                print(f"   the wire: {json.dumps(a[-1], sort_keys=True)[:800]}")
            for fr in traceback.extract_tb(e.__traceback__)[-4:]:
                print(f"   at {fr.filename.split('/')[-1]}:{fr.lineno} {fr.name}: {fr.line}")
            sys.stdout.flush(); os._exit(0)
        if refusal_key and isinstance(got, dict) and refusal_key in got:
            print(f"WIRES_CAUSE {label} refusal: {str(got[refusal_key])[:1500]}")
            sys.stdout.flush(); os._exit(0)
        if refusal_key and isinstance(got, dict):
            ws = got.get("wires", [])
            keys = sorted({k for w in ws for k in w})
            print(f"EDGES ok: {len(ws)} wires, keys {keys}; first: {json.dumps(ws[0], sort_keys=True)[:500] if ws else None}",
                  flush=True)
        return got
    setattr(owner, name, f)


wrap(E, "edges", "edges.edges", "refusal")
wrap(IF, "check_wires", "check_wires")
inner_of = IF.Wire.of.__func__ if hasattr(IF.Wire.of, "__func__") else None
if inner_of is not None:
    def of(cls, x, _i=inner_of):
        try:
            return _i(cls, x)
        except Exception as e:
            print(f"WIRES_CAUSE Wire.of raised {type(e).__name__}: {str(e)[:1500]}\n   the wire: {json.dumps(x, sort_keys=True)[:800]}")
            sys.stdout.flush(); os._exit(0)
    IF.Wire.of = classmethod(of)
case = json.loads(gzip.open(sys.argv[2], "rt").read())
got = SR.verdicts(case)
print("WIRES_DIAG no cause caught; check:", json.dumps(got["check"], sort_keys=True)[:600], flush=True)
