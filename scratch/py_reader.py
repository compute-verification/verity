"""The Python readers on one class's wired Program: the record check's twin's Program stage (`desc_program`), proofs'
edges reader (`program_wires` -> `verity.commitments.periods.edges.edges`), and the twin's `own_rows` (form only)."""
import gzip, json, sys, time, traceback
sys.path.insert(0, ".")
import inference_only_record_reference as R
Malformed = R.reference.Malformed
prog, circ = sys.argv[1:3]
t0 = time.time()
def say(s): print(f"[{time.time() - t0:.0f} s] {s}", flush=True)
text = (gzip.open if prog.endswith(".gz") else open)(prog, "rt").read()
c = json.load(open(circ))
say(f"read {len(text)} bytes; {len(c['ids'])} templates")
def stage(name, f):
    try:
        r = f(); say(f"{name}: ACCEPTED"); return r
    except Malformed as m:
        say(f"{name}: REFUSED {m.code!r}")
    except Exception as e:
        say(f"{name}: REFUSED {type(e).__name__}: {e}"); traceback.print_exc(limit=4)
program = stage("desc_program", lambda: R.reference.desc_program(text))
if program is not None:
    say(f"walk_bound {R.reference.walk_bound(program)} vs FUEL//2 {R.reference.FUEL // 2}")
desc = json.loads(text)
root = desc["definitions"][desc["root"]]
say(f"root params: {len(root['body'].get('params', []))} e.g. {[p if isinstance(p, str) else p.get('name') for p in root['body'].get('params', [])[:4]]}")
import verity.commitments.periods.edges as E
_pw = E._part_wires
def _traced(R, fu, templates, cuts_of, k, i, q, s0):
    try:
        return _pw(R, fu, templates, cuts_of, k, i, q, s0)
    except E.Refusal as x:
        node = R.nodes[k]; port, ty = node["params"][i]
        rt = fu.get(k, (None,))[0]
        say(f"edges refusal {x} at root node {k} (template {rt}), argument {i} port {port!r}, "
            f"{_leaves(ty)} leaves, cut {cuts_of(rt, port) if rt else None}, members {node['members']}, dims {node['args'][i]['dims']}, s0 {s0}")
        say(f"  part {json.dumps(q, default=str)[:900]}")
        raise
_leaves = E._leaves
E._part_wires = _traced
w = stage("edges", lambda: R.program_wires(desc, c["ids"], c["cuts"]))
if w is not None:
    say(f"edges: {len(w)} wires")
if program is not None:
    stage("own_rows(form_only)", lambda: R.own_rows(program, True))
