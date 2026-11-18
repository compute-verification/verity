import importlib.util, json, copy, traceback
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v = load("record_difftest_vectors"); ref = v.ref
from verity.circuits.types import Value, Tuple, Array
MIX = Tuple(Value(16), Value(8), Value(8))
_lo = v.leaves_of
def leaves_of(c):
    if isinstance(c.type, Tuple):
        return [x for i in range(len(c.type.fields)) for x in leaves_of(c[i])]
    return [c] if isinstance(c.type, Value) else [x for i in range(c.type.n) for x in leaves_of(c[i])]
v.leaves_of = leaves_of
def onto2(name):
    W = v.template(f"{name}Writer", [("a", Value(16))], [MIX, Value(16)],
                   [[("op", 16, [(0, 0)]), ("op", 8, [(0, 0)]), ("op", 8, [(0, 0)])], [("op", 16, [(0, 0)])]])
    R = v.template(f"{name}Reader", [("v", MIX)], [Value(16)], [[("op", 16, [(0, 0)])]])
    def body(B, S, a, z):
        B.batch(W, a, axes=(0,))
        return B.call(R, z)
    return v.Edged(1, [("a", Array(2, Value(16))), ("z", MIX)], Value(16), body, [(W, 2, v._units(1, 2)), (R, 1, v._units(1, 1))])
v.EDGED["scratch: a mixed-width field onto a row of its type"] = (lambda: onto2("EdgeMix"), v._repoint(["n", 4, 0, 3], 5), None, "?")
c = v.edged_case("scratch: a mixed-width field onto a row of its type")
desc = json.loads(c["instance"]["program"])
for i, n in enumerate(desc["definitions"][desc["root"]]["body"]["nodes"]): print(i, json.dumps(n)[:200])
try:
    print("detail", ref.check_detail(c["instance"], c["trusted"]))
except Exception as e:
    traceback.print_exc()
try:
    ref.expand(c["instance"], c["trusted"])
except Exception as e:
    print("cause", repr(e.__cause__)[:400]); tb = traceback.extract_tb(e.__traceback__); print("raised", e.args[:1], [(f.name, f.lineno) for f in tb[-2:]]); print(str(getattr(e,'args',[''])[-1])[:400])
json.dump(c, open("/tmp/ioread/lw_case.json", "w"))
