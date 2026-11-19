import json, sys
from pathlib import Path
sys.path.insert(0, "tests/verity/circuits")
import test_restrict as TR
import verity_difftest as D
from verity.circuits import boolean as BL
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Tuple

A4 = TR.A4
T, T2 = TR.T, bind(CompositeDefinition("RestrictToyT2", 1, (), lambda S: ((("a", A4), ("k", A4)), A4), lambda B, S, a, k: array_of(
    [B.call(BL.AND, BL.operands(B.call(BL.XOR, BL.operands(a[i], k[i])), a[(i + 3) % 4])) for i in range(4)]), register=False))


def body(B, S, x, k):
    u0 = B.call(T, x, k)
    u1 = B.call(T2, u0, k)
    u2 = B.call(T, u1, k)
    return tuple_of(u2, B.call(T2, u2, x))


P = Program(CompositeDefinition("TwoTemplates", 1, (), lambda S: ((("x", A4), ("k", A4)), Tuple(A4, A4)), body, register=False))
q = PO.template_instances_query([T, T2])
units = [descriptor_id(nd.fn) for nd in P.root.body.nodes if nd.form != "primitive"]
c = b"".join(descriptor_chunks(P))
port = D.Port("spot-difftest", "restrict", lambda _j: None, None,
              package=Path("verity/protocols/proofs/spot/verifier/lean"), of_record="lean")
res = {"units": units, "bindings": P.bindings, "blocks": []}
with D.Server(port.argv()) as s:
    for lo, hi in ((0, 2), (1, 4), (0, 4)):
        b = R.restrict(c, {"query": q}, [(lo, hi)])
        kind, body_ = s.call("restrict", {"descriptor": c.decode(), "units": list(range(lo, hi)), "python": b.descriptor.decode()})
        out = json.loads(body_) if kind == "output" else body_
        res["blocks"].append({"units": [lo, hi], "templates": units[lo:hi], "kind": kind,
                              "agree": out.get("agree") if isinstance(out, dict) else None,
                              "lowers": out.get("lowers") if isinstance(out, dict) else out[:400]})
print(json.dumps(res, indent=1))
