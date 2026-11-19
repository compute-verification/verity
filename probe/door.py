import sys, traceback
sys.path.insert(0, "tests/verity/protocols/resource_accounting/memory/pous/verifier")
import test_pous_setup_proof as T
from verity.circuits import restrict as R
from verity.circuits.codec import decode_program
c, win, part, reg = T.registered()
_, _, setup = T.setup_case()
program, _q = setup.circuit().program()
print("bindings", program.bindings, "supplied", program.supplied)
print("inputs", [nd.fn.id for nd in program.root.body.nodes if nd.fn.id.endswith("Input16_v1") or "Input" in nd.fn.id][:8])
try:
    b = R.restrict(c, part, [(5, 6)])
    print("restricted", b.program.bindings)
except Exception:
    traceback.print_exc(file=sys.stdout)
