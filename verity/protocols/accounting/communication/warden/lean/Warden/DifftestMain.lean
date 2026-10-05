import Warden.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetTiming.Code.Difftest.main`, audited under `Warden`.

The audit runs it interpreted, about two minutes on four cores for the 10,000 instances. It is the one test that holds
the Python audit to the warden program, which `Proofs.Warden.Code` proves equal to the reference definitions the
guarantees are about: the Python tests mirror the guarantees' statements on small examples, and a definition that
drifts from its Python counterpart passes both sides' own tests. -/

def main (args : List String) : IO UInt32 := NetTiming.Code.Difftest.main args
