import Proofs.NetworkCertifier.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetTiming.Difftest.main`, audited under `NetTiming`.

The audit runs it interpreted, about two minutes on four cores for the 10,000 instances. It is the one test that holds
the Python audit to `NetTiming`'s definitions, which the pins are about: the Python tests mirror the pins' statements on
small examples, and a definition that drifts from its Python counterpart passes both sides' own tests. -/

def main (args : List String) : IO UInt32 := NetTiming.Difftest.main args
