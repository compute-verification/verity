module

public import NetworkAttester.Difftest
meta import NetworkAttester.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetworkAttester.Difftest.main`, audited under `NetworkAttester`.

The audit runs it interpreted, about two minutes on four cores for the 10,000 instances. It is the one test that holds
the Python audit to network attestation's program, which `Proofs.Certifier.Code` proves equal to the reference definitions the
guarantees are about: the Python tests mirror the guarantees' statements on small examples, and a definition that
drifts from its Python counterpart passes both sides' own tests. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkAttester.Difftest.main args
