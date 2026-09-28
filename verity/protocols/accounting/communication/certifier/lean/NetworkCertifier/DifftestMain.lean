import NetworkCertifier.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetworkCertifier.Difftest.main`, audited under `NetworkCertifier`.

The audit runs it interpreted, about two minutes on four cores for the 10,000 instances. It is the one test that holds
the Python audit to the certifier program, which `Proofs.NetworkCertifier.Code` proves equal to the reference definitions the
guarantees are about: the Python tests mirror the guarantees' statements on small examples, and a definition that
drifts from its Python counterpart passes both sides' own tests. -/

def main (args : List String) : IO UInt32 := NetworkCertifier.Difftest.main args
