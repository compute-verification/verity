module

public import InferenceOnlyVerifier.Difftest
meta import InferenceOnlyVerifier.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.Difftest.main`, audited under `InferenceOnlyVerifier`. It is the one
test that holds the Lean check to the Python reference, which computes the same numbers straight from the definitions. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.Difftest.main args
