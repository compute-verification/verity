module

public import InferenceOnlyCheck.Difftest
meta import InferenceOnlyCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.Difftest.main`, audited under `InferenceOnlyCheck`. It is the one
test that holds the Lean check to the Python reference, which computes the same numbers straight from the definitions. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.Difftest.main args
