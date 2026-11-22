module

public import InferenceOnlyVerifier.ThetaDifftest
meta import InferenceOnlyVerifier.ThetaDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.ThetaDifftest.main`, audited under
`InferenceOnlyVerifier`. It holds `InferenceOnlyVerifier.Theta.check` to the Python reference and to the unrolled check on the same cases. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.ThetaDifftest.main args
