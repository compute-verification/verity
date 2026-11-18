module

public import InferenceOnlyVerifier.DescLiveness
meta import InferenceOnlyVerifier.DescLiveness

/-! The description-sized check's liveness run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.DescLiveness.main`, audited under
`InferenceOnlyVerifier`. It holds `InferenceOnlyVerifier.Desc.check` to the unrolled check, `InferenceOnlyVerifier.check`, on Program shapes it once walked member
by member or iteration by iteration, and on the same shapes at `2^22` to `2^52` of them to its own verdict. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.DescLiveness.main args
