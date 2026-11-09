import InferenceOnlyCheck.DescLiveness

/-! The description-sized check's liveness run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.DescLiveness.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Desc.check` to the unrolled check, `InferenceOnlyCheck.check`, on Program shapes it once walked member
by member or iteration by iteration, and on the same shapes at `2^22` to `2^52` of them to its own verdict. -/

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.DescLiveness.main args
