import NciCheck.DescLiveness

/-! The description-sized check's liveness run (`runs` in `lean-audit.json`): `NciCheck.DescLiveness.main`, audited under
`NciCheck`. It holds `NciCheck.Desc.check` to the unrolled check, `NciCheck.check`, on Program shapes it once walked member
by member or iteration by iteration, and on the same shapes at `2^22` to `2^52` of them to its own verdict. -/

def main (args : List String) : IO UInt32 := NciCheck.DescLiveness.main args
