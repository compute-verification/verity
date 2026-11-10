module

public import NciCheck.DescDifftest
meta import NciCheck.DescDifftest

/-! The description-sized check's difftest run (`runs` in `lean-audit.json`): `NciCheck.DescDifftest.main`, audited under
`NciCheck`. It holds `NciCheck.Desc.check` to the unrolled check, `NciCheck.check`, on the same declarations. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NciCheck.DescDifftest.main args
