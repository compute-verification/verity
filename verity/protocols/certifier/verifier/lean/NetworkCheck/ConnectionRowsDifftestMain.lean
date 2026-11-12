module

public import NetworkCheck.ConnectionRowsDifftest
meta import NetworkCheck.ConnectionRowsDifftest

/-! The connection rows' difftest's run (`runs` in `lean-audit.json`): `NetworkCheck.ConnectionRowsDifftest.main`,
audited under `NetworkCheck`. It holds `connection_rows.py`'s tables, domains, heads and registration read to
`NetworkCheck.ConnectionRows` on the generated instances. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.ConnectionRowsDifftest.main args
