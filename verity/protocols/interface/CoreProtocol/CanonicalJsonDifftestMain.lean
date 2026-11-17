module

public import CoreProtocol.CanonicalJsonDifftest

/-! # `core-canonical-json-difftest`: `Core.CanonicalJsonDifftest.table` behind `Difftest.serve`

The executable's entry point (`lakefile.toml`'s `core-canonical-json-difftest`, not a default target: the difftest
harness builds it by name). It reads one request a line on standard input, `{"fn": NAME, "input": JSON}`, for
`canonical-parsers`, and prints one line for each: `{"output": JSON}`, `{"reject": REASON}` or `{"error": MESSAGE}`.
`tools/difftest/tests/test_canonical_json_port.py` drives it against the Python mirror. -/

@[expose] public section

def main : IO Unit := Difftest.serve Core.CanonicalJsonDifftest.table
