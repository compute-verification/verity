module

public import CoreProtocol.PeriodDifftest

/-! # `core-periods-difftest`: `Core.PeriodDifftest.table` behind `Difftest.serve`

The executable's entry point (`lakefile.toml`'s `core-periods-difftest`, not a default target: the difftest harness
builds it by name). It reads one request a line on standard input, `{"fn": NAME, "input": JSON}`, for `span`, `within`, `cut`
and `monitoring`, and prints one line for each: `{"output": JSON}`, `{"reject": REASON}` or `{"error": MESSAGE}`.
`verity/periods/tests/test_periods_difftest.py` drives it against `verity.periods`. -/

@[expose] public section

def main : IO Unit := Difftest.serve Core.PeriodDifftest.table
