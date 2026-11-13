module

public import RemoteAudit.Difftest

/-! `remote-audit-difftest`: `RemoteAudit.Difftest.table` behind `Difftest.serve`. It reads one request a line on
standard input, `{"fn": NAME, "input": JSON}`, until the input ends, and prints one line for each: `{"output": JSON}`;
`{"reject": REASON}`, where the function refuses its input; or `{"error": MESSAGE}`, a fault of the line protocol (a
line that is no request, an unknown `fn`, an output with no canonical form). The package's `tests/` drive it
(`ra_cases.EXE`), holding `audit.py`, `receipts.py`, `codec.py`, `store.py`, `line.py`, `pooled.py`, `schedule.py`,
`p2.py` and `p2_party.py` to the Lean. -/

@[expose] public section

def main : IO Unit := Difftest.serve RemoteAudit.Difftest.table
