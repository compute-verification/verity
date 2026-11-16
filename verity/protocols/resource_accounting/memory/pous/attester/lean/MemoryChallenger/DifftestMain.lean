module

public import MemoryChallenger.Difftest

/-! `memory-challenger-difftest`: `MemoryChallenger.Difftest.table` behind `Difftest.serve`. It reads one request a line
on standard input, `{"fn": NAME, "input": JSON}`, until the input ends, and prints one line for each:
`{"output": JSON}`; `{"reject": REASON}`, where the function refuses its input; or `{"error": MESSAGE}`, a fault of the
line protocol (a line that is no request, an unknown `fn`, an output with no canonical form). `tests/test_formats.py`
drives it against `memory_challenger.py`, `streamed_wipe.py` and `verity.commitments.crypto.lms`, and
`tests/test_attest.py` against `verity.commitments.crypto.lms`, `verity.commitments.merkle` and
`verity.commitments.attest`. -/

@[expose] public section

def main : IO Unit := Difftest.serve MemoryChallenger.Difftest.table
