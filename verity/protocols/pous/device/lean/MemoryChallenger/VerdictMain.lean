module

public import Difftest
public import MemoryChallenger.Judge

/-! `memory-challenger-verdict`: the one function `verdict` (`MemoryChallenger.Judge.verdictFn`) behind
`Difftest.serve`. It reads one request a line on standard input, `{"fn": "verdict", "input": RECORD}`, the record
`memory_challenger.verdict`'s arguments as JSON, until the input ends, and prints one line for each: `{"output": JSON}`,
the whole verdict (`accepted`, `late`, `reasons`, `wrong`); `{"reject": REASON}`, where it refuses the record; or
`{"error": MESSAGE}`, a fault of the line protocol. `verdict.py` beside the package sends it records,
`tests/test_verdict.py` holds it to `memory_challenger.verdict`, and `benchmarks/pous/memory_challenger/live.py` judges
its sessions through `verdict.py`. -/

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.Judge.verdictFn)])
