module

public import Difftest
public import MemoryChallenger.WipeJudge

/-! `streamed-wipe-verdict`: the one function `verdict` (`MemoryChallenger.StreamedWipe.verdictFn`) behind
`Difftest.serve`. It reads one request a line on standard input, `{"fn": "verdict", "input": RECORD}`, the record
`streamed_wipe.verdict`'s arguments as JSON, until the input ends, and prints one line for each: `{"output": JSON}`, the
whole verdict (`accepted`, `egress_mib`, `reasons`); `{"reject": REASON}`, where it refuses the record; or
`{"error": MESSAGE}`, a fault of the line protocol. `wipe_verdict.py` beside the package sends it records,
`tests/test_wipe_verdict.py` holds it to `streamed_wipe.verdict`, and `benchmarks/pous/streamed_wipe/bench.py` judges
its session through `wipe_verdict.py`. -/

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", MemoryChallenger.StreamedWipe.verdictFn)])
