module

public import MemoryAttester.WipeProtocol

/-! `wipe-protocol`: the command line of `MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`, the
streamed-label wipe's verdict under S1's LMS, held to the served node NAME (`MemoryAttester.WipeProtocol.mainOf`). `id`
prints the protocol id; `--server NAME check PARAMS COMMITMENT EVIDENCE` reads three files and prints the outcome's
canonical JSON line; `--server NAME serve` reads one case a line on standard input, `{"commitment": hex, "evidence":
hex, "params": JSON}`, and prints one outcome line for each. Each refuses by name a case of another server than NAME,
and `serve` or `check` with no server is refused by name (`unheld`). An accepted wipe's outcome publishes `foreign_bits`
and `eps` among its fields (`certify`), and a session not yet signed is undecided. `tests/test_wipe_protocol.py` drives
it on its named wipes, and `wipe_run.py` runs it on a period's wipe case when the period names it among its tools. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryAttester.WipeProtocol.main args
