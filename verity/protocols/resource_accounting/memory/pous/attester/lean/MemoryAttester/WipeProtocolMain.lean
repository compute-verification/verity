module

public import MemoryAttester.WipeProtocol

/-! `wipe-protocol`: the command line of `MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`, the
streamed-label wipe's verdict under S1's LMS, held to the served node NAME and the fill floor BYTES the card file
gives (`MemoryAttester.WipeProtocol.mainOf`). `id` prints the protocol id; `--server NAME --fill BYTES check
PARAMS COMMITMENT EVIDENCE` reads three files and prints the outcome's canonical JSON line; `--server NAME --fill BYTES
serve` reads one case a line on standard input, `{"commitment": hex, "evidence": hex, "params": JSON}`, and prints one
outcome line for each. Each refuses by name a case of another server than NAME, and one whose labels fill less than
BYTES (`underFilled`); `serve` or `check` without both is refused by name (`unheld`). An accepted wipe's outcome
publishes `foreign_bits` and `eps` among its fields (`certify`), and a session not yet signed is undecided.
`tests/test_wipe_protocol.py` drives it on its named wipes, and `wipe_run.py` runs it on a period's wipe case when the
period names it among its tools. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryAttester.WipeProtocol.main args
