module

public import MemoryChallenger.WipeProtocol

/-! `wipe-protocol`: the core protocol command line (`Core.Protocol.main`) of `MemoryChallenger.WipeProtocol.protocol`,
`verity/wipe/v0`, the streamed-label wipe's verdict under S1's LMS. `id` prints the protocol id; `check PARAMS
COMMITMENT EVIDENCE` reads three files and prints the outcome's canonical JSON line; `serve` reads one case a line on
standard input, `{"commitment": hex, "evidence": hex, "params": JSON}`, and prints one outcome line for each. An
accepted wipe's outcome publishes `foreign_bits` and `eps` among its fields (`certify`), and a session not yet signed is
undecided. `tests/test_wipe_protocol.py` drives `serve` on its named wipes, and `wipe_period.py`, in the verifier's
Python, runs it on a period's wipe case when the period names it among its tools. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.WipeProtocol.protocol.main args
