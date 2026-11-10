# The memory challenger's verdicts in Lean (`core_memory_challenger`)

This Lake package is code: the memory challenger's two verdicts as executable Lean, their record formats and crypto, and
each verdict behind the core protocol interface (`Core.Protocol`). It imports Lean core, Flock's verifier for its
hashes and canonical JSON, and core's randomness, LMS, attest, protocol interface and difftest packages; no Mathlib and
nothing of `Security/`. `Security/Properties/Pous/` states its properties against these functions (the area's README,
one directory up).

## Top-level API

- `MemoryChallenger.verdict` (`MemoryChallenger/Verdict.lean`): the check of a signed run of timed rounds against the
  commitment published before it; it fails closed, and its reasons are the Python's word for word.
- `MemoryChallenger.protocol` (`MemoryChallenger/Protocol.lean`): that verdict as a core protocol, the executable
  `memory-challenger`; its public output is the verdict alone.
- `MemoryChallenger.Judge.judge` (`MemoryChallenger/Judge.lean`): the verdict on a record as its caller holds it, the
  executable `memory-challenger-verdict`, which `verdict.py` calls behind `memory_challenger.verdict`'s signature.
- `MemoryChallenger.StreamedWipe.verdict` (`MemoryChallenger/StreamedWipe.lean`): the streamed-label wipe's check of a
  signed run; it refuses a batch of more than one read, which the guarantee doesn't cover.
- `MemoryChallenger.WipeProtocol.protocol` (`MemoryChallenger/WipeProtocol.lean`): the wipe as a core protocol,
  `verity/wipe/v0`, the executable `wipe-protocol`; an accepted, covered run publishes `foreign_bits` at
  `eps = 2^-128` (`certify`), and an unsigned one is undecided.
- `MemoryChallenger.StreamedWipe.judge` (`MemoryChallenger/WipeJudge.lean`): the wipe's verdict on a record as its caller
  holds it, the executable `streamed-wipe-verdict`, which `wipe_verdict.judge` calls (`lean`, `both` or `python`).
- `MemoryChallenger.Records` and `MemoryChallenger.SignedMessage`: the coin tree, the run's commitment, the round
  record, the records root and the message the device signs.
- `MemoryChallenger.Difftest` (`memory-challenger-difftest`): the functions `tests/` holds to the Python.

`memory-challenger-test` and `wipe-protocol-test` are the two protocols under ids of their own that accept only the
test signers' schemes, for the difftests.

## Build and test

~~~bash
lake build                                       # here: the library and every executable
# from the root:
python3 tools/verity/lean/check.py --build verity/protocols/pous/device/lean   # the Lean check (lean-audit.json)
uv run tools/verity/check/suites.py verity/protocols/pous/device/lean          # the difftests in tests/
~~~
