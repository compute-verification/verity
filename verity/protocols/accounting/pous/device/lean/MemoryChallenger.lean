module

public import Attest
public import MemoryChallenger.SignedMessage
public import MemoryChallenger.Records
public import MemoryChallenger.Verdict
public import MemoryChallenger.Protocol
public import MemoryChallenger.Judge
public import MemoryChallenger.StreamedWipe
public import MemoryChallenger.WipeJudge
public import MemoryChallenger.WipeProtocol
public import MemoryChallenger.Difftest

/-!
# The memory challenger's verdicts, as executable Lean

The library of the package `core_memory_challenger`: the memory challenger's two verdicts, their record formats and
crypto, and each verdict behind the core protocol interface (`Core.Protocol`).

* `Records`, the coin tree, the session commitment, the round record and the records root, and `SignedMessage`, the
  message a session's record signs;
* `Verdict`, the check of a signed session of PoUS's timed rounds against the commitment published before it; `Judge`,
  that verdict on a record as its caller holds it; `Protocol`, the same verdict as a core protocol;
* `StreamedWipe`, the streamed-label wipe's formats and its check of a signed session; `WipeJudge`, that verdict on a
  record as its caller holds it; `WipeProtocol`, the wipe as a core protocol (`verity/wipe/v0`);
* `Difftest`, the functions `tests/` holds to the Python, among them those of `Attest`, the package `core_attest`'s
  period tree and checks, which this module imports.
-/

@[expose] public section
