module

public import Attest
public import MemoryAttester.SignedMessage
public import MemoryAttester.Records
public import MemoryAttester.Verdict
public import MemoryAttester.Protocol
public import MemoryAttester.Judge
public import MemoryAttester.StreamedWipe
public import MemoryAttester.WipeJudge
public import MemoryAttester.WipeProtocol
public import MemoryAttester.WipeCount
public import MemoryAttester.SetupRecord
public import MemoryAttester.InProofs
public import MemoryAttester.Difftest

/-!
# The memory challenger's verdicts, as executable Lean

The library of the package `core_memory_challenger`: the memory challenger's two verdicts, their record formats and
crypto, and each verdict behind the core protocol interface (`Core.Protocol`).

* `Records`, the coin tree, the session commitment, the round record and the records root, and `SignedMessage`, the
  message a session's record signs;
* `Verdict`, the check of a signed session of PoUS's timed rounds against the commitment published before it; `Judge`,
  that verdict on a record as its caller holds it; `Protocol`, the same verdict as a core protocol;
* `StreamedWipe`, the streamed-label wipe's formats and its check of a signed session; `WipeJudge`, that verdict on a
  record as its caller holds it; `WipeProtocol`, the wipe as a core protocol (`verity/wipe/v0`); `WipeCount`, what a
  composition's tie reads off the wipe's case and the wipe's check with its count held;
* `SetupRecord`, the setup's published outputs the memory challenger's device signs at setup, under a session's key
  and id, and its check against a session (`setupRecordOk`);
* `InProofs`, the session's verdict from the outcomes of its `k + 2` proofs, linked only by their committed rows
  (`runInProofs`, `pous.verifier.session_proofs`' `check_session`);
* `Difftest`, the functions `tests/` holds to the Python, among them those of `Attest`, the package `core_attest`'s
  period tree and checks, which this module imports.
-/

@[expose] public section
