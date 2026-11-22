"""The memory attester, the auditor-trusted device that runs PoUS's timed rounds and the streamed wipe's reads and signs
each. Its Lean is `lean/` (the Lake package `core_memory_challenger`), whose difftests hold it to the rounds' Python, PoUS's
`memory_attester`; its timed audit's process is `timed_audit`, a client of a responder's wire whose verdict PoUS's
Lean check (`pous_check`'s `pous-timed-audit`) is held to. `challenger_party` is its party on the challenger host, over
the messages and ids PoUS holds (`challenger_session`); the streamed wipe's is
`verity.protocols.accounting.memory.wipe.attester.wipe_challenger`. `continuous` is the continuous audit's memory
attester.

Trust in the device and its host is the named assumption `trusted-device` (`TrustedDevice`,
`security/Definitions/Core/Assumptions.lean`). What its verdicts mean is stated against the Lean's functions:
`MemoryAttester.Properties.VerdictAccepts` and `VerdictSigned` (`security/Properties/Pous/MemoryAttester.lean`) for the
timed rounds, `MemoryAttester.Properties.WipeSurvivorBound` (`security/Properties/Pous/Wipe.lean`) and
`Pous.Component.WipeAccepts` (`security/Properties/Pous/Component.lean`) for the wipe."""
