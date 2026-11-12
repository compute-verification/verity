module

public import RemoteAudit.Basic
public import RemoteAudit.Codec
public import RemoteAudit.Plan
public import RemoteAudit.Receipts
public import RemoteAudit.Window
public import RemoteAudit.Session
public import RemoteAudit.Shared
public import RemoteAudit.Flight
public import RemoteAudit.Registry
public import RemoteAudit.Wire
public import RemoteAudit.Unboxed
public import RemoteAudit.Store
public import RemoteAudit.Pooled
public import RemoteAudit.Line
public import RemoteAudit.Schedule
public import RemoteAudit.P2
public import RemoteAudit.P2Session

/-!
# PoUB's verifier, as executable Lean

The library of the package `remote_audit`, the definition that the Python of `verity.protocols.poub.verifier` is held
to (`audit.py` and what it calls, by `tests/`):

* `Basic`, the byte encodings, SHA-512 and frame-v3-sha512 openings; `Codec`, the store's codec, decoding; `Plan`, a
  pass's plan; `Receipts`, the seeds, the receipts and their tree; `Wire`, the frame a prover sends openings in;
* `Window`, the check of the sampled receipts' openings (`verifyWindow`); `Session`, a session's bookkeeping, a state
  machine over the events the verifier sees; `Shared`, that session with one openings coin over several windows, and
  `Flight`, with every coin sent after the root before it; `Registry`, the verifier's registrations;
* `Store`, the decision on a registered store bound to a pinned model, its payload root through `Unboxed`; `Pooled`, the
  openings' pooled bound; `Line`, the memory line held over time; `Schedule`, the schedule check;
* `P2` and `P2Session`, a registration, receipts, openings and sessions over PoUS's P2 store.

The modules the executables are built on are not imported here, and each is a root of the Lean check of its own
(`roots` in `lean-audit.json`): `RemoteAudit.Difftest` (`remote-audit-difftest`), `RemoteAudit.Verifier`
(`remote-audit-verifier`), `RemoteAudit.Protocol` (`remote-audit-protocol`) and `RemoteAudit.P2Protocol`
(`remote-audit-p2-protocol`).
-/

@[expose] public section
