module

public import Poub.Basic
public import Poub.Codec
public import Poub.Plan
public import Poub.Receipts
public import Poub.Window
public import Poub.Session
public import Poub.Shared
public import Poub.Flight
public import Poub.Registry
public import Poub.Wire
public import Poub.Pooled
public import Poub.Line
public import Poub.Schedule
public import Poub.P2
public import Poub.P2Session

/-!
# PoUB's verifier, as executable Lean

The library of the package `poub`, the definition that the Python of `verity.protocols.accounting.poub.verifier` is held
to (`audit.py` and what it calls, by `tests/`):

* `Basic`, the byte encodings, SHA-512 and frame-v3-sha512 openings; `Codec`, the store's codec, decoding; `Plan`, a
  pass's plan; `Receipts`, the seeds, the receipts and their tree; `Wire`, the frame a prover sends openings in;
* `Window`, the check of the sampled receipts' openings (`verifyWindow`); `Session`, a session's bookkeeping, a state
  machine over the events the verifier sees; `Shared`, that session with one openings coin over several windows, and
  `Flight`, with every coin sent after the root before it; `Registry`, the verifier's registrations;
* `Pooled`, the openings' pooled bound; `Line`, the memory line held over time; `Schedule`, the schedule check;
* `P2` and `P2Session`, a registration, receipts, openings and sessions over PoUS's P2 store.

The modules the executables are built on are not imported here, and each is a root of the Lean check of its own
(`roots` in `lean-audit.json`): `Poub.Difftest` (`poub-difftest`), `Poub.Verifier`
(`poub-verifier`), `Poub.Protocol` (`poub-protocol`) and `Poub.P2Protocol`
(`poub-p2-protocol`), with the bytes of P2's close and their lemmas (`Poub.P2Close`).
-/

@[expose] public section
