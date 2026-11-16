module

public import Properties.Poub.Remote.HonestLog.Case

/-!
# The deployed check accepts an honest one-window log (`HonestLogAccepted`)

`HonestLog.Case` holds the case of an honest period on one node: the registration of one block under the deployed
codec, the parameters (one window of 100 ms, one sample, a 500 ms deadline, the lifetime cap of 300 windows, unpaced and
not deferred) and the evidence, its two coins and the verifier's log (the session's start, the pass's root, the window's
end, its sample and the opening). `HonestLogAccepted` says the deployed check (`Poub.Protocol.check`) accepts it.
The honest witnesses of `LineHeldAtSubprotocol` and `PublishedEps` take it, and nothing in Lean proves it yet: the
kernel doesn't evaluate the log's decoding (`Flock.unhex` on its block of 4096 bytes) or the window's sample
(`receiptSample`, which `Session.close` runs).

It is what `test_honest_log.py` (the PoUB verifier's Lean tests) checks: `poub-protocol`, the compiled check, accepts
the case with the Python verifier's public outputs and profile, and `poub-difftest`'s session of it samples the Python
verifier's receipt at the window's close; and the module is `honest_log.py`'s rendering of the case.
-/

@[expose] public section

namespace Properties.Poub.Subprotocol

/-- **The deployed check accepts the honest one-window log** (witnessed outside Lean: `poub-protocol` accepts it). -/
def HonestLogAccepted : Prop :=
  ∃ (reg : Poub.Registration) (pub : Poub.Protocol.Public) (prof : Lean.Json),
    Poub.Difftest.registrationOf HonestLog.registration = .ok reg ∧
      Poub.Protocol.check HonestLog.params reg HonestLog.evidence = .accept pub prof

end Properties.Poub.Subprotocol
