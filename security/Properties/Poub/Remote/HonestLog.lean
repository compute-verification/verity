module

public import Properties.Poub.Remote.HonestLog.Case

/-! The deployed check accepts an honest one-window log (`HonestLogAccepted`), which nothing in Lean proves: the
kernel doesn't evaluate the log's decoding or the window's sample. -/

@[expose] public section

namespace Properties.Poub.Subprotocol

/-- **The deployed check accepts the honest one-window log** (witnessed outside Lean: `poub-protocol` accepts it). -/
def HonestLogAccepted : Prop :=
  ∃ (reg : Poub.Registration) (pub : Poub.Protocol.Public) (prof : Lean.Json),
    Poub.Difftest.registrationOf HonestLog.registration = .ok reg ∧
      Poub.Protocol.check HonestLog.params reg HonestLog.evidence = .accept pub prof

end Properties.Poub.Subprotocol
