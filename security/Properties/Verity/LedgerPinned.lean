module

public import Properties.Challenger
public import Definitions.Core.Site

/-! A ledger line signed under a key the registration pins (`KeyPinned`, `LinePinned`, `LinesPinned`). -/

@[expose] public section

namespace Properties.Verity

open Lean (Json)
open Spot.Challenger

/-- **The registration pins the public key `pk`**: some location of the auditor's in the Site the log `log` builds is
registered under `pk` in lower-case hex, and the auditor's own key for that location (`A.key`) is that string. -/
def KeyPinned (log : _root_.Verity.Registration) (A : _root_.Verity.Registration.Auditor) (pk : ByteArray) : Prop :=
  ∃ x ∈ (_root_.Verity.Site.ofLog log).locs, x.kind.auditor = true ∧ x.key = Flock.hexOf pk ∧ A.key x.id = some x.key

/-- **A ledger line signed under a pinned key**: `line` parses as a ledger entry, the entry's message under `key`'s LMS
identifier is defined, LMS verifies the entry's signature over that message under `key`, and the registration pins
`key`'s public key. -/
def LinePinned (log : _root_.Verity.Registration) (A : _root_.Verity.Registration.Auditor) (key : ChallengerKey)
    (line : Json) : Prop :=
  ∃ e m, ledgerEntry? line = some e ∧ ledgerMessage (Properties.Challenger.ident key.pub) e.entry = .ok m ∧
    Properties.Challenger.LmsVerifies key m e.sig ∧ KeyPinned log A key.pub

def LinesPinned (log : _root_.Verity.Registration) (A : _root_.Verity.Registration.Auditor) (key : ChallengerKey)
    (lines : List Json) : Prop :=
  ∀ line ∈ lines, LinePinned log A key line

/-- **Not vacuous**: under a key the registration doesn't pin, a list holding a line is not pinned. -/
theorem linesPinned_unpinned {log : _root_.Verity.Registration} {A : _root_.Verity.Registration.Auditor}
    {key : ChallengerKey} {lines : List Json} {line : Json} (hk : ¬KeyPinned log A key.pub) (hl : line ∈ lines) :
    ¬LinesPinned log A key lines := fun h =>
  let ⟨_, _, _, _, _, hp⟩ := h line hl
  hk hp

end Properties.Verity
