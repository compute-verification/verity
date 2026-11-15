module

public import Properties.Challenger
public import Definitions.Core.Site

/-!
# A ledger line signed under a key the registration pins

The challenger's ledger lines are what the auditor's device signs (`Properties.Challenger.History.ledger`); the reader
checks each one's signature under the key the parameters name (`Properties.Challenger.signatures`). That key is the
auditor's only where the registration pins it. These say so of a line and of a list of lines, against the registration
log's Site (`Verity.Site.ofLog`) and the auditor's own keys (`Verity.Registration.Auditor.key`):

* `KeyPinned`: a location of the auditor's (`Verity.Site.Kind.auditor`) in the log's Site is registered under the key,
  in lower-case hex (`Flock.hexOf`, the one form a key is compared in), and the auditor's own key for that location is
  the same string, as the registration's `Owners` check holds a registered key to;
* `LinePinned`: the line is a ledger entry (`Spot.Challenger.ledgerEntry?`), whose signature LMS verifies over the
  entry's message (`Spot.Challenger.ledgerMessage`) under a key the registration pins;
* `LinesPinned`: every line of a list is.

`linesPinned_unpinned` is the shape that keeps `LinesPinned` from being vacuous: under a key the registration doesn't
pin, no list holding a line is pinned.
-/

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

/-- **Every ledger line of `lines` is signed under a pinned key** (`LinePinned`, at `key`). -/
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
