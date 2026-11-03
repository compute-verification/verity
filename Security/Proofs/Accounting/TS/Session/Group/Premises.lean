import Proofs.Accounting.TS.Session.Group.Assumptions

/-!
# The premises a leaf's group of linked sessions takes

The named premises a statement about a leaf proved as a group of C-Flock sessions takes, one premise over all the
group's sessions each: `shape.premises` in `Security/lean-audit.json` names this module. Each is the named assumption
it abbreviates at every proof's every session, unchanged:

* `SessionsCR` (`Flock.SoundAtBridge.RowsSession.CR`): C-Flock's `EndToEndRegisteredRows`' named assumptions,
  `cr/sha-512` and `ecr/sha-512` for its finders, at every session and every unit's opening.
* `WindowLaws` (`Accounting.Assumptions.TS.WindowLaw`, `accounting-ts-window-law`): at each statement of the group, the
  window's coins read through its sessions have the sessions' law.
* `GroupRealizes` (`Accounting.Assumptions.TS.GroupRealizes`, `accounting-ts-group-realizes`): at every proof, the
  group's circuits realize the leaf's relation at the rows its sessions open.

They belong in `Security/Definitions/`, but they read `RowsSession`, which is in the proofs package, so they are here
until C-Flock's spec is extracted, as `Accounting.Assumptions.TS`'s other session premises are.
-/

namespace Accounting.Premises.TS

open Accounting.TS Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)

/-- **`RowsSession.CR` at every session of every proof**: `EndToEndRegisteredRows`' named assumptions. -/
abbrev SessionsCR {Ω : Type} {S nπ : ℕ} {ι : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession) : Prop :=
  ∀ x w i k, (sess x w i k).CR

/-- **`Accounting.Assumptions.TS.WindowLaw` at every statement of the group**. -/
abbrev WindowLaws {Ω : Type} [Fintype Ω] {S nπ : ℕ} {ι : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) : Prop :=
  ∀ k, Accounting.Assumptions.TS.WindowLaw (fun x w i => sess x w i k) (fun x w i => out x w i k)

/-- **`Accounting.Assumptions.TS.GroupRealizes` at every proof**. -/
abbrev GroupRealizes {Ω : Type} {S nπ : ℕ} {ι V : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession)
    (known : Paths Ω S → Fin S → Fin nπ → ι → Accounting.TS.Lanes.Known)
    (full : Paths Ω S → Fin S → Fin nπ → ι → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → ι → List _root_.Flock.Registered.Port)
    (gate : Paths Ω S → Fin S → Fin nπ → Prop) (opened : Paths Ω S → Fin S → Fin nπ → V)
    (rel : Paths Ω S → Fin S → Fin nπ → V → Prop) : Prop :=
  ∀ x w i, Accounting.Assumptions.TS.GroupRealizes (sess x w i) (known x w i) (full x w i) (want x w i) (gate x w i)
    (opened x w i) (rel x w i)

end Accounting.Premises.TS
