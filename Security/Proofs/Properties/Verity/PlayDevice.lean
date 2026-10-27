import Properties.Verity.Game
import Proofs.Properties.Challenger

/-!
# What the challenger's properties say of the play's device

The play's device (`Vocab.deviceOf`) runs one session under its registered key, the one `PlayDevice.session` writes
from the play's releases (`PlayDevice.mem_sessions`). So where `roundCoins` accepts under that key,
`RoundsByDevice` names that session: it released each use's coins, its log shows the uses' roots, each order pair in
order and each group's uses one run each, or the finder outputs a break from one use's record
(`Vocab.rounds_at_play`). A break at the play's device under its registered key is a SHA-512 collision the finder
outputs, or a forgery at the play (`Vocab.forgery`), the signature verifying under the registered key over a message
the play's device never signed (`Vocab.break_at_play`).
-/

namespace Verity

open SampledProofs.Challenger

theorem PlayDevice.beq_self (pk : ByteArray) : (pk == pk) = true := by
  show ByteArray.beq pk pk = true
  simp [ByteArray.beq]

/-- Under its key, the play's device ran exactly the session `PlayDevice.session` writes from the releases. -/
theorem PlayDevice.mem_sessions {D : PlayDevice} {pk : ByteArray} {rs : List Release} {s : Properties.Challenger.Session} :
    s ∈ (D.device pk rs).sessions pk ↔ s = D.session rs := by
  simp [PlayDevice.device, PlayDevice.beq_self]

namespace Vocab

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **`RoundsByDevice` at the play's device**: where `roundCoins` reads `ξs` for the uses `us` under a key whose public
key is the registered one, the session the play's device wrote from the play's releases is opened withheld on the
window by its ledger, released each use's coins, shows the uses' roots, each pair of `order` in order and each group's
uses one run each; or the finder outputs a break at the play's device from one use's record. -/
theorem rounds_at_play (hR : Properties.Challenger.RoundsByDevice) {θ : ByteArray} (o : Outcome) {key : ChallengerKey}
    {w : String} {us : List Use} {order : List (String × String)} {groups : List (List String)} {coins : Lean.Json}
    {ξs : List ByteArray} (hk : key.pub = (registeredKey θ).2) (h : roundCoins key w us order groups coins = .ok ξs) :
    (Properties.Challenger.OpensOn (V.deviceOf θ o) key.pub w ((V.device θ).session o.log).id ∧
        (∀ (i : Nat) (u : Use), us[i]? = some u → ξs[i]? = some (((V.device θ).session o.log).released u.label u.after)) ∧
        (∀ ℓ ∈ ((V.device θ).session o.log).families.map (·.1),
          ((V.device θ).session o.log).rootsOf ℓ = Use.roots us ℓ) ∧
        (∀ ab ∈ order, ((V.device θ).session o.log).Before ab.1 ab.2) ∧
        ∀ g ∈ groups, ((V.device θ).session o.log).Runs g us) ∨
      ∃ (i : Nat) (u : Use), us[i]? = some u ∧
        Properties.Challenger.Break (V.deviceOf θ o) key u.label (Properties.Challenger.useCoins coins i) := by
  rcases hR (V.deviceOf θ o) key w us order groups coins ξs h with ⟨s, hs, rest⟩ | hb
  · left
    have hs' : s = (V.device θ).session o.log := by
      rw [hk] at hs
      exact PlayDevice.mem_sessions.1 hs
    subst hs'
    exact rest
  · exact .inr hb

/-- **A break at the play's device under its registered key** is a SHA-512 collision the finder outputs, or a forgery
at the play that the finder outputs: a signature verifying under the registered key over a message the play's device
never signed. -/
theorem break_at_play {θ : ByteArray} {o : Outcome} {key : ChallengerKey} {label : String} {coins : Lean.Json}
    (hk : key = ⟨(registeredKey θ).1, (registeredKey θ).2⟩)
    (hb : Properties.Challenger.Break (V.deviceOf θ o) key label coins) :
    (∃ x y, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y ∧
        Properties.Challenger.findBreak (V.deviceOf θ o) key label coins = some (.collision x y)) ∨
      ∃ m sig, Properties.Challenger.findBreak (V.deviceOf θ o) key label coins = some (.forgery m sig) ∧
        V.forgery θ o m sig := by
  subst hk
  obtain ⟨w, hw, hv⟩ := hb
  cases w with
  | collision x y => exact .inl ⟨x, y, hv.1, hv.2, hw⟩
  | forgery m sig => exact .inr ⟨m, sig, hw, hv.1, hv.2⟩

end Vocab

end Verity
