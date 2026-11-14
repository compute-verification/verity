module

public import Properties.Accounting.Remote.InProofs
public import Proofs.Properties.Accounting.InProofs
public import Proofs.Properties.Accounting.Remote.Window

/-!
# Proof of PoUB's read session inside proofs (`Properties.Accounting.Remote.InProofs`)

`checkWindow` on as many openings as draws is its first rejection in draw order, or accept (`checkWindow_eq`); a draw
whose opening `checkOpening` passes rejects nothing. So an accepted run whose decoded openings `checkWindow` refuses has a
draw whose value fails `OpeningOk`, and `Gives` at `right` gives it at `OpeningOk ∘ decode` (`Gives.mono`).
-/

@[expose] public section

open RemoteAudit

namespace Properties.RemoteAudit.InProofs.SecurityProofs

open Properties.RemoteAudit Properties.RemoteAudit.InProofs
open Verity.InProofs Properties.Accounting.InProofs Verity.Game

/-- Openings every one of which `checkOpening` passes at its draw make `checkWindow` accept. -/
theorem checkWindow_ofFn_accept (reg : Registration) (digest : ByteArray) (appends : Array Append)
    (w : WindowRecord) (drawn : Array (Nat × Nat)) (o : Fin drawn.size → Opening)
    (h : ∀ k, OpeningOk reg digest appends w drawn k (o k)) :
    checkWindow reg digest appends w drawn (Array.ofFn o) = .accept := by
  rw [checkWindow_eq]
  have hnone : (List.range drawn.size).findSome? (rejectAt reg digest appends w drawn (Array.ofFn o)) = none := by
    rw [List.findSome?_eq_none_iff]
    intro k hk
    have hk' := List.mem_range.1 hk
    have hok := h ⟨k, hk'⟩
    unfold OpeningOk at hok
    unfold rejectAt
    have hd : drawn[k]! = drawn[k] := getElem!_pos drawn k hk'
    have ho : (Array.ofFn o)[k]! = o ⟨k, hk'⟩ := by
      rw [getElem!_pos (Array.ofFn o) k (by simpa using hk'), Array.getElem_ofFn]
    rw [hd, ho, Option.map_eq_none_iff]
    exact hok
  simp [hnone]

theorem spanGives : SpanGives := by
  intro reg digest appends w drawn Ω μ V r right decode ε g h
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  classical
  obtain ⟨-, hne⟩ := hω
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro hall
  refine hne (checkWindow_ofFn_accept reg digest appends w drawn _ fun k => ?_)
  by_contra hk
  exact hall k (by simp [Run.failing, hk])

end Properties.RemoteAudit.InProofs.SecurityProofs
