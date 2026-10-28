import Proofs.Accounting.TS.Session.Period.Guarantees
import Proofs.Accounting.TS.SessionSound
import Proofs.Accounting.FillerLuck
import Proofs.Pouw.Period.Honest

/-!
# PoUW's proof system soundness from its proof layer (the proof)

`Accounting.SecurityProofs.TS.CheckProofSystemSound` is a union bound over a window's sessions: where the check accepts
a window and some drawn unit `u` fails its relation, the session `sessOf x w u` that proves `u` is accepted and its
statement fails at its opening, so the event lies in the union of the window's `n_π` unsound sessions, each at most
`δ`. `WindowSessions.serviceSound_of_sessions` composes it with `Accounting.SecurityProofs.TS.SessionSoundAt`: a
window's sessions that are C-Flock sessions meeting that guarantee's named premises give `ServiceSound` at
`n_π · (δ_s + δ)`.

The witnesses are at H1's honest live-salt period (`Properties.Pouw.Period.SecurityProofs.HonestH1`), one window and
one session: there the check accepts every window on every path and its layer is sound at `0`, and with relations
that every opening fails `ServiceSound` fails at `0`, since the check accepts and draws a unit on every path.

This module imports `Proofs.Accounting.TS.SessionSound`, whose imports make `X` a token, so `Proofs/Accounting.lean`
imports it and no theorem here binds `X`.
-/

namespace Accounting.TS.WindowSessions

open Pouw.RelativeSecurity

/-- **A window's drawn-unit failures are a union of its unsound sessions**: the window is accepted and some drawn unit
fails its relation with probability at most `n_π · δ` when each session of the layer is unsound with probability at
most `δ`. -/
theorem prob_drawn_le {Ω U V : Type} [Fintype Ω] {n nπ : ℕ} (acc : Paths Ω n → Fin n → Prop)
    (drawn : Paths Ω n → Fin n → U → Prop) (key : U → ℕ)
    (opened : Paths Ω n → Fin n → ℕ → ℕ → V) (rel : Paths Ω n → Fin n → U → (ℕ → V) → Prop)
    (sessOf : Paths Ω n → Fin n → U → Fin nπ) {δ : ℝ}
    (h : (layer acc drawn key opened rel sessOf).SoundAt δ) (w : Fin n) :
    prob (fun x => acc x w ∧ ∃ u, drawn x w u ∧ ¬ rel x w u (opened x w (key u))) ≤ nπ * δ := by
  calc prob (fun x => acc x w ∧ ∃ u, drawn x w u ∧ ¬ rel x w u (opened x w (key u)))
      ≤ prob (fun x => ∃ i ∈ (Finset.univ : Finset (Fin nπ)),
          (layer acc drawn key opened rel sessOf).Unsound x w i) :=
        prob_mono fun x ⟨ha, u, hd, hr⟩ =>
          ⟨sessOf x w u, Finset.mem_univ _, ha, fun hall => hr (hall u hd rfl)⟩
    _ ≤ ∑ i : Fin nπ, prob (fun x => (layer acc drawn key opened rel sessOf).Unsound x w i) :=
        prob_exists_le Finset.univ (fun i x => (layer acc drawn key opened rel sessOf).Unsound x w i)
    _ ≤ ∑ _i : Fin nπ, δ := Finset.sum_le_sum fun i _ => h w i
    _ = nπ * δ := by simp

end Accounting.TS.WindowSessions

open Pouw.RelativeSecurity Accounting.TS.WindowSessions in
/-- **PoUW's proof system is sound against its check, from its sessions' soundness**
(`Accounting.Guarantees.TS.CheckProofSystemSound`): `prob_drawn_le` at the check's layer, whose verdict is the check's
acceptance and whose drawn units are the units it publishes. -/
theorem Accounting.SecurityProofs.TS.CheckProofSystemSound : Accounting.Guarantees.TS.CheckProofSystemSound := by
  intro C Ξ V _ _ m Np n nπ p ws opened rel sessOf δ h w
  refine (prob_mono fun x hx => ?_).trans (prob_drawn_le _ _ Fin.val opened rel sessOf h w)
  obtain ⟨pub, prof, hc, u, hu, hr⟩ := hx
  exact ⟨⟨pub, prof, hc⟩, u, ⟨pub, prof, hc, hu⟩, hr⟩

namespace Accounting.TS.WindowSessions

open Pouw.RelativeSecurity Flock.SoundAtBridge
open scoped ENNReal

/-- **`ServiceSound` from a window's C-Flock sessions**: when the check layer's proofs are C-Flock sessions `sess` that
meet `Accounting.Guarantees.TS.SessionSoundAt`'s named premises, the check accepts a window and some drawn unit fails
its relation with probability at most `n_π · (δ_s + δ)`. -/
theorem serviceSound_of_sessions {C Ξ V : Type} [Fintype C] [Fintype Ξ] {m Np n nπ : ℕ} (p : PouwCheck.Params)
    (ws : Properties.Pouw.Period.Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (sessOf : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → Fin nπ)
    (sess : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → ℕ → ℕ → V)
    (δs δ : ℝ)
    (hCR : ∀ x w i, (sess x w i).CR)
    (hAcc : Accounting.Assumptions.TS.AcceptedLive (checkLayer p ws opened rel sessOf) sess out)
    (hLaw : Accounting.Assumptions.TS.WindowLaw sess out)
    (hChk : Accounting.Assumptions.TS.ServedChecksHold (checkLayer p ws opened rel sessOf) sess out rows)
    (hLink : Accounting.Assumptions.TS.ServedRootsLinked (checkLayer p ws opened rel sessOf) sess out rows δs)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    Properties.Pouw.Period.ServiceSound p ws opened rel (nπ * (δs + δ)) :=
  Accounting.SecurityProofs.TS.CheckProofSystemSound p ws opened rel sessOf (δs + δ)
    (Accounting.SecurityProofs.TS.SessionSoundAt (checkLayer p ws opened rel sessOf) sess out rows δs δ hCR hAcc hLaw
      hChk hLink hδ hδ0)

end Accounting.TS.WindowSessions

namespace Accounting.SecurityProofs.TS

open Pouw.RelativeSecurity Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs.HonestH1
open Accounting.TS.WindowSessions

/-- **The honest instance meets `CheckProofSystemSound`'s premise**: at H1's honest live-salt period of one window, one
session a window, the deployed check accepts the window on every path, and its layer is sound at `0`. -/
theorem CheckProofSystemSound_honest :
    (∀ x w, (checkLayer prm (windows 1) opened (stagedAt salt staged) (fun _ _ _ => (0 : Fin 1))).accepted x w 0) ∧
      Accounting.Assumptions.TS.ProofSoundAt
        (checkLayer prm (windows 1) opened (stagedAt salt staged) (fun _ _ _ => (0 : Fin 1))) 0 :=
  ⟨fun x w => accepted_all 1 x w,
    fun _ _ => le_of_eq
      (Accounting.SecurityProofs.ComposePeriod.Honest.prob_none fun _ h => h.2 fun _ _ _ => trivial)⟩

/-- **`CheckProofSystemSound`'s conclusion fails at `0`** when every opening fails its relation: at H1's honest period
of one window the deployed check accepts and publishes a drawn unit on every path. -/
theorem CheckProofSystemSound_nontrivial :
    ¬ ServiceSound prm (windows 1) opened (fun _ _ _ _ => False) 0 := by
  intro h
  haveI : Nonempty (Paths Ω1 1) := ⟨((Paths.const true _, (Equiv.refl _, ())), ())⟩
  refine absurd (h 0) (not_le.2 (lt_of_lt_of_le one_pos
    ((Accounting.SecurityProofs.FillerLuck.Cell.prob_all (Ω := Paths Ω1 1)).symm.le.trans
      (prob_mono fun x _ => ?_))))
  obtain ⟨pub, prof, hc⟩ := accepted_all 1 x 0
  obtain ⟨-, -, hd⟩ := drawIsCoin 1 x 0 pub prof hc
  exact ⟨pub, prof, hc, Accounting.Compose.drawOf hK24 (Accounting.TS.coinAt x 0).2 ⟨0, by decide⟩,
    (hd _).2 ⟨⟨0, by decide⟩, rfl⟩, fun h => h⟩

end Accounting.SecurityProofs.TS
