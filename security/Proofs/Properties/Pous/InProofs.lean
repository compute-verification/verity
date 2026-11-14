module

public import Properties.Pous.InProofs
public import Proofs.Properties.Accounting.InProofs
public import Proofs.Properties.Pous.MemoryChallenger

/-!
# Proof of PoUS's session inside proofs (`Properties.Pous.InProofs`)

When every proof block's value meets its piece (`PieceOk`), the carried rows open to one value and the verifier's clear
checks pass, the session the run opens meets every hypothesis of the converse of `VerdictAccepts`
(`verdict_accepted_of`), so `verdict` accepts it. So an accepted run in the event, with no collision, has a proof block
whose value fails `PieceOk`, and `Gives` at `right` gives it at `PieceOk` (`Gives.mono`).
-/

@[expose] public section

namespace MemoryChallenger.Properties.InProofs.SecurityProofs

open MemoryChallenger MemoryChallenger.Properties.InProofs MemoryChallenger.Properties.SecurityProofs
open Verity.InProofs Properties.Accounting.InProofs Verity.Game

theorem sessionGives : SessionGives := by
  intro schemes trusted expected c openings scheme pk idx Ω μ r right ε g h hc
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  classical
  obtain ⟨ha, ⟨⟨p, hp⟩, ht, hco⟩, hrej, hn⟩ := hω
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro hall
  have hok : ∀ u, PieceOk schemes scheme pk c expected idx u (r.ports u ω) := fun u => by
    by_contra hu
    exact hall u (by simp [Run.failing, hu])
  obtain ⟨hm, hrd⟩ := (hc ω ha).resolve_right hn
  have h0 := hok ⟨0, by omega⟩
  have h1 := hok ⟨1, by omega⟩
  simp only [PieceOk, ↓reduceIte] at h0
  simp only [PieceOk, one_ne_zero, ↓reduceIte] at h1
  obtain ⟨hmsg, he, hsz, htime⟩ := h1
  set sd := r.ports ⟨1, by omega⟩ ω
  set sg := r.ports ⟨0, by omega⟩ ω
  have hround : ∀ j (hj : j < sd.rounds.size), sd.rounds[j] = (r.ports ⟨j + 2, by omega⟩ ω).round := fun j hj => by
    have := hrd j (by omega)
    rw [Array.getElem?_eq_getElem hj] at this
    exact Option.some.inj this
  rw [Bool.eq_false_iff] at hrej
  refine hrej (verdict_accepted_of schemes trusted expected _ c openings p hp ht sd.m hmsg ?_ rfl he hsz ?_)
  · show lmsVerify p pk c.session sd.m sg.sig = .ok true
    rw [← hm]
    exact h0 p hp
  · intro j hj
    have hj' : j < c.k := by simp only [sessionOf] at hj; omega
    obtain ⟨o, ho, hoi, hvc⟩ := hco j hj'
    have hpj := hok ⟨j + 2, by omega⟩
    simp only [PieceOk, show j + 2 ≠ 0 by omega, show j + 2 ≠ 1 by omega, ↓reduceIte, Nat.add_sub_cancel] at hpj
    obtain ⟨hrj, hri, hb, hd⟩ := hpj
    simp only [sessionOf] at hj ⊢
    simp only [hround j hj]
    have ht := htime j hj
    simp only [hround j hj] at ht
    exact ⟨⟨o, ho, hrj, by rw [hoi, hri], hvc⟩, hb, hd, ht⟩

/-- The values of a session that ended nowhere: no digest, no signature, no rounds. -/
def noSession : Vals := ⟨.empty, .empty, "", #[], default⟩

theorem sessionGives_nontrivial : SessionGivesNontrivial := by
  refine ⟨default, .empty, fun _ => noSession, fun h => ?_⟩
  have hclear : ClearOk SCHEMES [ByteArray.empty] LMS .empty default #[] (fun _ => 0) :=
    ⟨⟨_, rfl⟩, List.mem_singleton.2 rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  have hrej : (verdict SCHEMES [ByteArray.empty] (fun _ => none)
      (sessionOf default LMS .empty noSession noSession) default #[]).accepted = false := by
    rw [Bool.eq_false_iff]
    intro ha
    obtain ⟨-, -, he, -⟩ := verdictAccepts _ _ _ _ _ _ ha
    exact absurd he (by decide)
  have hn : ¬ Hash.CollisionIn Verity.sha512 [] := by simp [Hash.CollisionIn]
  simp only [prPMF, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq] at h
  simp only [sessionOf] at hrej
  simp only [constRun] at h
  simp only [hclear, hrej, hn, sessionOf, and_true, not_false_eq_true, ↓reduceIte] at h
  exact absurd h (by simp)

end MemoryChallenger.Properties.InProofs.SecurityProofs
