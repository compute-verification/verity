import Proofs.Accounting.TS.Session.Lanes.Guarantees
import Proofs.Accounting.TS.CheckedSound
import Proofs.Pouw.RelativeSecurity

/-!
# A leaf's 33 linked sessions realize one unit (the proof)

`Accounting.SecurityProofs.TS.LeafLanesSound`. The lanes cover the leaf (`Accounting.TS.Lanes.cover`: cell `(i, j)` is
lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and column slot `2⌊j/8⌋ + (j mod 2)`), so a leaf whose 33
statements each hold at their committed rows is served right (`unitServed_of`): `LeafChecks` and `LeafShared` give each
lane's `LaneOk` and the combine's `CombineOk` at the one leaf and partials record. An unsound proof of the leaf layer is
then, at one of its 33 sessions, an unsound proof of the checked layer (`CheckedSessionSound`'s `SoundAt δ`) or an
accepted outcome that doesn't draw every unit, which under `kd = n` has probability `0` in every session's game
(`RowsSession.prob_undrawn`) and so on the window (`WindowLaw`). The union bound over the 33 sessions gives `33 δ`.

This module imports `Proofs.Flock.SoundAtBridge` (through `Proofs.Accounting.TS.CheckedSound`), whose imports make `X`
a token, so `Proofs/Accounting.lean` imports it.
-/

namespace Flock.SoundAtBridge.RowsSession

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkHidden

variable (s : RowsSession)

/-- **Under full checking no outcome of the session's game misses a unit.** -/
theorem prob_undrawn (hk : s.kd = s.n) : prob (fun o : s.Out => o.2.1 ≠ Finset.univ) s.G s.τ = 0 :=
  prob_audit_eq_zero s.L s.Reg (liveOfZJ s.dg (planZHJ s.dj s.y₀) s.N s.wr) (fun o => o.2.1 ≠ Finset.univ)
    (fun _ ω _ => nonpos_iff_eq_zero.1
      ((prob_mono (fun _ ht => ht (s.draw_full hk ω)) _ _).trans (prob_false _ _).le)) s.τ

end Flock.SoundAtBridge.RowsSession

namespace Accounting.TS.Lanes

open Accounting.TS.Checked Flock.SoundAtBridge

/-- **The 32 lanes cover the leaf**: cell `(i, j)` is lane `4 (i mod 8) + ⌊(j mod 8)/2⌋`'s at row slot `⌊i/8⌋` and
column slot `2⌊j/8⌋ + (j mod 2)`. -/
theorem cover (i j : Fin 32) : ∃ l s t, rowOf l s = i ∧ colOf l t = j := by
  have hi := i.isLt
  have hj := j.isLt
  refine ⟨⟨4 * (i.val % 8) + (j.val % 8) / 2, by omega⟩, ⟨i.val / 8, by omega⟩, ⟨2 * (j.val / 8) + j.val % 2, by omega⟩,
    Fin.ext ?_, Fin.ext ?_⟩
  · show (4 * (i.val % 8) + (j.val % 8) / 2) / 4 + 8 * (i.val / 8) = i.val
    omega
  · show 2 * ((4 * (i.val % 8) + (j.val % 8) / 2) % 4) + 8 * ((2 * (j.val / 8) + j.val % 2) / 2) +
      (2 * (j.val / 8) + j.val % 2) % 2 = j.val
    omega

/-- **A leaf whose 33 statements hold at their committed rows is served right**: where each session's committed rows
are the registered reads of a wire assignment its evaluation agrees with at every unit, `LeafChecks` and the sessions'
decodes (`LeafShared` at one path) give every lane's `LaneOk` and the combine's `CombineOk` at the one leaf `L` and
partials record `parts`, and the lanes cover the leaf. -/
theorem unitServed_of {K nw : ℕ} {Pub : Type} {dq : List ℤ → ℤ}
    {part : Pub → Fin 32 → (Fin 4 → Row K) → (Fin 8 → Row K) → Fin nw → BitVec 32}
    {sess : Option (Fin 32) → RowsSession} {decL : ∀ s : RowsSession, (ℕ → ℕ → Option Bool) → LaneView K nw Pub}
    {decC : ∀ s : RowsSession, (ℕ → ℕ → Option Bool) → CombineView nw} {pub : Pub}
    {parts : Fin 32 → Fin nw → BitVec 32} {receipt : Fin nw → BitVec 32} {L : Leaf K}
    (hC : Accounting.Assumptions.TS.LeafChecks dq part sess decL decC)
    (hL : ∀ l, decL (sess (some l)) (openedReg (sess (some l))) = laneView L pub (parts l) l)
    (hCb : decC (sess none) (openedReg (sess none)) = ⟨parts, receipt, true⟩)
    (hV : ∀ k, ∃ vals : Fin (sess k).Circ.N → Bool,
      openedReg (sess k) = rowsReg (sess k) vals ∧ ∀ u, ∀ g ∈ (sess k).P.io u, (sess k).Circ.eval vals g = vals g) :
    UnitServed dq part pub receipt L := by
  have hlane : ∀ l, LaneOk dq part (laneView L pub (parts l) l) := by
    intro l
    obtain ⟨vals, hv, he⟩ := hV (some l)
    have hd : decL (sess (some l)) (rowsReg (sess (some l)) vals) = laneView L pub (parts l) l := by
      rw [← hv]; exact hL l
    have h := hC.1 l vals he ((congrArg LaneView.ok hd).trans rfl)
    rw [hd] at h
    exact h
  have hcomb : CombineOk (⟨parts, receipt, true⟩ : CombineView nw) := by
    obtain ⟨vals, hv, he⟩ := hV none
    have hd : decC (sess none) (rowsReg (sess none) vals) = ⟨parts, receipt, true⟩ := by
      rw [← hv]; exact hCb
    have h := hC.2 vals he ((congrArg CombineView.ok hd).trans rfl)
    rw [hd] at h
    exact h
  refine ⟨fun i j => ?_, ?_⟩
  · obtain ⟨l, s, t, rfl, rfl⟩ := cover i j
    exact (hlane l).2 s t
  · have hp : parts = fun l => part pub l (laneA L l) (laneB L l) := funext fun l => (hlane l).1
    exact hcomb.symm.trans (congrArg xorAll hp)

end Accounting.TS.Lanes

open Accounting.TS Accounting.TS.Checked Accounting.TS.Lanes Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg in
/-- **A leaf's 33 linked C-Flock sessions realize one unit** (`Accounting.Guarantees.TS.LeafLanesSound`). -/
theorem Accounting.SecurityProofs.TS.LeafLanesSound : Accounting.Guarantees.TS.LeafLanesSound := by
  intro Ω _ S nπ K nw Pub sess out ok dq part decL decC pub parts receipt leaf right δ hCR hLaw hk hN hS hok hδ hδ0
    hC hSh
  have hSound : ∀ k, (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun _ _ => True)).SoundAt δ := fun k =>
    (Accounting.SecurityProofs.TS.CheckedSessionSound (fun x w i => sess x w i k) (fun x w i => out x w i k) ok
      (fun _ _ => True) δ (fun x w i => hCR x w i k) (hLaw k) (fun x w i => hk x w i k) (fun x w i => hN x w i k)
      (fun x w i => hS x w i k) (fun x w i => hok x w i k) (fun x w i => hδ x w i k) hδ0).2.2.2
  have hUndrawn : ∀ k (w : Fin S) (i : Fin nπ),
      Pouw.RelativeSecurity.prob (fun x => (out x w i k).2.1 ≠ Finset.univ) ≤ 0 := fun k w i =>
    (hLaw k w i fun s o => o.2.1 ≠ Finset.univ).trans (avg_le _ le_rfl fun x => le_of_eq
      ((congrArg ENNReal.toReal (RowsSession.prob_undrawn (sess x w i k) (hk x w i k))).trans ENNReal.toReal_zero))
  intro w i
  have hIn : ∀ x, (leafLayer sess out dq part pub receipt leaf right).Unsound x w i →
      ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
        (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
        (out x w i k).2.1 ≠ Finset.univ := by
    intro x ⟨hacc, hrel⟩
    by_contra hno
    push_neg at hno
    apply hrel
    refine unitServed_of (hC x w i) ((hSh x w i).1) ((hSh x w i).2) fun k => ?_
    obtain ⟨hk1, hk2⟩ := hno k (Finset.mem_univ _)
    have hR : CheckedRel (sess x w i k) (out x w i k) ok (openedReg (sess x w i k)) := by
      by_contra hR
      exact hk1 ⟨hacc k, hR⟩
    obtain ⟨-, vals, hv, he⟩ := hR
    exact ⟨vals, hv, fun u => he u (by rw [hk2]; exact Finset.mem_univ u)⟩
  calc Pouw.RelativeSecurity.prob (fun x => (leafLayer sess out dq part pub receipt leaf right).Unsound x w i)
      ≤ Pouw.RelativeSecurity.prob (fun x => ∃ k ∈ (Finset.univ : Finset (Option (Fin 32))),
          (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
          (out x w i k).2.1 ≠ Finset.univ) := Pouw.RelativeSecurity.prob_mono hIn
    _ ≤ ∑ k ∈ (Finset.univ : Finset (Option (Fin 32))), Pouw.RelativeSecurity.prob (fun x =>
          (checkedLayer (fun x w i => sess x w i k) (fun x w i => out x w i k) ok (fun _ _ => True)).Unsound x w i ∨
          (out x w i k).2.1 ≠ Finset.univ) := Pouw.RelativeSecurity.prob_exists_le _ _
    _ ≤ ∑ _k ∈ (Finset.univ : Finset (Option (Fin 32))), δ := Finset.sum_le_sum fun k _ =>
          (Pouw.RelativeSecurity.prob_or_le _ _).trans
            ((add_le_add (hSound k w i) (hUndrawn k w i)).trans (add_zero δ).le)
    _ = 33 * δ := by simp
