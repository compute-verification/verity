module

public import Proofs.Flock.Soundness.Registered.Defs
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Binding.Weights
import Proofs.Flock.Soundness.Game

/-! Registered reads bind the registrant's rows under `cr/sha-512`: accepting while a registered read's leaf is not
the registrant's has probability at most `qR²/2^513` (`Reads.prob_off_le`). -/

@[expose] public section

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

theorem collides_bytes (x : ByteArray × ByteArray) (h : Flock.Registered.Collides x) :
    Collides (x.1.data.toList, x.2.data.toList) :=
  ⟨fun e => h.1 (ByteArray.ext (Array.ext' e)), Subtype.ext (by simp only [H512_val, bytes_of_list]; exact h.2)⟩

theorem TwoOpenings.collides (o : TwoOpenings) (h : o.Conflict) : Collides o.pair := by
  obtain ⟨hne, h1, h2⟩ := h
  simp only [TwoOpenings.pair]
  exact collides_bytes _ (Flock.Registered.opensLeaf_binding o.port o.pos o.leaf₁.1 o.path₁ o.leaf₂.1 o.path₂ o.leaf₁.2
    o.leaf₂.2 (fun e => hne (Subtype.ext e)) h1 h2)

theorem TwoOpenings.isColl (o : TwoOpenings) (h : o.Conflict) : StrictCR.IsColl H512 (some o.pair) :=
  ⟨o.pair.1, o.pair.2, congrArg some Prod.mk.eta.symm, o.collides h⟩

theorem TwoOpenings.conflict_strict {β : Type} (g : Game β) (s : Strategy g) (o : β → TwoOpenings) {cost : β → ℝ}
    {q : ℝ} (hcost : ∀ b, cost b ≤ q)
    (hCR : Assumptions.SHA512CRStrict H512 g s (fun b => some (o b).pair) cost q) :
    (prob (fun b => (o b).Conflict) g s).toReal ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (prob_mono (fun b hb => (o b).isColl hb) g s)).trans (hCR hcost)

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Reads

variable {lay : Layout P plan tab decode} {rs : HmRows lay} (rd : Reads lay rs)

theorem regLeaf_eq {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) {p : lay.Pos} (hp : rd.isRead p) :
    lay.regLeaf R p = rd.leaf p ∨ Collides (rd.openings R path p).pair := by
  by_cases h : lay.regLeaf R p = rd.leaf p
  · exact Or.inl h
  · exact Or.inr ((rd.openings R path p).collides ⟨h, hc p hp, rd.committed p hp⟩)

/-- The row the witness holds at a registered read is the row the registrant committed (`wv`), or SHA-512 collides in
hm96's leaf (`hm96Pair`) or between the session's opening of the registered root and the registrant's. -/
theorem registered_weights (hHm : rs.Computes) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    (M : Msg plan tab S R u) (hM : Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u)
    (hrd : rd.isRead (lay.pos g)) {path : lay.Pos → ByteArray} (hc : rd.Checked R path) :
    lay.row S R u M (lay.pos g) = rd.wv (lay.pos g) ∨
      Collides (hm96Pair (rs.enc (lay.row S R u M (lay.pos g))) (lay.salt S R u M (lay.pos g))
        (rs.enc (rd.wv (lay.pos g))) (rd.ws (lay.pos g))) ∨
      Collides (rd.openings R path (lay.pos g)).pair := by
  rcases rd.regLeaf_eq hc hrd with h | h
  · exact (rs.registered_weights_hm96 hHm hu M hM hg (rd.read_row _ hrd) (rd.wv _) (rd.ws _) h.symm).imp_right Or.inl
  · exact Or.inr (Or.inr h)

theorem offAt_conflict {R : Reg} {path : lay.Pos → ByteArray} (hc : rd.Checked R path) (h : rd.Off R) :
    (rd.openings R path (rd.offAt R)).Conflict := by
  have hs : rd.isRead (rd.offAt R) ∧ lay.regLeaf R (rd.offAt R) ≠ rd.leaf (rd.offAt R) := Classical.epsilon_spec h
  exact ⟨hs.2, hc _ hs.1, rd.committed _ hs.1⟩

theorem prob_off_le {β : Type} (G : Game β) (σ : Strategy G) (R : Reg) (accept : β → Prop)
    (path : β → lay.Pos → ByteArray) (hchk : ∀ o, accept o → rd.Checked R (path o)) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o => rd.pick R (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ rd.Off R) G σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : prob (fun o => accept o ∧ rd.Off R) G σ ≤ prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ :=
    prob_mono (fun o ⟨ha, hoff⟩ => (rd.openings R (path o) (rd.offAt R)).isColl (rd.offAt_conflict (hchk o ha) hoff)) G σ
  have hev : (prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ).toReal ≤ qR ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    have := StrictCR.Finder.CR.le hR
    rwa [StrictCR.oneRun_ev] at this
  have hfin : prob (fun o => StrictCR.IsColl H512 (rd.pick R (path o))) G σ ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  exact hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 hev)

end Reads

end FlockSoundness.Registered
