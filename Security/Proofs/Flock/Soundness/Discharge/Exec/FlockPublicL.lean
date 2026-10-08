import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkedL
import Proofs.Flock.Soundness.Audit.FlockPublic

/-!
# Public wires: the constant and the zero are the statement's, not the prover's

The link theorem's value layer (`XplurL`) is the plurality of a trial's readings at each commit string. On the constant
and on the statement's zero, that plurality is the prover's only in form. Every satisfying message of every drawn unit's
table decodes the constant to 1 (`UnitPlace.decode_one`) and the zero to 0 (its forced-zero rows). So the committed
transcript here is `Xpub ones zeros vals`: 1 on `ones`, the gates the units' constants read; 0 on `zeros`, the statement's
zero; and `vals` elsewhere.

* `linkEvB_of_pub`: at a drawn unit, when the public wires decode to their values in every satisfying message of its
  table (`PubDecodes`), the link event at `Xpub ones zeros vals` is the link event at `vals`. So the link theorem's bound holds
  unchanged for `Xpub ones zeros XplurL` (`flock_batched_linkSoundE_pubL`), and with it the count curve and the drawn-unit
  bound (`flock_batched_count_publicL`, `flock_batched_drawn_publicL`).
* `pubDecodes_of_place`: `PubDecodes` for the decoder that reads each drawn unit from its place. Two facts about the
  statement suffice:
  - `ConstCols`: only a unit's constant column sits on a gate of `ones`. No input column of a unit reads a constant,
    which is a fact of the circuit's wiring.
  - `ZeroCols`: every column on a gate of `zeros` carries 0 in every satisfying witness. These are the forced-zero rows,
    which the verifier's Δ copies to the slot inputs that read the zero.

This discharges the end-to-end theorem's `hOne` and the zero's binding (`E2E.lean`): no hypothesis on the prover's
committed values remains in their place, only these two facts about the statement.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The link term is monotone in the link event**: if at every drawn unit the link event at `Xc'` implies the one at
`Xc`, every bound on the link term at `Xc` bounds it at `Xc'`. -/
theorem linkSoundE_monoL {decode : DecoderB (C := C) (Reg := Reg) plan tab}
    {Xc Xc' : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)} {k : ℕ} (hk : 1 ≤ k)
    (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    (hsub : ∀ R τ S u, u ∈ S → ∀ T, P.LinkEvB plan tab decode S R u (Xc' R τ) T →
      P.LinkEvB plan tab decode S R u (Xc R τ) T)
    {δlink : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (h : (analysisBEL P H E Lf plan tab decode Xc k hk hr hlow).LinkSound δlink) :
    (analysisBEL P H E Lf plan tab decode Xc' k hk hr hlow).LinkSound δlink := by
  classical
  intro R τ tgt
  refine le_trans (avg_mono fun ω => ?_) (h R τ tgt)
  cases htg : tgt ω with
  | none => simp only [atTgt, Option.elim_none, le_refl]
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · show ENNReal.ofReal (accProbL H E Lf plan (τ ω) *
          extractProbAccBL execArith H E Lf (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStratL H E Lf plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc' R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤
        ENNReal.ofReal (accProbL H E Lf plan (τ ω) *
          extractProbAccBL execArith H E Lf (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStratL H E Lf plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k))
      apply ENNReal.ofReal_le_ofReal
      have hr1 : 0 ≤ 1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k := by
        linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
      refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left ?_ (expect_nonneg (ind_nonneg _) _ _)) hr1
      unfold extractProbAccBL
      refine expectAcc_mono _ _ _ _ fun bs => ?_
      unfold ind
      split_ifs with h1 h2
      · exact le_refl _
      · exact absurd (hsub R τ _ u hu _ h1) h2
      · exact zero_le_one
      · exact le_refl _
    · exact le_refl _

variable {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

/-- **The link theorem at the public transcript**: `flock_batched_linkSoundEL`'s bound, prover by prover
(`linkBoundCRL`), holds for `Xpub ones zeros XplurL` when the public wires decode to their values. -/
theorem flock_batched_linkSoundE_pubL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode) {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ}
    (ht : 0 ≤ t') :
    (analysisBEL P (L := L) H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) k hk
        (fun S R j => (hr S R j).trans_lt hρ) hlow).LinkSound
      (linkBoundCRL P H E Lf plan tab vb k Rw M ρ t') :=
  linkSoundE_monoL P H E Lf plan tab hk _ hlow (fun _ _ _ _ hu _ h => P.linkEvB_of_pub plan tab hpub hu h)
    (flock_batched_linkSoundEL P H E Lf plan tab vb hlow hk hRw hM hρ hr ht)

/-- **The compiled-layer count curve at the public transcript**: `flock_batched_count_linkedL` with the committed
transcript 1 on `ones`, 0 on `zeros` and the plurality elsewhere, when the public wires decode to their values. -/
theorem flock_batched_count_publicL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        K ≤ (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_countEL P H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundE_pubL P H E Lf plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

/-- **Drawn units at the public transcript**: `flock_batched_drawn_linkedL` with the committed transcript 1 on `ones`,
0 on `zeros` and the plurality elsewhere, when the public wires decode to their values. -/
theorem flock_batched_drawn_publicL (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_drawnEL P H E Lf plan tab decode (fun R τ => Xpub ones zeros (XplurL P H E Lf plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundE_pubL P H E Lf plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

end FlockSoundness.Discharge.Exec
