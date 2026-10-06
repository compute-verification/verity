import Proofs.Flock.Soundness.Audit.FlockLinked

/-!
# Public wires: the constant and the zero are the statement's, not the prover's

The link theorem's value layer (`Xplur`) is the plurality of a trial's readings at each commit string. On the constant
and on the statement's zero, that plurality is the prover's only in form. Every satisfying message of every drawn unit's
table decodes the constant to 1 (`UnitPlace.decode_one`) and the zero to 0 (its forced-zero rows). So the committed
transcript here is `Xpub ones zeros vals`: 1 on `ones`, the gates the units' constants read; 0 on `zeros`, the statement's
zero; and `vals` elsewhere.

* `linkEvB_of_pub`: at a drawn unit, when the public wires decode to their values in every satisfying message of its
  table (`PubDecodes`), the link event at `Xpub ones zeros vals` is the link event at `vals`. So the link theorem's bound holds
  unchanged for `Xpub ones zeros Xplur` (`flock_batched_linkSoundE_pub`), and with it the count curve and the drawn-unit
  bound (`flock_batched_count_public`, `flock_batched_drawn_public`).
* `pubDecodes_of_place`: `PubDecodes` for the decoder that reads each drawn unit from its place. Two facts about the
  statement suffice:
  - `ConstCols`: only a unit's constant column sits on a gate of `ones`. No input column of a unit reads a constant,
    which is a fact of the circuit's wiring.
  - `ZeroCols`: every column on a gate of `zeros` carries 0 in every satisfying witness. These are the forced-zero rows,
    which the verifier's Δ copies to the slot inputs that read the zero.

This discharges the end-to-end theorem's `hOne` and the zero's binding (`E2E.lean`): no hypothesis on the prover's
committed values remains in their place, only these two facts about the statement.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The committed transcript with its public wires**: 1 on `ones` (the gates the units' constants read), 0 on `zeros`
(the statement's zero) and `vals` elsewhere. -/
noncomputable def Xpub {N : ℕ} (ones zeros : Finset (Fin N)) (vals : Fin N → Bool) : Fin N → Bool :=
  fun g => if g ∈ ones then true else if g ∈ zeros then false else vals g

theorem Xpub_of_mem_ones {N : ℕ} {ones zeros : Finset (Fin N)} {vals : Fin N → Bool} {g : Fin N} (h : g ∈ ones) :
    Xpub ones zeros vals g = true := by
  simp [Xpub, h]

theorem Xpub_of_mem_zeros {N : ℕ} {ones zeros : Finset (Fin N)} {vals : Fin N → Bool} {g : Fin N} (h1 : g ∉ ones)
    (h0 : g ∈ zeros) : Xpub ones zeros vals g = false := by
  simp [Xpub, h1, h0]

theorem Xpub_of_not_mem {N : ℕ} {ones zeros : Finset (Fin N)} {vals : Fin N → Bool} {g : Fin N} (h1 : g ∉ ones)
    (h0 : g ∉ zeros) : Xpub ones zeros vals g = vals g := by
  simp [Xpub, h1, h0]

namespace Partition.IsRowsUnit

variable {C : Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {R : Rows} (I : P.IsRowsUnit u R)

/-- **Every committed wire of a rows unit is one of its columns**: a gate of the unit is a computed row's, and a gate its
rows read is a column's. -/
theorem exists_wire_of_mem_io {g : Fin C.N} (hg : g ∈ P.io u) : ∃ c, I.wire c = g := by
  classical
  simp only [io, Finset.mem_filter] at hg
  rcases hg.2 with hu | ⟨h, hh, hgh⟩
  · obtain ⟨i, hi⟩ := I.onto g hu
    exact ⟨_, hi⟩
  · obtain ⟨i, rfl⟩ := I.onto h hh
    rw [I.op i] at hgh
    simp only [Op.args, List.mem_append, List.mem_map] at hgh
    rcases hgh with ⟨c, -, rfl⟩ | ⟨c, -, rfl⟩ <;> exact ⟨c, rfl⟩

end Partition.IsRowsUnit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The public wires decode to their values**: every message packing a satisfying witness of a drawn unit's table
decodes, on the unit's committed wires, each gate of `ones` to 1 and each gate of `zeros` outside `ones` to 0. -/
def PubDecodes (decode : DecoderB (C := C) (Reg := Reg) plan tab) (ones zeros : Finset (Fin C.N)) : Prop :=
  ∀ S R u, u ∈ S → ∀ M : Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols,
    (tabOf plan tab S R u).S.Satisfies
      (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
        (tabOf plan tab S R u).sch.level₀.logCols M) →
    ∀ g ∈ P.io u, (g ∈ ones → decode S R u M g = true) ∧ (g ∉ ones → g ∈ zeros → decode S R u M g = false)

/-- **The link event at the public transcript is the link event at `vals`**, at a drawn unit whose table's satisfying
messages decode the public wires to their values: a wire that decodes differently from `Xpub ones zeros vals` is not a
public wire, so it decodes differently from `vals`. -/
theorem linkEvB_of_pub {decode : DecoderB (C := C) (Reg := Reg) plan tab} {ones zeros : Finset (Fin C.N)}
    (hpub : P.PubDecodes plan tab decode ones zeros) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    {vals : Fin C.N → Bool} {T : Oracle 𝔽} (h : P.LinkEvB plan tab decode S R u (Xpub ones zeros vals) T) :
    P.LinkEvB plan tab decode S R u vals T := by
  refine ⟨h.1, fun M hM hsat => ?_⟩
  obtain ⟨g, hg, hne⟩ := h.2 M hM hsat
  refine ⟨g, hg, fun heq => hne ?_⟩
  obtain ⟨h1, h0⟩ := hpub S R u hu M hsat g hg
  by_cases hg1 : g ∈ ones
  · rw [Xpub_of_mem_ones hg1]; exact h1 hg1
  · by_cases hg0 : g ∈ zeros
    · rw [Xpub_of_mem_zeros hg1 hg0]; exact h0 hg1 hg0
    · rw [Xpub_of_not_mem hg1 hg0]; exact heq

/-- **The link term is monotone in the link event**: if at every drawn unit the link event at `Xc'` implies the one at
`Xc`, every bound on the link term at `Xc` bounds it at `Xc'`. -/
theorem linkSoundE_mono {decode : DecoderB (C := C) (Reg := Reg) plan tab}
    {Xc Xc' : (R : Reg) → Cont L Reg (batchedSession H E plan) R → (Fin C.N → Bool)} {k : ℕ} (hk : 1 ≤ k)
    (hr : ∀ S R j, rateB (plan S R) j k < 1) (hlow : P.LoweringSoundB plan tab decode)
    (hsub : ∀ R τ S u, u ∈ S → ∀ T, P.LinkEvB plan tab decode S R u (Xc' R τ) T →
      P.LinkEvB plan tab decode S R u (Xc R τ) T)
    {δlink : (R : Reg) → Cont L Reg (batchedSession H E plan) R → ℝ≥0∞}
    (h : (P.analysisBE H E plan tab decode Xc k hk hr hlow).LinkSound δlink) :
    (P.analysisBE H E plan tab decode Xc' k hk hr hlow).LinkSound δlink := by
  classical
  intro R τ tgt
  refine le_trans (avg_mono fun ω => ?_) (h R τ tgt)
  cases htg : tgt ω with
  | none => simp only [atTgt, Option.elim_none, le_refl]
  | some u =>
    simp only [atTgt, Option.elim_some]
    split_ifs with hu
    · show ENNReal.ofReal (accProb H E plan (τ ω) *
          extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStrat H E plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc' R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤
        ENNReal.ofReal (accProb H E plan (τ ω) *
          extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k (batchStrat H E plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (Xc R τ)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k))
      apply ENNReal.ofReal_le_ofReal
      have hr1 : 0 ≤ 1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k := by
        linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
      refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left ?_ (expect_nonneg (ind_nonneg _) _ _)) hr1
      unfold extractProbAccB
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

/-- **The link theorem at the public transcript**: `flock_batched_linkSoundE`'s bound, prover by prover
(`linkBoundCR`), holds for `Xpub ones zeros Xplur` when the public wires decode to their values. -/
theorem flock_batched_linkSoundE_pub (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode) {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ}
    (ht : 0 ≤ t') :
    (P.analysisBE (L := L) H E plan tab decode (fun R τ => Xpub ones zeros (P.Xplur H E plan tab vb R τ k Rw)) k hk
        (fun S R j => (hr S R j).trans_lt hρ) hlow).LinkSound
      (P.linkBoundCR H E plan tab vb k Rw M ρ t') :=
  P.linkSoundE_mono H E plan tab hk _ hlow (fun _ _ _ _ hu _ h => P.linkEvB_of_pub plan tab hpub hu h)
    (P.flock_batched_linkSoundE H E plan tab vb hlow hk hRw hM hρ hr ht)

/-- **The compiled-layer count curve at the public transcript**: `flock_batched_count_linked` with the committed
transcript 1 on `ones`, 0 on `zeros` and the plurality elsewhere, when the public wires decode to their values. -/
theorem flock_batched_count_public (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_countE H E plan tab decode (fun R τ => Xpub ones zeros (P.Xplur H E plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE_pub H E plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

/-- **Drawn units at the public transcript**: `flock_batched_drawn_linked` with the committed transcript 1 on `ones`,
0 on `zeros` and the plurality elsewhere, when the public wires decode to their values. -/
theorem flock_batched_drawn_public (ones zeros : Finset (Fin C.N)) (hpub : P.PubDecodes plan tab decode ones zeros)
    (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧
        ¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_drawnE H E plan tab decode (fun R τ => Xpub ones zeros (P.Xplur H E plan tab vb R τ k Rw)) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE_pub H E plan tab vb ones zeros hpub hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

/-! ## The public wires from the drawn units' places -/

/-- **Only a unit's constant column sits on a gate of `ones`**, at every drawn unit's place: no input column of a unit
reads a constant, and a unit's constant is outside it. A fact of the circuit's wiring. -/
def ConstCols (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (ones : Finset (Fin C.N)) : Prop :=
  ∀ S R u (h : u ∈ S) (c : Fin (place S R u h).R.w), (place S R u h).inst.wire c ∈ ones →
    c = (place S R u h).R.one

/-- **`ConstCols` for one constant gate**: when every drawn unit's constant column sits on the gate `one`, no other column
does, since an instance shares a gate only between input columns. -/
theorem constCols_singleton (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (one : Fin C.N)
    (h : ∀ S R u (hu : u ∈ S), (place S R u hu).inst.wire (place S R u hu).R.one = one) :
    P.ConstCols plan tab place {one} := by
  intro S R u hu c hc
  rw [Finset.mem_singleton, ← h S R u hu] at hc
  rcases (place S R u hu).inst.shared _ _ hc with he | ⟨_, h2⟩
  · exact he
  · simp only [Rows.one] at h2
    omega

/-- **The zero's columns carry 0**, at every drawn unit's place: a column on a gate of `zeros` carries 0 in every witness
satisfying the table's statement. These are the forced-zero rows, which the verifier's Δ copies to the slot inputs that
read the zero. -/
def ZeroCols (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) (zeros : Finset (Fin C.N)) : Prop :=
  ∀ S R u (h : u ∈ S) (c : Fin (place S R u h).R.w), (place S R u h).inst.wire c ∈ zeros →
    ∀ z : Witness (tabOf plan tab S R u).S.m, (tabOf plan tab S R u).S.Satisfies z →
      (tabOf plan tab S R u).S.block z (place S R u h).o ((place S R u h).col c) = 0

/-- **The public wires decode to their values from the drawn units' places**: a committed wire of a drawn unit is one of
its columns; on a gate of `ones` that column is the constant's, which a satisfying witness holds at 1, and on a gate of
`zeros` it carries 0. -/
theorem pubDecodes_of_place (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u)
    {ones zeros : Finset (Fin C.N)} (hc : P.ConstCols plan tab place ones) (hz : P.ZeroCols plan tab place zeros) :
    P.PubDecodes plan tab (P.placeDecoder plan tab place) ones zeros := by
  intro S R u hu M hsat g hg
  have hdec : P.placeDecoder plan tab place S R u M = (place S R u hu).decode
      (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
        (tabOf plan tab S R u).sch.level₀.logCols M) := by
    unfold placeDecoder; rw [dite_eq_left_of_eq_true (eq_true hu)]
  rw [hdec]
  have hc' := hc S R u hu
  have hz' := hz S R u hu
  generalize witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
    (tabOf plan tab S R u).sch.level₀.logCols M = z at hsat ⊢
  generalize place S R u hu = pl at hc' hz' ⊢
  obtain ⟨c, rfl⟩ := pl.inst.exists_wire_of_mem_io hg
  have hval : pl.decode z (pl.inst.wire c) = bit ((tabOf plan tab S R u).S.block z pl.o (pl.col c)) := by
    simp only [UnitPlace.decode]
    exact pl.inst.decode_wire (pl.agree hsat) c
  rw [hval]
  refine ⟨fun h1 => ?_, fun _ h0 => ?_⟩
  · rw [hc' c h1, pl.placed.one_eq hsat pl.o]
    rfl
  · rw [hz' c h0 z hsat]
    rfl

end Partition

end FlockSoundness.Audit
