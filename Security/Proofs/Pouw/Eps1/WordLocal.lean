import Proofs.Pouw.Eps1.Basics

/-!
# Theorem B in M3: a word-local program's instructions

QSTAR's Theorem B (note:20261017T0344Z-draft-eps1-qstar, (b); gap list E6, E7: note:20261017T0801Z-draft-lean-stmt):
a word-local program costs at least 4 instructions a word at the zero instance and at least 8 at a general (int8)
instance.

Block `τ`'s live part `L` is its writer and the writer's ancestors inside the block. The writer's value is a function
of `L`'s input slots and the previous segment's output (`out_determined`). On the fiber of `τ = (ℓ, i, j)`, where only
segment `ℓ` of row `i` and column `j` vary and row `i`'s earlier segments are zero, that output is `0`, a row-side free
word is a function of the row's segment and a column-side one of the column. The writer determines the column (at a
general instance, the row's segment too), so at least 7 column-side slots and 1 row-side slot (8 and 8) feed `L`, and
`block_outside_slots` caps `L`'s outside slots at `2|L| + 1`.
-/

namespace Pouw.SecurityProofs.Eps1.WordLocal

open Pouw.Eps1 Pouw.SecurityProofs.Eps1 Finset

section Block

variable {Rd Cd : Type} {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}

/-- `a` is an argument of `b`, an instruction of block `τ`. -/
def ArgIn (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (τ : Tau m n Lseg) (a b : Fin G) : Prop :=
  blk b = τ ∧ ∃ j s, P.prog.arg b j = .inr s ∧ M3.earlier b s = a

open Classical in
/-- Block `τ`'s live part: its writer and the writer's ancestors inside the block. -/
noncomputable def live (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (τ : Tau m n Lseg) :
    Finset (Fin G) :=
  univ.filter fun t => blk t = τ ∧ Relation.ReflTransGen (ArgIn P blk τ) t (P.out τ)

theorem mem_live {P : Prog Rd Cd m n Lseg FR FU G} {blk : Fin G → Tau m n Lseg} {τ : Tau m n Lseg} {t : Fin G} :
    t ∈ live P blk τ ↔ blk t = τ ∧ Relation.ReflTransGen (ArgIn P blk τ) t (P.out τ) := by
  simp [live]

theorem out_mem_live {P : Prog Rd Cd m n Lseg FR FU G} {blk : Fin G → Tau m n Lseg} (hW : P.WordLocal blk)
    (τ : Tau m n Lseg) : P.out τ ∈ live P blk τ :=
  mem_live.mpr ⟨hW.1 τ, .refl⟩

theorem live_used {P : Prog Rd Cd m n Lseg FR FU G} {blk : Fin G → Tau m n Lseg} {τ : Tau m n Lseg} {t : Fin G}
    (ht : t ∈ live P blk τ) (hne : t ≠ P.out τ) :
    ∃ t' ∈ live P blk τ, ∃ j s, P.prog.arg t' j = .inr s ∧ M3.earlier t' s = t := by
  rcases (mem_live.mp ht).2.cases_head with h | ⟨c, ⟨hc, j, s, harg, he⟩, hcr⟩
  · exact absurd h hne
  · exact ⟨c, mem_live.mpr ⟨hc, hcr⟩, j, s, harg, he⟩

theorem live_closed {P : Prog Rd Cd m n Lseg FR FU G} {blk : Fin G → Tau m n Lseg} {τ : Tau m n Lseg} {t : Fin G}
    {j : Fin 3} {s : Fin t.val} (ht : t ∈ live P blk τ) (harg : P.prog.arg t j = .inr s)
    (hb : blk (M3.earlier t s) = τ) : M3.earlier t s ∈ live P blk τ :=
  mem_live.mpr ⟨hb, .head ⟨(mem_live.mp ht).1, j, s, harg, rfl⟩ (mem_live.mp ht).2⟩

theorem live_card_le (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (τ : Tau m n Lseg) :
    (live P blk τ).card ≤ (univ.filter fun t => blk t = τ).card :=
  card_le_card fun t ht => mem_filter.mpr ⟨mem_univ t, (mem_live.mp ht).1⟩

/-- The writer's value is a function of its live part's input slots and the previous segment's output. -/
theorem out_determined (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (hW : P.WordLocal blk)
    (τ : Tau m n Lseg) (x x' : Rd) (y y' : Cd) (v v' : Fin G → Word) (hv : P.prog.IsRun (P.inputs x y) v)
    (hv' : P.prog.IsRun (P.inputs x' y') v')
    (hprev : ∀ h : 0 < τ.1.val, v (P.out (τ.prev h)) = v' (P.out (τ.prev h)))
    (hr : ∀ t ∈ live P blk τ, ∀ j a, P.prog.arg t j = .inl (.inl a) → P.fr a x = P.fr a x')
    (hu : ∀ t ∈ live P blk τ, ∀ j b, P.prog.arg t j = .inl (.inr b) → P.fu b y = P.fu b y') :
    v (P.out τ) = v' (P.out τ) := by
  have hτ : ∀ (σ : Tau m n Lseg) (hσ : 0 < σ.1.val), σ = τ →
      v (P.out (σ.prev hσ)) = v' (P.out (σ.prev hσ)) := by
    rintro σ hσ rfl
    exact hprev hσ
  have key : ∀ k, ∀ t ∈ live P blk τ, t.val < k → v t = v' t := by
    intro k
    induction k with
    | zero => intro t _ h; omega
    | succ k ih =>
      intro t ht hk
      have harg : ∀ j, P.prog.argVal (P.inputs x y) v t j = P.prog.argVal (P.inputs x' y') v' t j := by
        intro j
        unfold M3.argVal
        split
        · rename_i c hc
          rcases c with a | b
          · exact hr t ht j a hc
          · exact hu t ht j b hc
        · rename_i s hs
          by_cases hl : M3.earlier t s ∈ live P blk τ
          · exact ih _ hl (by simp only [M3.earlier]; omega)
          · rcases hW.2 t j s hs with hb | ⟨h, he⟩
            · exact absurd (live_closed ht hs (hb.trans (mem_live.mp ht).1)) hl
            · rw [he]
              exact hτ _ h (mem_live.mp ht).1
      rw [hv t, hv' t, harg 0, harg 1, harg 2]
  exact key _ _ (out_mem_live hW τ) (Nat.lt_succ_self _)

/-- The live part's row-side and column-side input slots number at most `2|L| + 1` (`block_outside_slots`). -/
theorem slots_le (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) (hW : P.WordLocal blk)
    (τ : Tau m n Lseg) (Sr Su : Finset (Fin 3 × Fin G))
    (hSr : ∀ tj ∈ Sr, tj.2 ∈ live P blk τ ∧ ∃ a, P.prog.arg tj.2 tj.1 = .inl (.inl a))
    (hSu : ∀ tj ∈ Su, tj.2 ∈ live P blk τ ∧ ∃ b, P.prog.arg tj.2 tj.1 = .inl (.inr b)) :
    Sr.card + Su.card ≤ 2 * (live P blk τ).card + 1 := by
  classical
  refine le_trans ?_ (block_outside_slots P blk hW τ (live P blk τ) (fun t ht => (mem_live.mp ht).1)
    (out_mem_live hW τ) (fun t ht hne => live_used ht hne))
  rw [← card_union_of_disjoint]
  · apply card_le_card
    intro tj htj
    rw [mem_filter, Finset.product_eq_sprod, mem_product]
    rcases mem_union.mp htj with h | h
    · obtain ⟨hl, a, ha⟩ := hSr tj h
      exact ⟨⟨mem_univ _, hl⟩, fun s hs => by rw [ha] at hs; cases hs⟩
    · obtain ⟨hl, b, hb⟩ := hSu tj h
      exact ⟨⟨mem_univ _, hl⟩, fun s hs => by rw [hb] at hs; cases hs⟩
  · rw [disjoint_left]
    intro tj h1 h2
    obtain ⟨_, a, ha⟩ := hSr tj h1
    obtain ⟨_, b, hb⟩ := hSu tj h2
    rw [ha] at hb
    cases hb

/-- Blocks partition the instructions, so `c` instructions a block make `c·L·m·n` at least. -/
theorem count_blocks (blk : Fin G → Tau m n Lseg) (c : ℕ)
    (h : ∀ τ, c ≤ (univ.filter fun t => blk t = τ).card) : c * (Lseg * m * n) ≤ G := by
  have hG := card_eq_sum_card_fiberwise (f := blk) (s := univ) (t := univ) (fun _ _ => mem_coe.mpr (mem_univ _))
  rw [card_univ, Fintype.card_fin] at hG
  calc c * (Lseg * m * n) = ∑ _τ : Tau m n Lseg, c := by
        simp only [sum_const, card_univ, Fintype.card_prod, Fintype.card_fin, smul_eq_mul]; ring
    _ ≤ ∑ τ : Tau m n Lseg, (univ.filter fun t => blk t = τ).card := sum_le_sum fun τ _ => h τ
    _ = G := hG.symm

end Block

/-- A 32-bit word tells apart the integers in `[−2^31, 2^31)`. -/
theorem ofInt_inj {a b : ℤ} (ha : -2 ^ 31 ≤ a ∧ a < 2 ^ 31) (hb : -2 ^ 31 ≤ b ∧ b < 2 ^ 31)
    (h : BitVec.ofInt 32 a = BitVec.ofInt 32 b) : a = b := by
  have h' := congrArg BitVec.toInt h
  rwa [BitVec.toInt_ofInt, BitVec.toInt_ofInt, Int.bmod_eq_of_le (by norm_num; omega) (by norm_num; omega),
    Int.bmod_eq_of_le (by norm_num; omega) (by norm_num; omega)] at h'

section Zero

variable {m n Lseg : ℕ}

/-- The zero instance's fiber of `τ`: segment `τ.1` of row `τ.2.1` is `r`, every other dither bit is zero. -/
def rowF (τ : Tau m n Lseg) (r : Fin 32 → Bool) : Dither m Lseg :=
  fun i l p => if i = τ.2.1 ∧ l = τ.1 then r p else false

/-- Column `τ.2.2` is `u`, every other entry is `1`. -/
def colF (τ : Tau m n Lseg) (u : Fin 32 → ℤ) : Cols n :=
  fun j p => if j = τ.2.2 then u p else 1

theorem colF_valid (τ : Tau m n Lseg) (u : Fin 32 → ℤ) (hu : ∀ p, 1 ≤ |u p| ∧ |u p| ≤ 64) :
    ColsValid (colF τ u) := by
  intro j p
  unfold colF
  split_ifs
  · exact hu p
  · norm_num

theorem zeroSeg_rowF_ne (τ : Tau m n Lseg) (r : Fin 32 → Bool) (U : Cols n) (l : Fin Lseg) (i : Fin m)
    (j : Fin n) (hl : l ≠ τ.1) : zeroSeg (rowF τ r) U l i j = 0 := by
  simp [zeroSeg, rowF, hl]

theorem zeroPrefix_rowF (τ σ : Tau m n Lseg) (r : Fin 32 → Bool) (U : Cols n) (hσ : σ.1 ≤ τ.1) :
    zeroPrefix (rowF τ r) U σ = 0 :=
  sum_eq_zero fun l hl => zeroSeg_rowF_ne τ r U l _ _ (ne_of_lt (lt_of_lt_of_le (mem_filter.mp hl).2 hσ))

theorem zeroOut_fiber (τ : Tau m n Lseg) (r : Fin 32 → Bool) (u : Fin 32 → ℤ) :
    zeroOut (rowF τ r) (colF τ u) τ = BitVec.ofInt 32 (∑ p, if r p then u p else 0) := by
  rw [zeroOut, zeroPrefix_rowF τ τ r _ le_rfl, zero_add]
  simp [zeroSeg, rowF, colF]

theorem zeroOut_fiber_prev (τ : Tau m n Lseg) (r : Fin 32 → Bool) (U : Cols n) (h : 0 < τ.1.val) :
    zeroOut (rowF τ r) U (τ.prev h) = 0 := by
  have hle : (τ.prev h).1 ≤ τ.1 := by simp only [Tau.prev, Fin.le_def]; omega
  have hne : (τ.prev h).1 ≠ τ.1 := by simp only [Tau.prev, ne_eq, Fin.ext_iff]; omega
  rw [zeroOut, zeroPrefix_rowF τ _ r U hle, zeroSeg_rowF_ne τ r U _ _ _ hne]
  rfl

/-- **Theorem B's block bound, zero instance**: each block's live part has at least 4 instructions. -/
theorem block_zero {FR FU : Type} {G : ℕ} (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (blk : Fin G → Tau m n Lseg) (hC : P.Computes (fun _ U => ColsValid U) zeroOut) (hW : P.WordLocal blk)
    (τ : Tau m n Lseg) : 4 ≤ (live P blk τ).card := by
  classical
  let run : (Fin 32 → Bool) → (Fin 32 → ℤ) → Fin G → Word := fun r u =>
    (run_exists P.prog (P.inputs (rowF τ r) (colF τ u))).choose
  have hrun : ∀ r u, P.prog.IsRun (P.inputs (rowF τ r) (colF τ u)) (run r u) := fun r u =>
    (run_exists P.prog (P.inputs (rowF τ r) (colF τ u))).choose_spec
  have hout : ∀ r u, (∀ p, 1 ≤ |u p| ∧ |u p| ≤ 64) →
      run r u (P.out τ) = BitVec.ofInt 32 (∑ p, if r p then u p else 0) := fun r u hu => by
    rw [hC _ _ (colF_valid τ u hu) _ (hrun r u) τ, zeroOut_fiber]
  have hprev : ∀ r u, (∀ p, 1 ≤ |u p| ∧ |u p| ≤ 64) → ∀ h : 0 < τ.1.val, run r u (P.out (τ.prev h)) = 0 :=
    fun r u hu h => by rw [hC _ _ (colF_valid τ u hu) _ (hrun r u), zeroOut_fiber_prev]
  let Sr := ((univ : Finset (Fin 3)) ×ˢ live P blk τ).filter fun tj =>
    ∃ a, P.prog.arg tj.2 tj.1 = .inl (.inl a)
  let Su := ((univ : Finset (Fin 3)) ×ˢ live P blk τ).filter fun tj =>
    ∃ b, P.prog.arg tj.2 tj.1 = .inl (.inr b)
  have hS := slots_le P blk hW τ Sr Su
    (fun tj h => ⟨(mem_product.mp (mem_filter.mp h).1).2, (mem_filter.mp h).2⟩)
    (fun tj h => ⟨(mem_product.mp (mem_filter.mp h).1).2, (mem_filter.mp h).2⟩)
  -- the column side: the writer determines column `j`, so at least 7 column-side slots
  have hu7 : 7 ≤ Su.card := by
    let e := Su.equivFin
    let g : (Fin 32 → ℤ) → Fin 3 × Fin G → Word := fun u tj =>
      match P.prog.arg tj.2 tj.1 with
      | .inl c => P.inputs (rowF τ fun _ => false) (colF τ u) c
      | .inr _ => 0
    refine words_needed _ (fun u k => g u (e.symm k).1) fun u u' hu hu' hf => funext fun p => ?_
    let r : Fin 32 → Bool := fun q => decide (q = p)
    have hd := out_determined P blk hW τ (rowF τ r) (rowF τ r) (colF τ u) (colF τ u') _ _ (hrun r u) (hrun r u')
      (fun h => by rw [hprev r u hu h, hprev r u' hu' h]) (fun _ _ _ _ _ => rfl) (fun t ht j b hb => by
        have hmem : ((j, t) : Fin 3 × Fin G) ∈ Su := mem_filter.mpr ⟨mem_product.mpr ⟨mem_univ _, ht⟩, b, hb⟩
        have hk := congrFun hf (e ⟨_, hmem⟩)
        simp only [Equiv.symm_apply_apply, g, hb] at hk
        simpa [Prog.inputs] using hk)
    rw [hout r u hu, hout r u' hu'] at hd
    simp only [r, decide_eq_true_eq, sum_ite_eq', mem_univ, ite_true] at hd
    have h1 := abs_le.mp (hu p).2
    have h2 := abs_le.mp (hu' p).2
    exact ofInt_inj ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ hd
  -- the row side: the writer depends on the row's segment, so at least one row-side slot
  have hr1 : 1 ≤ Sr.card := by
    rw [Nat.one_le_iff_ne_zero, Ne, card_eq_zero]
    intro h0
    have hu1 : ∀ p : Fin 32, 1 ≤ |(1 : ℤ)| ∧ |(1 : ℤ)| ≤ 64 := fun _ => by norm_num
    have hd := out_determined P blk hW τ (rowF τ fun _ => false) (rowF τ fun q => decide (q = 0))
      (colF τ fun _ => 1) (colF τ fun _ => 1) _ _ (hrun _ _) (hrun _ _)
      (fun h => by rw [hprev _ _ hu1 h, hprev _ _ hu1 h])
      (fun t ht j a ha => by
        have hmem : ((j, t) : Fin 3 × Fin G) ∈ Sr := mem_filter.mpr ⟨mem_product.mpr ⟨mem_univ _, ht⟩, a, ha⟩
        rw [h0] at hmem
        exact absurd hmem (notMem_empty _))
      (fun _ _ _ _ _ => rfl)
    rw [hout _ _ hu1, hout _ _ hu1] at hd
    simp at hd
  omega

end Zero

/-- **Theorem B, zero instance**: a word-local program computing Dither's prefix words has at least `4·L·m·n`
instructions. -/
theorem wordLocalZero : Pouw.Guarantees.Eps1.WordLocalZero := by
  intro m n Lseg FR FU G P blk hC hW
  exact count_blocks blk 4 fun τ => (block_zero P blk hC hW τ).trans (live_card_le P blk τ)

end Pouw.SecurityProofs.Eps1.WordLocal
