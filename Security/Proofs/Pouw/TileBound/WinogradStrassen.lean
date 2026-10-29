import Proofs.Pouw.TileBound.Proofs

/-!
# A program that beats the honest count in the abstract slot model (a later bundle; outside the 36 merged pins)

`docs/pouw/fp4-tile-model.md` §7. One Strassen level on 2 × 2 blocks, each of its 7 sub-products done by
Winograd's commutative inner-product trick, for `⟨2m, 4, 2n⟩`:

~~~text
(U V)_ij = (U_i0 + V_1j)(U_i1 + V_0j) − U_i0·U_i1 − V_0j·V_1j
~~~

As a `SlotProg` with every slot in `int8`, it is correct (`CrossOn` on every entry). It is `Representable` under every
block partition: each slot input reads at most 4 entries (2 of X and 2 of Y, checked by
`winograd_strassen_check.py`), and the proof bounds that by 8, within int8's capacity of 10.
It costs `7(mn + m + n)` against honest `8mn`. That is below honest
whenever `7(m + n) < mn`, for instance at `m = n = 16`. So the model's int8 cell is open for a real program, not only for
the numbers `support_admits_open` exhibits. On the card, tile index sharing and integer scale classes stop it (§7.3).
-/

namespace Pouw.TileBound

open Finset

/-- Strassen's X-side coefficients: `M_t = (Σ_pq α t p q · X_pq)(Σ_qr β t q r · Y_qr)`. -/
def sα : Fin 7 → Fin 2 → Fin 2 → ℚ :=
  ![![![1, 0], ![0, 1]], ![![0, 0], ![1, 1]], ![![1, 0], ![0, 0]], ![![0, 0], ![0, 1]],
    ![![1, 1], ![0, 0]], ![![-1, 0], ![1, 0]], ![![0, 1], ![0, -1]]]

/-- Strassen's Y-side coefficients. -/
def sβ : Fin 7 → Fin 2 → Fin 2 → ℚ :=
  ![![![1, 0], ![0, 1]], ![![1, 0], ![0, 0]], ![![0, 1], ![0, -1]], ![![-1, 0], ![1, 0]],
    ![![0, 0], ![0, 1]], ![![1, 1], ![0, 0]], ![![0, 0], ![1, 1]]]

/-- Strassen's output coefficients: `C_pr = Σ_t γ t p r · M_t`. -/
def sγ : Fin 7 → Fin 2 → Fin 2 → ℚ :=
  ![![![1, 0], ![0, 1]], ![![0, 0], ![1, -1]], ![![0, 1], ![0, 1]], ![![1, 0], ![1, 0]],
    ![![-1, 1], ![0, 0]], ![![0, 0], ![0, 1]], ![![1, 0], ![0, 0]]]

/-- **Strassen's identity**, as a polynomial identity in the 8 block values. -/
theorem strassen_identity (x y : Fin 2 → Fin 2 → ℚ) (p r : Fin 2) :
    ∑ t, sγ t p r * (∑ p', ∑ q, sα t p' q * x p' q) * (∑ q, ∑ r', sβ t q r' * y q r') = ∑ q, x p q * y q r := by
  fin_cases p <;> fin_cases r <;>
    simp [Fin.sum_univ_succ, sα, sβ, sγ] <;> ring

/-! ## Index splits and the program -/

/-- Block `p`, index `i` within it: `finProdFinEquiv (p, i) = i + m·p`. -/
abbrev rowE (m : ℕ) : Fin 2 × Fin m ≃ Fin (2 * m) := finProdFinEquiv

/-- The inner index: block `q`, position `c` within it, `c + 2q`. -/
abbrev kE : Fin 2 × Fin 2 ≃ Fin 4 := finProdFinEquiv

variable {m n : ℕ}

/-- `U_t[i, c]`, the X-side pre-add of sub-product `t` at row `i` and inner position `c`, as a linear form. -/
def Uf (t : Fin 7) (i : Fin m) (c : Fin 2) : Fin (2 * m) × Fin 4 → ℚ := fun e =>
  if ((rowE m).symm e.1).2 = i ∧ (kE.symm e.2).2 = c then sα t ((rowE m).symm e.1).1 (kE.symm e.2).1 else 0

/-- `V_t[c, j]`, the Y-side pre-add. -/
def Vf (t : Fin 7) (c : Fin 2) (j : Fin n) : Fin 4 × Fin (2 * n) → ℚ := fun e =>
  if ((rowE n).symm e.2).2 = j ∧ (kE.symm e.1).2 = c then sβ t (kE.symm e.1).1 ((rowE n).symm e.2).1 else 0

/-- Where sub-product `t`'s entry `(i, j)` goes: output `(p·m + i, r·n + j)` with coefficient `γ t p r`. -/
def Wf (t : Fin 7) (i : Fin m) (j : Fin n) : Fin (2 * m) × Fin (2 * n) → ℚ := fun o =>
  if ((rowE m).symm o.1).2 = i ∧ ((rowE n).symm o.2).2 = j then sγ t ((rowE m).symm o.1).1 ((rowE n).symm o.2).1
  else 0

/-- The product index: per sub-product `t`, Winograd's mixed products `(i, j)`, then its row and column corrections. -/
abbrev WSIdx (m n : ℕ) := Fin 7 × ((Fin m × Fin n) ⊕ (Fin m ⊕ Fin n))

/-- **The Winograd–Strassen program**, every slot `int8`. Mixed product `(t, i, j)` is
`(U_i0 + V_1j)(U_i1 + V_0j)`, row correction `(t, i)` is `U_i0·U_i1`, column correction `(t, j)` is `V_0j·V_1j`. -/
def wsProg (m n : ℕ) : SlotProg ℚ (2 * m) 4 (2 * n) (WSIdx m n) where
  a τ := match τ.2 with
    | .inl ij => Uf τ.1 ij.1 0
    | .inr (.inl i) => Uf τ.1 i 0
    | .inr (.inr _) => 0
  b τ := match τ.2 with
    | .inl ij => Vf τ.1 1 ij.2
    | .inr (.inl _) => 0
    | .inr (.inr j) => Vf τ.1 0 j
  a' τ := match τ.2 with
    | .inl ij => Uf τ.1 ij.1 1
    | .inr (.inl i) => Uf τ.1 i 1
    | .inr (.inr _) => 0
  b' τ := match τ.2 with
    | .inl ij => Vf τ.1 0 ij.2
    | .inr (.inl _) => 0
    | .inr (.inr j) => Vf τ.1 1 j
  w τ o := match τ.2 with
    | .inl ij => Wf τ.1 ij.1 ij.2 o
    | .inr (.inl i) => -∑ j, Wf τ.1 i j o
    | .inr (.inr j) => -∑ i, Wf τ.1 i j o
  fmt _ := .int8

theorem sum_Uf (t : Fin 7) (i : Fin m) (c : Fin 2) (X : Fin (2 * m) × Fin 4 → ℚ) :
    ∑ e, Uf t i c e * X e = ∑ p, ∑ q, sα t p q * X (rowE m (p, i), kE (q, c)) := by
  rw [← Equiv.sum_comp ((rowE m).prodCongr kE)]
  simp only [Uf, Equiv.prodCongr_apply, Prod.map, Equiv.symm_apply_apply, Fintype.sum_prod_type, ite_mul, zero_mul]
  simp [ite_and, Finset.sum_ite_eq']

theorem sum_Vf (t : Fin 7) (c : Fin 2) (j : Fin n) (Y : Fin 4 × Fin (2 * n) → ℚ) :
    ∑ e, Vf t c j e * Y e = ∑ q, ∑ r, sβ t q r * Y (kE (q, c), rowE n (r, j)) := by
  rw [← Equiv.sum_comp (kE.prodCongr (rowE n))]
  simp only [Vf, Equiv.prodCongr_apply, Prod.map, Equiv.symm_apply_apply, Fintype.sum_prod_type, ite_mul, zero_mul]
  simp [ite_and, Finset.sum_ite_eq']

theorem Wf_at (t : Fin 7) (i i' : Fin m) (j j' : Fin n) (p r : Fin 2) :
    Wf t i' j' (rowE m (p, i), rowE n (r, j)) = if i = i' ∧ j = j' then sγ t p r else 0 := by
  simp [Wf, Equiv.symm_apply_apply]

/-- **The program computes `X·Y` at every input.** Winograd's identity per sub-product entry, then Strassen's identity
per inner position. -/
theorem wsProg_eval (X : Fin (2 * m) × Fin 4 → ℚ) (Y : Fin 4 × Fin (2 * n) → ℚ) (o : Fin (2 * m) × Fin (2 * n)) :
    (wsProg m n).toQuadProg.eval X Y o = matMul X Y o := by
  obtain ⟨o1, o2⟩ := o
  obtain ⟨⟨P, i⟩, rfl⟩ := (rowE m).surjective o1
  obtain ⟨⟨R, j⟩, rfl⟩ := (rowE n).surjective o2
  let u : Fin 7 → Fin 2 → ℚ := fun t c => ∑ p, ∑ q, sα t p q * X (rowE m (p, i), kE (q, c))
  let v : Fin 7 → Fin 2 → ℚ := fun t c => ∑ q, ∑ r, sβ t q r * Y (kE (q, c), rowE n (r, j))
  have hev : (wsProg m n).toQuadProg.eval X Y (rowE m (P, i), rowE n (R, j)) =
      ∑ t, sγ t P R * ((u t 0 + v t 1) * (u t 1 + v t 0) - u t 0 * u t 1 - v t 0 * v t 1) := by
    unfold QuadProg.eval
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun t _ => ?_
    rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type]
    simp only [wsProg, sum_Uf, sum_Vf, Wf_at, Pi.zero_apply, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
    simp [ite_and, ite_mul, Finset.sum_ite_eq, Finset.sum_neg_distrib, u, v]
    ring
  rw [hev]
  have hw : ∀ t, sγ t P R * ((u t 0 + v t 1) * (u t 1 + v t 0) - u t 0 * u t 1 - v t 0 * v t 1) =
      sγ t P R * (u t 0 * v t 0) + sγ t P R * (u t 1 * v t 1) := fun t => by ring
  simp only [hw, Finset.sum_add_distrib]
  have hc : ∀ c : Fin 2, ∑ t, sγ t P R * (u t c * v t c) =
      ∑ q, X (rowE m (P, i), kE (q, c)) * Y (kE (q, c), rowE n (R, j)) := by
    intro c
    have h := strassen_identity (fun p q => X (rowE m (p, i), kE (q, c))) (fun q r => Y (kE (q, c), rowE n (r, j))) P R
    rw [← h]
    exact Finset.sum_congr rfl fun t _ => by simp only [u, v]; ring
  rw [hc 0, hc 1]
  unfold matMul
  rw [← Equiv.sum_comp kE]
  simp [Fintype.sum_prod_type, Fin.sum_univ_two]
  ring

/-- **Correct on every entry.** -/
theorem wsProg_crossOn : (wsProg m n).toQuadProg.CrossOn (fun _ => True) (fun _ => True) :=
  crossOn_of_eval _ fun _ _ _ _ o => ⟨wsProg_eval _ _ o, wsProg_eval _ _ o, wsProg_eval _ _ o⟩

/-! ## Representable in int8 under every block partition, and the cost -/

open Classical in
theorem XBlocks.touches_le_support {k' : ℕ} {β : Type*} [Fintype β] (B : XBlocks m k' β)
    (f : Fin m × Fin k' → ℚ) : B.touches f ≤ #{e | f e ≠ 0} := by
  unfold XBlocks.touches
  calc #{b | B.Touches f b} ≤ #((({e | f e ≠ 0} : Finset (Fin m × Fin k'))).image B.blk) := by
        apply Finset.card_le_card
        intro b hb
        obtain ⟨_, e, h1, h2⟩ := Finset.mem_filter.1 hb
        exact Finset.mem_image.2 ⟨e, Finset.mem_filter.2 ⟨Finset.mem_univ _, h2⟩, h1⟩
    _ ≤ _ := Finset.card_image_le

open Classical in
theorem YBlocks.touches_le_support {k' : ℕ} {β : Type*} [Fintype β] (B : YBlocks k' n β)
    (f : Fin k' × Fin n → ℚ) : B.touches f ≤ #{e | f e ≠ 0} := by
  unfold YBlocks.touches
  calc #{b | B.Touches f b} ≤ #((({e | f e ≠ 0} : Finset (Fin k' × Fin n))).image B.blk) := by
        apply Finset.card_le_card
        intro b hb
        obtain ⟨_, e, h1, h2⟩ := Finset.mem_filter.1 hb
        exact Finset.mem_image.2 ⟨e, Finset.mem_filter.2 ⟨Finset.mem_univ _, h2⟩, h1⟩
    _ ≤ _ := Finset.card_image_le

open Classical in
/-- A pre-add `U_t[i, c]` reads at most 4 entries: one per block `(p, q)`. -/
theorem card_support_Uf (t : Fin 7) (i : Fin m) (c : Fin 2) : #{e | Uf t i c e ≠ 0} ≤ 4 := by
  calc #{e | Uf t i c e ≠ 0} ≤
        #((univ : Finset (Fin 2 × Fin 2)).image fun pq => (rowE m (pq.1, i), kE (pq.2, c))) := by
        apply Finset.card_le_card
        intro e he
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at he
        refine Finset.mem_image.2 ⟨(((rowE m).symm e.1).1, (kE.symm e.2).1), Finset.mem_univ _, ?_⟩
        unfold Uf at he
        split_ifs at he with h
        · obtain ⟨h1, h2⟩ := h
          refine Prod.ext ?_ ?_
          · show rowE m (_, i) = e.1
            rw [← h1]
            exact (rowE m).apply_symm_apply e.1
          · show kE (_, c) = e.2
            rw [← h2]
            exact kE.apply_symm_apply e.2
        · exact absurd rfl he
    _ ≤ 4 := Finset.card_image_le.trans (by simp)

open Classical in
/-- A pre-add `V_t[c, j]` reads at most 4 entries. -/
theorem card_support_Vf (t : Fin 7) (c : Fin 2) (j : Fin n) : #{e | Vf t c j e ≠ 0} ≤ 4 := by
  calc #{e | Vf t c j e ≠ 0} ≤
        #((univ : Finset (Fin 2 × Fin 2)).image fun qr => (kE (qr.1, c), rowE n (qr.2, j))) := by
        apply Finset.card_le_card
        intro e he
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at he
        refine Finset.mem_image.2 ⟨((kE.symm e.1).1, ((rowE n).symm e.2).1), Finset.mem_univ _, ?_⟩
        unfold Vf at he
        split_ifs at he with h
        · obtain ⟨h1, h2⟩ := h
          refine Prod.ext ?_ ?_
          · show kE (_, c) = e.1
            rw [← h2]
            exact kE.apply_symm_apply e.1
          · show rowE n (_, j) = e.2
            rw [← h1]
            exact (rowE n).apply_symm_apply e.2
        · exact absurd rfl he
    _ ≤ 4 := Finset.card_image_le.trans (by simp)

/-- **Representable in int8, under every block partition.** Each slot input reads at most 4 entries (2 of X and 2 of Y,
as the lane's script checks); this proof bounds each pre-add by 4 entries, so each input by 8, within int8's capacity of
10. Since a form reads no more blocks than entries, the bound holds for any `BX`, `BY`, NVFP4's included (at `k = 4`
NVFP4's 16-wide blocks are one block per row, which doesn't matter here). -/
theorem wsProg_representable {β γ : Type*} [Fintype β] [Fintype γ] (BX : XBlocks (2 * m) 4 β)
    (BY : YBlocks 4 (2 * n) γ) : (wsProg m n).Representable BX BY := by
  intro τ
  obtain ⟨t, s⟩ := τ
  have hU : ∀ i c, BX.touches (Uf t i c) ≤ 4 := fun i c =>
    (XBlocks.touches_le_support _ _).trans (card_support_Uf t i c)
  have hV : ∀ c j, BY.touches (Vf t c j) ≤ 4 := fun c j =>
    (YBlocks.touches_le_support _ _).trans (card_support_Vf t c j)
  rcases s with ⟨i, j⟩ | i | j
  · simp only [wsProg, capacity]
    have := hU i 0; have := hU i 1; have := hV 1 j; have := hV 0 j
    omega
  · simp only [wsProg, capacity, YBlocks.touches_zero]
    have := hU i 0; have := hU i 1
    omega
  · simp only [wsProg, capacity, XBlocks.touches_zero]
    have := hV 0 j; have := hV 1 j
    omega

/-- Its cost: `7(mn + m + n)` int8 slots at price 1. -/
theorem wsProg_cost : (wsProg m n).cost = 7 * (m * n + (m + n)) := by
  simp [SlotProg.cost, wsProg, price, Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_sum,
    Fintype.card_fin]

/-- **A program that beats the honest count in the abstract slot model.** For `7(m + n) < mn`, the int8
Winograd–Strassen program for `⟨2m, 4, 2n⟩` is correct on every entry, representable within int8's capacity under every
block partition, and costs less than FP4's price for every product. So the model's int8 cell, 3 to 10 blocks per input, is
open for a real program. -/
theorem int8_winograd_strassen_beats (m n : ℕ) (h : 7 * (m + n) < m * n) :
    ∃ P : SlotProg ℚ (2 * m) 4 (2 * n) (Fin 7 × ((Fin m × Fin n) ⊕ (Fin m ⊕ Fin n))),
      P.toQuadProg.CrossOn (fun _ => True) (fun _ => True) ∧
      (∀ {β γ : Type} [Fintype β] [Fintype γ] (BX : XBlocks (2 * m) 4 β) (BY : YBlocks 4 (2 * n) γ),
        P.Representable BX BY) ∧ (∀ t, P.fmt t = .int8) ∧
      P.cost < price .fp4 * nLive (m := 2 * m) (k := 4) (n := 2 * n) (fun _ => True) (fun _ => True) := by
  refine ⟨wsProg m n, wsProg_crossOn, fun BX BY => wsProg_representable BX BY, fun _ => rfl, ?_⟩
  rw [wsProg_cost, nLive_all]
  have h' : ((7 * (m + n) : ℕ) : ℚ) < ((m * n : ℕ) : ℚ) := by exact_mod_cast h
  push_cast at h' ⊢
  norm_num [price]
  nlinarith

/-- At `m = n = 16`, that is `⟨32, 4, 32⟩`: 2,016 int8 slots against honest 4,096 FP4 slots, 2,016 < 2,048. -/
theorem int8_winograd_strassen_beats_32 :
    ∃ P : SlotProg ℚ (2 * 16) 4 (2 * 16) (Fin 7 × ((Fin 16 × Fin 16) ⊕ (Fin 16 ⊕ Fin 16))),
      P.toQuadProg.CrossOn (fun _ => True) (fun _ => True) ∧
      (∀ {β γ : Type} [Fintype β] [Fintype γ] (BX : XBlocks (2 * 16) 4 β) (BY : YBlocks 4 (2 * 16) γ),
        P.Representable BX BY) ∧ (∀ t, P.fmt t = .int8) ∧
      P.cost < price .fp4 * nLive (m := 2 * 16) (k := 4) (n := 2 * 16) (fun _ => True) (fun _ => True) :=
  int8_winograd_strassen_beats 16 16 (by norm_num)

end Pouw.TileBound
