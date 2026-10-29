import Definitions.Pouw.PearlC.DeviceFp4
import Proofs.Pouw.PearlC.Fp4Freeze
import Proofs.Pouw.PearlC.PeelExactProofs

/-!
# P2 on Pearl-C4's U, `Fp4Dev.peelFp4` (staging, nothing pinned)

`PeelExactProofs` proves P2 on Pearl-C's U at the E4M3 atom. Pearl-C4's peel factors use the FP4 atom: `C̃`, `A′·F_B`,
`B̃·F_A` and `G = F_Bᵀ·F_A` are `cTilde` chains on block-scaled E2M1 rows (`F`'s columns at scale `1.0`), while `P_A`,
`P_B` and U's four BF16 K16 steps are `PeelExact`'s (`paRow?`, `pbRow?`, `uWord?`). So the facts carry over with the chain
words' error left as a parameter.

**(a) The noise cancels exactly.** The exact counterpart in the pinned order (`uQ4`: every chain, FMA, add, `RN_bf16` and
K16 step exact) is `(A′ − E_A·F_Aᵀ)·(B̃ − E_B·F_Bᵀ)ᵀ` on any values (`uQ4_cancel`, by
`PeelExactProofs.peelUQ_cancel`). On the served operands (`opVal`: code value times block scale; `basisVal`; the peel's noise rows `beVal`), with Pearl-C4's rows
`Ẽ = eTilde E a_E`, each row entry is `E·a_E/16` exactly (`bfVal_eTilde`, all 16 codes × 128 scale bytes by the kernel),
so the noise the peel cancels is the noise atom's, `E′ = E·a_E·2⁻⁴` against `F` (`uQ4_cancel_served`).

**(b) The rounding is bounded.** Wherever the served entry is finite, `peelFp4_err` puts it within `peelBound4` of `uQ4`
on the named range `PeelRange4`: each FP4 chain word of the entry within its `δ` of its exact dot (`ChainsNear4`), and
U's four K16 steps on the K16 range (`URange4`, `PeelExactProofs.URange`). The FP32 FMAs and add, `RN_bf16` and the K16
steps (width 26, `eps16`) are bounded by `PeelExactProofs`' lemmas as they stand. Where the chains are exact runs
(`ExactRuns4`: `Fp4Freeze.ExactRun` on each chain from `+0`), the exactness lemma `Fp4Freeze.cTilde_dot` puts
every chain word on its exact dot (`chainNear_exact`, through `atomDot_sum`), so the bound is U's own rounding
(`peelFp4_err_exact`, `peelBound4Exact`). `Fp4Freeze` shows D₄ gives no exact run (`noLoss_gap`), so `ExactRuns4` is a
per-run hypothesis, not a consequence of the domain. This is option (b) of the exactness ruling: a claim that
equates `C̃` with the exact product carries a per-word exact-run condition, here `ExactRuns4 i j` on entry `(i, j)`
alone, and it loses the inexact words. `peelFp4_err` itself needs no exactness.

**(c) Salt-dependence**, by witness at `sm120Nvf4` (`served4_salt_dependent`): the row `x ≡ 1` of `k = 128` with the
pin's scales (`α = 1344`, `β = f32(336·f32(16/√2346.125))`), formed by `formRow` and served by `peelFp4` with
`Ẽ = eTilde E a_E`, under two salts whose lines differ in one code: the served words differ.
-/

namespace Pouw.PearlC.PeelFp4Proofs

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactProofs
  Matrix

variable {m n k : Nat}

/-! ## Values -/

/-- A formed row's value at position `l`: its code's value times its block's scale (`Fp4Freeze.scaleAt`, `0` where the
byte is absent or invalid). -/
def rowVal (p : Fp4Atom) (k : Nat) (a : Row4) : Fin k → ℚ :=
  fun l => decode (a.1.getD l 0) * Fp4Freeze.scaleAt p.fmt a.2 (l / p.block)

/-- A formed operand's values; a row outside the model is `0`. -/
def opVal (p : Fp4Atom) (k : Nat) (A : Op4 m) : Matrix (Fin m) (Fin k) ℚ :=
  fun i => match A i with
    | some a => rowVal p k a
    | none => 0

/-- A basis's values: `F` enters every chain at scale `1.0`. -/
def basisVal (F : Matrix (Fin k) (Fin 32) Nib) : Matrix (Fin k) (Fin 32) ℚ := F.map decode

/-- The peel's noise rows' values (BF16 words). -/
def beVal (be : Matrix (Fin m) (Fin 32) Nat) : Matrix (Fin m) (Fin 32) ℚ := be.map bfVal

/-- The noise the atom adds, per row: `E′ = E·a_E·2⁻⁴`, the line at its scale byte times `F`'s scale `2⁻⁴`. -/
def noiseE (E : Fin m → Fin 32 → Nib) (aE : Fin m → Fin 256) : Matrix (Fin m) (Fin 32) ℚ :=
  fun i r => decode (E i r) * codeVal (aE i) / 16

/-! ## (a) The noise cancels exactly -/

section Exact

variable (A' : Matrix (Fin m) (Fin k) ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (FA FB : Matrix (Fin k) (Fin 32) ℚ)
  (EA : Matrix (Fin m) (Fin 32) ℚ) (EB : Matrix (Fin n) (Fin 32) ℚ)

/-- `C̃_ij`'s chain, exact. -/
def cQ4 (i : Fin m) (j : Fin n) : ℚ := ∑ l, A' i l * B' j l

/-- `(A′·F_B)_ip`'s chain, exact. -/
def afQ4 (i : Fin m) (p : Fin 32) : ℚ := ∑ l, A' i l * FB l p

/-- `(B̃·F_A)_jq`'s chain, exact. -/
def bfQ4 (j : Fin n) (q : Fin 32) : ℚ := ∑ l, B' j l * FA l q

/-- `G_pq = (F_Bᵀ·F_A)_pq`'s chain, exact. -/
def gQ4 (p q : Fin 32) : ℚ := ∑ l, FB l p * FA l q

/-- `P_A,i` exact: `[E_A,i ‖ A′·F_B]`, the `RN_bf16` of the second block dropped. -/
def paQ4 (i : Fin m) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then EA i ⟨t.val, h⟩ else afQ4 A' FB i ⟨t.val - 32, by have := t.isLt; omega⟩

/-- `acc_q`'s FMA chain, exact: `Σ_p E_B,jp·G_pq`. -/
def accQ4 (j : Fin n) (q : Fin 32) : ℚ := ∑ p, EB j p * gQ4 FA FB p q

/-- `P_B,j` exact: `[acc_q − (B̃·F_A)_jq ‖ −E_B,j]`, the FP32 difference's rounding and its `RN_bf16` dropped. -/
def pbQ4 (j : Fin n) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then accQ4 FA FB EB j ⟨t.val, h⟩ - bfQ4 B' FA j ⟨t.val, h⟩
  else -EB j ⟨t.val - 32, by have := t.isLt; omega⟩

/-- **U's exact counterpart in the pinned order**: from `C̃_ij`, four exact K16 steps over the slices of `P_A,i` and
`P_B,j`. -/
def uQ4 (i : Fin m) (j : Fin n) : ℚ :=
  (List.range 4).foldl (fun acc s => k16Q acc (sliceQ (paQ4 A' FB EA i) s) (sliceQ (pbQ4 B' FA FB EB j) s))
    (cQ4 A' B' i j)

theorem uQ4_eq_sum (i : Fin m) (j : Fin n) :
    uQ4 A' B' FA FB EA EB i j = cQ4 A' B' i j + ∑ t, paQ4 A' FB EA i t * pbQ4 B' FA FB EB j t := by
  unfold uQ4; exact foldl_k16Q _ _ _

/-- The exact counterpart is Pearl's U with unit `β`, `C̃ = A′·B̃ᵀ` and the noise `E_A, F_A, E_B, F_B` (`peelUQ`). -/
theorem uQ4_eq_peelUQ (i : Fin m) (j : Fin n) :
    uQ4 A' B' FA FB EA EB i j = peelUQ (A' * B'ᵀ) A' B' EA FA EB FB i j := by
  rw [uQ4_eq_sum]
  simp only [peelUQ, Matrix.add_apply, Matrix.mul_apply, transpose_apply]
  congr 1
  refine (Fintype.sum_equiv (finSumFinEquiv : Fin 32 ⊕ Fin 32 ≃ Fin 64) _ _ ?_).symm
  rintro (t | t)
  · have hv : ((finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) : Nat) = t := rfl
    have e1 : paQ4 A' FB EA i (finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) = EA i t := by simp [paQ4, hv]
    have e2 : pbQ4 B' FA FB EB j (finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) =
        accQ4 FA FB EB j t - bfQ4 B' FA j t := by simp [pbQ4, hv]
    rw [e1, e2]
    simp only [fromCols_apply_inl]
    congr 1
    simp only [accQ4, gQ4, bfQ4, Matrix.mul_apply, Matrix.sub_apply, transpose_apply]
    simp only [Finset.sum_mul, Finset.mul_sum, sub_mul, Finset.sum_sub_distrib]
    rw [Finset.sum_comm]
    simp only [mul_assoc]
  · have hv : ((finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) : Nat) = 32 + t := rfl
    have e1 : paQ4 A' FB EA i (finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) = afQ4 A' FB i t := by
      simp [paQ4, hv]
    have e2 : pbQ4 B' FA FB EB j (finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) = -EB j t := by
      simp [pbQ4, hv]
    rw [e1, e2]
    simp only [fromCols_apply_inr, afQ4, Matrix.mul_apply, Matrix.neg_apply, mul_neg]

/-- **(a) The noise cancels exactly**: U's exact counterpart in the pinned order is `(A′ − E_A·F_Aᵀ)·(B̃ − E_B·F_Bᵀ)ᵀ`. -/
theorem uQ4_cancel (i : Fin m) (j : Fin n) :
    uQ4 A' B' FA FB EA EB i j = ((A' - EA * FAᵀ) * (B' - EB * FBᵀ)ᵀ) i j := by
  rw [uQ4_eq_peelUQ, peelUQ_cancel]

end Exact

/-- **Ẽ is exact**: for every code `c` and every UE4M3 scale byte `s` (sign bit clear), the BF16 word
`bf16(|c|·s/16)` with `c`'s sign has the value `c·s/16`. -/
theorem bfVal_eTilde_elem : ∀ c : Nib, ∀ s : Fin 256, s.val < 128 →
    bfVal (bf16WordOf (|decode c| * codeVal s / 16) ||| (if c.sign then 0x8000 else 0)) = decode c * codeVal s / 16 := by
  decide +kernel

/-- `eTilde`'s entries are `E·a_E/16` exactly. -/
theorem bfVal_eTilde (e : Fin 32 → Nib) (aE : Fin 256) (h : aE.val < 128) (q : Fin 32) :
    bfVal (eTilde e aE q) = decode (e q) * codeVal aE / 16 :=
  bfVal_eTilde_elem (e q) aE h

theorem beVal_eTilde (E : Fin m → Fin 32 → Nib) (aE : Fin m → Fin 256) (h : ∀ i, (aE i).val < 128) :
    beVal (fun i => eTilde (E i) (aE i)) = noiseE E aE := by
  funext i r
  exact bfVal_eTilde (E i) (aE i) (h i) r

/-- **(a) on the served operands**: with Pearl-C4's noise rows `Ẽ = eTilde E a_E`, U's exact counterpart is
`(A′ − E′_A·F_Aᵀ)·(B̃ − E′_B·F_Bᵀ)ᵀ`, where `A′`, `B̃` are the formed operands' values, `F` the bases' values and
`E′ = E·a_E·2⁻⁴` the noise the atom adds (`noiseE`). -/
theorem uQ4_cancel_served (p : Fp4Atom) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (EA : Fin m → Fin 32 → Nib) (EB : Fin n → Fin 32 → Nib) (aA : Fin m → Fin 256) (aB : Fin n → Fin 256)
    (hA : ∀ i, (aA i).val < 128) (hB : ∀ j, (aB j).val < 128) (i : Fin m) (j : Fin n) :
    uQ4 (opVal p k A) (opVal p k B) (basisVal FA) (basisVal FB) (beVal fun i => eTilde (EA i) (aA i))
        (beVal fun j => eTilde (EB j) (aB j)) i j =
      ((opVal p k A - noiseE EA aA * (basisVal FA)ᵀ) * (opVal p k B - noiseE EB aB * (basisVal FB)ᵀ)ᵀ) i j := by
  rw [uQ4_cancel, beVal_eTilde EA aA hA, beVal_eTilde EB aB hB]

/-! ## `peelFp4` as rows and steps -/

/-- An FP4 chain word of the peel: `cTilde`, finite. -/
def ch4 (p : Fp4Atom) (x y : List Nib) (sx sy : List (Fin 256)) : Option Nat :=
  (cTilde p x y sx sy).bind PeelExact.finite?

/-- `peelFp4`'s `C̃` word and rows `P_A,i`, `P_B,j` before U's four steps. -/
def peelRows4 (p : Fp4Atom) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) (i : Fin m) (j : Fin n) :
    Option (Nat × List Nat × List Nat) := do
  let a ← A i
  let b ← B j
  let one := onesScales p k
  let c ← ch4 p a.1 b.1 a.2 b.2
  let af ← (List.ofFn fun q => ch4 p a.1 (colF FB q) a.2 one).mapM id
  let bf ← (List.ofFn fun q => ch4 p b.1 (colF FA q) b.2 one).mapM id
  let g ← (List.ofFn fun q => (List.ofFn fun q' => ch4 p (colF FB q) (colF FA q') one one).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  pure (c, pa, pb)

theorem peelFp4_eq_rows (p : Fp4Atom) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
    (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) (i : Fin m) (j : Fin n) :
    peelFp4 p A B FA FB beA beB i j = (peelRows4 p A B FA FB beA beB i j).bind fun r => uWord? r.1 r.2.1 r.2.2 := by
  unfold peelFp4 peelRows4 ch4
  simp only [Option.bind_eq_bind, Option.bind_assoc, Option.pure_def, Option.bind_some]

/-- **`peelFp4` from its parts**: the entry is `uWord? c pa pb` when the rows, chains and peel rows are as given. The
conformance checks assemble `peelFp4`'s entries from their parts this way. -/
theorem peelFp4_of_parts {p : Fp4Atom} {A : Op4 m} {B : Op4 n} {FA FB : Matrix (Fin k) (Fin 32) Nib}
    {beA : Matrix (Fin m) (Fin 32) Nat} {beB : Matrix (Fin n) (Fin 32) Nat} {i : Fin m} {j : Fin n} {a b : Row4}
    {c : Nat} {af bf pa pb : List Nat} {g : List (List Nat)} (hA : A i = some a) (hB : B j = some b)
    (hc : ch4 p a.1 b.1 a.2 b.2 = some c)
    (haf : (List.ofFn fun q => ch4 p a.1 (colF FB q) a.2 (onesScales p k)).mapM id = some af)
    (hbf : (List.ofFn fun q => ch4 p b.1 (colF FA q) b.2 (onesScales p k)).mapM id = some bf)
    (hg : (List.ofFn fun q => (List.ofFn fun q' =>
      ch4 p (colF FB q) (colF FA q') (onesScales p k) (onesScales p k)).mapM id).mapM id = some g)
    (hpa : paRow? (List.ofFn (beA i)) af = some pa) (hpb : pbRow? (List.ofFn (beB j)) bf g = some pb) :
    peelFp4 p A B FA FB beA beB i j = uWord? c pa pb := by
  rw [peelFp4_eq_rows]
  simp only [peelRows4, hA, hB, hc, haf, hbf, hg, hpa, hpb, Option.bind_eq_bind, Option.bind_some, Option.pure_def]

/-- A list of `N` options, each `some`, is `some` of their values: how the conformance checks and the witness assemble a
factor row or `G` from checks of its rows or entries. -/
theorem mapM_ofFn_some {β : Type} {N : Nat} (f : Fin N → Option β) (g : Fin N → β) (h : ∀ q, f q = some (g q)) :
    (List.ofFn f).mapM id = some (List.ofFn g) := by
  induction N with
  | zero => rfl
  | succ N ih =>
    rw [List.ofFn_succ, List.ofFn_succ, List.mapM_cons, ih (fun q => f q.succ) (fun q => g q.succ) (fun q => h q.succ),
      h 0]
    rfl

theorem peelRows4_spec {p : Fp4Atom} {A : Op4 m} {B : Op4 n} {FA FB : Matrix (Fin k) (Fin 32) Nib}
    {beA : Matrix (Fin m) (Fin 32) Nat} {beB : Matrix (Fin n) (Fin 32) Nat} {i : Fin m} {j : Fin n} {c : Nat}
    {pa pb : List Nat} (h : peelRows4 p A B FA FB beA beB i j = some (c, pa, pb)) :
    ∃ a b af bf g, A i = some a ∧ B j = some b ∧ ch4 p a.1 b.1 a.2 b.2 = some c ∧
      (List.ofFn fun q => ch4 p a.1 (colF FB q) a.2 (onesScales p k)).mapM id = some af ∧
      (List.ofFn fun q => ch4 p b.1 (colF FA q) b.2 (onesScales p k)).mapM id = some bf ∧
      (List.ofFn fun q => (List.ofFn fun q' =>
        ch4 p (colF FB q) (colF FA q') (onesScales p k) (onesScales p k)).mapM id).mapM id = some g ∧
      paRow? (List.ofFn (beA i)) af = some pa ∧ pbRow? (List.ofFn (beB j)) bf g = some pb := by
  unfold peelRows4 at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨a, hA, b, hB, c', hc, af, haf, bf, hbf, g, hg, pa', hpa, pb', hpb, rfl, rfl, rfl⟩ := h
  exact ⟨a, b, af, bf, g, hA, hB, hc, haf, hbf, hg, hpa, hpb⟩

/-! ## (b) The rounding is bounded -/

/-- **An FP4 chain word within `δ` of `v`**: wherever the chain is finite, its word has a value within `δ` of `v`. -/
def ChainNear (p : Fp4Atom) (x y : List Nib) (sx sy : List (Fin 256)) (v δ : ℚ) : Prop :=
  ∀ w, ch4 p x y sx sy = some w → ∃ cv, wordQ w = some cv ∧ |cv - v| ≤ δ

/-- A chain's exact dot, position by position: the code values times their blocks' scales. -/
def flatDot (p : Fp4Atom) (x y : List Nib) (sx sy : List (Fin 256)) : ℚ :=
  ∑ l ∈ Finset.range x.length, decode (x.getD l 0) * decode (y.getD l 0) *
    (Fp4Freeze.scaleAt p.fmt sx (l / p.block) * Fp4Freeze.scaleAt p.fmt sy (l / p.block))

theorem sum_range_mul (H : ℕ → ℚ) (a b : ℕ) :
    ∑ l ∈ Finset.range (a * b), H l = ∑ u ∈ Finset.range a, ∑ r ∈ Finset.range b, H (u * b + r) := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_mul, Finset.sum_range_add, ih, Finset.sum_range_succ]

theorem zip_sum (l1 l2 : List Nib) (h : l1.length = l2.length) :
    ((l1.zip l2).map fun x => Fp4.decode x.1 * Fp4.decode x.2).sum =
      ∑ r ∈ Finset.range l1.length, Fp4.decode (l1.getD r 0) * Fp4.decode (l2.getD r 0) := by
  induction l1 generalizing l2 with
  | nil => simp
  | cons x xs ih =>
    cases l2 with
    | nil => simp at h
    | cons y ys =>
      rw [List.zip_cons_cons, List.map_cons, List.sum_cons, List.length_cons, Finset.sum_range_succ',
        ih ys (by simpa using h)]
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      ring

theorem slice_getElem? {α : Type} (l : List α) (u w r : ℕ) (hr : r < w) :
    (Fp4Freeze.slice l u w)[r]? = l[u * w + r]? := by
  rw [Fp4Freeze.slice, List.getElem?_take, ite_eq_left hr, List.getElem?_drop]

theorem slice_getD {α : Type} (l : List α) (u w r : ℕ) (d : α) (hr : r < w) :
    (Fp4Freeze.slice l u w).getD r d = l.getD (u * w + r) d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, slice_getElem? l u w r hr]

theorem slice_length {α : Type} (l : List α) (u w : ℕ) (h : (u + 1) * w ≤ l.length) :
    (Fp4Freeze.slice l u w).length = w := by
  rw [Nat.succ_mul] at h
  simp only [Fp4Freeze.slice, List.length_take, List.length_drop]
  omega

theorem scaleAt_slice (f : ScaleFmt) (s : List (Fin 256)) (u w i : ℕ) (hi : i < w) :
    Fp4Freeze.scaleAt f (Fp4Freeze.slice s u w) i = Fp4Freeze.scaleAt f s (u * w + i) := by
  simp only [Fp4Freeze.scaleAt, slice_getElem? s u w i hi]

theorem div_group {g B j r : ℕ} (hg : 0 < g) (hgB : g ∣ B) (hr : r < g) : (j * g + r) / B = j * g / B := by
  obtain ⟨c, rfl⟩ := hgB
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul, Nat.mul_div_cancel _ hg]
  congr 1
  rw [add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hr, zero_add]

/-- **The atoms' exact dots are the flat exact dot**: `Fp4Freeze.atomDot`, summed over a shaped chain's atoms, is
`flatDot`, on an atom whose groups lie inside its blocks inside its firings (`Fp4Freeze.BlockShape`). -/
theorem atomDot_sum (p : Fp4Atom) (hp : Fp4Freeze.BlockShape p) {x y : List Nib} {sx sy : List (Fin 256)}
    (hs : Fp4Freeze.Shaped p x y sx sy) :
    ∑ u ∈ Finset.range (x.length / p.k), Fp4Freeze.atomDot p x y sx sy u = flatDot p x y sx sy := by
  obtain ⟨hG, ⟨c, hc⟩, ⟨S, hS⟩, hK⟩ := hp
  obtain ⟨hyx, hxk, -, -⟩ := hs
  have hB : 0 < p.block := by
    rcases Nat.eq_zero_or_pos p.block with h | h
    · rw [h, zero_mul] at hS; omega
    · exact h
  have hsc : p.scales = S := by unfold Fp4Atom.scales; rw [hS, Nat.mul_div_cancel_left _ hB]
  have hkg : p.k / p.group * p.group = p.k := Nat.div_mul_cancel ⟨c * S, by rw [hS, hc, mul_assoc]⟩
  set N := x.length / p.k
  have hN : x.length = N * p.k := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hxk)).symm
  unfold flatDot
  rw [hN, sum_range_mul]
  refine Finset.sum_congr rfl fun u hu => ?_
  have hu' : (u + 1) * p.k ≤ x.length := by
    rw [hN]; exact Nat.mul_le_mul_right _ (Finset.mem_range.1 hu)
  have hlx := slice_length x u p.k hu'
  have hly := slice_length y u p.k (hyx ▸ hu')
  unfold Fp4Freeze.atomDot
  rw [← hkg, sum_range_mul, hkg]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' : (j + 1) * p.group ≤ p.k := by
    rw [← hkg]; exact Nat.mul_le_mul_right _ (Finset.mem_range.1 hj)
  have hjg : j * p.group < p.k := by rw [Nat.succ_mul] at hj'; omega
  have hjS : j * p.group / p.block < p.scales := by
    rw [hsc]
    exact (Nat.div_lt_iff_lt_mul hB).2 (calc j * p.group < p.k := hjg
      _ = S * p.block := by rw [hS, mul_comm])
  unfold Fp4Freeze.groupDot
  have hgx := slice_length _ j p.group (by rw [hlx]; exact hj' : (j + 1) * p.group ≤ (Fp4Freeze.slice x u p.k).length)
  have hgy := slice_length _ j p.group (by rw [hly]; exact hj' : (j + 1) * p.group ≤ (Fp4Freeze.slice y u p.k).length)
  rw [zip_sum _ _ (by rw [hgx, hgy]), hgx, Finset.sum_mul, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r hr => ?_
  have hr' : r < p.group := Finset.mem_range.1 hr
  have hin : j * p.group + r < p.k := by rw [Nat.succ_mul] at hj'; omega
  have hdiv : (u * p.k + (j * p.group + r)) / p.block = u * p.scales + j * p.group / p.block := by
    rw [hsc, hS, show u * (p.block * S) = p.block * (u * S) by ring, Nat.mul_add_div hB,
      div_group hG ⟨c, hc⟩ hr']
  rw [slice_getD _ _ _ _ _ hr', slice_getD _ _ _ _ _ hr', slice_getD _ _ _ _ _ hin, slice_getD _ _ _ _ _ hin,
    scaleAt_slice _ _ _ _ _ hjS, scaleAt_slice _ _ _ _ _ hjS, hdiv]
  ring

/-- **The exactness lemma on one chain** (`Fp4Freeze.cTilde_dot`): on an exact run from `+0`, the peel's chain
word is `flatDot`. -/
theorem chainNear_exact {p : Fp4Atom} (hp : Fp4Freeze.BlockShape p) {x y : List Nib} {sx sy : List (Fin 256)}
    (hs : Fp4Freeze.Shaped p x y sx sy) (hr : Fp4Freeze.ExactRun p [] 0 x y sx sy (x.length / p.k)) :
    ChainNear p x y sx sy (flatDot p x y sx sy) 0 := by
  obtain ⟨w, hw, -, hq⟩ := Fp4Freeze.cTilde_dot p hs hr
  intro w' h'
  have hww : w' = w := by
    simp only [ch4, hw, Option.bind_some, PeelExact.finite?] at h'
    by_cases hh : (w >>> 23) % 256 = 255 <;> simp [hh] at h'
    exact h'.symm
  subst hww
  exact ⟨_, hq, by rw [atomDot_sum p hp hs, sub_self, abs_zero]⟩

theorem scaleOne_value (f : ScaleFmt) : scaleValue f (scaleOne f) = some 1 := by
  cases f <;> decide +kernel

theorem scaleAt_ones (p : Fp4Atom) (hp : Fp4Freeze.BlockShape p) {k : ℕ} (hk : p.k ∣ k) {l : ℕ} (hl : l < k) :
    Fp4Freeze.scaleAt p.fmt (onesScales p k) (l / p.block) = 1 := by
  unfold Fp4Freeze.scaleAt onesScales
  rw [List.getElem?_replicate, ite_eq_left (Nat.div_lt_div_of_lt_of_dvd (dvd_trans hp.2.2.1 hk) hl)]
  simp [scaleOne_value]

theorem ones_length (p : Fp4Atom) (hp : Fp4Freeze.BlockShape p) {k : ℕ} (hk : p.k ∣ k) :
    (onesScales p k).length = k / p.k * p.scales := by
  obtain ⟨-, -, ⟨S, hS⟩, hK⟩ := hp
  obtain ⟨t, rfl⟩ := hk
  have hB : 0 < p.block := by
    rcases Nat.eq_zero_or_pos p.block with h | h
    · rw [h, zero_mul] at hS; omega
    · exact h
  unfold onesScales Fp4Atom.scales
  rw [List.length_replicate, Nat.mul_div_cancel_left _ hK, hS, mul_assoc, Nat.mul_div_cancel_left _ hB,
    Nat.mul_div_cancel_left _ hB, mul_comm]

theorem flatDot_fin (p : Fp4Atom) {k : ℕ} {x y : List Nib} {sx sy : List (Fin 256)} (hx : x.length = k) :
    flatDot p x y sx sy = ∑ l : Fin k, decode (x.getD l 0) * Fp4Freeze.scaleAt p.fmt sx (l / p.block) *
      (decode (y.getD l 0) * Fp4Freeze.scaleAt p.fmt sy (l / p.block)) := by
  unfold flatDot
  rw [hx, Finset.sum_range]
  exact Finset.sum_congr rfl fun l _ => by ring

section Bound

variable (p : Fp4Atom) (A : Op4 m) (B : Op4 n) (FA FB : Matrix (Fin k) (Fin 32) Nib)
  (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
  (δC : ℚ) (δA δB : Fin 32 → ℚ) (δG : Fin 32 → Fin 32 → ℚ)

/-- **The entry's FP4 chains within their `δ`s**: on the entry's rows `a = A i`, `b = B j`, the words of `C̃_ij`
(`δC`), `(A′·F_B)_ip` (`δA p`), `(B̃·F_A)_jq` (`δB q`) and `G_pq` (`δG p q`) are within those of their exact dots. -/
def ChainsNear4 (i : Fin m) (j : Fin n) : Prop :=
  ∀ a b, A i = some a → B j = some b →
    ChainNear p a.1 b.1 a.2 b.2 (cQ4 (opVal p k A) (opVal p k B) i j) δC ∧
    (∀ q, ChainNear p a.1 (colF FB q) a.2 (onesScales p k) (afQ4 (opVal p k A) (basisVal FB) i q) (δA q)) ∧
    (∀ q, ChainNear p b.1 (colF FA q) b.2 (onesScales p k) (bfQ4 (opVal p k B) (basisVal FA) j q) (δB q)) ∧
    ∀ q q', ChainNear p (colF FB q) (colF FA q') (onesScales p k) (onesScales p k)
      (gQ4 (basisVal FA) (basisVal FB) q q') (δG q q')

/-- **The entry's FP4 chains exact** (every `δ = 0`): each chain word's value is its exact dot. It follows from exact
runs (`chainsExact4_of_runs`). -/
def ChainsExact4 (i : Fin m) (j : Fin n) : Prop :=
  ChainsNear4 p A B FA FB 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) i j

/-- **The entry's FP4 chains are exact runs** (`Fp4Freeze.ExactRun` from `+0`): on the entry's rows `a = A i`,
`b = B j`, of length `k` and shaped for the chain (`Fp4Freeze.Shaped`), the chains of `C̃_ij`, `(A′·F_B)_ip`,
`(B̃·F_A)_jq` and `G_pq`. D₄ gives no exact run (`Fp4Freeze.noLoss_gap`); it is a per-run hypothesis, which the census
measures. -/
def ExactRuns4 (i : Fin m) (j : Fin n) : Prop :=
  ∀ a b, A i = some a → B j = some b → a.1.length = k ∧ Fp4Freeze.Shaped p a.1 b.1 a.2 b.2 ∧
    Fp4Freeze.ExactRun p [] 0 a.1 b.1 a.2 b.2 (k / p.k) ∧
    (∀ q, Fp4Freeze.ExactRun p [] 0 a.1 (colF FB q) a.2 (onesScales p k) (k / p.k)) ∧
    (∀ q, Fp4Freeze.ExactRun p [] 0 b.1 (colF FA q) b.2 (onesScales p k) (k / p.k)) ∧
    ∀ q q', Fp4Freeze.ExactRun p [] 0 (colF FB q) (colF FA q') (onesScales p k) (onesScales p k) (k / p.k)

/-- **Exact runs make the chains exact**, by the exactness lemma (`Fp4Freeze.cTilde_dot`) on each chain. -/
theorem chainsExact4_of_runs (hp : Fp4Freeze.BlockShape p) (hk : p.k ∣ k) (i : Fin m) (j : Fin n)
    (h : ExactRuns4 p A B FA FB i j) : ChainsExact4 p A B FA FB i j := by
  intro a b hA hB
  obtain ⟨hla, hs, hC, hAF, hBF, hG⟩ := h a b hA hB
  obtain ⟨hba, hmod, hsa, hsb⟩ := hs
  have hlb : b.1.length = k := hba.trans hla
  have hcol : ∀ F : Matrix (Fin k) (Fin 32) Nib, ∀ q, (colF F q).length = k := fun F q => by simp [colF]
  have hones := ones_length p hp hk
  have hmodk : k % p.k = 0 := Nat.mod_eq_zero_of_dvd hk
  have hA' : opVal p k A i = rowVal p k a := by unfold opVal; rw [hA]
  have hB' : opVal p k B j = rowVal p k b := by unfold opVal; rw [hB]
  have hgetD : ∀ F : Matrix (Fin k) (Fin 32) Nib, ∀ q (l : Fin k), (colF F q).getD l 0 = F l q :=
    fun F q l => getD_ofFn (fun t => F t q) 0 l
  have hsh : ∀ {x y : List Nib} {sx sy : List (Fin 256)} {v : ℚ}, x.length = k → Fp4Freeze.Shaped p x y sx sy →
      Fp4Freeze.ExactRun p [] 0 x y sx sy (k / p.k) → v = flatDot p x y sx sy → ChainNear p x y sx sy v 0 := by
    intro x y sx sy v hx hs hr hv
    subst hv
    exact chainNear_exact hp hs (hx ▸ hr)
  refine ⟨hsh hla ⟨hba, hmod, hsa, hsb⟩ hC ?_, fun q => hsh hla ?_ (hAF q) ?_,
    fun q => hsh hlb ?_ (hBF q) ?_, fun q q' => hsh (hcol FB q) ?_ (hG q q') ?_⟩
  · rw [flatDot_fin p hla]
    unfold cQ4
    exact Finset.sum_congr rfl fun l _ => by rw [hA', hB']; rfl
  · exact ⟨(hcol FB q).trans hla.symm, hmod, hsa, by rw [hones, hla]⟩
  · rw [flatDot_fin p hla]
    unfold afQ4
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [hA', hgetD, scaleAt_ones p hp hk l.isLt]
    simp [rowVal, basisVal]
  · exact ⟨(hcol FA q).trans hlb.symm, by rw [hlb]; exact hmodk, by rw [hlb, hsb, hla],
      by rw [hones, hlb]⟩
  · rw [flatDot_fin p hlb]
    unfold bfQ4
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [hB', hgetD, scaleAt_ones p hp hk l.isLt]
    simp [rowVal, basisVal]
  · exact ⟨(hcol FA q').trans (hcol FB q).symm, by rw [hcol]; exact hmodk, by rw [hones, hcol],
      by rw [hones, hcol]⟩
  · rw [flatDot_fin p (hcol FB q)]
    unfold gQ4
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [hgetD, hgetD, scaleAt_ones p hp hk l.isLt]
    simp [basisVal]

/-- **U's steps on the K16 range** over the served rows (`PeelExactProofs.URange`). -/
def URange4 (i : Fin m) (j : Fin n) : Prop :=
  ∀ c pa pb, peelRows4 p A B FA FB beA beB i j = some (c, pa, pb) → URange pa pb (List.range 4) c

/-- **The range of (b)**: the chains within their `δ`s, and U's steps on the K16 range. -/
def PeelRange4 (i : Fin m) (j : Fin n) : Prop :=
  ChainsNear4 p A B FA FB δC δA δB δG i j ∧ URange4 p A B FA FB beA beB i j

/-- `P_A,i`'s rounded entry `32 + q`. -/
def paErr4 (i : Fin m) (q : Fin 32) : ℚ :=
  δA q + (|afQ4 (opVal p k A) (basisVal FB) i q| + δA q) / 2 ^ 8 + 2 ^ (-134 : ℤ)

/-- `P_B,j`'s rounded entry `q`. -/
def pbErr4 (j : Fin n) (q : Fin 32) : ℚ :=
  pbErrOf (beB j) (fun p' => gQ4 (basisVal FA) (basisVal FB) p' q) (fun p' => δG p' q)
    (bfQ4 (opVal p k B) (basisVal FA) j q) (δB q)

/-- Position `t`'s row error: one row's entry error times the other row's exact entry. -/
def rowErr4 (i : Fin m) (j : Fin n) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then
    |paQ4 (opVal p k A) (basisVal FB) (beVal beA) i t| * pbErr4 p B FA FB beB δB δG j ⟨t.val, h⟩
  else
    paErr4 p A FB δA i ⟨t.val - 32, by have := t.isLt; omega⟩ *
      |pbQ4 (opVal p k B) (basisVal FA) (basisVal FB) (beVal beB) j t|

/-- **(b)'s bound**: U's four steps, `((1 + eps16)^4 − 1)·(|C̃| + Σ_t |P_A,t·P_B,t| + errors)`, `C̃`'s `δC`, and each
row entry's error times the other row's exact entry. -/
def peelBound4 (i : Fin m) (j : Fin n) : ℚ :=
  ((1 + eps16) ^ 4 - 1) * (|cQ4 (opVal p k A) (opVal p k B) i j| + δC +
      ∑ t, (|paQ4 (opVal p k A) (basisVal FB) (beVal beA) i t *
          pbQ4 (opVal p k B) (basisVal FA) (basisVal FB) (beVal beB) j t| +
        rowErr4 p A B FA FB beA beB δA δB δG i j t)) +
    δC + ∑ t, rowErr4 p A B FA FB beA beB δA δB δG i j t

/-- **(b)'s bound with exact chains**: U's own rounding (the FP32 FMAs and add, `RN_bf16`, the four K16 steps). -/
def peelBound4Exact (i : Fin m) (j : Fin n) : ℚ :=
  peelBound4 p A B FA FB beA beB 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) i j

/-- **(b) The rounding is bounded**: wherever `peelFp4`'s entry is finite, on `PeelRange4`, it is within `peelBound4` of
its exact counterpart `uQ4` on the served operands. -/
theorem peelFp4_err (i : Fin m) (j : Fin n) {v : ℚ} (hv : (peelFp4 p A B FA FB beA beB i j).bind wordQ = some v)
    (hR : PeelRange4 p A B FA FB beA beB δC δA δB δG i j) :
    |v - uQ4 (opVal p k A) (opVal p k B) (basisVal FA) (basisVal FB) (beVal beA) (beVal beB) i j| ≤
      peelBound4 p A B FA FB beA beB δC δA δB δG i j := by
  set A' := opVal p k A
  set B' := opVal p k B
  set FA' := basisVal FA
  set FB' := basisVal FB
  rw [peelFp4_eq_rows] at hv
  obtain ⟨u, hu, huv⟩ := Option.bind_eq_some_iff.1 hv
  obtain ⟨⟨c, pa, pb⟩, hrows, hu'⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨a, b, af, bf, g, hA, hB, hc, haf, hbf, hg, hpa, hpb⟩ := peelRows4_spec hrows
  obtain ⟨hC, hAF, hBF, hG⟩ := hR.1 a b hA hB
  obtain ⟨cv, hcv, hce⟩ := hC c hc
  obtain ⟨uv, huv', he⟩ := uWord_err pa pb hcv hu' (hR.2 c pa pb hrows)
  rw [huv] at huv'
  cases huv'
  obtain ⟨hafl, hafw⟩ := mapM_ofFn _ 0 haf
  obtain ⟨-, hbfw⟩ := mapM_ofFn _ 0 hbf
  obtain ⟨hgl, hgw⟩ := mapM_ofFn _ [] hg
  have hgw' : ∀ q q' : Fin 32, ch4 p (colF FB q) (colF FA q') (onesScales p k) (onesScales p k) =
      some ((g.getD q []).getD q' 0) := fun q q' => (mapM_ofFn _ 0 (hgw q)).2 q'
  obtain ⟨hpa1, hpa2⟩ := paRow_spec hpa (by simp)
  obtain ⟨hpb1, hpb2⟩ := pbRow_spec hpb (by simp)
  have hrow : ∀ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) -
      paQ4 A' FB' (beVal beA) i t * pbQ4 B' FA' FB' (beVal beB) j t| ≤ rowErr4 p A B FA FB beA beB δA δB δG i j t := by
    intro t
    by_cases ht : t.val < 32
    · have e1 : bfVal (pa.getD t 0) = paQ4 A' FB' (beVal beA) i t := by
        rw [hpa1 t ht, show (t : Nat) = ((⟨t.val, ht⟩ : Fin 32) : Nat) from rfl, getD_ofFn]
        unfold paQ4; rw [dite_eq_left ht]; rfl
      obtain ⟨acc, s, hacc, hs, hh⟩ := hpb1 t ht
      have hpbe := pbEntry_err (be := beB j) (G := fun p' => gQ4 FA' FB' p' ⟨t.val, ht⟩)
        (δG := fun p' => δG p' ⟨t.val, ht⟩) hgl (fun p' => hG p' ⟨t.val, ht⟩ _ (hgw' p' ⟨t.val, ht⟩))
        (hBF ⟨t.val, ht⟩ _ (hbfw ⟨t.val, ht⟩)) hacc hs hh
      have e2 : pbQ4 B' FA' FB' (beVal beB) j t =
          (∑ p', bfVal (beB j p') * gQ4 FA' FB' p' ⟨t.val, ht⟩) - bfQ4 B' FA' j ⟨t.val, ht⟩ := by
        unfold pbQ4 accQ4; rw [dite_eq_left ht]; rfl
      unfold rowErr4
      rw [dite_eq_left ht, e1, ← mul_sub, abs_mul, e2]
      exact mul_le_mul_of_nonneg_left hpbe (abs_nonneg _)
    · have hp : t.val - 32 < 32 := by have := t.isLt; omega
      have e1 : bfVal (pb.getD t 0) = pbQ4 B' FA' FB' (beVal beB) j t := by
        have h1 := hpb2 (t.val - 32) hp
        rw [show 32 + (t.val - 32) = t.val by omega,
          show t.val - 32 = ((⟨t.val - 32, hp⟩ : Fin 32) : Nat) from rfl, getD_ofFn] at h1
        rw [h1, bfVal_xor]
        unfold pbQ4; rw [dite_eq_right ht]; rfl
      have hpa' : rnBf16? (af.getD (t.val - 32) 0) = some (pa.getD t 0) := by
        have h1 := hpa2 (t.val - 32) (by rw [hafl]; exact hp)
        rwa [show 32 + (t.val - 32) = t.val by omega] at h1
      have hpae := paEntry_err (A := afQ4 A' FB' i ⟨t.val - 32, hp⟩) (δ := δA ⟨t.val - 32, hp⟩)
        (hAF ⟨t.val - 32, hp⟩ _ (hafw _)) hpa'
      have e2 : paQ4 A' FB' (beVal beA) i t = afQ4 A' FB' i ⟨t.val - 32, hp⟩ := by
        unfold paQ4; rw [dite_eq_right ht]
      unfold rowErr4
      rw [dite_eq_right ht, e1, ← sub_mul, abs_mul, e2]
      exact mul_le_mul_of_nonneg_right hpae (abs_nonneg _)
  have hq := uQ4_eq_sum A' B' FA' FB' (beVal beA) (beVal beB) i j
  rw [hq]
  set S := ∑ t : Fin 64, bfVal (pa.getD t 0) * bfVal (pb.getD t 0)
  set Sq := ∑ t, paQ4 A' FB' (beVal beA) i t * pbQ4 B' FA' FB' (beVal beB) j t
  have hd : |S - Sq| ≤ ∑ t, rowErr4 p A B FA FB beA beB δA δB δG i j t := by
    rw [← Finset.sum_sub_distrib]
    exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun t _ => hrow t)
  have hCa : |cv| ≤ |cQ4 A' B' i j| + δC := by
    have := abs_sub_abs_le_abs_sub cv (cQ4 A' B' i j); linarith
  have hS' : ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)| ≤
      ∑ t, (|paQ4 A' FB' (beVal beA) i t * pbQ4 B' FA' FB' (beVal beB) j t| +
        rowErr4 p A B FA FB beA beB δA δB δG i j t) := by
    refine Finset.sum_le_sum fun t _ => ?_
    have := abs_sub_abs_le_abs_sub (bfVal (pa.getD t 0) * bfVal (pb.getD t 0))
      (paQ4 A' FB' (beVal beA) i t * pbQ4 B' FA' FB' (beVal beB) j t)
    linarith [hrow t]
  have hK : 0 ≤ (1 + eps16) ^ 4 - 1 := by
    have := one_le_pow₀ (n := 4) (by linarith [eps16_nonneg] : (1 : ℚ) ≤ 1 + eps16); linarith
  have hce' : |cv - cQ4 A' B' i j| ≤ δC := hce
  calc |v - (cQ4 A' B' i j + Sq)| = |(v - (cv + S)) + (cv - cQ4 A' B' i j) + (S - Sq)| := by ring_nf
    _ ≤ |v - (cv + S)| + |cv - cQ4 A' B' i j| + |S - Sq| := abs_add_three _ _ _
    _ ≤ ((1 + eps16) ^ 4 - 1) * (|cv| + ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)|) + δC +
        ∑ t, rowErr4 p A B FA FB beA beB δA δB δG i j t := add_le_add (add_le_add he hce') hd
    _ ≤ peelBound4 p A B FA FB beA beB δC δA δB δG i j := by
        unfold peelBound4
        exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left (by linarith) hK) le_rfl) le_rfl

/-- **(b) on exact runs**: where the entry's FP4 chains are exact runs (`ExactRuns4`, through `Fp4Freeze.cTilde_dot`)
and U's steps are on the K16 range, the served entry is within `peelBound4Exact`, U's own rounding, of `uQ4`. At
`sm120Nvf4`, `hp` is `Fp4Freeze.sm120Nvf4_blockShape` and `hk` is `64 ∣ k`. -/
theorem peelFp4_err_exact (hp : Fp4Freeze.BlockShape p) (hk : p.k ∣ k) (i : Fin m) (j : Fin n) {v : ℚ}
    (hv : (peelFp4 p A B FA FB beA beB i j).bind wordQ = some v) (hX : ExactRuns4 p A B FA FB i j)
    (hU : URange4 p A B FA FB beA beB i j) :
    |v - uQ4 (opVal p k A) (opVal p k B) (basisVal FA) (basisVal FB) (beVal beA) (beVal beB) i j| ≤
      peelBound4Exact p A B FA FB beA beB i j :=
  peelFp4_err p A B FA FB beA beB 0 _ _ _ i j hv ⟨chainsExact4_of_runs p A B FA FB hp hk i j hX, hU⟩

end Bound

/-! ## (c) Salt-dependence at `sm120Nvf4` -/

/-- The witness row: `x ≡ 1` (BF16 `1.0`), `k = 128`. -/
def wX : Fin 128 → ℚ := fun _ => 1

/-- The pin's `α` at `x ≡ 1` (`pearl_c4.stats`): `2688/(1 + 4·(1/4)) = 1344`, the word `0x44A80000`. -/
def wAlpha : ℚ := wordVal 0x44A80000

/-- The pin's `β` at `x ≡ 1`: `f32(f32(1/4·1344)·f32(16/√2346.125))`, the word `0x42DDFADB` (`a_E = 0x6E`). -/
def wBeta : ℚ := wordVal 0x42DDFADB

/-- A constant line of code `c`. -/
def wLine (c : Nib) : Fin 32 → Nib := fun _ => c

/-- Salt 2's `E_A`: code `0x4` (`2.0`) at position 0, `0x2` (`1.0`) elsewhere. -/
def wLine2 : Fin 32 → Nib := fun r => if r.val = 0 then 4 else 2

/-- Both salts' bases: every code `0x2` (`1.0`). -/
def wBasis : Matrix (Fin 128) (Fin 32) Nib := fun _ _ => 2

/-- The served U word under the salt whose lines are `EA`, `EB` and bases `FA`, `FB`: `A = B = x ≡ 1`, each formed by
`formRow` at the pin's scales (`A` against `F_A`, `B` against `F_B`), then `peelFp4 sm120Nvf4` with
`Ẽ = eTilde E (e4m3 β)`. -/
def wServed4 (EA EB : Fin 32 → Nib) (FA FB : Matrix (Fin 128) (Fin 32) Nib) : Option Nat :=
  peelFp4 sm120Nvf4 (m := 1) (n := 1) (fun _ => formRow wAlpha wBeta wX EA FA) (fun _ => formRow wAlpha wBeta wX EB FB)
    FA FB (fun _ => eTilde EA (e4m3 wBeta)) (fun _ => eTilde EB (e4m3 wBeta)) 0 0

/-- The formed row under both salts: every code `0x7` (`6.0`, saturated) and every scale byte `0x78`. -/
def wRow : Row4 := (List.replicate 128 7, List.replicate 8 0x78)

/-- Salt 1's `P_A`: `Ẽ_A = 0x40E0` (`1·a_E/16 = 7`) and `bf16(A′·F_B) = 0x4840` (`192`). -/
def wPa1 : List Nat := List.replicate 32 0x40E0 ++ List.replicate 32 0x4840

/-- Salt 2's `P_A`: `Ẽ_A`'s first entry `0x4160` (`14`). -/
def wPa2 : List Nat := 0x4160 :: List.replicate 31 0x40E0 ++ List.replicate 32 0x4840

/-- Both salts' `P_B`: `bf16(Ẽ_B·G − B̃·F_A) = 0xC824` (`−164`) and `−Ẽ_B = 0xC0E0`. -/
def wPb : List Nat := List.replicate 32 0xC824 ++ List.replicate 32 0xC0E0

theorem wForm1 : formRow wAlpha wBeta wX (wLine 2) wBasis = some wRow := by decide +kernel

theorem wForm2 : formRow wAlpha wBeta wX wLine2 wBasis = some wRow := by decide +kernel

theorem wC : ch4 sm120Nvf4 wRow.1 wRow.1 wRow.2 wRow.2 = some 0x4D900000 := by decide +kernel

theorem wAF0 : ch4 sm120Nvf4 wRow.1 (colF wBasis 0) wRow.2 (onesScales sm120Nvf4 128) = some 0x48400000 := by
  decide +kernel

theorem wG0 : ch4 sm120Nvf4 (colF wBasis 0) (colF wBasis 0) (onesScales sm120Nvf4 128) (onesScales sm120Nvf4 128) =
    some 0x43000000 := by decide +kernel

/-- `A′·F_B` (and `B̃·F_A`): every column of the constant basis is column 0. -/
theorem wAF : (List.ofFn fun q => ch4 sm120Nvf4 wRow.1 (colF wBasis q) wRow.2 (onesScales sm120Nvf4 128)).mapM id =
    some (List.ofFn fun _ : Fin 32 => 0x48400000) :=
  mapM_ofFn_some _ _ fun _ => wAF0

theorem wG : (List.ofFn fun q => (List.ofFn fun q' => ch4 sm120Nvf4 (colF wBasis q) (colF wBasis q')
    (onesScales sm120Nvf4 128) (onesScales sm120Nvf4 128)).mapM id).mapM id =
    some (List.ofFn fun _ : Fin 32 => List.ofFn fun _ : Fin 32 => 0x43000000) :=
  mapM_ofFn_some _ _ fun _ => mapM_ofFn_some _ _ fun _ => wG0

theorem wPA1 : paRow? (List.ofFn (eTilde (wLine 2) (e4m3 wBeta))) (List.ofFn fun _ : Fin 32 => 0x48400000) =
    some wPa1 := by decide +kernel

theorem wPA2 : paRow? (List.ofFn (eTilde wLine2 (e4m3 wBeta))) (List.ofFn fun _ : Fin 32 => 0x48400000) =
    some wPa2 := by decide +kernel

theorem wPB : pbRow? (List.ofFn (eTilde (wLine 2) (e4m3 wBeta))) (List.ofFn fun _ : Fin 32 => 0x48400000)
    (List.ofFn fun _ : Fin 32 => List.ofFn fun _ : Fin 32 => 0x43000000) = some wPb := by decide +kernel

theorem wU1 : uWord? 0x4D900000 wPa1 wPb = some 0x4D522000 := by decide +kernel

theorem wU2 : uWord? 0x4D900000 wPa2 wPb = some 0x4D510100 := by decide +kernel

/-- A salt's served word from its parts: both rows formed to `wRow`, then `P_A`, `P_B` and U's steps. -/
theorem wServed4_of_parts {EA EB : Fin 32 → Nib} {pa pb : List Nat} {u : Nat}
    (hA : formRow wAlpha wBeta wX EA wBasis = some wRow) (hB : formRow wAlpha wBeta wX EB wBasis = some wRow)
    (hpa : paRow? (List.ofFn (eTilde EA (e4m3 wBeta))) (List.ofFn fun _ : Fin 32 => 0x48400000) = some pa)
    (hpb : pbRow? (List.ofFn (eTilde EB (e4m3 wBeta))) (List.ofFn fun _ : Fin 32 => 0x48400000)
      (List.ofFn fun _ : Fin 32 => List.ofFn fun _ : Fin 32 => 0x43000000) = some pb)
    (hu : uWord? 0x4D900000 pa pb = some u) : wServed4 EA EB wBasis wBasis = some u :=
  (peelFp4_of_parts (A := fun _ : Fin 1 => formRow wAlpha wBeta wX EA wBasis)
    (B := fun _ : Fin 1 => formRow wAlpha wBeta wX EB wBasis) (beA := fun _ => eTilde EA (e4m3 wBeta))
    (beB := fun _ => eTilde EB (e4m3 wBeta)) (i := 0) (j := 0) hA hB wC wAF wAF wG hpa hpb).trans hu

/-- **The two salts' served words**: `0x4D522000` (`220,332,032`) under the all-`1.0` lines and `0x4D510100`
(`219,156,480`) with `E_A`'s first code `2.0`, assembled from their parts by `peelFp4_of_parts`, each part by the kernel.
`pearl_c4.py` gives the same two words. The formed rows and `C̃ = 0x4D900000` are the same under both salts (the codes
saturate at `6`), so U moves with the noise it cancels. -/
theorem wServed4_words : wServed4 (wLine 2) (wLine 2) wBasis wBasis = some 0x4D522000 ∧
    wServed4 wLine2 (wLine 2) wBasis wBasis = some 0x4D510100 :=
  ⟨wServed4_of_parts wForm1 wForm1 wPA1 wPB wU1, wServed4_of_parts wForm2 wForm1 wPA2 wPB wU2⟩

/-- **(c) Salt-dependence at `sm120Nvf4`**: the two salts' served entries differ, as values. -/
theorem served4_salt_dependent :
    (wServed4 (wLine 2) (wLine 2) wBasis wBasis).bind wordQ ≠ (wServed4 wLine2 (wLine 2) wBasis wBasis).bind wordQ := by
  rw [wServed4_words.1, wServed4_words.2]
  decide +kernel

end Pouw.PearlC.PeelFp4Proofs
