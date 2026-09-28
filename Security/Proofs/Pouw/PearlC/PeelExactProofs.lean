import Definitions.Pouw.PearlC.PeelExact
import Proofs.Pouw.PearlC.Peel
import Proofs.Pouw.PearlC.AtomErr

/-!
# P2 on the served U (staging, nothing pinned)

`Peel.lean`'s clean-up facts (a)–(c), proved on `PeelExact.peelExact`, the pinned order the verifier replays, rather
than on exact peel factors.

**(a) The noise cancels exactly.** Each pinned step has an exact counterpart, with every truncation and rounding
replaced by the exact value: the chains `C̃_ij`, `A′·F_B`, `B̃·F_A` and `G = F_Bᵀ·F_A` (`chainQ`: `cQ`, `afQ`, `bfQ`,
`gQ`), the rows `P_A,i = [be_A,i ‖ A′·F_B]` (`paQ`) and `P_B,j = [Σ_p be_B,jp·G_pq − B̃·F_A ‖ −be_B,j]` (`accQ`, `pbQ`),
and U's four K16 steps from `C̃_ij` (`k16Q`, `uQ`). When `32 ∣ k`, `uQ` is #311's U with unit `β` on the served noise
(`uQ_eq_peelUQ`, on rational factors since the rows are BF16 words, not codes: `peelUQ`): `E_A`, `E_B` are the values
of the peel's rows `beA`, `beB` and `F_A`, `F_B` the bases' values. So it is `(A′ − N_A)·(B̃ − N_B)ᵀ` with
`N_A = be_A·F_Aᵀ`, `N_B = be_B·F_Bᵀ` (`uQ_cancel`, by `peelUQ_cancel`). Which noise the rows hold: under v1 they are
`bf16(E′)`, and `RN_bf16` of an E4M3 value is the value (`bf16_e4m3`, all 256 codes by the kernel), so the served rows'
values are `Noise.atomLines`' noise lines (`bfVals_v1`) and `uQ` is §5's `(A′ − E′_A·F_Aᵀ)·(B̃ − E′_B·F_Bᵀ)ᵀ`
(`uQ_cancel_v1`, as `Peel.peelU_cancel_atomMix`). Under v0 they are
`bf16(β·bf16(E))`, each within `(2^−7 + 2^−16)·|β·e|` of `β·e` (`bf16_v0_err`), so the served `N_A` is within
`(2^−7 + 2^−16)·Σ_t |β·e_t·f_t|` of `β·N_A` (`servedNA_v0_err`): under v0 the peel cancels the noise it is served, which
is `β·N` up to those two BF16 roundings.

**(b) The rounding is bounded.** Wherever the served entry is finite, `peelExactVal_err` puts it within `peelBound` of
`uQ`, on the named range `PeelRange`. The range asks, for every E4M3 chain of the entry (`C̃_ij`, the rows of `A′·F_B`,
`B̃·F_A` and `G`), `ChainRange`: each `hopperStep` on `H8Range`, its codes zero or normal (`ZeroOrNormal8`, no E4M3
subnormal) and its exponent in `[−113, 120]` unless every term is zero. For U's four K16 steps it asks `URange`: each
step on `K16Range`, its BF16 words zero or normal and its exponent in `[−101, 121]` unless every term is zero. FP32 adds
and FMAs need no range: a finite result is within `|v|·2^−24 + 2^−150` (`addWord_err`, `fmaWord_err`), and `RN_bf16`
within `|x|·2^−8 + 2^−134` (`rnBf16_err`). The stages: one K16 step is within `eps16 ≈ 2^−20.6` of its exact sum,
relative to `|acc| + Σ|x·y|` (`k16Step_rel`); one E4M3 step within `eps8 ≈ 2^−7.9` (`hopperStep_rel`); an E4M3 chain
of `w` windows within `chainBound`, about `(epsC + w·2^−24)·Σ|a·b|` with `epsC ≈ 2^−5.9` a window of at most four atoms
and its promotion (`chainWord_err`: only the promotions' `2^−24` compounds, by `fold_rel2`); and `P_B`'s FMA chain within
`(1 + 2^−24)^32 − 1` (`accChain_err`). `peelBound` composes them in exact quantities: U's steps
`((1 + eps16)^4 − 1)·(|C̃| + Σ_t |P_A,t·P_B,t| + errors)`, `C̃`'s `chainBound`, and each row entry's error times the
other row's exact entry (`rowErr`, from `paEntry_err` and `pbEntry_err`).

**(c) Salt-dependence**, by witness: at `Peel.lean`'s degenerate inputs (`x = 0` with `β = 1`, `B = 0` with
`β = 1/3`, `m = n = 1`, `k = 32`), the salts whose noises are `wNoise 1 1` and `wNoise 2 (−1)` give the served entries
`0` and `10` (`served_live`, `served_salt_dependent`). For all inputs, (a) says what the entry is up to (b)'s rounding:
`(A′ − N_A)·(B̃ − N_B)ᵀ`, a function of the salt through `N_A` and `N_B`.
-/

namespace Pouw.PearlC.PeelExactProofs

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Matrix

variable {m n k : Nat}

/-! ## (a) The noise cancels exactly -/

/-- A BF16 word's value: the FP32 word `h <<< 16`'s, `0` where that is not finite. -/
def bfVal (h : Nat) : ℚ := (wordQ (h <<< 16)).getD 0

/-- #311's U at unit `β` on rational factors: `C̃ + [E_A ‖ A′·F_B]·[(E_B·F_Bᵀ − B̃)·F_A ‖ −E_B]ᵀ`, `peelU`'s form for
noise rows that are BF16 values rather than E4M3 codes, as the served rows are. -/
def peelUQ (Ct : Matrix (Fin m) (Fin n) ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (B' : Matrix (Fin n) (Fin k) ℚ)
    (EA : Matrix (Fin m) (Fin 32) ℚ) (FA : Matrix (Fin k) (Fin 32) ℚ) (EB : Matrix (Fin n) (Fin 32) ℚ)
    (FB : Matrix (Fin k) (Fin 32) ℚ) : Matrix (Fin m) (Fin n) ℚ :=
  Ct + fromCols EA (A' * FB) * (fromCols ((EB * FBᵀ - B') * FA) (-EB))ᵀ

/-- `peelUQ`'s cancellation identity, `Peel.peelU_cancel`'s on rational factors: with `C̃ = A′·B̃ᵀ` exact, it is
`(A′ − E_A·F_Aᵀ)·(B̃ − E_B·F_Bᵀ)ᵀ`. -/
theorem peelUQ_cancel (A' : Matrix (Fin m) (Fin k) ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (EA : Matrix (Fin m) (Fin 32) ℚ)
    (FA : Matrix (Fin k) (Fin 32) ℚ) (EB : Matrix (Fin n) (Fin 32) ℚ) (FB : Matrix (Fin k) (Fin 32) ℚ) :
    peelUQ (A' * B'ᵀ) A' B' EA FA EB FB = (A' - EA * FAᵀ) * (B' - EB * FBᵀ)ᵀ := by
  simp only [peelUQ, transpose_fromCols, fromCols_mul_fromRows, transpose_mul, transpose_sub, transpose_neg,
    Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_neg, Matrix.mul_assoc]
  abel

/-- One atom's dot product, exact. -/
def atomQ (x : AtomIn) : ℚ := ∑ l, codeVal (x.1 l) * codeVal (x.2 l)

/-- `chainWord?`'s exact counterpart, every `hopperStep` and every promotion exact: the sum of the atoms' dot products. -/
def chainQ (atoms : List AtomIn) : ℚ := (atoms.map atomQ).sum

/-- `k16Step`'s exact counterpart: the accumulator plus the 16 products. -/
def k16Q (acc : ℚ) (x y : Fin 16 → ℚ) : ℚ := acc + ∑ l, x l * y l

/-- `slice16` on a row of values: positions `16s, …, 16s + 15`. -/
def sliceQ (row : Fin 64 → ℚ) (s : Nat) : Fin 16 → ℚ :=
  fun l => if h : 16 * s + l.val < 64 then row ⟨16 * s + l.val, h⟩ else 0

section Exact

variable (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
  (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)

/-- `C̃_ij`'s chain, exact. -/
def cQ (i : Fin m) (j : Fin n) : ℚ := chainQ (rowAtoms (Aq i) (Bq j))

/-- `(A′·F_B)_ip`'s chain, exact. -/
def afQ (i : Fin m) (p : Fin 32) : ℚ := chainQ (rowAtoms (Aq i) (FBᵀ p))

/-- `(B̃·F_A)_jq`'s chain, exact. -/
def bfQ (j : Fin n) (q : Fin 32) : ℚ := chainQ (rowAtoms (Bq j) (FAᵀ q))

/-- `G_pq = (F_Bᵀ·F_A)_pq`'s chain, exact. -/
def gQ (p q : Fin 32) : ℚ := chainQ (rowAtoms (FBᵀ p) (FAᵀ q))

/-- `P_A,i` exact: `[be_A,i ‖ A′·F_B]`, the `RN_bf16` of the second block dropped. -/
def paQ (i : Fin m) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then bfVal (beA i ⟨t.val, h⟩) else afQ Aq FB i ⟨t.val - 32, by have := t.isLt; omega⟩

/-- `acc_q`'s FMA chain, exact: `Σ_p be_B,jp·G_pq`. -/
def accQ (j : Fin n) (q : Fin 32) : ℚ := ∑ p, bfVal (beB j p) * gQ FA FB p q

/-- `P_B,j` exact: `[acc_q − (B̃·F_A)_jq ‖ −be_B,j]`, the FP32 difference's rounding and its `RN_bf16` dropped. -/
def pbQ (j : Fin n) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then accQ FA FB beB j ⟨t.val, h⟩ - bfQ Bq FA j ⟨t.val, h⟩
  else -bfVal (beB j ⟨t.val - 32, by have := t.isLt; omega⟩)

/-- **U's exact counterpart in the pinned order**: from `C̃_ij`, four exact K16 steps over the slices of `P_A,i` and
`P_B,j`. -/
def uQ (i : Fin m) (j : Fin n) : ℚ :=
  (List.range 4).foldl (fun acc s => k16Q acc (sliceQ (paQ Aq FB beA i) s) (sliceQ (pbQ Bq FA FB beB j) s))
    (cQ Aq Bq i j)

end Exact

/-- The exact chain over one output's atoms is the dot product of the two rows, when `32 ∣ k`. -/
theorem chainQ_rowAtoms (hk : 32 ∣ k) (a b : Fin k → Code) :
    chainQ (rowAtoms a b) = ∑ l, codeVal (a l) * codeVal (b l) := by
  unfold chainQ rowAtoms atomQ
  rw [List.map_ofFn, List.sum_ofFn]
  simp only [Function.comp_def]
  rw [← Fintype.sum_prod_type']
  refine Fintype.sum_equiv (finProdFinEquiv.trans (finCongr (Nat.div_mul_cancel hk))) _ _ (fun x => ?_)
  have e : (finProdFinEquiv.trans (finCongr (Nat.div_mul_cancel hk))) x =
      ⟨32 * x.1.val + x.2.val, by have := x.1.isLt; have := x.2.isLt; omega⟩ := by
    ext; simp [finProdFinEquiv]; ring
  rw [e]

/-- The four exact K16 steps add the 64 products `P_A,i t · P_B,j t` to the accumulator. -/
theorem foldl_k16Q (c : ℚ) (pa pb : Fin 64 → ℚ) :
    (List.range 4).foldl (fun acc s => k16Q acc (sliceQ pa s) (sliceQ pb s)) c = c + ∑ t, pa t * pb t := by
  rw [show List.range 4 = [0, 1, 2, 3] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, k16Q]
  have h : ∑ t, pa t * pb t = ∑ s : Fin 4, ∑ l : Fin 16, sliceQ pa s l * sliceQ pb s l := by
    rw [← Fintype.sum_prod_type']
    refine (Fintype.sum_equiv (finProdFinEquiv : Fin 4 × Fin 16 ≃ Fin 64) _ _ (fun x => ?_)).symm
    have hx : 16 * x.1.val + x.2.val < 64 := by have := x.1.isLt; have := x.2.isLt; omega
    have e : (finProdFinEquiv : Fin 4 × Fin 16 ≃ Fin 64) x = ⟨16 * x.1.val + x.2.val, hx⟩ := by
      ext; simp [finProdFinEquiv]; ring
    simp only [sliceQ, hx, ↓reduceDIte, e]
  rw [h, Fin.sum_univ_four]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two]
  norm_num
  ring

theorem rowScale_one {p q : Nat} (M : Matrix (Fin p) (Fin q) ℚ) : rowScale (fun _ => 1) M = M := by
  simp only [rowScale, diagonal_one, Matrix.one_mul]

section Exact

variable (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
  (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)

/-- **The exact counterpart is #311's U on the served noise**: with unit `β`, `E_A`, `E_B` the values of `beA`, `beB`
and `F_A`, `F_B` the bases' values (`peelUQ`). -/
theorem uQ_eq_peelUQ (hk : 32 ∣ k) (i : Fin m) (j : Fin n) :
    uQ Aq Bq FA FB beA beB i j =
      peelUQ (ctExact Aq Bq) (vals Aq) (vals Bq) (beA.map bfVal) (vals FA) (beB.map bfVal) (vals FB) i j := by
  unfold uQ
  rw [foldl_k16Q]
  simp only [peelUQ, Matrix.add_apply, Matrix.mul_apply, transpose_apply]
  congr 1
  · rw [cQ, chainQ_rowAtoms hk]; simp [ctExact, Matrix.mul_apply, vals]
  · refine (Fintype.sum_equiv (finSumFinEquiv : Fin 32 ⊕ Fin 32 ≃ Fin 64) _ _ ?_).symm
    rintro (t | t)
    · have hv : ((finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) : Nat) = t := rfl
      have e1 : paQ Aq FB beA i (finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) = bfVal (beA i t) := by
        simp [paQ, hv]
      have e2 : pbQ Bq FA FB beB j (finSumFinEquiv (m := 32) (n := 32) (Sum.inl t)) =
          accQ FA FB beB j t - bfQ Bq FA j t := by
        simp [pbQ, hv]
      rw [e1, e2]
      simp only [fromCols_apply_inl, Matrix.map_apply]
      congr 1
      simp only [accQ, gQ, bfQ, chainQ_rowAtoms hk, Matrix.mul_apply, Matrix.sub_apply, transpose_apply, vals,
        Matrix.map_apply]
      simp only [Finset.sum_mul, Finset.mul_sum, sub_mul, Finset.sum_sub_distrib]
      rw [Finset.sum_comm]
      simp only [mul_assoc]
    · have hv : ((finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) : Nat) = 32 + t := rfl
      have e1 : paQ Aq FB beA i (finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) = afQ Aq FB i t := by
        simp [paQ, hv]
      have e2 : pbQ Bq FA FB beB j (finSumFinEquiv (m := 32) (n := 32) (Sum.inr t)) = -bfVal (beB j t) := by
        simp [pbQ, hv]
      rw [e1, e2]
      simp only [fromCols_apply_inr]
      simp only [afQ, chainQ_rowAtoms hk, Matrix.mul_apply, Matrix.neg_apply, vals, Matrix.map_apply, transpose_apply,
        mul_neg]

/-- **(a) The noise cancels exactly**: U's exact counterpart in the pinned order is `(A′ − N_A)·(B̃ − N_B)ᵀ` for the
served noise `N_A = be_A·F_Aᵀ`, `N_B = be_B·F_Bᵀ` (on values). -/
theorem uQ_cancel (hk : 32 ∣ k) (i : Fin m) (j : Fin n) :
    uQ Aq Bq FA FB beA beB i j =
      ((vals Aq - beA.map bfVal * (vals FA)ᵀ) * (vals Bq - beB.map bfVal * (vals FB)ᵀ)ᵀ) i j := by
  rw [uQ_eq_peelUQ Aq Bq FA FB beA beB hk, ctExact, peelUQ_cancel]

end Exact

/-! ### Which noise `beA` and `beB` hold -/

/-- **`RN_bf16` of an E4M3 value is the value**: the pin's `bf16(E′)` is `E′` exactly, for each of the 256 codes (an
E4M3 value has a 4-bit significand and an exponent in `[−9, 8]`, inside BF16's normal range). -/
theorem bf16_e4m3 : ∀ c : Code, (rnBf16? (f32Word (codeVal c))).map bfVal = some (codeVal c) := by decide +kernel

/-- **Under v1** the peel's rows are `bf16(E′)`, the noise lines' codes cast to BF16 (`hb`): their values are the
lines' values, the noise lines of `Noise.atomLines`, which the atom-mix added. -/
theorem bfVals_v1 {p : Nat} (β : Fin p → ℚ) (E : Matrix (Fin p) (Fin 32) Code) (be : Matrix (Fin p) (Fin 32) Nat)
    (hb : ∀ i t, rnBf16? (f32Word (codeVal (castLines β E i t))) = some (be i t)) :
    be.map bfVal = vals (castLines β E) := by
  ext i t
  have e := bf16_e4m3 (castLines β E i t)
  rw [hb i t] at e
  exact Option.some_injective _ e

/-- **(a) under v1**: the exact counterpart of U on the atom-mix's operands, with the peel's rows `bf16(E′)` and the
lines' bases, is `(A′ − E′_A·F_Aᵀ)·(B̃ − E′_B·F_Bᵀ)ᵀ`, §5's cancellation (as `Peel.peelU_cancel_atomMix`). -/
theorem uQ_cancel_v1 (hk : 32 ∣ k) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (αA βA : Fin m → ℚ) (αB βB : Fin n → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ)
    (z : Noise m n k 32)
    (hA : ∀ i t, rnBf16? (f32Word (codeVal (castLines βA z.EA i t))) = some (beA i t))
    (hB : ∀ j t, rnBf16? (f32Word (codeVal (castLines βB z.EB j t))) = some (beB j t)) (i : Fin m) (j : Fin n) :
    uQ (FormChain.atomMix.quantA αA βA X z) (FormChain.atomMix.quantB αB βB B z) z.FA z.FB beA beB i j =
      ((vals (FormChain.atomMix.quantA αA βA X z) - vals (castLines βA z.EA) * (vals z.FA)ᵀ) *
        (vals (FormChain.atomMix.quantB αB βB B z) - vals (castLines βB z.EB) * (vals z.FB)ᵀ)ᵀ) i j := by
  rw [uQ_cancel _ _ _ _ _ _ hk, bfVals_v1 βA z.EA beA hA, bfVals_v1 βB z.EB beB hB]

/-- **Under v0** the peel's rows are `bf16(β·bf16(E))`: each served entry is within `(2^−7 + 2^−16)·|β·e|` of `β·e`,
so the served `N_A = be_A·F_Aᵀ` is within `(2^−7 + 2^−16)·Σ_t |β·e_t·f_t|` of `β·N_A` entry by entry. -/
theorem bf16_v0_err (β e : ℚ) : |bf16Rn (β * bf16Rn e) - β * e| ≤ (2 / 2 ^ 8 + 1 / 2 ^ 16) * |β * e| := by
  have e1 := Peel.bf16Rn_sub_le e
  have e2 := Peel.bf16Rn_sub_le (β * bf16Rn e)
  have hb : |bf16Rn e| ≤ |e| + |e| / 2 ^ 8 := by
    have := abs_sub_abs_le_abs_sub (bf16Rn e) e; linarith
  have hs : bf16Rn (β * bf16Rn e) - β * e = (bf16Rn (β * bf16Rn e) - β * bf16Rn e) + β * (bf16Rn e - e) := by ring
  rw [hs]
  calc _ ≤ |bf16Rn (β * bf16Rn e) - β * bf16Rn e| + |β * (bf16Rn e - e)| := abs_add_le _ _
    _ ≤ |β| * |bf16Rn e| / 2 ^ 8 + |β| * (|e| / 2 ^ 8) := by
        rw [abs_mul β (bf16Rn e - e)]; rw [abs_mul] at e2; gcongr
    _ ≤ |β| * (|e| + |e| / 2 ^ 8) / 2 ^ 8 + |β| * (|e| / 2 ^ 8) := by gcongr
    _ = _ := by rw [abs_mul]; ring

theorem servedNA_v0_err {FA : Matrix (Fin k) (Fin 32) Code} {beA : Matrix (Fin m) (Fin 32) Nat}
    (βA : Fin m → ℚ) (EA : Matrix (Fin m) (Fin 32) ℚ)
    (hA : ∀ i t, bfVal (beA i t) = bf16Rn (βA i * bf16Rn (EA i t))) (i : Fin m) (l : Fin k) :
    |(beA.map bfVal * (vals FA)ᵀ) i l - βA i * (EA * (vals FA)ᵀ) i l| ≤
      (2 / 2 ^ 8 + 1 / 2 ^ 16) * ∑ t, |βA i * EA i t * codeVal (FA l t)| := by
  simp only [Matrix.mul_apply, transpose_apply, vals, Matrix.map_apply, hA, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun t _ => ?_)
  have := bf16_v0_err (βA i) (EA i t)
  rw [show bf16Rn (βA i * bf16Rn (EA i t)) * codeVal (FA l t) - βA i * (EA i t * codeVal (FA l t)) =
    (bf16Rn (βA i * bf16Rn (EA i t)) - βA i * EA i t) * codeVal (FA l t) by ring, abs_mul, abs_mul (βA i * EA i t)]
  calc _ ≤ (2 / 2 ^ 8 + 1 / 2 ^ 16) * |βA i * EA i t| * |codeVal (FA l t)| := by gcongr
    _ = _ := by ring

/-! ## (b) The rounding is bounded -/

section Rounding

open Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs

/-! ### The K16 step -/

/-- A BF16 word's value: `±s·2^(max(E, 1) − 134)`, for exponent field `E ≠ 255`. -/
def bf16Value (h : Nat) : ℚ := (if bf16Sign h then -1 else 1) * (bf16Sig h : ℚ) * 2 ^ ((bf16ExpOr1 h : ℤ) - 134)

theorem fp32Value_bf16 (h : Nat) : fp32Value (h <<< 16) =
    if bf16Exp h = 255 then (if bf16Man h = 0 then .inf (bf16Sign h) else .nan)
    else if bf16Exp h = 0 then (if bf16Man h = 0 then .fin Term.zero else .fin ⟨bf16Sign h, -126, bf16Man h * 2 ^ 16⟩)
    else .fin ⟨bf16Sign h, (bf16Exp h : ℤ) - 127, 2 ^ 23 + bf16Man h * 2 ^ 16⟩ := by
  have h1 : (h <<< 16) >>> 23 % 256 = bf16Exp h := by
    unfold bf16Exp; rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]; omega
  have h2 : (h <<< 16) % 2 ^ 23 = bf16Man h * 2 ^ 16 := by
    unfold bf16Man; rw [Nat.shiftLeft_eq]; omega
  have h3 : ((h <<< 16) >>> 31 % 2 == 1) = bf16Sign h := by
    unfold bf16Sign; rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
    congr 1; omega
  have hm : bf16Man h * 2 ^ 16 < 2 ^ 23 := by unfold bf16Man; omega
  have h4 : 1 <<< 23 ||| bf16Man h * 2 ^ 16 = 2 ^ 23 + bf16Man h * 2 ^ 16 := by
    rw [← Nat.shiftLeft_add_eq_or_of_lt hm, Nat.one_shiftLeft]
  have h5 : (bf16Man h * 2 ^ 16 = 0) = (bf16Man h = 0) := by simp
  unfold fp32Value
  simp only [h1, h2, h3, h4, h5]

theorem wordQ_bf16 {h : Nat} (he : bf16Exp h ≠ 255) : wordQ (h <<< 16) = some (bf16Value h) := by
  unfold wordQ bf16Value bf16Sig bf16ExpOr1
  rw [fp32Value_bf16, ite_eq_right he]
  by_cases h0 : bf16Exp h = 0
  · rw [ite_eq_left h0, h0]
    by_cases hm : bf16Man h = 0
    · simp [hm, Value.finite?, Term.value, Term.zero]
    · simp only [hm, ite_false, Value.finite?, Option.map_some, Term.value, ite_true]
      congr 1
      rw [show max 0 1 = 1 from rfl, mul_assoc, mul_assoc]
      congr 1
      push_cast
      rw [mul_assoc]
      congr 1
      norm_num
  · simp only [h0, ite_false, Value.finite?, Option.map_some, Term.value]
    congr 1
    rw [show max (bf16Exp h) 1 = bf16Exp h by omega]
    have e : (2 : ℚ) ^ ((bf16Exp h : ℤ) - 127 - 23) * 2 ^ (16 : ℕ) = 2 ^ ((bf16Exp h : ℤ) - 134) := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    rw [← e]; push_cast; ring

theorem bfVal_eq {h : Nat} (he : bf16Exp h ≠ 255) : bfVal h = bf16Value h := by
  rw [bfVal, wordQ_bf16 he]; rfl

theorem sign_xor (a b : Bool) :
    (if (a ^^ b) = true then (-1 : ℚ) else 1) = (if a = true then -1 else 1) * (if b = true then -1 else 1) := by
  cases a <;> cases b <;> norm_num

theorem bf16Product_value (x y : Nat) : (bf16Product x y).value = bf16Value x * bf16Value y := by
  have e0 : bf16Value x * bf16Value y = ((if bf16Sign x then -1 else 1) * (if bf16Sign y then -1 else 1)) *
      (2 ^ ((bf16ExpOr1 x : ℤ) - 134) * 2 ^ ((bf16ExpOr1 y : ℤ) - 134)) * ((bf16Sig x : ℚ) * bf16Sig y) := by
    unfold bf16Value; ring
  unfold bf16Product
  by_cases h : (bf16Sig x * bf16Sig y) <<< 9 = 0
  · rw [ite_eq_left h]
    have : bf16Sig x * bf16Sig y = 0 := by rw [Nat.shiftLeft_eq] at h; simpa using h
    have h' : (bf16Sig x : ℚ) * bf16Sig y = 0 := by exact_mod_cast this
    rw [e0, h']
    simp [Term.value, Term.zero]
  · rw [ite_eq_right h, e0]
    simp only [Term.value, Nat.shiftLeft_eq, sign_xor]
    have e : (2 : ℚ) ^ ((bf16ExpOr1 x : ℤ) + bf16ExpOr1 y - 254 - 23) * 2 ^ (9 : ℕ) =
        2 ^ ((bf16ExpOr1 x : ℤ) - 134) * 2 ^ ((bf16ExpOr1 y : ℤ) - 134) := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero, ← zpow_add₀ two_ne_zero]
      congr 1; push_cast; ring
    rw [← e]; push_cast; ring

/-- The K16 step's alignment exponent: the largest exponent over the accumulator term and the nonzero products, at
least −133. -/
def k16Exp (tc : Term) (x y : Fin 16 → Nat) : ℤ := maxExp (-133) (tc :: List.ofFn fun l => bf16Product (x l) (y l))

/-- The K16 step on a finite accumulator term, on finite BF16 words: one group of `tc` and the 16 products. -/
theorem k16Step_fin {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (x y : Fin 16 → Nat)
    (hx : ∀ l, bf16Exp (x l) ≠ 255) (hy : ∀ l, bf16Exp (y l) ≠ 255) :
    k16Step w x y = packValue
      (let t := groupSum 26 (-133) (tc :: List.ofFn fun l => bf16Product (x l) (y l))
       if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t) := by
  have hp : (fun l => bf16ProductTotal (x l) (y l)) = fun l => Value.fin (bf16Product (x l) (y l)) := by
    funext l
    have h1 : bf16IsNaN (x l) = false := by simp [bf16IsNaN, hx l]
    have h2 : bf16IsNaN (y l) = false := by simp [bf16IsNaN, hy l]
    have h3 : bf16IsInf (x l) = false := by simp [bf16IsInf, hx l]
    have h4 : bf16IsInf (y l) = false := by simp [bf16IsInf, hy l]
    simp [bf16ProductTotal, h1, h2, h3, h4]
  unfold k16Step hopperBf16K16
  simp only [hw, hp]
  unfold groupStepTotal
  have h1 : ((Value.fin tc :: List.ofFn fun l => Value.fin (bf16Product (x l) (y l))).any Value.isNaN) = false := by
    rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isNaN]
  have h2 : ∀ s, ((Value.fin tc :: List.ofFn fun l => Value.fin (bf16Product (x l) (y l))).any (Value.isInf s)) =
      false := by
    intro s; rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isInf]
  have h3 : (Value.fin tc :: List.ofFn fun l => Value.fin (bf16Product (x l) (y l))).filterMap Value.finite? =
      tc :: List.ofFn fun l => bf16Product (x l) (y l) := by
    simp [Value.finite?]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.and_self]

/-! ### The first truncation on the 26-bit adder -/

theorem toAdder26 (m : Nat) : toAdder 26 m = 4 * m := by
  unfold toAdder; rw [ite_eq_left (by norm_num), Nat.shiftLeft_eq]; norm_num; ring

theorem aligned26_eq (E : ℤ) (t : Term) :
    aligned 26 E t = (if t.neg then -1 else 1) * ((4 * t.mag / 2 ^ (E - t.exp).toNat : ℕ) : ℤ) := by
  by_cases hm : t.mag = 0
  · simp [aligned, hm]
  · simp only [aligned, hm, ite_false, toAdder26, Nat.shiftRight_eq_div_pow]
    cases t.neg <;> simp

theorem value26_eq {E : ℤ} {t : Term} (hE : t.exp ≤ E) :
    t.value = (if t.neg then -1 else 1) * (((4 * t.mag : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat) * 2 ^ (E - 25) := by
  unfold Term.value
  have : t.exp - 23 = (E - 25) - (((E - t.exp).toNat : ℕ) : ℤ) + 2 := by omega
  rw [this, zpow_add₀ two_ne_zero, zpow_sub₀ two_ne_zero, zpow_natCast]
  push_cast
  field_simp

theorem aligned26_eq_truncZ (E : ℤ) (t : Term) :
    aligned 26 E t = truncZ ((if t.neg then -1 else 1) * (((4 * t.mag : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat)) := by
  rw [aligned26_eq]
  have hq : ((4 * t.mag / 2 ^ (E - t.exp).toNat : Nat) : Int) =
      (⌊((4 * t.mag : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat⌋₊ : Int) := by
    congr 1
    rw [← Nat.floor_div_eq_div (K := ℚ)]
    simp
  have hnn : 0 ≤ ((4 * t.mag : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat := by positivity
  rw [hq]
  generalize ((4 * t.mag : ℕ) : ℚ) / 2 ^ (E - t.exp).toNat = q at hnn ⊢
  unfold truncZ
  cases t.neg
  · simp only [Bool.false_eq_true, ite_false, one_mul, ite_eq_left hnn]
  · simp only [ite_true, neg_one_mul]
    by_cases h0 : q = 0
    · subst h0; simp
    · rw [ite_eq_right (by intro h; exact h0 (by linarith)), neg_neg]

/-- An aligned term on the 26-bit adder, on the grid `2^(E − 25)`, is its value truncated toward zero to the grid. -/
theorem aligned26_value {E : ℤ} {t : Term} (hE : t.mag ≠ 0 → t.exp ≤ E) :
    (aligned 26 E t : ℚ) * 2 ^ (E - 25) = truncTo (2 ^ (E - 25)) t.value := by
  by_cases hm : t.mag = 0
  · have : t.value = 0 := by simp [Term.value, hm]
    rw [this, truncTo_zero]; simp [aligned, hm]
  · rw [value26_eq (hE hm), aligned26_eq_truncZ, truncTo_mul (two_zpow_pos _)]

theorem aligned26_natAbs_le (E : ℤ) (t : Term) : (aligned 26 E t).natAbs ≤ 4 * t.mag := by
  rw [aligned26_eq]
  have : 4 * t.mag / 2 ^ (E - t.exp).toNat ≤ 4 * t.mag := Nat.div_le_self _ _
  split <;> simp only [neg_one_mul, one_mul, Int.natAbs_neg, Int.natAbs_natCast] <;> omega

theorem bf16Sig_le (x : Nat) : bf16Sig x ≤ 255 := by unfold bf16Sig bf16Man; split <;> omega

theorem bf16Product_mag_lt (x y : Nat) : (bf16Product x y).mag < 2 ^ 25 := by
  by_cases h : (bf16Sig x * bf16Sig y) <<< 9 = 0
  · simp [bf16Product, h, Term.zero]
  · simp only [bf16Product]
    rw [ite_eq_right h]
    show (bf16Sig x * bf16Sig y) <<< 9 < 2 ^ 25
    have := Nat.mul_le_mul (bf16Sig_le x) (bf16Sig_le y)
    rw [Nat.shiftLeft_eq]
    omega

/-! ### The second truncation, to 24 significant bits -/

/-- Truncation toward zero to 24 significant bits. -/
def trunc24 (y : ℚ) : ℚ := if y = 0 then 0 else truncTo (2 ^ (binade |y| - 23)) y

/-- Truncation to 24 significant bits moves `y` by at most `|y|·2^−23`. -/
theorem abs_trunc24_sub_le (y : ℚ) : |trunc24 y - y| ≤ |y| / 2 ^ 23 := by
  rcases eq_or_ne y 0 with rfl | hy
  · simp [trunc24]
  have h1 := Freeze.abs_sub_truncTo_lt (two_zpow_pos (binade |y| - 23)) y
  have h2 := (binade_spec (abs_pos.2 hy)).1
  have hg : (2 : ℚ) ^ (binade |y| - 23) ≤ |y| / 2 ^ 23 := by
    rw [zpow_sub₀ two_ne_zero]
    exact div_le_div_of_nonneg_right h2 (by norm_num)
  rw [trunc24, ite_eq_right hy, abs_sub_comm]
  linarith

theorem floor_shift24 (n L : ℕ) : (⌊(n : ℚ) * 2 ^ ((23 : ℤ) - L)⌋ : ℚ) =
    ((if 24 < L + 1 then n >>> (L + 1 - 24) else n <<< (24 - (L + 1)) : ℕ) : ℚ) := by
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, ← Nat.floor_div_eq_div (K := ℚ)]
    have : (23 : ℤ) - L = -((L + 1 - 24 : ℕ) : ℤ) := by omega
    rw [this, zpow_neg, zpow_natCast, ← div_eq_mul_inv]
    have hnn : (0 : ℚ) ≤ (n : ℚ) / 2 ^ (L + 1 - 24) := by positivity
    rw [← Int.natCast_floor_eq_floor hnn]
    push_cast; rfl
  · rw [Nat.shiftLeft_eq]
    have : (23 : ℤ) - L = ((24 - (L + 1) : ℕ) : ℤ) := by omega
    rw [this, zpow_natCast]
    have e : (n : ℚ) * 2 ^ (24 - (L + 1)) = ((n * 2 ^ (24 - (L + 1)) : ℕ) : ℤ) := by push_cast; ring
    rw [e, Int.floor_intCast]; push_cast; ring

theorem fromAdder26 (m : Nat) : fromAdder 26 m = m / 4 := by
  unfold fromAdder; rw [ite_eq_left (by norm_num), Nat.shiftRight_eq_div_pow]

/-- The 26-bit adder's normalised magnitude, back on the 24-bit grid, is the magnitude normalised to 24 bits. -/
theorem shift26_24 (n L : ℕ) :
    fromAdder 26 (if 26 < L + 1 then n >>> (L + 1 - 26) else n <<< (26 - (L + 1))) =
      if 24 < L + 1 then n >>> (L + 1 - 24) else n <<< (24 - (L + 1)) := by
  rw [fromAdder26]
  split_ifs with h1 h2 h2
  · rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul,
      show (4 : ℕ) = 2 ^ 2 from rfl, ← pow_add]
    congr 2; omega
  · omega
  · have : L + 1 = 25 ∨ L + 1 = 26 := by omega
    rcases this with h | h <;> simp only [h, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow] <;> omega
  · rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, show 26 - (L + 1) = (24 - (L + 1)) + 2 by omega, pow_add,
      ← mul_assoc]
    exact Nat.mul_div_cancel _ (by norm_num)

/-- `normalize` on the 26-bit adder, on a nonzero sum below `2^32` at an exponent of at least −101: a normal term at an
exponent of at most `E + 6`, the sum truncated toward zero to 24 significant bits. -/
theorem normalize26_spec {E total : ℤ} (h0 : total ≠ 0) (hE1 : -101 ≤ E) (hT : total.natAbs < 2 ^ 32) :
    2 ^ 23 ≤ (normalize 26 E total).mag ∧ (normalize 26 E total).mag < 2 ^ 24 ∧
      -126 ≤ (normalize 26 E total).exp ∧ (normalize 26 E total).exp ≤ E + 6 ∧
      (normalize 26 E total).value = trunc24 ((total : ℚ) * 2 ^ (E - 25)) := by
  have hn : total.natAbs ≠ 0 := by omega
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self total.natAbs
  have hL : total.natAbs.log2 < 32 := by
    by_contra h
    have : 2 ^ 32 ≤ 2 ^ total.natAbs.log2 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hq := floor_shift24 total.natAbs total.natAbs.log2
  set L := total.natAbs.log2 with hLdef
  set m0 := (if 24 < L + 1 then total.natAbs >>> (L + 1 - 24) else total.natAbs <<< (24 - (L + 1)))
    with hm0
  have hb1 : 2 ^ 23 ≤ m0 ∧ m0 < 2 ^ 24 := by
    rw [hm0]; split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow]
      constructor
      · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]; rwa [show 23 + (L + 1 - 24) = L by omega]
      · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]; rwa [show 24 + (L + 1 - 24) = L + 1 by omega]
    · rw [Nat.shiftLeft_eq]
      constructor
      · calc 2 ^ 23 = 2 ^ L * 2 ^ (24 - (L + 1)) := by rw [← pow_add]; congr 1; omega
          _ ≤ _ := Nat.mul_le_mul_right _ hL1
      · calc total.natAbs * 2 ^ (24 - (L + 1)) < 2 ^ (L + 1) * 2 ^ (24 - (L + 1)) :=
              Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
          _ = 2 ^ 24 := by rw [← pow_add]; congr 1; omega
  have hbl : bitLength total.natAbs = L + 1 := by unfold bitLength; rw [ite_eq_right hn]
  have hne : ¬ (E + ((L + 1 : ℕ) : ℤ) - ((26 : ℕ) : ℤ) < -126) := by omega
  unfold normalize
  simp only [h0, ite_false, hbl, hne, shift26_24, ← hm0]
  rw [ite_eq_right (by omega)]
  dsimp only
  refine ⟨by omega, by omega, by omega, by omega, ?_⟩
  have hy0 : (total : ℚ) * 2 ^ (E - 25) ≠ 0 := mul_ne_zero (by exact_mod_cast h0) (two_zpow_ne _)
  have habs : |(total : ℚ) * 2 ^ (E - 25)| = (total.natAbs : ℚ) * 2 ^ (E - 25) := by
    rw [abs_mul, abs_of_pos (two_zpow_pos _), Nat.cast_natAbs, Int.cast_abs]
  have hbin : binade |(total : ℚ) * 2 ^ (E - 25)| = L + (E - 25) := by
    rw [habs, binade_nat_mul hn]
  unfold trunc24 truncTo Term.value
  rw [ite_eq_right hy0, hbin, habs]
  have hdiv : (total.natAbs : ℚ) * 2 ^ (E - 25) / 2 ^ ((L : ℤ) + (E - 25) - 23) =
      (total.natAbs : ℚ) * 2 ^ ((23 : ℤ) - L) := by
    rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; ring
  rw [hdiv, hq]
  have hs : (total : ℚ) * 2 ^ (E - 25) < 0 ↔ total < 0 := by
    rw [mul_neg_iff]; constructor
    · rintro (⟨-, h⟩ | ⟨h, -⟩)
      · exact absurd h (not_lt.2 (two_zpow_pos _).le)
      · exact_mod_cast h
    · intro h; right; exact ⟨by exact_mod_cast h, two_zpow_pos _⟩
  have hpow : (2 : ℚ) ^ (E + ((L + 1 : ℕ) : ℤ) - ((26 : ℕ) : ℤ) - 23) = 2 ^ ((L : ℤ) + (E - 25) - 23) := by
    congr 1; push_cast; ring
  rw [hpow]
  by_cases hneg : total < 0
  · simp only [hneg, decide_true, ite_true, hs.2 hneg]
  · simp only [hneg, decide_false, ite_false, Bool.false_eq_true, (not_congr hs).2 hneg]

/-! ### The step -/

/-- **The K16 step's value.** On a finite accumulator term `tc` and finite BF16 words, with the step's exponent
`E = k16Exp tc x y` in `[−101, 121]`, the step writes a finite word whose value is two truncations toward zero: the
accumulator and each product to the grid `g = 2^(E − 25)`, then their sum to 24 significant bits. -/
theorem k16Step_value {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (x y : Fin 16 → Nat)
    (hx : ∀ l, bf16Exp (x l) ≠ 255) (hy : ∀ l, bf16Exp (y l) ≠ 255) (hE1 : -101 ≤ k16Exp tc x y)
    (hE2 : k16Exp tc x y ≤ 121) :
    wordQ (k16Step w x y) = some (trunc24 (truncTo (2 ^ (k16Exp tc x y - 25)) tc.value +
      ∑ l, truncTo (2 ^ (k16Exp tc x y - 25)) (bfVal (x l) * bfVal (y l)))) := by
  obtain ⟨hm2, -⟩ := Freeze.fp32Value_fin hw
  set E := k16Exp tc x y with hEdef
  have hcE : tc.mag ≠ 0 → tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self
  have hle : ∀ l, (bf16Product (x l) (y l)).mag ≠ 0 → (bf16Product (x l) (y l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  set total := alignedSum 26 E (tc :: List.ofFn fun l => bf16Product (x l) (y l)) with htdef
  rw [k16Step_fin hw x y hx hy]
  unfold groupSum
  change wordQ (packValue (let t := normalize 26 E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t)) = _
  have hsum : total = aligned 26 E tc + ∑ l, aligned 26 E (bf16Product (x l) (y l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have htot : (total : ℚ) * 2 ^ (E - 25) =
      truncTo (2 ^ (E - 25)) tc.value + ∑ l, truncTo (2 ^ (E - 25)) (bfVal (x l) * bfVal (y l)) := by
    rw [hsum]; push_cast; rw [add_mul, Finset.sum_mul, aligned26_value hcE, add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned26_value (hle l), bf16Product_value, bfVal_eq (hx l), bfVal_eq (hy l)]
  have hT : total.natAbs < 2 ^ 32 := by
    have hc : (aligned 26 E tc).natAbs ≤ 4 * tc.mag := aligned26_natAbs_le E tc
    have h1 : |∑ l : Fin 16, aligned 26 E (bf16Product (x l) (y l))| ≤
        ∑ l : Fin 16, |aligned 26 E (bf16Product (x l) (y l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin 16, |aligned 26 E (bf16Product (x l) (y l))| ≤ ∑ _l : Fin 16, (2 ^ 27 : ℤ) := by
      refine Finset.sum_le_sum fun l _ => ?_
      have := aligned26_natAbs_le E (bf16Product (x l) (y l))
      have := bf16Product_mag_lt (x l) (y l)
      rw [Int.abs_eq_natAbs]; omega
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    rw [Int.abs_eq_natAbs] at h1
    rw [hsum]
    omega
  clear_value total E
  rw [← htot]
  by_cases h0 : total = 0
  · rw [h0]
    simp [normalize, Term.zero, packValue, packTerm, Freeze.wordQ_zero, trunc24]
  obtain ⟨m1, m2, e1, e3, hv⟩ := normalize26_spec h0 hE1 hT
  have e2 : (normalize 26 E total).exp ≤ 127 := by omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (by omega)), packValue, wordQ_packTerm m1 m2 e1 e2, hv]

/-- **The K16 step's error.** Under `k16Step_value`'s hypotheses, the step's value is within
`17·g + (|S| + 17·g)·2^−23` of the exact sum `S = c + Σ_l x_l·y_l`, for the grid `g = 2^(E − 25)`. -/
theorem k16Step_err {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (x y : Fin 16 → Nat)
    (hx : ∀ l, bf16Exp (x l) ≠ 255) (hy : ∀ l, bf16Exp (y l) ≠ 255) (hE1 : -101 ≤ k16Exp tc x y)
    (hE2 : k16Exp tc x y ≤ 121) :
    |PearlC.wordVal (k16Step w x y) - (tc.value + ∑ l, bfVal (x l) * bfVal (y l))| ≤
      17 * 2 ^ (k16Exp tc x y - 25) +
        (|tc.value + ∑ l, bfVal (x l) * bfVal (y l)| + 17 * 2 ^ (k16Exp tc x y - 25)) / 2 ^ 23 := by
  have hv := k16Step_value hw x y hx hy hE1 hE2
  set g : ℚ := 2 ^ (k16Exp tc x y - 25) with hg_def
  have hg : 0 < g := two_zpow_pos _
  set S := tc.value + ∑ l, bfVal (x l) * bfVal (y l) with hS
  set T := truncTo g tc.value + ∑ l, truncTo g (bfVal (x l) * bfVal (y l)) with hT
  have hwv : PearlC.wordVal (k16Step w x y) = trunc24 T := by
    unfold PearlC.wordVal; rw [hv]
  have hTS : |T - S| ≤ 17 * g := by
    have e : T - S = (truncTo g tc.value - tc.value) +
        ∑ l, (truncTo g (bfVal (x l) * bfVal (y l)) - bfVal (x l) * bfVal (y l)) := by
      rw [hT, hS, Finset.sum_sub_distrib]; ring
    have h1 : |truncTo g tc.value - tc.value| ≤ g := by
      rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
    have h2 : |∑ l, (truncTo g (bfVal (x l) * bfVal (y l)) - bfVal (x l) * bfVal (y l))| ≤ 16 * g := by
      calc _ ≤ ∑ l, |truncTo g (bfVal (x l) * bfVal (y l)) - bfVal (x l) * bfVal (y l)| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _l : Fin 16, g := Finset.sum_le_sum fun l _ => by
            rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
        _ = 16 * g := by simp
    rw [e]
    obtain ⟨a1, a2⟩ := abs_le.1 h1
    obtain ⟨b1, b2⟩ := abs_le.1 h2
    exact abs_le.2 ⟨by linarith, by linarith⟩
  have h24 := abs_trunc24_sub_le T
  have hTa : |T| ≤ |S| + 17 * g := by
    have := abs_sub_abs_le_abs_sub T S; linarith
  rw [hwv]
  obtain ⟨a1, a2⟩ := abs_le.1 h24
  obtain ⟨b1, b2⟩ := abs_le.1 hTS
  have h3 : |T| / 2 ^ 23 ≤ (|S| + 17 * g) / 2 ^ 23 := div_le_div_of_nonneg_right hTa (by norm_num)
  exact abs_le.2 ⟨by linarith, by linarith⟩

/-! ### The step's relative error on a range -/

/-- A BF16 word that is zero or normal: its exponent field is below 255, and zero only with a zero mantissa. -/
def ZeroOrNormal (h : Nat) : Prop := bf16Exp h ≠ 255 ∧ (bf16Exp h = 0 → bf16Man h = 0)

/-- **The K16 range** of one step on the accumulator term `tc` and the words `x`, `y`: every word is zero or normal,
and the step's exponent is at most 121, and at least −101 unless every term is zero. -/
def K16Range (tc : Term) (x y : Fin 16 → Nat) : Prop :=
  (∀ l, ZeroOrNormal (x l)) ∧ (∀ l, ZeroOrNormal (y l)) ∧ k16Exp tc x y ≤ 121 ∧
    (-101 ≤ k16Exp tc x y ∨ (tc.mag = 0 ∧ ∀ l, (bf16Product (x l) (y l)).mag = 0))

/-- The K16 step's relative error on the range, `17·2^−25 + (1 + 17·2^−25)·2^−23`, about `2^−20.6`. -/
def eps16 : ℚ := 17 / 2 ^ 25 + (1 + 17 / 2 ^ 25) / 2 ^ 23

theorem eps16_nonneg : 0 ≤ eps16 := by unfold eps16; positivity

theorem bf16Sig_normal {h : Nat} (hh : ZeroOrNormal h) (h0 : bf16Sig h ≠ 0) : 128 ≤ bf16Sig h := by
  unfold bf16Sig at *
  split_ifs at * with he
  · exact absurd (hh.2 he) h0
  · omega

theorem bf16Product_normal {x y : Nat} (hx : ZeroOrNormal x) (hy : ZeroOrNormal y)
    (hm : (bf16Product x y).mag ≠ 0) : 2 ^ 23 ≤ (bf16Product x y).mag := by
  have hmag : (bf16Product x y).mag = 0 ∨ (bf16Product x y).mag = bf16Sig x * bf16Sig y * 2 ^ 9 := by
    by_cases h : (bf16Sig x * bf16Sig y) <<< 9 = 0
    · left; simp only [bf16Product]; rw [ite_eq_left h]; rfl
    · right; simp only [bf16Product]; rw [ite_eq_right h]
      show (bf16Sig x * bf16Sig y) <<< 9 = _
      rw [Nat.shiftLeft_eq]
  rcases hmag with h | h
  · exact absurd h hm
  · rw [h] at hm ⊢
    have h1 : bf16Sig x ≠ 0 := fun h' => hm (by simp [h'])
    have h2 : bf16Sig y ≠ 0 := fun h' => hm (by simp [h'])
    have := Nat.mul_le_mul (bf16Sig_normal hx h1) (bf16Sig_normal hy h2)
    omega

theorem two_zpow_term_le {t : Term} (h : 2 ^ 23 ≤ t.mag) : (2 : ℚ) ^ t.exp ≤ |t.value| := by
  have e : (2 : ℚ) ^ t.exp = (2 : ℚ) ^ (23 : ℕ) * 2 ^ (t.exp - 23) := by
    rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  rw [e, AtomErr.abs_termValue]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast h) (two_zpow_pos _).le

/-- On the range, the step's grid is bounded by its terms: `2^E ≤ |c| + Σ_l |x_l·y_l|`. -/
theorem two_zpow_k16Exp_le {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) {x y : Fin 16 → Nat}
    (hx : ∀ l, ZeroOrNormal (x l)) (hy : ∀ l, ZeroOrNormal (y l)) (hE1 : -101 ≤ k16Exp tc x y) :
    (2 : ℚ) ^ k16Exp tc x y ≤ |tc.value| + ∑ l, |bfVal (x l) * bfVal (y l)| := by
  have hB : 0 ≤ ∑ l, |bfVal (x l) * bfVal (y l)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hc := abs_nonneg tc.value
  unfold k16Exp at hE1 ⊢
  rcases AtomErr.maxExp_cases (-133) (tc :: List.ofFn fun l => bf16Product (x l) (y l)) with e | ⟨t, ht, hm, e⟩
  · rw [e] at hE1; norm_num at hE1
  · rw [e] at hE1 ⊢
    rcases List.mem_cons.1 ht with rfl | ht
    · rcases AtomErr.fp32Value_cases hw with h0 | he | h23
      · exact absurd h0 hm
      · rw [he] at hE1; norm_num at hE1
      · have := two_zpow_term_le h23; linarith
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      have h1 := two_zpow_term_le (bf16Product_normal (hx l) (hy l) hm)
      rw [bf16Product_value, ← bfVal_eq (hx l).1, ← bfVal_eq (hy l).1] at h1
      have h2 : |bfVal (x l) * bfVal (y l)| ≤ ∑ l, |bfVal (x l) * bfVal (y l)| :=
        Finset.single_le_sum (f := fun l => |bfVal (x l) * bfVal (y l)|) (fun _ _ => abs_nonneg _)
          (Finset.mem_univ l)
      linarith

/-- The step on zero terms writes `+0`. -/
theorem k16Step_zero {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (x y : Fin 16 → Nat)
    (hx : ∀ l, bf16Exp (x l) ≠ 255) (hy : ∀ l, bf16Exp (y l) ≠ 255) (h0 : tc.mag = 0)
    (hp : ∀ l, (bf16Product (x l) (y l)).mag = 0) : wordQ (k16Step w x y) = some 0 := by
  rw [k16Step_fin hw x y hx hy]
  have hs : ∀ E, alignedSum 26 E (tc :: List.ofFn fun l => bf16Product (x l) (y l)) = 0 := by
    intro E
    unfold alignedSum
    apply List.sum_eq_zero
    intro a ha
    simp only [List.mem_map, List.mem_cons, List.mem_ofFn] at ha
    obtain ⟨t, ht, rfl⟩ := ha
    rcases ht with rfl | ⟨l, rfl⟩
    · simp [aligned, h0]
    · simp [aligned, hp l]
  have hn : ∀ E, normalize 26 E 0 = Term.zero := fun E => by unfold normalize; rw [ite_eq_left rfl]
  unfold groupSum
  rw [hs, hn]
  show wordQ (packValue (if Term.zero.mag ≠ 0 ∧ 127 < Term.zero.exp then .inf Term.zero.neg else .fin Term.zero)) = _
  rw [ite_eq_right (fun h => absurd rfl h.1)]
  exact Freeze.wordQ_zero

theorem rel_of_err {v S B g : ℚ} (he : |v - S| ≤ 17 * g + (|S| + 17 * g) / 2 ^ 23) (hS : |S| ≤ B)
    (hg : g ≤ B / 2 ^ 25) : |v - S| ≤ eps16 * B := by
  calc _ ≤ _ := he
    _ ≤ 17 * (B / 2 ^ 25) + (B + 17 * (B / 2 ^ 25)) / 2 ^ 23 := by gcongr
    _ = eps16 * B := by unfold eps16; ring

/-- **The K16 step's relative error on the range**: the word is finite, and within `eps16·(|c| + Σ_l |x_l·y_l|)` of
`c + Σ_l x_l·y_l`. -/
theorem k16Step_rel {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) {x y : Fin 16 → Nat}
    (hR : K16Range tc x y) :
    ∃ v, wordQ (k16Step w x y) = some v ∧
      |v - (tc.value + ∑ l, bfVal (x l) * bfVal (y l))| ≤
        eps16 * (|tc.value| + ∑ l, |bfVal (x l) * bfVal (y l)|) := by
  obtain ⟨hx, hy, hE2, hE1 | ⟨h0, hp⟩⟩ := hR
  · have hv := k16Step_value hw x y (fun l => (hx l).1) (fun l => (hy l).1) hE1 hE2
    refine ⟨_, hv, ?_⟩
    have he := k16Step_err hw x y (fun l => (hx l).1) (fun l => (hy l).1) hE1 hE2
    have hwv : PearlC.wordVal (k16Step w x y) = trunc24 (truncTo (2 ^ (k16Exp tc x y - 25)) tc.value +
        ∑ l, truncTo (2 ^ (k16Exp tc x y - 25)) (bfVal (x l) * bfVal (y l))) := by
      unfold PearlC.wordVal; rw [hv]
    rw [hwv] at he
    have hB := two_zpow_k16Exp_le hw hx hy hE1
    have hS : |tc.value + ∑ l, bfVal (x l) * bfVal (y l)| ≤ |tc.value| + ∑ l, |bfVal (x l) * bfVal (y l)| := by
      have h1 := abs_add_le tc.value (∑ l, bfVal (x l) * bfVal (y l))
      have h2 := Finset.abs_sum_le_sum_abs (fun l => bfVal (x l) * bfVal (y l)) Finset.univ
      linarith
    have hg : (2 : ℚ) ^ (k16Exp tc x y - 25) ≤ (|tc.value| + ∑ l, |bfVal (x l) * bfVal (y l)|) / 2 ^ 25 := by
      rw [zpow_sub₀ two_ne_zero]
      exact div_le_div_of_nonneg_right hB (by norm_num)
    exact rel_of_err he hS hg
  · refine ⟨0, k16Step_zero hw x y (fun l => (hx l).1) (fun l => (hy l).1) h0 hp, ?_⟩
    have hc : tc.value = 0 := by simp [Term.value, h0]
    have hp' : ∀ l, bfVal (x l) * bfVal (y l) = 0 := by
      intro l
      rw [bfVal_eq (hx l).1, bfVal_eq (hy l).1, ← bf16Product_value]
      simp [Term.value, hp l]
    simp only [hc, hp', Finset.sum_const_zero, add_zero, sub_zero, abs_zero, abs_zero, mul_zero, le_refl]

/-! ### U's four steps -/

/-- **The K16 range along U's steps**: the range at each step's accumulator, as the steps before it wrote it. -/
def URange (pa pb : List Nat) : List Nat → Nat → Prop
  | [], _ => True
  | s :: ss, acc => (∀ tc, fp32Value acc = .fin tc → K16Range tc (slice16 pa s) (slice16 pb s)) ∧
      ∀ acc', uStep? pa pb acc s = some acc' → URange pa pb ss acc'

/-- Slice `s`'s products `P_A,i · P_B,j`, as values. -/
def sliceP (pa pb : List Nat) (s : Nat) (l : Fin 16) : ℚ := bfVal (slice16 pa s l) * bfVal (slice16 pb s l)

theorem wordQ_fin {w : Nat} {v : ℚ} (h : wordQ w = some v) : ∃ t, fp32Value w = .fin t ∧ t.value = v := by
  unfold wordQ at h
  cases hv : fp32Value w with
  | nan => simp [hv, Value.finite?] at h
  | inf s => simp [hv, Value.finite?] at h
  | fin t => simp only [hv, Value.finite?, Option.map_some, Option.some.injEq] at h; exact ⟨t, rfl, h⟩

theorem uFold_err (pa pb : List Nat) (ss : List Nat) : ∀ (c u : Nat) (cv : ℚ), wordQ c = some cv →
    ss.foldlM (uStep? pa pb) c = some u → URange pa pb ss c →
    ∃ uv, wordQ u = some uv ∧
      |uv - (cv + (ss.map fun s => ∑ l, sliceP pa pb s l).sum)| ≤
        ((1 + eps16) ^ ss.length - 1) * (|cv| + (ss.map fun s => ∑ l, |sliceP pa pb s l|).sum) := by
  induction ss with
  | nil =>
    intro c u cv hc hu _
    simp only [List.foldlM_nil, pure, Option.some.injEq] at hu
    subst hu
    exact ⟨cv, hc, by simp⟩
  | cons s ss ih =>
    intro c u cv hc hu hR
    rw [List.foldlM_cons] at hu
    obtain ⟨c', hc', hu'⟩ := Option.bind_eq_some_iff.1 hu
    obtain ⟨tc, htc, htv⟩ := wordQ_fin hc
    have hstep : k16Step c (slice16 pa s) (slice16 pb s) = c' := by
      unfold uStep? finite? at hc'
      split_ifs at hc'
      exact Option.some.inj hc'
    obtain ⟨v, hv, he⟩ := k16Step_rel htc (hR.1 tc htc)
    rw [hstep] at hv
    rw [htv] at he
    obtain ⟨uv, huv, he'⟩ := ih c' u v hv hu' (hR.2 c' hc')
    refine ⟨uv, huv, ?_⟩
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    set C := (1 + eps16) ^ ss.length with hC
    set P := (ss.map fun s => ∑ l, |sliceP pa pb s l|).sum
    set R := (ss.map fun s => ∑ l, sliceP pa pb s l).sum
    set Q := ∑ l, |sliceP pa pb s l|
    set S1 := ∑ l, sliceP pa pb s l
    have hε := eps16_nonneg
    have hC1 : 1 ≤ C := one_le_pow₀ (by linarith)
    have hP : 0 ≤ P := List.sum_nonneg (by simp; intro _ _; exact Finset.sum_nonneg fun _ _ => abs_nonneg _)
    have hQ : 0 ≤ Q := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hSQ : |S1| ≤ Q := Finset.abs_sum_le_sum_abs _ _
    change |v - (cv + S1)| ≤ eps16 * (|cv| + Q) at he
    have ha : |v| ≤ (1 + eps16) * (|cv| + Q) := by
      have := abs_sub_abs_le_abs_sub v (cv + S1)
      have := abs_add_le cv S1
      nlinarith [abs_nonneg cv]
    calc |uv - (cv + (S1 + R))| = |(uv - (v + R)) + (v - (cv + S1))| := by ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ (C - 1) * (|v| + P) + eps16 * (|cv| + Q) := add_le_add he' he
      _ ≤ (C - 1) * ((1 + eps16) * (|cv| + Q) + P) + eps16 * (|cv| + Q) := by gcongr
      _ ≤ (C * (1 + eps16) - 1) * (|cv| + (Q + P)) := by
          nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℚ) ≤ C) hε) hP]
      _ = _ := by rw [hC, ← pow_succ]

theorem sum_slices (f : ℕ → ℚ) :
    ((List.range 4).map fun s => ∑ l : Fin 16, f (16 * s + l)).sum = ∑ t : Fin 64, f t := by
  rw [show List.range 4 = [0, 1, 2, 3] from rfl]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  have h : ∑ t : Fin 64, f t = ∑ s : Fin 4, ∑ l : Fin 16, f (16 * s + l) := by
    rw [← Fintype.sum_prod_type']
    refine (Fintype.sum_equiv (finProdFinEquiv : Fin 4 × Fin 16 ≃ Fin 64) _ _ (fun x => ?_)).symm
    have e : ((finProdFinEquiv : Fin 4 × Fin 16 ≃ Fin 64) x : ℕ) = 16 * x.1.val + x.2.val := by
      simp [finProdFinEquiv]; ring
    rw [e]
  rw [h, Fin.sum_univ_four]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two]
  norm_num
  ring

/-- **U's four steps on the range**: `uWord?` writes a finite word within `((1 + eps16)^4 − 1)·(|c| + Σ_t |P_A,t·P_B,t|)`
of `c + Σ_t P_A,t·P_B,t`, over the 64 positions of the served rows. -/
theorem uWord_err {c u : Nat} {cv : ℚ} (pa pb : List Nat) (hc : wordQ c = some cv) (hu : uWord? c pa pb = some u)
    (hR : URange pa pb (List.range 4) c) :
    ∃ uv, wordQ u = some uv ∧
      |uv - (cv + ∑ t : Fin 64, bfVal (pa.getD t 0) * bfVal (pb.getD t 0))| ≤
        ((1 + eps16) ^ 4 - 1) * (|cv| + ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)|) := by
  obtain ⟨uv, huv, he⟩ := uFold_err pa pb (List.range 4) c u cv hc hu hR
  refine ⟨uv, huv, ?_⟩
  have h1 := sum_slices fun t => bfVal (pa.getD t 0) * bfVal (pb.getD t 0)
  have h2 := sum_slices fun t => |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)|
  simp only [sliceP, slice16, List.length_range] at he
  rw [← h1, ← h2]
  exact he

/-! ### The entry's decomposition into U's steps, `C̃`'s chain and the rows -/

/-- The served factor rows of entry `(i, j)`: `C̃`'s word and the rows `P_A,i`, `P_B,j`, as `peelExact` computes them
before U's four steps. -/
def peelRows (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
    (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) (i : Fin m) (j : Fin n) :
    Option (Nat × List Nat × List Nat) := do
  let c ← chainWord? (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun p => chainWord? (rowAtoms (Aq i) (FBᵀ p))).mapM id
  let bf ← (List.ofFn fun q => chainWord? (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun p => (List.ofFn fun q => chainWord? (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  pure (c, pa, pb)

theorem peelExact_eq_rows (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (i : Fin m) (j : Fin n) :
    peelExact Aq Bq FA FB beA beB i j = (peelRows Aq Bq FA FB beA beB i j).bind fun r => uWord? r.1 r.2.1 r.2.2 := by
  unfold peelExact peelRows
  simp only [Option.bind_eq_bind, Option.bind_assoc, Option.pure_def, Option.bind_some]

theorem finite?_eq_some {w w' : Nat} (h : finite? w = some w') : w' = w ∧ ∃ v, wordQ w = some v := by
  unfold finite? at h
  split_ifs at h with he
  refine ⟨(Option.some.inj h).symm, ?_⟩
  unfold wordQ fp32Value
  simp only [he, ite_false]
  split_ifs <;> simp [Value.finite?]

/-- A chain's word is finite. -/
theorem chainWord_fin {atoms : List AtomIn} {c : Nat} (h : chainWord? atoms = some c) : ∃ v, wordQ c = some v := by
  unfold chainWord? at h
  suffices H : ∀ (gs : List (List AtomIn)) (t : Nat), (∃ v, wordQ t = some v) →
      gs.foldlM (fun t g => (windowWord? g).bind (promoteWord? t)) t = some c → ∃ v, wordQ c = some v from
    H _ 0 ⟨0, Freeze.wordQ_zero⟩ h
  intro gs
  induction gs with
  | nil => intro t ht h; simp only [List.foldlM_nil, pure, Option.some.injEq] at h; subst h; exact ht
  | cons g gs ih =>
    intro t _ h
    rw [List.foldlM_cons] at h
    obtain ⟨t', ht', h⟩ := Option.bind_eq_some_iff.1 h
    refine ih t' ?_ h
    obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 ht'
    unfold promoteWord? at hw
    obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 hw
    unfold addWord? at hs
    split at hs
    · obtain ⟨rfl, hv⟩ := finite?_eq_some hs
      split_ifs
      · exact ⟨0, Freeze.wordQ_zero⟩
      · exact hv
    · cases hs

/-- **The served U against its exact counterpart**, entry by entry: on the K16 range along U's steps, the served
entry is within U's four steps' `((1 + eps16)^4 − 1)·(|C̃| + Σ_t |P_A,t·P_B,t|)`, plus `C̃`'s chain error, plus the
rows' errors `Σ_t |P_A,t·P_B,t − P̂_A,t·P̂_B,t|`, of the exact counterpart `uQ` (whose rows `P̂` are `paQ`, `pbQ`). -/
theorem peelExact_err (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (i : Fin m) (j : Fin n) {c : Nat} {pa pb : List Nat} {v : ℚ}
    (hrows : peelRows Aq Bq FA FB beA beB i j = some (c, pa, pb))
    (hv : peelExactVal Aq Bq FA FB beA beB i j = some v) (hR : URange pa pb (List.range 4) c) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤
      ((1 + eps16) ^ 4 - 1) * (|PearlC.wordVal c| + ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)|) +
        |PearlC.wordVal c - cQ Aq Bq i j| +
        ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) -
          paQ Aq FB beA i t * pbQ Bq FA FB beB j t| := by
  unfold peelExactVal at hv
  rw [peelExact_eq_rows, hrows] at hv
  change (uWord? c pa pb).bind wordQ = some v at hv
  obtain ⟨u, hu, huv⟩ := Option.bind_eq_some_iff.1 hv
  have hc : ∃ cv, wordQ c = some cv := by
    unfold peelRows at hrows
    obtain ⟨c', hc', h⟩ := Option.bind_eq_some_iff.1 hrows
    have : c' = c := by
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq,
        Prod.mk.injEq] at h
      obtain ⟨_, _, _, _, _, _, _, _, _, _, h⟩ := h
      exact h.1
    subst this
    exact chainWord_fin hc'
  obtain ⟨cv, hcv⟩ := hc
  obtain ⟨uv, huv', he⟩ := uWord_err pa pb hcv hu hR
  rw [huv] at huv'
  cases huv'
  have hwc : PearlC.wordVal c = cv := by unfold PearlC.wordVal; rw [hcv]
  rw [hwc]
  have hq : uQ Aq Bq FA FB beA beB i j = cQ Aq Bq i j + ∑ t, paQ Aq FB beA i t * pbQ Bq FA FB beB j t := by
    unfold uQ; exact foldl_k16Q _ _ _
  rw [hq]
  set S := ∑ t : Fin 64, bfVal (pa.getD t 0) * bfVal (pb.getD t 0)
  set Sq := ∑ t, paQ Aq FB beA i t * pbQ Bq FA FB beB j t
  have hd : |S - Sq| ≤ ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) -
      paQ Aq FB beA i t * pbQ Bq FA FB beB j t| := by
    rw [← Finset.sum_sub_distrib]; exact Finset.abs_sum_le_sum_abs _ _
  calc |v - (cQ Aq Bq i j + Sq)| = |(v - (cv + S)) + (cv - cQ Aq Bq i j) + (S - Sq)| := by ring_nf
    _ ≤ |v - (cv + S)| + |cv - cQ Aq Bq i j| + |S - Sq| := abs_add_three _ _ _
    _ ≤ _ := by gcongr

/-! ### FP32 add and FMA -/

theorem wordQ_inf (sb : Nat) (hs : sb ≤ 1) : wordQ (sb * 2 ^ 31 + 0x7F800000) = none := by
  rcases Nat.le_one_iff_eq_zero_or_eq_one.1 hs with rfl | rfl <;> decide

/-- **`f32Word`'s error where the word is finite**: within half an FP32 ulp of `y`, at most `|y|·2^−24`, or `2^−150`
among the subnormals (`AtomErr.f32Word_spec` without its range: a finite word is its own range). -/
theorem f32Word_fin_err {y v : ℚ} (hfin : wordQ (f32Word y) = some v) :
    |v - y| ≤ max (|y| / 2 ^ 24) (2 ^ (-150 : ℤ)) := by
  rcases eq_or_ne y 0 with rfl | hy0
  · have : f32Word 0 = 0 := by simp [f32Word]
    rw [this, Freeze.wordQ_zero, Option.some.injEq] at hfin
    subst hfin
    simp only [sub_zero, abs_zero]
    exact le_max_of_le_right (two_zpow_pos _).le
  have ha : 0 < |y| := abs_pos.2 hy0
  obtain ⟨hb1, hb2⟩ := binade_spec ha
  set E₀ := max (binade |y|) (-126) with hE₀
  set u : ℚ := 2 ^ (E₀ - 23) with hu_def
  have hu : 0 < u := two_zpow_pos _
  set R := roundEven (|y| / u) with hR
  have hR0 : 0 ≤ R := le_trans (Int.floor_nonneg.2 (div_nonneg ha.le hu.le)) (floor_le_roundEven _)
  have hlt : |y| / u < 2 ^ 24 := by
    rw [div_lt_iff₀ hu]
    calc |y| < 2 ^ (binade |y| + 1) := hb2
      _ ≤ 2 ^ (E₀ + 1) := zpow_le_zpow_right₀ one_le_two (by omega)
      _ = 2 ^ 24 * u := by rw [hu_def, two_pow_24]
  have hR24 : R ≤ 2 ^ 24 := by
    have h1 := roundEven_le (|y| / u)
    have h2 : ⌊|y| / u⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hlt)
    omega
  have hR23 : -126 ≤ binade |y| → 2 ^ 23 ≤ R := by
    intro h
    have hE : E₀ = binade |y| := max_eq_left h
    have h1 : (2 : ℚ) ^ 23 ≤ |y| / u := by
      rw [le_div_iff₀ hu, hu_def, hE, two_pow_23]; exact hb1
    have h2 : (2 ^ 23 : ℤ) ≤ ⌊|y| / u⌋ := Int.le_floor.2 (by exact_mod_cast h1)
    exact le_trans h2 (floor_le_roundEven _)
  have herr : abs ((R : ℚ) * u - |y|) ≤ u / 2 := by
    have h := AtomErr.abs_roundEven_sub_le (|y| / u)
    rw [← hR] at h
    have e : (R : ℚ) * u - |y| = ((R : ℚ) - |y| / u) * u := by rw [sub_mul, div_mul_cancel₀ _ hu.ne']
    rw [e, abs_mul, abs_of_pos hu]
    nlinarith [abs_nonneg ((R : ℚ) - |y| / u)]
  have hhalf : u / 2 ≤ max (|y| / 2 ^ 24) (2 ^ (-150 : ℤ)) := by
    rcases le_or_gt (-126) (binade |y|) with h | h
    · apply le_max_of_le_left
      have hE : E₀ = binade |y| := max_eq_left h
      have : u * 2 ^ 24 = 2 * 2 ^ binade |y| := by
        rw [hu_def, hE, mul_comm, two_pow_24, zpow_add_one₀ two_ne_zero]; ring
      rw [div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
    · apply le_max_of_le_right
      have hE : E₀ = -126 := max_eq_right h.le
      rw [hu_def, hE]; norm_num
  set sb : ℕ := if y < 0 then 1 else 0 with hsb
  have hs1 : sb ≤ 1 := by rw [hsb]; split <;> omega
  have hs2 : (if y < 0 then 2 ^ 31 else 0) = sb * 2 ^ 31 := by rw [hsb]; split <;> simp
  have hsgn : (if (sb == 1) = true then (-1 : ℚ) else 1) * |y| = y := by
    rw [hsb]; by_cases h : y < 0
    · simp [h, abs_of_neg h]
    · simp [h, abs_of_nonneg (not_lt.1 h)]
  obtain ⟨N₀, hN₀⟩ : ∃ n : ℕ, (n : ℤ) = R := ⟨R.toNat, Int.toNat_of_nonneg hR0⟩
  have hNt : R.toNat = N₀ := by omega
  have key : ∃ t : Term, fp32Value (f32Word y) = .fin t ∧ t.value = (if (sb == 1) = true then -1 else 1) * (R * u) := by
    have hw : f32Word y =
        let E := if N₀ = 2 ^ 24 then E₀ + 1 else E₀
        let N := if N₀ = 2 ^ 24 then 2 ^ 23 else N₀
        if 127 < E then sb * 2 ^ 31 + 0x7F800000
        else if N < 2 ^ 23 then sb * 2 ^ 31 + N
        else sb * 2 ^ 31 + (E + 127).toNat * 2 ^ 23 + (N - 2 ^ 23) := by
      unfold f32Word
      simp only [ite_eq_right ha.ne', ← hE₀, ← hu_def, ← hR, hNt, hs2]
    rw [hw] at hfin ⊢
    dsimp only at hfin ⊢
    have hR' : (R : ℚ) = N₀ := by rw [← hN₀]; norm_cast
    by_cases hc : N₀ = 2 ^ 24
    · by_cases h127 : 127 < E₀ + 1
      · simp only [hc, ite_true, h127] at hfin
        rw [wordQ_inf sb hs1] at hfin; cases hfin
      simp only [hc, ite_true, h127, ite_false, show ¬ (2 ^ 23 < 2 ^ 23) by omega, Nat.sub_self, add_zero]
      have hef : (E₀ + 1 + 127).toNat < 255 := by omega
      have hv := AtomErr.fp32Value_word sb _ 0 hs1 hef (by norm_num)
      rw [Nat.add_zero, ite_eq_right (by omega)] at hv
      rw [hv]
      refine ⟨_, rfl, ?_⟩
      unfold Term.value
      have hx : (((E₀ + 1 + 127).toNat : ℕ) : ℤ) - 127 - 23 = (E₀ - 23) + 1 := by omega
      simp only [hx]
      rw [hR', hc, hu_def, zpow_add_one₀ two_ne_zero]
      push_cast; ring
    · by_cases h127 : 127 < E₀
      · simp only [hc, ite_false, h127, ite_true] at hfin
        rw [wordQ_inf sb hs1] at hfin; cases hfin
      simp only [hc, ite_false, h127]
      by_cases hsub : N₀ < 2 ^ 23
      · simp only [hsub, ite_true]
        have hbin : binade |y| < -126 := by
          by_contra h; have := hR23 (not_lt.1 h); omega
        have hE : E₀ = -126 := max_eq_right hbin.le
        rw [show sb * 2 ^ 31 + N₀ = sb * 2 ^ 31 + 0 * 2 ^ 23 + N₀ by ring,
          AtomErr.fp32Value_word sb 0 N₀ hs1 (by norm_num) hsub, ite_eq_left rfl]
        by_cases h0 : N₀ = 0
        · rw [ite_eq_left h0]
          refine ⟨_, rfl, ?_⟩
          rw [hR', h0]; simp [Term.value, Term.zero]
        · rw [ite_eq_right h0]
          refine ⟨_, rfl, ?_⟩
          unfold Term.value
          rw [hR', hu_def, hE]; norm_num
      · simp only [hsub, ite_false]
        have hef : (E₀ + 127).toNat < 255 := by omega
        have hm : N₀ - 2 ^ 23 < 2 ^ 23 := by omega
        rw [AtomErr.fp32Value_word sb _ _ hs1 hef hm, ite_eq_right (by omega)]
        refine ⟨_, rfl, ?_⟩
        unfold Term.value
        rw [hR', show 2 ^ 23 + (N₀ - 2 ^ 23) = N₀ by omega, hu_def,
          show (((E₀ + 127).toNat : ℕ) : ℤ) - 127 - 23 = E₀ - 23 by omega]
        ring
  obtain ⟨t, ht, hv⟩ := key
  have hvt : v = t.value := by
    unfold wordQ at hfin; rw [ht] at hfin
    simp only [Value.finite?, Option.map_some, Option.some.injEq] at hfin
    exact hfin.symm
  rw [hvt]
  have e : t.value - y = (if (sb == 1) = true then (-1 : ℚ) else 1) * (R * u - |y|) := by
    rw [hv, mul_sub, hsgn]
  rw [e, abs_mul]
  have h1 : |(if (sb == 1) = true then (-1 : ℚ) else 1)| = 1 := by split <;> simp
  rw [h1, one_mul]
  exact le_trans herr hhalf

theorem max_le_add_of_nonneg {a b : ℚ} (ha : 0 ≤ a) (hb : 0 ≤ b) : max a b ≤ a + b :=
  max_le (by linarith) (by linarith)

theorem wordQ_negZero : wordQ 0x80000000 = some 0 := by
  unfold wordQ fp32Value; norm_num [Value.finite?, Term.zero, Term.value]

/-- One rounding of a nonzero-or-signed-zero value: the finite word is within `|v|·2^−24 + 2^−150` of `v`. -/
theorem rnWord_err {v : ℚ} {nz : Bool} {r : ℚ} (h : wordQ (rnWord v nz) = some r) :
    |r - v| ≤ |v| / 2 ^ 24 + 2 ^ (-150 : ℤ) := by
  have h150 := two_zpow_pos (-150)
  unfold rnWord at h
  by_cases hv : v = 0
  · rw [ite_eq_left hv] at h
    have : r = 0 := by
      cases nz
      · simpa [Freeze.wordQ_zero] using h.symm
      · simpa [wordQ_negZero] using h.symm
    rw [this, hv]; simp only [sub_zero, abs_zero, zero_div, zero_add]; exact h150.le
  · rw [ite_eq_right hv] at h
    exact le_trans (f32Word_fin_err h) (max_le_add_of_nonneg (by positivity) h150.le)

/-- **FP32 add's error**: a finite sum word is within `|x + y|·2^−24 + 2^−150` of `x + y`. -/
theorem addWord_err {a b s : Nat} {x y : ℚ} (ha : wordQ a = some x) (hb : wordQ b = some y)
    (hs : addWord? a b = some s) : ∃ sv, wordQ s = some sv ∧ |sv - (x + y)| ≤ |x + y| / 2 ^ 24 + 2 ^ (-150 : ℤ) := by
  unfold addWord? at hs
  rw [ha, hb] at hs
  obtain ⟨rfl, sv, hsv⟩ := finite?_eq_some hs
  exact ⟨sv, hsv, rnWord_err hsv⟩

/-- **FMA's error**: a finite word is within `|x·y + z|·2^−24 + 2^−150` of `x·y + z`. -/
theorem fmaWord_err {a b c s : Nat} {x y z : ℚ} (ha : wordQ a = some x) (hb : wordQ b = some y)
    (hc : wordQ c = some z) (hs : fmaWord? a b c = some s) :
    ∃ sv, wordQ s = some sv ∧ |sv - (x * y + z)| ≤ |x * y + z| / 2 ^ 24 + 2 ^ (-150 : ℤ) := by
  unfold fmaWord? at hs
  rw [ha, hb, hc] at hs
  obtain ⟨rfl, sv, hsv⟩ := finite?_eq_some hs
  exact ⟨sv, hsv, rnWord_err hsv⟩

/-! ### Sign flips -/

theorem xor31_fields (w : Nat) :
    ((w ^^^ 0x80000000) >>> 23) % 256 = (w >>> 23) % 256 ∧ (w ^^^ 0x80000000) % 2 ^ 23 = w % 2 ^ 23 ∧
      ((w ^^^ 0x80000000) >>> 31) % 2 = 1 - (w >>> 31) % 2 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [Nat.shiftRight_xor_distrib, show (0x80000000 : Nat) >>> 23 = 256 from rfl,
      show (256 : Nat) = 2 ^ 8 from rfl, Nat.xor_mod_two_pow]
    simp
  · rw [Nat.xor_mod_two_pow]; simp
  · rw [Nat.shiftRight_xor_distrib, show (0x80000000 : Nat) >>> 31 = 1 from rfl,
      show (2 : Nat) = 2 ^ 1 from rfl, Nat.xor_mod_two_pow]
    have hb : (w >>> 31) % 2 ^ 1 < 2 := Nat.mod_lt _ (by norm_num)
    generalize (w >>> 31) % 2 ^ 1 = b at hb ⊢
    rcases (by omega : b = 0 ∨ b = 1) with rfl | rfl <;> rfl

/-- `negWord` negates a word's value. -/
theorem wordQ_xor31 (w : Nat) : wordQ (w ^^^ 0x80000000) = (wordQ w).map (- ·) := by
  obtain ⟨h1, h2, h3⟩ := xor31_fields w
  unfold wordQ fp32Value
  simp only [h1, h2, h3]
  have hb : (w >>> 31) % 2 < 2 := Nat.mod_lt _ (by norm_num)
  generalize (w >>> 31) % 2 = b at hb ⊢
  generalize (w >>> 23) % 256 = e
  generalize w % 2 ^ 23 = m
  rcases (by omega : b = 0 ∨ b = 1) with rfl | rfl <;> split_ifs <;> simp [Value.finite?, Term.value, Term.zero]

theorem wordQ_negWord (w : Nat) : wordQ (negWord w) = (wordQ w).map (- ·) := wordQ_xor31 w

/-- Flipping a BF16 word's sign bit negates its value. -/
theorem bfVal_xor (h : Nat) : bfVal (h ^^^ 0x8000) = -bfVal h := by
  unfold bfVal
  rw [Nat.shiftLeft_xor_distrib, show (0x8000 : Nat) <<< 16 = 0x80000000 from rfl, wordQ_xor31]
  cases wordQ (h <<< 16) <;> simp


/-! ### `RN_bf16` -/

/-- A finite FP32 word's value from its fields: sign bit `s`, exponent field `e` and mantissa `m`,
`±((e = 0 ? 0 : 2^23) + m)·2^(max(e, 1) − 150)`. -/
theorem wordQ_fields {w : Nat} (he : (w >>> 23) % 256 ≠ 255) :
    wordQ w = some ((if (w >>> 31) % 2 = 1 then (-1 : ℚ) else 1) *
      ((((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) : ℚ) *
        (2 : ℚ) ^ (((max ((w >>> 23) % 256) 1 : ℕ) : ℤ) - 150))) := by
  have hm : w % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  have h4 : 1 <<< 23 ||| w % 2 ^ 23 = 2 ^ 23 + w % 2 ^ 23 := by
    rw [← Nat.shiftLeft_add_eq_or_of_lt hm, Nat.one_shiftLeft]
  unfold wordQ fp32Value
  simp only [ite_eq_right he, h4]
  generalize (w >>> 23) % 256 = e at he ⊢
  generalize w % 2 ^ 23 = m
  generalize (w >>> 31) % 2 = b
  by_cases e0 : e = 0
  · subst e0
    by_cases m0 : m = 0
    · subst m0; simp [Value.finite?, Term.value, Term.zero]
    · simp only [ite_true, m0, ite_false, Value.finite?, Option.map_some, Term.value, beq_iff_eq]
      norm_num
  · simp only [e0, ite_false, Value.finite?, Option.map_some, Term.value, beq_iff_eq, Option.some.injEq]
    rw [show max e 1 = e by omega]
    push_cast
    rw [show (e : ℤ) - 127 - 23 = (e : ℤ) - 150 by ring]
    ring

theorem two_zpow_shift (a b : ℕ) (hab : b ≤ a) :
    (2 : ℚ) ^ ((a : ℤ) - 134) = ((2 ^ (a - b + 16) : ℕ) : ℚ) * 2 ^ ((b : ℤ) - 150) := by
  rw [Nat.cast_pow, Nat.cast_ofNat, ← zpow_natCast, ← zpow_add₀ two_ne_zero]
  congr 1
  push_cast [hab]
  ring

theorem rn_bits (H sb e q lo : ℕ) (hsb : sb ≤ 1) (he : e ≤ 254) (hq : q < 128) (hlo : lo < 65536) :
    ((4294967296 * H + 2147483648 * sb + 8388608 * e + 65536 * q + lo) + 32767 +
        (4294967296 * H + 2147483648 * sb + 8388608 * e + 65536 * q + lo) / 65536 % 2) / 65536 % 65536 =
      32768 * sb + 128 * e + q + (lo + 32767 + q % 2) / 65536 := by
  have h1 : (4294967296 * H + 2147483648 * sb + 8388608 * e + 65536 * q + lo) / 65536 =
      65536 * H + 32768 * sb + 128 * e + q := by omega
  rw [h1]
  have h2 : (65536 * H + 32768 * sb + 128 * e + q) % 2 = q % 2 := by omega
  rw [h2]
  have hr : (lo + 32767 + q % 2) / 65536 ≤ 1 := by omega
  have h3 : ((4294967296 * H + 2147483648 * sb + 8388608 * e + 65536 * q + lo) + 32767 + q % 2) / 65536 =
      65536 * H + (32768 * sb + 128 * e + q + (lo + 32767 + q % 2) / 65536) := by omega
  rw [h3]
  omega

theorem out_fields (sb e q r : ℕ) (hsb : sb ≤ 1) (he : e ≤ 254) (hq : q < 128) (hr : r ≤ 1)
    (hov : ¬ 32640 ≤ (32768 * sb + 128 * e + q + r) % 32768) :
    (32768 * sb + 128 * e + q + r) / 32768 % 2 = sb ∧ (32768 * sb + 128 * e + q + r) / 128 % 256 ≠ 255 ∧
      (q + r < 128 → (32768 * sb + 128 * e + q + r) / 128 % 256 = e ∧ (32768 * sb + 128 * e + q + r) % 128 = q + r) ∧
      (128 ≤ q + r → (32768 * sb + 128 * e + q + r) / 128 % 256 = e + 1 ∧ (32768 * sb + 128 * e + q + r) % 128 = 0) := by
  have hX : 128 * e + q + r < 32640 := by
    by_contra h
    have : 128 * e + q + r = 32640 := by omega
    rw [show 32768 * sb + 128 * e + q + r = 32768 * sb + 32640 by omega] at hov
    omega
  refine ⟨by omega, by omega, fun h => ⟨by omega, by omega⟩, fun h => ⟨by omega, by omega⟩⟩

/-- **`RN_bf16`'s error**: a BF16 word `rnBf16?` writes is finite and within `|x|·2^−8 + 2^−134` of the FP32 value
`x`: half a BF16 ulp, `2^15` units of the FP32 word's last place. -/
theorem rnBf16_err {w h : Nat} {x : ℚ} (hx : wordQ w = some x) (hh : rnBf16? w = some h) :
    bf16Exp h ≠ 255 ∧ |bfVal h - x| ≤ |x| / 2 ^ 8 + 2 ^ (-134 : ℤ) := by
  have he : (w >>> 23) % 256 ≠ 255 := by
    intro he
    unfold wordQ fp32Value at hx
    simp only [he, ite_true] at hx
    split_ifs at hx <;> simp [Value.finite?] at hx
  have hfin : finite? w = some w := by unfold finite?; rw [ite_eq_right he]
  unfold rnBf16? at hh
  rw [hfin, Option.bind_some] at hh
  dsimp only at hh
  by_cases hov : 0x7F80 ≤ ((w + 0x7FFF + (w >>> 16) % 2) >>> 16) % 65536 % 32768
  · rw [ite_eq_left hov] at hh; cases hh
  rw [ite_eq_right hov, Option.some.injEq] at hh
  subst hh
  rw [wordQ_fields he, Option.some.injEq] at hx
  subst hx
  obtain ⟨e, hed⟩ : ∃ e, (w >>> 23) % 256 = e := ⟨_, rfl⟩
  obtain ⟨m, hmd⟩ : ∃ m, w % 2 ^ 23 = m := ⟨_, rfl⟩
  obtain ⟨sb, hsd⟩ : ∃ sb, (w >>> 31) % 2 = sb := ⟨_, rfl⟩
  obtain ⟨out, hod⟩ : ∃ o, ((w + 0x7FFF + (w >>> 16) % 2) >>> 16) % 65536 = o := ⟨_, rfl⟩
  rw [hed, hmd, hsd]
  rw [hod] at hov ⊢
  rw [hed] at he
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at hed hsd hod hmd
  have hsb : sb ≤ 1 := by omega
  have hA : w / 8388608 / 256 = w / 2147483648 := by rw [Nat.div_div_eq_div_mul]
  have hB : w / 2147483648 / 2 = w / 4294967296 := by rw [Nat.div_div_eq_div_mul]
  obtain ⟨q, hqd⟩ : ∃ q, m / 65536 = q := ⟨_, rfl⟩
  obtain ⟨lo, hlod⟩ : ∃ lo, m % 65536 = lo := ⟨_, rfl⟩
  have hq : q < 128 := by omega
  have hlo : lo < 65536 := by omega
  have hm : m = 65536 * q + lo := by omega
  have hw : w = 4294967296 * (w / 4294967296) + 2147483648 * sb + 8388608 * e + 65536 * q + lo := by omega
  have hout := rn_bits (w / 4294967296) sb e q lo hsb (by omega) hq hlo
  rw [← hw, hod] at hout
  have he256 : e < 256 := by omega
  clear hod hed hsd hmd hw hA hB hqd hlod
  subst hm
  obtain ⟨r, hrd⟩ : ∃ r, (lo + 32767 + q % 2) / 65536 = r := ⟨_, rfl⟩
  rw [hrd] at hout
  subst hout
  have hr : r ≤ 1 := by omega
  obtain ⟨hs1, hE1, hnc, hc⟩ := out_fields sb e q r hsb (by omega) hq hr hov
  have hsgn : bf16Sign (32768 * sb + 128 * e + q + r) = (sb == 1) := by
    unfold bf16Sign; simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; rw [hs1]
  have hE : bf16Exp (32768 * sb + 128 * e + q + r) ≠ 255 := by
    unfold bf16Exp; simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]; exact hE1
  refine ⟨hE, ?_⟩
  rw [bfVal_eq hE]
  unfold bf16Value bf16Sig bf16ExpOr1
  rw [hsgn]
  set P : ℚ := 2 ^ (((max e 1 : ℕ) : ℤ) - 150) with hP
  have hP0 : 0 < P := two_zpow_pos _
  set X : ℕ := (if e = 0 then 0 else 2 ^ 23) + (65536 * q + lo) with hX
  obtain ⟨Z, hZ, hZX⟩ : ∃ Z : ℕ, ((if bf16Exp (32768 * sb + 128 * e + q + r) = 0 then
      bf16Man (32768 * sb + 128 * e + q + r) else 128 + bf16Man (32768 * sb + 128 * e + q + r) : ℕ) : ℚ) *
      2 ^ (((max (bf16Exp (32768 * sb + 128 * e + q + r)) 1 : ℕ) : ℤ) - 134) = (Z : ℚ) * P ∧
      (Z : ℤ) - X ≤ 2 ^ 15 ∧ (X : ℤ) - Z ≤ 2 ^ 15 := by
    unfold bf16Exp bf16Man
    simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]
    by_cases hcr : q + r < 128
    · obtain ⟨h1, h2⟩ := hnc hcr
      rw [h1, h2, two_zpow_shift _ _ le_rfl, Nat.sub_self, zero_add, ← mul_assoc, ← Nat.cast_mul]
      refine ⟨_, rfl, ?_, ?_⟩ <;> rw [hX] <;> split_ifs <;> push_cast <;> omega
    · obtain ⟨h1, h2⟩ := hc (by omega)
      rw [h1, h2, ite_eq_right (by omega : ¬ (e + 1 = 0)), add_zero]
      by_cases e0 : e = 0
      · subst e0
        rw [two_zpow_shift _ _ le_rfl, Nat.sub_self, zero_add, ← mul_assoc, ← Nat.cast_mul]
        refine ⟨_, rfl, ?_, ?_⟩ <;> rw [hX] <;> simp only [ite_true] <;> push_cast <;> omega
      · rw [two_zpow_shift _ _ (show max e 1 ≤ max (e + 1) 1 by omega),
          show max (e + 1) 1 - max e 1 = 1 by omega, ← mul_assoc, ← Nat.cast_mul]
        refine ⟨_, rfl, ?_, ?_⟩ <;> rw [hX] <;> simp only [e0, ite_false] <;> push_cast <;> omega
  rw [mul_assoc, hZ]
  have hs : ((if (sb == 1) = true then -1 else 1 : ℚ)) = (if sb = 1 then -1 else 1) := by simp
  rw [hs, ← mul_sub, abs_mul, abs_mul, abs_mul]
  have h1 : |(if sb = 1 then (-1 : ℚ) else 1)| = 1 := by split <;> simp
  rw [h1, one_mul, one_mul, ← sub_mul, abs_mul, abs_of_pos hP0, abs_of_nonneg (by positivity : (0 : ℚ) ≤ X)]
  have hd : |(Z : ℚ) - X| ≤ 2 ^ 15 := by
    rw [abs_le]; constructor
    · have : ((X : ℤ) - Z : ℚ) ≤ 2 ^ 15 := by exact_mod_cast hZX.2
      push_cast at this; linarith
    · have : ((Z : ℤ) - X : ℚ) ≤ 2 ^ 15 := by exact_mod_cast hZX.1
      push_cast at this; linarith
  have hfin' : |(Z : ℚ) - X| * P ≤ 2 ^ 15 * P := mul_le_mul_of_nonneg_right hd hP0.le
  by_cases e0 : e = 0
  · have : 2 ^ 15 * P = 2 ^ (-134 : ℤ) := by rw [hP, e0]; norm_num
    have h8 : 0 ≤ (X : ℚ) * P / 2 ^ 8 := by positivity
    calc |(Z : ℚ) - X| * P ≤ 2 ^ 15 * P := hfin'
      _ = 2 ^ (-134 : ℤ) := this
      _ ≤ _ := le_add_of_nonneg_left h8
  · have hX23 : (2 : ℚ) ^ 23 ≤ X := by
      rw [hX, ite_eq_right e0]; push_cast; linarith [(Nat.cast_nonneg (65536 * q + lo) : (0 : ℚ) ≤ _)]
    have : 2 ^ 15 * P ≤ (X : ℚ) * P / 2 ^ 8 := by
      rw [le_div_iff₀ (by norm_num)]; nlinarith
    have h134 := two_zpow_pos (-134)
    linarith

/-! ### Folds of rounded steps -/

/-- A range along a fold of word steps: `R` at each step's accumulator, as the steps before it wrote it. -/
def FoldRange {α : Type} (f : Nat → α → Option Nat) (R : Nat → α → Prop) : List α → Nat → Prop
  | [], _ => True
  | x :: xs, c => R c x ∧ ∀ c', f c x = some c' → FoldRange f R xs c'

theorem foldRange_true {α : Type} (f : Nat → α → Option Nat) :
    ∀ (xs : List α) (c : Nat), FoldRange f (fun _ _ => True) xs c
  | [], _ => trivial
  | _ :: xs, _ => ⟨trivial, fun _ _ => foldRange_true f xs _⟩

/-- **A fold of rounded steps**: when each step on the range writes a finite word within `ε·(|c| + a_x)` of
`c + t_x`, with `|t_x| ≤ a_x`, the fold is within `((1 + ε)^n − 1)·(|c| + Σ a_x)` of `c + Σ t_x`. -/
theorem fold_rel {α : Type} (f : Nat → α → Option Nat) (R : Nat → α → Prop) (t a : α → ℚ) {ε : ℚ} (hε : 0 ≤ ε)
    (ht : ∀ x, |t x| ≤ a x)
    (hstep : ∀ c x c' cv, R c x → wordQ c = some cv → f c x = some c' →
      ∃ v, wordQ c' = some v ∧ |v - (cv + t x)| ≤ ε * (|cv| + a x)) :
    ∀ (xs : List α) (c u : Nat) (cv : ℚ), wordQ c = some cv → xs.foldlM f c = some u → FoldRange f R xs c →
      ∃ uv, wordQ u = some uv ∧
        |uv - (cv + (xs.map t).sum)| ≤ ((1 + ε) ^ xs.length - 1) * (|cv| + (xs.map a).sum) := by
  intro xs
  induction xs with
  | nil =>
    intro c u cv hc hu _
    simp only [List.foldlM_nil, pure, Option.some.injEq] at hu
    subst hu
    exact ⟨cv, hc, by simp⟩
  | cons x xs ih =>
    intro c u cv hc hu hR
    rw [List.foldlM_cons] at hu
    obtain ⟨c', hc', hu'⟩ := Option.bind_eq_some_iff.1 hu
    obtain ⟨v, hv, he⟩ := hstep c x c' cv hR.1 hc hc'
    obtain ⟨uv, huv, he'⟩ := ih c' u v hv hu' (hR.2 c' hc')
    refine ⟨uv, huv, ?_⟩
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    set C := (1 + ε) ^ xs.length with hC
    set P := (xs.map a).sum
    set Rs := (xs.map t).sum
    have hC1 : 1 ≤ C := one_le_pow₀ (by linarith)
    have hP : 0 ≤ P := List.sum_nonneg (by
      simp only [List.mem_map]; rintro _ ⟨y, _, rfl⟩; exact le_trans (abs_nonneg _) (ht y))
    have hQ : 0 ≤ a x := le_trans (abs_nonneg _) (ht x)
    have hSQ := ht x
    have ha : |v| ≤ (1 + ε) * (|cv| + a x) := by
      have := abs_sub_abs_le_abs_sub v (cv + t x)
      have := abs_add_le cv (t x)
      nlinarith [abs_nonneg cv]
    calc |uv - (cv + (t x + Rs))| = |(uv - (v + Rs)) + (v - (cv + t x))| := by ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ (C - 1) * (|v| + P) + ε * (|cv| + a x) := add_le_add he' he
      _ ≤ (C - 1) * ((1 + ε) * (|cv| + a x) + P) + ε * (|cv| + a x) := by gcongr
      _ ≤ (C * (1 + ε) - 1) * (|cv| + (a x + P)) := by
          nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℚ) ≤ C) hε) hP]
      _ = _ := by rw [hC, ← pow_succ]

/-- **A fold of rounded steps, two rates**: when each step on the range writes a finite word within `ε·|c| + b_x` of
`c + t_x`, with `|t_x| ≤ a_x` and `0 ≤ b_x`, the fold is within `((1 + ε)^n − 1)·(|c| + Σ a_x) + (1 + ε)^n·Σ b_x` of
`c + Σ t_x`: only the rate on the running total compounds. -/
theorem fold_rel2 {α : Type} (f : Nat → α → Option Nat) (R : Nat → α → Prop) (t a b : α → ℚ) {ε : ℚ} (hε : 0 ≤ ε)
    (ht : ∀ x, |t x| ≤ a x) (hb : ∀ x, 0 ≤ b x)
    (hstep : ∀ c x c' cv, R c x → wordQ c = some cv → f c x = some c' →
      ∃ v, wordQ c' = some v ∧ |v - (cv + t x)| ≤ ε * |cv| + b x) :
    ∀ (xs : List α) (c u : Nat) (cv : ℚ), wordQ c = some cv → xs.foldlM f c = some u → FoldRange f R xs c →
      ∃ uv, wordQ u = some uv ∧ |uv - (cv + (xs.map t).sum)| ≤
        ((1 + ε) ^ xs.length - 1) * (|cv| + (xs.map a).sum) + (1 + ε) ^ xs.length * (xs.map b).sum := by
  intro xs
  induction xs with
  | nil =>
    intro c u cv hc hu _
    simp only [List.foldlM_nil, pure, Option.some.injEq] at hu
    subst hu
    exact ⟨cv, hc, by simp⟩
  | cons x xs ih =>
    intro c u cv hc hu hR
    rw [List.foldlM_cons] at hu
    obtain ⟨c', hc', hu'⟩ := Option.bind_eq_some_iff.1 hu
    obtain ⟨v, hv, he⟩ := hstep c x c' cv hR.1 hc hc'
    obtain ⟨uv, huv, he'⟩ := ih c' u v hv hu' (hR.2 c' hc')
    refine ⟨uv, huv, ?_⟩
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    set C := (1 + ε) ^ xs.length with hC
    set P := (xs.map a).sum
    set Bs := (xs.map b).sum
    set Rs := (xs.map t).sum
    have hC1 : 1 ≤ C := one_le_pow₀ (by linarith)
    have hP : 0 ≤ P := List.sum_nonneg (by
      simp only [List.mem_map]; rintro _ ⟨y, _, rfl⟩; exact le_trans (abs_nonneg _) (ht y))
    have hB : 0 ≤ Bs := List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨y, _, rfl⟩; exact hb y)
    have hQ : 0 ≤ a x := le_trans (abs_nonneg _) (ht x)
    have hbx := hb x
    have hSQ := ht x
    have ha : |v| ≤ (1 + ε) * |cv| + a x + b x := by
      have := abs_sub_abs_le_abs_sub v (cv + t x)
      have := abs_add_le cv (t x)
      linarith
    calc |uv - (cv + (t x + Rs))| = |(uv - (v + Rs)) + (v - (cv + t x))| := by ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ ((C - 1) * (|v| + P) + C * Bs) + (ε * |cv| + b x) := add_le_add he' he
      _ ≤ ((C - 1) * ((1 + ε) * |cv| + a x + b x + P) + C * Bs) + (ε * |cv| + b x) := by gcongr
      _ ≤ (C * (1 + ε) - 1) * (|cv| + (a x + P)) + C * (1 + ε) * (b x + Bs) := by
          nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℚ) ≤ C) hε) (by linarith : 0 ≤ a x + P + b x + Bs)]
      _ = _ := by rw [hC, ← pow_succ]

/-! ### The FMA chain -/

theorem fmaWord_some {a b c s : Nat} (h : fmaWord? a b c = some s) :
    ∃ x y z, wordQ a = some x ∧ wordQ b = some y ∧ wordQ c = some z := by
  unfold fmaWord? at h
  rcases ha : wordQ a with _ | x <;> rcases hb : wordQ b with _ | y <;> rcases hc : wordQ c with _ | z <;>
    simp only [ha, hb, hc, reduceCtorEq] at h
  exact ⟨x, y, z, rfl, rfl, rfl⟩

theorem wordVal_of {w : Nat} {v : ℚ} (h : wordQ w = some v) : PearlC.wordVal w = v := by
  unfold PearlC.wordVal; rw [h]

theorem bfVal_of {h : Nat} {v : ℚ} (hv : wordQ (h <<< 16) = some v) : bfVal h = v := by
  unfold bfVal; rw [hv]; rfl

/-- **The FMA chain's error**: `accChain? be gq` is within `((1 + 2^−24)^n − 1)·Σ_p (|be_p·g_p| + 2^−126)` of
`Σ_p be_p·g_p`, over the `n` pairs of `be` and `gq`. -/
theorem fmaStep_err {c c' : Nat} {x : Nat × Nat} {cv : ℚ} (hc : wordQ c = some cv)
    (hs : fmaWord? (x.1 <<< 16) x.2 c = some c') :
    ∃ v, wordQ c' = some v ∧ |v - (cv + bfVal x.1 * PearlC.wordVal x.2)| ≤
      1 / 2 ^ 24 * (|cv| + (|bfVal x.1 * PearlC.wordVal x.2| + 2 ^ (-126 : ℤ))) := by
  obtain ⟨p, g, z, hp, hg, hz⟩ := fmaWord_some hs
  rw [hc, Option.some.injEq] at hz
  subst hz
  obtain ⟨sv, hsv, hse⟩ := fmaWord_err hp hg hc hs
  refine ⟨sv, hsv, ?_⟩
  rw [bfVal_of hp, wordVal_of hg, add_comm cv]
  have h1 := abs_add_le (p * g) cv
  have e : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 24 * 2 ^ (-126 : ℤ) := by norm_num
  calc _ ≤ _ := hse
    _ ≤ (|p * g| + |cv|) / 2 ^ 24 + 2 ^ (-150 : ℤ) := by gcongr
    _ = 1 / 2 ^ 24 * (|cv| + (|p * g| + 2 ^ (-126 : ℤ))) := by rw [e]; ring

/-- **The FMA chain's error**: `accChain? be gq` is within `((1 + 2^−24)^n − 1)·Σ_p (|be_p·g_p| + 2^−126)` of
`Σ_p be_p·g_p`, over the `n` pairs of `be` and `gq`. -/
theorem accChain_err {be gq : List Nat} {acc : Nat} (h : accChain? be gq = some acc) :
    ∃ v, wordQ acc = some v ∧
      |v - ((be.zip gq).map fun x => bfVal x.1 * PearlC.wordVal x.2).sum| ≤
        ((1 + 1 / 2 ^ 24) ^ (be.zip gq).length - 1) *
          ((be.zip gq).map fun x => |bfVal x.1 * PearlC.wordVal x.2| + 2 ^ (-126 : ℤ)).sum := by
  have h126 := two_zpow_pos (-126)
  unfold accChain? at h
  have key := fold_rel (fun acc x => fmaWord? (x.1 <<< 16) x.2 acc) (fun _ _ => True)
    (fun x : Nat × Nat => bfVal x.1 * PearlC.wordVal x.2)
    (fun x : Nat × Nat => |bfVal x.1 * PearlC.wordVal x.2| + 2 ^ (-126 : ℤ))
    (ε := 1 / 2 ^ 24) (by positivity) (fun x => by linarith)
    (fun c x c' cv _ hc hs => fmaStep_err hc hs)
  obtain ⟨v, hv, he⟩ := key (be.zip gq) 0 acc 0 Freeze.wordQ_zero h (foldRange_true _ _ _)
  refine ⟨v, hv, ?_⟩
  rw [zero_add, abs_zero, zero_add] at he
  exact he

/-! ### The E4M3 chain -/

/-- An E4M3 code that is zero or normal: not NaN, and exponent field `0` only with a zero mantissa. -/
def ZeroOrNormal8 (c : Code) : Prop := c.isNaN = false ∧ (c.expField = 0 → c.manField = 0)

/-- **The E4M3 step's range** on the accumulator term `tc` and the codes `a`, `b`: every code is zero or normal, and
the step's exponent is at most 120, and at least −113 unless every term is zero. -/
def H8Range (tc : Term) (a b : Fin 32 → Code) : Prop :=
  (∀ l, ZeroOrNormal8 (a l)) ∧ (∀ l, ZeroOrNormal8 (b l)) ∧ Freeze.stepExp tc a b ≤ 120 ∧
    (-113 ≤ Freeze.stepExp tc a b ∨ (tc.mag = 0 ∧ ∀ l, (product (a l) (b l)).mag = 0))

/-- The E4M3 step's relative error on the range, `33·2^−13 + (1 + 33·2^−13)·2^−13`, about `2^−7.9`. -/
def eps8 : ℚ := 33 / 2 ^ 13 + (1 + 33 / 2 ^ 13) / 2 ^ 13

theorem eps8_nonneg : 0 ≤ eps8 := by unfold eps8; positivity

theorem val_eq_codeVal (c : Code) : val c = codeVal c := rfl

theorem code_sig_normal {c : Code} (hc : ZeroOrNormal8 c) (h0 : c.sig ≠ 0) : 8 ≤ c.sig := by
  unfold Code.sig at *
  split_ifs at * with he
  · exact absurd (hc.2 he) h0
  · omega

theorem product_normal8 {a b : Code} (ha : ZeroOrNormal8 a) (hb : ZeroOrNormal8 b) (hm : (product a b).mag ≠ 0) :
    (2 : ℚ) ^ (product a b).exp ≤ |val a * val b| := by
  have h := AtomErr.product_two_zpow ha.1 hb.1 hm
  have hs : a.sig * b.sig ≠ 0 := by
    intro h0; apply hm; unfold product; simp only [h0, Nat.zero_shiftLeft, ite_true]; rfl
  have h1 := code_sig_normal ha (fun h' => hs (by simp [h']))
  have h2 := code_sig_normal hb (fun h' => hs (by simp [h']))
  have h64 : (64 : ℚ) ≤ ((a.sig * b.sig : ℕ) : ℚ) := by exact_mod_cast (Nat.mul_le_mul h1 h2)
  have hp := two_zpow_pos (product a b).exp
  nlinarith

/-- On the range, the step's grid is bounded by its terms: `2^E ≤ |c| + Σ_l |a_l·b_l|`. -/
theorem two_zpow_stepExp_le' {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) {a b : Fin 32 → Code}
    (ha : ∀ l, ZeroOrNormal8 (a l)) (hb : ∀ l, ZeroOrNormal8 (b l)) (hE1 : -113 ≤ Freeze.stepExp tc a b) :
    (2 : ℚ) ^ Freeze.stepExp tc a b ≤ |tc.value| + ∑ l, |val (a l) * val (b l)| := by
  have hB : 0 ≤ ∑ l, |val (a l) * val (b l)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hc := abs_nonneg tc.value
  unfold Freeze.stepExp at hE1 ⊢
  rcases AtomErr.maxExp_cases (-139) (tc :: List.ofFn fun l => product (a l) (b l)) with e | ⟨t, ht, hm, e⟩
  · rw [e] at hE1; norm_num at hE1
  · rw [e] at hE1 ⊢
    rcases List.mem_cons.1 ht with rfl | ht
    · rcases AtomErr.fp32Value_cases hw with h0 | he | h23
      · exact absurd h0 hm
      · rw [he] at hE1; norm_num at hE1
      · have := two_zpow_term_le h23; linarith
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      have h1 := product_normal8 (ha l) (hb l) hm
      have h2 : |val (a l) * val (b l)| ≤ ∑ l, |val (a l) * val (b l)| :=
        Finset.single_le_sum (f := fun l => |val (a l) * val (b l)|) (fun _ _ => abs_nonneg _)
          (Finset.mem_univ l)
      linarith

/-- The step on zero terms writes `+0`. -/
theorem hopperStep_zero' {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (h0 : tc.mag = 0)
    (hp : ∀ l, (product (a l) (b l)).mag = 0) : wordQ (hopperStep w a b) = some 0 := by
  rw [Freeze.hopperStep_fin hw a b ha hb]
  have hs : ∀ E, alignedSum 14 E (tc :: List.ofFn fun l => product (a l) (b l)) = 0 := by
    intro E
    unfold alignedSum
    apply List.sum_eq_zero
    intro x hx
    simp only [List.mem_map, List.mem_cons, List.mem_ofFn] at hx
    obtain ⟨t, ht, rfl⟩ := hx
    rcases ht with rfl | ⟨l, rfl⟩
    · simp [aligned, h0]
    · simp [aligned, hp l]
  have hn : ∀ E, normalize 14 E 0 = Term.zero := fun E => by unfold normalize; rw [ite_eq_left rfl]
  unfold groupSum
  rw [hs, hn]
  show wordQ (packValue (if Term.zero.mag ≠ 0 ∧ 127 < Term.zero.exp then .inf Term.zero.neg else .fin Term.zero)) = _
  rw [ite_eq_right (fun h => absurd rfl h.1)]
  exact Freeze.wordQ_zero

theorem rel_of_err8 {v S B g : ℚ} (he : |v - S| ≤ 33 * g + (|S| + 33 * g) / 2 ^ 13) (hS : |S| ≤ B)
    (hg : g ≤ B / 2 ^ 13) : |v - S| ≤ eps8 * B := by
  calc _ ≤ _ := he
    _ ≤ 33 * (B / 2 ^ 13) + (B + 33 * (B / 2 ^ 13)) / 2 ^ 13 := by gcongr
    _ = eps8 * B := by unfold eps8; ring

/-- **The E4M3 step's relative error on the range**: the word is finite, and within `eps8·(|c| + Σ_l |a_l·b_l|)` of
`c + Σ_l a_l·b_l`. -/
theorem hopperStep_rel {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) {a b : Fin 32 → Code}
    (hR : H8Range tc a b) :
    ∃ v, wordQ (hopperStep w a b) = some v ∧
      |v - (tc.value + ∑ l, val (a l) * val (b l))| ≤ eps8 * (|tc.value| + ∑ l, |val (a l) * val (b l)|) := by
  obtain ⟨ha, hb, hE2, hE1 | ⟨h0, hp⟩⟩ := hR
  · have hv := AtomErr.hopperStep_value hw a b (fun l => (ha l).1) (fun l => (hb l).1) hE1 hE2
    refine ⟨_, hv, ?_⟩
    have he := AtomErr.hopperStep_err hw a b (fun l => (ha l).1) (fun l => (hb l).1) hE1 hE2
    rw [wordVal_of hv] at he
    have hB := two_zpow_stepExp_le' hw ha hb hE1
    have hS : |tc.value + ∑ l, val (a l) * val (b l)| ≤ |tc.value| + ∑ l, |val (a l) * val (b l)| := by
      have h1 := abs_add_le tc.value (∑ l, val (a l) * val (b l))
      have h2 := Finset.abs_sum_le_sum_abs (fun l => val (a l) * val (b l)) Finset.univ
      linarith
    have hg : (2 : ℚ) ^ (Freeze.stepExp tc a b - 13) ≤ (|tc.value| + ∑ l, |val (a l) * val (b l)|) / 2 ^ 13 := by
      rw [zpow_sub₀ two_ne_zero]
      exact div_le_div_of_nonneg_right hB (by norm_num)
    exact rel_of_err8 he hS hg
  · refine ⟨0, hopperStep_zero' hw a b (fun l => (ha l).1) (fun l => (hb l).1) h0 hp, ?_⟩
    have hc : tc.value = 0 := by simp [Term.value, h0]
    have hp' : ∀ l, val (a l) * val (b l) = 0 := by
      intro l
      rw [← productExact _ _ _ _ (val_eq (ha l).1) (val_eq (hb l).1)]
      simp [Term.value, hp l]
    simp only [hc, hp', Finset.sum_const_zero, add_zero, sub_zero, abs_zero, mul_zero, le_refl]

/-- An atom's absolute products, `Σ_l |a_l·b_l|`. -/
def atomAbs (x : AtomIn) : ℚ := ∑ l, |codeVal (x.1 l) * codeVal (x.2 l)|

/-- One window's step: `hopperStep` on the atom, the word finite. -/
def winStep (w : Nat) (x : AtomIn) : Option Nat := finite? (hopperStep w x.1 x.2)

/-- The range along a window's steps from `+0`. -/
def WindowRange (g : List AtomIn) : Prop :=
  FoldRange winStep (fun c x => ∀ tc, fp32Value c = .fin tc → H8Range tc x.1 x.2) g 0

/-- **A window's error**: its word is within `((1 + eps8)^n − 1)·Σ |a·b|` of its atoms' exact sum. -/
theorem windowWord_err {g : List AtomIn} {w : Nat} (h : windowWord? g = some w) (hR : WindowRange g) :
    ∃ v, wordQ w = some v ∧ |v - (g.map atomQ).sum| ≤ ((1 + eps8) ^ g.length - 1) * (g.map atomAbs).sum := by
  have key := fold_rel winStep (fun c x => ∀ tc, fp32Value c = .fin tc → H8Range tc x.1 x.2) atomQ atomAbs
    eps8_nonneg (fun x => Finset.abs_sum_le_sum_abs _ _)
    (fun c x c' cv hR hc hs => by
      obtain ⟨tc, htc, htv⟩ := wordQ_fin hc
      obtain ⟨rfl, -⟩ := finite?_eq_some hs
      obtain ⟨v, hv, he⟩ := hopperStep_rel htc (hR tc htc)
      exact ⟨v, hv, by rw [← htv]; exact he⟩)
  obtain ⟨v, hv, he⟩ := key g 0 w 0 Freeze.wordQ_zero h hR
  refine ⟨v, hv, ?_⟩
  rw [zero_add, abs_zero, zero_add] at he
  exact he

/-- The chain's step: a window's word promoted into the total. -/
def chainStep (t : Nat) (g : List AtomIn) : Option Nat := (windowWord? g).bind (promoteWord? t)

/-- **The chain's range**: every window's range, along the chain's windows. -/
def ChainRange (atoms : List AtomIn) : Prop :=
  FoldRange chainStep (fun _ g => WindowRange g) (atoms.toChunks Params.pi.G) 0

/-- A promotion's relative error: `(1 + 2^−24)·(1 + eps8)^4 − 1`, about `2^−5.9`, per window of at most four atoms. -/
def epsC : ℚ := (1 + 1 / 2 ^ 24) * (1 + eps8) ^ 4 - 1

theorem epsC_ge : 1 / 2 ^ 24 ≤ epsC := by
  unfold epsC
  have h := one_le_pow₀ (n := 4) (by linarith [eps8_nonneg] : (1 : ℚ) ≤ 1 + eps8)
  nlinarith

theorem foldRange_and {α : Type} (f : Nat → α → Option Nat) (R Q : Nat → α → Prop) :
    ∀ (xs : List α) (c : Nat), (∀ x ∈ xs, ∀ c, Q c x) → FoldRange f R xs c →
      FoldRange f (fun c x => Q c x ∧ R c x) xs c
  | [], _, _, _ => trivial
  | x :: xs, c, hQ, hR => ⟨⟨hQ x List.mem_cons_self c, hR.1⟩, fun c' hc' =>
      foldRange_and f R Q xs c' (fun y hy => hQ y (List.mem_cons_of_mem _ hy)) (hR.2 c' hc')⟩

theorem promoteWord_val {t w s : Nat} (h : promoteWord? t w = some s) :
    ∃ s', addWord? t w = some s' ∧ wordQ s = wordQ s' := by
  unfold promoteWord? at h
  obtain ⟨s', hs', rfl⟩ := Option.map_eq_some_iff.1 h
  refine ⟨s', hs', ?_⟩
  by_cases h0 : s' = 0x80000000
  · rw [ite_eq_left h0, h0, Freeze.wordQ_zero, wordQ_negZero]
  · rw [ite_eq_right h0]

theorem abs_listSum_le {α : Type} (f a : α → ℚ) (h : ∀ x, |f x| ≤ a x) :
    ∀ l : List α, |(l.map f).sum| ≤ (l.map a).sum
  | [] => by simp
  | x :: xs => by
    simp only [List.map_cons, List.sum_cons]
    exact le_trans (abs_add_le _ _) (add_le_add (h x) (abs_listSum_le f a h xs))

theorem chainStep_err {c c' : Nat} {g : List AtomIn} {cv : ℚ} (hg : g.length ≤ 4) (hR : WindowRange g)
    (hc : wordQ c = some cv) (hs : chainStep c g = some c') :
    ∃ v, wordQ c' = some v ∧
      |v - (cv + (g.map atomQ).sum)| ≤ 1 / 2 ^ 24 * |cv| + (epsC * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)) := by
  obtain ⟨W, hW, hp⟩ := Option.bind_eq_some_iff.1 hs
  obtain ⟨wv, hwv, hwe⟩ := windowWord_err hW hR
  obtain ⟨s', hs', hss⟩ := promoteWord_val hp
  obtain ⟨sv, hsv, hse⟩ := addWord_err hc hwv hs'
  refine ⟨sv, by rw [hss]; exact hsv, ?_⟩
  set A := (g.map atomAbs).sum
  set Wq := (g.map atomQ).sum
  have hWA : |Wq| ≤ A := abs_listSum_le atomQ atomAbs (fun x => Finset.abs_sum_le_sum_abs _ _) g
  have hε := eps8_nonneg
  have hK : (1 + eps8) ^ g.length ≤ (1 + eps8) ^ 4 := pow_le_pow_right₀ (by linarith) hg
  have hK1 : 1 ≤ (1 + eps8) ^ g.length := one_le_pow₀ (by linarith)
  have hA : 0 ≤ A := le_trans (abs_nonneg _) hWA
  have hwe' : |wv - Wq| ≤ ((1 + eps8) ^ 4 - 1) * A := le_trans hwe (by gcongr)
  have hwa : |wv| ≤ (1 + eps8) ^ 4 * A := by
    have := abs_sub_abs_le_abs_sub wv Wq; nlinarith
  have hcw : |cv + wv| ≤ |cv| + (1 + eps8) ^ 4 * A := le_trans (abs_add_le _ _) (by linarith)
  calc |sv - (cv + Wq)| = |(sv - (cv + wv)) + (wv - Wq)| := by ring_nf
    _ ≤ |sv - (cv + wv)| + |wv - Wq| := abs_add_le _ _
    _ ≤ (|cv| + (1 + eps8) ^ 4 * A) / 2 ^ 24 + 2 ^ (-150 : ℤ) + ((1 + eps8) ^ 4 - 1) * A := by
        gcongr; exact le_trans hse (by gcongr)
    _ = _ := by unfold epsC; ring

theorem chunksOf_length_le {α : Type} (n : Nat) : ∀ (l : List α), ∀ g ∈ Freeze.chunksOf n l, g.length ≤ n + 1
  | [] => by simp [Freeze.chunksOf]
  | x :: xs => by
    rw [Freeze.chunksOf_cons]
    intro g hg
    rcases List.mem_cons.1 hg with rfl | hg
    · simp only [List.length_cons, List.length_take]; omega
    · exact chunksOf_length_le n (xs.drop n) g hg
termination_by l => l.length
decreasing_by simp; omega

theorem chunksOf_sum {α : Type} (n : Nat) (f : α → ℚ) : ∀ (l : List α),
    ((Freeze.chunksOf n l).map fun g => (g.map f).sum).sum = (l.map f).sum
  | [] => by simp [Freeze.chunksOf]
  | x :: xs => by
    rw [Freeze.chunksOf_cons, List.map_cons, List.sum_cons, chunksOf_sum n f (xs.drop n)]
    conv_rhs => rw [← List.take_append_drop n xs]
    simp only [List.map_cons, List.sum_cons, List.map_append, List.sum_append]
    ring
termination_by l => l.length
decreasing_by simp; omega

theorem toChunks4 (l : List AtomIn) : l.toChunks Params.pi.G = Freeze.chunksOf 3 l := Freeze.toChunks_succ 3 l

/-- The chain's error bound over its `w` windows: `((1 + 2^−24)^w − 1)·Σ |a·b| + (1 + 2^−24)^w·(epsC·Σ |a·b| +
w·2^−150)`. The windows' error, `epsC`, does not compound; only the FP32 promotions' `2^−24` does. -/
def chainBound (atoms : List AtomIn) : ℚ :=
  ((1 + 1 / 2 ^ 24) ^ (atoms.toChunks Params.pi.G).length - 1) * (atoms.map atomAbs).sum +
    (1 + 1 / 2 ^ 24) ^ (atoms.toChunks Params.pi.G).length *
      (epsC * (atoms.map atomAbs).sum + (atoms.toChunks Params.pi.G).length * 2 ^ (-150 : ℤ))

/-- **The E4M3 chain's error on its range**: `chainWord?`'s word is within `chainBound` of the exact chain. -/
theorem chainWord_err {atoms : List AtomIn} {w : Nat} (h : chainWord? atoms = some w) (hR : ChainRange atoms) :
    ∃ v, wordQ w = some v ∧ |v - chainQ atoms| ≤ chainBound atoms := by
  have hlen : ∀ g ∈ atoms.toChunks Params.pi.G, ∀ _c : Nat, g.length ≤ 4 := by
    intro g hg _; rw [toChunks4] at hg; exact chunksOf_length_le 3 atoms g hg
  have hA : ∀ g : List AtomIn, |(g.map atomQ).sum| ≤ (g.map atomAbs).sum :=
    abs_listSum_le atomQ atomAbs (fun x => Finset.abs_sum_le_sum_abs _ _)
  have key := fold_rel2 chainStep (fun _ g => g.length ≤ 4 ∧ WindowRange g) (fun g => (g.map atomQ).sum)
    (fun g => (g.map atomAbs).sum) (fun g => epsC * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)) (by positivity) hA
    (fun g => add_nonneg (mul_nonneg (le_trans (by positivity) epsC_ge) (le_trans (abs_nonneg _) (hA g)))
      (by positivity))
    (fun c g c' cv hR hc hs => chainStep_err hR.1 hR.2 hc hs)
  obtain ⟨v, hv, he⟩ := key (atoms.toChunks Params.pi.G) 0 w 0 Freeze.wordQ_zero h
    (foldRange_and _ _ _ _ _ hlen hR)
  refine ⟨v, hv, ?_⟩
  rw [zero_add, abs_zero, zero_add] at he
  have e1 : ((atoms.toChunks Params.pi.G).map fun g => (g.map atomQ).sum).sum = chainQ atoms := by
    rw [toChunks4, chunksOf_sum]; rfl
  have e2 : ((atoms.toChunks Params.pi.G).map fun g => (g.map atomAbs).sum).sum = (atoms.map atomAbs).sum := by
    rw [toChunks4, chunksOf_sum]
  have e3 : ((atoms.toChunks Params.pi.G).map fun g => epsC * (g.map atomAbs).sum + 2 ^ (-150 : ℤ)).sum =
      epsC * (atoms.map atomAbs).sum + (atoms.toChunks Params.pi.G).length * 2 ^ (-150 : ℤ) := by
    rw [List.sum_map_add, List.map_const', List.sum_replicate, nsmul_eq_mul, List.sum_map_mul_left, e2]
  rw [e1, e2, e3] at he
  exact he

/-! ### The served rows, unpacked -/

theorem mapM_getD {α β : Type} (f : α → Option β) (dα : α) (dβ : β) :
    ∀ (l : List α) (l' : List β), l.mapM f = some l' →
      l'.length = l.length ∧ ∀ i < l.length, f (l.getD i dα) = some (l'.getD i dβ)
  | [], l', h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h
    subst h; simp
  | x :: xs, l', h => by
    rw [List.mapM_cons] at h
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
    obtain ⟨y, hy, ys, hys, rfl⟩ := h
    obtain ⟨hl, hi⟩ := mapM_getD f dα dβ xs ys hys
    refine ⟨by simp [hl], fun i hi' => ?_⟩
    rcases i with _ | i
    · simpa using hy
    · simpa using hi i (by simpa using hi')

theorem mapM_ofFn {β : Type} {N : Nat} (f : Fin N → Option β) (d : β) {l : List β}
    (h : (List.ofFn f).mapM id = some l) : l.length = N ∧ ∀ p : Fin N, f p = some (l.getD p d) := by
  obtain ⟨hl, hi⟩ := mapM_getD id none d _ _ h
  rw [List.length_ofFn] at hl
  refine ⟨hl, fun p => ?_⟩
  have := hi p (by simp)
  rwa [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn] at this

theorem getD_ofFn {β : Type} {N : Nat} (f : Fin N → β) (d : β) (p : Fin N) : (List.ofFn f).getD p d = f p := by
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem peelRows_spec {Aq : Matrix (Fin m) (Fin k) Code} {Bq : Matrix (Fin n) (Fin k) Code}
    {FA FB : Matrix (Fin k) (Fin 32) Code} {beA : Matrix (Fin m) (Fin 32) Nat} {beB : Matrix (Fin n) (Fin 32) Nat}
    {i : Fin m} {j : Fin n} {c : Nat} {pa pb : List Nat}
    (h : peelRows Aq Bq FA FB beA beB i j = some (c, pa, pb)) :
    ∃ af bf g, chainWord? (rowAtoms (Aq i) (Bq j)) = some c ∧
      (List.ofFn fun p => chainWord? (rowAtoms (Aq i) (FBᵀ p))).mapM id = some af ∧
      (List.ofFn fun q => chainWord? (rowAtoms (Bq j) (FAᵀ q))).mapM id = some bf ∧
      (List.ofFn fun p => (List.ofFn fun q => chainWord? (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id = some g ∧
      paRow? (List.ofFn (beA i)) af = some pa ∧ pbRow? (List.ofFn (beB j)) bf g = some pb := by
  unfold peelRows at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨c', hc, af, haf, bf, hbf, g, hg, pa', hpa, pb', hpb, rfl, rfl, rfl⟩ := h
  exact ⟨af, bf, g, hc, haf, hbf, hg, hpa, hpb⟩

theorem paRow_spec {be af pa : List Nat} (h : paRow? be af = some pa) (hbe : be.length = 32) :
    (∀ t < 32, pa.getD t 0 = be.getD t 0) ∧ ∀ p < af.length, rnBf16? (af.getD p 0) = some (pa.getD (32 + p) 0) := by
  unfold paRow? at h
  obtain ⟨hs, hhs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨-, hi⟩ := mapM_getD rnBf16? 0 0 af hs hhs
  refine ⟨fun t ht => List.getD_append _ _ _ _ (by omega), fun p hp => ?_⟩
  rw [List.getD_append_right _ _ _ _ (by omega), show 32 + p - be.length = p by omega]
  exact hi p hp

theorem pbRow_spec {be bf pb : List Nat} {g : List (List Nat)} (h : pbRow? be bf g = some pb) (hbe : be.length = 32) :
    (∀ q < 32, ∃ acc s, accChain? be (col g q) = some acc ∧ addWord? acc (negWord (bf.getD q 0)) = some s ∧
      rnBf16? s = some (pb.getD q 0)) ∧ ∀ p < 32, pb.getD (32 + p) 0 = be.getD p 0 ^^^ 0x8000 := by
  unfold pbRow? at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨mid, hmid, rfl⟩ := h
  obtain ⟨hl, hi⟩ := mapM_getD _ 0 0 _ mid hmid
  have hl' : mid.length = 32 := by rw [hl]; rfl
  refine ⟨fun q hq => ?_, fun p hp => ?_⟩
  · have := hi q (by simpa [Params.pi] using hq)
    rw [List.getD_eq_getElem _ _ (by simpa [Params.pi] using hq), List.getElem_range] at this
    simp only [Option.bind_eq_some_iff] at this
    obtain ⟨acc, hacc, s, hs, hh⟩ := this
    refine ⟨acc, s, hacc, hs, ?_⟩
    rw [List.getD_append _ _ _ _ (by omega)]
    exact hh
  · rw [List.getD_append_right _ _ _ _ (by omega), show 32 + p - mid.length = p by omega,
      List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map, List.getD_eq_getElem _ _ (by omega)]

/-! ### The rows' entries -/

/-- `P_A`'s rounded half: `RN_bf16` of a chain word within `δ` of `A` is within `δ + (|A| + δ)·2^−8 + 2^−134` of `A`. -/
theorem paEntry_err {aw h : Nat} {A δ : ℚ} (ha : ∃ av, wordQ aw = some av ∧ |av - A| ≤ δ) (hh : rnBf16? aw = some h) :
    |bfVal h - A| ≤ δ + (|A| + δ) / 2 ^ 8 + 2 ^ (-134 : ℤ) := by
  obtain ⟨av, hav, he⟩ := ha
  have hr := (rnBf16_err hav hh).2
  have h1 : |av| ≤ |A| + δ := by have := abs_sub_abs_le_abs_sub av A; linarith
  calc |bfVal h - A| = |(bfVal h - av) + (av - A)| := by ring_nf
    _ ≤ |bfVal h - av| + |av - A| := abs_add_le _ _
    _ ≤ (|A| + δ) / 2 ^ 8 + 2 ^ (-134 : ℤ) + δ := add_le_add (le_trans hr (by gcongr)) he
    _ = _ := by ring

/-- The bound on `P_B`'s rounded half: the FMA chain's error over the served `G` (within `δG` of `G`), then `B̃·F_A`'s
chain error `δB`, the FP32 difference's rounding and the `RN_bf16`, against `D = Σ_p be_p·G_p − B`. -/
def pbErrOf (be : Fin 32 → Nat) (G δG : Fin 32 → ℚ) (B δB : ℚ) : ℚ :=
  let D := (∑ p, bfVal (be p) * G p) - B
  let e1 := ((1 + 1 / 2 ^ 24) ^ 32 - 1) * ∑ p, (|bfVal (be p)| * (|G p| + δG p) + 2 ^ (-126 : ℤ)) +
    ∑ p, |bfVal (be p)| * δG p + δB
  let d := e1 + (|D| + e1) / 2 ^ 24 + 2 ^ (-150 : ℤ)
  d + (|D| + d) / 2 ^ 8 + 2 ^ (-134 : ℤ)

theorem zip_col {be : Fin 32 → Nat} {g : List (List Nat)} (hg : g.length = 32) (q : Nat) :
    (List.ofFn be).zip (PeelExact.col g q) = List.ofFn fun p : Fin 32 => (be p, (g.getD p []).getD q 0) := by
  apply List.ext_getElem
  · simp [PeelExact.col, hg]
  · intro p h1 h2
    have hp : p < g.length := by rw [hg]; simpa using h2
    simp only [List.getElem_zip, List.getElem_ofFn, PeelExact.col, List.getElem_map]
    rw [List.getD_eq_getElem _ _ hp]

/-- **`P_B`'s rounded half**: the served entry is within `pbErrOf` of `Σ_p be_p·G_p − B`. -/
theorem pbEntry_err {be : Fin 32 → Nat} {g : List (List Nat)} {q : Nat} {bw acc s h : Nat} {G δG : Fin 32 → ℚ}
    {B δB : ℚ} (hg : g.length = 32)
    (hgw : ∀ p : Fin 32, ∃ gv, wordQ ((g.getD p []).getD q 0) = some gv ∧ |gv - G p| ≤ δG p)
    (hb : ∃ bv, wordQ bw = some bv ∧ |bv - B| ≤ δB)
    (hacc : accChain? (List.ofFn be) (PeelExact.col g q) = some acc) (hs : addWord? acc (negWord bw) = some s)
    (hh : rnBf16? s = some h) :
    |bfVal h - ((∑ p, bfVal (be p) * G p) - B)| ≤ pbErrOf be G δG B δB := by
  choose gv hgv hge using hgw
  obtain ⟨bv, hbv, hbe⟩ := hb
  obtain ⟨av, hav, hae⟩ := accChain_err hacc
  rw [zip_col hg, List.length_ofFn, List.map_ofFn, List.map_ofFn, List.sum_ofFn, List.sum_ofFn] at hae
  simp only [Function.comp_apply, wordVal_of (hgv _)] at hae
  have hneg : wordQ (negWord bw) = some (-bv) := by rw [wordQ_negWord, hbv]; rfl
  obtain ⟨sv, hsv, hse⟩ := addWord_err hav hneg hs
  have hr := (rnBf16_err hsv hh).2
  set D := (∑ p, bfVal (be p) * G p) - B with hD
  set F : ℚ := (1 + 1 / 2 ^ 24) ^ 32 - 1 with hF
  have hF0 : 0 ≤ F := by rw [hF]; have := one_le_pow₀ (n := 32) (by norm_num : (1 : ℚ) ≤ 1 + 1 / 2 ^ 24); linarith
  have hδG : ∀ p, 0 ≤ δG p := fun p => le_trans (abs_nonneg _) (hge p)
  have h1 : |(∑ p, bfVal (be p) * gv p) - ∑ p, bfVal (be p) * G p| ≤ ∑ p, |bfVal (be p)| * δG p := by
    rw [← Finset.sum_sub_distrib]
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun p _ => ?_)
    rw [← mul_sub, abs_mul]; exact mul_le_mul_of_nonneg_left (hge p) (abs_nonneg _)
  have h2 : ∑ p, (|bfVal (be p) * gv p| + 2 ^ (-126 : ℤ)) ≤
      ∑ p, (|bfVal (be p)| * (|G p| + δG p) + 2 ^ (-126 : ℤ)) := by
    refine Finset.sum_le_sum fun p _ => ?_
    rw [abs_mul]
    have : |gv p| ≤ |G p| + δG p := by have := abs_sub_abs_le_abs_sub (gv p) (G p); linarith [hge p]
    have := mul_le_mul_of_nonneg_left this (abs_nonneg (bfVal (be p)))
    linarith
  set e1 := F * ∑ p, (|bfVal (be p)| * (|G p| + δG p) + 2 ^ (-126 : ℤ)) + ∑ p, |bfVal (be p)| * δG p + δB
  have hacc' : |av - ∑ p, bfVal (be p) * G p| ≤ e1 - δB := by
    have := mul_le_mul_of_nonneg_left h2 hF0
    calc |av - ∑ p, bfVal (be p) * G p| =
        |(av - ∑ p, bfVal (be p) * gv p) + ((∑ p, bfVal (be p) * gv p) - ∑ p, bfVal (be p) * G p)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ _ := add_le_add hae h1
      _ ≤ _ := by linarith
  have hdiff : |(av + -bv) - D| ≤ e1 := by
    calc |(av + -bv) - D| = |(av - ∑ p, bfVal (be p) * G p) - (bv - B)| := by rw [hD]; ring_nf
      _ ≤ |av - ∑ p, bfVal (be p) * G p| + |bv - B| := abs_sub _ _
      _ ≤ e1 := by linarith
  have habs : |av + -bv| ≤ |D| + e1 := by have := abs_sub_abs_le_abs_sub (av + -bv) D; linarith
  set d := e1 + (|D| + e1) / 2 ^ 24 + 2 ^ (-150 : ℤ)
  have hsd : |sv - D| ≤ d := by
    calc |sv - D| = |(sv - (av + -bv)) + ((av + -bv) - D)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ (|D| + e1) / 2 ^ 24 + 2 ^ (-150 : ℤ) + e1 := add_le_add (le_trans hse (by gcongr)) hdiff
      _ = d := by ring
  have hsa : |sv| ≤ |D| + d := by have := abs_sub_abs_le_abs_sub sv D; linarith
  show |bfVal h - D| ≤ d + (|D| + d) / 2 ^ 8 + 2 ^ (-134 : ℤ)
  calc |bfVal h - D| = |(bfVal h - sv) + (sv - D)| := by ring_nf
    _ ≤ _ := abs_add_le _ _
    _ ≤ (|D| + d) / 2 ^ 8 + 2 ^ (-134 : ℤ) + d := add_le_add (le_trans hr (by gcongr)) hsd
    _ = _ := by ring

/-! ### The served entry against its exact counterpart -/

section Bound

variable (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
  (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)

/-- `P_A,i`'s rounded entry `32 + p`: `(A′·F_B)_ip`'s chain bound, then its `RN_bf16`. -/
def paErr (i : Fin m) (p : Fin 32) : ℚ :=
  chainBound (rowAtoms (Aq i) (FBᵀ p)) + (|afQ Aq FB i p| + chainBound (rowAtoms (Aq i) (FBᵀ p))) / 2 ^ 8 +
    2 ^ (-134 : ℤ)

/-- `P_B,j`'s rounded entry `q`: `pbErrOf` with `G`'s and `B̃·F_A`'s chain bounds. -/
def pbErr (j : Fin n) (q : Fin 32) : ℚ :=
  pbErrOf (beB j) (fun p => gQ FA FB p q) (fun p => chainBound (rowAtoms (FBᵀ p) (FAᵀ q))) (bfQ Bq FA j q)
    (chainBound (rowAtoms (Bq j) (FAᵀ q)))

/-- Position `t`'s row error: the exact entry of one row times the other row's rounding. -/
def rowErr (i : Fin m) (j : Fin n) (t : Fin 64) : ℚ :=
  if h : t.val < 32 then |paQ Aq FB beA i t| * pbErr Bq FA FB beB j ⟨t.val, h⟩
  else paErr Aq FB i ⟨t.val - 32, by have := t.isLt; omega⟩ * |pbQ Bq FA FB beB j t|

/-- **The served entry's bound**: U's four K16 steps, `C̃`'s chain and the rows' errors, in exact quantities. -/
def peelBound (i : Fin m) (j : Fin n) : ℚ :=
  ((1 + eps16) ^ 4 - 1) * (|cQ Aq Bq i j| + chainBound (rowAtoms (Aq i) (Bq j)) +
      ∑ t, (|paQ Aq FB beA i t * pbQ Bq FA FB beB j t| + rowErr Aq Bq FA FB beA beB i j t)) +
    chainBound (rowAtoms (Aq i) (Bq j)) + ∑ t, rowErr Aq Bq FA FB beA beB i j t

/-- **The range**: every E4M3 chain of the entry on its range (`ChainRange`), and U's steps on the K16 range
(`URange`) over the served rows. -/
def PeelRange (i : Fin m) (j : Fin n) : Prop :=
  ChainRange (rowAtoms (Aq i) (Bq j)) ∧ (∀ p, ChainRange (rowAtoms (Aq i) (FBᵀ p))) ∧
    (∀ q, ChainRange (rowAtoms (Bq j) (FAᵀ q))) ∧ (∀ p q, ChainRange (rowAtoms (FBᵀ p) (FAᵀ q))) ∧
    ∀ c pa pb, peelRows Aq Bq FA FB beA beB i j = some (c, pa, pb) → URange pa pb (List.range 4) c

/-- **(b) The served U is within `peelBound` of its exact counterpart** on the range, wherever the entry is finite. -/
theorem peelExactVal_err (i : Fin m) (j : Fin n) {v : ℚ} (hv : peelExactVal Aq Bq FA FB beA beB i j = some v)
    (hR : PeelRange Aq Bq FA FB beA beB i j) :
    |v - uQ Aq Bq FA FB beA beB i j| ≤ peelBound Aq Bq FA FB beA beB i j := by
  have hv' := hv
  unfold peelExactVal at hv'
  rw [peelExact_eq_rows] at hv'
  obtain ⟨u, hu, -⟩ := Option.bind_eq_some_iff.1 hv'
  obtain ⟨⟨c, pa, pb⟩, hrows, -⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨af, bf, g, hc, haf, hbf, hg, hpa, hpb⟩ := peelRows_spec hrows
  obtain ⟨hRc, hRa, hRb, hRg, hRu⟩ := hR
  have he := peelExact_err Aq Bq FA FB beA beB i j hrows hv (hRu c pa pb hrows)
  obtain ⟨cv, hcv, hce⟩ := chainWord_err hc hRc
  obtain ⟨hafl, hafw⟩ := mapM_ofFn _ 0 haf
  obtain ⟨-, hbfw⟩ := mapM_ofFn _ 0 hbf
  obtain ⟨hgl, hgw⟩ := mapM_ofFn _ [] hg
  have hgw' : ∀ p q : Fin 32, chainWord? (rowAtoms (FBᵀ p) (FAᵀ q)) = some ((g.getD p []).getD q 0) :=
    fun p q => (mapM_ofFn _ 0 (hgw p)).2 q
  obtain ⟨hpa1, hpa2⟩ := paRow_spec hpa (by simp)
  obtain ⟨hpb1, hpb2⟩ := pbRow_spec hpb (by simp)
  have hrow : ∀ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0) - paQ Aq FB beA i t * pbQ Bq FA FB beB j t| ≤
      rowErr Aq Bq FA FB beA beB i j t := by
    intro t
    by_cases ht : t.val < 32
    · have e1 : bfVal (pa.getD t 0) = paQ Aq FB beA i t := by
        rw [hpa1 t ht, show (t : Nat) = ((⟨t.val, ht⟩ : Fin 32) : Nat) from rfl, getD_ofFn]
        unfold paQ; rw [dite_eq_left ht]
      obtain ⟨acc, s, hacc, hs, hh⟩ := hpb1 t ht
      have hpbe := pbEntry_err (be := beB j) (G := fun p => gQ FA FB p ⟨t.val, ht⟩)
        (δG := fun p => chainBound (rowAtoms (FBᵀ p) (FAᵀ ⟨t.val, ht⟩))) hgl
        (fun p => chainWord_err (hgw' p ⟨t.val, ht⟩) (hRg _ _))
        (chainWord_err (hbfw ⟨t.val, ht⟩) (hRb _)) hacc hs hh
      have e2 : pbQ Bq FA FB beB j t = (∑ p, bfVal (beB j p) * gQ FA FB p ⟨t.val, ht⟩) - bfQ Bq FA j ⟨t.val, ht⟩ := by
        unfold pbQ accQ; rw [dite_eq_left ht]
      unfold rowErr
      rw [dite_eq_left ht, e1, ← mul_sub, abs_mul, e2]
      exact mul_le_mul_of_nonneg_left hpbe (abs_nonneg _)
    · have hp : t.val - 32 < 32 := by have := t.isLt; omega
      have e1 : bfVal (pb.getD t 0) = pbQ Bq FA FB beB j t := by
        have h1 := hpb2 (t.val - 32) hp
        rw [show 32 + (t.val - 32) = t.val by omega,
          show t.val - 32 = ((⟨t.val - 32, hp⟩ : Fin 32) : Nat) from rfl, getD_ofFn] at h1
        rw [h1, bfVal_xor]
        unfold pbQ; rw [dite_eq_right ht]
      have hpa' : rnBf16? (af.getD (t.val - 32) 0) = some (pa.getD t 0) := by
        have h1 := hpa2 (t.val - 32) (by rw [hafl]; exact hp)
        rwa [show 32 + (t.val - 32) = t.val by omega] at h1
      have hpae := paEntry_err (A := afQ Aq FB i ⟨t.val - 32, hp⟩)
        (δ := chainBound (rowAtoms (Aq i) (FBᵀ ⟨t.val - 32, hp⟩))) (chainWord_err (hafw _) (hRa _)) hpa'
      have e2 : paQ Aq FB beA i t = afQ Aq FB i ⟨t.val - 32, hp⟩ := by unfold paQ; rw [dite_eq_right ht]
      unfold rowErr
      rw [dite_eq_right ht, e1, ← sub_mul, abs_mul, e2]
      exact mul_le_mul_of_nonneg_right hpae (abs_nonneg _)
  set δC := chainBound (rowAtoms (Aq i) (Bq j))
  have hwc : PearlC.wordVal c = cv := wordVal_of hcv
  have hC : |PearlC.wordVal c - cQ Aq Bq i j| ≤ δC := by rw [hwc]; exact hce
  have hCa : |PearlC.wordVal c| ≤ |cQ Aq Bq i j| + δC := by
    have := abs_sub_abs_le_abs_sub (PearlC.wordVal c) (cQ Aq Bq i j); linarith
  have hS : ∑ t : Fin 64, |bfVal (pa.getD t 0) * bfVal (pb.getD t 0)| ≤
      ∑ t, (|paQ Aq FB beA i t * pbQ Bq FA FB beB j t| + rowErr Aq Bq FA FB beA beB i j t) := by
    refine Finset.sum_le_sum fun t _ => ?_
    have := abs_sub_abs_le_abs_sub (bfVal (pa.getD t 0) * bfVal (pb.getD t 0))
      (paQ Aq FB beA i t * pbQ Bq FA FB beB j t)
    linarith [hrow t]
  have hK : 0 ≤ (1 + eps16) ^ 4 - 1 := by
    have := one_le_pow₀ (n := 4) (by linarith [eps16_nonneg] : (1 : ℚ) ≤ 1 + eps16); linarith
  calc |v - uQ Aq Bq FA FB beA beB i j| ≤ _ := he
    _ ≤ peelBound Aq Bq FA FB beA beB i j := by
        unfold peelBound
        gcongr ?_ + ?_ + ?_
        · exact mul_le_mul_of_nonneg_left (by linarith) hK
        · exact Finset.sum_le_sum fun t _ => hrow t

end Bound

end Rounding

/-! ## (c) The terms stay salt-dependent on the pinned U -/

theorem mapM_id_replicate {β : Type} {n : Nat} (hn : n ≠ 0) (y : Option β) :
    (List.replicate n y).mapM id = y.map (List.replicate n) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn
  clear hn
  induction n with
  | zero => cases y <;> rfl
  | succ n ih => rw [List.replicate_succ, List.mapM_cons, ih]; cases y <;> rfl

theorem mapM_const {α β : Type} {l : List α} (hl : l ≠ []) (y : Option β) :
    l.mapM (fun _ => y) = y.map (List.replicate l.length) := by
  induction l with
  | nil => exact absurd rfl hl
  | cons x l ih =>
    rcases l with _ | ⟨x', l⟩
    · cases y <;> rfl
    · rw [List.mapM_cons, ih (List.cons_ne_nil _ _)]; cases y <;> rfl

theorem mapM_congr_mem {α β : Type} {l : List α} {f g : α → Option β} (h : ∀ x ∈ l, f x = g x) :
    l.mapM f = l.mapM g := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [List.mapM_cons, List.mapM_cons, h x List.mem_cons_self,
      ih fun y hy => h y (List.mem_cons_of_mem _ hy)]

/-- `pbRow?` on a constant `B̃·F_A` row and a constant `G`: every column's FMA chain is the same. -/
theorem pbRow_const (be : List Nat) (b gw : Nat) :
    pbRow? be (List.replicate 32 b) (List.replicate 32 (List.replicate 32 gw)) =
      (do
        let acc ← accChain? be (List.replicate 32 gw)
        let h ← rnBf16? (← addWord? acc (negWord b))
        pure (List.replicate 32 h ++ be.map (· ^^^ 0x8000))) := by
  unfold pbRow?
  have hc : ∀ q ∈ List.range Params.pi.r,
      (do
        let acc ← accChain? be (PeelExact.col (List.replicate 32 (List.replicate 32 gw)) q)
        rnBf16? (← addWord? acc (negWord ((List.replicate 32 b).getD q 0)))) =
      (do
        let acc ← accChain? be (List.replicate 32 gw)
        rnBf16? (← addWord? acc (negWord b))) := by
    intro q hq
    have hq' : q < 32 := List.mem_range.1 hq
    rw [PeelExact.col, List.map_replicate, List.getD_replicate _ hq', List.getD_replicate _ hq']
  rw [mapM_congr_mem hc, mapM_const (by simp [Params.pi])]
  rw [List.length_range]
  rcases accChain? be (List.replicate 32 gw) with _ | acc
  · rfl
  · rcases h1 : addWord? acc (negWord b) with _ | w
    · simp only [Option.bind_eq_bind, Option.bind_some, h1, Option.bind_none, Option.map_none]
    · rcases h2 : rnBf16? w with _ | h
      · simp only [Option.bind_eq_bind, Option.bind_some, h1, h2, Option.bind_none, Option.map_none]
      · simp only [Option.bind_eq_bind, Option.bind_some, h1, h2, Option.map_some]
        rfl

/-- **`peelExact` on constant operand and basis rows**: four chains, one `P_A` row, one FMA chain and U's four steps. -/
theorem peelExact_const (a b f g : Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat)
    (i : Fin m) (j : Fin n) :
    peelExact (k := k) (fun _ _ => a) (fun _ _ => b) (fun _ _ => f) (fun _ _ => g) beA beB i j =
      (do
        let c ← chainWord? (rowAtoms (fun _ : Fin k => a) (fun _ => b))
        let af ← chainWord? (rowAtoms (fun _ : Fin k => a) (fun _ => g))
        let bf ← chainWord? (rowAtoms (fun _ : Fin k => b) (fun _ => f))
        let gw ← chainWord? (rowAtoms (fun _ : Fin k => g) (fun _ => f))
        let pa ← paRow? (List.ofFn (beA i)) (List.replicate 32 af)
        let pb ← pbRow? (List.ofFn (beB j)) (List.replicate 32 bf) (List.replicate 32 (List.replicate 32 gw))
        uWord? c pa pb) := by
  have hT : ∀ c : Code, (fun _ _ => c : Matrix (Fin k) (Fin 32) Code)ᵀ = fun _ _ => c := fun _ => rfl
  unfold peelExact
  simp only [hT, List.ofFn_const, mapM_id_replicate (show (32 : Nat) ≠ 0 by norm_num), Option.bind_eq_bind,
    Option.bind_map, Function.comp_def]

/-- The witness noise (`m = n = 1`, `k = r = 32`), as codes: `E_A`'s row is `(a, 1, …, 1)`, `E_B`'s is `(b, 1, …, 1)`,
and every `F` entry is `1`, each `e4m3` of its value. -/
def wNoise (a b : ℚ) : Noise 1 1 32 32 :=
  ⟨fun _ t => e4m3 (if t.val = 0 then a else 1), fun _ _ => e4m3 1, fun _ t => e4m3 (if t.val = 0 then b else 1),
    fun _ _ => e4m3 1⟩

/-- v1's served noise words: each noise line `E′ = lineCodes β e` as a BF16 word, `RN_bf16` of its FP32 word (exact, as
`E′` is an E4M3 value). -/
def beV1 {p : Nat} (β : Fin p → ℚ) (E : Matrix (Fin p) (Fin 32) Code) : Matrix (Fin p) (Fin 32) Nat :=
  fun i t => (rnBf16? (f32Word (codeVal (castLines β E i t)))).getD 0

/-- The witness's `A′`: `x = 0` with the floor's `β = 1` (`α = 1`), formed by the atom-mix. -/
def wA (z : Noise 1 1 32 32) : Matrix (Fin 1) (Fin 32) Code := FormChain.atomMix.quantA (fun _ => 1) (fun _ => 1) 0 z

/-- The witness's `B̃`: `B = 0` with `B`'s floor `β = 1/3` (`α = 1`), formed by the atom-mix. -/
def wB (z : Noise 1 1 32 32) : Matrix (Fin 1) (Fin 32) Code :=
  FormChain.atomMix.quantB (fun _ => 1) (fun _ => 1 / 3) 0 z

/-- The served U's entry at the witness under the salt whose noise is `z`: `peelExactVal` on the atom-mix operands, the
bases' codes and v1's noise words. -/
def wServed (z : Noise 1 1 32 32) : Option ℚ :=
  peelExactVal (wA z) (wB z) (fun l t => z.FA l t) (fun l t => z.FB l t)
    (beV1 (fun _ => 1) z.EA) (beV1 (fun _ => 1 / 3) z.EB) 0 0

theorem wServed_eq (a b : ℚ) (cA cB : Code)
    (hA : FormChain.atomMix.elem 1 0 1 ((wNoise a b).EA 0) (fun _ => e4m3 1) = cA)
    (hB : FormChain.atomMix.elem 1 0 (1 / 3) ((wNoise a b).EB 0) (fun _ => e4m3 1) = cB) :
    wServed (wNoise a b) =
      (peelExact (k := 32) (fun _ _ => cA) (fun _ _ => cB) (fun _ _ => e4m3 1) (fun _ _ => e4m3 1)
        (beV1 (fun _ => 1) (wNoise a b).EA) (beV1 (fun _ => 1 / 3) (wNoise a b).EB) 0 0).bind wordQ := by
  have hA' : wA (wNoise a b) = fun _ _ => cA := by rw [← hA]; rfl
  have hB' : wB (wNoise a b) = fun _ _ => cB := by rw [← hB]; rfl
  unfold wServed peelExactVal
  rw [hA', hB']
  rfl

set_option maxHeartbeats 0 in
/-- **(c) on the served U**: at `Peel.lean`'s degenerate inputs (`x = 0` with `β = 1`, `B = 0` with `β = 1/3`), two
salts give different entries of `peelExactVal`: `0` under the all-ones noise and `10` when `E_A`'s row starts with `2`
and `E_B`'s with `−1`. -/
theorem served_live : wServed (wNoise 1 1) = some 0 ∧ wServed (wNoise 2 (-1)) = some 10 := by
  constructor
  · rw [wServed_eq 1 1 (e4m3 32) (e4m3 11) (by decide +kernel) (by decide +kernel), peelExact_const]
    decide +kernel
  · rw [wServed_eq 2 (-1) (e4m3 32) (e4m3 10) (by decide +kernel) (by decide +kernel), peelExact_const]
    decide +kernel

/-- The two salts' served entries differ. -/
theorem served_salt_dependent : wServed (wNoise 1 1) ≠ wServed (wNoise 2 (-1)) := by
  rw [served_live.1, served_live.2]
  decide

end Pouw.PearlC.PeelExactProofs
