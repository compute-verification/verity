/-
Level 3, L3-A (PROTOCOL.md §17.4 stage 1d): the lincheck fold's factorization. The in-block tensor is
`e(i) = lw[i mod 64] · eq(ρ, ⌊i/64⌋)`. A slot of `2^sl` bits at position `q` sees `e(q·2^sl + c) = e_lo(c) · eq(ρ_hi, q)`,
with `e_lo = lw ⊗ eq(ρ_lo)`, `ρ_lo` the first `sl − 6` coordinates and `ρ_hi` the rest. That is why `Stmt.fold` folds
each slot type once on `e_lo` and scales it per slot by `eqTable ρ_hi`.
-/
import FlockLevel3.Assembly
import FlockLevel3.Nodes
import Flock.Statement

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-- `eq(r, b)` over any commutative ring (the soundness model's `eqAt`). -/
def eqAt {R : Type} [CommRing R] {n : ℕ} (r : Fin n → R) (b : ℕ) : R :=
  ∏ j : Fin n, if b.testBit j then r j else 1 + r j

/-- **eq over a split point.** The first `a` coordinates read the low `a` bits, the rest the high bits. -/
theorem eqAt_split {R : Type} [CommRing R] (a b : ℕ) (r : Fin (a + b) → R) (u : ℕ) :
    eqAt r u = eqAt (fun j : Fin a => r (Fin.castAdd b j)) (u % 2 ^ a) *
      eqAt (fun j : Fin b => r (Fin.natAdd a j)) (u / 2 ^ a) := by
  unfold eqAt
  rw [Fin.prod_univ_add]
  congr 1
  · refine Finset.prod_congr rfl fun j _ => ?_
    simp [Nat.testBit_mod_two_pow, j.isLt]
  · refine Finset.prod_congr rfl fun j _ => ?_
    simp [Nat.testBit_div_two_pow, add_comm]

/-- The executable's `eqTable` is `eqAt` on its coordinates. -/
theorem eqTable_eqAt (r : Array F128) (b : ℕ) (hb : b < 2 ^ r.size) :
    GF128.of (eqTable r)[b]! = eqAt (fun j : Fin r.size => GF128.of r[j.val]!) b := by
  rw [getElem!_pos _ b (by rw [eqTable_size]; exact hb), eqTable_get]
  unfold eqAt
  refine Finset.prod_congr rfl fun j _ => ?_
  beta_reduce
  rw [getElem!_pos r j.val j.isLt]; rfl

theorem extract_get! (rho : Array F128) (lo hi j : ℕ) (hj : lo + j < hi) (hhi : hi ≤ rho.size) :
    (rho.extract lo hi)[j]! = rho[lo + j]! := by
  rw [getElem!_pos _ j (by simp; omega), getElem!_pos rho (lo + j) (by omega), Array.getElem_extract]

theorem tensor_get (lw rho : Array F128) (k : ℕ) (hk : k < lw.size * 2 ^ rho.size) :
    (Stmt.tensor lw rho)[k]! = lw[k % lw.size]! * (eqTable rho)[k / lw.size]! := by
  unfold Stmt.tensor
  rw [getElem!_pos _ k (by simp [eqTable_size]; exact hk), Array.getElem_ofFn]

/-- **The fold's factorization.** For a slot of `2^sl` bits (`6 ≤ sl ≤ 6 + |ρ|`) at position `q` and an offset `c`
inside it, the in-block tensor is the slot's local tensor at `c` times `eq(ρ_hi, q)`. -/
theorem slot_factor (lw rho : Array F128) (hlw : lw.size = 64) (sl : ℕ) (hsl : 6 ≤ sl) (hslr : sl - 6 ≤ rho.size)
    (q c : ℕ) (hq : q < 2 ^ (rho.size - (sl - 6))) (hc : c < 2 ^ sl) :
    GF128.of (lw[(q * 2 ^ sl + c) % 64]! * (eqTable rho)[(q * 2 ^ sl + c) / 64]!) =
      GF128.of (Stmt.tensor lw (rho.extract 0 (sl - 6)))[c]! * GF128.of (eqTable (rho.extract (sl - 6) rho.size))[q]! := by
  set a := sl - 6 with ha
  set b := rho.size - a with hb
  have hsl' : sl = a + 6 := by omega
  have hsz : rho.size = a + b := by omega
  have h64 : (2 : ℕ) ^ sl = 2 ^ a * 64 := by rw [hsl', pow_add]; rfl
  -- the address split
  have hmod : (q * 2 ^ sl + c) % 64 = c % 64 := by
    rw [h64, show q * (2 ^ a * 64) = 64 * (q * 2 ^ a) by ring, Nat.mul_add_mod]
  have hdiv : (q * 2 ^ sl + c) / 64 = q * 2 ^ a + c / 64 := by
    rw [h64, show q * (2 ^ a * 64) = 64 * (q * 2 ^ a) by ring, Nat.mul_add_div (by norm_num)]
  have hc64 : c / 64 < 2 ^ a := by rw [Nat.div_lt_iff_lt_mul (by norm_num), ← h64]; exact hc
  have hu : q * 2 ^ a + c / 64 < 2 ^ rho.size := by
    rw [hsz, pow_add]
    calc q * 2 ^ a + c / 64 < q * 2 ^ a + 2 ^ a := by omega
      _ = (q + 1) * 2 ^ a := by ring
      _ ≤ 2 ^ b * 2 ^ a := Nat.mul_le_mul_right _ (by omega)
      _ = 2 ^ a * 2 ^ b := mul_comm _ _
  rw [hmod, hdiv, of_mul, eqTable_eqAt rho _ hu]
  -- split eq(ρ, ·) at a
  have hsplit := eqAt_split a b (fun j : Fin (a + b) => GF128.of rho[j.val]!) (q * 2 ^ a + c / 64)
  have e1 : eqAt (fun j : Fin rho.size => GF128.of rho[j.val]!) (q * 2 ^ a + c / 64) =
      eqAt (fun j : Fin (a + b) => GF128.of rho[j.val]!) (q * 2 ^ a + c / 64) := by
    unfold eqAt; exact Fintype.prod_equiv (finCongr hsz) _ _ (fun j => rfl)
  rw [e1, hsplit]
  have hlo : (q * 2 ^ a + c / 64) % 2 ^ a = c / 64 := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hc64]
  have hhi : (q * 2 ^ a + c / 64) / 2 ^ a = q := by
    rw [Nat.mul_comm, Nat.mul_add_div (Nat.two_pow_pos a), Nat.div_eq_of_lt hc64, add_zero]
  rw [hlo, hhi]
  -- the local tensor at c and eq(ρ_hi, q)
  have hlsz : (rho.extract 0 a).size = a := by simp; omega
  have hhsz : (rho.extract a rho.size).size = b := by simp; omega
  have hct : c < lw.size * 2 ^ (rho.extract 0 a).size := by rw [hlw, hlsz, mul_comm, ← h64]; exact hc
  rw [tensor_get lw _ c hct, hlw, of_mul, eqTable_eqAt _ _ (by rw [hlsz]; exact hc64),
    eqTable_eqAt _ _ (by rw [hhsz]; exact hq)]
  have e2 : eqAt (fun j : Fin a => GF128.of rho[(Fin.castAdd b j : Fin (a + b)).val]!) (c / 64) =
      eqAt (fun j : Fin (rho.extract 0 a).size => GF128.of (rho.extract 0 a)[j.val]!) (c / 64) := by
    unfold eqAt
    refine Fintype.prod_equiv (finCongr hlsz.symm) _ _ (fun j => ?_)
    have hj := j.isLt
    simp only [finCongr_apply, Fin.val_cast, Fin.coe_castAdd]
    rw [extract_get! rho 0 a j.val (by omega) (by omega), zero_add]
  have e3 : eqAt (fun j : Fin b => GF128.of rho[(Fin.natAdd a j : Fin (a + b)).val]!) q =
      eqAt (fun j : Fin (rho.extract a rho.size).size => GF128.of (rho.extract a rho.size)[j.val]!) q := by
    unfold eqAt
    refine Fintype.prod_equiv (finCongr hhsz.symm) _ _ (fun j => ?_)
    have hj := j.isLt
    simp only [finCongr_apply, Fin.val_cast, Fin.coe_natAdd]
    rw [extract_get! rho a rho.size j.val (by omega) le_rfl]
  rw [e2, e3]
  ring

end FlockLevel3
