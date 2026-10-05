import Mathlib.LinearAlgebra.Lagrange
import FlockSoundness.Protocol.Rounds
import FlockSoundness.Protocol.Defs
import FlockSoundness.Protocol.Game.Union

/-!
# The lincheck phase (PROTOCOL.md §10)

For one codeword near the committed table, whose witness `z` makes the `c` claim and the region
claims true: the doomed zerocheck says the `a` claim, the `b` claim or the pin fails. Then the
lincheck ends with a true `ab` claim for `z` with probability at most `(2 + 2(k_log − 6) + 63)/|F|`
(`value_lincheck_le`):

* **α, β (L1, L2).** The lincheck starts from `α·v_a + v_b + β`; the truth is
  `α·a(z) + b(z) + β·pin(z)`, which is `Σ_c comb[c]·h[c]` for the witness's block-folded columns
  `h` (`combSum_eq`). The difference is a nonzero polynomial of degree 1 in `(α, β)`: one coin each.
* **The rounds (L3).** Each keeps a false claim false except with probability `2/|F|`
  (`Rounds.lincheckRound_sound`), about the vectors folded so far (`value_lincheckRounds_le`).
* **The final check and `r'` (L4, L5).** A passing check forces `z_partial` off the folded columns;
  the `ab` claim's value `Σ_s L_s(r')·z_partial[s]` then misses the truth `Σ_s L_s(r')·h'[s]`
  (`abEval_eq`) except with probability `63/|F|`, a nonzero polynomial of degree below 64 through
  the skip nodes (`prCoin_lagS_le`).

`lincheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Equality weights and sums over split indices -/

/-- Splitting an equality weight at coordinate `a`: low bits `w < 2^a`, high bits `o`. -/
theorem eqAt_split_at {R : Type} [CommRing R] {N : ℕ} (a b : ℕ) (hN : a + b = N) (r : Fin N → R)
    (w o : ℕ) (hw : w < 2 ^ a) :
    eqAt r (w + 2 ^ a * o) = eqAt (fun j : Fin a => r ⟨j.val, by omega⟩) w *
      eqAt (fun j : Fin b => r ⟨a + j.val, by omega⟩) o := by
  subst hN
  unfold eqAt
  rw [Fin.prod_univ_add]
  congr 1
  · refine Finset.prod_congr rfl fun j _ => ?_
    rw [add_comm w, Nat.testBit_two_pow_mul_add o hw]
    simp only [Fin.val_castAdd, j.isLt, if_true]
    rfl
  · refine Finset.prod_congr rfl fun j _ => ?_
    rw [add_comm w, Nat.testBit_two_pow_mul_add o hw]
    simp only [Fin.val_natAdd, show ¬ a + j.val < a by omega, if_false,
      show a + j.val - a = j.val by omega]
    rfl

/-- Sums over `Fin (m·n)` as double sums, the index being `j + n·i`. -/
theorem sum_fin_mul {M : Type} [AddCommMonoid M] {m n : ℕ} (f : Fin (m * n) → M) :
    ∑ x, f x = ∑ i : Fin m, ∑ j : Fin n, f (finProdFinEquiv (i, j)) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]

/-- Addresses split into blocks: `a = o·2^k_log + i`. -/
theorem sum_addr {M : Type} [AddCommMonoid M] (S : Statement) (f : Fin (2 ^ S.m) → M) :
    ∑ a, f a = ∑ o : Fin (2 ^ S.nbl), ∑ i : Fin (2 ^ S.kLog), f (S.addr o i) := by
  have h : 2 ^ S.nbl * 2 ^ S.kLog = 2 ^ S.m := by rw [← pow_add, Nat.sub_add_cancel S.hk]
  rw [← (finCongr h).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl fun i _ => ?_
  congr 1
  ext
  simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, Statement.addr]
  ring

/-! ## Two coin bounds -/

/-! ## The rounds -/

/-! ## The witness through the lincheck's eyes -/

/-- The witness's block-folded columns: `h[c] = Σ_o eq(ρ_out, o)·ω[o·2^k_log + c]`. -/
def hcol (S : Statement) (ρOut : Fin S.nbl → F) (ω : Witness S.m) : ℕ → F := fun c =>
  if hc : c < 2 ^ S.kLog then
    ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((ω (S.addr o ⟨c, hc⟩)).val : F)
  else 0

theorem two_pow_kLog (S : Statement) : 2 ^ S.kLog = 2 ^ (S.kLog - 6) * 64 := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel S.hk6]

/-- A claim at a point `p ‖ ρ_out` (inner bits, then the block bits) evaluates through the
block-folded columns. -/
theorem claim_eval_split (S : Statement) (c : Claim F S.m) (p : Fin (S.kLog - 6) → F)
    (ρOut : Fin S.nbl → F)
    (hp : ∀ j : Fin (S.kLog - 6), c.x ⟨j.val, by have := S.hk; omega⟩ = p j)
    (hout : ∀ j : Fin S.nbl,
      c.x ⟨S.kLog - 6 + j.val, by
        have := S.hk; have := S.hk6; have hj : j.val < S.m - S.kLog := j.isLt; omega⟩ = ρOut j)
    (ω : Witness S.m) :
    c.eval ω = ∑ i : Fin (2 ^ S.kLog), c.skip ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
      eqAt p (i.val / 64) * hcol S ρOut ω i.val := by
  have hk := S.hk
  have hk6 := S.hk6
  have hnbl : S.nbl = S.m - S.kLog := rfl
  have hpow := two_pow_kLog S
  unfold Claim.eval
  rw [sum_addr S, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [hcol, dif_pos i.isLt, Finset.mul_sum]
  refine Finset.sum_congr rfl fun o _ => ?_
  have hval : (S.addr o i).val = 64 * (2 ^ (S.kLog - 6) * o.val) + i.val := by
    simp only [Statement.addr, hpow]; ring
  have hmod : (S.addr o i).val % 64 = i.val % 64 := by
    rw [hval, Nat.mul_add_mod]
  have hdiv : (S.addr o i).val / 64 = i.val / 64 + 2 ^ (S.kLog - 6) * o.val := by
    rw [hval, Nat.mul_add_div (by norm_num)]; ring
  have hi : i.val / 64 < 2 ^ (S.kLog - 6) := by
    have := lt_of_lt_of_eq i.isLt hpow; omega
  have hskip : c.skip ⟨(S.addr o i).val % 64, Nat.mod_lt _ (by norm_num)⟩ =
      c.skip ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ := congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv, eqAt_split_at (S.kLog - 6) S.nbl (by omega) c.x _ _ hi]
  simp only [hp, hout]
  ring

/-- `ρ_in`: the zerocheck's challenges for the inner bits (as the lincheck reads them). -/
def ρIn (S : Statement) (zc : ZcOut F S.m) : Fin (S.kLog - 6) → F :=
  fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩

/-! ## The phase -/

end FlockSoundness
