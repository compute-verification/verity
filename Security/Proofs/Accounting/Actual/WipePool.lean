import Models.Accounting.Guarantees.Actual.WipePool
import Proofs.Core.Game.Prob

/-!
# The memory cap's lifetime pool

`Accounting.Guarantees.Actual.WipePool` and `WipePoolDelta`, by Ville's inequality on the run tree, proved by finite
optional stopping: from an un-held total `a`, the probability that every wipe passes with the total at least `H` is at
most `exp(k (a − H))` (`WipePlay.prEv_le_exp`), by induction on the tree. At a wipe the draws that pass challenge no
un-held block, which happens with probability `(1 − x)^k ≤ exp(−k x)` (`WipePlay.prCoin_miss_le`), and each
continuation is at most `exp(k (a + x − H))`, so the wipe is at most `exp(−k x) · exp(k (a + x − H)) = exp(k (a − H))`.

The honest wipe (`WipePlay.oneWipe`), which passes exactly when no draw challenges an un-held block, meets the premise
(`oneWipe_sound`) and has the event at `H = x` with probability `((|B| − |U|) / |B|)^k` (`oneWipe_prEv`), which is
positive when some block is held.
-/

namespace Accounting

open scoped ENNReal
open Verity.Game

namespace WipePlay

variable {k : ℕ}

open Classical in
theorem prEv_stop (E : ℝ → Prop) (a : ℝ) : (WipePlay.stop : WipePlay k).prEv E a = if E a then 1 else 0 := rfl

theorem prEv_coin (E : ℝ → Prop) (C : Type) [Fintype C] [Nonempty C] (next : C → WipePlay k) (a : ℝ) :
    (WipePlay.coin C next).prEv E a = avg fun c => (next c).prEv E a := rfl

theorem prEv_wipe (E : ℝ → Prop) (B : Type) [Fintype B] [Nonempty B] (U : Finset B)
    (pass : (Fin k → B) → Bool) (next : (Fin k → B) → WipePlay k) (a : ℝ) :
    (WipePlay.wipe B U pass next).prEv E a =
      avg fun ω => if pass ω = true then (next ω).prEv E (a + unheld U) else 0 := rfl

theorem natFrac_le_exp {a b : ℕ} (hab : a ≤ b) (hb : 0 < b) :
    ((b - a : ℕ) : ℝ≥0∞) / (b : ℝ≥0∞) ≤ ENNReal.ofReal (Real.exp (-((a : ℝ) / b))) := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  rw [← ENNReal.ofReal_natCast (b - a), ← ENNReal.ofReal_natCast b, ← ENNReal.ofReal_div_of_pos hb']
  refine ENNReal.ofReal_le_ofReal ?_
  rw [Nat.cast_sub hab, sub_div, div_self hb'.ne']
  exact Real.one_sub_le_exp_neg _

theorem one_le_ofReal_exp {x : ℝ} (hx : 0 ≤ x) : 1 ≤ ENNReal.ofReal (Real.exp x) := by
  rw [← ENNReal.ofReal_one]
  exact ENNReal.ofReal_le_ofReal (Real.one_le_exp hx)

/-- An average of a function at most `c`, and zero off `D`, is at most `Pr[D] · c`. -/
theorem avg_le_prCoin_mul {C : Type} [Fintype C] (D : C → Prop) (f : C → ℝ≥0∞) (c : ℝ≥0∞)
    (h : ∀ ω, f ω ≤ c) (h0 : ∀ ω, ¬ D ω → f ω = 0) : avg f ≤ prCoin D * c := by
  classical
  calc avg f ≤ avg fun ω => (if D ω then 1 else 0) * c := avg_mono fun ω => by
        by_cases hD : D ω
        · rw [if_pos hD, one_mul]
          exact h ω
        · rw [h0 ω hD]
          exact zero_le
    _ = prCoin D * c := by rw [avg_mul_const, avg_ite_eq_prCoin]

/-- `k` uniform draws on `B` all miss `U` with probability `((|B| − |U|) / |B|)^k`. -/
theorem prCoin_miss {B : Type} [Fintype B] [Nonempty B] (U : Finset B) :
    prCoin (fun ω : Fin k → B => ∀ i, ω i ∉ U) =
      (((Fintype.card B - U.card : ℕ) : ℝ≥0∞) / Fintype.card B) ^ k := by
  classical
  calc prCoin (fun ω : Fin k → B => ∀ i, ω i ∉ U)
      = prCoin (fun ω : Fin k → B => ∀ i ∈ (Finset.univ : Finset (Fin k)), ω i ∉ U) := by
        congr 1
        funext ω
        simp
    _ = ∏ _i : Fin k, prCoin (fun b : B => b ∉ U) :=
        prCoin_pi_forall (Ω := fun _ : Fin k => B) Finset.univ (fun _ b => b ∉ U)
    _ = (((Fintype.card B - U.card : ℕ) : ℝ≥0∞) / Fintype.card B) ^ k := by
        rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin, prCoin_eq_card,
          Finset.filter_notMem_eq_sdiff, Finset.card_univ_sdiff]

/-- `k` uniform draws on `B` all miss `U` with probability at most `exp(−k x)`, `x = |U| / |B|`. -/
theorem prCoin_miss_le {B : Type} [Fintype B] [Nonempty B] (U : Finset B) :
    prCoin (fun ω : Fin k → B => ∀ i, ω i ∉ U) ≤ ENNReal.ofReal (Real.exp (-((k : ℝ) * unheld U))) := by
  rw [prCoin_miss, unheld, ← mul_neg, Real.exp_nat_mul, ENNReal.ofReal_pow (Real.exp_pos _).le]
  exact pow_le_pow_left₀ zero_le (natFrac_le_exp (Finset.card_le_univ U) Fintype.card_pos) k

/-- **Optional stopping on the run tree**: from an un-held total `a`, every wipe passes with the total at least `H`
with probability at most `exp(k (a − H))`. -/
theorem prEv_le_exp (H : ℝ) :
    ∀ (P : WipePlay k) (a : ℝ), P.UnheldFails →
      P.prEv (fun x => H ≤ x) a ≤ ENNReal.ofReal (Real.exp ((k : ℝ) * (a - H)))
  | .stop, a, _ => by
      rw [prEv_stop]
      split_ifs with hE
      · exact one_le_ofReal_exp (mul_nonneg (Nat.cast_nonneg _) (sub_nonneg.2 hE))
      · exact zero_le
  | @WipePlay.coin _ C _ _ next, a, hP => by
      rw [prEv_coin]
      exact (avg_mono fun c => prEv_le_exp H (next c) a (hP c)).trans (avg_const _).le
  | @WipePlay.wipe _ B _ _ U pass next, a, hP => by
      obtain ⟨hU, hnext⟩ := hP
      rw [prEv_wipe]
      calc avg (fun ω => if pass ω = true then (next ω).prEv (fun x => H ≤ x) (a + unheld U) else 0)
          ≤ prCoin (fun ω : Fin k → B => ∀ i, ω i ∉ U) *
              ENNReal.ofReal (Real.exp ((k : ℝ) * (a + unheld U - H))) := by
            refine avg_le_prCoin_mul (fun ω : Fin k → B => ∀ i, ω i ∉ U) _ _ (fun ω => ?_) (fun ω hω => ?_)
            · show (if pass ω = true then (next ω).prEv (fun x => H ≤ x) (a + unheld U) else 0) ≤ _
              by_cases hp : pass ω = true
              · rw [if_pos hp]
                exact prEv_le_exp H (next ω) _ (hnext ω)
              · rw [if_neg hp]
                exact zero_le
            · show (if pass ω = true then (next ω).prEv (fun x => H ≤ x) (a + unheld U) else 0) = 0
              rw [if_neg fun hp => hω (hU ω hp)]
        _ ≤ ENNReal.ofReal (Real.exp (-((k : ℝ) * unheld U))) *
              ENNReal.ofReal (Real.exp ((k : ℝ) * (a + unheld U - H))) :=
            mul_le_mul_left (prCoin_miss_le U) _
        _ = ENNReal.ofReal (Real.exp ((k : ℝ) * (a - H))) := by
            rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
            congr 2
            ring

/-- **The honest wipe** over the store `B` with un-held set `U`: it passes exactly when no draw challenges a block of
`U`, and then the run stops. -/
def oneWipe (B : Type) [Fintype B] [Nonempty B] [DecidableEq B] (U : Finset B) : WipePlay k :=
  .wipe B U (fun ω => decide (∀ i, ω i ∉ U)) fun _ => .stop

theorem oneWipe_sound (B : Type) [Fintype B] [Nonempty B] [DecidableEq B] (U : Finset B) :
    Assumptions.TimedRoundSound (oneWipe (k := k) B U) := by
  show (∀ ω : Fin k → B, decide (∀ i, ω i ∉ U) = true → ∀ i, ω i ∉ U) ∧ ∀ _ : Fin k → B, True
  exact ⟨fun _ hω => of_decide_eq_true hω, fun _ => trivial⟩

/-- The honest wipe reaches the pool's event at `H = x` with probability `((|B| − |U|) / |B|)^k`. -/
theorem oneWipe_prEv (B : Type) [Fintype B] [Nonempty B] [DecidableEq B] (U : Finset B) :
    (oneWipe (k := k) B U).prEv (fun x => unheld U ≤ x) 0 =
      (((Fintype.card B - U.card : ℕ) : ℝ≥0∞) / Fintype.card B) ^ k := by
  rw [← prCoin_miss, oneWipe, prEv_wipe, ← avg_ite_eq_prCoin]
  congr 1
  funext ω
  rw [prEv_stop, zero_add, if_pos le_rfl]
  by_cases h : ∀ i, ω i ∉ U
  · rw [if_pos (decide_eq_true h), if_pos h]
  · rw [if_neg fun hd => h (of_decide_eq_true hd), if_neg h]

end WipePlay

namespace SecurityProofs.Actual

theorem WipePool : Guarantees.Actual.WipePool := by
  intro k P H hP
  have key := WipePlay.prEv_le_exp H P 0 hP
  rwa [zero_sub, mul_neg] at key

theorem WipePoolDelta : Guarantees.Actual.WipePoolDelta := by
  intro k P δ hk hδ hP
  refine (WipePool k P (Real.log (1 / δ) / k) hP).trans (le_of_eq ?_)
  rw [mul_div_cancel₀ _ (Nat.cast_ne_zero.2 hk.ne'), one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

end SecurityProofs.Actual

end Accounting
