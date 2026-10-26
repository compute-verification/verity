import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Proofs.Accounting.Defs.Actual.WipePool
import Proofs.Core.Game.Prob

/-!
# The memory cap's lifetime pool, as lemmas over an abstract run tree

Step 7 of the memory-cap restatement ("One delta"), by Ville's inequality over a run tree of wipes
(`Accounting.WipePlay`), each fixing its un-held set `U` and so its fraction `x_t = |U| / |B|` before its fresh coins:

* `WipePool`: on a tree meeting `UnheldFails`, `Pr[every wipe passes ∧ Σ_t x_t ≥ H] ≤ exp(−k H)`.
* `WipePoolDelta`: at `H = ln(1/δ)/k`, at most `δ`.

Neither bound depends on the number of wipes: the tree is finite, of any depth. At `k = 1` each wipe is one read with
its own `U`, which gives the pool per read, adaptive within a wipe as well.

They are lemmas, not guarantees. `U` is a label on the tree, and `UnheldFails` (a passing draw challenges no block of
`U`) holds of every tree once each `U` is empty, so it is no physical premise. The pool's physical premise is
memory-accounting's restatement of it on the challenger's verdicts (MEM-55), which uses these lemmas at `k = 1`. The
lemmas count only what the wipes see: a block dropped after one wipe and rebuilt before the next is in no `U`.

The proof is finite optional stopping: from an un-held total `a`, the probability that every wipe passes with the total
at least `H` is at most `exp(k (a − H))` (`WipePlay.prEv_le_exp`), by induction on the tree. At a wipe the draws that
pass challenge no un-held block, which happens with probability `(1 − x)^k ≤ exp(−k x)` (`WipePlay.prCoin_miss_le`),
and each continuation is at most `exp(k (a + x − H))`, so the wipe is at most
`exp(−k x) · exp(k (a + x − H)) = exp(k (a − H))`.

The honest wipe (`WipePlay.oneWipe`), which passes exactly when no draw challenges an un-held block, meets
`UnheldFails` (`oneWipe_sound`) and has the event at `H = x` with probability `((|B| − |U|) / |B|)^k`
(`oneWipe_prEv`), which is positive when some block is held.
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
        · rw [ite_eq_left hD, one_mul]
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
              · rw [ite_eq_left hp]
                exact prEv_le_exp H (next ω) _ (hnext ω)
              · rw [ite_eq_right hp]
                exact zero_le
            · show (if pass ω = true then (next ω).prEv (fun x => H ≤ x) (a + unheld U) else 0) = 0
              rw [ite_eq_right fun hp => hω (hU ω hp)]
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
    (oneWipe (k := k) B U).UnheldFails := by
  show (∀ ω : Fin k → B, decide (∀ i, ω i ∉ U) = true → ∀ i, ω i ∉ U) ∧ ∀ _ : Fin k → B, True
  exact ⟨fun _ hω => of_decide_eq_true hω, fun _ => trivial⟩

/-- The honest wipe reaches the pool's event at `H = x` with probability `((|B| − |U|) / |B|)^k`. -/
theorem oneWipe_prEv (B : Type) [Fintype B] [Nonempty B] [DecidableEq B] (U : Finset B) :
    (oneWipe (k := k) B U).prEv (fun x => unheld U ≤ x) 0 =
      (((Fintype.card B - U.card : ℕ) : ℝ≥0∞) / Fintype.card B) ^ k := by
  rw [← prCoin_miss, oneWipe, prEv_wipe, ← avg_ite_eq_prCoin]
  congr 1
  funext ω
  rw [prEv_stop, zero_add, ite_eq_left le_rfl]
  by_cases h : ∀ i, ω i ∉ U
  · rw [ite_eq_left (decide_eq_true h), ite_eq_left h]
  · rw [ite_eq_right fun hd => h (of_decide_eq_true hd), ite_eq_right h]

end WipePlay

namespace SecurityProofs.Actual

/-- **The lifetime pool.** For any run tree of wipes of `k` rounds (`P : WipePlay k`, any finite number of wipes, each
fixing its un-held set before its coins) on which a passing draw challenges no block of a wipe's un-held set, and any
`H`, the probability that every wipe passes and the un-held fractions sum to at least `H` is at most `exp(−k H)`. -/
theorem WipePool :
    ∀ (k : ℕ) (P : WipePlay k) (H : ℝ), P.UnheldFails →
      P.prEv (fun x => H ≤ x) 0 ≤ ENNReal.ofReal (Real.exp (-((k : ℝ) * H))) := by
  intro k P H hP
  have key := WipePlay.prEv_le_exp H P 0 hP
  rwa [zero_sub, mul_neg] at key

/-- **The lifetime pool at `δ`.** For `k > 0` rounds a wipe and `δ > 0`, on a run tree meeting `UnheldFails`, the
probability that every wipe passes and the un-held fractions sum to at least `ln(1/δ)/k` is at most `δ`. -/
theorem WipePoolDelta :
    ∀ (k : ℕ) (P : WipePlay k) (δ : ℝ), 0 < k → 0 < δ → P.UnheldFails →
      P.prEv (fun x => Real.log (1 / δ) / k ≤ x) 0 ≤ ENNReal.ofReal δ := by
  intro k P δ hk hδ hP
  refine (WipePool k P (Real.log (1 / δ) / k) hP).trans (le_of_eq ?_)
  rw [mul_div_cancel₀ _ (Nat.cast_ne_zero.2 hk.ne'), one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

end SecurityProofs.Actual

end Accounting
