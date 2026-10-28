import Definitions.Core.Guarantees.Challenger
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Prob

/-!
# The sample from a challenger's coin: proof

`SampleAfterRecorded`. The verifier's coin `i` is the challenger's flip `i` (`coin_of_log`), and the set of units fixed
before it reads only the flips before `i` (`FlipsAfterRecorded`), so resampling flip `i` leaves the set where it is
while the average over the challenger's flips stays the same (`Game.avg_update`). For each fixed set of at least `K`
units, a uniform coin's draw misses it with probability the law's `escape`, at most `miss K`.
-/

namespace Verity

open scoped ENNReal

namespace Game

/-- Resampling one coordinate of a uniform tuple leaves its average unchanged. -/
theorem avg_update {C : Type} [Fintype C] [Nonempty C] {n : Nat} (k : Fin n) (f : (Fin n → C) → ℝ≥0∞) :
    avg (fun ω => avg fun c => f (Function.update ω k c)) = avg f := by
  let e : (Fin n → C) × C ≃ (Fin n → C) × C :=
    { toFun := fun p => (Function.update p.1 k p.2, p.1 k)
      invFun := fun p => (Function.update p.1 k p.2, p.1 k)
      left_inv := fun p => by simp
      right_inv := fun p => by simp }
  have hsum : ∑ p : (Fin n → C) × C, f (Function.update p.1 k p.2) = ∑ p : (Fin n → C) × C, f p.1 :=
    Fintype.sum_equiv e _ _ fun p => rfl
  have hC : (Fintype.card C : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp only [avg, ← Finset.mul_sum]
  congr 1
  rw [← Fintype.sum_prod_type' (fun ω c => f (Function.update ω k c)), hsum,
    Fintype.sum_prod_type' (fun ω (_ : C) => f ω)]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← Finset.mul_sum, ← mul_assoc,
    ENNReal.inv_mul_cancel hC (ENNReal.natCast_ne_top _), one_mul]

end Game

namespace Challenger

open Verity.Assumptions

/-- A coin the verifier reads from a challenger's log at index `i` is the challenger's flip `i`. -/
theorem coin_of_log {R C : Type} {n : Nat} {log : (Fin n → C) → Log R C} {vc : (Fin n → C) → List C}
    (hlog : ChallengerLogsCoins log) (hread : VerifierReadsLog vc log) {ω : Fin n → C} {i : Nat} {c : C}
    (hc : (vc ω)[i]? = some c) : ∃ h : i < n, ω ⟨i, h⟩ = c := by
  rw [hread ω, List.prefix_iff_eq_take.mp (hlog ω), List.getElem?_take, List.getElem?_ofFn] at hc
  split_ifs at hc with h1 h2
  exact ⟨h2, Option.some_injective _ hc⟩

namespace Log

variable {R C : Type}

theorem coins_map_released (l : List C) : coins (l.map (Entry.released (R := R))) = l := by
  induction l with
  | nil => rfl
  | cons c l ih => simp [coins, ih]

theorem before_map_released (l : List C) (i : Nat) :
    before (l.map (Entry.released (R := R))) i = (l.take i).map Entry.released := by
  induction l generalizing i with
  | nil => simp [before]
  | cons c l ih => cases i <;> simp [before, ih]

/-- **A challenger that releases its flips in order, with nothing else in its log, meets both of its premises**, so
they are not contradictory. -/
theorem released_in_order {n : Nat} :
    ChallengerLogsCoins (fun ω : Fin n → C => (List.ofFn ω).map (Entry.released (R := R))) ∧
      FlipsAfterRecorded (fun ω : Fin n → C => (List.ofFn ω).map (Entry.released (R := R))) := by
  refine ⟨fun ω => by rw [coins_map_released], fun i ω ω' h => ?_⟩
  simp only [before_map_released]
  congr 1
  refine List.ext_getElem? fun j => ?_
  simp only [List.getElem?_take, List.getElem?_ofFn]
  split_ifs with h1 h2 <;> first | rfl | simp [h ⟨j, h2⟩ h1]

end Log

end Challenger

namespace SecurityProofs

open Verity.Assumptions Verity.Challenger Game

theorem SampleAfterRecorded : Guarantees.SampleAfterRecorded := by
  intro U R L n log vc i bad K hlog hflip hread
  by_cases hi : i < n
  swap
  · refine le_trans (prCoin_mono (q := fun _ => False) ?_) (by rw [prCoin_false]; exact bot_le)
    rintro ω ⟨c, hc, -⟩
    exact hi (coin_of_log hlog hread hc).1
  set k : Fin n := ⟨i, hi⟩
  set B : (Fin n → L.Ω) → Finset U := fun ω => bad ((log ω).before i)
  have hB : ∀ ω c, B (Function.update ω k c) = B ω := fun ω c =>
    congrArg bad (hflip i _ _ fun j hj => Function.update_of_ne (Fin.ne_of_lt hj) _ _)
  calc prCoin (fun ω => ∃ c, (vc ω)[i]? = some c ∧ K ≤ (bad ((log ω).before i)).card ∧
          Disjoint (L.draw c) (bad ((log ω).before i)))
      ≤ prCoin (fun ω => K ≤ (B ω).card ∧ Disjoint (L.draw (ω k)) (B ω)) := by
        refine prCoin_mono fun ω ⟨c, hc, hK, hd⟩ => ?_
        obtain ⟨_, rfl⟩ := coin_of_log hlog hread hc
        exact ⟨hK, hd⟩
    _ = avg (fun ω => prCoin fun c => K ≤ (B ω).card ∧ Disjoint (L.draw c) (B ω)) := by
        unfold prCoin
        rw [← avg_update k]
        congr 1; funext ω; congr 1; funext c
        beta_reduce
        rw [hB, Function.update_self]
    _ ≤ avg (fun _ : Fin n → L.Ω => L.miss K) := by
        refine avg_mono fun ω => ?_
        by_cases hK : K ≤ (B ω).card
        · exact le_trans (prCoin_mono fun c h => h.2)
            (le_iSup₂ (f := fun (B : Finset U) (_ : K ≤ B.card) => L.escape B) (B ω) hK)
        · refine le_trans (prCoin_mono (q := fun _ => False) fun c h => hK h.1) ?_
          rw [prCoin_false]
          exact bot_le
    _ = L.miss K := avg_const _

end SecurityProofs

end Verity
