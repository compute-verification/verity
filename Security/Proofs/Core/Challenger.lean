import Definitions.Core.Guarantees.Challenger
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Prob

/-!
# The sample from a challenger's coin: proof

`SampleAfterRecorded`. The verifier's coin `i` is the challenger's flip `i` (`coin_of_log`), and the set of blocks fixed
before it reads only the flips before `i` (`FlipsAfterRecorded`), so resampling flip `i` leaves the set where it is
while the average over the challenger's flips stays the same (`Game.avg_update`). For each fixed set of at least `K`
blocks, a uniform coin's draw misses it with probability the law's `escape`, at most `miss K`.

Its witnesses: `SampleAfterRecorded_honest`, a log that releases coins around a recorded commitment and meets every
premise, and `SampleAfterRecorded_nontrivial`, a log that flips before it records, where the bound fails.
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

/-- **`SampleAfterRecorded`'s honest lemma.** A challenger that releases coin 0, records a commitment the prover made
after seeing it, then releases coin 1, meets all three premises, with the verifier reading both coins from its log; so
for every law, every set of blocks fixed by the log before coin 1 and every `K`, the bound holds there. -/
theorem SampleAfterRecorded_honest {U : Type} (L : Law U) (bad : Log L.Ω L.Ω → Finset U) (K : Nat) :
    ChallengerLogsCoins (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) ∧
      FlipsAfterRecorded
        (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) ∧
      VerifierReadsLog (fun ω : Fin 2 → L.Ω => [ω 0, ω 1])
        (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) ∧
      prCoin (fun ω : Fin 2 → L.Ω => ∃ c, [ω 0, ω 1][1]? = some c ∧
        K ≤ (bad (Log.before [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)] 1)).card ∧
        Disjoint (L.draw c) (bad (Log.before [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)] 1)))
        ≤ L.miss K := by
  have hlog : ChallengerLogsCoins
      (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) := fun ω => by
    simp [Log.coins, List.ofFn_succ]
  have hflip : FlipsAfterRecorded
      (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) := by
    intro i ω ω' h
    match i with
    | 0 => rfl
    | 1 => simp [Log.before, h 0 (by decide)]
    | i + 2 =>
      simp only [Log.before]
      rw [h 0 (Nat.lt_of_lt_of_le (by decide) (Nat.le_add_left 2 i)),
        h 1 (Nat.lt_of_lt_of_le (by decide) (Nat.le_add_left 2 i))]
  have hread : VerifierReadsLog (fun ω : Fin 2 → L.Ω => [ω 0, ω 1])
      (fun ω : Fin 2 → L.Ω => [Entry.released (ω 0), Entry.recorded (ω 0), Entry.released (ω 1)]) := fun ω => by
    simp [Log.coins]
  exact ⟨hlog, hflip, hread, SampleAfterRecorded L 2 _ _ 1 bad K hlog hflip hread⟩

/-- The law of one fair bit that samples the bit it is. -/
private def bitLaw : Law Bool where
  Ω := Bool
  draw c := {c}

/-- **`SampleAfterRecorded`'s nontrivial lemma: without `FlipsAfterRecorded` the bound fails.** A challenger that flips
its coin before it records the commitment the coin samples (a commitment that names the other bit) meets
`ChallengerLogsCoins` and `VerifierReadsLog` but not `FlipsAfterRecorded`, and the block set that commitment fixes, chosen
after the coin, escapes the sample of one fair bit always, while the law lets a set of one block escape half the time. -/
theorem SampleAfterRecorded_nontrivial :
    ∃ (L : Law Bool) (log : (Fin 1 → L.Ω) → Log Bool L.Ω) (vc : (Fin 1 → L.Ω) → List L.Ω)
      (bad : Log Bool L.Ω → Finset Bool),
      ChallengerLogsCoins log ∧ VerifierReadsLog vc log ∧ ¬ FlipsAfterRecorded log ∧
        L.miss 1 < prCoin (fun ω => ∃ c, (vc ω)[0]? = some c ∧ 1 ≤ (bad ((log ω).before 0)).card ∧
          Disjoint (L.draw c) (bad ((log ω).before 0))) := by
  let log : (Fin 1 → Bool) → Log Bool Bool := fun ω => [Entry.recorded (!ω 0), Entry.released (ω 0)]
  let bad : Log Bool Bool → Finset Bool := fun l => match l with
    | [Entry.recorded b] => {b}
    | _ => ∅
  refine ⟨bitLaw, log, fun ω => [ω 0], bad, fun ω => by simp [log, Log.coins, List.ofFn_succ], fun ω => by
    simp [log, Log.coins], fun h => ?_, ?_⟩
  · have := h 0 (fun _ => true) (fun _ => false) fun j hj => absurd hj (Nat.not_lt_zero _)
    simp [log, Log.before] at this
  · have hall : ∀ ω : Fin 1 → Bool, ∃ c, [ω 0][0]? = some c ∧ 1 ≤ (bad ((log ω).before 0)).card ∧
        Disjoint (bitLaw.draw c) (bad ((log ω).before 0)) := fun ω =>
      ⟨ω 0, rfl, by simp [log, bad, Log.before], by
        simp only [log, bad, Log.before, bitLaw]
        generalize ω 0 = b
        cases b <;> decide⟩
    have h1 : prCoin (fun ω : Fin 1 → Bool => ∃ c, [ω 0][0]? = some c ∧ 1 ≤ (bad ((log ω).before 0)).card ∧
        Disjoint (bitLaw.draw c) (bad ((log ω).before 0))) = 1 := by
      unfold prCoin
      simp only [hall, ite_true]
      exact avg_const 1
    have hmiss : bitLaw.miss 1 ≤ 2⁻¹ := by
      refine iSup₂_le fun B hB => ?_
      obtain ⟨b, hb⟩ := Finset.card_pos.mp hB
      refine le_trans (prCoin_mono (q := fun c : Bool => c ≠ b) fun c hc hcb => ?_) ?_
      · exact Finset.disjoint_left.mp hc (by simp [bitLaw, hcb]) hb
      · cases b <;> simp [prCoin, avg]
    exact lt_of_le_of_lt hmiss (lt_of_lt_of_eq (ENNReal.inv_lt_one.2 (by norm_num)) h1.symm)

end SecurityProofs

end Verity
