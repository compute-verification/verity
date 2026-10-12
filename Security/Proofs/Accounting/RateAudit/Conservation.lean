import Mathlib
import Models.Accounting.Guarantees.RateAuditRemote
import Proofs.Accounting.RateAudit.Interactive

/-!
# Conservation over tiers: the proof

Every ingress step reads one address, of one tier, so the reads summed over the tiers are the run's ingress
(`IChip.sum_tierBits`), and the footprint is at most that. A run with `k` correct receipts falls in one of two events
(`InteractiveDevice.guess_or_le`, `InteractiveDevice.correct_le_relayed`), by how the receipts came:

- with side sources that ignore the oracle, a word never queried with a correct receipt anyway: `2^-h` a word, `N/2^h`
  over the words (`RelayDevice.pr_guess_or_compress`); with any side sources, at least `m` correct receipts never
  queried, relayed in: `RateAuditInteractive`'s `C(N, m)·2^L/2^(h·m)` (`RelayDevice.pr_relay_or_compress`);
- otherwise the chip queried `k` (or `k − m`) words, and what it had besides the oracle (`taken`: on-chip state,
  footprint, link bits and rebuilt words) fell short of the stream and those words by `λ`: `RateAudit`'s compression,
  `C(q, k)·2^-λ`.

No sample is drawn: these are `RateAudit`'s and `RateAuditInteractive`'s arguments on the correct receipts themselves.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le)

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- The reads of every tier sum to the ingress. -/
theorem sum_tierBits {τ : ℕ} (tier : ℕ → Fin τ) (w : IWorld N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    ∑ t, p.tierBits tier t w v = p.ingressBits w v := by
  induction p generalizing v with
  | done a => simp [tierBits, ingressBits]
  | tag x k ih => exact ih _ v
  | ingress a k ih =>
    simp only [tierBits, ingressBits, Finset.sum_add_distrib, ih, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

end IChip

namespace InteractiveDevice

variable {N b h ρ S L q : ℕ}

theorem sum_tierReads {τ : ℕ} (A : InteractiveDevice N b h ρ S L q) (tier : ℕ → Fin τ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : ∑ t, A.tierReads tier t E R O = A.ingress E R O :=
  IChip.sum_tierBits tier _ _ []

theorem footprint_le_sum_tierReads {τ : ℕ} (A : InteractiveDevice N b h ρ S L q) (tier : ℕ → Fin τ) (E : Store N b)
    (R : Bits ρ) (O : Oracle N b h ρ) : A.footprint E R O ≤ ∑ t, A.tierReads tier t E R O := by
  rw [sum_tierReads]
  exact footprint_le_ingress A E R O

theorem toRelay_rebuilds (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (A.toRelay.chip (A.toRelay.state E)).rebuilds (A.toRelay.world E R O) = A.rebuilt E R O := by
  change ((A.chip (A.state E)).toChip []).rebuilds (A.toRelay.world E R O) = _
  rw [IChip.toChip_rebuilds, lift_world]
  rfl

theorem toRelay_correct (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toRelay.correct E R O = A.correct E R O := by
  simp only [RelayDevice.correct, correct, toRelay_tags]

/-- What the relay device had besides the oracle is at most the reads of every tier, the state, the link and the
rebuilt words. -/
theorem taken_le {τ : ℕ} (A : InteractiveDevice N b h ρ S L q) (tier : ℕ → Fin τ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : taken A.toRelay E R O ≤ ∑ t, A.tierReads tier t E R O + S + L + A.rebuilt E R O * b := by
  have h1 := footprint_le_sum_tierReads A tier E R O
  have h2 := (A.toRelay.bounded (A.toRelay.state E) (A.toRelay.world E R O)).1
  unfold taken
  rw [toRelay_footprint, toRelay_rebuilds]
  omega

/-- A correct receipt for a word never queried, or every correct receipt's word queried. -/
theorem guess_or_le (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (∃ j, j ∉ queried A.toRelay E R O ∧ A.toRelay.tags E R O j = O (R, j, E j)) ∨
      (A.correct E R O).card ≤ (queried A.toRelay E R O).card := by
  by_cases hg : ∃ j, j ∉ queried A.toRelay E R O ∧ A.toRelay.tags E R O j = O (R, j, E j)
  · exact Or.inl hg
  · refine Or.inr (Finset.card_le_card (fun j hj => ?_))
    simp only [correct, Finset.mem_filter, Finset.mem_univ, true_and] at hj
    by_contra hq
    exact hg ⟨j, hq, by rw [toRelay_tags]; exact hj⟩

/-- The correct receipts are the relayed ones and the queried ones. -/
theorem correct_le_relayed (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (A.correct E R O).card ≤ (A.toRelay.relayed E R O).card + (queried A.toRelay E R O).card := by
  rw [← toRelay_correct]
  exact Finset.card_le_card_sdiff_add_card

end InteractiveDevice

namespace RelayDevice

variable {N b h ρ S L q : ℕ}

/-- With side sources that ignore the oracle: a guessed receipt, or `k` queried words and too little else, with
probability at most `N/2^h + C(q, k)/2^λ`. -/
theorem pr_guess_or_compress (A : RelayDevice N b h ρ S L q) (hA : ∀ E R O O', A.side E R O = A.side E R O')
    (k lam : ℕ) (hk : 0 < k) :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        (∃ j, j ∉ queried A x.1 x.2.1 x.2.2 ∧ A.tags x.1 x.2.1 x.2.2 j = x.2.2 (x.2.1, j, x.1 j)) ∨
          (k ≤ (queried A x.1 x.2.1 x.2.2).card ∧ taken A x.1 x.2.1 x.2.2 + lam < k * b + ρ))
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam := by
  classical
  refine (pr_or_le _ _).trans (add_le_add ?_ ?_)
  · refine (pr_exists_le _).trans ?_
    have hj : ∀ j : Fin N, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        j ∉ queried A x.1 x.2.1 x.2.2 ∧ A.tags x.1 x.2.1 x.2.2 j = x.2.2 (x.2.1, j, x.1 j)) ≤
        ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun j =>
      pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
        rw [one_mul, Fintype.card_prod, Fintype.card_prod]
        refine card_filter_prod_le _ _ (Fintype.card (Bits ρ) * Fintype.card (Oracle N b h ρ)) (fun E => ?_)
        refine card_filter_prod_le _ _ _ (fun R => ?_)
        exact card_guess_le A E R j (hA E R))
    refine (Finset.sum_le_sum (fun j _ => hj j)).trans (le_of_eq ?_)
    simp [div_eq_mul_inv]
  · refine (pr_le_of_card _ (2 ^ lam) (q.choose k) (by positivity) (by
      rw [card_filter_equiv (Equiv.prodAssoc (Store N b) (Bits ρ) (Oracle N b h ρ)).symm]
      refine (card_filter_prod_le' _ _ (q.choose k * Fintype.card (Store N b × Bits ρ)) (fun O => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod]; ring))
      exact card_compress_le A O k lam hk)).trans (le_of_eq ?_)
    push_cast
    rfl

/-- With any side sources: `m` relayed receipts, or `k − m` queried words and too little else, with probability at
most `C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`. -/
theorem pr_relay_or_compress (A : RelayDevice N b h ρ S L q) (k m lam : ℕ) (hk : 0 < k - m) :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        m ≤ (A.relayed x.1 x.2.1 x.2.2).card ∨
          (k - m ≤ (queried A x.1 x.2.1 x.2.2).card ∧ taken A x.1 x.2.1 x.2.2 + lam < (k - m) * b + ρ))
      ≤ (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  classical
  refine (pr_or_le _ _).trans (add_le_add ?_ ?_)
  · refine (pr_le_of_card _ (2 ^ (h * m)) (N.choose m * 2 ^ L) (by positivity) (by
      rw [Fintype.card_prod, Fintype.card_prod]
      refine (card_filter_prod_le _ _
        (Fintype.card (Bits ρ) * (N.choose m * 2 ^ L * Fintype.card (Oracle N b h ρ))) (fun E => ?_)).trans
        (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun R => ?_)
      exact card_relay_le A E R m)).trans (le_of_eq ?_)
    push_cast
    rfl
  · refine (pr_le_of_card _ (2 ^ lam) (q.choose (k - m)) (by positivity) (by
      rw [card_filter_equiv (Equiv.prodAssoc (Store N b) (Bits ρ) (Oracle N b h ρ)).symm]
      refine (card_filter_prod_le' _ _ (q.choose (k - m) * Fintype.card (Store N b × Bits ρ)) (fun O => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod]; ring))
      exact card_compress_le A O (k - m) lam hk)).trans (le_of_eq ?_)
    push_cast
    rfl

end RelayDevice

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (pr_le_of_card pr_mono)

/-- **Conservation over tiers, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateConservationRO`). -/
theorem RateConservationRO : Accounting.Guarantees.RateConservationRO := by
  classical
  intro N b h ρ S L q k lam τ tier A hA
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp at hx), Finset.card_empty, zero_mul, zero_mul]
  refine (pr_mono (fun x hx => ?_)).trans
    (A.toRelay.pr_guess_or_compress (InteractiveDevice.toRelay_side A hA) k lam hk)
  obtain ⟨hc, hlt⟩ := hx
  rcases InteractiveDevice.guess_or_le A x.1 x.2.1 x.2.2 with hg | hq
  · exact Or.inl hg
  · have := InteractiveDevice.taken_le A tier x.1 x.2.1 x.2.2
    exact Or.inr ⟨hc.trans hq, by omega⟩

/-- **Conservation over tiers, with any side sources** (`Accounting.Guarantees.RateConservation`). -/
theorem RateConservation : Accounting.Guarantees.RateConservation := by
  classical
  intro N b h ρ S L q k m lam τ tier A
  rcases Nat.eq_zero_or_pos (k - m) with hkm | hk
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [hkm] at hx), Finset.card_empty, zero_mul, zero_mul]
  refine (pr_mono (fun x hx => ?_)).trans (A.toRelay.pr_relay_or_compress k m lam hk)
  obtain ⟨hc, hlt⟩ := hx
  by_cases hm : m ≤ (A.toRelay.relayed x.1 x.2.1 x.2.2).card
  · exact Or.inl hm
  · have h1 := InteractiveDevice.correct_le_relayed A x.1 x.2.1 x.2.2
    have h2 := InteractiveDevice.taken_le A tier x.1 x.2.1 x.2.2
    exact Or.inr ⟨by omega, by omega⟩

/-- The capacities bound the footprint. -/
theorem footprint_mul_lt_of_capped {N b h ρ S L q τ cEnc fl : ℕ} {tier : ℕ → Fin τ} {cap : Fin τ → ℕ}
    (A : InteractiveDevice N b h ρ S L q) (hcap : A.TierCapped tier cap) (hfl : (∑ t, cap t) * cEnc < fl)
    (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : A.toRelay.footprint E R O * cEnc < fl := by
  rw [InteractiveDevice.toRelay_footprint]
  refine lt_of_le_of_lt (Nat.mul_le_mul_right cEnc ?_) hfl
  exact (InteractiveDevice.footprint_le_sum_tierReads A tier E R O).trans
    (Finset.sum_le_sum (fun t _ => hcap (A.state E) (A.world E R O) t))

/-- **Correct receipts are bounded by the tiers' capacity, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateConservationCapRO`). -/
theorem RateConservationCapRO : Accounting.Guarantees.RateConservationCapRO := by
  classical
  intro N b h ρ S L q k F cTag cEnc lam τ tier cap A hA hcap hfl
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp [floor] at hfl
  refine (pr_mono (fun x hx => ?_)).trans
    (A.toRelay.pr_guess_or_compress (InteractiveDevice.toRelay_side A hA) k lam hk)
  obtain ⟨hc, hF⟩ := hx
  rcases InteractiveDevice.guess_or_le A x.1 x.2.1 x.2.2 with hg | hq
  · exact Or.inl hg
  · have hk' := hc.trans hq
    refine Or.inr ⟨hk', taken_lt_of_floor A.toRelay x.1 x.2.1 x.2.2 hk' ?_
      (footprint_mul_lt_of_capped A hcap hfl x.1 x.2.1 x.2.2)⟩
    rwa [InteractiveDevice.toRelay_compute]

/-- **Correct receipts are bounded by the tiers' capacity, with any side sources**
(`Accounting.Guarantees.RateConservationCap`). -/
theorem RateConservationCap : Accounting.Guarantees.RateConservationCap := by
  classical
  intro N b h ρ S L q k m F cTag cEnc lam τ tier cap A hcap hfl
  rcases Nat.eq_zero_or_pos (k - m) with hkm | hk
  · simp [floor, hkm] at hfl
  refine (pr_mono (fun x hx => ?_)).trans (A.toRelay.pr_relay_or_compress k m lam hk)
  obtain ⟨hc, hF⟩ := hx
  by_cases hm : m ≤ (A.toRelay.relayed x.1 x.2.1 x.2.2).card
  · exact Or.inl hm
  · have h1 := InteractiveDevice.correct_le_relayed A x.1 x.2.1 x.2.2
    have hq : k - m ≤ (queried A.toRelay x.1 x.2.1 x.2.2).card := by omega
    refine Or.inr ⟨hq, taken_lt_of_floor A.toRelay x.1 x.2.1 x.2.2 hq ?_
      (footprint_mul_lt_of_capped A hcap hfl x.1 x.2.1 x.2.2)⟩
    rwa [InteractiveDevice.toRelay_compute]

/-- **Freed bandwidth is spent on another tier or rebuilt** (`Accounting.Guarantees.RateDisplacement`). -/
theorem RateDisplacement : Accounting.Guarantees.RateDisplacement := by
  intro N b h ρ S L q k D lam τ tier t₀ A hA
  refine (pr_mono (fun x hx => ?_)).trans (RateConservationRO N b h ρ S L q k lam τ tier A hA)
  obtain ⟨hc, h1, h2⟩ := hx
  have e := Finset.add_sum_erase Finset.univ (fun t => A.tierReads tier t x.1 x.2.1 x.2.2) (Finset.mem_univ t₀)
  exact ⟨hc, by omega⟩

end Accounting.SecurityProofs
