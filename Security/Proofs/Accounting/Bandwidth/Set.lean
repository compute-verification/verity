import Mathlib
import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.Bandwidth.Honest

/-!
# The bandwidth audit v0 over a co-audited set: the proof

The set is one `Device` over its members' stores, so `AuditRO`'s lemmas apply to it as they are; only the sample
differs, `s` blocks of each member's store. An accepted run with low ingress falls in one of three events, by the
blocks whose contents the set queried the oracle at (`queried`):

- at least `d·k` of them, with low ingress: `AuditRO`'s compression at `d·k` blocks (`card_compress_le`);
- otherwise some member has fewer than `k` of its blocks queried (`exists_member_lt`), and either its sample fell
  among them, `((k − 1)/N)^s` (`card_sample_member`), or one of its sampled blocks has the right tag without a query,
  `2^-h` each (`card_guess_le`). Summed over the members: `d·(((k − 1)/N)^s + s/2^h)`.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace AuditSet

variable {d N : ℕ}

/-- Member `i`'s blocks among a set of the joint store's blocks. -/
def memberBlocks (K : Finset (Fin (d * N))) (i : Fin d) : Finset (Fin N) :=
  Finset.univ.filter (fun j => member i j ∈ K)

/-- Fewer than `d·k` blocks leave some member with fewer than `k` of its own. -/
theorem exists_member_lt (K : Finset (Fin (d * N))) (k : ℕ) (hK : K.card < d * k) :
    ∃ i, (memberBlocks K i).card < k := by
  classical
  by_contra hall
  push Not at hall
  refine absurd hK (not_lt.2 ?_)
  calc d * k = ∑ _i : Fin d, k := by simp
    _ ≤ ∑ i, (memberBlocks K i).card := Finset.sum_le_sum (fun i _ => hall i)
    _ = (Finset.univ.filter (fun p : Fin d × Fin N => member p.1 p.2 ∈ K)).card := by
        rw [Finset.card_filter, Fintype.sum_prod_type]
        refine Finset.sum_congr rfl (fun i _ => ?_)
        rw [memberBlocks, Finset.card_filter]
    _ ≤ K.card := by
        refine Finset.card_le_card_of_injOn (fun p => member p.1 p.2) (fun p hp => ?_) ?_
        · simpa using hp
        · intro p _ p' _ hpp
          exact finProdFinEquiv.injective hpp

/-- The samples whose member-`i` part falls in `K`: a `(|K|/N)^s` fraction. -/
theorem card_sample_member (i : Fin d) (K : Finset (Fin N)) (s : ℕ) :
    (Finset.univ.filter (fun I : Fin d → Fin s → Fin N => ∀ t, I i t ∈ K)).card * N ^ s =
      K.card ^ s * Fintype.card (Fin d → Fin s → Fin N) := by
  classical
  have hset : Finset.univ.filter (fun I : Fin d → Fin s → Fin N => ∀ t, I i t ∈ K) =
      Fintype.piFinset (Function.update (fun _ => (Finset.univ : Finset (Fin s → Fin N))) i
        (Fintype.piFinset (fun _ => K))) := by
    ext I
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
    constructor
    · intro hI i'
      by_cases h : i' = i
      · subst h
        simpa [Fintype.mem_piFinset] using hI
      · simp [Function.update_of_ne h]
    · intro hI t
      have := hI i
      simp only [Function.update_self, Fintype.mem_piFinset] at this
      exact this t
  have hd : (Finset.univ \ {i} : Finset (Fin d)).card + 1 = d := by
    rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_singleton, Finset.card_univ,
      Fintype.card_fin]
    have := i.2
    omega
  have hrest : ∏ x ∈ Finset.univ \ {i},
      (Function.update (fun _ => (Finset.univ : Finset (Fin s → Fin N))) i (Fintype.piFinset (fun _ => K)) x).card =
      (N ^ s) ^ (Finset.univ \ {i} : Finset (Fin d)).card := by
    rw [Finset.prod_congr rfl (g := fun _ => N ^ s) (fun x hx => ?_), Finset.prod_const]
    have hxi : x ≠ i := by simpa using hx
    simp [Function.update_of_ne hxi]
  rw [hset, Fintype.card_piFinset, Finset.prod_eq_mul_prod_sdiff_singleton_of_mem (Finset.mem_univ i), hrest,
    Function.update_self, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ]
  simp only [Fintype.card_fun, Fintype.card_fin]
  calc K.card ^ s * (N ^ s) ^ (Finset.univ \ {i}).card * N ^ s
      = K.card ^ s * (N ^ s) ^ ((Finset.univ \ {i} : Finset (Fin d)).card + 1) := by ring
    _ = K.card ^ s * (N ^ s) ^ d := by rw [hd]

end AuditSet

end Bandwidth

namespace Bandwidth.SecurityProofs

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth.AuditRO Bandwidth.AuditSet

/-- **The bandwidth audit v0 over a co-audited set** (`Bandwidth.Guarantees.AuditSet`). -/
theorem AuditSet : Bandwidth.Guarantees.AuditSet := by
  classical
  intro d N b h S c G q s k lam A
  have hsplit : ∀ x : Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N),
      SetAccepts A x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < d * (k * b) →
      (∃ i : Fin d, ((memberBlocks (queried A x.1 x.2.1) i).card < k ∧
            ∀ t, x.2.2 i t ∈ memberBlocks (queried A x.1 x.2.1) i) ∨
          ∃ t, member i (x.2.2 i t) ∉ queried A x.1 x.2.1 ∧
            A.tags x.1 x.2.1 (member i (x.2.2 i t)) = x.2.1 (member i (x.2.2 i t), x.1 (member i (x.2.2 i t)))) ∨
      (d * k ≤ (queried A x.1 x.2.1).card ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < d * k * b) := by
    rintro ⟨E, O, I⟩ ⟨hacc, hlow⟩
    dsimp only at hacc hlow ⊢
    by_cases hk : d * k ≤ (queried A E O).card
    · exact Or.inr ⟨hk, by rwa [mul_assoc]⟩
    · obtain ⟨i, hi⟩ := exists_member_lt (queried A E O) k (by omega)
      refine Or.inl ⟨i, ?_⟩
      by_cases hall : ∀ t, I i t ∈ memberBlocks (queried A E O) i
      · exact Or.inl ⟨hi, hall⟩
      · push Not at hall
        obtain ⟨t, ht⟩ := hall
        exact Or.inr ⟨t, by simpa [memberBlocks] using ht, hacc i t⟩
  have hΩ : Fintype.card (Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N)) =
      Fintype.card (Store (d * N) b) *
        (Fintype.card (Oracle (d * N) b h) * Fintype.card (Fin d → Fin s → Fin N)) := by
    simp only [Fintype.card_prod]
  by_cases hI : Fintype.card (Fin d → Fin s → Fin N) = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hI]; simp)]
    exact bot_le
  have hNs : ∀ _i : Fin d, N ^ s ≠ 0 := fun i hN => hI (by
    simp only [Fintype.card_fun, Fintype.card_fin, hN]
    exact zero_pow (Nat.pos_of_ne_zero (by rintro rfl; exact i.elim0)).ne')
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ?_ ?_)
  · refine (pr_exists_le _).trans ?_
    have hmem : ∀ i : Fin d, pr (fun x : Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N) =>
        ((memberBlocks (queried A x.1 x.2.1) i).card < k ∧
            ∀ t, x.2.2 i t ∈ memberBlocks (queried A x.1 x.2.1) i) ∨
          ∃ t, member i (x.2.2 i t) ∉ queried A x.1 x.2.1 ∧
            A.tags x.1 x.2.1 (member i (x.2.2 i t)) = x.2.1 (member i (x.2.2 i t), x.1 (member i (x.2.2 i t)))) ≤
        (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h := fun i => by
      refine (pr_or_le _ _).trans (add_le_add ?_ ?_)
      · -- member `i`'s sample misses the blocks never queried
        have hc := pr_le_of_card (fun x : Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N) =>
            (memberBlocks (queried A x.1 x.2.1) i).card < k ∧ ∀ t, x.2.2 i t ∈ memberBlocks (queried A x.1 x.2.1) i)
            (N ^ s) ((k - 1) ^ s) (hNs i) (by
          rw [hΩ]
          refine (card_filter_prod_le _ _
            (Fintype.card (Oracle (d * N) b h) * ((k - 1) ^ s * Fintype.card (Fin d → Fin s → Fin N)))
            (fun E => ?_)).trans (le_of_eq (by ring))
          refine card_filter_prod_le _ _ _ (fun O => ?_)
          by_cases hk : (memberBlocks (queried A E O) i).card < k
          · simp only [hk, true_and]
            rw [card_sample_member]
            exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
          · simp [hk])
        refine hc.trans (le_of_eq ?_)
        rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
      · -- a guessed tag among member `i`'s samples
        refine (pr_exists_le _).trans ?_
        have ht : ∀ t : Fin s, pr (fun x : Store (d * N) b × Oracle (d * N) b h × (Fin d → Fin s → Fin N) =>
            member i (x.2.2 i t) ∉ queried A x.1 x.2.1 ∧
              A.tags x.1 x.2.1 (member i (x.2.2 i t)) = x.2.1 (member i (x.2.2 i t), x.1 (member i (x.2.2 i t)))) ≤
            ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun t =>
          pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
            rw [hΩ, one_mul]
            refine (card_filter_prod_le _ _
              (Fintype.card (Oracle (d * N) b h) * Fintype.card (Fin d → Fin s → Fin N))
              (fun E => ?_)).trans (le_of_eq (by simp))
            refine card_filter_prod_le' _ _ _ (fun I => ?_)
            exact card_guess_le A E (member i (I i t)))
        refine (Finset.sum_le_sum (fun t _ => ht t)).trans (le_of_eq ?_)
        simp [div_eq_mul_inv]
    refine (Finset.sum_le_sum (fun i _ => hmem i)).trans (le_of_eq ?_)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · -- compression at `d·k` blocks
    refine (pr_le_of_card _ (2 ^ lam) (q.choose (d * k)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose (d * k) * Fintype.card (Store (d * N) b)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod]; ring))
      exact card_compress_le A y.1 (d * k) lam)).trans (le_of_eq ?_)
    push_cast
    rfl

/-- **Completeness of the co-audited set** (`Bandwidth.Guarantees.SetHonestPasses`): the honest device over the
members' stores side by side, each member reading its own store once. -/
theorem SetHonestPasses : Bandwidth.Guarantees.SetHonestPasses :=
  fun d N b h => ⟨Bandwidth.Completeness.honest (d * N) b h, fun E O _ I =>
    ⟨fun i => (Bandwidth.Completeness.honest_accepts E O (fun t => member i (I i t))).1,
      by rw [(Bandwidth.Completeness.honest_accepts (s := 0) E O Fin.elim0).2, mul_assoc]⟩⟩

end Bandwidth.SecurityProofs
