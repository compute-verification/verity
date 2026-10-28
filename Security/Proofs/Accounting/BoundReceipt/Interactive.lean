import Proofs.Accounting.BoundReceipt.Audit

/-!
# Receipts bound to the matmul when the locations' answers depend on the mix (`e > 0`)

At each value of the mix the node is a `Node` (`InteractiveNode.under`), so `BoundReceiptLink`'s compressions apply
as they are, each at a fixed mix. The guess is taken over the node's queries instead of one location's: if neither
location queried block `j`'s point, a mix that differs only there gives the same receipts (`QueryDetermined`), so a
right receipt there is a guess, at most one value of the mix in `2^h` for each location's receipt.
-/

namespace Accounting.BoundReceipt.InteractiveNode

open Accounting.Rate
open Pous (Bits)
open Bandwidth (card_line_le)

variable {N a b h S L q Lz qz : ℕ}

theorem under_ofNode (A : Node N a b h S L q Lz qz) (G : Mix N a b h) : (ofNode A).under G = A := rfl

/-- A node whose answers ignore the mix has `QueryDetermined`: each location's run changes nowhere it didn't query. -/
theorem ofNode_queryDetermined (A : Node N a b h S L q Lz qz) : (ofNode A).QueryDetermined := by
  intro E z G G' hg hr
  exact ⟨(Chip.run_congr ((A.gpu z).world E noStream (atZ G z)) (atZ G' z) _ hg).1,
    (Chip.run_congr ((A.rest E).world E z G) (atE G' E) _ hr).1⟩

/-- If neither location made block `j`'s receipt under `G₁`, a mix `G₂` equal to `G₁` off `(j, E_j, z_j)` gives the
same receipts at both locations. -/
theorem receipts_off (A : InteractiveNode N a b h S L q Lz qz) (hA : A.QueryDetermined) (E : Payloads N b)
    (z : Payloads N a) (j : Fin N) (G₁ G₂ : Mix N a b h)
    (hg : ¬ (A.under G₁).MadeAtGpu E z G₁ j) (hr : ¬ (A.under G₁).MadeAtRest E z G₁ j)
    (hoff : ∀ x, x ≠ (j, E j, z j) → G₁ x = G₂ x) :
    (A.under G₂).gpuReceipts E z G₂ = (A.under G₁).gpuReceipts E z G₁ ∧
      (A.under G₂).restReceipts E z G₂ = (A.under G₁).restReceipts E z G₁ := by
  refine hA E z G₁ G₂ (fun x hx => (hoff _ fun he => hg ?_).symm) (fun y hy => (hoff _ fun he => hr ?_).symm)
  · obtain ⟨r, j', x'⟩ := x
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, -⟩ := he
    rw [noStream_eq r] at hx
    exact hx
  · obtain ⟨r, j', y'⟩ := y
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, -, rfl⟩ := he
    rw [noStream_eq r] at hy
    exact hy

/-- The GPU's receipt for a block neither location made is right for at most a `2^-h` fraction of mixes. -/
theorem card_guess_gpu (A : InteractiveNode N a b h S L q Lz qz) (hA : A.QueryDetermined) (E : Payloads N b)
    (z : Payloads N a) (j : Fin N)
    [DecidablePred (fun G : Mix N a b h => (A.under G).gpuReceipts E z G j = G (j, E j, z j) ∧
      ¬ (A.under G).MadeAtGpu E z G j ∧ ¬ (A.under G).MadeAtRest E z G j)] :
    (Finset.univ.filter (fun G : Mix N a b h => (A.under G).gpuReceipts E z G j = G (j, E j, z j) ∧
      ¬ (A.under G).MadeAtGpu E z G j ∧ ¬ (A.under G).MadeAtRest E z G j)).card * 2 ^ h ≤
      Fintype.card (Mix N a b h) := by
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (j, E j, z j) _ (fun G₁ h₁ G₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at h₁ h₂
  have ht := (receipts_off A hA E z j G₁ G₂ h₁.2.1 h₁.2.2 hoff).1
  funext x
  by_cases hx : x = (j, E j, z j)
  · rw [hx, ← h₁.1, ← h₂.1, ht]
  · exact hoff x hx

/-- The rest's receipt for a block neither location made is right for at most a `2^-h` fraction of mixes. -/
theorem card_guess_rest (A : InteractiveNode N a b h S L q Lz qz) (hA : A.QueryDetermined) (E : Payloads N b)
    (z : Payloads N a) (j : Fin N)
    [DecidablePred (fun G : Mix N a b h => (A.under G).restReceipts E z G j = G (j, E j, z j) ∧
      ¬ (A.under G).MadeAtGpu E z G j ∧ ¬ (A.under G).MadeAtRest E z G j)] :
    (Finset.univ.filter (fun G : Mix N a b h => (A.under G).restReceipts E z G j = G (j, E j, z j) ∧
      ¬ (A.under G).MadeAtGpu E z G j ∧ ¬ (A.under G).MadeAtRest E z G j)).card * 2 ^ h ≤
      Fintype.card (Mix N a b h) := by
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (j, E j, z j) _ (fun G₁ h₁ G₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at h₁ h₂
  have ht := (receipts_off A hA E z j G₁ G₂ h₁.2.1 h₁.2.2 hoff).2
  funext x
  by_cases hx : x = (j, E j, z j)
  · rw [hx, ← h₁.1, ← h₂.1, ht]
  · exact hoff x hx

end Accounting.BoundReceipt.InteractiveNode

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.BoundReceipt Accounting.BoundReceipt.Node
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le
  pr_eq_zero_of_card)
open Bandwidth.AuditRO (card_sample)

/-- **The link relay at `e > 0`** (`Accounting.Guarantees.BoundReceiptLinkInteractive`). -/
theorem BoundReceiptLinkInteractive : Accounting.Guarantees.BoundReceiptLinkInteractive := by
  intro N a b h S L q Lz qz s k m F cTag cEnc w Fz lam A hA hm
  rw [Bandwidth.pr_coins_flat]
  dsimp only
  classical
  rcases Nat.eq_zero_or_pos (k - m) with hkm | hk0
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [floor, hkm] at hx), Finset.card_empty, zero_mul,
      zero_mul]
  let gQ : Payloads N b → Payloads N a → Mix N a b h → Finset (Fin N) :=
    fun E z G => queried ((A.under G).gpu z).toRelay E noStream (atZ G z)
  let rQ : Payloads N b → Payloads N a → Mix N a b h → Finset (Fin N) :=
    fun E z G => queried ((A.under G).rest E).view z noStream (atE G E)
  have hsplit : ∀ x : Payloads N b × (Payloads N a × Mix N a b h × (Fin s → Fin N)),
      ((A.under x.2.2.1).Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧
        (A.under x.2.2.1).compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
        (A.under x.2.2.1).tileWork w x.1 x.2.1 x.2.2.1 ≤ Fz ∧
        (A.under x.2.2.1).footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b 0 S L F cTag cEnc (k - m) lam) →
      (((gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1).card < k ∧
          ∀ i, x.2.2.2 i ∈ gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1) ∨
        (∃ i, ((A.under x.2.2.1).gpuReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ (A.under x.2.2.1).MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i) ∧
            ¬ (A.under x.2.2.1).MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i)) ∨
          ((A.under x.2.2.1).restReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ (A.under x.2.2.1).MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i) ∧
            ¬ (A.under x.2.2.1).MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i)))) ∨
      ((k - m ≤ (gQ x.1 x.2.1 x.2.2.1).card ∧
          taken ((A.under x.2.2.1).gpu x.2.1).toRelay x.1 noStream (atZ x.2.2.1 x.2.1) + lam < (k - m) * b + 0) ∨
        (m + 1 ≤ (rQ x.1 x.2.1 x.2.2.1).card ∧
          taken ((A.under x.2.2.1).rest x.1).view x.2.1 noStream (atE x.2.2.1 x.1) + lam < (m + 1) * a + 0)) := by
    rintro ⟨E, z, G, I⟩ ⟨hacc, hc, hz, hf⟩
    dsimp only at hacc hc hz hf ⊢
    have hgQ : ∀ j, j ∈ gQ E z G ↔ (A.under G).MadeAtGpu E z G j := fun j => by
      simp only [gQ, queried, Finset.mem_filter, Finset.mem_univ, true_and]
      rfl
    have hrQ : ∀ j, j ∈ rQ E z G ↔ (A.under G).MadeAtRest E z G j := fun j => Rest.mem_view_queried _ E z G j
    by_cases hk : k ≤ (gQ E z G ∪ rQ E z G).card
    · refine Or.inr ?_
      by_cases hr : m + 1 ≤ (rQ E z G).card
      · refine Or.inr ⟨hr, ?_⟩
        rw [Rest.view_taken]
        have hL := (((A.under G).rest E).bounded (((A.under G).rest E).world E z G)).1
        unfold Node.tileWork at hz
        generalize ((A.under G).rest E).chip.linkBits (((A.under G).rest E).world E z G) = lb at hL ⊢
        generalize ((A.under G).rest E).chip.rebuilds (((A.under G).rest E).world E z G) = G' at hz ⊢
        have h1 : (lb + G' * a + lam) * w < (m + 1) * a * w := by
          calc (lb + G' * a + lam) * w = (lb + lam) * w + G' * w * a := by ring
            _ ≤ (Lz + lam) * w + Fz * a := by gcongr
            _ < (m + 1) * a * w := hm
        have := Nat.lt_of_mul_lt_mul_right h1
        omega
      · have hq : k - m ≤ (gQ E z G).card := by
          have := Finset.card_union_le (gQ E z G) (rQ E z G)
          omega
        exact Or.inl ⟨hq, taken_lt_of_floor ((A.under G).gpu z).toRelay E noStream (atZ G z) hq hc hf⟩
    · refine Or.inl ?_
      by_cases hall : ∀ i, I i ∈ gQ E z G ∪ rQ E z G
      · exact Or.inl ⟨by omega, hall⟩
      · push Not at hall
        obtain ⟨i, hi⟩ := hall
        simp only [Finset.mem_union, not_or, hgQ, hrQ] at hi
        refine Or.inr ⟨i, ?_⟩
        rcases hacc i with h | h
        · exact Or.inl ⟨h, hi.1, hi.2⟩
        · exact Or.inr ⟨h, hi.1, hi.2⟩
  have hΩ : Fintype.card (Payloads N b × (Payloads N a × Mix N a b h × (Fin s → Fin N))) =
      Fintype.card (Payloads N b) * (Fintype.card (Payloads N a) * (Fintype.card (Mix N a b h) * N ^ s)) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans ?_
  refine (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ((pr_or_le _ _).trans (add_le_add ?_ ?_))).trans
    (le_of_eq (add_assoc _ _ _).symm)
  · -- the sample misses the blocks made nowhere
    have hc := pr_le_of_card (fun x : Payloads N b × (Payloads N a × Mix N a b h × (Fin s → Fin N)) =>
        (gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1).card < k ∧
          ∀ i, x.2.2.2 i ∈ gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1) (N ^ s) ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card (Payloads N a) * (Fintype.card (Mix N a b h) * ((k - 1) ^ s * N ^ s)))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun z => ?_)
      refine card_filter_prod_le _ _ _ (fun G => ?_)
      by_cases hk : (gQ E z G ∪ rQ E z G).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- a guessed receipt: its point queried at neither location
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : Payloads N b × (Payloads N a × Mix N a b h × (Fin s → Fin N)) =>
        ((A.under x.2.2.1).gpuReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ (A.under x.2.2.1).MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i) ∧
            ¬ (A.under x.2.2.1).MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i)) ∨
          ((A.under x.2.2.1).restReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ (A.under x.2.2.1).MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i) ∧
            ¬ (A.under x.2.2.1).MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i))) ≤ 2 / 2 ^ h := fun i => by
      refine (pr_or_le _ _).trans ((add_le_add ?_ ?_).trans (le_of_eq (two_div_eq h)))
      · refine pr_le_of_card _ (2 ^ h) 1 (by positivity) ?_
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Payloads N a) *
          (Fintype.card (Mix N a b h) * Fintype.card (Fin s → Fin N))) (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun z => ?_)
        exact card_filter_prod_le' _ _ _ (fun I => InteractiveNode.card_guess_gpu A hA E z (I i))
      · refine pr_le_of_card _ (2 ^ h) 1 (by positivity) ?_
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Payloads N a) *
          (Fintype.card (Mix N a b h) * Fintype.card (Fin s → Fin N))) (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun z => ?_)
        exact card_filter_prod_le' _ _ _ (fun I => InteractiveNode.card_guess_rest A hA E z (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_div_assoc', mul_comm]
  · -- the GPU made `k − m`: compressing the store, at each mix
    refine (pr_le_of_card _ (2 ^ lam) (q.choose (k - m)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose (k - m) * Fintype.card (Payloads N b)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      have := card_compress_le ((A.under y.2.1).gpu y.1).toRelay (atZ y.2.1 y.1) (k - m) lam hk0
      rw [← card_filter_noStream] at this
      simpa [Fintype.card_prod] using this)).trans (le_of_eq ?_)
    push_cast
    rfl
  · -- the rest made `m + 1`: compressing the tile outputs, at each mix
    refine (pr_le_of_card _ (2 ^ lam) (qz.choose (m + 1)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (qz.choose (m + 1) * Fintype.card (Payloads N a) *
        (Fintype.card (Mix N a b h) * N ^ s)) (fun E => ?_)).trans (le_of_eq (by ring))
      refine (card_filter_prod_le' _ _ (qz.choose (m + 1) * Fintype.card (Payloads N a)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]))
      have := card_compress_le ((A.under y.1).rest E).view (atE y.1 E) (m + 1) lam (Nat.succ_pos m)
      rw [← card_filter_noStream] at this
      simpa [Fintype.card_prod] using this)).trans (le_of_eq ?_)
    push_cast
    rfl


/-- At a node whose answers ignore the mix, `BoundReceiptLinkInteractive` is `BoundReceiptLink`. -/
theorem boundReceiptLink_of_interactive : Accounting.Guarantees.BoundReceiptLink :=
  fun N a b h S L q Lz qz s k m F cTag cEnc w Fz lam A hm =>
    BoundReceiptLinkInteractive N a b h S L q Lz qz s k m F cTag cEnc w Fz lam (InteractiveNode.ofNode A)
      (InteractiveNode.ofNode_queryDetermined A) hm

end Accounting.SecurityProofs
