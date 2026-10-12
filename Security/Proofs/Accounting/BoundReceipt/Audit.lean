import Mathlib
import Models.Accounting.Guarantees.BoundReceipt
import Proofs.Accounting.RateAudit.Audit

/-!
# Receipts bound to the matmul: the proofs

Each location is a `RelayDevice` of `RateAudit`'s game, whose oracle is the mix at what the location holds:

- the GPU at tile outputs `z` is its device (`Device.toRelay`), over the store, at the mix `atZ G z`;
- the rest at a store `E` is a device over the tile outputs (`Rest.view`), with no on-chip state, its memory inlined
  into its program (`Chip.inline`), so that it has no footprint, its tile work as rebuilds and what the GPU sent as
  side reads, at the mix `atE G E`.

A receipt a location never queried is a guess (`card_guess_at_le`, `RateAudit`'s guessed tag at a point of the mix).
An accepted node whose blocks were made at fewer than `k` places is the sample's chance; otherwise either the rest made
`m + 1` (`RateAudit`'s compression of the tile outputs, at `C(q_z, m + 1)/2^λ`) or the GPU made `k − m` (its
compression of the store, at `C(q, k − m)/2^λ`).
-/

namespace Accounting.Rate.Chip

variable {N b h ρ : ℕ} {α : Type}

/-- The program with its device-memory reads answered by `m`: a program that reads no device memory. -/
def inline (m : ℕ → Bool) : Chip N b h ρ α → Chip N b h ρ α
  | done a => done a
  | tag x k => tag x (fun y => (k y).inline m)
  | ingress a k => (k (m a)).inline m
  | side a k => side a (fun v => (k v).inline m)
  | stream i k => stream i (fun v => (k v).inline m)
  | rebuild j k => rebuild j (fun v => (k v).inline m)

theorem inline_run (m : ℕ → Bool) (w : World N b h ρ) (c : Chip N b h ρ α) :
    (c.inline m).run w = c.run { w with mem := m } := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

theorem inline_queries (m : ℕ → Bool) (w : World N b h ρ) (c : Chip N b h ρ α) :
    (c.inline m).queries w = c.queries { w with mem := m } := by
  induction c with
  | done a => rfl
  | tag x k ih => exact congrArg (x :: ·) (ih _)
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

theorem inline_linkBits (m : ℕ → Bool) (w : World N b h ρ) (c : Chip N b h ρ α) :
    (c.inline m).linkBits w = c.linkBits { w with mem := m } := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact congrArg (· + 1) (ih _)
  | stream i k ih => exact congrArg (· + 1) (ih _)
  | rebuild j k ih => exact ih _

theorem inline_rebuilds (m : ℕ → Bool) (w : World N b h ρ) (c : Chip N b h ρ α) :
    (c.inline m).rebuilds w = c.rebuilds { w with mem := m } := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact congrArg (· + 1) (ih _)

theorem inline_addrs (m : ℕ → Bool) (w : World N b h ρ) (c : Chip N b h ρ α) : (c.inline m).addrs w = ∅ := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

end Accounting.Rate.Chip

namespace Accounting.BoundReceipt

open Accounting.Rate
open Pous (Bits)

variable {N a b h S L q Lz qz : ℕ}

/-- The rest at one store as a relay device over the tile outputs: no on-chip state, its memory inlined into its
program, what the GPU sent as its side sources and its tile work as rebuilds. -/
def Rest.view (B : Rest N a h Lz qz) : RelayDevice N a h 0 0 Lz qz where
  state := fun _ => noStream
  mem := fun _ _ => false
  side := fun z _ _ => B.side z
  chip := fun _ => B.chip.inline B.mem
  bounded := fun _ w => by
    rw [Chip.inline_linkBits, Chip.inline_queries]
    exact B.bounded _

theorem Rest.view_world (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) :
    { B.view.world z noStream (atE G E) with mem := B.mem } = B.world E z G := rfl

theorem Rest.view_tags (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) :
    B.view.tags z noStream (atE G E) = B.receipts E z G :=
  Chip.inline_run _ _ _

theorem Rest.mem_view_queried (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) (j : Fin N) :
    j ∈ queried B.view z noStream (atE G E) ↔ (noStream, j, z j) ∈ B.chip.queries (B.world E z G) := by
  simp only [queried, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Iff.of_eq (congrArg _ (Chip.inline_queries _ _ _))

theorem Rest.view_taken (B : Rest N a h Lz qz) (E : Store N b) (z : Store N a) (G : Mix N a b h) :
    taken B.view z noStream (atE G E) =
      B.chip.linkBits (B.world E z G) + B.chip.rebuilds (B.world E z G) * a := by
  simp only [taken, RelayDevice.footprint]
  change ((B.chip.inline B.mem).addrs _).card + 0 + (B.chip.inline B.mem).linkBits _ +
    (B.chip.inline B.mem).rebuilds _ * a = _
  rw [Chip.inline_addrs, Chip.inline_linkBits, Chip.inline_rebuilds]
  simp only [Finset.card_empty, zero_add]
  rfl

theorem noStream_eq (R : Bits 0) : R = noStream := funext (fun i => i.elim0)

end Accounting.BoundReceipt

namespace Accounting.Rate

open Pous (Bits)
open Bandwidth (card_line_le)

/-- `RateAudit`'s guessed tag at a point of a larger random oracle: for an oracle `φ G` read from `G : X → Bits h`
along an injection `ι`, and side sources that ignore the oracle, a word the device never queried has the right tag,
`G` at `ι`'s image of its point, for at most a `2^-h` fraction of `G`. -/
theorem card_guess_at_le {X : Type} [Fintype X] [DecidableEq X] {N b h ρ S L q : ℕ} (A : RelayDevice N b h ρ S L q)
    (E : Store N b) (R : Bits ρ) (j : Fin N) (φ : (X → Bits h) → Oracle N b h ρ) (ι : Bits ρ × Fin N × Bits b → X)
    (hι : Function.Injective ι) (hφ : ∀ G x, φ G x = G (ι x)) (x₀ : X) (hx₀ : ι (R, j, E j) = x₀)
    (hA : ∀ O O', A.side E R O = A.side E R O')
    [DecidablePred (fun G : X → Bits h => j ∉ queried A E R (φ G) ∧ A.tags E R (φ G) j = G x₀)] :
    (Finset.univ.filter (fun G : X → Bits h => j ∉ queried A E R (φ G) ∧ A.tags E R (φ G) j = G x₀)).card * 2 ^ h ≤
      Fintype.card (X → Bits h) := by
  classical
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le x₀ _ (fun G₁ h₁ G₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, queried] at h₁ h₂
  have hoff' : ∀ x, x ≠ (R, j, E j) → φ G₂ x = φ G₁ x := fun x hx => by
    rw [hφ, hφ]
    exact (hoff (ι x) (fun he => hx (hι (he.trans hx₀.symm)))).symm
  have hq : ∀ x ∈ (A.chip (A.state E)).queries (A.world E R (φ G₁)), φ G₂ x = (A.world E R (φ G₁)).oracle x :=
    fun x hx => hoff' x (fun he => h₁.1 (he ▸ hx))
  have hw : A.world E R (φ G₂) = { A.world E R (φ G₁) with oracle := φ G₂ } := by
    simp only [RelayDevice.world, hA (φ G₂) (φ G₁)]
  have ht : A.tags E R (φ G₂) j = A.tags E R (φ G₁) j := by
    simp only [RelayDevice.tags]
    rw [hw]
    exact congrFun (Chip.run_congr (A.world E R (φ G₁)) (φ G₂) _ hq).1 j
  funext x
  by_cases hx : x = x₀
  · subst hx
    rw [← h₁.2, ← h₂.2, ht]
  · exact hoff x hx

/-- A count over stores and the empty stream is a count over stores. -/
theorem card_filter_noStream {X : Type} [Fintype X] (P : X × Bits 0 → Prop) [DecidablePred P] :
    (Finset.univ.filter (fun x : X => P (x, BoundReceipt.noStream))).card = (Finset.univ.filter P).card := by
  classical
  rw [card_filter_equiv (Equiv.prodUnique X (Bits 0)).symm]
  congr 1
  refine Finset.filter_congr (fun y _ => ?_)
  rw [show y = (y.1, BoundReceipt.noStream) from Prod.ext rfl (BoundReceipt.noStream_eq y.2)]
  rfl

end Accounting.Rate

namespace Accounting.BoundReceipt.Node

open Accounting.Rate
open Pous (Bits)

variable {N a b h S L q Lz qz : ℕ}

theorem inj_atZ (z : Store N a) : Function.Injective (fun x : Bits 0 × Fin N × Bits b => (x.2.1, x.2.2, z x.2.1)) := by
  rintro ⟨r₁, j₁, x₁⟩ ⟨r₂, j₂, x₂⟩ he
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl, -⟩ := he
  rw [noStream_eq r₁, noStream_eq r₂]

theorem inj_atE (E : Store N b) : Function.Injective (fun x : Bits 0 × Fin N × Bits a => (x.2.1, E x.2.1, x.2.2)) := by
  rintro ⟨r₁, j₁, y₁⟩ ⟨r₂, j₂, y₂⟩ he
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, -, rfl⟩ := he
  rw [noStream_eq r₁, noStream_eq r₂]

/-- The GPU's guess: a receipt it never made is right for at most a `2^-h` fraction of mixes. -/
theorem card_guess_gpu (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (j : Fin N)
    [DecidablePred (fun G : Mix N a b h => A.gpuReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtGpu E z G j)] :
    (Finset.univ.filter (fun G : Mix N a b h =>
      A.gpuReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtGpu E z G j)).card * 2 ^ h ≤
      Fintype.card (Mix N a b h) := by
  classical
  refine le_trans (Nat.mul_le_mul_right _ (Finset.card_le_card (fun G hG => ?_)))
    (card_guess_at_le (A.gpu z).toRelay E noStream j (fun G => atZ G z) _ (inj_atZ z) (fun _ _ => rfl)
      (j, E j, z j) rfl (fun _ _ => rfl))
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, queried] at hG ⊢
  exact ⟨hG.2, hG.1⟩

/-- The rest's guess: a receipt it never made is right for at most a `2^-h` fraction of mixes. -/
theorem card_guess_rest (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a) (j : Fin N)
    [DecidablePred (fun G : Mix N a b h => A.restReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtRest E z G j)] :
    (Finset.univ.filter (fun G : Mix N a b h =>
      A.restReceipts E z G j = G (j, E j, z j) ∧ ¬ A.MadeAtRest E z G j)).card * 2 ^ h ≤
      Fintype.card (Mix N a b h) := by
  classical
  refine le_trans (Nat.mul_le_mul_right _ (Finset.card_le_card (fun G hG => ?_)))
    (card_guess_at_le (A.rest E).view z noStream j (fun G => atE G E) _ (inj_atE E) (fun _ _ => rfl)
      (j, E j, z j) rfl (fun _ _ => rfl))
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hG ⊢
  rw [Rest.mem_view_queried, Rest.view_tags]
  exact ⟨hG.2, hG.1⟩

end Accounting.BoundReceipt.Node

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.BoundReceipt Accounting.BoundReceipt.Node
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le
  pr_eq_zero_of_card)
open Bandwidth.AuditRO (card_sample)

theorem two_div_eq (h : ℕ) : ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) + ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) =
    2 / 2 ^ h := by
  rw [ENNReal.div_add_div_same]
  push_cast
  rw [one_add_one_eq_two]

/-- **Co-location** (`Accounting.Guarantees.Colocation`). -/
theorem Colocation : Accounting.Guarantees.Colocation := by
  intro N a b h S L q Lz qz A j E z
  classical
  refine (pr_or_le _ _).trans ((add_le_add ?_ ?_).trans (le_of_eq (two_div_eq h)))
  · exact pr_le_of_card _ (2 ^ h) 1 (by positivity) (by rw [one_mul]; exact card_guess_gpu A E z j)
  · exact pr_le_of_card _ (2 ^ h) 1 (by positivity) (by rw [one_mul]; exact card_guess_rest A E z j)

/-- **The link's share** (`Accounting.Guarantees.BoundReceiptShare`). -/
theorem BoundReceiptShare : Accounting.Guarantees.BoundReceiptShare := by
  intro a b w L Lz Fz lam ha hw
  have hd : 0 < a * w := Nat.mul_pos ha hw
  set X := (Lz + lam) * w + Fz * a with hX
  have hm : covered a w Lz Fz lam * (a * w) ≤ X := Nat.div_mul_le_self X (a * w)
  refine ⟨?_, ?_⟩
  · have := Nat.lt_mul_div_succ X hd
    unfold covered
    rw [← hX]
    calc X < a * w * (X / (a * w) + 1) := this
      _ = (X / (a * w) + 1) * a * w := by ring
  · calc (L + covered a w Lz Fz lam * b) * min a b * w
        = L * min a b * w + covered a w Lz Fz lam * b * min a b * w := by ring
      _ ≤ L * b * w + covered a w Lz Fz lam * b * a * w :=
          add_le_add (Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (min_le_right a b)))
            (Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (min_le_left a b)))
      _ = b * (L * w + covered a w Lz Fz lam * (a * w)) := by ring
      _ ≤ b * (L * w + X) := by gcongr
      _ = b * ((L + Lz + lam) * w + Fz * a) := by rw [hX]; ring

/-- **The link relay** (`Accounting.Guarantees.BoundReceiptLink`). -/
theorem BoundReceiptLink : Accounting.Guarantees.BoundReceiptLink := by
  intro N a b h S L q Lz qz s k m F cTag cEnc w Fz lam A hm
  classical
  rcases Nat.eq_zero_or_pos (k - m) with hkm | hk0
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [floor, hkm] at hx), Finset.card_empty, zero_mul,
      zero_mul]
  let gQ : Store N b → Store N a → Mix N a b h → Finset (Fin N) :=
    fun E z G => queried (A.gpu z).toRelay E noStream (atZ G z)
  let rQ : Store N b → Store N a → Mix N a b h → Finset (Fin N) :=
    fun E z G => queried (A.rest E).view z noStream (atE G E)
  have hsplit : ∀ x : Store N b × BoundReceipt.Coins N a b h s,
      (A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
        A.tileWork w x.1 x.2.1 x.2.2.1 ≤ Fz ∧
        A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b 0 S L F cTag cEnc (k - m) lam) →
      (((gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1).card < k ∧
          ∀ i, x.2.2.2 i ∈ gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1) ∨
        (∃ i, (A.gpuReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ A.MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i)) ∨
          (A.restReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) =
              x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ A.MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i)))) ∨
      ((k - m ≤ (gQ x.1 x.2.1 x.2.2.1).card ∧
          taken (A.gpu x.2.1).toRelay x.1 noStream (atZ x.2.2.1 x.2.1) + lam < (k - m) * b + 0) ∨
        (m + 1 ≤ (rQ x.1 x.2.1 x.2.2.1).card ∧
          taken (A.rest x.1).view x.2.1 noStream (atE x.2.2.1 x.1) + lam < (m + 1) * a + 0)) := by
    rintro ⟨E, z, G, I⟩ ⟨hacc, hc, hz, hf⟩
    dsimp only at hacc hc hz hf ⊢
    have hgQ : ∀ j, j ∈ gQ E z G ↔ A.MadeAtGpu E z G j := fun j => by
      simp only [gQ, queried, Finset.mem_filter, Finset.mem_univ, true_and]
      rfl
    have hrQ : ∀ j, j ∈ rQ E z G ↔ A.MadeAtRest E z G j := fun j => Rest.mem_view_queried _ E z G j
    by_cases hk : k ≤ (gQ E z G ∪ rQ E z G).card
    · refine Or.inr ?_
      by_cases hr : m + 1 ≤ (rQ E z G).card
      · refine Or.inr ⟨hr, ?_⟩
        rw [Rest.view_taken]
        have hL := ((A.rest E).bounded ((A.rest E).world E z G)).1
        unfold Node.tileWork at hz
        generalize (A.rest E).chip.linkBits ((A.rest E).world E z G) = lb at hL ⊢
        generalize (A.rest E).chip.rebuilds ((A.rest E).world E z G) = G' at hz ⊢
        have h1 : (lb + G' * a + lam) * w < (m + 1) * a * w := by
          calc (lb + G' * a + lam) * w = (lb + lam) * w + G' * w * a := by ring
            _ ≤ (Lz + lam) * w + Fz * a := by gcongr
            _ < (m + 1) * a * w := hm
        have := Nat.lt_of_mul_lt_mul_right h1
        omega
      · have hq : k - m ≤ (gQ E z G).card := by
          have := Finset.card_union_le (gQ E z G) (rQ E z G)
          omega
        exact Or.inl ⟨hq, taken_lt_of_floor (A.gpu z).toRelay E noStream (atZ G z) hq hc hf⟩
    · refine Or.inl ?_
      by_cases hall : ∀ i, I i ∈ gQ E z G ∪ rQ E z G
      · exact Or.inl ⟨by omega, hall⟩
      · push Not at hall
        obtain ⟨i, hi⟩ := hall
        simp only [Finset.mem_union, not_or, hgQ, hrQ] at hi
        refine Or.inr ⟨i, ?_⟩
        rcases hacc i with h | h
        · exact Or.inl ⟨h, hi.1⟩
        · exact Or.inr ⟨h, hi.2⟩
  have hΩ : Fintype.card (Store N b × BoundReceipt.Coins N a b h s) =
      Fintype.card (Store N b) * (Fintype.card (Store N a) * (Fintype.card (Mix N a b h) * N ^ s)) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans ?_
  refine (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ((pr_or_le _ _).trans (add_le_add ?_ ?_))).trans
    (le_of_eq (add_assoc _ _ _).symm)
  · -- the sample misses the blocks made nowhere
    have hc := pr_le_of_card (fun x : Store N b × BoundReceipt.Coins N a b h s =>
        (gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1).card < k ∧
          ∀ i, x.2.2.2 i ∈ gQ x.1 x.2.1 x.2.2.1 ∪ rQ x.1 x.2.1 x.2.2.1) (N ^ s) ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card (Store N a) * (Fintype.card (Mix N a b h) * ((k - 1) ^ s * N ^ s)))
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
  · -- a guessed receipt, at either location
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : Store N b × BoundReceipt.Coins N a b h s =>
        (A.gpuReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) = x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ A.MadeAtGpu x.1 x.2.1 x.2.2.1 (x.2.2.2 i)) ∨
          (A.restReceipts x.1 x.2.1 x.2.2.1 (x.2.2.2 i) = x.2.2.1 (x.2.2.2 i, x.1 (x.2.2.2 i), x.2.1 (x.2.2.2 i)) ∧
            ¬ A.MadeAtRest x.1 x.2.1 x.2.2.1 (x.2.2.2 i))) ≤ 2 / 2 ^ h := fun i => by
      refine (pr_or_le _ _).trans ((add_le_add ?_ ?_).trans (le_of_eq (two_div_eq h)))
      · refine pr_le_of_card _ (2 ^ h) 1 (by positivity) ?_
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Store N a) *
          (Fintype.card (Mix N a b h) * Fintype.card (Fin s → Fin N))) (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun z => ?_)
        exact card_filter_prod_le' _ _ _ (fun I => card_guess_gpu A E z (I i))
      · refine pr_le_of_card _ (2 ^ h) 1 (by positivity) ?_
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Store N a) *
          (Fintype.card (Mix N a b h) * Fintype.card (Fin s → Fin N))) (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun z => ?_)
        exact card_filter_prod_le' _ _ _ (fun I => card_guess_rest A E z (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_div_assoc', mul_comm]
  · -- the GPU made `k − m`: compressing the store
    refine (pr_le_of_card _ (2 ^ lam) (q.choose (k - m)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose (k - m) * Fintype.card (Store N b)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      have := card_compress_le (A.gpu y.1).toRelay (atZ y.2.1 y.1) (k - m) lam hk0
      rw [← card_filter_noStream] at this
      simpa [Fintype.card_prod] using this)).trans (le_of_eq ?_)
    push_cast
    rfl
  · -- the rest made `m + 1`: compressing the tile outputs
    refine (pr_le_of_card _ (2 ^ lam) (qz.choose (m + 1)) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (qz.choose (m + 1) * Fintype.card (Store N a) *
        (Fintype.card (Mix N a b h) * N ^ s)) (fun E => ?_)).trans (le_of_eq (by ring))
      refine (card_filter_prod_le' _ _ (qz.choose (m + 1) * Fintype.card (Store N a)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]))
      have := card_compress_le (A.rest E).view (atE y.1 E) (m + 1) lam (Nat.succ_pos m)
      rw [← card_filter_noStream] at this
      simpa [Fintype.card_prod] using this)).trans (le_of_eq ?_)
    push_cast
    rfl

end Accounting.SecurityProofs
