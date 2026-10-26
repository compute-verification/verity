import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Sigma
import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe: the counting argument

`Pr[accepted ∧ keeps z] ≤ (g/N)^k + k · C(N, g+1) · 2^(M+E) / 2^(q+(g+1)·b)` in the ideal game (`win_ideal_le`), from
two counts over the runs `((z, c), L)`:

- **sampling** (`sampling_count`): for any sets `H h` of at most `g` indices, one for each history `h`, at most `g^k`
  challenge vectors put every index in the set of its history;
- **compression** (`compression_count`): for fixed challenges and a fixed history, at most
  `C(N, g+1) · 2^M · 2^E · 2^(b·(N−g−1))` pairs `(z, L)` keep `z` and answer more than `g` labels after that history.
  For each set `T` of `g+1` indices, `(z, L) ↦ (σ, ξ, L off T)` is injective on the runs that keep `z` and answer all
  of `T`: `z` is what `keep` returns, and `L` on `T` is what `answer` returns.

A run that wins either answers at most `g` labels after every history it sees, and is counted by sampling, or answers
more than `g` after some round's history while keeping `z`, and is counted by compression at that round
(`win_count`).
-/

namespace PousStreamedWipe

open Finset Pous Pous.StreamedWipe
open scoped ENNReal

theorem revealed_zero {N k : ℕ} (c : Fin (k + 1) → Fin N) : revealed c 0 = [] := by
  simp [revealed]

theorem revealed_succ {N k : ℕ} (c : Fin (k + 1) → Fin N) (j : Fin k) :
    revealed c j.succ = c 0 :: revealed (Fin.tail c) j := by
  simp [revealed, List.ofFn_succ]
  rfl

/-- **Sampling.** If every history's set has at most `g` indices, at most `g^k` challenge vectors put each index in
the set of the history before it. -/
theorem sampling_count {N g : ℕ} (k : ℕ) :
    ∀ H : List (Fin N) → Finset (Fin N), (∀ h, (H h).card ≤ g) →
      (univ.filter fun c : Fin k → Fin N => ∀ j : Fin k, c j ∈ H (revealed c j)).card ≤ g ^ k := by
  induction k with
  | zero => intro H _; exact (card_filter_le _ _).trans (by simp)
  | succ k ih =>
    intro H hH
    classical
    let S := univ.filter fun c : Fin (k + 1) → Fin N => ∀ j : Fin (k + 1), c j ∈ H (revealed c j)
    let T : Fin N → Finset (Fin k → Fin N) := fun x =>
      univ.filter fun c : Fin k → Fin N => ∀ j : Fin k, c j ∈ H (x :: revealed c j)
    have hmaps : Set.MapsTo (fun c : Fin (k + 1) → Fin N => (⟨c 0, Fin.tail c⟩ : Σ _ : Fin N, Fin k → Fin N))
        (S : Set _) ((H []).sigma T : Set _) := by
      intro c hc
      simp only [S, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hc
      simp only [coe_sigma, Set.mem_sigma_iff, mem_coe, T, mem_filter, mem_univ, true_and]
      refine ⟨by simpa [revealed_zero] using hc 0, fun j => ?_⟩
      simpa [revealed_succ, Fin.tail] using hc j.succ
    have hinj : Set.InjOn (fun c : Fin (k + 1) → Fin N => (⟨c 0, Fin.tail c⟩ : Σ _ : Fin N, Fin k → Fin N))
        (S : Set _) := by
      intro c _ d _ hcd
      simp only [Sigma.mk.injEq] at hcd
      obtain ⟨h0, ht⟩ := hcd
      rw [← Fin.cons_self_tail c, ← Fin.cons_self_tail d, h0, eq_of_heq ht]
    calc S.card ≤ ((H []).sigma T).card := card_le_card_of_injOn _ hmaps hinj
      _ = ∑ x ∈ H [], (T x).card := card_sigma _ _
      _ ≤ (H []).card • g ^ k := sum_le_card_nsmul _ _ _ fun x _ => ih (fun h => H (x :: h)) fun _ => hH _
      _ ≤ g • g ^ k := Nat.mul_le_mul_right _ (hH [])
      _ = g ^ (k + 1) := by rw [smul_eq_mul, pow_succ, mul_comm]

/-- The indices `P` answers correctly after the history `h`, in the run with prior content `z` and labels `L`. -/
noncomputable def answered {N k b q M E : ℕ} (P : Prover N k b q M E) (z : Bits q) (L : Fin N → Bits b)
    (h : List (Fin N)) : Finset (Fin N) := by
  classical
  exact univ.filter fun i => P.answer (P.fill z L).1 (P.fill z L).2 h i = L i

theorem mem_answered {N k b q M E : ℕ} (P : Prover N k b q M E) (z : Bits q) (L : Fin N → Bits b)
    (h : List (Fin N)) (i : Fin N) : i ∈ answered P z L h ↔ P.answer (P.fill z L).1 (P.fill z L).2 h i = L i := by
  classical
  unfold answered
  simp

open Classical in
/-- **Compression.** For fixed challenges `c` and a fixed history `h`, at most `C(N, g+1) · 2^M · 2^E · 2^(b·(N−g−1))`
pairs `(z, L)` keep `z` and answer more than `g` labels after `h`. -/
theorem compression_count {N k b q M E : ℕ} (g : ℕ) (P : Prover N k b q M E) (c : Fin k → Fin N)
    (h : List (Fin N)) :
    (univ.filter fun x : Bits q × (Fin N → Bits b) =>
        Keeps P x.1 x.2 c ∧ g < (answered P x.1 x.2 h).card).card
      ≤ N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))) := by
  classical
  let ST : Finset (Fin N) → Finset (Bits q × (Fin N → Bits b)) := fun T =>
    univ.filter fun x => Keeps P x.1 x.2 c ∧ T ⊆ answered P x.1 x.2 h
  have hsub : (univ.filter fun x : Bits q × (Fin N → Bits b) =>
      Keeps P x.1 x.2 c ∧ g < (answered P x.1 x.2 h).card) ⊆ (powersetCard (g + 1) univ).biUnion ST := by
    intro x hx
    simp only [mem_filter, mem_univ, true_and] at hx
    obtain ⟨T, hT, hTc⟩ := exists_subset_card_eq (s := answered P x.1 x.2 h) (n := g + 1) hx.2
    simp only [mem_biUnion, mem_powersetCard, subset_univ, true_and, ST, mem_filter, mem_univ]
    exact ⟨T, hTc, hx.1, hT⟩
  have hST : ∀ T ∈ powersetCard (g + 1) (univ : Finset (Fin N)),
      (ST T).card ≤ 2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1)) := by
    intro T hT
    have hTc : T.card = g + 1 := (mem_powersetCard.1 hT).2
    let f : Bits q × (Fin N → Bits b) → Bits M × Bits E × ({i // i ∉ T} → Bits b) :=
      fun x => ((P.fill x.1 x.2).1, (P.fill x.1 x.2).2, fun i => x.2 i.1)
    have hinj : Set.InjOn f (ST T : Set _) := by
      intro x hx y hy hxy
      simp only [ST, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hx hy
      simp only [f, Prod.mk.injEq] at hxy
      obtain ⟨hσ, hξ, hoff⟩ := hxy
      have hz : x.1 = y.1 := by
        rw [← hx.1, ← hy.1, hσ, hξ]
      have hL : x.2 = y.2 := by
        funext i
        by_cases hi : i ∈ T
        · rw [← (mem_answered P _ _ h i).1 (hx.2 hi), ← (mem_answered P _ _ h i).1 (hy.2 hi), hσ, hξ]
        · exact congrFun hoff ⟨i, hi⟩
      exact Prod.ext hz hL
    calc (ST T).card ≤ (univ : Finset (Bits M × Bits E × ({i // i ∉ T} → Bits b))).card :=
          card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) hinj
      _ = 2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1)) := by
          rw [card_univ, Fintype.card_prod, Fintype.card_prod, Fintype.card_fun, Fintype.card_fun,
            Fintype.card_fun, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_fin,
            Fintype.card_fin, Fintype.card_subtype_compl, Fintype.card_coe, hTc, Fintype.card_fin, mul_assoc]
  calc _ ≤ ((powersetCard (g + 1) univ).biUnion ST).card := card_le_card hsub
    _ ≤ ∑ T ∈ powersetCard (g + 1) univ, (ST T).card := card_biUnion_le
    _ ≤ (powersetCard (g + 1) (univ : Finset (Fin N))).card • (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))) :=
        sum_le_card_nsmul _ _ _ hST
    _ = N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))) := by
        rw [card_powersetCard, card_univ, Fintype.card_fin, smul_eq_mul]

/-- **The count.** The winning runs `((z, c), L)` number at most `2^q · 2^(b·N) · g^k` (sampling) plus
`k · N^k · C(N, g+1) · 2^M · 2^E · 2^(b·(N−g−1))` (compression, at some round). -/
theorem win_count {N k b q M E : ℕ} (g : ℕ) (P : Prover N k b q M E) [DecidablePred (wipeDistinguisher P)] :
    (univ.filter (wipeDistinguisher P)).card ≤
      2 ^ q * (2 ^ b) ^ N * g ^ k + k * (N ^ k * (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))))) := by
  classical
  let H : Bits q → (Fin N → Bits b) → List (Fin N) → Finset (Fin N) := fun z L h =>
    if (answered P z L h).card ≤ g then answered P z L h else ∅
  let good := univ.filter fun x : (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) =>
    ∀ j : Fin k, x.1.2 j ∈ H x.1.1 x.2 (revealed x.1.2 j)
  let bad : Fin k → Finset ((Bits q × (Fin k → Fin N)) × (Fin N → Bits b)) := fun j =>
    univ.filter fun x => Keeps P x.1.1 x.2 x.1.2 ∧ g < (answered P x.1.1 x.2 (revealed x.1.2 j)).card
  have hsplit : univ.filter (wipeDistinguisher P) ⊆ good ∪ univ.biUnion bad := by
    intro x hx
    simp only [mem_filter, mem_univ, true_and, wipeDistinguisher] at hx
    obtain ⟨hacc, hkeep⟩ := hx
    by_cases hb : ∃ j : Fin k, g < (answered P x.1.1 x.2 (revealed x.1.2 j)).card
    · obtain ⟨j, hj⟩ := hb
      refine mem_union_right _ (mem_biUnion.2 ⟨j, mem_univ _, ?_⟩)
      simp only [bad, mem_filter, mem_univ, true_and]
      exact ⟨hkeep, hj⟩
    · simp only [not_exists, not_lt] at hb
      refine mem_union_left _ ?_
      simp only [good, mem_filter, mem_univ, true_and]
      intro j
      simp only [H, hb j, ↓reduceIte]
      exact (mem_answered P _ _ _ _).2 (hacc j)
  have hgood : good.card ≤ 2 ^ q * (2 ^ b) ^ N * g ^ k := by
    let fib : (Bits q × (Fin N → Bits b)) → Finset (Fin k → Fin N) := fun y =>
      univ.filter fun c => ∀ j : Fin k, c j ∈ H y.1 y.2 (revealed c j)
    have hmaps : Set.MapsTo
        (fun x : (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) =>
          (⟨(x.1.1, x.2), x.1.2⟩ : Σ _ : Bits q × (Fin N → Bits b), Fin k → Fin N))
        (good : Set _) ((univ : Finset (Bits q × (Fin N → Bits b))).sigma fib : Set _) := by
      intro x hx
      simp only [good, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hx
      simp only [coe_sigma, Set.mem_sigma_iff, coe_univ, Set.mem_univ, true_and, mem_coe, fib, mem_filter]
      exact ⟨mem_univ _, hx⟩
    have hinj : Set.InjOn
        (fun x : (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) =>
          (⟨(x.1.1, x.2), x.1.2⟩ : Σ _ : Bits q × (Fin N → Bits b), Fin k → Fin N)) (good : Set _) := by
      rintro ⟨⟨z, c⟩, L⟩ _ ⟨⟨z', c'⟩, L'⟩ _ hxy
      simp only [Sigma.mk.injEq, Prod.mk.injEq, heq_eq_eq] at hxy
      obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hxy
      rfl
    have hfib : ∀ y ∈ (univ : Finset (Bits q × (Fin N → Bits b))), (fib y).card ≤ g ^ k := fun y _ =>
      sampling_count k (H y.1 y.2) fun h => by
        simp only [H]
        split_ifs with hc
        · exact hc
        · simp
    calc good.card ≤ ((univ : Finset (Bits q × (Fin N → Bits b))).sigma fib).card :=
          card_le_card_of_injOn _ hmaps hinj
      _ = ∑ y, (fib y).card := card_sigma _ _
      _ ≤ (univ : Finset (Bits q × (Fin N → Bits b))).card • g ^ k := sum_le_card_nsmul _ _ _ hfib
      _ = 2 ^ q * (2 ^ b) ^ N * g ^ k := by
          rw [card_univ, Fintype.card_prod, Fintype.card_fun, Fintype.card_fun, Fintype.card_fun,
            Fintype.card_bool, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin, smul_eq_mul]
  have hbad : ∀ j ∈ (univ : Finset (Fin k)),
      (bad j).card ≤ N ^ k * (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1)))) := by
    intro j _
    let fib : (Fin k → Fin N) → Finset (Bits q × (Fin N → Bits b)) := fun c =>
      univ.filter fun y => Keeps P y.1 y.2 c ∧ g < (answered P y.1 y.2 (revealed c j)).card
    have hmaps : Set.MapsTo
        (fun x : (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) =>
          (⟨x.1.2, (x.1.1, x.2)⟩ : Σ _ : Fin k → Fin N, Bits q × (Fin N → Bits b)))
        (bad j : Set _) ((univ : Finset (Fin k → Fin N)).sigma fib : Set _) := by
      intro x hx
      simp only [bad, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hx
      simp only [coe_sigma, Set.mem_sigma_iff, coe_univ, Set.mem_univ, true_and, mem_coe, fib, mem_filter]
      exact ⟨mem_univ _, hx⟩
    have hinj : Set.InjOn
        (fun x : (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) =>
          (⟨x.1.2, (x.1.1, x.2)⟩ : Σ _ : Fin k → Fin N, Bits q × (Fin N → Bits b))) (bad j : Set _) := by
      rintro ⟨⟨z, c⟩, L⟩ _ ⟨⟨z', c'⟩, L'⟩ _ hxy
      simp only [Sigma.mk.injEq] at hxy
      obtain ⟨rfl, hrest⟩ := hxy
      simp only [heq_eq_eq, Prod.mk.injEq] at hrest
      obtain ⟨rfl, rfl⟩ := hrest
      rfl
    calc (bad j).card ≤ ((univ : Finset (Fin k → Fin N)).sigma fib).card := card_le_card_of_injOn _ hmaps hinj
      _ = ∑ c, (fib c).card := card_sigma _ _
      _ ≤ (univ : Finset (Fin k → Fin N)).card • (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1)))) :=
          sum_le_card_nsmul _ _ _ fun c _ => compression_count g P c (revealed c j)
      _ = N ^ k * (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1)))) := by
          rw [card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, smul_eq_mul]
  calc (univ.filter (wipeDistinguisher P)).card ≤ (good ∪ univ.biUnion bad).card := card_le_card hsplit
    _ ≤ good.card + (univ.biUnion bad).card := card_union_le _ _
    _ ≤ good.card + ∑ j, (bad j).card := Nat.add_le_add_left card_biUnion_le _
    _ ≤ 2 ^ q * (2 ^ b) ^ N * g ^ k +
          (univ : Finset (Fin k)).card • (N ^ k * (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))))) :=
        Nat.add_le_add hgood (sum_le_card_nsmul _ _ _ hbad)
    _ = _ := by rw [card_univ, Fintype.card_fin, smul_eq_mul]

end PousStreamedWipe
