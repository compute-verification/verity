import Mathlib
import Definitions.Accounting.Guarantees.Bandwidth
import Definitions.Accounting.Guarantees.RemotePass
import Proofs.Accounting.Bandwidth.Chip

/-!
# PoUB's per-pass lemma, the bandwidth audit v0 in the random-oracle model: the proof

An accepted run with low ingress falls in one of three events, by whether the device queried the oracle at the
contents of at least `k` blocks (`queried`):

- fewer than `k`, and every sampled block among them: the sample's chance, `((k − 1)/N)^s` (`card_sample`);
- a sampled block it never queried, with the right tag anyway: `2^-h` per sampled block, because changing the oracle
  at that one point changes neither the run nor the tag the device committed to (`card_guess_le`);
- at least `k`, with low ingress: `C(q, k)·2^-λ`, by compression (`card_compress_le`). The chip's state, the bits
  device memory, side sources and regeneration answered with, and which `k` of its queries named blocks, determine
  those `k` blocks given the oracle, since replaying the chip recomputes its queries (`Chip.replay_trace`).
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

variable {N b h S c G q : ℕ}

/-- The blocks whose contents the device queried the oracle at. -/
def queried (A : Prover N b h S c G q) (E : Store N b) (O : Oracle N b h) : Finset (Fin N) :=
  Finset.univ.filter (fun j => (j, E j) ∈ (A.chip (A.state E)).queries (A.world E O))

namespace AuditRO

/-! ## The sample misses the blocks never queried -/

theorem card_sample (K : Finset (Fin N)) (s : ℕ) [DecidablePred (fun I : Fin s → Fin N => ∀ i, I i ∈ K)] :
    (Finset.univ.filter (fun I : Fin s → Fin N => ∀ i, I i ∈ K)).card = K.card ^ s := by
  have : Finset.univ.filter (fun I : Fin s → Fin N => ∀ i, I i ∈ K) = Fintype.piFinset (fun _ => K) := by
    ext I; simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset]
  simp

/-! ## A guessed tag -/

theorem card_guess_le (A : Prover N b h S c G q) (E : Store N b) (j : Fin N)
    [DecidablePred (fun O : Oracle N b h => j ∉ queried A E O ∧ A.tags E O j = O (j, E j))] :
    (Finset.univ.filter (fun O : Oracle N b h => j ∉ queried A E O ∧ A.tags E O j = O (j, E j))).card * 2 ^ h ≤
      Fintype.card (Oracle N b h) := by
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (j, E j) _ (fun O₁ h₁ O₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, queried] at h₁ h₂
  have hq : ∀ x ∈ (A.chip (A.state E)).queries (A.world E O₁), O₂ x = (A.world E O₁).oracle x :=
    fun x hx => (hoff x (fun he => h₁.1 (by simpa [he] using hx))).symm
  have ht : A.tags E O₂ j = A.tags E O₁ j := congrFun (Chip.run_congr (A.world E O₁) O₂ _ hq).1 j
  funext x
  by_cases hx : x = (j, E j)
  · subst hx
    rw [← h₁.2, ← h₂.2, ht]
  · exact hoff x hx

/-! ## Compression -/

/-- The stream a padded trace replays from. -/
def stream (r : ℕ) (bits : Bits r) (n : ℕ) : Bool := if hn : n < r then bits ⟨n, hn⟩ else false

/-- The queries a code (the chip's state, a padded trace and a set of query positions) replays to. -/
def decoded (A : Prover N b h S c G q) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    List (Fin N × Bits b) :=
  (A.chip code.1).replay O (stream r code.2.1)

open Classical in
/-- The stores a code describes: those holding, at each named position's query, its block. -/
noncomputable def fib (A : Prover N b h S c G q) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    Finset (Store N b) :=
  Finset.univ.filter (fun E => ∀ p ∈ code.2.2, ∀ x, (decoded A O r code)[p]? = some x → E x.1 = x.2)

open Classical in
/-- The blocks a code names. -/
noncomputable def named (A : Prover N b h S c G q) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    Finset (Fin N) :=
  Finset.univ.filter (fun j => ∃ p ∈ code.2.2, ∃ x, (decoded A O r code)[p]? = some x ∧ x.1 = j)

theorem card_fib (A : Prover N b h S c G q) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    (fib A O r code).card ≤ (2 ^ b) ^ (N - (named A O r code).card) := by
  classical
  calc (fib A O r code).card ≤ (Finset.univ : Finset (↥(named A O r code)ᶜ → Bits b)).card := by
        refine Finset.card_le_card_of_injOn (fun E j => E j) (fun _ _ => Finset.mem_univ _) ?_
        intro E₁ h₁ E₂ h₂ he
        simp only [fib, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
        funext j
        by_cases hj : j ∈ named A O r code
        · simp only [named, Finset.mem_filter, Finset.mem_univ, true_and] at hj
          obtain ⟨p, hp, x, hx, rfl⟩ := hj
          rw [h₁ p hp x hx, h₂ p hp x hx]
        · exact congrFun he ⟨j, by simpa using hj⟩
    _ = (2 ^ b) ^ (N - (named A O r code).card) := by
        simp

theorem card_compress_le (A : Prover N b h S c G q) (O : Oracle N b h) (k lam : ℕ)
    [DecidablePred (fun E : Store N b => k ≤ (queried A E O).card ∧ A.ingress E O + S + c + G * b + lam < k * b)] :
    (Finset.univ.filter (fun E : Store N b =>
        k ≤ (queried A E O).card ∧ A.ingress E O + S + c + G * b + lam < k * b)).card * 2 ^ lam ≤
      q.choose k * Fintype.card (Store N b) := by
  classical
  by_cases hpar : S + lam ≤ k * b ∧ k ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun E _ ⟨hk, hlt⟩ =>
      hpar ⟨by omega, hk.trans ((Finset.card_le_univ _).trans (by simp))⟩)]
    simp
  obtain ⟨hSl, hkN⟩ := hpar
  set r := k * b - S - lam with hr
  let codes : Finset (Bits S × Bits r × Finset ℕ) := Finset.univ ×ˢ Finset.univ ×ˢ (Finset.range q).powersetCard k
  have hsub : Finset.univ.filter (fun E : Store N b =>
        k ≤ (queried A E O).card ∧ A.ingress E O + S + c + G * b + lam < k * b) ⊆
      (codes.filter (fun code => k ≤ (named A O r code).card)).biUnion (fib A O r) := by
    intro E hE
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hE
    obtain ⟨hk, hlow⟩ := hE
    obtain ⟨hside, hreg, hlen⟩ := A.bounded (A.state E) (A.world E O)
    have hT : ((A.chip (A.state E)).trace (A.world E O)).length < r := by
      have h1 := Chip.trace_length (A.world E O) (A.chip (A.state E))
      have h2 : (A.chip (A.state E)).regens (A.world E O) * b ≤ G * b := Nat.mul_le_mul_right _ hreg
      simp only [Prover.ingress] at hlow
      omega
    obtain ⟨Kk, hKk, hKkcard⟩ := Finset.exists_subset_card_eq hk
    set qs := (A.chip (A.state E)).queries (A.world E O) with hqs
    have hmem : ∀ j ∈ Kk, (j, E j) ∈ qs := fun j hj => by simpa [queried] using hKk hj
    have hpos : ∀ j ∈ Kk, qs[qs.idxOf (j, E j)]? = some (j, E j) := fun j hj => by
      have hlt : qs.idxOf (j, E j) < qs.length := List.idxOf_lt_length_of_mem (hmem j hj)
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    let bits : Bits r := fun i => ((A.chip (A.state E)).trace (A.world E O)).getD i false
    have hdec : decoded A O r (A.state E, bits, Kk.image (fun j => qs.idxOf (j, E j))) = qs := by
      have hstream : stream r bits = fun n => ((A.chip (A.state E)).trace (A.world E O) ++ []).getD n false := by
        funext n
        simp only [stream, bits, List.append_nil]
        split_ifs with hn
        · rfl
        · rw [List.getD_eq_default _ _ (by omega)]
      simp only [decoded, hstream]
      exact Chip.replay_trace (A.world E O) (A.chip (A.state E)) []
    refine Finset.mem_biUnion.2 ⟨(A.state E, bits, Kk.image (fun j => qs.idxOf (j, E j))), ?_, ?_⟩
    · refine Finset.mem_filter.2 ⟨?_, ?_⟩
      · simp only [codes, Finset.mem_product, Finset.mem_univ, true_and, Finset.mem_powersetCard]
        refine ⟨fun p hp => ?_, ?_⟩
        · obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hp
          exact Finset.mem_range.2 ((List.idxOf_lt_length_of_mem (hmem j hj)).trans_le hlen)
        · rw [Finset.card_image_of_injOn, hKkcard]
          intro j₁ h₁ j₂ h₂ he
          have h12 := (hpos j₁ h₁).symm.trans ((congrArg (fun p => qs[p]?) he).trans (hpos j₂ h₂))
          simp only [Option.some.injEq, Prod.mk.injEq] at h12
          exact h12.1
      · rw [← hKkcard]
        refine Finset.card_le_card (fun j hj => ?_)
        simp only [named, Finset.mem_filter, Finset.mem_univ, true_and, hdec]
        exact ⟨_, Finset.mem_image_of_mem _ hj, (j, E j), hpos j hj, rfl⟩
    · simp only [fib, Finset.mem_filter, Finset.mem_univ, true_and, hdec]
      intro p hp x hx
      obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hp
      rw [hpos j hj] at hx
      cases hx
      rfl
  have hcodes : codes.card = 2 ^ S * (2 ^ r * q.choose k) := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hstore : Fintype.card (Store N b) = (2 ^ b) ^ N := by simp
  calc (Finset.univ.filter (fun E : Store N b =>
          k ≤ (queried A E O).card ∧ A.ingress E O + S + c + G * b + lam < k * b)).card * 2 ^ lam
      ≤ (codes.card * (2 ^ b) ^ (N - k)) * 2 ^ lam := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        refine (Finset.sum_le_card_nsmul _ _ ((2 ^ b) ^ (N - k)) (fun code hc => ?_)).trans ?_
        · have := (Finset.mem_filter.1 hc).2
          exact (card_fib A O r code).trans (Nat.pow_le_pow_right (by positivity) (by omega))
        · rw [smul_eq_mul]
          exact Nat.mul_le_mul_right _ (Finset.card_filter_le _ _)
    _ = q.choose k * Fintype.card (Store N b) := by
        have e1 : S + r + lam = k * b := by omega
        have e2 : (2 : ℕ) ^ (k * b) * (2 ^ b) ^ (N - k) = (2 ^ b) ^ N := by
          rw [mul_comm k b, pow_mul, ← pow_add, Nat.add_sub_cancel' hkN]
        rw [hcodes, hstore, ← e2, ← e1, pow_add, pow_add]
        ring

end AuditRO

theorem pr_eq_zero_of_card {Ω : Type} [Fintype Ω] (P : Ω → Prop) (h : Fintype.card Ω = 0) : pr P = 0 := by
  classical
  unfold pr
  have : (Finset.univ.filter P).card = 0 := Nat.eq_zero_of_le_zero (h ▸ Finset.card_le_univ _)
  simp [this]

end Bandwidth

namespace Bandwidth.SecurityProofs

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth.AuditRO

/-- **PoUB's per-pass lemma, the bandwidth audit v0 in the random-oracle model**
(`Accounting.Guarantees.Remote.AuditRO`). -/
theorem _root_.Accounting.SecurityProofs.Remote.AuditRO : Accounting.Guarantees.Remote.AuditRO := by
  classical
  intro N b h S c G q s k lam A
  have hsplit : ∀ x : Store N b × Oracle N b h × (Fin s → Fin N),
      Accepts A x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < k * b →
      (((queried A x.1 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ queried A x.1 x.2.1) ∨
        (∃ i, x.2.2 i ∉ queried A x.1 x.2.1 ∧ A.tags x.1 x.2.1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i)))) ∨
      (k ≤ (queried A x.1 x.2.1).card ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < k * b) := by
    rintro ⟨E, O, I⟩ ⟨hacc, hlow⟩
    dsimp only at hacc hlow ⊢
    by_cases hk : k ≤ (queried A E O).card
    · exact Or.inr ⟨hk, hlow⟩
    · by_cases hall : ∀ i, I i ∈ queried A E O
      · exact Or.inl (Or.inl ⟨by omega, hall⟩)
      · push Not at hall
        obtain ⟨i, hi⟩ := hall
        exact Or.inl (Or.inr ⟨i, hi, hacc i⟩)
  have hΩ : Fintype.card (Store N b × Oracle N b h × (Fin s → Fin N)) =
      Fintype.card (Store N b) * (Fintype.card (Oracle N b h) * N ^ s) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)
  · -- the sample misses the blocks never queried
    have hc := pr_le_of_card (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (queried A x.1 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ queried A x.1 x.2.1) (N ^ s) ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (Fintype.card (Oracle N b h) * ((k - 1) ^ s * N ^ s))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (queried A E O).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- a guessed tag
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        x.2.2 i ∉ queried A x.1 x.2.1 ∧ A.tags x.1 x.2.1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ≤
        ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun i =>
      pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Oracle N b h) * Fintype.card (Fin s → Fin N))
          (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le' _ _ _ (fun I => ?_)
        exact card_guess_le A E (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    simp [div_eq_mul_inv]
  · -- compression
    refine (pr_le_of_card _ (2 ^ lam) (q.choose k) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose k * Fintype.card (Store N b)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact card_compress_le A y.1 k lam)).trans (le_of_eq ?_)
    push_cast
    rfl

end Bandwidth.SecurityProofs
