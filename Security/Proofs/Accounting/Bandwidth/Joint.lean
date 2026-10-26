import Mathlib
import Definitions.Accounting.Guarantees.Bandwidth
import Proofs.Accounting.Bandwidth.AuditRO

/-!
# Bandwidth adds, by counting (Mbw): the proof

`AuditRO`'s three events, for the pair (store, `z`). The sample and a guessed tag are `AuditRO`'s for each pair. The
compression (`card_compress_le`) describes the pair: the chip's state, the bits it read and which `k` of its queries
named blocks determine, given the oracle, those `k` blocks (replaying the chip recomputes its queries,
`Chip.replay_trace`) and `z` (replaying it recomputes its value, `Chip.replayRun_trace`, and its value is `z`). So the
code has room for `k·b + n` bits less what the device read, not `k·b`.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Chip

variable {N b h : ℕ} {α : Type}

/-- A run's value, recomputed from the oracle and a stream of the bits it was answered with otherwise. -/
def replayRun (O : Oracle N b h) : Chip N b h α → (ℕ → Bool) → α
  | done a, _ => a
  | tag x k, f => (k (O x)).replayRun O f
  | ingress _ k, f => (k (f 0)).replayRun O (fun n => f (n + 1))
  | side _ k, f => (k (f 0)).replayRun O (fun n => f (n + 1))
  | regen _ k, f => (k (fun i => f i)).replayRun O (fun n => f (n + b))

theorem replayRun_trace (w : World N b h) (c : Chip N b h α) (l : List Bool) :
    c.replayRun w.oracle (fun n => (c.trace w ++ l).getD n false) = c.run w := by
  induction c generalizing l with
  | done a => simp [replayRun, run]
  | tag x k ih => simp only [replayRun, run, trace, ih]
  | ingress a k ih => simpa [replayRun, run, trace] using ih _ l
  | side a k ih => simpa [replayRun, run, trace] using ih _ l
  | regen j k ih =>
    simp only [replayRun, run, trace, List.append_assoc]
    have hb : (fun i : Fin b => (List.ofFn (w.block j) ++ ((k (w.block j)).trace w ++ l)).getD i false) =
        w.block j := by
      funext i
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp)]
      simp
    have hs : (fun n => (List.ofFn (w.block j) ++ ((k (w.block j)).trace w ++ l)).getD (n + b) false) =
        fun n => ((k (w.block j)).trace w ++ l).getD n false := by
      funext n
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.getD_eq_getElem?_getD]
      simp
    rw [hb, hs, ih]

end Chip

namespace JointDevice

variable {N b h S c G q n : ℕ}

/-- The blocks whose contents the chip queried the oracle at. -/
def queried (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (O : Oracle N b h) : Finset (Fin N) :=
  Finset.univ.filter (fun j => (j, E j) ∈ (A.chip (A.state E z)).queries (A.world E z O))

end JointDevice

namespace AuditJoint

open JointDevice AuditRO

variable {N b h S c G q n : ℕ}

/-! ## A guessed tag -/

theorem card_guess_le (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (j : Fin N)
    [DecidablePred (fun O : Oracle N b h => j ∉ A.queried E z O ∧ (A.out E z O).1 j = O (j, E j))] :
    (Finset.univ.filter (fun O : Oracle N b h => j ∉ A.queried E z O ∧ (A.out E z O).1 j = O (j, E j))).card *
        2 ^ h ≤ Fintype.card (Oracle N b h) := by
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (j, E j) _ (fun O₁ h₁ O₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, JointDevice.queried] at h₁ h₂
  have hq : ∀ x ∈ (A.chip (A.state E z)).queries (A.world E z O₁), O₂ x = (A.world E z O₁).oracle x :=
    fun x hx => (hoff x (fun he => h₁.1 (by simpa [he] using hx))).symm
  have ht : (A.out E z O₂).1 j = (A.out E z O₁).1 j :=
    congrFun (congrArg Prod.fst (Chip.run_congr (A.world E z O₁) O₂ _ hq).1) j
  funext x
  by_cases hx : x = (j, E j)
  · subst hx
    rw [← h₁.2, ← h₂.2, ht]
  · exact hoff x hx

/-! ## Compression of the pair -/

/-- The queries a code (the chip's state, a padded trace and a set of query positions) replays to. -/
def decoded (A : JointDevice N b h S c G q n) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    List (Fin N × Bits b) :=
  (A.chip code.1).replay O (stream r code.2.1)

/-- The value a code replays to. -/
def value (A : JointDevice N b h S c G q n) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    Bits n :=
  ((A.chip code.1).replayRun O (stream r code.2.1)).2

open Classical in
/-- The pairs a code describes: its value, and stores holding, at each named position's query, its block. -/
noncomputable def fib (A : JointDevice N b h S c G q n) (O : Oracle N b h) (r : ℕ)
    (code : Bits S × Bits r × Finset ℕ) : Finset (Store N b × Bits n) :=
  Finset.univ.filter (fun x => x.2 = value A O r code ∧
    ∀ p ∈ code.2.2, ∀ y, (decoded A O r code)[p]? = some y → x.1 y.1 = y.2)

open Classical in
/-- The blocks a code names. -/
noncomputable def named (A : JointDevice N b h S c G q n) (O : Oracle N b h) (r : ℕ)
    (code : Bits S × Bits r × Finset ℕ) : Finset (Fin N) :=
  Finset.univ.filter (fun j => ∃ p ∈ code.2.2, ∃ y, (decoded A O r code)[p]? = some y ∧ y.1 = j)

theorem card_fib (A : JointDevice N b h S c G q n) (O : Oracle N b h) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    (fib A O r code).card ≤ (2 ^ b) ^ (N - (named A O r code).card) := by
  classical
  calc (fib A O r code).card ≤ (Finset.univ : Finset (↥(named A O r code)ᶜ → Bits b)).card := by
        refine Finset.card_le_card_of_injOn (fun x j => x.1 j) (fun _ _ => Finset.mem_univ _) ?_
        intro x₁ h₁ x₂ h₂ he
        simp only [fib, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
        refine Prod.ext (funext fun j => ?_) (h₁.1.trans h₂.1.symm)
        by_cases hj : j ∈ named A O r code
        · simp only [named, Finset.mem_filter, Finset.mem_univ, true_and] at hj
          obtain ⟨p, hp, y, hy, rfl⟩ := hj
          rw [h₁.2 p hp y hy, h₂.2 p hp y hy]
        · exact congrFun he ⟨j, by simpa using hj⟩
    _ = (2 ^ b) ^ (N - (named A O r code).card) := by
        simp

theorem card_compress_le (A : JointDevice N b h S c G q n) (hA : A.Recovers) (O : Oracle N b h) (k lam : ℕ)
    [DecidablePred (fun x : Store N b × Bits n =>
      k ≤ (A.queried x.1 x.2 O).card ∧ A.ingress x.1 x.2 O + S + c + G * b + lam < k * b + n)] :
    (Finset.univ.filter (fun x : Store N b × Bits n =>
        k ≤ (A.queried x.1 x.2 O).card ∧ A.ingress x.1 x.2 O + S + c + G * b + lam < k * b + n)).card * 2 ^ lam ≤
      q.choose k * Fintype.card (Store N b × Bits n) := by
  classical
  by_cases hpar : S + lam ≤ k * b + n ∧ k ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun x _ ⟨hk, hlt⟩ =>
      hpar ⟨by omega, hk.trans ((Finset.card_le_univ _).trans (by simp))⟩)]
    simp
  obtain ⟨hSl, hkN⟩ := hpar
  set r := k * b + n - S - lam with hr
  let codes : Finset (Bits S × Bits r × Finset ℕ) := Finset.univ ×ˢ Finset.univ ×ˢ (Finset.range q).powersetCard k
  have hsub : Finset.univ.filter (fun x : Store N b × Bits n =>
        k ≤ (A.queried x.1 x.2 O).card ∧ A.ingress x.1 x.2 O + S + c + G * b + lam < k * b + n) ⊆
      (codes.filter (fun code => k ≤ (named A O r code).card)).biUnion (fib A O r) := by
    rintro ⟨E, z⟩ hE
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hE
    obtain ⟨hk, hlow⟩ := hE
    obtain ⟨hside, hreg, hlen⟩ := A.bounded (A.state E z) (A.world E z O)
    have hT : ((A.chip (A.state E z)).trace (A.world E z O)).length < r := by
      have h1 := Chip.trace_length (A.world E z O) (A.chip (A.state E z))
      have h2 : (A.chip (A.state E z)).regens (A.world E z O) * b ≤ G * b := Nat.mul_le_mul_right _ hreg
      have h3 : A.ingress E z O = (A.chip (A.state E z)).ingressBits (A.world E z O) := rfl
      omega
    obtain ⟨Kk, hKk, hKkcard⟩ := Finset.exists_subset_card_eq hk
    set qs := (A.chip (A.state E z)).queries (A.world E z O) with hqs
    have hmem : ∀ j ∈ Kk, (j, E j) ∈ qs := fun j hj => by simpa [JointDevice.queried] using hKk hj
    have hpos : ∀ j ∈ Kk, qs[qs.idxOf (j, E j)]? = some (j, E j) := fun j hj => by
      have hlt : qs.idxOf (j, E j) < qs.length := List.idxOf_lt_length_of_mem (hmem j hj)
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    let bits : Bits r := fun i => ((A.chip (A.state E z)).trace (A.world E z O)).getD i false
    have hstream : stream r bits = fun m => ((A.chip (A.state E z)).trace (A.world E z O) ++ []).getD m false := by
      funext m
      simp only [stream, bits, List.append_nil]
      split_ifs with hm
      · rfl
      · rw [List.getD_eq_default _ _ (by omega)]
    let code := (A.state E z, bits, Kk.image (fun j => qs.idxOf (j, E j)))
    have hdec : decoded A O r code = qs := by
      simp only [decoded, code, hstream]
      exact Chip.replay_trace (A.world E z O) (A.chip (A.state E z)) []
    have hval : value A O r code = z := by
      simp only [value, code, hstream]
      exact (congrArg Prod.snd (Chip.replayRun_trace (A.world E z O) (A.chip (A.state E z)) [])).trans (hA E z O)
    refine Finset.mem_biUnion.2 ⟨code, ?_, ?_⟩
    · refine Finset.mem_filter.2 ⟨?_, ?_⟩
      · simp only [codes, code, Finset.mem_product, Finset.mem_univ, true_and, Finset.mem_powersetCard]
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
    · simp only [fib, Finset.mem_filter, Finset.mem_univ, true_and, hdec, hval]
      intro p hp y hy
      obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hp
      rw [hpos j hj] at hy
      cases hy
      rfl
  have hcodes : codes.card = 2 ^ S * (2 ^ r * q.choose k) := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hpair : Fintype.card (Store N b × Bits n) = (2 ^ b) ^ N * 2 ^ n := by simp
  calc (Finset.univ.filter (fun x : Store N b × Bits n =>
          k ≤ (A.queried x.1 x.2 O).card ∧ A.ingress x.1 x.2 O + S + c + G * b + lam < k * b + n)).card * 2 ^ lam
      ≤ (codes.card * (2 ^ b) ^ (N - k)) * 2 ^ lam := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        refine (Finset.sum_le_card_nsmul _ _ ((2 ^ b) ^ (N - k)) (fun code hc => ?_)).trans ?_
        · have := (Finset.mem_filter.1 hc).2
          exact (card_fib A O r code).trans (Nat.pow_le_pow_right (by positivity) (by omega))
        · rw [smul_eq_mul]
          exact Nat.mul_le_mul_right _ (Finset.card_filter_le _ _)
    _ = q.choose k * Fintype.card (Store N b × Bits n) := by
        have e1 : S + r + lam = k * b + n := by omega
        have e2 : (2 : ℕ) ^ (k * b) * (2 ^ b) ^ (N - k) = (2 ^ b) ^ N := by
          rw [mul_comm k b, pow_mul, ← pow_add, Nat.add_sub_cancel' hkN]
        rw [hcodes, hpair, ← e2]
        have e3 : (2 : ℕ) ^ S * 2 ^ r * 2 ^ lam = 2 ^ (k * b) * 2 ^ n := by
          rw [← pow_add, ← pow_add, ← pow_add, e1]
        calc 2 ^ S * (2 ^ r * q.choose k) * (2 ^ b) ^ (N - k) * 2 ^ lam
            = q.choose k * (2 ^ S * 2 ^ r * 2 ^ lam) * (2 ^ b) ^ (N - k) := by ring
          _ = q.choose k * (2 ^ (k * b) * (2 ^ b) ^ (N - k) * 2 ^ n) := by rw [e3]; ring

end AuditJoint

end Bandwidth

namespace Bandwidth.SecurityProofs

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth.AuditRO Bandwidth.AuditJoint Bandwidth.JointDevice

/-- **Bandwidth adds, by counting (Mbw), in the random-oracle model** (`Bandwidth.Guarantees.AuditJoint`). -/
theorem AuditJoint : Bandwidth.Guarantees.AuditJoint := by
  classical
  intro N b h S c G q n s k lam A hA
  have hsplit : ∀ x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N),
      A.Accepts x.1.1 x.1.2 x.2.1 x.2.2 ∧ A.ingress x.1.1 x.1.2 x.2.1 + S + c + G * b + lam < k * b + n →
      (((A.queried x.1.1 x.1.2 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ A.queried x.1.1 x.1.2 x.2.1) ∨
        (∃ i, x.2.2 i ∉ A.queried x.1.1 x.1.2 x.2.1 ∧
          (A.out x.1.1 x.1.2 x.2.1).1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i)))) ∨
      (k ≤ (A.queried x.1.1 x.1.2 x.2.1).card ∧
        A.ingress x.1.1 x.1.2 x.2.1 + S + c + G * b + lam < k * b + n) := by
    rintro ⟨⟨E, z⟩, O, I⟩ ⟨hacc, hlow⟩
    dsimp only at hacc hlow ⊢
    by_cases hk : k ≤ (A.queried E z O).card
    · exact Or.inr ⟨hk, hlow⟩
    · by_cases hall : ∀ i, I i ∈ A.queried E z O
      · exact Or.inl (Or.inl ⟨by omega, hall⟩)
      · push Not at hall
        obtain ⟨i, hi⟩ := hall
        exact Or.inl (Or.inr ⟨i, hi, hacc i⟩)
  have hΩ : Fintype.card ((Store N b × Bits n) × Oracle N b h × (Fin s → Fin N)) =
      Fintype.card (Store N b × Bits n) * (Fintype.card (Oracle N b h) * N ^ s) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)
  · -- the sample misses the blocks never queried
    have hc := pr_le_of_card (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (A.queried x.1.1 x.1.2 x.2.1).card < k ∧ ∀ i, x.2.2 i ∈ A.queried x.1.1 x.1.2 x.2.1) (N ^ s) ((k - 1) ^ s)
        hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _ (Fintype.card (Oracle N b h) * ((k - 1) ^ s * N ^ s))
        (fun x => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (A.queried x.1 x.2 O).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- a guessed tag
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        x.2.2 i ∉ A.queried x.1.1 x.1.2 x.2.1 ∧
          (A.out x.1.1 x.1.2 x.2.1).1 (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ≤
        ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun i =>
      pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _ (Fintype.card (Oracle N b h) * Fintype.card (Fin s → Fin N))
          (fun x => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le' _ _ _ (fun I => ?_)
        exact card_guess_le A x.1 x.2 (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    simp [div_eq_mul_inv]
  · -- compression of the pair
    refine (pr_le_of_card _ (2 ^ lam) (q.choose k) (by positivity) (by
      rw [hΩ]
      refine (card_filter_prod_le' _ _ (q.choose k * Fintype.card (Store N b × Bits n)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact card_compress_le A hA y.1 k lam)).trans (le_of_eq ?_)
    push_cast
    rfl

end Bandwidth.SecurityProofs
