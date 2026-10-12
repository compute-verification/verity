import Mathlib
import Models.Accounting.Guarantees.RateAuditPartial
import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.PartialChip

/-!
# The rate audit with partial completion: the proof

`Audit.lean`'s argument, with completions in place of rebuilds. An accepted run whose compute is within `F` and whose
footprint is short of the floor at `c_enc = rounds · c_root` falls in one of three events, by whether the device queried
the oracle, under the certified stream, at the contents of at least `k` words (`queried`):

- fewer than `k`, and every sampled word among them: the sample's chance, `((k − 1)/N)^s`;
- a sampled word it never queried, with the right tag anyway: `2^-h` per sampled word (`card_guess_le`);
- at least `k`: then the run's state, footprint, link bits and completed bits (`taken`) fall short of the stream and
  those `k` words by `λ` bits (`taken_lt_of_floor`), because tagging `k` words leaves `F − k·c_tag` of the compute for
  completions, and under `WorkHardness` that completes at most `(F − k·c_tag)·b/(rounds · c_root)` bits
  (`PChip.completed_mul_le`). That happens for at most a `C(q, k)·2^-λ` fraction of stores and streams, by compression
  (`card_compress_le`): the trace records `x` bits per completion of `x` (`PChip.trace_length`), and replaying the chip
  recomputes its queries (`PChip.replay_trace`).

The honest device is `RateHonest`'s, as a partial device (`Device.toPartial`).
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_line_le)

variable {N b h ρ S L q : ℕ}

namespace PartialDevice

/-- The words whose contents the device queried the oracle at, under the certified stream. -/
def queried (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : Finset (Fin N) :=
  Finset.univ.filter (fun j => (R, j, E j) ∈ (A.chip (A.state E)).queries (A.world E R O))

/-- What a run had besides the oracle, in bits: its on-chip state, its footprint, the link's bits and the bits it
completed. -/
def taken (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  A.footprint E R O + S + (A.chip (A.state E)).linkBits (A.world E R O) + A.completed E R O

theorem card_queried_le (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (A.queried E R O).card ≤ ((A.chip (A.state E)).queries (A.world E R O)).length := by
  classical
  refine (Finset.card_le_card_of_injOn (fun j => (R, j, E j)) (fun j hj => ?_) ?_).trans (List.toFinset_card_le _)
  · simpa [queried] using hj
  · intro j₁ _ j₂ _ he
    exact (Prod.mk.inj (Prod.mk.inj he).2).1

/-- A run that queried at least `k` words within the compute budget `F` and whose footprint is short of the floor at
`c_enc = rounds · c_root` had less than the stream and `k` words: tagging `k` words leaves `F − k·c_tag` for
completions, which under `WorkHardness` complete at most `(F − k·c_tag)·b/(rounds · c_root)` bits. -/
theorem taken_lt_of_floor (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ)
    {k F cTag rounds cRoot lam : ℕ} {cost : ℕ → ℕ} (hw : Assumptions.WorkHardness b rounds cRoot cost)
    (hk : k ≤ (A.queried E R O).card) (hc : A.compute cTag cost E R O ≤ F)
    (hf : A.footprint E R O * (rounds * cRoot) < floor b ρ S L F cTag (rounds * cRoot) k lam) :
    A.taken E R O + lam < k * b + ρ := by
  have hQ := hk.trans (card_queried_le A E R O)
  have hL := (A.bounded (A.state E) (A.world E R O)).1
  have hG := PChip.completed_mul_le hw (A.world E R O) (A.chip (A.state E))
  unfold PartialDevice.compute at hc
  unfold floor at hf
  unfold taken PartialDevice.completed
  generalize rounds * cRoot = cEnc at hf hG
  generalize ((A.chip (A.state E)).queries (A.world E R O)).length = Q at hQ hc
  generalize (A.chip (A.state E)).work cost (A.world E R O) = Wk at hc hG
  generalize (A.chip (A.state E)).completed (A.world E R O) = G at hG ⊢
  generalize (A.chip (A.state E)).linkBits (A.world E R O) = lb at hL ⊢
  generalize A.footprint E R O = fp at hf ⊢
  have h0 := Nat.lt_sub_iff_add_lt.1 hf
  have h1 : Wk ≤ F - k * cTag := by
    have := Nat.mul_le_mul_right cTag hQ
    omega
  have h2 : G * cEnc ≤ (F - k * cTag) * b :=
    calc G * cEnc ≤ b * Wk := hG
      _ ≤ b * (F - k * cTag) := Nat.mul_le_mul_left b h1
      _ = (F - k * cTag) * b := by ring
  have h3 : (fp + S + (L - ρ) + lam + G) * cEnc < k * b * cEnc := by
    have e : (fp + S + (L - ρ) + lam + G) * cEnc = fp * cEnc + (S + (L - ρ) + lam) * cEnc + G * cEnc := by ring
    omega
  have h4 := Nat.lt_of_mul_lt_mul_right h3
  omega

/-! ## A guessed tag -/

/-- A word the device never queried has the right tag for at most a `2^-h` fraction of oracles. -/
theorem card_guess_le (A : PartialDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ) (j : Fin N)
    [DecidablePred (fun O : Oracle N b h ρ => j ∉ A.queried E R O ∧ A.tags E R O j = O (R, j, E j))] :
    (Finset.univ.filter (fun O : Oracle N b h ρ => j ∉ A.queried E R O ∧ A.tags E R O j = O (R, j, E j))).card *
        2 ^ h ≤ Fintype.card (Oracle N b h ρ) := by
  classical
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  rw [← hY]
  refine card_line_le (R, j, E j) _ (fun O₁ h₁ O₂ h₂ hoff => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, queried] at h₁ h₂
  have hq : ∀ x ∈ (A.chip (A.state E)).queries (A.world E R O₁), O₂ x = (A.world E R O₁).oracle x :=
    fun x hx => (hoff x (fun he => h₁.1 (he ▸ hx))).symm
  have hw : A.world E R O₂ = { A.world E R O₁ with oracle := O₂ } := rfl
  have ht : A.tags E R O₂ j = A.tags E R O₁ j := by
    simp only [PartialDevice.tags]
    rw [hw]
    exact congrFun (PChip.run_congr (A.world E R O₁) O₂ _ hq).1 j
  funext x
  by_cases hx : x = (R, j, E j)
  · subst hx
    rw [← h₁.2, ← h₂.2, ht]
  · exact hoff x hx

/-! ## Compression -/

/-- The queries a code (the chip's state, a padded trace and a set of query positions) replays to. -/
def decoded (A : PartialDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ) (code : Bits S × Bits r × Finset ℕ) :
    List (Bits ρ × Fin N × Bits b) :=
  (A.chip code.1).replay O (fun _ => none) (Bandwidth.AuditRO.stream r code.2.1)

open Classical in
/-- The stores and streams a code describes: those whose stream is each named position's query's, and whose word
there holds the query's contents. -/
noncomputable def fib (A : PartialDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ)
    (code : Bits S × Bits r × Finset ℕ) : Finset (Store N b × Bits ρ) :=
  Finset.univ.filter (fun x => ∀ p ∈ code.2.2, ∀ y, (A.decoded O r code)[p]? = some y → y.1 = x.2 ∧ x.1 y.2.1 = y.2.2)

open Classical in
/-- The words a code names. -/
noncomputable def named (A : PartialDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ)
    (code : Bits S × Bits r × Finset ℕ) : Finset (Fin N) :=
  Finset.univ.filter (fun j => ∃ p ∈ code.2.2, ∃ y, (A.decoded O r code)[p]? = some y ∧ y.2.1 = j)

/-- A code that names a word fixes the stream, and the words it names. -/
theorem card_fib (A : PartialDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ) (code : Bits S × Bits r × Finset ℕ)
    (hn : 0 < (A.named O r code).card) :
    (A.fib O r code).card ≤ (2 ^ b) ^ (N - (A.named O r code).card) := by
  classical
  obtain ⟨j₀, hj₀⟩ := Finset.card_pos.1 hn
  simp only [named, Finset.mem_filter, Finset.mem_univ, true_and] at hj₀
  obtain ⟨p₀, hp₀, y₀, hy₀, -⟩ := hj₀
  calc (A.fib O r code).card ≤ (Finset.univ : Finset (↥(A.named O r code)ᶜ → Bits b)).card := by
        refine Finset.card_le_card_of_injOn (fun x j => x.1 j) (fun _ _ => Finset.mem_univ _) ?_
        intro x₁ h₁ x₂ h₂ he
        simp only [fib, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
        refine Prod.ext (funext fun j => ?_) ((h₁ p₀ hp₀ y₀ hy₀).1.symm.trans (h₂ p₀ hp₀ y₀ hy₀).1)
        by_cases hj : j ∈ A.named O r code
        · simp only [named, Finset.mem_filter, Finset.mem_univ, true_and] at hj
          obtain ⟨p, hp, y, hy, rfl⟩ := hj
          rw [(h₁ p hp y hy).2, (h₂ p hp y hy).2]
        · exact congrFun he ⟨j, by simpa using hj⟩
    _ = (2 ^ b) ^ (N - (A.named O r code).card) := by
        simp

theorem card_compress_le (A : PartialDevice N b h ρ S L q) (O : Oracle N b h ρ) (k lam : ℕ) (hk : 0 < k)
    [DecidablePred (fun x : Store N b × Bits ρ =>
      k ≤ (A.queried x.1 x.2 O).card ∧ A.taken x.1 x.2 O + lam < k * b + ρ)] :
    (Finset.univ.filter (fun x : Store N b × Bits ρ =>
        k ≤ (A.queried x.1 x.2 O).card ∧ A.taken x.1 x.2 O + lam < k * b + ρ)).card * 2 ^ lam ≤
      q.choose k * Fintype.card (Store N b × Bits ρ) := by
  classical
  by_cases hpar : S + lam ≤ k * b + ρ ∧ k ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun x _ ⟨hk', hlt⟩ =>
      hpar ⟨by unfold taken at hlt; omega, hk'.trans ((Finset.card_le_univ _).trans (by simp))⟩)]
    simp
  obtain ⟨hSl, hkN⟩ := hpar
  set r := k * b + ρ - S - lam with hr
  let codes : Finset (Bits S × Bits r × Finset ℕ) := Finset.univ ×ˢ Finset.univ ×ˢ (Finset.range q).powersetCard k
  have hsub : Finset.univ.filter (fun x : Store N b × Bits ρ =>
        k ≤ (A.queried x.1 x.2 O).card ∧ A.taken x.1 x.2 O + lam < k * b + ρ) ⊆
      (codes.filter (fun code => k ≤ (A.named O r code).card)).biUnion (A.fib O r) := by
    rintro ⟨E, R⟩ hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
    obtain ⟨hkq, hlow⟩ := hx
    obtain ⟨-, hlen⟩ := A.bounded (A.state E) (A.world E R O)
    have hT : ((A.chip (A.state E)).trace (A.world E R O) ∅).length < r := by
      have h1 := PChip.trace_length (A.world E R O) (A.chip (A.state E)) ∅
      simp only [Finset.sdiff_empty] at h1
      simp only [taken, PartialDevice.footprint, PartialDevice.completed] at hlow
      omega
    obtain ⟨Kk, hKk, hKkcard⟩ := Finset.exists_subset_card_eq hkq
    set qs := (A.chip (A.state E)).queries (A.world E R O) with hqs
    have hmem : ∀ j ∈ Kk, (R, j, E j) ∈ qs := fun j hj => by simpa [queried] using hKk hj
    have hpos : ∀ j ∈ Kk, qs[qs.idxOf (R, j, E j)]? = some (R, j, E j) := fun j hj => by
      have hlt : qs.idxOf (R, j, E j) < qs.length := List.idxOf_lt_length_of_mem (hmem j hj)
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    let bits : Bits r := fun i => ((A.chip (A.state E)).trace (A.world E R O) ∅).getD i false
    have hdec : A.decoded O r (A.state E, bits, Kk.image (fun j => qs.idxOf (R, j, E j))) = qs := by
      have hstream : Bandwidth.AuditRO.stream r bits =
          fun n => ((A.chip (A.state E)).trace (A.world E R O) ∅ ++ []).getD n false := by
        funext n
        simp only [Bandwidth.AuditRO.stream, bits, List.append_nil]
        split_ifs with hn
        · rfl
        · rw [List.getD_eq_default _ _ (by omega)]
      simp only [decoded, hstream]
      exact PChip.replay_trace (A.world E R O) (A.chip (A.state E)) ∅ (fun _ => none) (fun a => by simp) []
    refine Finset.mem_biUnion.2 ⟨(A.state E, bits, Kk.image (fun j => qs.idxOf (R, j, E j))), ?_, ?_⟩
    · refine Finset.mem_filter.2 ⟨?_, ?_⟩
      · simp only [codes, Finset.mem_product, Finset.mem_univ, true_and, Finset.mem_powersetCard]
        refine ⟨fun p hp => ?_, ?_⟩
        · obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hp
          exact Finset.mem_range.2 ((List.idxOf_lt_length_of_mem (hmem j hj)).trans_le hlen)
        · rw [Finset.card_image_of_injOn, hKkcard]
          intro j₁ h₁ j₂ h₂ he
          have h12 := (hpos j₁ h₁).symm.trans ((congrArg (fun p => qs[p]?) he).trans (hpos j₂ h₂))
          exact (Prod.mk.inj (Prod.mk.inj (Option.some.inj h12)).2).1
      · rw [← hKkcard]
        refine Finset.card_le_card (fun j hj => ?_)
        simp only [named, Finset.mem_filter, Finset.mem_univ, true_and, hdec]
        exact ⟨_, Finset.mem_image_of_mem _ hj, (R, j, E j), hpos j hj, rfl⟩
    · simp only [fib, Finset.mem_filter, Finset.mem_univ, true_and, hdec]
      intro p hp y hy
      obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hp
      rw [hpos j hj] at hy
      cases hy
      exact ⟨rfl, rfl⟩
  have hcodes : codes.card = 2 ^ S * (2 ^ r * q.choose k) := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hΩ : Fintype.card (Store N b × Bits ρ) = (2 ^ b) ^ N * 2 ^ ρ := by simp
  calc (Finset.univ.filter (fun x : Store N b × Bits ρ =>
          k ≤ (A.queried x.1 x.2 O).card ∧ A.taken x.1 x.2 O + lam < k * b + ρ)).card * 2 ^ lam
      ≤ (codes.card * (2 ^ b) ^ (N - k)) * 2 ^ lam := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        refine (Finset.sum_le_card_nsmul _ _ ((2 ^ b) ^ (N - k)) (fun code hc => ?_)).trans ?_
        · have := (Finset.mem_filter.1 hc).2
          exact (card_fib A O r code (by omega)).trans (Nat.pow_le_pow_right (by positivity) (by omega))
        · rw [smul_eq_mul]
          exact Nat.mul_le_mul_right _ (Finset.card_filter_le _ _)
    _ = q.choose k * Fintype.card (Store N b × Bits ρ) := by
        have e1 : (2 : ℕ) ^ S * 2 ^ r * 2 ^ lam = 2 ^ (k * b) * 2 ^ ρ := by
          rw [← pow_add, ← pow_add, ← pow_add]
          congr 1
          omega
        have e2 : (2 : ℕ) ^ (k * b) * (2 ^ b) ^ (N - k) = (2 ^ b) ^ N := by
          rw [mul_comm k b, pow_mul, ← pow_add, Nat.add_sub_cancel' hkN]
        rw [hcodes, hΩ, ← e2]
        calc 2 ^ S * (2 ^ r * q.choose k) * (2 ^ b) ^ (N - k) * 2 ^ lam
            = q.choose k * (2 ^ S * 2 ^ r * 2 ^ lam) * (2 ^ b) ^ (N - k) := by ring
          _ = q.choose k * (2 ^ (k * b) * (2 ^ b) ^ (N - k) * 2 ^ ρ) := by rw [e1]; ring

end PartialDevice

/-- A device of `RateAudit.lean`'s game as a partial device: its rebuilds are completions of all `b` bits. -/
def Device.toPartial (A : Device N b h ρ S L q) : PartialDevice N b h ρ S L q where
  state := A.state
  mem := A.mem
  side := A.side
  chip st := (A.chip st).toPartial
  bounded st w := by
    obtain ⟨-, -, -, h4, h5, -⟩ := Chip.toPartial_eq w (A.chip st)
    rw [h4, h5]
    exact A.bounded st w

/-- As a partial device, a device commits to the same tags with the same footprint and ingress, completes `b` bits per
rebuilt word and pays `cost b` for each. -/
theorem Device.toPartial_eq (A : Device N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toPartial.tags E R O = A.tags E R O ∧ A.toPartial.footprint E R O = A.footprint E R O ∧
      A.toPartial.ingress E R O = A.ingress E R O ∧
      A.toPartial.completed E R O = (A.chip (A.state E)).rebuilds (A.world E R O) * b ∧
      ∀ (cTag : ℕ) (cost : ℕ → ℕ), A.toPartial.compute cTag cost E R O =
        ((A.chip (A.state E)).queries (A.world E R O)).length * cTag +
          (A.chip (A.state E)).rebuilds (A.world E R O) * cost b := by
  obtain ⟨e1, e2, e3, -, e5, e6⟩ := Chip.toPartial_eq (A.world E R O) (A.chip (A.state E))
  refine ⟨e1, congrArg Finset.card e3, e2, e6, fun cTag cost => ?_⟩
  have ew := Chip.toPartial_work cost (A.world E R O) (A.chip (A.state E))
  change ((A.chip (A.state E)).toPartial.queries (A.world E R O)).length * cTag +
      (A.chip (A.state E)).toPartial.work cost (A.world E R O) = _
  rw [e5, ew]

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le
  pr_eq_zero_of_card)
open Bandwidth.AuditRO (card_sample)

/-- **The rate audit with partial completion, in the random-oracle model** (`Accounting.Guarantees.RateAuditPartial`). -/
theorem RateAuditPartial : Accounting.Guarantees.RateAuditPartial := by
  intro N b h ρ S L q s k F cTag rounds cRoot lam cost A hw
  classical
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp [floor] at hx), Finset.card_empty, zero_mul, zero_mul]
  have hsplit : ∀ x : Store N b × Coins N b h ρ s,
      (A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cost x.1 x.2.1 x.2.2.1 ≤ F ∧
        A.footprint x.1 x.2.1 x.2.2.1 * (rounds * cRoot) < floor b ρ S L F cTag (rounds * cRoot) k lam) →
      (((A.queried x.1 x.2.1 x.2.2.1).card < k ∧ ∀ i, x.2.2.2 i ∈ A.queried x.1 x.2.1 x.2.2.1) ∨
        (∃ i, x.2.2.2 i ∉ A.queried x.1 x.2.1 x.2.2.1 ∧
          A.tags x.1 x.2.1 x.2.2.1 (x.2.2.2 i) = x.2.2.1 (x.2.1, x.2.2.2 i, x.1 (x.2.2.2 i)))) ∨
      (k ≤ (A.queried x.1 x.2.1 x.2.2.1).card ∧ A.taken x.1 x.2.1 x.2.2.1 + lam < k * b + ρ) := by
    rintro ⟨E, R, O, I⟩ ⟨hacc, hc, hf⟩
    dsimp only at hacc hc hf ⊢
    by_cases hk : k ≤ (A.queried E R O).card
    · exact Or.inr ⟨hk, A.taken_lt_of_floor E R O hw hk hc hf⟩
    · by_cases hall : ∀ i, I i ∈ A.queried E R O
      · exact Or.inl (Or.inl ⟨by omega, hall⟩)
      · push Not at hall
        obtain ⟨i, hi⟩ := hall
        exact Or.inl (Or.inr ⟨i, hi, hacc i⟩)
  have hΩ : Fintype.card (Store N b × Coins N b h ρ s) =
      Fintype.card (Store N b) * (Fintype.card (Bits ρ) * (Fintype.card (Oracle N b h ρ) * N ^ s)) := by
    simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]
  by_cases hN : N ^ s = 0
  · rw [pr_eq_zero_of_card _ (by rw [hΩ, hN]; simp)]
    exact bot_le
  refine (pr_mono hsplit).trans ?_
  refine (pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add ?_ ?_)) ?_)
  · -- the sample misses the words never queried
    have hc := pr_le_of_card (fun x : Store N b × Coins N b h ρ s =>
        (A.queried x.1 x.2.1 x.2.2.1).card < k ∧ ∀ i, x.2.2.2 i ∈ A.queried x.1 x.2.1 x.2.2.1) (N ^ s)
        ((k - 1) ^ s) hN (by
      rw [hΩ]
      refine (card_filter_prod_le _ _
        (Fintype.card (Bits ρ) * (Fintype.card (Oracle N b h ρ) * ((k - 1) ^ s * N ^ s)))
        (fun E => ?_)).trans (le_of_eq (by ring))
      refine card_filter_prod_le _ _ _ (fun R => ?_)
      refine card_filter_prod_le _ _ _ (fun O => ?_)
      by_cases hk : (A.queried E R O).card < k
      · simp only [hk, true_and]
        rw [card_sample]
        exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_left (by omega) _)
      · simp [hk])
    refine hc.trans (le_of_eq ?_)
    rw [Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, div_eq_mul_inv, mul_pow, ENNReal.inv_pow]
  · -- a guessed tag
    refine (pr_exists_le _).trans ?_
    have hi : ∀ i : Fin s, pr (fun x : Store N b × Coins N b h ρ s =>
        x.2.2.2 i ∉ A.queried x.1 x.2.1 x.2.2.1 ∧
          A.tags x.1 x.2.1 x.2.2.1 (x.2.2.2 i) = x.2.2.1 (x.2.1, x.2.2.2 i, x.1 (x.2.2.2 i))) ≤
        ((1 : ℕ) : ℝ≥0∞) / ((2 ^ h : ℕ) : ℝ≥0∞) := fun i =>
      pr_le_of_card _ (2 ^ h) 1 (by positivity) (by
        rw [hΩ, one_mul]
        refine (card_filter_prod_le _ _
          (Fintype.card (Bits ρ) * (Fintype.card (Oracle N b h ρ) * Fintype.card (Fin s → Fin N)))
          (fun E => ?_)).trans (le_of_eq (by simp))
        refine card_filter_prod_le _ _ _ (fun R => ?_)
        refine card_filter_prod_le' _ _ _ (fun I => ?_)
        exact A.card_guess_le E R (I i))
    refine (Finset.sum_le_sum (fun i _ => hi i)).trans (le_of_eq ?_)
    simp [div_eq_mul_inv]
  · -- compression
    refine (pr_le_of_card _ (2 ^ lam) (q.choose k) (by positivity) (by
      rw [hΩ, card_filter_equiv (Equiv.prodAssoc (Store N b) (Bits ρ) (Oracle N b h ρ × (Fin s → Fin N))).symm]
      refine (card_filter_prod_le' _ _ (q.choose k * Fintype.card (Store N b × Bits ρ)) (fun y => ?_)).trans
        (le_of_eq (by simp only [Fintype.card_prod, Fintype.card_fun, Fintype.card_fin]; ring))
      exact A.card_compress_le y.1 k lam hk0)).trans (le_of_eq ?_)
    push_cast
    rfl

/-- **Completeness of the rate audit with partial completion** (`Accounting.Guarantees.RatePartialHonest`):
`RateHonest`'s device, as a partial device. -/
theorem RatePartialHonest : Accounting.Guarantees.RatePartialHonest := by
  intro N b h ρ
  refine ⟨(Completeness.honest N b h ρ).toPartial, fun cTag cost E R O s I => ?_⟩
  obtain ⟨hacc, hfp, hin, -⟩ := Completeness.honest_accepts (h := h) cTag 0 E R O I
  obtain ⟨-, -, -, -, h5, h6⟩ := Completeness.chip_eq (h := h) ((Completeness.honest N b h ρ).world E R O)
  obtain ⟨t1, t2, t3, t4, t5⟩ := (Completeness.honest N b h ρ).toPartial_eq E R O
  have hc : (Completeness.honest N b h ρ).chip ((Completeness.honest N b h ρ).state E) = Completeness.chip N b h ρ :=
    rfl
  refine ⟨fun i => ?_, t2.trans hfp, t3.trans hin, ?_, ?_⟩
  · rw [t1]
    exact hacc i
  · rw [t4, hc, h5, zero_mul]
  · rw [t5 cTag cost, hc, h5, h6, zero_mul, add_zero]

end Accounting.SecurityProofs
