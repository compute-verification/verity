module

public import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# `Q_s` by the commit strings a sample reads, the shared ones counted once

`Q_s = Σ_c |Ω_c|/|Ω|` is the factor of the link term (`linkBoundZC`): the expected number of commit strings the sampled
proof blocks read. `qs_le_io` bounds it by the committed wires the sampled proof blocks read or write. A wire is not a
commit string: the bits of one row share a commit string, and a commit string every proof block reads (the constant's,
the zero's) is one string however many proof blocks a sample holds.

* `qs_card_le_split`, `qs_le_split`: when every wire of proof block `u` sits at a commit string of `T u`, at most `r`
  of them, or at one of a set `E` shared by all proof blocks, `Q_s ≤ k_d·r + |E|` plus the share of the samples in a
  set `bad` times `n·r`, where every sample outside `bad` holds at most `k_d` proof blocks. A sample of `bad` may hold
  every proof block.
* `qs_le_common`: the same with no sample in `bad`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession StrictCR
open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ}

section QsSplit

variable {K : Type} [Field K] (A : Arith F K) {C : Verity.Boolean.Circuit} {n : ℕ}
  (P : Audit.Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **The commit strings the samples read, counted**: `Σ_c |Ω_c| ≤ |Ω|·(k_d·r + |E|) + |bad|·n·r`. Each wire of proof
block `u` sits at a commit string of `T u` (at most `r`) or of `E`, a set shared by all proof blocks; a sample outside
`bad` holds at most `k_d` proof blocks, and one in `bad` at most all `n`. -/
theorem qs_card_le_split (L : Law n) (T : Fin n → Finset vb.Pos) (E : Finset vb.Pos) (kd r : ℕ)
    (hT : ∀ u, (T u).card ≤ r) (hpos : ∀ u, ∀ g ∈ P.io u, vb.pos g ∈ T u ∨ vb.pos g ∈ E)
    (bad : Finset L.Ω) (hgood : ∀ ω, ω ∉ bad → (L.draw ω).card ≤ kd) :
    ∑ c : vb.Pos, Fintype.card (ΩcZ A P planZ tab vb L c) ≤
      Fintype.card L.Ω * (kd * r + E.card) + bad.card * (n * r) := by
  classical
  have hω : ∀ ω, (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω).card ≤
      (kd * r + E.card) + (if ω ∈ bad then n * r else 0) := fun ω => by
    have hsub : (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω) ⊆ (L.draw ω).biUnion T ∪ E := by
      intro c hc
      obtain ⟨u, hu, g, hg, rfl⟩ := (Finset.mem_filter.1 hc).2
      rcases hpos u g hg with h | h
      · exact Finset.mem_union_left _ (Finset.mem_biUnion.2 ⟨u, hu, h⟩)
      · exact Finset.mem_union_right _ h
    have hbi : ((L.draw ω).biUnion T).card ≤ (L.draw ω).card * r :=
      Finset.card_biUnion_le.trans ((Finset.sum_le_sum fun u _ => hT u).trans (le_of_eq (by
        rw [Finset.sum_const, smul_eq_mul])))
    have hd : (L.draw ω).card * r ≤ kd * r + (if ω ∈ bad then n * r else 0) := by
      by_cases hb : ω ∈ bad
      · rw [ite_eq_left hb]
        have hn : (L.draw ω).card ≤ n := by
          have := Finset.card_le_univ (L.draw ω)
          rwa [Fintype.card_fin] at this
        exact (Nat.mul_le_mul_right r hn).trans (Nat.le_add_left _ _)
      · rw [ite_eq_right hb, Nat.add_zero]
        exact Nat.mul_le_mul_right r (hgood ω hb)
    calc (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω).card
        ≤ ((L.draw ω).biUnion T ∪ E).card := Finset.card_le_card hsub
      _ ≤ ((L.draw ω).biUnion T).card + E.card := Finset.card_union_le _ _
      _ ≤ (L.draw ω).card * r + E.card := Nat.add_le_add_right hbi _
      _ ≤ (kd * r + (if ω ∈ bad then n * r else 0)) + E.card := Nat.add_le_add_right hd _
      _ = (kd * r + E.card) + (if ω ∈ bad then n * r else 0) := Nat.add_right_comm _ _ _
  calc ∑ c : vb.Pos, Fintype.card (ΩcZ A P planZ tab vb L c)
      = ∑ c : vb.Pos, (Finset.univ.filter fun ω => ReadsDrawZ A P planZ tab vb c ω).card :=
        Finset.sum_congr rfl fun c _ => Fintype.card_of_subtype _ fun ω => by simp
    _ = ∑ ω : L.Ω, (Finset.univ.filter fun c => ReadsDrawZ A P planZ tab vb c ω).card := by
        simp only [Finset.card_filter]
        exact Finset.sum_comm
    _ ≤ ∑ ω : L.Ω, ((kd * r + E.card) + (if ω ∈ bad then n * r else 0)) := Finset.sum_le_sum fun ω _ => hω ω
    _ = Fintype.card L.Ω * (kd * r + E.card) + bad.card * (n * r) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, smul_eq_mul, Finset.sum_ite_mem,
          Finset.univ_inter, Finset.sum_const, smul_eq_mul]

/-- **`Q_s ≤ (k_d·r + |E|) + (|bad|/|Ω|)·n·r`**: `qs_card_le_split` as the average `Q_s = Σ_c |Ω_c|/|Ω|`. The first
term is the samples of at most `k_d` proof blocks, each reading at most `r` commit strings of its own and the `|E|`
shared ones; the second is the samples of `bad`, which may hold every proof block. -/
theorem qs_le_split (L : Law n) (T : Fin n → Finset vb.Pos) (E : Finset vb.Pos) (kd r : ℕ)
    (hT : ∀ u, (T u).card ≤ r) (hpos : ∀ u, ∀ g ∈ P.io u, vb.pos g ∈ T u ∨ vb.pos g ∈ E)
    (bad : Finset L.Ω) (hgood : ∀ ω, ω ∉ bad → (L.draw ω).card ≤ kd) :
    ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω ≤
      ((kd : ℝ) * r + E.card) + (bad.card : ℝ) / Fintype.card L.Ω * ((n : ℝ) * r) := by
  have hsum := qs_card_le_split A P planZ tab vb L T E kd r hT hpos bad hgood
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hcast := (Nat.cast_le (α := ℝ)).2 hsum
  push_cast at hcast
  have hb : (bad.card : ℝ) / Fintype.card L.Ω * Fintype.card L.Ω = bad.card := div_mul_cancel₀ _ hΩ.ne'
  have hmul : (((kd : ℝ) * r + E.card) + (bad.card : ℝ) / Fintype.card L.Ω * ((n : ℝ) * r)) * Fintype.card L.Ω =
      Fintype.card L.Ω * ((kd : ℝ) * r + E.card) + bad.card * ((n : ℝ) * r) := by
    calc (((kd : ℝ) * r + E.card) + (bad.card : ℝ) / Fintype.card L.Ω * ((n : ℝ) * r)) * Fintype.card L.Ω
        = Fintype.card L.Ω * ((kd : ℝ) * r + E.card) +
            ((bad.card : ℝ) / Fintype.card L.Ω * Fintype.card L.Ω) * ((n : ℝ) * r) := by ring
      _ = Fintype.card L.Ω * ((kd : ℝ) * r + E.card) + bad.card * ((n : ℝ) * r) := by rw [hb]
  rw [← Finset.sum_div, div_le_iff₀ hΩ, hmul]
  exact hcast

/-- **`Q_s ≤ k_d·r + |E|`** at samples of at most `k_d` proof blocks: `qs_le_split` with no sample in `bad`. -/
theorem qs_le_common (L : Law n) (T : Fin n → Finset vb.Pos) (E : Finset vb.Pos) (kd r : ℕ)
    (hT : ∀ u, (T u).card ≤ r) (hpos : ∀ u, ∀ g ∈ P.io u, vb.pos g ∈ T u ∨ vb.pos g ∈ E)
    (hd : ∀ ω, (L.draw ω).card ≤ kd) :
    ∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω ≤ (kd : ℝ) * r + E.card := by
  have h := qs_le_split A P planZ tab vb L T E kd r hT hpos ∅ fun ω _ => hd ω
  rw [Finset.card_empty, Nat.cast_zero, zero_div, zero_mul, add_zero] at h
  exact h

end QsSplit

end FlockSoundness.Discharge.ZkLink
