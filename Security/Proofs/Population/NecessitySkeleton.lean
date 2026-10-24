import Definitions.Population.Necessity
import Proofs.Population.Leakage
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Abel

/-!
# The skeleton theorem: tier 1 gives the sampling certificate, and each tier-1 item is needed

**Tier 1** is: (i) the transcript is committed before the coins, and the draw distribution is fixed before the
transcript; (ii) local checks; (iii) robust determinacy, which for opaque units is single assignment and acyclicity
(`Dataflow.determinate_iff_checked`).

* The probability side. `prob_accepts`: a transcript fixed before the coins passes `k` independent draws with
  probability `(1 − π(B))^k`, and `0` when an always-proved unit is wrong.
* `certificate`, **from (i) alone**: for every transcript committed before the coins, the verifier accepts it while
  `B ∩ A ≠ ∅` or `π(B) > ε` with probability at most `(1 − ε)^k`.
* `accepts_local`, **what (ii) buys**: the verdict reads only the always-proved and drawn units' positions.
* `skeleton`, **tier 1 for a population**: with probability at least `1 − (1 − ε)^k`, an accepted run has `B ∩ A = ∅`,
  `π(B) ≤ ε`, and every egress slot where it differs from the honest run inside the wrong units' cone (by (iii)).
  `integrity_inferred`: drawing by cone egress, `π ∝ E`, at most `ε · Σ_u E(u)` egress slots differ.
* `inference_iff`, **tier 2**: a bound `g` on a policy's damage holds at confidence `1 − δ` exactly when every transcript
  with damage above `g` and no always-proved unit wrong passes with probability at most `δ`, that is
  `(1 − π(B))^k ≤ δ`. `inferable_of_amp`: an amplification bound `K` gives the bound `K · C · ε`.
  `not_inferable_of_massless`: a transcript with damage and no draw weight on its wrong units passes with
  probability one, so no bound below its damage is inferable at any confidence.

**The counterexamples, one family per item.**
* (i), a transcript after the coins: `adaptive_dodges`, every draw passes with every undrawn unit wrong
  (`flags_free` meets the hypothesis). A draw after the transcript: `late_draw_dodges`, the point mass on a correct
  unit passes with probability one.
* (ii), a non-local check: `checksum_untestable`, a decision that reads fewer than `n` positions and accepts every
  transcript whose `n` values sum to zero accepts every transcript.
* (iii), a cycle: `Dataflow.cycle_two_fixed_points` and `Dataflow.loop_egress` (POP1). A free value: `free_value`, a
  slot no unit writes, copied to the egress by a correct unit, makes the egress arbitrary with no wrong unit, even with
  every unit always proved. Determinacy that isn't robust: `gate_determinate` and `gate_free`, a population with
  exactly one all-correct assignment in which one wrong unit, writing one slot with no egress in its cone
  (`gate_coneEgress`), makes `k` egress bits arbitrary (`gate_egress`).
-/

namespace Population.Iop

open Finset

variable {U T : Type}

/-! ## The probability side -/

section Prob

variable [Fintype U] (π : U → ℝ) (k : ℕ)

theorem prob_mono (hπ : ∀ u, 0 ≤ π u) {E E' : (Fin k → U) → Prop} (h : ∀ d, E d → E' d) :
    prob π k E ≤ prob π k E' := by
  classical
  unfold prob
  refine sum_le_sum fun d _ => ?_
  by_cases hE : E d
  · simp [hE, h d hE]
  · rw [ite_eq_right hE]
    split_ifs
    · exact prod_nonneg fun i _ => hπ _
    · exact le_rfl

theorem prob_false : prob π k (fun _ => False) = 0 := by
  simp [prob]

theorem prob_congr {E E' : (Fin k → U) → Prop} (h : ∀ d, E d ↔ E' d) : prob π k E = prob π k E' := by
  have : E = E' := funext fun d => propext (h d)
  rw [this]

/-- **Independent draws**: the probability that all `k` draws land in `g` is `π(g)^k`. -/
theorem prob_all (g : U → Prop) [DecidablePred g] :
    prob π k (fun d => ∀ i, g (d i)) = (∑ u ∈ univ.filter g, π u) ^ k := by
  classical
  rw [sum_filter, Fintype.sum_pow]
  unfold prob
  refine sum_congr rfl fun d _ => ?_
  rw [prod_ite_zero]
  simp

theorem mem_bad {chk : U → T → Prop} {σ : T} {u : U} : u ∈ bad chk σ ↔ ¬ chk u σ := by
  classical
  simp [bad]

theorem disjoint_bad_iff (chk : U → T → Prop) (A : Finset U) (σ : T) :
    Disjoint (bad chk σ) A ↔ ∀ u ∈ A, chk u σ := by
  classical
  simp only [bad, disjoint_left, mem_filter, mem_univ, true_and]
  exact ⟨fun h u hu => by_contra fun hc => h hc hu, fun h u hc hu => hc (h u hu)⟩

theorem mass_bad_le_one (hπ : Dist π) (chk : U → T → Prop) (σ : T) : mass π (bad chk σ) ≤ 1 := by
  rw [← hπ.sum_one]
  exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun u _ _ => hπ.nonneg u

open Classical in
theorem sum_good (hπ : Dist π) (chk : U → T → Prop) (σ : T) :
    ∑ u ∈ univ.filter (fun u => chk u σ), π u = 1 - mass π (bad chk σ) := by
  have h := sum_filter_add_sum_filter_not univ (fun u => chk u σ) π
  rw [hπ.sum_one] at h
  have : mass π (bad chk σ) = ∑ u ∈ univ.filter (fun u => ¬ chk u σ), π u := rfl
  rw [this]
  linarith

/-- **A transcript fixed before the coins passes** with probability `(1 − π(B))^k` when no always-proved unit is
wrong. -/
theorem prob_accepts (hπ : Dist π) (chk : U → T → Prop) (A : Finset U) (σ : T) (hd : Disjoint (bad chk σ) A) :
    prob π k (fun d => Accepts chk A d σ) = (1 - mass π (bad chk σ)) ^ k := by
  classical
  have hA := (disjoint_bad_iff chk A σ).1 hd
  rw [← sum_good π hπ chk σ, ← prob_all]
  exact prob_congr π k fun d => ⟨fun h => h.2, fun h => ⟨hA, h⟩⟩

/-- **It never passes when an always-proved unit is wrong.** -/
theorem prob_accepts_eq_zero (chk : U → T → Prop) (A : Finset U) (σ : T) (hd : ¬ Disjoint (bad chk σ) A) :
    prob π k (fun d => Accepts chk A d σ) = 0 := by
  have hA : ¬ ∀ u ∈ A, chk u σ := fun h => hd ((disjoint_bad_iff chk A σ).2 h)
  rw [← prob_false π k]
  exact prob_congr π k fun d => by simp only [Accepts, iff_false, not_and]; exact fun h => absurd h hA

/-- **The sampling certificate**, from tier 1's first item alone: for every transcript committed before the coins
(`σ` does not depend on the draw) and every draw fixed before it (`π`, `A` and `k` do not depend on `σ`), the
verifier accepts while an always-proved unit is wrong or `π(B) > ε` with probability at most `(1 − ε)^k`. A
randomized prover is a mixture of such transcripts, so the bound holds for it too. -/
theorem certificate (hπ : Dist π) (chk : U → T → Prop) (A : Finset U) {ε : ℝ} (hε : ε ≤ 1) (σ : T) :
    prob π k (fun d => Accepts chk A d σ ∧ ¬ Certified chk π A ε σ) ≤ (1 - ε) ^ k := by
  have h1e : 0 ≤ 1 - ε := by linarith
  by_cases hc : Certified chk π A ε σ
  · rw [prob_congr π k (E' := fun _ => False) fun d => by simp [hc], prob_false]
    exact pow_nonneg h1e k
  · refine (prob_mono π k hπ.nonneg fun d h => h.1).trans ?_
    by_cases hd : Disjoint (bad chk σ) A
    · rw [prob_accepts π k hπ chk A σ hd]
      have hm : ε < mass π (bad chk σ) := lt_of_not_ge fun h => hc ⟨hd, h⟩
      exact pow_le_pow_left₀ (by linarith [mass_bad_le_one π hπ chk σ]) (by linarith) k
    · rw [prob_accepts_eq_zero π k chk A σ hd]
      exact pow_nonneg h1e k

/-! ## Tier 2: which damage bounds the certificate supports -/

/-- **Which bounds are inferable**: the damage bound `g` holds at confidence `1 − δ` for every transcript exactly when
every transcript with damage above `g` and no always-proved unit wrong passes the draw with probability
`(1 − π(B))^k ≤ δ`. The best inferable bound is the largest damage among transcripts with
`(1 − π(B))^k > δ`. -/
theorem inference_iff (hπ : Dist π) (chk : U → T → Prop) (A : Finset U) (dmg : T → ℝ) (g δ : ℝ)
    (hδ : 0 ≤ δ) :
    (∀ σ, prob π k (fun d => Accepts chk A d σ ∧ g < dmg σ) ≤ δ) ↔
      ∀ σ, Disjoint (bad chk σ) A → g < dmg σ → (1 - mass π (bad chk σ)) ^ k ≤ δ := by
  constructor
  · intro h σ hd hg
    have := h σ
    rwa [prob_congr π k (E' := fun d => Accepts chk A d σ) fun d => by simp [hg],
      prob_accepts π k hπ chk A σ hd] at this
  · intro h σ
    by_cases hg : g < dmg σ
    · rw [prob_congr π k (E' := fun d => Accepts chk A d σ) fun d => by simp [hg]]
      by_cases hd : Disjoint (bad chk σ) A
      · rw [prob_accepts π k hπ chk A σ hd]
        exact h σ hd hg
      · rw [prob_accepts_eq_zero π k chk A σ hd]
        exact hδ
    · rw [prob_congr π k (E' := fun _ => False) fun d => by simp [hg], prob_false]
      exact hδ

/-- **An amplification bound is a damage bound**: with damage at most `K` times the wrong units' cost, and cost at
most `C` times the draw weight, an accepted transcript has damage above `K C ε` with probability at most
`(1 − ε)^k`. With `π ∝ c`, `C = Σ_u c(u)`. -/
theorem inferable_of_amp (hπ : Dist π) {chk : U → T → Prop} {A : Finset U} {c : U → ℝ} {dmg : T → ℝ} {K C : ℝ}
    (hK : 0 ≤ K) (hC : 0 ≤ C) (hamp : AmpLe chk A c dmg K) (hc : ∀ u, c u ≤ C * π u) {ε : ℝ} (hε : ε ≤ 1)
    (σ : T) : prob π k (fun d => Accepts chk A d σ ∧ K * C * ε < dmg σ) ≤ (1 - ε) ^ k := by
  refine (prob_mono π k hπ.nonneg fun d h => ⟨h.1, fun hcert => ?_⟩).trans (certificate π k hπ chk A hε σ)
  have h1 : ∑ u ∈ bad chk σ, c u ≤ C * mass π (bad chk σ) := by
    unfold mass
    rw [mul_sum]
    exact sum_le_sum fun u _ => hc u
  have h2 : dmg σ ≤ K * C * ε := by
    calc dmg σ ≤ K * ∑ u ∈ bad chk σ, c u := hamp σ hcert.1
      _ ≤ K * (C * mass π (bad chk σ)) := mul_le_mul_of_nonneg_left h1 hK
      _ ≤ K * (C * ε) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hcert.2 hC) hK
      _ = K * C * ε := (mul_assoc _ _ _).symm
  exact absurd h.2 (not_lt.2 h2)

/-- **No bound below a weightless damage is inferable**: a transcript with no always-proved unit wrong and no draw
weight on its wrong units passes with probability one. -/
theorem not_inferable_of_massless (hπ : Dist π) (chk : U → T → Prop) (A : Finset U) (σ : T)
    (hd : Disjoint (bad chk σ) A) (h0 : mass π (bad chk σ) = 0) :
    prob π k (fun d => Accepts chk A d σ) = 1 := by
  rw [prob_accepts π k hπ chk A σ hd, h0]
  simp

end Prob

/-! ## (ii) What local checks buy, and a non-local check -/

/-- **The verdict reads only the always-proved and drawn units' positions** when the checks are local. -/
theorem accepts_local {Q V : Type} {reads : U → Finset Q} {chk : U → (Q → V) → Prop} (hl : Local reads chk)
    (A : Finset U) {k : ℕ} (d : Fin k → U) {σ τ : Q → V}
    (h : ∀ u, (u ∈ A ∨ ∃ i, d i = u) → ∀ q ∈ reads u, σ q = τ q) :
    Accepts chk A d σ ↔ Accepts chk A d τ := by
  unfold Accepts
  constructor
  · rintro ⟨hA, hd⟩
    exact ⟨fun u hu => (hl u σ τ (h u (Or.inl hu))).1 (hA u hu),
      fun i => (hl _ σ τ (h _ (Or.inr ⟨i, rfl⟩))).1 (hd i)⟩
  · rintro ⟨hA, hd⟩
    exact ⟨fun u hu => (hl u σ τ (h u (Or.inl hu))).2 (hA u hu),
      fun i => (hl _ σ τ (h _ (Or.inr ⟨i, rfl⟩))).2 (hd i)⟩

/-- **A dense check is untestable from a bounded read**: if every transcript agrees on `R` with one that the check
`G` accepts, a decision that reads `R` and accepts every `G` transcript accepts every transcript. -/
theorem untestable_of_dense {Q V : Type} (G : (Q → V) → Prop) (R : Finset Q) (dec : (R → V) → Prop)
    (hcomp : ∀ τ : Q → V, G τ → dec fun r : R => τ r) (hdense : ∀ σ : Q → V, ∃ τ, G τ ∧ ∀ q ∈ R, τ q = σ q)
    (σ : Q → V) : dec fun r : R => σ r := by
  obtain ⟨τ, hG, hτ⟩ := hdense σ
  have : (fun r : R => τ r) = fun r : R => σ r := funext fun r => hτ r r.2
  exact this ▸ hcomp τ hG

/-- The checksum: the `n` values sum to zero. A unit whose check is the checksum reads all `n` positions. -/
def checksum {n : ℕ} {V : Type} [AddCommGroup V] (σ : Fin n → V) : Prop := ∑ i, σ i = 0

theorem checksum_dense {n : ℕ} {V : Type} [AddCommGroup V] (R : Finset (Fin n)) (hR : R.card < n)
    (σ : Fin n → V) : ∃ τ, checksum τ ∧ ∀ q ∈ R, τ q = σ q := by
  classical
  obtain ⟨j, hj⟩ : ∃ j, j ∉ R := by
    by_contra h
    push Not at h
    have := card_le_card fun x (_ : x ∈ (univ : Finset (Fin n))) => h x
    simp at this
    omega
  refine ⟨Function.update σ j (σ j - ∑ i, σ i), ?_,
    fun q hq => Function.update_of_ne (by rintro rfl; exact hj hq) _ _⟩
  unfold checksum
  have hsplit := sum_sdiff (f := σ) (singleton_subset_iff.2 (mem_univ j))
  rw [sum_singleton] at hsplit
  rw [sum_update_of_mem (mem_univ j), ← hsplit]
  abel

/-- **A non-local check isn't implied by sampled local reads**: a decision that reads fewer than `n` positions and
accepts every transcript whose `n` values sum to zero accepts every transcript. So a unit whose check is the checksum,
always proved and read through any fewer than `n` positions, passes when it is wrong. -/
theorem checksum_untestable {n : ℕ} {V : Type} [AddCommGroup V] (R : Finset (Fin n)) (hR : R.card < n)
    (dec : (R → V) → Prop) (hcomp : ∀ τ : Fin n → V, checksum τ → dec fun r : R => τ r) (σ : Fin n → V) :
    dec fun r : R => σ r :=
  untestable_of_dense checksum R dec hcomp (checksum_dense R hR) σ

/-! ## (i) A transcript after the coins, and a draw after the transcript -/

/-- **A transcript chosen after the coins dodges the draw**: when every set of units is some transcript's wrong set,
the strategy that reads the draw passes every draw with every undrawn unit outside `A` wrong. -/
theorem adaptive_dodges [Fintype U] [DecidableEq U] {chk : U → T → Prop} (hfree : ∀ B : Finset U, ∃ σ, bad chk σ = B)
    (A : Finset U) (k : ℕ) :
    ∃ σ : (Fin k → U) → T, ∀ d, Accepts chk A d (σ d) ∧ bad chk (σ d) = (A ∪ univ.image d)ᶜ := by
  choose g hg using hfree
  have pass : ∀ (B : Finset U) u, u ∉ B → chk u (g B) := by
    intro B u hu
    classical
    have : u ∉ bad chk (g B) := by rw [hg]; exact hu
    simpa [bad] using this
  exact ⟨fun d => g (A ∪ univ.image d)ᶜ, fun d =>
    ⟨⟨fun u hu => pass _ u (by simp [hu]), fun i => pass _ _ (by simp)⟩, hg _⟩⟩

/-- Its wrong set holds all but `|A| + k` units, so under the uniform draw `π(B) ≥ 1 − (|A| + k)/|U|`. -/
theorem adaptive_card [Fintype U] [DecidableEq U] (A : Finset U) {k : ℕ} (d : Fin k → U) :
    Fintype.card U ≤ ((A ∪ univ.image d)ᶜ).card + A.card + k := by
  rw [card_compl]
  have h1 := card_union_le A (univ.image d)
  have h2 : (univ.image d).card ≤ k := card_image_le.trans (by simp)
  have h3 := card_le_univ (A ∪ univ.image d)
  omega

/-- **A draw chosen after the transcript dodges it**: the point mass on a correct unit is a draw distribution, and
the transcript passes it with probability one, whatever its wrong set. -/
theorem late_draw_dodges [Fintype U] [DecidableEq U] (chk : U → T → Prop) (A : Finset U) (σ : T) {u₀ : U}
    (h₀ : chk u₀ σ) (hA : ∀ u ∈ A, chk u σ) (k : ℕ) :
    Dist (fun u => if u = u₀ then (1 : ℝ) else 0) ∧
      prob (fun u => if u = u₀ then (1 : ℝ) else 0) k (fun d => Accepts chk A d σ) = 1 := by
  have hπ : Dist (fun u => if u = u₀ then (1 : ℝ) else 0) :=
    ⟨fun u => by split_ifs <;> norm_num, by simp⟩
  refine ⟨hπ, not_inferable_of_massless _ k hπ chk A σ ((disjoint_bad_iff chk A σ).2 hA) ?_⟩
  classical
  unfold mass
  refine sum_eq_zero fun u hu => ?_
  have hu' : ¬ chk u σ := by simpa [bad] using hu
  have : u ≠ u₀ := fun e => hu' (e ▸ h₀)
  simp [this]

/-- The flags population (POP1) meets `adaptive_dodges`'s hypothesis: every set of units is some assignment's wrong
set. -/
theorem flags_free (n : ℕ) (B : Finset (Fin n)) :
    ∃ σ, bad (corr (Dataflow.flags n) (Dataflow.flagsFns n)) σ = B := by
  classical
  refine ⟨fun s => decide (s ∈ B), ?_⟩
  have key : ∀ u, corr (Dataflow.flags n) (Dataflow.flagsFns n) u (fun s => decide (s ∈ B)) ↔ u ∉ B := by
    intro u
    constructor
    · intro h hu
      have := h ⟨u, by simp [Dataflow.flags]⟩
      simp [Dataflow.flagsFns, hu] at this
    · intro hu ⟨s, hs⟩
      have : s = u := by simpa [Dataflow.flags] using hs
      subst this
      simp [Dataflow.flagsFns, hu]
  ext u
  rw [mem_bad, key, not_not]

/-! ## (iii) Determinacy: a free value, and determinacy that isn't robust -/

section Free

/-- `k` units, unit `i` reading the slot `inl i`, which no unit writes and isn't an input, and writing the egress
slot `inr i`. Acyclic, and no unit writes an input; only coverage fails. -/
def freeVals (k : ℕ) : Dataflow (Fin k) (Fin k ⊕ Fin k) where
  input := ∅
  reads i := {Sum.inl i}
  writes i := {Sum.inr i}

/-- Unit `i` copies its read. -/
def freeFns (k : ℕ) {V : Type} : (freeVals k).Fns V := fun i r _ => r ⟨Sum.inl i, mem_singleton_self _⟩

theorem freeVals_acyclic (k : ℕ) : (freeVals k).Acyclic := by
  intro u h
  obtain ⟨b, hb⟩ := Relation.TransGen.head'_iff.1 h
  obtain ⟨s, hs, hs'⟩ := hb.1
  simp [freeVals] at hs hs'
  rw [hs] at hs'
  exact Sum.inr_ne_inl hs'

theorem freeVals_not_covered {k : ℕ} (hk : 0 < k) : ¬ (freeVals k).Covered := by
  intro h
  obtain ⟨u, hu⟩ := h (Sum.inl ⟨0, hk⟩) (by simp [freeVals])
  simp [freeVals] at hu

/-- **A free value makes the egress arbitrary with no wrong unit**: for every egress value `y` some assignment makes
every unit correct and shows `y`. Every unit passes, so checking every unit (the always-proved set `univ`) doesn't
help. -/
theorem free_value (k : ℕ) {V : Type} (y : Fin k → V) :
    ∃ σ : Fin k ⊕ Fin k → V, (∀ u, (freeVals k).Correct (freeFns k) σ u) ∧ ∀ i, σ (Sum.inr i) = y i := by
  refine ⟨Sum.elim y y, fun u ⟨s, hs⟩ => ?_, fun i => rfl⟩
  have : s = Sum.inr u := by simpa [freeVals] using hs
  subst this
  rfl

end Free

section Gate

/-- The gate's units and slots: `none` is the gate unit and its slot `t`, `some (false, i)` the unit writing `y i`,
`some (true, i)` the unit writing `z i`. Every unit writes the slot of its own name. -/
abbrev Gx (k : ℕ) := Option (Bool × Fin k)

/-- `y i` and `z i` copy each other; the gate reads everything and writes `t`. -/
def gate (k : ℕ) : Dataflow (Gx k) (Gx k) where
  input := ∅
  reads
    | none => univ
    | some (b, i) => {some (!b, i)}
  writes u := {u}

/-- The gate writes `¬ t` when some `y i` is set, which no correct `t` can satisfy, and `false` otherwise. -/
def gateFns (k : ℕ) : (gate k).Fns Bool
  | none => fun r _ =>
      if ∃ i, r ⟨some (false, i), mem_univ _⟩ = true then !(r ⟨none, mem_univ _⟩) else false
  | some (b, i) => fun r _ => r ⟨some (!b, i), mem_singleton_self _⟩

theorem gate_correct_some {k : ℕ} {σ : Gx k → Bool} {b : Bool} {i : Fin k} :
    (gate k).Correct (gateFns k) σ (some (b, i)) ↔ σ (some (b, i)) = σ (some (!b, i)) := by
  constructor
  · intro h
    exact h ⟨some (b, i), mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = some (b, i) := by simpa [gate] using hs
    subst this
    exact h

theorem gate_correct_none {k : ℕ} {σ : Gx k → Bool} :
    (gate k).Correct (gateFns k) σ none ↔
      σ none = if ∃ i, σ (some (false, i)) = true then !(σ none) else false := by
  constructor
  · intro h
    exact h ⟨none, mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = none := by simpa [gate] using hs
    subst this
    exact h

/-- **The gate is determinate at its functions**: exactly one assignment, all `false`, makes every unit correct. -/
theorem gate_determinate (k : ℕ) (a : Gx k → Bool) : DeterminateAt (gate k) (gateFns k) a := by
  refine ⟨fun _ => false, ⟨fun s hs => by simp [gate] at hs, fun u => ?_⟩, ?_⟩
  · rcases u with _ | ⟨b, i⟩
    · rw [gate_correct_none]
      simp
    · rw [gate_correct_some]
  · rintro σ ⟨-, hσ⟩
    have hyz : ∀ b i, σ (some (b, i)) = σ (some (!b, i)) := fun b i => gate_correct_some.1 (hσ _)
    have ht := gate_correct_none.1 (hσ none)
    have hy : ∀ i, σ (some (false, i)) = false := by
      intro i
      by_contra hi
      have hex : ∃ i, σ (some (false, i)) = true := ⟨i, by simpa using hi⟩
      rw [ite_eq_left hex] at ht
      cases h : σ none <;> simp [h] at ht
    have hnone : σ none = false := by
      rw [ht, ite_eq_right]
      simp [hy]
    funext s
    rcases s with _ | ⟨b, i⟩
    · exact hnone
    · cases b
      · exact hy i
      · rw [hyz true i]
        exact hy i

/-- **But not robustly**: for every `y`, some assignment whose only possibly wrong unit is the gate shows `y` on the
`y` slots. -/
theorem gate_free (k : ℕ) (y : Fin k → Bool) :
    ∃ σ : Gx k → Bool, (gate k).wrong (gateFns k) σ ⊆ {none} ∧ ∀ i, σ (some (false, i)) = y i := by
  let σ : Gx k → Bool := fun s => match s with
    | none => false
    | some (_, i) => y i
  refine ⟨σ, fun u hu => ?_, fun i => rfl⟩
  rcases u with _ | ⟨b, i⟩
  · exact mem_singleton_self _
  · exfalso
    classical
    have hu' : ¬ (gate k).Correct (gateFns k) σ (some (b, i)) := by simpa [Dataflow.wrong] using hu
    exact hu' (gate_correct_some.2 rfl)

/-- The `y` slots, the egress. -/
def gateOut (k : ℕ) : Finset (Gx k) := univ.image fun i => some (false, i)

/-- **The gate's cone holds no egress**: only the gate reads `t`, so `E(gate) = 0`, and a draw weighted by cone egress
gives it no weight. -/
theorem gate_coneEgress (k : ℕ) : (gate k).coneEgress (gateOut k) {none} = ∅ := by
  classical
  have hdown : ∀ v, Relation.ReflTransGen (gate k).Dep none v → v = none := by
    intro v h
    induction h with
    | refl => rfl
    | tail _ hd ih =>
      subst ih
      obtain ⟨s, hs, hs'⟩ := hd
      have : s = none := by simpa [gate] using hs
      subst this
      rename_i c _
      rcases c with _ | ⟨b, i⟩
      · rfl
      · simp [gate] at hs'
  ext s
  simp only [Dataflow.coneEgress, mem_inter, Dataflow.writesOf, mem_biUnion, notMem_empty, iff_false, not_and]
  intro hs ⟨v, hv, hsv⟩
  have hv' := hdown v (((gate k).mem_down).1 hv |>.elim fun u ⟨hu, h⟩ => by
    rw [mem_singleton] at hu; exact hu ▸ h)
  subst hv'
  have : s = none := by simpa [gate] using hsv
  subst this
  simp [gateOut] at hs

/-- **With the gate wrong, the egress takes all `2^k` values**, though the gate writes one slot and its cone holds no
egress. Determinacy at fixed functions, without robustness, bounds neither integrity nor control. -/
theorem gate_egress (k : ℕ) (a : Gx k → Bool) :
    2 ^ k ≤ ((gate k).egress (gateFns k) a ∅ (gateOut k) {none}).card := by
  classical
  let φ : (Fin k → Bool) → (↥(gateOut k) → Bool) := fun y o => match o.1 with
    | none => false
    | some (_, i) => y i
  have hmem : ∀ y, φ y ∈ (gate k).egress (gateFns k) a ∅ (gateOut k) {none} := by
    intro y
    obtain ⟨σ, hw, hσ⟩ := gate_free k y
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨σ, ⟨fun s hs => by simp [gate] at hs, hw⟩, funext fun o => ?_⟩
    obtain ⟨o, ho⟩ := o
    obtain ⟨i, -, rfl⟩ := mem_image.1 ho
    exact hσ i
  have hinj : Function.Injective φ := by
    intro y y' h
    funext i
    exact congrFun h ⟨some (false, i), mem_image.2 ⟨i, mem_univ _, rfl⟩⟩
  have := card_le_card_of_injOn φ (s := univ) (fun y _ => hmem y) hinj.injOn
  simpa using this

theorem gate_singleAssignment (k : ℕ) : (gate k).SingleAssignment :=
  fun s => ⟨fun h => by simp [gate] at h, fun _ => ⟨s, by simp [gate], fun u hu => by simpa [gate, eq_comm] using hu⟩⟩

end Gate

/-! ## Tier 1 for a population -/

section Skeleton

variable {S : Type} (P : Dataflow U S)

theorem bad_corr [Fintype U] {V : Type} (f : P.Fns V) (σ : S → V) : bad (corr P f) σ = P.wrong f σ := rfl

/-- **Integrity's damage lies in the wrong units' cone** (by robust determinacy): in a checked population, every egress
slot where a run differs from the honest run is in the cone egress of its wrong units. -/
theorem damaged_sub_coneEgress [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest σ : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = a s)
    (O : Finset S) : damaged O σ honest ⊆ P.coneEgress O (P.wrong f σ) := by
  classical
  intro o ho
  simp only [damaged, mem_filter] at ho
  refine (P.mem_coneEgress).2 ⟨ho.1, ?_⟩
  by_contra hno
  exact ho.2 (P.eq_off_cone (P.covered_of_singleAssignment h.1) h.2 f
    (fun t ht => (hσ t ht).trans (hh t ht).symm) le_rfl (by simp [P.wrong_eq_empty hc]) hno)

theorem card_damaged_le [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest σ : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = a s)
    (O : Finset S) : (damaged O σ honest).card ≤ ∑ u ∈ P.wrong f σ, (P.coneEgress O {u}).card :=
  (card_le_card (damaged_sub_coneEgress P h f a hh hc hσ O)).trans (P.card_coneEgress_le O _)

/-- **The skeleton theorem**: in a checked population (tier 1's (ii) by type, (iii) by the checks), for every run
committed before the coins on the given inputs and every draw fixed before it, the verifier accepts while any of
`B ∩ A = ∅`, `π(B) ≤ ε` and "every damaged egress slot is in `B`'s cone" fails with probability at most `(1 − ε)^k`. -/
theorem skeleton [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) {π : U → ℝ} (hπ : Dist π)
    (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) (O : Finset S) (σ : S → V) (hσ : ∀ s ∈ P.input, σ s = a s) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ ¬ (Disjoint (P.wrong f σ) A ∧ mass π (P.wrong f σ) ≤ ε ∧
      damaged O σ honest ⊆ P.coneEgress O (P.wrong f σ))) ≤ (1 - ε) ^ k := by
  refine (prob_mono π k hπ.nonneg
    (E' := fun d => Accepts (corr P f) A d σ ∧ ¬ Certified (corr P f) π A ε σ)
    fun d hd => ⟨hd.1, fun hcert => hd.2 ⟨hcert.1, hcert.2,
      damaged_sub_coneEgress P h f a hh hc hσ O⟩⟩).trans ?_
  exact certificate π k hπ (corr P f) A hε σ

/-- **Integrity, inferred**: drawing by cone egress (`E(u) ≤ C · π(u)`), an accepted run differs from the honest run
on more than `C ε` egress slots with probability at most `(1 − ε)^k`. Amplification `1` at the cost `E`. -/
theorem integrity_inferred [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) {π : U → ℝ} (hπ : Dist π)
    (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) (O : Finset S) {C : ℝ} (hC : 0 ≤ C)
    (hE : ∀ u, ((P.coneEgress O {u}).card : ℝ) ≤ C * π u) (σ : S → V) (hσ : ∀ s ∈ P.input, σ s = a s) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ C * ε < (damaged O σ honest).card) ≤ (1 - ε) ^ k := by
  classical
  have hamp : AmpLe (corr P f) A (fun u => ((P.coneEgress O {u}).card : ℝ))
      (fun τ => if ∀ s ∈ P.input, τ s = a s then ((damaged O τ honest).card : ℝ) else 0) 1 := by
    intro τ _
    rw [one_mul, bad_corr]
    dsimp only
    split_ifs with hτ
    · exact_mod_cast card_damaged_le P h f a hh hc hτ O
    · exact sum_nonneg fun _ _ => Nat.cast_nonneg _
  have := inferable_of_amp π k hπ zero_le_one hC hamp hE hε σ
  rw [ite_eq_left hσ, one_mul] at this
  exact this

end Skeleton

end Population.Iop
