import Batteries.Tactic.OpenPrivate
import Properties.NetworkPeriod.Bernoulli
import Properties.NetworkPeriod.Placed
import Proofs.Flock.Claims.Sampling.Count
import Proofs.Properties.SampledProofs

/-!
# `WrongBounds` at the bound the audit publishes for a Bernoulli draw

The sampled audit publishes `SampledProofs.Bound.bernoulliBound population p δ` as its bound on wrong units for a
Bernoulli law at the rate `p` (`Bound.boundOf`). Its acceptance probability is exact there: a design that draws its
population independently at `p`, however the rate is written, and every other class whole misses a set `B` that meets
no class drawn whole with probability `(1 − p)^|B|` (`classesPa_bernoulli`), so a possible incorrect set `F` at `δ`
lies in the population and has `(1 − p)^|F| > δ`. The code's search finds every such count (`le_bernoulliBound`): its
doubling loop stops at a count past `population` or one that `(1 − p)^·` puts below `δ`, and its bisection keeps that
end until the other is one below it.

* `wrongBounds_of_bernoulliBound`: `WrongBounds` at `boundOf population (.bernoulli p) δ`, at a design that is
  `BernoulliDrawn` at `population` and `p`.
* `wrongBounds_of_accept_bernoulli`: the same at the bound an accepted one-stage audit publishes for a level its
  profile draws by a Bernoulli law (`Properties.SampledProofs.bernoulliBounds`), at the parameters' registered `δ`.
* `bernoulliDrawn_deployed`: the premise holds at the deployed shape, a population at a rate written in other terms
  and an input class drawn whole.
-/

namespace Properties.NetworkPeriod.Bernoulli

open Claims.Sampling Finset
open private escapes from SampledProofs.Bound

theorem escapes_eq (p δ : Core.Ratio) (m : ℕ) :
    escapes p δ m = decide (δ.num * p.den ^ m ≤ (p.den - p.num) ^ m * δ.den) := rfl

/-- `(1 − p)^m` falls with `m`, so a count that keeps it at least `δ` keeps every smaller count there. -/
theorem escapes_anti {p δ : Core.Ratio} (hp : 0 < p.den) {a b : ℕ} (hab : a ≤ b) (h : escapes p δ b = true) :
    escapes p δ a = true := by
  rw [escapes_eq, decide_eq_true_eq] at h ⊢
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hab
  refine Nat.le_of_mul_le_mul_right ?_ (pow_pos hp k)
  calc δ.num * p.den ^ a * p.den ^ k = δ.num * p.den ^ (a + k) := by ring
    _ ≤ (p.den - p.num) ^ (a + k) * δ.den := h
    _ = (p.den - p.num) ^ a * δ.den * (p.den - p.num) ^ k := by ring
    _ ≤ (p.den - p.num) ^ a * δ.den * p.den ^ k := Nat.mul_le_mul_left _ (Nat.pow_le_pow_left (Nat.sub_le _ _) k)

/-- A property of what follows a loop over `[0, r)`, from an invariant its steps keep and the property of the state
a step that breaks leaves (`Count.forIn_inv`). -/
theorem run_forIn_bind {β γ : Type} (G : γ → Prop) (P : ℕ → β → Prop) (Q : β → Prop) (r : ℕ) (init : β)
    (f : ℕ → β → Id (ForInStep β)) (g : β → Id γ) (h0 : P 0 init)
    (hf : ∀ K b, P K b → (∃ b', (f K b).run = .done b' ∧ Q b') ∨ ∃ b', (f K b).run = .yield b' ∧ P (K + 1) b')
    (hg : ∀ b, Q b ∨ P r b → G (g b).run) : G (Id.run (forIn (List.range' 0 r) init f >>= g)) := by
  rw [Id.run_bind]
  refine hg _ ?_
  simpa only [Nat.zero_add] using Count.forIn_inv f P Q hf r 0 init h0

/-- **The code's search finds every escaping count**: each `m ≤ population` with `(1 − p)^m ≥ δ` is at most
`bernoulliBound population p δ`. The doubling loop ends at a `hi` past `population` or with `(1 − p)^hi < δ`, and the
bisection keeps that end while it halves the gap to `lo`, so it returns a `lo` one below such an end. -/
theorem le_bernoulliBound {population : ℕ} {p δ : Core.Ratio} (hp : 0 < p.den) {m : ℕ} (hm : m ≤ population)
    (he : escapes p δ m = true) : m ≤ SampledProofs.Bound.bernoulliBound population p δ := by
  have hpow : population < 2 ^ (population.log2 + 1) := Nat.lt_log2_self
  unfold SampledProofs.Bound.bernoulliBound
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one]
  split
  · exact hm
  refine run_forIn_bind (fun x => m ≤ x) (fun K (b : ℕ × ℕ) => b.2 = 2 ^ K)
    (fun b => population < b.2 ∨ escapes p δ b.2 = false) _ _ _ _ ?_ ?_ ?_
  · rfl
  · intro K b hb
    obtain ⟨lo, hi⟩ := b
    dsimp only at hb ⊢
    split_ifs with h
    · left
      exact ⟨_, rfl, by simpa using h⟩
    · right
      exact ⟨_, rfl, by show 2 * hi = 2 ^ (K + 1); rw [hb]; ring⟩
  · intro b h1
    obtain ⟨lo, hi⟩ := b
    dsimp only at h1 ⊢
    have hhi : population < min hi (population + 1) ∨ escapes p δ (min hi (population + 1)) = false := by
      by_cases hlt : population < hi
      · left
        have := Nat.min_eq_right (show population + 1 ≤ hi by omega)
        omega
      · rw [Nat.min_eq_left (show hi ≤ population + 1 by omega)]
        rcases h1 with h | h
        · exact Or.inr (h.resolve_left hlt)
        · have : population < hi := by
            rw [h]
            exact lt_of_lt_of_le hpow (Nat.pow_le_pow_right (by norm_num) (by omega))
          exact absurd this hlt
    refine run_forIn_bind (fun x => m ≤ x)
      (fun K (b : ℕ × ℕ) => (population < b.2 ∨ escapes p δ b.2 = false) ∧ b.2 - b.1 ≤ 2 ^ (population.log2 + 1 - K))
      (fun b => (population < b.2 ∨ escapes p δ b.2 = false) ∧ b.2 ≤ b.1 + 1) _ _ _ _ ?_ ?_ ?_
    · refine ⟨hhi, ?_⟩
      show min hi (population + 1) - lo ≤ 2 ^ (population.log2 + 1 - 0)
      have := Nat.min_le_right hi (population + 1)
      rw [Nat.sub_zero]
      omega
    · intro K b hb
      obtain ⟨lo', hi'⟩ := b
      dsimp only at hb ⊢
      obtain ⟨hinv, hgap⟩ := hb
      have halve : ¬ hi' ≤ lo' + 1 → hi' - lo' ≤ 2 * 2 ^ (population.log2 + 1 - (K + 1)) := fun h => by
        rcases Nat.eq_zero_or_pos (population.log2 + 1 - K) with h0 | h0
        · rw [h0, pow_zero] at hgap
          omega
        · rwa [← pow_succ', show population.log2 + 1 - (K + 1) + 1 = population.log2 + 1 - K by omega]
      split_ifs with h h'
      · left
        exact ⟨_, rfl, hinv, h⟩
      · right
        refine ⟨_, rfl, hinv, ?_⟩
        show hi' - (lo' + hi') / 2 ≤ 2 ^ (population.log2 + 1 - (K + 1))
        have := halve h
        omega
      · right
        refine ⟨_, rfl, Or.inr (by simpa using h'), ?_⟩
        show (lo' + hi') / 2 - lo' ≤ 2 ^ (population.log2 + 1 - (K + 1))
        have := halve h
        omega
    · intro b hend
      obtain ⟨lo', hi'⟩ := b
      dsimp only at hend ⊢
      have hfin : (population < hi' ∨ escapes p δ hi' = false) ∧ hi' ≤ lo' + 1 := by
        rcases hend with h | ⟨h1, h2⟩
        · exact h
        · refine ⟨h1, ?_⟩
          have h3 : hi' - lo' ≤ 2 ^ 0 := h2.trans (Nat.pow_le_pow_right (by norm_num) (by omega))
          rw [pow_zero] at h3
          omega
      show m ≤ lo'
      by_contra hlt
      rcases hfin.1 with h | h
      · omega
      · have h3 := escapes_anti hp (show hi' ≤ m by omega) he
        rw [h] at h3
        cases h3

/-- A rate written `a/b` misses a proof block as often as the same rate written `p`. -/
theorem rate_eq {a b : ℕ} {p : Core.Ratio} (hb : 0 < b) (hab : a ≤ b) (hp : 0 < p.den) (h : a * p.den = p.num * b) :
    ((b - a : ℕ) : ℝ) / b = ((p.den - p.num : ℕ) : ℝ) / p.den := by
  have hpn : p.num ≤ p.den := by
    refine Nat.le_of_mul_le_mul_right ?_ hb
    calc p.num * b = a * p.den := h.symm
      _ ≤ b * p.den := Nat.mul_le_mul_right _ hab
      _ = p.den * b := Nat.mul_comm _ _
  have h' : (a : ℝ) * p.den = p.num * b := by exact_mod_cast h
  have hb' : (b : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hb.ne'
  have hp' : (p.den : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hp.ne'
  rw [div_eq_div_iff hb' hp', Nat.cast_sub hab, Nat.cast_sub hpn, sub_mul, sub_mul, h']
  ring

/-- **A design whose every rule draws at the rate `p` or draws all misses `B` with probability `(1 − p)^k`**, where
`B` meets no class drawn all, `k` the number of `B`'s members among its classes' proof blocks, from slot `o`: a class
drawn all holds none of `B`, so it misses `B` surely. -/
theorem classesPa_bernoulli {p : Core.Ratio} (hp : 0 < p.den) : ∀ (rs : List OneStage)
    (cs : List (List (List (List ℕ)))) (o : ℕ) (B : Finset ℕ), rs.length = cs.length →
    (∀ x ∈ rs.zip cs, x.1.WellFormed (strataSizes x.2)) →
    (∀ r ∈ rs, r = .all ∨ ∃ a b, r = .bernoulli a b ∧ a * p.den = p.num * b) →
    Disjoint B (NetTiming.allSlotsFrom rs cs o) →
    classesPa rs cs o B =
      (((p.den - p.num : ℕ) : ℝ) / p.den) ^ (B.filter fun v => o ≤ v ∧ v < o + cs.flatten.flatten.length).card
  | [], [], o, B, _, _, _, _ => by
    have h : B.filter (fun v => o ≤ v ∧ v < o + ([] : List (List (List (List ℕ)))).flatten.flatten.length) = ∅ :=
      filter_eq_empty_iff.2 fun v _ => by simp only [List.flatten_nil, List.length_nil, add_zero]; omega
    show (1 : ℝ) = _
    rw [h, card_empty, pow_zero]
  | [], _ :: _, _, _, h, _, _, _ => absurd h (by simp)
  | _ :: _, [], _, _, h, _, _, _ => absurd h (by simp)
  | r :: rs, c :: cs, o, B, h, hw, hr, hd => by
    simp only [NetTiming.allSlotsFrom, Finset.disjoint_union_right] at hd
    rw [classesPa, classesPa_bernoulli hp rs cs (o + c.flatten.length) B (by simpa using h)
      (fun x hx => hw x (by rw [List.zip_cons_cons]; exact List.mem_cons_of_mem _ hx))
      (fun r' h' => hr r' (List.mem_cons_of_mem _ h')) hd.2]
    have hw1 := hw (r, c) (by simp)
    rcases hr r List.mem_cons_self with rfl | ⟨a, b, rfl, hab⟩
    · have hI : Disjoint B (Finset.Ico o (o + c.flatten.length)) := hd.1
      have hw0 : window o c.flatten.length B = ∅ := by
        rw [window, Finset.image_eq_empty, Finset.filter_eq_empty_iff]
        intro v hv hvI
        exact Finset.disjoint_left.1 hI hv (Finset.mem_Ico.2 hvI)
      have h1 : OneStage.Pa (strataSizes c) .all (window o c.flatten.length B) = 1 := by
        simp [OneStage.Pa, hw0]
      rw [h1, one_mul]
      congr 2
      ext v
      simp only [mem_filter, List.flatten_cons, List.flatten_append, List.length_append]
      by_cases hv : v ∈ B
      · have hv' : ¬(o ≤ v ∧ v < o + c.flatten.length) := fun hvI =>
          Finset.disjoint_left.1 hI hv (Finset.mem_Ico.2 hvI)
        simp only [hv, true_and]
        omega
      · simp [hv]
    · obtain ⟨hb0, hba⟩ : 0 < b ∧ a ≤ b := hw1
      simp only [OneStage.Pa]
      rw [rate_eq hb0 hba hp hab, ← pow_add, Count.card_window,
        ← card_union_of_disjoint (disjoint_filter.2 fun v _ h1 h2 => by omega)]
      congr 2
      ext v
      simp only [mem_union, mem_filter]
      by_cases hv : v ∈ B
      · simp only [hv, true_and, List.flatten_cons, List.flatten_append, List.length_append]
        omega
      · simp [hv]

end Properties.NetworkPeriod.Bernoulli

namespace Properties.NetworkPeriod

open Claims.Sampling _root_.Properties.NetworkPeriod.Bernoulli
open private escapes from SampledProofs.Bound

/-- **`WrongBounds` at the bound the audit publishes for a Bernoulli draw**: at a design that draws its population
independently at `p` and every other class whole (`BernoulliDrawn`), every possible incorrect set at `δ` holds at most
`boundOf population (.bernoulli p) δ` proof blocks. It meets no class drawn all, so it lies among the population's
at most `population` blocks and is accepted with probability exactly `(1 − p)^|F|` (`classesPa_bernoulli`), and the
code's search finds every count `(1 − p)^·` keeps at least `δ` (`le_bernoulliBound`). -/
theorem wrongBounds_of_bernoulliBound {s : AuditStatement} {P : Partition} {population : ℕ} {p : Core.Ratio}
    (hb : BernoulliDrawn s.design P population p) (δ : Core.Ratio) (hδd : 0 < δ.den) :
    WrongBounds s P ((δ.num : ℝ) / δ.den) (SampledProofs.Bound.boundOf population (.bernoulli p) δ) := by
  obtain ⟨⟨hlen, hwf⟩, hp, hr, hpop⟩ := hb
  intro F hF
  classical
  simp only [NetTiming.faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, hM, hdis⟩ := hF
  have hcard : F.card ≤ population :=
    (Finset.card_le_card fun v hv => Finset.mem_sdiff.2 ⟨hsub hv, Finset.disjoint_left.1 hdis hv⟩).trans hpop
  have hPa : s.design.Pa P F = (((p.den - p.num : ℕ) : ℝ) / p.den) ^ F.card := by
    rw [Design.Pa, classesPa_bernoulli hp _ _ 0 F hlen hwf hr hdis]
    congr 2
    refine Finset.filter_true_of_mem fun v hv => ⟨Nat.zero_le v, ?_⟩
    have := Finset.mem_range.1 (hsub hv)
    unfold Partition.blocks at this
    omega
  have hlt : (δ.num : ℝ) / δ.den < s.design.Pa P F := hM
  rw [hPa, div_pow, div_lt_div_iff₀ (by exact_mod_cast hδd) (pow_pos (by exact_mod_cast hp) _)] at hlt
  have he : escapes p δ F.card = true := by
    rw [escapes_eq, decide_eq_true_eq]
    exact_mod_cast hlt.le
  exact le_bernoulliBound hp hcard he

/-- **`WrongBounds` at the bound an accepted one-stage audit publishes, at its registered `δ`**: where the audit's
profile draws a level by a Bernoulli law at `r` (`Properties.SampledProofs.bernoulliBounds`) and the statement's
design draws the level's units at `r` and every other class whole (`BernoulliDrawn`). -/
theorem wrongBounds_of_accept_bernoulli {pa : SampledProofs.OneStage.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStage.Evidence} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : SampledProofs.OneStage.check pa c e = .accept pub prof) {r : Core.Ratio} {l : Core.Level}
    (hl : l ∈ prof.levels) (hd : l.draw = some (.bernoulli r)) {s : AuditStatement} {P : Partition}
    (hb : BernoulliDrawn s.design P l.size r) (hδ : 0 < pa.delta.den) :
    WrongBounds s P ((pa.delta.num : ℝ) / pa.delta.den) pub.bound := by
  rw [Properties.SampledProofs.bernoulliBounds pa c e pub prof h r l hl hd]
  exact wrongBounds_of_bernoulliBound hb pa.delta hδ

/-- **`BernoulliDrawn` holds at the deployed shape**: a population of two proof blocks drawn at `1/2`, written `2/4`,
then an input class of one drawn all. -/
theorem bernoulliDrawn_deployed :
    BernoulliDrawn ⟨[.bernoulli 2 4, .all]⟩ ⟨[[[[0], [1]]], [[[2]]]]⟩ 2 ⟨1, 2⟩ := by
  refine ⟨⟨rfl, ?_⟩, by decide, ?_, by decide⟩
  · intro x hx
    simp only [List.zip_cons_cons, List.zip_nil_left, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact ⟨by decide, by decide⟩
    · trivial
  · intro r hr
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl
    · exact Or.inr ⟨2, 4, rfl, rfl⟩
    · exact Or.inl rfl

end Properties.NetworkPeriod
