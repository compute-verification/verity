import Batteries.Tactic.OpenPrivate
import Properties.NetworkPeriod.Bernoulli
import Properties.NetworkPeriod.Placed
import Proofs.Flock.Claims.Sampling.Count
import Proofs.Properties.SampledProofs

/-!
# `WrongBounds` at the bound the audit publishes for a Bernoulli draw

The sampled audit publishes `SampledProofs.Bound.bernoulliBound population p δ` as its bound on wrong units for a
Bernoulli law at the rate `p` (`Bound.boundOf`). Its acceptance probability is exact there: a design that draws every
proof block independently at `p` misses a set `B` of them with probability `(1 − p)^|B|` (`classesPa_bernoulli`), so a
possible incorrect set `F` at `δ` has `(1 − p)^|F| > δ`. The code's search finds every such count
(`le_bernoulliBound`): its doubling loop stops at a count past `population` or one that `(1 − p)^·` puts below `δ`, and
its bisection keeps that end until the other is one below it.

* `wrongBounds_of_bernoulliBound`: `WrongBounds` at `boundOf population (.bernoulli p) δ`, at a design that is
  `BernoulliDrawn` at `population` and `p`.
* `wrongBounds_of_accept_bernoulli`: the same at the bound an accepted one-stage audit publishes for a level its
  profile draws by a Bernoulli law (`Properties.SampledProofs.bernoulliBounds`), at the parameters' registered `δ`.
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

/-- **A design whose every rule draws at the rate `num/den` misses `B` with probability `((den − num)/den)^k`**, `k`
the number of `B`'s members among its classes' proof blocks, from slot `o`. -/
theorem classesPa_bernoulli {num den : ℕ} : ∀ (rs : List OneStage) (cs : List (List (List (List ℕ)))) (o : ℕ)
    (B : Finset ℕ), (∀ r ∈ rs, r = .bernoulli num den) → rs.length = cs.length →
    classesPa rs cs o B =
      (((den - num : ℕ) : ℝ) / den) ^ (B.filter fun v => o ≤ v ∧ v < o + cs.flatten.flatten.length).card
  | [], [], o, B, _, _ => by
    have h : B.filter (fun v => o ≤ v ∧ v < o + ([] : List (List (List (List ℕ)))).flatten.flatten.length) = ∅ :=
      filter_eq_empty_iff.2 fun v _ => by simp only [List.flatten_nil, List.length_nil, add_zero]; omega
    show (1 : ℝ) = _
    rw [h, card_empty, pow_zero]
  | [], _ :: _, _, _, _, h => absurd h (by simp)
  | _ :: _, [], _, _, _, h => absurd h (by simp)
  | r :: rs, c :: cs, o, B, hr, h => by
    rw [classesPa, hr r List.mem_cons_self, classesPa_bernoulli rs cs (o + c.flatten.length) B
      (fun r' h' => hr r' (List.mem_cons_of_mem _ h')) (by simpa using h)]
    simp only [OneStage.Pa]
    rw [← pow_add, Count.card_window, ← card_union_of_disjoint (disjoint_filter.2 fun v _ h1 h2 => by omega)]
    congr 2
    ext v
    simp only [mem_union, mem_filter]
    by_cases hv : v ∈ B
    · simp only [hv, true_and, List.flatten_cons, List.flatten_append, List.length_append]
      omega
    · simp [hv]

end Properties.NetworkPeriod.Bernoulli

namespace Properties.NetworkPeriod

open Claims.Sampling Bernoulli
open private escapes from SampledProofs.Bound

/-- **`WrongBounds` at the bound the audit publishes for a Bernoulli draw**: at a design that draws every proof block
independently at `p` (`BernoulliDrawn`), every possible incorrect set at `δ` holds at most
`boundOf population (.bernoulli p) δ` proof blocks, since it is accepted with probability exactly `(1 − p)^|F|`
(`classesPa_bernoulli`) and the code's search finds every count `(1 − p)^·` keeps at least `δ` (`le_bernoulliBound`). -/
theorem wrongBounds_of_bernoulliBound {s : AuditStatement} {P : Partition} {population : ℕ} {p : Core.Ratio}
    (hb : BernoulliDrawn s.design P population p) (δ : Core.Ratio) (hδd : 0 < δ.den) :
    WrongBounds s P ((δ.num : ℝ) / δ.den) (SampledProofs.Bound.boundOf population (.bernoulli p) δ) := by
  intro F hF
  classical
  simp only [NetTiming.faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, hM, -⟩ := hF
  have hcard : F.card ≤ P.blocks.length := by simpa using Finset.card_le_card hsub
  have hPa : s.design.Pa P F = (((p.den - p.num : ℕ) : ℝ) / p.den) ^ F.card := by
    rw [Design.Pa, classesPa_bernoulli _ _ 0 F hb.2.1 hb.1.1]
    congr 2
    refine Finset.filter_true_of_mem fun v hv => ⟨Nat.zero_le v, ?_⟩
    have := Finset.mem_range.1 (hsub hv)
    unfold Partition.blocks at this
    omega
  have hlt : (δ.num : ℝ) / δ.den < s.design.Pa P F := hM
  rw [hPa] at hlt
  rcases Nat.eq_zero_or_pos p.den with h0 | hp
  · have : F.card = 0 := by
      by_contra hne
      rw [h0, Nat.cast_zero, div_zero, zero_pow hne] at hlt
      exact absurd hlt (not_lt.2 (by positivity))
    rw [this]
    exact Nat.zero_le _
  · have he : escapes p δ F.card = true := by
      rw [escapes_eq, decide_eq_true_eq]
      rw [div_pow, div_lt_div_iff₀ (by exact_mod_cast hδd) (pow_pos (by exact_mod_cast hp) _)] at hlt
      exact_mod_cast hlt.le
    exact le_bernoulliBound hp (hcard.trans hb.2.2) he

/-- **`WrongBounds` at the bound an accepted one-stage audit publishes, at its registered `δ`**: where the audit's
profile draws a level by a Bernoulli law at `r` (`Properties.SampledProofs.bernoulliBounds`) and the statement's
design draws every proof block at `r` over the level's units (`BernoulliDrawn`). -/
theorem wrongBounds_of_accept_bernoulli {pa : SampledProofs.OneStage.Params} {c : SampledProofs.OneStage.Commitment}
    {e : SampledProofs.OneStage.Evidence} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : SampledProofs.OneStage.check pa c e = .accept pub prof) {r : Core.Ratio} {l : Core.Level}
    (hl : l ∈ prof.levels) (hd : l.draw = some (.bernoulli r)) {s : AuditStatement} {P : Partition}
    (hb : BernoulliDrawn s.design P l.size r) (hδ : 0 < pa.delta.den) :
    WrongBounds s P ((pa.delta.num : ℝ) / pa.delta.den) pub.bound := by
  rw [Properties.SampledProofs.bernoulliBounds pa c e pub prof h r l hl hd]
  exact wrongBounds_of_bernoulliBound hb pa.delta hδ

end Properties.NetworkPeriod
