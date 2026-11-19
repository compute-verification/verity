module

public import Proofs.Flock.Soundness.ExecStratified
import Batteries.Tactic.OpenPrivate
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.ExecCheck
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The strata the verifier accepts are a partition's

`SamplingDesign.execStrata_of_ofJson`: a sample file `Flock.Draw.UnitSample.ofJson` accepts under a stratified sampling design has an assignment `σ`
of its population's units to the strata that the strata describe (`ExecStrata`), each stratum's `k` at most its size. The
hypotheses `ExecStrata σ k S` and `k s ≤ (stratum σ s).card` that the stratified e2e theorems take are so not conditions
on the statement's form: they follow from the verifier accepting.

* `StrataExec.ofJson_stratified`: what `ofJson` checks of a stratified sampling design: `strataProblem` finds nothing, and each stratum
  passed `stratumOfJson`, whose units are non-empty ranges `lo < hi` (`stratumOfJson_ranges`).
* `StrataExec.strataProblem_none`: what `strataProblem` checks: `1 ≤ k ≤ pop`, each stratum's `k` in `[1, size]`,
  and the ranges, sorted by their starts, chain from `0` to `pop` (`Chain`).
* `StrataExec.chain_facts`: a chain of non-empty ranges from `a` to `b` covers each unit of `[a, b)` exactly once;
  `StrataExec.chain_covers`: any chain from `a` to `b` covers each unit of `[a, b)`.
* `SamplingDesign.execStrata_of_strataProblem`: those checks give `σ`, each unit's stratum the one whose ranges hold it.
* At a work sampling design: `StrataExec.ofJson_work`, `ofJson` checks that `workProblem` finds nothing, and
  `StrataExec.workProblem_chain`, which checks that the ranges, sorted by their starts, chain from `0` to `pop`.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Sampling

open FlockSoundness

open Finset Flock.Draw

namespace StrataExec

/-- The ranges `l` follow each other from `a` to `b`: each starts where the last ended. -/
def Chain : ℕ → ℕ → List (ℕ × ℕ) → Prop
  | a, b, [] => a = b
  | a, b, r :: rs => r.1 = a ∧ Chain r.2 b rs

/-- The units of ranges `rs`, in order. -/
def flat (rs : List (ℕ × ℕ)) : List ℕ := rs.flatMap fun r => List.range' r.1 (r.2 - r.1)

/-- `Stratum.unit` on the list of its ranges. -/
def unitL : List (ℕ × ℕ) → ℕ → ℕ
  | [], _ => 0
  | r :: rs, j => if j < r.2 - r.1 then r.1 + j else unitL rs (j - (r.2 - r.1))

/-- Range `r` holds unit `u`. -/
def hits (u : ℕ) (r : ℕ × ℕ) : Bool := decide (r.1 ≤ u ∧ u < r.2)

theorem unit_eq (s : Stratum) (j : Nat) : s.unit j = unitL s.units.toList j := by
  unfold Stratum.unit
  simp only [← Array.forIn_toList, Id.run_bind]
  generalize s.units.toList = l
  induction l generalizing j with
  | nil => rfl
  | cons r rs ih =>
    simp only [List.forIn_cons, unitL, Id.run_bind]
    by_cases hj : j < r.2 - r.1
    · simp [hj]
    · simp only [hj, ite_false]
      exact ih _

theorem size_eq (s : Stratum) : s.size = (flat s.units.toList).length := by
  unfold Stratum.size flat
  rw [← Array.foldl_toList]
  generalize s.units.toList = l
  suffices ∀ acc, l.foldl (fun acc (r : Nat × Nat) => acc + (r.2 - r.1)) acc =
      acc + (l.flatMap fun r => List.range' r.1 (r.2 - r.1)).length by simpa using this 0
  induction l with
  | nil => intro acc; simp
  | cons r rs ih => intro acc; simp [ih]; omega

theorem unitL_getElem : ∀ (rs : List (Nat × Nat)) (j : Nat) (h : j < (flat rs).length), unitL rs j = (flat rs)[j]
  | [], j, h => by simp [flat] at h
  | r :: rs, j, h => by
    simp only [flat, List.flatMap_cons] at h ⊢
    unfold unitL
    split
    · next hj => rw [List.getElem_append_left (by simpa using hj)]; simp
    · next hj =>
      rw [List.getElem_append_right (by simp; omega)]
      have := unitL_getElem rs (j - (r.2 - r.1)) (by simp [flat] at h ⊢; omega)
      simp only [flat] at this
      simpa using this

theorem count_flat (u : Nat) : ∀ rs : List (Nat × Nat), (flat rs).count u = rs.countP (hits u)
  | [] => by simp [flat]
  | r :: rs => by
    have ih := count_flat u rs
    simp only [flat, List.flatMap_cons, List.count_append, List.countP_cons] at ih ⊢
    rw [ih]
    simp only [hits]
    rw [(List.nodup_range' 1).count]
    simp only [List.mem_range']
    by_cases h1 : r.1 ≤ u ∧ u < r.2
    · rw [ite_eq_left ⟨u - r.1, by omega, by omega⟩]
      simp [h1]; omega
    · rw [ite_eq_right (by rintro ⟨i, hi, rfl⟩; omega)]
      simp [h1]

theorem chain_facts : ∀ (a b : Nat) (l : List (Nat × Nat)), Chain a b l → (∀ r ∈ l, r.1 < r.2) →
    a ≤ b ∧ (∀ r ∈ l, a ≤ r.1 ∧ r.2 ≤ b) ∧ ∀ u, l.countP (hits u) = if a ≤ u ∧ u < b then 1 else 0
  | a, b, [], h, _ => by
    cases h
    exact ⟨Nat.le_refl _, by simp, fun u => by simp⟩
  | a, b, r :: rs, ⟨h1, h2⟩, hlt => by
    obtain ⟨hab, hin, hc⟩ := chain_facts r.2 b rs h2 (fun r' hr' => hlt r' (List.mem_cons_of_mem _ hr'))
    have hr := hlt r List.mem_cons_self
    refine ⟨by omega, fun r' hr' => ?_, fun u => ?_⟩
    · rcases List.mem_cons.1 hr' with rfl | hr'
      · omega
      · have := hin r' hr'; omega
    · rw [List.countP_cons, hc u]
      simp only [hits, decide_eq_true_eq]
      split <;> split <;> split <;> omega

/-- A chain of ranges from `a` to `b` covers each unit of `[a, b)`: some range holds it. -/
theorem chain_covers : ∀ {a b : ℕ} {l : List (ℕ × ℕ)}, Chain a b l → ∀ u, a ≤ u → u < b →
    ∃ r ∈ l, r.1 ≤ u ∧ u < r.2
  | a, b, [], h, u, hau, hub => by cases h; omega
  | a, b, r :: rs, ⟨h1, h2⟩, u, hau, hub => by
    by_cases hu : u < r.2
    · exact ⟨r, List.mem_cons_self, by omega, hu⟩
    · obtain ⟨r', hr', h⟩ := chain_covers h2 u (by omega) hub
      exact ⟨r', List.mem_cons_of_mem _ hr', h⟩

theorem qsort_perm {α : Type} (lt : α → α → Bool) (as : Array α) : (as.qsort lt).toList.Perm as.toList := by
  exact Array.perm_iff_toList_perm.1 (SampleExec.qsort_perm as lt)

theorem loopA {β : Type} (l : List β) (b : β → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit)))
    (hb : ∀ i, b i (none, ()) = ForInStep.yield (none, ()) ∨ ∃ m, b i (none, ()) = ForInStep.done (some (some m), ())) :
    (∃ m, (forIn (m := Id) l (none, ()) b).run.1 = some (some m)) ∨
      ((forIn (m := Id) l (none, ()) b).run.1 = none ∧ ∀ i ∈ l, b i (none, ()) = ForInStep.yield (none, ())) := by
  induction l with
  | nil => right; exact ⟨rfl, by simp⟩
  | cons i is ih =>
    rw [List.forIn_cons]
    rcases hb i with h | ⟨m, h⟩
    · rw [h]
      rcases ih with hm | ⟨hn, hall⟩
      · left; exact hm
      · right
        refine ⟨hn, fun j hj => ?_⟩
        rcases List.mem_cons.1 hj with rfl | hj
        · exact h
        · exact hall j hj
    · left; rw [h]; exact ⟨m, rfl⟩

theorem loopB (msg : String) (l : List (Nat × Nat)) (a : Nat) :
    let r := forIn (m := Id) l ((none : Option (Option String)), a) (fun x s =>
      if (x.fst != s.snd) = true then pure (ForInStep.done (some (some msg), s.snd))
      else pure (ForInStep.yield (none, x.snd)))
    r.run.1 = some (some msg) ∨ (r.run.1 = none ∧ Chain a r.run.2 l) := by
  induction l generalizing a with
  | nil => right; exact ⟨rfl, rfl⟩
  | cons x xs ih =>
    simp only [List.forIn_cons]
    by_cases hx : x.fst = a
    · have hne : (x.fst != a) = false := by simp [hx]
      simp only [hne, Bool.false_eq_true, ite_false]
      rcases ih x.snd with h | ⟨hn, hc⟩
      · left; exact h
      · right; exact ⟨hn, hx, hc⟩
    · have hne : (x.fst != a) = true := by simp [hx]
      simp only [hne, ite_true]
      left; rfl

/-- A loop each of whose steps from no early return continues or returns some `m` never returns `none` early. -/
theorem forIn_ne_some_none {β : Type} (l : List β)
    (b : β → Option (Option String) × Unit → Id (ForInStep (Option (Option String) × Unit)))
    (hb : ∀ i, b i (none, ()) = ForInStep.yield (none, ()) ∨ ∃ m, b i (none, ()) = ForInStep.done (some (some m), ())) :
    (forIn (m := Id) l (none, ()) b).run.1 ≠ some none := by
  induction l with
  | nil => simp [Id.run]
  | cons i is ih =>
    rw [List.forIn_cons]
    rcases hb i with h | ⟨m, h⟩
    · rw [h]; exact ih
    · rw [h]; intro h'; cases h'

theorem strataProblem_none {k : Nat} {S : Array Stratum} {pop : Nat} (h : strataProblem k S pop = none) :
    1 ≤ k ∧ k ≤ pop ∧ (∀ s ∈ S, 1 ≤ s.k ∧ s.k ≤ s.size) ∧
      Chain 0 pop ((S.foldl (fun x1 x2 => x1 ++ x2.units) #[]).qsort fun a b => decide (a.fst < b.fst)).toList := by
  unfold strataProblem at h
  split at h
  · simp at h
  · next hk =>
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Id.run_bind] at h
    have hk' : 1 ≤ k ∧ k ≤ pop := by simp at hk; omega
    rcases loopA (List.range' 0 [:S.size].size) (fun i __s =>
            if ((S.extract 0 i).any fun x => x.template == S[i]!.template) = true then
              pure
                (ForInStep.done
                  (some (some (toString "U2: template '" ++ toString S[i]!.template ++ toString "' twice")), ()))
            else
              if (S[i]!.k < 1 || S[i]!.k > S[i]!.size) = true then
                pure
                  (ForInStep.done
                    (some
                        (some
                          (toString "U2: stratum '" ++ toString S[i]!.template ++ toString "': k is not in [1, " ++
                            toString S[i]!.size ++ toString "]")),
                      ()))
              else pure (ForInStep.yield (none, ()))) (by
        intro i
        split
        · right; exact ⟨_, rfl⟩
        · split
          · right; exact ⟨_, rfl⟩
          · left; rfl) with ⟨m, hm⟩ | ⟨hn, hall⟩
    · simp only [hm] at h
      cases h
    · simp only [hn, Id.run_bind] at h
      rcases loopB "U2: the strata don't tile the population"
          ((S.foldl (fun (x1 : Array (Nat × Nat)) (x2 : Stratum) => x1 ++ x2.units) #[]).qsort
            fun (a b : Nat × Nat) => decide (a.fst < b.fst)).toList 0 with hm | ⟨hn2, hc⟩
      · simp only [Array.forIn_toList] at hm
        simp only [hm] at h
        cases h
      · simp only [Array.forIn_toList] at hn2 hc
        simp only [hn2] at h
        refine ⟨hk'.1, hk'.2, fun s hs => ?_, ?_⟩
        · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hs
          have := hall i (by simp [List.mem_range']; omega)
          split at this
          · cases this
          · split at this
            · cases this
            · next hne =>
              simp only [getElem!_pos S i hi, Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_lt] at hne
              obtain ⟨h1, h2⟩ := hne
              exact ⟨by omega, h2⟩
        · split at h
          · next heq => rw [← beq_iff_eq.1 heq]; exact hc
          · cases h

/-- What `workProblem` checks of the ranges: sorted by their starts, they chain from `0` to `pop`. -/
theorem workProblem_chain {k : ℕ} {S : Array WorkStratum} {pop : ℕ} (h : workProblem k S pop = none) :
    Chain 0 pop ((S.foldl (fun x1 x2 => x1 ++ x2.units) #[]).qsort fun a b => decide (a.fst < b.fst)).toList := by
  unfold workProblem at h
  split at h
  · simp at h
  · simp only [Std.Legacy.Range.forIn_eq_forIn_range'] at h
    split at h
    · simp at h
    · simp only [Id.run_bind] at h
      split at h
      · next r heq =>
        have hr : r = none := h
        subst hr
        exact absurd heq (forIn_ne_some_none _ _ fun i => by
          split
          · exact Or.inr ⟨_, rfl⟩
          · split
            · exact Or.inr ⟨_, rfl⟩
            · split
              · exact Or.inr ⟨_, rfl⟩
              · exact Or.inl rfl)
      · simp only [Id.run_bind] at h
        rcases loopB "U2: the strata don't tile the population"
            ((S.foldl (fun (x1 : Array (ℕ × ℕ)) (x2 : WorkStratum) => x1 ++ x2.units) #[]).qsort
              fun (a b : ℕ × ℕ) => decide (a.fst < b.fst)).toList 0 with hm | ⟨hn2, hc⟩
        · simp only [Array.forIn_toList] at hm
          simp only [hm] at h
          cases h
        · simp only [Array.forIn_toList] at hn2 hc
          simp only [hn2] at h
          split at h
          · next heq => rw [← beq_iff_eq.1 heq]; exact hc
          · cases h

theorem foldl_units (S : Array Stratum) :
    (S.foldl (fun (x1 : Array (ℕ × ℕ)) (x2 : Stratum) => x1 ++ x2.units) #[]).toList =
      S.toList.flatMap fun s => s.units.toList := by
  rw [← Array.foldl_toList]
  suffices ∀ acc : Array (ℕ × ℕ), (S.toList.foldl (fun x1 x2 => x1 ++ x2.units) acc).toList =
      acc.toList ++ S.toList.flatMap fun s => s.units.toList by simpa using this #[]
  induction S.toList with
  | nil => intro acc; simp
  | cons s ss ih => intro acc; simp [ih]

theorem foldl_units_work (S : Array WorkStratum) :
    (S.foldl (fun (x1 : Array (ℕ × ℕ)) (x2 : WorkStratum) => x1 ++ x2.units) #[]).toList =
      S.toList.flatMap fun s => s.units.toList := by
  rw [← Array.foldl_toList]
  suffices ∀ acc : Array (ℕ × ℕ), (S.toList.foldl (fun x1 x2 => x1 ++ x2.units) acc).toList =
      acc.toList ++ S.toList.flatMap fun s => s.units.toList by simpa using this #[]
  induction S.toList with
  | nil => intro acc; simp
  | cons s ss ih => intro acc; simp [ih]

theorem stratumK_le {k z pop : ℕ} (hz : 1 ≤ z) (hk : k ≤ pop) (hk1 : 1 ≤ k) : stratumK k z pop ≤ z := by
  unfold stratumK
  refine max_le hz (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
  have : k * z ≤ pop * z := Nat.mul_le_mul_right z hk
  nlinarith

/-! ### The checks `UnitSample.ofJson` makes of a stratified sampling design -/

section ofJson

open FlockSoundness.Refine Lean

open private stratumOfJson from Commitments.Draw

theorem ok_of_mem_mapM_list {α β : Type} (f : α → Flock.V β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → ∀ b ∈ r, ∃ a ∈ l, f a = .ok b
  | [], r, h, b, hb => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; simp at hb
  | a :: l, r, h, b, hb => by
    simp only [List.mapM_cons] at h
    obtain ⟨c, hc, h⟩ := ebind_ok h
    obtain ⟨cs, hcs, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, hc⟩
    · obtain ⟨a', ha', h'⟩ := ok_of_mem_mapM_list f l cs hcs b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

theorem ok_of_mem_mapM {α β : Type} (f : α → Flock.V β) (as : Array α) (bs : Array β) (h : as.mapM f = .ok bs) :
    ∀ b ∈ bs, ∃ a ∈ as, f a = .ok b := by
  obtain ⟨l⟩ := as
  rw [List.mapM_toArray] at h
  cases hl : l.mapM f with
  | error e => rw [hl] at h; cases h
  | ok r =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    intro b hb
    obtain ⟨a, ha, h'⟩ := ok_of_mem_mapM_list f l r hl b (by simpa using hb)
    exact ⟨a, by simpa using ha, h'⟩

/-- A stratum `stratumOfJson` accepts has non-empty units, each a non-empty range `lo < hi`. -/
theorem stratumOfJson_ranges {j : Json} {s : Stratum} (h : stratumOfJson j = .ok s) :
    s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2 := by
  unfold stratumOfJson at h
  simp only [throw_bind'] at h
  repeat' walk_step
  have hm := ‹Array.mapM _ _ = Except.ok _›
  have hne := ‹¬ (_ : Array (ℕ × ℕ)).isEmpty = true›
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  refine ⟨fun h0 => hne (Array.isEmpty_iff.2 h0), fun r hr => ?_⟩
  obtain ⟨a, -, h⟩ := ok_of_mem_mapM _ _ _ hm r hr
  clear hm hne
  repeat' walk_step
  have hlt := ‹¬ (_ : ℕ) ≥ _›
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact Nat.lt_of_not_ge hlt

/-- A sample file `UnitSample.ofJson` accepts under a stratified sampling design: `strataProblem` found nothing, and every stratum
passed `stratumOfJson`. -/
theorem ofJson_stratified {j : Json} {D : UnitSample} {k : ℕ} {S : Array Stratum} (h : UnitSample.ofJson j = .ok D)
    (hl : D.samplingDesign = .stratified k S) :
    strataProblem k S D.population = none ∧ ∀ s ∈ S, s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2 := by
  unfold UnitSample.ofJson at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    cases hl
  all_goals
    have hlaw := ‹(_ : Flock.V Flock.Draw.SamplingDesign) = Except.ok (Flock.Draw.SamplingDesign.stratified k S)›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq, Flock.Draw.SamplingDesign.stratified.injEq] at hlaw
  all_goals
    obtain ⟨-, rfl⟩ := ‹_ ∧ _ = S›
    have hp := ‹∀ (why : String), strataProblem _ _ _ = some why → False›
    refine ⟨Option.eq_none_iff_forall_ne_some.2 fun w hw => hp w hw, fun s hs => ?_⟩
    obtain ⟨a, -, ha⟩ := ok_of_mem_mapM _ _ _ ‹Array.mapM stratumOfJson _ = Except.ok _› s hs
    exact stratumOfJson_ranges ha

/-- A sample file `UnitSample.ofJson` accepts under a work sampling design: `workProblem` found nothing. -/
theorem ofJson_work {j : Json} {D : UnitSample} {k : ℕ} {S : Array WorkStratum} (h : UnitSample.ofJson j = .ok D)
    (hl : D.samplingDesign = .work k S) : workProblem k S D.population = none := by
  unfold UnitSample.ofJson at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    cases hl
  all_goals
    have hlaw := ‹(_ : Flock.V Flock.Draw.SamplingDesign) = Except.ok (Flock.Draw.SamplingDesign.work k S)›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq, Flock.Draw.SamplingDesign.work.injEq] at hlaw
  all_goals
    obtain ⟨-, rfl⟩ := ‹_ ∧ _ = S›
    have hp := ‹∀ (why : String), workProblem _ _ _ = some why → False›
    exact Option.eq_none_iff_forall_ne_some.2 fun w hw => hp w hw

open private strataMeetRate rateMiss from Commitments.Draw

theorem mapM_keep_list {α β γ : Type} (F : α × β → Flock.V (α × β × γ)) (hF : ∀ a b, F a = .ok b → (b.1, b.2.1) = a) :
    ∀ (l : List (α × β)) (r : List (α × β × γ)), l.mapM F = .ok r → r.map (fun x => (x.1, x.2.1)) = l
  | [], r, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; rfl
  | a :: l, r, h => by
    simp only [List.mapM_cons] at h
    obtain ⟨c, hc, h⟩ := ebind_ok h
    obtain ⟨cs, hcs, h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [List.map_cons, hF a c hc, mapM_keep_list F hF l cs hcs]

theorem mapM_keep {α β γ : Type} (F : α × β → Flock.V (α × β × γ)) (hF : ∀ a b, F a = .ok b → (b.1, b.2.1) = a)
    (as : Array (α × β)) (bs : Array (α × β × γ)) (h : as.mapM F = .ok bs) : bs.map (fun x => (x.1, x.2.1)) = as := by
  obtain ⟨l⟩ := as
  rw [List.mapM_toArray] at h
  cases hl : l.mapM F with
  | error e => rw [hl] at h; cases h
  | ok r =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simp [mapM_keep_list F hF l r hl]

/-- The population `meetsRate` reads: the strata, each with its template's amount. -/
theorem pop_facts {table : Array (String × ℕ × ℕ)} {strata : Array (String × Array (ℕ × ℕ))}
    {pop : Array (String × Array (ℕ × ℕ) × ℕ)} {F : String × Array (ℕ × ℕ) → Flock.V (String × Array (ℕ × ℕ) × ℕ)}
    (hF : ∀ a b, F a = .ok b → (b.1, b.2.1) = a ∧ amountOf table a.1 = .ok b.2.2) (hm : strata.mapM F = .ok pop) :
    pop.map (fun x => (x.1, x.2.1)) = strata ∧ ∀ x ∈ pop, amountOf table x.1 = .ok x.2.2 := by
  refine ⟨mapM_keep F (fun a b h => (hF a b h).1) strata pop hm, fun x hx => ?_⟩
  obtain ⟨a, -, ha⟩ := ok_of_mem_mapM F strata pop hm x hx
  obtain ⟨h1, h2⟩ := hF a x ha
  subst h1
  exact h2

theorem meetsRate_den {num den : ℕ} {table : Array (String × ℕ × ℕ)} {strata : Array (String × Array (ℕ × ℕ))}
    {d : Flock.Draw.SamplingDesign} (h : meetsRate num den table strata d = .ok ()) : 0 < den := by
  rcases Nat.eq_zero_or_pos den with rfl | hd
  · simp only [meetsRate, beq_self_eq_true, ↓reduceIte, throw_bind'] at h
    exact absurd h (throw_ne_ok _ _)
  · exact hd

private theorem strataMeetRate_ok {num den : ℕ} {pop : Array (String × Array (ℕ × ℕ) × ℕ)} {ss : Array Stratum}
    (h : strataMeetRate num den pop ss = .ok ()) :
    ss.size = pop.size ∧ ∀ s ∈ ss, ∃ rs c, (s.template, rs, c) ∈ pop ∧ s.units = rs ∧
      (s.k = s.size ∨ s.size * c * num ≤ s.k * den) := by
  have hsz : ss.size = pop.size := by
    by_contra hne
    have hb : (ss.size != pop.size) = true := by simpa using hne
    simp only [strataMeetRate, hb, ↓reduceIte, throw_bind'] at h
    exact absurd h (throw_ne_ok _ _)
  refine ⟨hsz, ?_⟩
  unfold strataMeetRate at h
  simp only [throw_bind'] at h
  repeat' walk_step
  have hl := ‹forIn _ _ _ = Except.ok _›
  clear h
  have key := forIn_range_forall (P := fun i => ∃ rs c, (ss[i]!.template, rs, c) ∈ pop ∧ ss[i]!.units = rs ∧
      (ss[i]!.k = ss[i]!.size ∨ ss[i]!.size * c * num ≤ ss[i]!.k * den)) ?step hl
  case step =>
    intro a _ _ st t h
    try dsimp only at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      refine ⟨⟨_, rfl⟩, ?_⟩
      have hf := ‹pop.find? _ = some _›
      have hu := ‹¬ (ss[a]!.units != _) = true›
      have hr := ‹rateMiss _ _ _ _ _ _ = Except.ok _›
      unfold rateMiss at hr
      split at hr
      · next hm =>
        simp only [meetsAt, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hm
        have hmem := Array.mem_of_find?_eq_some hf
        have hp := Array.find?_some hf
        simp at hp
        subst hp
        exact ⟨_, _, hmem, by simpa using hu, hm⟩
      · exact absurd hr (throw_ne_ok _ _)
  intro s hs
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hs
  have := key i (Nat.zero_le _) hi
  simpa only [getElem!_pos ss i hi] using this

/-- What `meetsRate` checks of a stratified design: its strata are the population's, one a template, each sampled
whole or meeting `n_s · c_s · NUM ≤ k_s · DEN` at its template's amount `c_s`. -/
theorem meetsRate_stratified {num den k : ℕ} {table : Array (String × ℕ × ℕ)}
    {strata : Array (String × Array (ℕ × ℕ))} {ss : Array Stratum}
    (h : meetsRate num den table strata (.stratified k ss) = .ok ()) :
    0 < den ∧ ss.size = strata.size ∧ ∀ s ∈ ss, ∃ c, (s.template, s.units) ∈ strata ∧
      amountOf table s.template = .ok c ∧ (s.k = s.size ∨ s.size * c * num ≤ s.k * den) := by
  refine ⟨meetsRate_den h, ?_⟩
  unfold meetsRate at h
  simp only [throw_bind'] at h
  repeat' walk_step
  try dsimp only at h
  have hm := ‹Array.mapM _ strata = Except.ok _›
  have hrest := h
  clear h
  obtain ⟨hkeep, hamt⟩ := pop_facts (fun a b h => by
    obtain ⟨t, rs⟩ := a
    dsimp only at h
    repeat' walk_step
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, ‹_›⟩) hm
  obtain ⟨hsz, hall⟩ := strataMeetRate_ok hrest
  refine ⟨by rw [← hkeep, Array.size_map]; exact hsz, fun s hs => ?_⟩
  obtain ⟨rs, c, hmem, hu, hr⟩ := hall s hs
  refine ⟨c, ?_, hamt _ hmem, hr⟩
  rw [← hkeep, hu]
  exact Array.mem_map.2 ⟨_, hmem, rfl⟩

/-- What `meetsRate` checks of a subset design: over the population's strata, each with its template's amount, the
`k` units are all `n` or meet `n · c · NUM ≤ k · DEN` at the largest amount `c`. -/
theorem meetsRate_subset {num den k : ℕ} {table : Array (String × ℕ × ℕ)} {strata : Array (String × Array (ℕ × ℕ))}
    (h : meetsRate num den table strata (.subset k) = .ok ()) :
    0 < den ∧ ∃ pop : Array (String × Array (ℕ × ℕ) × ℕ), pop.map (fun x => (x.1, x.2.1)) = strata ∧
      (∀ x ∈ pop, amountOf table x.1 = .ok x.2.2) ∧
      (k = pop.foldl (fun n x => n + ({ template := x.1, units := x.2.1, k := 0 } : Stratum).size) 0 ∨
        pop.foldl (fun n x => n + ({ template := x.1, units := x.2.1, k := 0 } : Stratum).size) 0 *
          pop.foldl (fun c x => max c x.2.2) 0 * num ≤ k * den) := by
  refine ⟨meetsRate_den h, ?_⟩
  unfold meetsRate at h
  simp only [throw_bind'] at h
  repeat' walk_step
  try dsimp only at h
  have hm := ‹Array.mapM _ strata = Except.ok _›
  have hr := ‹rateMiss _ _ _ _ _ _ = Except.ok _›
  clear h
  obtain ⟨hkeep, hamt⟩ := pop_facts (fun a b h => by
    obtain ⟨t, rs⟩ := a
    dsimp only at h
    repeat' walk_step
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, ‹_›⟩) hm
  unfold rateMiss at hr
  split at hr
  · next hc =>
    simp only [meetsAt, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
    exact ⟨_, hkeep, hamt, hc⟩
  · exact absurd hr (throw_ne_ok _ _)

end ofJson

end StrataExec

namespace SamplingDesign

open StrataExec

/-- A stratified sampling design the verifier accepts describes a partition: `strataProblem` finding nothing, with
`stratumOfJson`'s checks that each stratum's units are non-empty ranges `lo < hi`, gives the assignment `σ` of the
population's units to the strata that the strata describe, each stratum's `k` at most its size. -/
theorem execStrata_of_strataProblem {k pop : ℕ} {S : Array Stratum}
    (hlt : ∀ s ∈ S, ∀ r ∈ s.units, r.1 < r.2) (hne : ∀ s ∈ S, s.units ≠ #[]) (h : strataProblem k S pop = none) :
    ∃ σ : Fin pop → Fin S.size, ExecStrata σ (fun s => S[s].k) (fun s => S[s]) ∧
      ∀ s, S[s].k ≤ (stratum σ s).card := by
  obtain ⟨-, -, hks, hc⟩ := strataProblem_none h
  set Q := (S.foldl (fun (x1 : Array (ℕ × ℕ)) (x2 : Stratum) => x1 ++ x2.units) #[]).qsort
    fun (a b : ℕ × ℕ) => decide (a.fst < b.fst) with hQ
  have hperm := StrataExec.qsort_perm (fun (a b : ℕ × ℕ) => decide (a.fst < b.fst))
    (S.foldl (fun (x1 : Array (ℕ × ℕ)) (x2 : Stratum) => x1 ++ x2.units) #[])
  rw [StrataExec.foldl_units] at hperm
  have hmemL : ∀ r ∈ (S.toList.flatMap fun s => s.units.toList), r.1 < r.2 := by
    intro r hr
    obtain ⟨s, hs, hr⟩ := List.mem_flatMap.1 hr
    exact hlt s (Array.mem_def.2 hs) r (Array.mem_def.2 hr)
  obtain ⟨-, hin, hcnt⟩ := StrataExec.chain_facts 0 pop Q.toList hc fun r hr => hmemL r (hperm.subset hr)
  let c : Fin S.size → ℕ → ℕ := fun i u => (StrataExec.flat S[i].units.toList).count u
  have hsum : ∀ u, ∑ i, c i u = if u < pop then 1 else 0 := by
    intro u
    have h1 := (hperm.countP_eq (StrataExec.hits u)).symm.trans (hcnt u)
    rw [List.countP_flatMap] at h1
    simp only [Nat.zero_le, true_and] at h1
    rw [← h1]
    simp only [c, StrataExec.count_flat]
    rw [← List.sum_ofFn]
    congr 1
    apply List.ext_getElem <;> simp
  have hbound : ∀ (i : Fin S.size) u, u ∈ StrataExec.flat S[i].units.toList → u < pop := by
    intro i u hu
    obtain ⟨r, hr, hu⟩ := List.mem_flatMap.1 hu
    have hrQ : r ∈ Q.toList := hperm.symm.subset
      (List.mem_flatMap.2 ⟨S[i], Array.mem_def.1 (Array.getElem_mem _), hr⟩)
    have := (hin r hrQ).2
    simp [List.mem_range'] at hu
    omega
  have huniq : ∀ i j u, c i u ≠ 0 → c j u ≠ 0 → i = j := by
    intro i j u hi hj
    by_contra hij
    have h2 : c i u + c j u ≤ ∑ l, c l u := by
      rw [← Finset.sum_pair (f := fun l => c l u) hij]
      exact Finset.sum_le_sum_of_subset (Finset.subset_univ _)
    rw [hsum u] at h2
    split at h2 <;> omega
  have hex : ∀ u : Fin pop, ∃ i, c i u ≠ 0 := by
    intro u
    have : ∑ i, c i u ≠ 0 := by rw [hsum u, ite_eq_left u.isLt]; exact one_ne_zero
    obtain ⟨i, -, hi⟩ := Finset.exists_ne_zero_of_sum_ne_zero this
    exact ⟨i, hi⟩
  classical
  let σ : Fin pop → Fin S.size := fun u => (hex u).choose
  have hσ : ∀ u i, σ u = i ↔ c i u ≠ 0 := fun u i =>
    ⟨fun h => h ▸ (hex u).choose_spec, fun h => huniq _ _ _ (hex u).choose_spec h⟩
  have hset : ∀ i : Fin S.size, (stratum σ i).map Fin.valEmbedding = (StrataExec.flat S[i].units.toList).toFinset := by
    intro i
    ext x
    simp only [stratum, Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply,
      List.mem_toFinset]
    constructor
    · rintro ⟨a, ha, rfl⟩
      exact List.count_pos_iff.1 (Nat.pos_of_ne_zero ((hσ a i).1 ha))
    · intro hx
      exact ⟨⟨x, hbound i x hx⟩, (hσ _ i).2 (Nat.pos_iff_ne_zero.1 (List.count_pos_iff.2 hx)), rfl⟩
  have hnodup : ∀ i : Fin S.size, (StrataExec.flat S[i].units.toList).Nodup := by
    intro i
    refine List.nodup_iff_count_le_one.2 fun u => ?_
    have h1 : (StrataExec.flat S[i].units.toList).count u ≤ ∑ j, c j u :=
      Finset.single_le_sum (f := fun j => c j u) (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
    rw [hsum u] at h1
    split at h1 <;> omega
  have hcard : ∀ i : Fin S.size, S[i].size = (stratum σ i).card := fun i => by
    rw [← Finset.card_map Fin.valEmbedding, hset i, List.toFinset_card_of_nodup (hnodup i), StrataExec.size_eq]
  refine ⟨σ, fun i => ⟨rfl, hcard i, ?_⟩, fun i => ?_⟩
  swap
  · rw [← hcard i]
    exact (hks _ (Array.getElem_mem _)).2
  · rw [hset i]
    ext x
    simp only [Finset.mem_image, Finset.mem_range, List.mem_toFinset, unit_eq, StrataExec.size_eq]
    constructor
    · rintro ⟨j, hj, rfl⟩
      rw [unitL_getElem _ _ hj]
      exact List.getElem_mem hj
    · intro hx
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hx
      exact ⟨j, hj, unitL_getElem _ _ hj⟩

/-- **The strata of an accepted sample file describe a partition**: a sample file `UnitSample.ofJson` accepts under a stratified
sampling design `stratified k S` has an assignment `σ` of its population's units to the strata `S` that `S` describes, each
stratum's `k` at most its size. -/
theorem execStrata_of_ofJson {j : Lean.Json} {D : UnitSample} {k : ℕ} {S : Array Stratum}
    (h : UnitSample.ofJson j = .ok D) (hl : D.samplingDesign = .stratified k S) :
    ∃ σ : Fin D.population → Fin S.size, ExecStrata σ (fun s => S[s].k) (fun s => S[s]) ∧
      ∀ s, S[s].k ≤ (stratum σ s).card := by
  obtain ⟨hp, hS⟩ := ofJson_stratified h hl
  exact execStrata_of_strataProblem (fun s hs => (hS s hs).2) (fun s hs => (hS s hs).1) hp

end SamplingDesign

end Spot.Sampling
