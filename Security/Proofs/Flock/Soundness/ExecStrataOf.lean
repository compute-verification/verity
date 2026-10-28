import Proofs.Flock.Soundness.ExecStratified
import Proofs.Flock.Soundness.Refine.Setup

/-!
# The strata the verifier accepts are a partition's

`Law.execStrata_of_ofJson`: a draw file `Flock.Draw.UnitDraw.ofJson` accepts under a stratified law has an assignment `σ`
of its population's units to the strata that the strata describe (`ExecStrata`), each stratum's `k` at most its size. The
hypotheses `ExecStrata σ k S` and `k s ≤ (stratum σ s).card` that the stratified e2e theorems take are so not conditions
on the statement's form: they follow from the verifier accepting.

* `StrataExec.ofJson_stratified`: what `ofJson` checks of a stratified law: `strataProblem` finds nothing, and each stratum
  passed `stratumOfJson`, whose units are non-empty ranges `lo < hi` (`stratumOfJson_ranges`).
* `StrataExec.strataProblem_none`: what `strataProblem` checks: `1 ≤ k ≤ pop`, each stratum's `k` the rule's
  (`stratumK`), and the ranges, sorted by their starts, chain from `0` to `pop` (`Chain`).
* `StrataExec.chain_facts`: a chain of non-empty ranges from `a` to `b` covers each unit of `[a, b)` exactly once.
* `Law.execStrata_of_strataProblem`: those checks give `σ`, each unit's stratum the one whose ranges hold it.
-/

namespace FlockSoundness.Audit

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

theorem qsort_perm {α : Type} (lt : α → α → Bool) (as : Array α) : (as.qsort lt).toList.Perm as.toList := by
  unfold Array.qsort
  split
  · exact List.Perm.refl _
  · exact Array.perm_iff_toList_perm.1 (DrawExec.sort_perm lt _ _ _ _ _ _ _ rfl)

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

theorem strataProblem_none {k : Nat} {S : Array Stratum} {pop : Nat} (h : strataProblem k S pop = none) :
    1 ≤ k ∧ k ≤ pop ∧ (∀ s ∈ S, s.k = stratumK k s.size pop) ∧
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
              if (S[i]!.k != stratumK k S[i]!.size pop) = true then
                pure
                  (ForInStep.done
                    (some
                        (some
                          (toString "U2: stratum '" ++ toString S[i]!.template ++ toString "': k is not the rule's")),
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
            · next hne => simpa [getElem!_pos S i hi] using hne
        · split at h
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

theorem stratumK_le {k z pop : ℕ} (hz : 1 ≤ z) (hk : k ≤ pop) (hk1 : 1 ≤ k) : stratumK k z pop ≤ z := by
  unfold stratumK
  refine max_le hz (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_))
  have : k * z ≤ pop * z := Nat.mul_le_mul_right z hk
  nlinarith

/-! ### The checks `UnitDraw.ofJson` makes of a stratified law -/

section ofJson

open FlockSoundness.Refine Lean

open private stratumOfJson from Flock.Draw

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

/-- A draw file `UnitDraw.ofJson` accepts under a stratified law: `strataProblem` found nothing, and every stratum
passed `stratumOfJson`. -/
theorem ofJson_stratified {j : Json} {D : UnitDraw} {k : ℕ} {S : Array Stratum} (h : UnitDraw.ofJson j = .ok D)
    (hl : D.law = .stratified k S) :
    strataProblem k S D.population = none ∧ ∀ s ∈ S, s.units ≠ #[] ∧ ∀ r ∈ s.units, r.1 < r.2 := by
  unfold UnitDraw.ofJson at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    cases hl
  all_goals
    have hlaw := ‹(_ : Flock.V Flock.Draw.Law) = Except.ok (Flock.Draw.Law.stratified k S)›
    simp only [pure, Except.pure, Except.ok.injEq, reduceCtorEq, Flock.Draw.Law.stratified.injEq] at hlaw
  all_goals
    obtain ⟨-, rfl⟩ := ‹_ ∧ _ = S›
    have hp := ‹∀ (why : String), strataProblem _ _ _ = some why → False›
    refine ⟨Option.eq_none_iff_forall_ne_some.2 fun w hw => hp w hw, fun s hs => ?_⟩
    obtain ⟨a, -, ha⟩ := ok_of_mem_mapM _ _ _ ‹Array.mapM stratumOfJson _ = Except.ok _› s hs
    exact stratumOfJson_ranges ha

end ofJson

end StrataExec

namespace Law

open StrataExec

/-- A stratified law the verifier accepts describes a partition: `strataProblem` finding nothing, with
`stratumOfJson`'s checks that each stratum's units are non-empty ranges `lo < hi`, gives the assignment `σ` of the
population's units to the strata that the strata describe, each stratum's `k` at most its size. -/
theorem execStrata_of_strataProblem {k pop : ℕ} {S : Array Stratum}
    (hlt : ∀ s ∈ S, ∀ r ∈ s.units, r.1 < r.2) (hne : ∀ s ∈ S, s.units ≠ #[]) (h : strataProblem k S pop = none) :
    ∃ σ : Fin pop → Fin S.size, ExecStrata σ (fun s => S[s].k) (fun s => S[s]) ∧
      ∀ s, S[s].k ≤ (stratum σ s).card := by
  obtain ⟨hk1, hkp, hks, hc⟩ := strataProblem_none h
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
  · have hmem : S[i] ∈ S := Array.getElem_mem _
    rw [← hcard i, hks _ hmem]
    refine stratumK_le ?_ hkp hk1
    rw [StrataExec.size_eq]
    obtain ⟨r, rs, hrs⟩ := List.exists_cons_of_ne_nil (fun h0 => hne _ hmem (Array.toList_eq_nil_iff.1 h0))
    have hr := hlt _ hmem r (Array.mem_def.2 (hrs ▸ List.mem_cons_self))
    rw [hrs]
    simp [StrataExec.flat]
    omega
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

/-- **The strata of an accepted draw file describe a partition**: a draw file `UnitDraw.ofJson` accepts under a stratified
law `stratified k S` has an assignment `σ` of its population's units to the strata `S` that `S` describes, each
stratum's `k` at most its size. -/
theorem execStrata_of_ofJson {j : Lean.Json} {D : UnitDraw} {k : ℕ} {S : Array Stratum}
    (h : UnitDraw.ofJson j = .ok D) (hl : D.law = .stratified k S) :
    ∃ σ : Fin D.population → Fin S.size, ExecStrata σ (fun s => S[s].k) (fun s => S[s]) ∧
      ∀ s, S[s].k ≤ (stratum σ s).card := by
  obtain ⟨hp, hS⟩ := ofJson_stratified h hl
  exact execStrata_of_strataProblem (fun s hs => (hS s hs).2) (fun s hs => (hS s hs).1) hp

end Law

end FlockSoundness.Audit
