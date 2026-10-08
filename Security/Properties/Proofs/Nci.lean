import Properties.Proofs.Nci.Lemmas

/-!
# The NCI check's properties, proved

Each statement of `Properties.Nci`, proved of `NciCheck.evaluate` from the lemmas about its steps
(`Properties.Proofs.Nci.Lemmas`).
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json toJson)
open Flock.Program (Prog)

theorem parses_unique {j : Json} {i i' : Instance} (h : Parses j i) (h' : Parses j i') : i = i' := by
  have h1 := (parse_ok j i).2 h
  rw [(parse_ok j i').2 h', Except.ok.injEq] at h1
  exact h1.symm

/-- **Each refusal is its condition.** -/
theorem RejectsExactly : Properties.RejectsExactly := by
  intro j k
  rw [evaluate_error, parse_error, Malformed, or_assoc]
  refine or_congr_right (or_congr_right (exists_congr fun i => ?_))
  rw [parse_ok]
  refine and_congr_right fun _ => or_congr_right (exists_congr fun P => and_congr_right fun _ => ?_)
  rw [circuitOf_error]
  refine or_congr_right ⟨?_, ?_⟩
  · rintro ⟨c, hc, hd⟩
    exact ⟨(circuitOf_isOk P).1 ⟨c, hc⟩, (divisionOf_error hc _ _ k).1 hd⟩
  · rintro ⟨hg, hd⟩
    obtain ⟨c, hc⟩ := (circuitOf_isOk P).2 hg
    exact ⟨c, hc, (divisionOf_error hc _ _ k).2 hd⟩

/-- **A verdict exactly on a well-formed instance.** -/
theorem AcceptsExactly : Properties.AcceptsExactly := by
  intro j
  constructor
  · rintro ⟨out, h⟩
    obtain ⟨i, P, c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 h
    obtain ⟨ht, hs, ho, he⟩ := (divisionOf_isOk h3 _ _).1 ⟨dv, h4⟩
    exact ⟨i, P, (parse_ok j i).1 h1, h2, (circuitOf_isOk P).1 ⟨c, h3⟩, ht, hs, ho, he⟩
  · rintro ⟨i, P, hi, hP, hg, ht, hs, ho, he⟩
    obtain ⟨c, hc⟩ := (circuitOf_isOk P).2 hg
    obtain ⟨dv, hd⟩ := (divisionOf_isOk hc _ _).2 ⟨ht, hs, ho, he⟩
    exact ⟨_, (evaluate_ok j _).2 ⟨i, P, c, dv, (parse_ok j i).2 hi, hP, hc, hd, rfl⟩⟩

theorem check_error (j : Json) (k : String) :
    (check j).getObjVal? "error" = .ok (.str k) ↔ evaluate j = .error k := by
  unfold check
  rcases h : evaluate j with e | out
  · show Except.ok (Json.str e) = _ ↔ _
    simp [eq_comm]
  · obtain ⟨i, P, c, dv, -, -, -, -, rfl⟩ := (evaluate_ok j out).1 h
    show Except.error _ = _ ↔ _
    simp

/-- **`check` refuses exactly the malformed.** -/
theorem CheckRefuses : Properties.CheckRefuses := fun j k => (check_error j k).trans (RejectsExactly j k)

/-- **`check` gives a verdict exactly on a well-formed instance.** -/
theorem CheckAccepts : Properties.CheckAccepts := by
  intro j
  rw [← AcceptsExactly j]
  simp only [ne_eq, check_error]
  rcases evaluate j with e | out <;> simp

/-- **What a verdict is about.** -/
theorem Decodes : Properties.Decodes := by
  intro j out h
  obtain ⟨i, P, c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 h
  refine ⟨i, P, (parse_ok j i).1 h1, h2, circuitFrom h3, laysOut_circuitFrom h3, ?_,
    dv.n, exists_partition h3 h4 (laysOut_circuitFrom h3)⟩
  have hmem := (divisionOf_ok h3 h4).2.1
  refine ⟨Finset.univ.filter fun g => g.val ∈ dv.theta, fun g => ?_, fun g hg => ?_⟩
  · rw [Finset.mem_filter, mem_iff_index, ← hmem]
    simp
  · refine theta_inputs h3 h4 (laysOut_circuitFrom h3) g ?_
    rw [mem_iff_index, ← hmem]
    exact (Finset.mem_filter.1 hg).2

/-- One step of `failure`: IU `u`'s verdict. -/
def failStep (gamma num den : ℕ) (r : Array (ℕ × ℕ)) (u : ℕ) : Option (ℕ × String) :=
  if r[u]!.1 > gamma then some (u, "size") else if r[u]!.2 * den > num * r[u]!.1 then some (u, "nci") else none

theorem failure_eq (gamma num den : ℕ) (r : Array (ℕ × ℕ)) :
    failure gamma num den r = (List.range r.size).findSome? (failStep gamma num den r) := rfl

theorem failStep_fst (gamma num den : ℕ) (r : Array (ℕ × ℕ)) (u v : ℕ) (rule : String)
    (h : failStep gamma num den r u = some (v, rule)) : v = u := by
  unfold failStep at h
  split_ifs at h <;> simp only [Option.some.injEq, Prod.mk.injEq] at h <;> exact h.1.symm

namespace Setting

variable {P : Prog} {theta owner : Array Json} {c : Circuit} {dv : Division} {d : Verity.Declaration} {n : ℕ}
  {U : Verity.Partition d.D n} (S : Setting P theta owner c dv d n U)
include S

theorem failStep_none (gamma num den : ℕ) (hden : 0 < den) (u : Fin n) :
    failStep gamma num den (ius c dv) u.val = none ↔ Meets d U gamma ((num : ℝ) / den) u := by
  rw [failStep, S.ius_get u, Meets, le_ratio_iff _ _ _ _ hden]
  dsimp only
  split_ifs with h1 h2 <;> simp <;> omega

theorem failStep_some (gamma num den : ℕ) (hden : 0 < den) (u : Fin n) (v : ℕ) (rule : String) :
    failStep gamma num den (ius c dv) u.val = some (v, rule) ↔ v = u.val ∧
      ((rule = "size" ∧ gamma < d.D.size (U.gates u)) ∨
       (rule = "nci" ∧ d.D.size (U.gates u) ≤ gamma ∧
         (num : ℝ) / den * d.D.size (U.gates u) < (Nci.novelReads d (U.gates u) : ℝ))) := by
  rw [failStep, S.ius_get u, ratio_lt_iff _ _ _ _ hden]
  dsimp only
  split_ifs with h1 h2
  · simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨rfl, rfl⟩; exact ⟨rfl, Or.inl ⟨rfl, h1⟩⟩
    · rintro ⟨rfl, (⟨rfl, -⟩ | ⟨rfl, h, -⟩)⟩
      · exact ⟨rfl, rfl⟩
      · omega
  · simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨rfl, rfl⟩; exact ⟨rfl, Or.inr ⟨rfl, by omega, by omega⟩⟩
    · rintro ⟨rfl, (⟨rfl, h⟩ | ⟨rfl, -, -⟩)⟩
      · omega
      · exact ⟨rfl, rfl⟩
  · simp only [false_iff, not_and, not_or]
    intro _
    exact ⟨fun _ h => h1 h, fun _ _ h => h2 (by omega)⟩

theorem failure_none (gamma num den : ℕ) (hden : 0 < den) :
    failure gamma num den (ius c dv) = none ↔ ∀ u, Meets d U gamma ((num : ℝ) / den) u := by
  rw [failure_eq, findSome_range_none, S.ius_size]
  constructor
  · intro h u
    exact (S.failStep_none gamma num den hden u).1 (h u.val u.isLt)
  · intro h u hu
    exact (S.failStep_none gamma num den hden ⟨u, hu⟩).2 (h ⟨u, hu⟩)

theorem failure_some (gamma num den : ℕ) (hden : 0 < den) (u : Fin n) (rule : String) :
    failure gamma num den (ius c dv) = some (u.val, rule) ↔
      (∀ u' < u, Meets d U gamma ((num : ℝ) / den) u') ∧
        ((rule = "size" ∧ gamma < d.D.size (U.gates u)) ∨
         (rule = "nci" ∧ d.D.size (U.gates u) ≤ gamma ∧
           (num : ℝ) / den * d.D.size (U.gates u) < (Nci.novelReads d (U.gates u) : ℝ))) := by
  rw [failure_eq, findSome_range_some, S.ius_size]
  constructor
  · rintro ⟨i, hi, hs, hj⟩
    have hiu := failStep_fst gamma num den _ i u.val rule hs
    subst hiu
    refine ⟨fun u' hu' => (S.failStep_none gamma num den hden u').1 (hj u'.val hu'), ?_⟩
    exact ((S.failStep_some gamma num den hden u u.val rule).1 hs).2
  · rintro ⟨hj, hr⟩
    refine ⟨u.val, u.isLt, (S.failStep_some gamma num den hden u u.val rule).2 ⟨rfl, hr⟩, fun j hj' => ?_⟩
    exact (S.failStep_none gamma num den hden ⟨j, by omega⟩).2 (hj ⟨j, by omega⟩ hj')

end Setting

theorem toJson_nat_inj {a b : ℕ} (h : toJson a = toJson b) : a = b := by
  simp [toJson, Lean.JsonNumber.fromNat] at h
  exact h

/-- **The verdict is the definition's.** -/
theorem Reports : Properties.Reports := by
  intro j out i P d n U hev hpar hdec hL hθ hU
  obtain ⟨i', P', c, dv, h1, h2, h3, h4, rfl⟩ := (evaluate_ok j out).1 hev
  obtain rfl := parses_unique ((parse_ok j i').1 h1) hpar
  rw [hdec, Except.ok.injEq] at h2
  subst h2
  have S : Setting P i'.theta i'.owner c dv d n U := ⟨h3, h4, hL, hθ, hU⟩
  have hden : 0 < i'.den := hpar.2.2.2.2.2
  intro α
  have hcompl : (verdict c.N (ius c dv) (failure i'.gamma i'.num i'.den (ius c dv))).getObjVal? "complies" =
      .ok (.bool (failure i'.gamma i'.num i'.den (ius c dv)).isNone) := rfl
  have hfail : (verdict c.N (ius c dv) (failure i'.gamma i'.num i'.den (ius c dv))).getObjVal? "failure" =
      .ok (match failure i'.gamma i'.num i'.den (ius c dv) with
        | none => Json.null
        | some (u, rule) => Json.arr #[toJson u, Json.str rule]) := rfl
  have hnone := S.failure_none i'.gamma i'.num i'.den hden
  refine ⟨?_, ?_, ?_, ?_, ?_, fun u rule => ?_⟩
  · rw [S.N_eq]; rfl
  · show Except.ok (Json.arr ((ius c dv).map fun (s, x) => Json.arr #[toJson s, toJson x])) = _
    congr 2
    apply Array.ext
    · simpa using S.ius_size
    · intro k hk1 hk2
      have hk : k < n := by simpa using hk2
      rw [Array.getElem_map, Array.getElem_ofFn, ← getElem!_pos _ k (by simpa using hk1)]
      rw [S.ius_get ⟨k, hk⟩]
  · rw [hcompl, ← hnone]
    cases failure i'.gamma i'.num i'.den (ius c dv) <;> simp
  · rw [hcompl, ← hnone]
    cases failure i'.gamma i'.num i'.den (ius c dv) <;> simp
  · rw [hfail, ← hnone]
    rcases failure i'.gamma i'.num i'.den (ius c dv) with _ | ⟨v, r⟩ <;> simp
  · rw [hfail, ← S.failure_some i'.gamma i'.num i'.den hden u rule]
    rcases failure i'.gamma i'.num i'.den (ius c dv) with _ | ⟨v, r⟩
    · simp
    · simp only [Except.ok.injEq, Json.arr.injEq, Option.some.injEq, Prod.mk.injEq]
      constructor
      · intro h
        simp at h
        exact ⟨toJson_nat_inj h.1, h.2⟩
      · rintro ⟨rfl, rfl⟩; rfl

/-- **A verdict of compliance witnesses the policy.** -/
theorem Witnesses : Properties.Witnesses := by
  intro j out i P d hev hpar hdec hL hθ hc
  obtain ⟨i', P', c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 hev
  obtain rfl := parses_unique ((parse_ok j i').1 h1) hpar
  rw [hdec, Except.ok.injEq] at h2
  subst h2
  obtain ⟨U, hU⟩ := exists_partition h3 h4 hL
  exact ⟨dv.n, U, (Reports j out i' P d dv.n U hev hpar hdec hL hθ hU).2.2.1.1 hc⟩

end NciCheck.Properties.SecurityProofs
