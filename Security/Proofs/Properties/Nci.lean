import Proofs.Properties.Nci.Lemmas
import Proofs.Properties.Nci.Desc

/-!
# The NCI check's properties, proved

Each statement of `Properties.Nci`, proved of `NciCheck.evaluate` from the lemmas about its steps
(`Proofs.Properties.Nci.Lemmas`), and the description-sized check's summaries against it (`sizes_eq_tally`, with
`Proofs.Properties.Nci.Desc`).
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json toJson)
open Flock.Program (Prog GateTable)

theorem parses_unique {j : Json} {i i' : Instance} (h : Parses j i) (h' : Parses j i') : i = i' := by
  have h1 := (parse_ok j i).2 h
  rw [(parse_ok j i').2 h', Except.ok.injEq] at h1
  exact h1.symm

/-- `evaluate j` refuses with `k` exactly when `Malformed j k`. -/
theorem refused_iff_malformed (j : Json) (k : String) : evaluate j = .error k ↔ Malformed j k := by
  rw [evaluate_error, parse_error, Malformed, or_assoc]
  refine or_congr_right (or_congr_right (exists_congr fun i => ?_))
  rw [parse_ok]
  refine and_congr_right fun _ => decodeProgram_cases _ _ _ _ fun P => ?_
  rw [circuitOf_error]
  refine or_congr_right ⟨?_, ?_⟩
  · rintro ⟨c, hc, hd⟩
    exact ⟨(circuitOf_isOk P).1 ⟨c, hc⟩, (divisionOf_error hc _ _ k).1 hd⟩
  · rintro ⟨hg, hd⟩
    obtain ⟨c, hc⟩ := (circuitOf_isOk P).2 hg
    exact ⟨c, hc, (divisionOf_error hc _ _ k).2 hd⟩

/-- `evaluate j` gives a verdict exactly when `WellFormed j`. -/
theorem verdict_iff_wellFormed (j : Json) : (∃ out, evaluate j = .ok out) ↔ WellFormed j := by
  constructor
  · rintro ⟨out, h⟩
    obtain ⟨i, P, c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 h
    obtain ⟨hP, hr⟩ := (decodeProgram_ok _ _).1 h2
    obtain ⟨ht, hs, ho, he⟩ := (divisionOf_isOk h3 _ _).1 ⟨dv, h4⟩
    exact ⟨i, P, (parse_ok j i).1 h1, hP, hr, (circuitOf_isOk P).1 ⟨c, h3⟩, ht, hs, ho, he⟩
  · rintro ⟨i, P, hi, hP, hr, hg, ht, hs, ho, he⟩
    obtain ⟨c, hc⟩ := (circuitOf_isOk P).2 hg
    obtain ⟨dv, hd⟩ := (divisionOf_isOk hc _ _).2 ⟨ht, hs, ho, he⟩
    exact ⟨_, (evaluate_ok j _).2 ⟨i, P, c, dv, (parse_ok j i).2 hi, (decodeProgram_ok _ _).2 ⟨hP, hr⟩, hc, hd, rfl⟩⟩

theorem refusalWith_error (e : String) (m : Option String) :
    (refusalWith e m).getObjVal? "error" = .ok (.str e) := by
  unfold refusalWith
  rcases m with _ | m <;> rfl

theorem refusal_error (e : String) : (refusal e).getObjVal? "error" = .ok (.str e) :=
  refusalWith_error e _

theorem refusalOf_error (j : Json) (e : String) : (refusalOf j e).getObjVal? "error" = .ok (.str e) := by
  unfold refusalOf
  split
  · exact refusalWith_error _ _
  · exact refusal_error _

theorem check_error (j : Json) (k : String) :
    (check j).getObjVal? "error" = .ok (.str k) ↔ evaluate j = .error k := by
  unfold check
  rcases h : evaluate j with e | out
  · show (refusalOf j e).getObjVal? "error" = _ ↔ _
    rw [refusalOf_error]
    simp [eq_comm]
  · obtain ⟨i, P, c, dv, -, -, -, -, rfl⟩ := (evaluate_ok j out).1 h
    show Except.error _ = _ ↔ _
    simp

theorem check_verdict_iff_wellFormed (j : Json) :
    (∀ k, (check j).getObjVal? "error" ≠ .ok (.str k)) ↔ WellFormed j := by
  rw [← verdict_iff_wellFormed j]
  simp only [ne_eq, check_error]
  rcases evaluate j with e | out <;> simp

/-- **A malformed instance is refused with its condition's code.** -/
theorem RejectsExactly : Properties.RejectsExactly :=
  fun j k h hm => h ((refused_iff_malformed j k).2 hm)

/-- **A verdict only on a well-formed instance.** -/
theorem AcceptsExactly : Properties.AcceptsExactly :=
  fun j out h => (verdict_iff_wellFormed j).1 ⟨out, h⟩

/-- **`check` refuses a malformed instance with its condition's code.** -/
theorem CheckRefuses : Properties.CheckRefuses :=
  fun j k h hm => h ((check_error j k).2 ((refused_iff_malformed j k).2 hm))

/-- **`check` gives a verdict only on a well-formed instance.** -/
theorem CheckAccepts : Properties.CheckAccepts :=
  fun j h => (check_verdict_iff_wellFormed j).1 h

/-- **What a verdict is about.** -/
theorem Decodes : Properties.Decodes := by
  intro j out h
  obtain ⟨i, P, c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 h
  refine ⟨i, P, (parse_ok j i).1 h1, ((decodeProgram_ok _ _).1 h2).1, circuitFrom h3, laysOut_circuitFrom h3, ?_,
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

variable {P : Prog} {theta owner : Array Json} {c : GateTable} {dv : Division} {d : Verity.Registered} {n : ℕ}
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
  intro j out hev i P d hpar hdec hL hθ hU
  obtain ⟨i', P', c, dv, h1, h2, h3, h4, rfl⟩ := (evaluate_ok j out).1 hev
  obtain rfl := parses_unique ((parse_ok j i').1 h1) hpar
  replace h2 := ((decodeProgram_ok _ _).1 h2).1
  rw [hdec, Except.ok.injEq] at h2
  subst h2
  have S : Setting P i'.theta i'.owner c dv d d.n d.𝓘 := ⟨h3, h4, hL, hθ, hU⟩
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
      have hk : k < d.n := by simpa using hk2
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
  intro j out hev i P d hpar hdec hL hθ hc
  obtain ⟨i', P', c, dv, h1, h2, h3, h4, -⟩ := (evaluate_ok j out).1 hev
  obtain rfl := parses_unique ((parse_ok j i').1 h1) hpar
  replace h2 := ((decodeProgram_ok _ _).1 h2).1
  rw [hdec, Except.ok.injEq] at h2
  subst h2
  obtain ⟨U, hU⟩ := exists_partition h3 h4 hL
  refine ⟨⟨dv.n, U, (Reports j out hev i' P (.ofPartition d.D d.θ d.θ_input dv.n U) hpar hdec hL hθ hU).2.2.1.1 hc⟩,
    fun hI => (Reports j out hev i' P d hpar hdec hL hθ hI).2.2.1.1 hc⟩

/-! ## Completeness

The converses of the refusals and of the verdict: the check refuses only the malformed, and gives a verdict on every
well-formed instance. Each comes with an instance where its case holds. A refusal's is concrete (`Json.null`, which has
no fields). A well-formed instance needs a program the decoder decodes, and the decoder is `partial` (`Prog.gate` and
the descriptor's resolution), so the kernel can't run it: the instance takes what the decoder returns as a hypothesis,
a Program whose canonical layout has no gates, with θ and the owners empty. -/

/-- **A refusal is its condition** (the converse of `RejectsExactly`). -/
theorem malformed_of_refused (j : Json) (k : String) (h : evaluate j = .error k) : Malformed j k :=
  (refused_iff_malformed j k).1 h

/-- **A well-formed instance gets a verdict** (the converse of `AcceptsExactly`). -/
theorem verdict_of_wellFormed (j : Json) (h : WellFormed j) : ∃ out, evaluate j = .ok out :=
  (verdict_iff_wellFormed j).2 h

/-- **`check`'s refusal is its condition** (the converse of `CheckRefuses`). -/
theorem check_malformed_of_refused (j : Json) (k : String) (h : (check j).getObjVal? "error" = .ok (.str k)) :
    Malformed j k :=
  malformed_of_refused j k ((check_error j k).1 h)

/-- **`check` gives a verdict on a well-formed instance** (the converse of `CheckAccepts`). -/
theorem check_verdict_of_wellFormed (j : Json) (h : WellFormed j) :
    ∀ k, (check j).getObjVal? "error" ≠ .ok (.str k) :=
  (check_verdict_iff_wellFormed j).2 h

/-- **A refusal's case holds**: `Json.null` is refused, with `instance`. -/
theorem refused_instance : evaluate Json.null = .error "instance" := rfl

/-- `check` refuses `Json.null` with `instance`. -/
theorem check_refused_instance : (check Json.null).getObjVal? "error" = .ok (.str "instance") :=
  (check_error _ _).2 refused_instance

/-- The instance of program text `s` with θ and the owners empty, `Γ = 0` and `α = 0 / 1`. -/
def noGates (s : String) : Json :=
  Json.mkObj [("program", .str s), ("theta", .arr #[]), ("owner", .arr #[]), ("gamma", toJson (0 : ℕ)),
    ("alpha", .arr #[toJson (0 : ℕ), toJson (1 : ℕ)])]

/-- **A well-formed instance's case holds**: where the decoder decodes `s` to a Program with no gates, and the width
analysis finds no reference read at another width, `noGates s` is well-formed. -/
theorem wellFormed_instance {s : String} {P : Prog} (hP : Flock.Program.decode s.toUTF8 = .ok P)
    (hr : ¬ Reinterprets P) (h0 : gates P = 0) : WellFormed (noGates s) := by
  refine ⟨⟨s, #[], #[], 0, 0, 1⟩, P, ⟨rfl, rfl, rfl, ⟨_, rfl, rfl⟩, ⟨_, _, rfl, rfl, rfl⟩, Nat.one_pos⟩, hP, hr,
    fun g hg => absurd hg (by omega), fun t ht => absurd ht (by simp), by simp [h0], fun g hg => absurd hg (by omega), ?_⟩
  rintro ⟨g, u, hg, -⟩
  exact absurd hg (by simp)

/-! ## The description-sized check -/

/-- **`NciCheck.Desc.sizes` is the unrolled tally.** On a division that is the expansion of the root nodes' IUs `own`
(root node `k` holding gates `[offs[k], offs[k + 1])`, `offs` from `0` to the circuit's `N`, each of them in IU
`own[k]`), the description-sized check's size of each IU is the one `NciCheck.tally` counts gate by gate. -/
theorem sizes_eq_tally (c : GateTable) (d : Division) (offs : Array ℕ) (own : Array (Option ℕ))
    (h0 : offs[0]! = 0) (hN : offs[own.size]! = c.N) (hmono : ∀ k < own.size, offs[k]! ≤ offs[k + 1]!)
    (hexp : ∀ k < own.size, ∀ g, offs[k]! ≤ g → g < offs[k + 1]! → d.owner[g]! = own[k]!) (u : ℕ) (hu : u < d.n) :
    (NciCheck.Desc.sizes offs own d.n)[u]! = (tally c d)[u]!.1 := by
  rw [Desc.sizes_eq_count offs own d.n c.N (fun g => d.owner[g]!) h0 hN hmono hexp u hu, ((tally_spec c d).2 u hu).1]

end NciCheck.Properties.SecurityProofs
