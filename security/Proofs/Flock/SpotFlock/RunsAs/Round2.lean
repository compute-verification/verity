module

public import Proofs.Flock.SpotFlock.RunsAs.Statements

/-!
# Round 2's check gives the proved verifier's statement checks: the proofs

Each statement of `SpotFlock.RunsAs.Statements`, proved from what a passing round 2 says (`Spot.SecondStage.check_ok`):

* the model's interior order and names are round 2's (`interiorOrder_of_check`, `namesOk_of_check`): each reads only
  the two records and the replay sample, which the running verifier holds as the model does (`OwnAgrees`);
* the model's code check of the `t`-th statement is round 2's (`codesOk_of_check`). Round 2 checks it against the first
  part of its template, which is the `t`-th: it is the `t`-th part's template, and no earlier statement is of it, so no
  earlier part is (`find?_first`). And the model's check reads, of the verdicts, only the statement's own
  (`Stmt.checked`), which pass `second-verified` and `second-tables` wherever round 2's verdicts of record do; its
  other checks read nothing the two differ in (`stmtCodes_core`, `stmtCodes_verified`);
* round 2 reads one statement a template where the templates' names are distinct (`count_of_check`);
* `readsOk`, `stmtOkAt` and `stmtOk` follow (`readsOk_of_check`, `stmtOkAt_of_check`, `stmtOk_of_check`), with the
  premises round 2 does not check.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)

/-! ## Facts of round 2's code -/

/-- `jeq` holds a value to itself wherever it holds a value to it: the value has a canonical text. -/
theorem jeq_self_of {a b : Json} (h : Spot.jeq a b = true) : Spot.jeq b b = true := by
  unfold Spot.jeq at h ⊢
  cases Flock.canon a <;> cases Flock.canon b <;> simp_all

/-- What a statement names of itself (`Stmt.checked`) sets its session up as one table. -/
theorem checked_tables {s : Spot.SecondStage.Stmt} {c : Spot.SecondStage.Checked} (h : s.checked = some c) :
    c.tables = Spot.SecondStage.ONE_TABLE := by
  simp only [Spot.SecondStage.Stmt.checked, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
    Option.some.injEq] at h
  obtain ⟨_, _, _, _, hc⟩ := h
  subst hc
  rfl

/-- **Round 2's per-statement checks read nothing `core` resets**: the reported reads, the block's return and the
supplied and fixed values. -/
theorem stmtCodes_core (own : Spot.SecondStage.Own) (v : Array Spot.SecondStage.Checked) (named : Bool) (k : ℕ)
    (p : Spot.SecondStage.Part) (s : Spot.SecondStage.Stmt) :
    Spot.SecondStage.stmtCodes { core own with verified := v } named k p s =
      Spot.SecondStage.stmtCodes { own with verified := v } named k p s :=
  rfl

/-- **Round 2's per-statement checks read the verdicts only in `second-verified` and `second-tables`**: with other
verdicts that pass those two as the running verifier's do, they fail the same codes. -/
theorem stmtCodes_verified (own : Spot.SecondStage.Own) (v : Array Spot.SecondStage.Checked) {named : Bool} {k : ℕ}
    {p : Spot.SecondStage.Part} {s : Spot.SecondStage.Stmt}
    (h1 : Spot.SecondStage.verifiedOk { own with verified := v } s = Spot.SecondStage.verifiedOk own s)
    (h2 : Spot.SecondStage.tablesOk { own with verified := v } s = Spot.SecondStage.tablesOk own s) :
    Spot.SecondStage.stmtCodes { own with verified := v } named k p s = Spot.SecondStage.stmtCodes own named k p s := by
  unfold Spot.SecondStage.stmtCodes
  rw [h1, h2]
  all_goals rfl

/-- The first element `p` holds at, where it holds at the `i`-th and at none before. -/
theorem find?_first {α : Type} (p : α → Bool) :
    ∀ (l : List α) (i : ℕ) (hi : i < l.length), p l[i] = true →
      (∀ j (hj : j < i), p (l[j]'(Nat.lt_trans hj hi)) = false) → l.find? p = some l[i]
  | [], _, hi, _, _ => absurd hi (Nat.not_lt_zero _)
  | a :: l, 0, _, hp, _ => by
    simp only [List.getElem_cons_zero] at hp ⊢
    simp [List.find?_cons, hp]
  | a :: l, i + 1, hi, hp, hj => by
    have ha : p a = false := hj 0 (Nat.succ_pos i)
    rw [List.find?_cons, ha]
    exact find?_first p l i (Nat.lt_of_succ_lt_succ hi) hp fun j hj' => hj (j + 1) (Nat.succ_lt_succ hj')

/-- A statement before the `t`-th is of a template among the first `t` statements'. -/
theorem mem_templates_take {ss : List Json} {j t : ℕ} (hj : j < t) (ht : t ≤ ss.length) {s : Spot.SecondStage.Stmt}
    (hs : Spot.SecondStage.Stmt.ofJson? (ss.getD j .null) = some s) :
    (Spot.SecondStage.templates (ss.take t)).contains s.template = true := by
  have hjl : j < ss.length := Nat.lt_of_lt_of_le hj ht
  rw [List.contains_iff_mem]
  unfold Spot.SecondStage.templates
  rw [List.mem_filterMap]
  refine ⟨ss[j], List.mem_iff_getElem.2 ⟨j, by rw [List.length_take]; omega, by simp⟩, ?_⟩
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjl, Option.getD_some] at hs
  simp only [hs, Option.map_some]

/-! ## The parts -/

theorem partsAt_size (h : Held) (q : ℕ) : (partsAt h q).size = h.ms.length := by
  simp [partsAt]

theorem partsAt_getElem? (h : Held) (q : ℕ) {t : ℕ} (ht : t < h.ms.length) :
    (partsAt h q)[t]? = some (partAt h q t) := by
  unfold partsAt
  rw [List.getElem?_toArray, List.getElem?_map, List.getElem?_range ht, Option.map_some]

/-- **The part at position `t`** is template `t`'s, `t` one of the templates. -/
theorem partsAt_template {h : Held} {q t : ℕ} {tm : String} (hi : (partsAt h q)[t]?.map (·.template) = some tm) :
    t < h.ms.length ∧ h.names.getD t "" = tm := by
  by_cases ht : t < h.ms.length
  · rw [partsAt_getElem? h q ht] at hi
    exact ⟨ht, Option.some.inj hi⟩
  · rw [getElem?_neg (partsAt h q) t (by simpa [partsAt_size] using ht)] at hi
    simp at hi

/-! ## What the agreement gives -/

theorem sample_of_agrees {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀) : own.sample = h.reg.sample (src c₀) :=
  congrArg Spot.SecondStage.Own.sample (show core own = ownOf h st rec draw c₀ from hA)

theorem parts_of_agrees {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀) : own.parts = some (partsAt h own.sample.size) := by
  rw [sample_of_agrees hA]
  exact congrArg Spot.SecondStage.Own.parts (show core own = ownOf h st rec draw c₀ from hA)

/-- **Round 2's parts are the model's**: where it passes, the parts it reads the statements against are `partsAt`. -/
theorem parts_of_check {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} {ps : Array Spot.SecondStage.Part} (hA : OwnAgrees h own st rec draw c₀)
    (hp : own.parts = some ps) : ps = partsAt h own.sample.size :=
  Option.some.inj (hp.symm.trans (parts_of_agrees hA))

/-- A passing round 2's statements are at most one a template. -/
theorem lt_of_check {h : Held} {own : Spot.SecondStage.Own} {ss : Array Json} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀)
    (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) {t : ℕ} (ht : t < ss.size) : t < h.ms.length := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  obtain ⟨s, -, -, -, -, hi, -⟩ := hst t ht
  exact (partsAt_template hi).1

/-! ## The clauses -/

theorem interiorOrder_of_check : InteriorOrderOfCheck := by
  intro h own ss st rec draw c₀ hA hc
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  exact (Spot.SecondStage.check_ok hc).2.1

theorem namesOk_of_check : NamesOkOfCheck := by
  intro h own ss st rec draw c₀ hA hc
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  exact Spot.SecondStage.interiorFailures_names (Spot.SecondStage.check_ok_interior hc)

theorem codesOk_of_check : CodesOkOfCheck := by
  intro h own ss st rec draw c₀ hA hc t ht S hS
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  obtain ⟨s, p, hs, hf, hd, hi, hcodes⟩ := hst t ht
  rw [show S = s from Option.some.inj (hS.symm.trans hs)]
  obtain ⟨htm, hname⟩ := partsAt_template hi
  -- the part round 2 checks the statement against is the `t`-th
  have hpt : p = partAt h own.sample.size t := by
    have hl : t < ((List.range h.ms.length).map (partAt h own.sample.size)).length := by
      rw [List.length_map, List.length_range]
      exact htm
    have hat : (fun x : Spot.SecondStage.Part => x.template == s.template)
        (((List.range h.ms.length).map (partAt h own.sample.size))[t]'hl) = true := by
      rw [List.getElem_map, List.getElem_range]
      exact beq_iff_eq.2 hname
    have hbefore : ∀ j (hj : j < t), (fun x : Spot.SecondStage.Part => x.template == s.template)
        (((List.range h.ms.length).map (partAt h own.sample.size))[j]'(Nat.lt_trans hj hl)) = false := by
      intro j hj
      rw [List.getElem_map, List.getElem_range]
      obtain ⟨sj, -, hsj, -, -, hij, -⟩ := hst j (Nat.lt_trans hj ht)
      obtain ⟨-, hnj⟩ := partsAt_template hij
      have hmem := mem_templates_take hj (by simpa using Nat.le_of_lt ht) hsj
      have hne : sj.template ≠ s.template := fun e => by
        rw [e, hd] at hmem
        exact Bool.false_ne_true hmem
      show (h.names.getD j "" == s.template) = false
      rw [hnj]
      exact beq_false_of_ne hne
    have key := find?_first _ _ t hl hat hbefore
    rw [List.getElem_map, List.getElem_range] at key
    exact Option.some.inj (hf.symm.trans ((List.find?_toArray _ _).trans key))
  -- the statement's own verdict passes what round 2's verdicts of record pass
  obtain ⟨hv, ht2, -⟩ := Spot.SecondStage.stmtCodes_nil hcodes
  obtain ⟨c, hcs, v, hvm, h1, h2, h3⟩ := Spot.SecondStage.verifiedOk_some hv
  have hone := jeq_self_of (Spot.SecondStage.tablesOk_one ht2 hcs hvm h1 h2 h3)
  have hreg := jeq_self_of h2
  have hdraw := jeq_self_of h3
  have hv1 : Spot.SecondStage.verifiedOk { own with verified := s.checked.toArray } s = true := by
    unfold Spot.SecondStage.verifiedOk
    rw [hcs]
    exact Array.any_eq_true'.2 ⟨c, Array.mem_singleton.2 rfl, by simp [hreg, hdraw]⟩
  have ht1 : Spot.SecondStage.tablesOk { own with verified := s.checked.toArray } s = true := by
    unfold Spot.SecondStage.tablesOk
    rw [hcs]
    refine Array.all_eq_true'.2 fun x hx => ?_
    obtain rfl : x = c := Array.mem_singleton.1 hx
    simp [checked_tables hcs, hone]
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  show (Spot.SecondStage.stmtCodes { core own with verified := s.checked.toArray }
    (Spot.SecondStage.byTemplate own.interior) (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t ""))
    (partAt h own.sample.size t) s).isEmpty = true
  rw [stmtCodes_core, hname, ← hpt, stmtCodes_verified own _ (hv1.trans hv.symm) (ht1.trans ht2.symm), hcodes]
  all_goals rfl

theorem count_of_check : CountOfCheck := by
  intro h own ss st rec draw c₀ hA hN hc
  obtain ⟨-, -, ps, hp, -, -, -, hcov, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  apply Nat.le_antisymm
  · by_contra hlt
    exact Nat.lt_irrefl _ (lt_of_check hA hc (t := h.ms.length) (by omega))
  · have hsub : ((List.range h.ms.length).map fun t => h.names.getD t "") ⊆
        Spot.SecondStage.templates ss.toList := by
      intro x hx
      obtain ⟨t, htr, rfl⟩ := List.mem_map.1 hx
      have hmemp : partAt h own.sample.size t ∈ (partsAt h own.sample.size).toList :=
        List.mem_map.2 ⟨t, htr, rfl⟩
      exact List.contains_iff_mem.1 (hcov _ hmemp)
    have hle := (List.Nodup.subperm hN hsub).length_le
    rw [List.length_map, List.length_range] at hle
    exact hle.trans ((List.length_filterMap_le _ _).trans (by simp))

theorem readsOk_of_check : ReadsOkOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hP hc t ht
  obtain ⟨xs, hxs, hlen, hw⟩ := hS
  obtain ⟨-, -, ps, -, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain ⟨s, -, hs, -, -, -, -⟩ := hst t ht
  have htm := lt_of_check hA hc ht
  obtain ⟨w, hdw, hjw⟩ := hw t (by omega)
  have hat : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by
    unfold stmtAt
    rw [hxs]
  have hdw' : FlockDeploy.decode (stmtAt h m₂ t) = some w := by
    rw [hat]
    exact hdw
  unfold readsOk
  rw [hx, hm₁, hd, hdw']
  simp only [Bool.and_eq_true]
  refine ⟨⟨interiorOrder_of_check h own ss st rec draw c₀ hA hc, ?_⟩, hP t htm w hdw'⟩
  rw [hjw, hs]
  exact codesOk_of_check h own ss st rec draw c₀ hA hc t ht s hs

theorem stmtOkAt_of_check : StmtOkAtOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hT hR hSc hP hc t ht
  have htm := lt_of_check hA hc ht
  have hr := readsOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hP hc t ht
  unfold stmtOkAt
  cases hpx : parse h x with
  | none => simp [parse, hx] at hpx
  | some s =>
    simp only [Bool.and_eq_true, beq_iff_eq]
    exact ⟨⟨⟨⟨rfl, hT s⟩, hR t htm⟩, hSc t htm⟩, hr⟩

theorem stmtOk_of_check : StmtOkOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hT hN hR hSc hP hc
  have hcount := count_of_check h own ss st rec draw c₀ hA hN hc
  obtain ⟨xs, hxs, hlen, -⟩ := id hS
  unfold stmtOk
  cases hpx : parse h x with
  | none => simp [parse, hx] at hpx
  | some s =>
    rw [hxs]
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
    refine ⟨⟨⟨rfl, hT s⟩, hlen.trans hcount⟩, fun t ht => ?_⟩
    have hat : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by
      unfold stmtAt
      rw [hxs]
    rw [← hat]
    exact ⟨⟨hR t ht, hSc t ht⟩,
      readsOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hP hc t (by omega)⟩

end SpotFlock.RunsAs
