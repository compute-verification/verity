module

public import Proofs.Flock.SpotFlock.RunsAs.Statements

/-! Round 2's check gives the proved verifier's statement checks: the proofs of `SpotFlock.RunsAs.Statements`. -/

@[expose] public section

namespace SpotFlock.RunsAs.Round2

open Claims
open Lean (Json)

theorem jeq_self_of {a b : Json} (h : Spot.jeq a b = true) : Spot.jeq b b = true := by
  revert h
  unfold Spot.jeq
  cases Flock.canon a <;> cases Flock.canon b <;> simp

theorem checked_tables {s : Spot.SecondStage.Stmt} {c : Spot.SecondStage.Checked} (h : s.checked = some c) :
    c.tables = Spot.SecondStage.ONE_TABLE := by
  simp only [Spot.SecondStage.Stmt.checked, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff] at h
  obtain ⟨_, _, h⟩ := h
  split at h <;> (obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.1 h; cases h; rfl)

theorem stmtCodes_core (own : Spot.SecondStage.Own) (v : Array Spot.SecondStage.Checked) (named : Bool) (k : ℕ)
    (p : Spot.SecondStage.Part) (s : Spot.SecondStage.Stmt) :
    Spot.SecondStage.stmtCodes { core own with verified := v } named k p s =
      Spot.SecondStage.stmtCodes { own with verified := v } named k p s :=
  rfl

theorem stmtCodes_verified (own : Spot.SecondStage.Own) (v : Array Spot.SecondStage.Checked) {named : Bool} {k : ℕ}
    {p : Spot.SecondStage.Part} {s : Spot.SecondStage.Stmt}
    (h1 : Spot.SecondStage.verifiedOk { own with verified := v } s = Spot.SecondStage.verifiedOk own s)
    (h2 : Spot.SecondStage.tablesOk { own with verified := v } s = Spot.SecondStage.tablesOk own s) :
    Spot.SecondStage.stmtCodes { own with verified := v } named k p s = Spot.SecondStage.stmtCodes own named k p s := by
  unfold Spot.SecondStage.stmtCodes
  rw [h1, h2]
  all_goals rfl

theorem find?_first {α : Type} (p : α → Bool) :
    ∀ (l : List α) (i : ℕ) (hi : i < l.length), p l[i] = true →
      (∀ j (hj : j < i), p (l[j]'(Nat.lt_trans hj hi)) = false) → l.find? p = some l[i]
  | [], _, hi, _, _ => absurd hi (Nat.not_lt_zero _)
  | a :: l, 0, _, hp, _ => by
    simp only [List.getElem_cons_zero] at hp ⊢
    simp [hp]
  | a :: l, i + 1, hi, hp, hj => by
    have ha : p a = false := hj 0 (Nat.succ_pos i)
    rw [List.find?_cons, ha]
    exact find?_first p l i (Nat.lt_of_succ_lt_succ hi) hp fun j hj' => hj (j + 1) (Nat.succ_lt_succ hj')

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

theorem partsAt_size (h : Held) (q : ℕ) : (partsAt h q).size = h.ms.length := by
  simp [partsAt]

theorem partsAt_getElem? (h : Held) (q : ℕ) {t : ℕ} (ht : t < h.ms.length) :
    (partsAt h q)[t]? = some (partAt h q t) := by
  unfold partsAt
  rw [List.getElem?_toArray, List.getElem?_map, List.getElem?_range ht, Option.map_some]

theorem partsAt_template {h : Held} {q t : ℕ} {tm : String} (hi : (partsAt h q)[t]?.map (·.template) = some tm) :
    t < h.ms.length ∧ h.names.getD t "" = tm := by
  by_cases ht : t < h.ms.length
  · rw [partsAt_getElem? h q ht] at hi
    exact ⟨ht, Option.some.inj hi⟩
  · rw [getElem?_neg (partsAt h q) t (by simpa [partsAt_size] using ht)] at hi
    simp at hi

theorem sample_of_agrees {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀) : own.sample = h.reg.sample (src c₀) :=
  congrArg Spot.SecondStage.Own.sample (show core own = ownOf h st rec draw c₀ from hA)

theorem parts_of_agrees {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀) : own.parts = some (partsAt h own.sample.size) := by
  rw [sample_of_agrees hA]
  exact congrArg Spot.SecondStage.Own.parts (show core own = ownOf h st rec draw c₀ from hA)

theorem parts_of_check {h : Held} {own : Spot.SecondStage.Own} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} {ps : Array Spot.SecondStage.Part} (hA : OwnAgrees h own st rec draw c₀)
    (hp : own.parts = some ps) : ps = partsAt h own.sample.size :=
  Option.some.inj (hp.symm.trans (parts_of_agrees hA))

theorem lt_of_check {h : Held} {own : Spot.SecondStage.Own} {ss : Array Json} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀)
    (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) {t : ℕ} (ht : t < ss.size) : t < h.ms.length := by
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  obtain ⟨s, -, -, -, -, hi, -⟩ := hst t ht
  exact (partsAt_template hi).1

theorem interiorOrder_of_check : InteriorOrderOfCheck := by
  intro h own ss st rec draw c₀ hA hc
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  exact (Spot.SecondStage.check_ok hc).2.1

theorem namesOk_of_check : NamesOkOfCheck := by
  intro h own ss st rec draw c₀ hA hc
  have hn := Spot.SecondStage.interiorFailures_names (Spot.SecondStage.check_ok_interior hc)
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  exact hn

theorem codesOk_of_check : CodesOkOfCheck := by
  intro h own ss st rec draw c₀ hA hc t ht S hS
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  obtain ⟨s, p, hs, hf, hd, hi, hcodes⟩ := hst t ht
  rw [show S = s from Option.some.inj (hS.symm.trans hs)]
  obtain ⟨htm, hname⟩ := partsAt_template hi
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
    have key := find?_first (fun x : Spot.SecondStage.Part => x.template == s.template) _ t hl hat hbefore
    rw [List.getElem_map, List.getElem_range] at key
    exact Option.some.inj (hf.symm.trans ((List.find?_toArray _ _).trans key))
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

theorem designOk_of_check {h : Held} {own : Spot.SecondStage.Own} {ss : Array Json} {st : Roots} {rec draw : Json}
    {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀)
    (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) :
    Spot.SecondStage.designOk (partsAt h (ownOf h st rec draw c₀).sample.size) (ownOf h st rec draw c₀) = true := by
  obtain ⟨-, -, ps, hp, hl, hw, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  rw [← show core own = ownOf h st rec draw c₀ from hA]
  unfold Spot.SecondStage.designOk
  rw [Bool.and_eq_true]
  exact ⟨hl, hw⟩

theorem inputsFailure_none {own : Spot.SecondStage.Own} (hR : RegistersNone own.interior) :
    Spot.SecondStage.inputsFailure own = none := by
  unfold Spot.SecondStage.inputsFailure
  dsimp only
  rw [Array.find?_eq_none]
  intro v hv
  exfalso
  obtain ⟨e, he, hev⟩ := Array.mem_filterMap.1 hv
  try dsimp only at hev
  have hstr : ∃ v', Spot.getD e "value" = .str v' := by
    revert hev
    split
    · rename_i v' hval
      intro _
      exact ⟨v', hval⟩
    · intro hnone
      cases hnone
  obtain ⟨v', hval⟩ := hstr
  revert he
  split
  · rename_i es hes
    intro he
    exact hR es hes e he v' hval
  · intro he
    simp at he

theorem inputsNone_of_registersNone {own : Spot.SecondStage.Own} (hR : RegistersNone own.interior) :
    InputsNone own :=
  inputsFailure_none (own := core own) hR

theorem inputsNone_of_emptied {own : Spot.SecondStage.Own}
    (hI : Spot.SecondStage.inputsFailure
      { own with fixed := #[], supplied := #[], reads := #[], tables := #[], widths := #[] } = none) :
    InputsNone own :=
  hI

theorem readsOk_of_check : ReadsOkOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hP hc t ht
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
  refine ⟨⟨⟨⟨⟨interiorOrder_of_check h own ss st rec draw c₀ hA hc, namesOk_of_check h own ss st rec draw c₀ hA hc⟩,
    ?_⟩, designOk_of_check hA hc⟩, ?_⟩, hP t htm w hdw'⟩
  · have hI : Spot.SecondStage.inputsFailure (ownOf h st rec draw c₀) = none := by
      rw [← show core own = ownOf h st rec draw c₀ from hA]
      exact hRN
    simp [hI]
  rw [hjw, hs]
  exact codesOk_of_check h own ss st rec draw c₀ hA hc t ht s hs

theorem stmtOkAt_of_check : StmtOkAtOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hT hR hSc hP hc t ht
  have htm := lt_of_check hA hc ht
  have hr := readsOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hP hc t ht
  unfold stmtOkAt
  cases hpx : parse h x with
  | none => simp [parse, hx] at hpx
  | some s =>
    simp only [Bool.and_eq_true, beq_iff_eq]
    exact ⟨⟨⟨⟨rfl, hT s⟩, hR t htm⟩, hSc t htm⟩, hr⟩

theorem stmtOk_of_check : StmtOkOfCheck := by
  intro h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hT hN hRec hR hSc hP hc
  have hcount := count_of_check h own ss st rec draw c₀ hA hN hc
  obtain ⟨xs, hxs, hlen, -⟩ := id hS
  unfold stmtOk
  cases hpx : parse h x with
  | none => simp [parse, hx] at hpx
  | some s =>
    rw [hxs]
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
    refine ⟨⟨⟨⟨rfl, hT s⟩, hlen.trans hcount⟩, hRec⟩, fun t ht => ?_⟩
    have hat : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by
      unfold stmtAt
      rw [hxs]
    rw [← hat]
    exact ⟨⟨hR t ht, hSc t ht⟩,
      readsOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hP hc t (by omega)⟩

theorem roundTwo_of_check {h : Held} (hN : NamesDistinct h) {own : Spot.SecondStage.Own} {ss : Array Json}
    {st : Roots} {rec draw : Json} {c₀ : FlockDeploy.Coin} (hA : OwnAgrees h own st rec draw c₀)
    (hc : (Spot.SecondStage.check true own (some ss)).codes = #[]) :
    Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true ∧
      Spot.SecondStage.namesOk (ownOf h st rec draw c₀) = true ∧
      Spot.SecondStage.designOk (partsAt h (ownOf h st rec draw c₀).sample.size) (ownOf h st rec draw c₀) = true ∧
      ∀ t < h.ms.length, ∃ S, Spot.SecondStage.Stmt.ofJson? (ss.getD t .null) = some S ∧
        S.template = h.names.getD t "" ∧ codesOk h (ownOf h st rec draw c₀) t S = true := by
  refine ⟨interiorOrder_of_check h own ss st rec draw c₀ hA hc, namesOk_of_check h own ss st rec draw c₀ hA hc,
    designOk_of_check hA hc, fun t ht => ?_⟩
  have hts : t < ss.size := by
    rw [count_of_check h own ss st rec draw c₀ hA hN hc]
    exact ht
  obtain ⟨-, -, ps, hp, -, -, hst, -, -⟩ := Spot.SecondStage.check_ok hc
  obtain rfl := parts_of_check hA hp
  obtain ⟨s, -, hs, -, -, hi, -⟩ := hst t hts
  have hg : ss.getD t .null = ss.toList.getD t .null := by
    simp [List.getD_eq_getElem?_getD]
  refine ⟨s, by rw [hg]; exact hs, (partsAt_template hi).2.symm, ?_⟩
  exact codesOk_of_check h own ss st rec draw c₀ hA hc t hts s hs

end SpotFlock.RunsAs.Round2
