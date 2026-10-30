import Properties.SampledProofs
import Proofs.Properties.Challenger

/-! # Proofs of sampled proofs' one-stage property (`Properties.SampledProofs`)

Each check of `OneStage.check` is a pipeline of failure codes (`addCode`, `push`) written with `let mut` and `if`, which
`do` elaborates into join points. A proof lifts the join points out (`extract_lets`), moves each `if` into the argument of
the join point it calls (`jp_ite`, `jp_ite'`), unfolds the join points one at a time, and reads the codes' emptiness off
the pipeline one check at a time (`ite_addCode_nil`, `ite_push_nil`). `acceptMeans_iff` says what the check accepts and
that it accepts all of it; `acceptMeans` is its first half and `acceptMeans_complete` the converse.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

open OneStage (Params Commitment Evidence Public)

set_option linter.unusedSimpArgs false

theorem jp_ite {α β : Type} (f : α → β) (b : Bool) (u v : α) :
    (if b = true then f u else f v) = f (if b = true then u else v) := by cases b <;> rfl
theorem jp_ite' {α β : Type} (f : α → β) (b : Bool) (u v : α) :
    (if b = true then (have x := u; f x) else f v) = f (if b = true then u else v) := by cases b <;> rfl
theorem jp_dite {α β : Type} (f : α → β) (p : Prop) [Decidable p] (u : p → α) (v : α) :
    (if h : p then f (u h) else f v) = f (if h : p then u h else v) := by by_cases h : p <;> simp [h]
theorem addCode_ne_nil (c : Array String) (x : String) : addCode c x ≠ #[] := by
  unfold addCode
  split
  · rename_i h; intro hc; subst hc; simp at h
  · simp

theorem ite_addCode_nil (b : Bool) (c : Array String) (x : String) :
    ((if b = true then addCode c x else c) = #[]) ↔ (b = false ∧ c = #[]) := by
  cases b <;> simp [addCode_ne_nil]

theorem fail_ne_nil (c : Check) (x : String) (w : Json) : (c.fail x w).codes ≠ #[] := by
  unfold Check.fail
  split
  · rename_i h; intro hc; rw [hc] at h; simp at h
  · simp

theorem ite_fail_nil (b : Bool) (c : Check) (x : String) (w : Json) :
    ((if b = true then c.fail x w else c).codes = #[]) ↔ (c.codes = #[] ∧ b = false) := by
  cases b <;> simp [fail_ne_nil]

theorem ite_ite_fail_nil (a b : Bool) (c : Check) (x : String) (w : Json) :
    ((if a = true then (if b = true then c.fail x w else c) else c).codes = #[]) ↔
      (c.codes = #[] ∧ (a = true → b = false)) := by
  cases a <;> cases b <;> simp [fail_ne_nil]
theorem ite_ite_ite_fail_nil (a b d : Bool) (c : Check) (x y : String) (w v : Json) :
    ((if a = true then (if b = true then c.fail x w else (if d = true then c.fail y v else c)) else c).codes = #[]) ↔
      (c.codes = #[] ∧ (a = true → b = false ∧ d = false)) := by
  cases a <;> cases b <;> cases d <;> simp [fail_ne_nil]
theorem dite_ite_fail_nil (p : Prop) [Decidable p] (b : p → Bool) (c : Check) (x : String) (w : Json) :
    ((if h : p then (if b h = true then c.fail x w else c) else c).codes = #[]) ↔
      (c.codes = #[] ∧ ∀ h : p, b h = false) := by
  by_cases hp : p
  · simp only [hp, dite_true]
    cases hb : b hp <;> simp [fail_ne_nil, hb]
  · simp [hp]

theorem map_some_inj {α : Type} (xs ys : Array α) : xs.map some = ys.map some ↔ xs = ys := by
  rw [← Array.toList_inj, Array.toList_map, Array.toList_map, List.map_inj_right (fun _ _ h => Option.some.inj h),
    Array.toList_inj]

theorem toJson_nat_int (n : Nat) : int? (Lean.toJson n) = some (n : Int) := by
  simp [Lean.toJson, int?, Lean.JsonNumber.fromNat]

theorem checkProved_nil (draw : Json) (us : Array Nat) (n : Nat) :
    (Draw.checkProvedFull draw (us.map Lean.toJson) (Lean.toJson n)).codes = #[] ↔
      n = (Draw.proved draw).size ∧ us.map (fun n : Nat => (n : Int)) = Draw.proved draw := by
  unfold Draw.checkProvedFull
  extract_lets want c0 j1 j0 c1
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp only [j0]
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp only [j1, Id.run_pure, c1, c0]
  simp only [ite_fail_nil, bne_eq_false_iff_eq, Array.map_map, Function.comp_def, toJson_nat_int,
    Option.some.injEq, Int.natCast_inj, true_and]
  rw [show (Array.map (fun (x : Nat) => some (x : Int)) us) = (us.map fun (x : Nat) => (x : Int)).map some by
    simp [Array.map_map, Function.comp_def], map_some_inj]

/-- A check of `Record.checks` that fails, other than the draw's. -/
def Fails (x : String × Check) : Bool := x.1 != "draw" && !x.2.codes.isEmpty

theorem none_fail_ite_push (b : Bool) (xs : Array (String × Check)) (y : String × Check) :
    (∀ x ∈ (if b = true then xs.push y else xs), Fails x = false) ↔
      (∀ x ∈ xs, Fails x = false) ∧ (b = true → Fails y = false) := by
  cases b <;> simp [Array.mem_push, or_imp, forall_and]

theorem none_fail_ite_ite_push (a b : Bool) (xs : Array (String × Check)) (y : String × Check) :
    (∀ x ∈ (if a = true then (if b = true then xs.push y else xs) else xs), Fails x = false) ↔
      (∀ x ∈ xs, Fails x = false) ∧ (a = true → b = true → Fails y = false) := by
  cases a <;> cases b <;> simp [Array.mem_push, or_imp, forall_and]

theorem fails_of_codes (n : String) (c : Check) (h1 : n ≠ "draw") (h2 : c.codes = #[n]) : Fails (n, c) = true := by
  simp [Fails, h1, h2]

theorem size_qsort {α : Type} (xs : Array α) (lt : α → α → Bool) : (xs.qsort lt).size = xs.size := by
  unfold Array.qsort
  split
  · rfl
  · simp

theorem none_fail_base (d p : Check) :
    (∀ x ∈ #[("draw", d), ("proved", p)], Fails x = false) ↔ p.codes = #[] := by
  simp [Fails, Array.isEmpty_iff]

theorem sameDict_iff (a b : Array (String × Json)) : Record.sameDict a b = true ↔ SameRoots a b := by
  simp only [Record.sameDict, SameRoots, Bool.and_eq_true, beq_iff_eq, Array.all_eq_true', Array.any_eq_true']

theorem qsort_eq_empty {α : Type} (xs : Array α) (lt : α → α → Bool) : xs.qsort lt = #[] ↔ xs = #[] := by
  rw [← Array.size_eq_zero_iff, size_qsort, Array.size_eq_zero_iff]

theorem checks_ok (reg receipt draw law : Json) (units : Array Int) (provedUnits : Array Json) (instances : Json)
    (digest : String) (served : Option (Array Int)) (sessionRoots : Option (Array (String × Json)))
    (opened : Option (Array (String × Json))) :
    (∀ x ∈ Record.checks reg receipt draw law units provedUnits instances digest served sessionRoots opened,
      Fails x = false) ↔
    (Draw.checkProvedFull draw provedUnits instances).codes = #[] ∧
    jeq (getD receipt "registration") (.str digest) = true ∧
    (∀ s, served = some s → s.qsort (· < ·) = units.qsort (· < ·)) ∧
    (∀ roots, sessionRoots = some roots → roots ≠ #[] ∧
      ∀ pr ∈ roots, ∃ r, (Record.byPort reg).find? (·.1 == pr.1) = some r ∧ jeq (getD r.2 "root") pr.2 = true) ∧
    ((field? reg "registered").isSome = true ∨ opened.isSome = true →
      SameRoots (opened.getD #[]) (Record.registeredRoots reg)) := by
  unfold Record.checks
  extract_lets o0 j3 want opened' wrong j2 registered j1 j0 o1
  rcases hs : served with _ | s <;> rcases hr : sessionRoots with _ | roots
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j0, hs]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j1, hr]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j2]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j3, Id.run_pure, o1, o0]
  all_goals simp only [none_fail_ite_push, none_fail_ite_ite_push]
  all_goals try rw [fails_of_codes "receipt" _ (by decide) rfl]
  all_goals try rw [fails_of_codes "served" _ (by decide) rfl]
  all_goals try rw [fails_of_codes "roots" _ (by decide) rfl]
  all_goals try rw [fails_of_codes "registered" _ (by decide) rfl]
  all_goals simp only [Bool.true_eq_false, imp_false, Bool.not_eq_true, Bool.not_eq_false', Bool.or_eq_false_iff,
    none_fail_base, Bool.or_eq_true, bne_eq_false_iff_eq, Array.isEmpty_iff, Array.isEmpty_eq_false_iff,
    not_or, sortStrings, qsort_eq_empty, Array.filterMap_eq_empty_iff,
    sameDict_iff, opened', registered,
    want, reduceCtorEq, false_implies, implies_true, and_true, true_and, Option.some.injEq, forall_eq', and_assoc]
  all_goals clear j0 j1 j2 j3 o0 o1 wrong want opened' registered
  case some.some =>
    constructor
    · rintro ⟨h1, h2, h3, h4, h5, h6⟩
      refine ⟨h1, h2, h3, h4, fun pr hpr => ?_, h6⟩
      have := h5 pr hpr; revert this; split <;> simp_all
    · rintro ⟨h1, h2, h3, h4, h5, h6⟩
      refine ⟨h1, h2, h3, h4, fun pr hpr => ?_, h6⟩
      obtain ⟨r, hr, hj⟩ := h5 pr hpr
      simp only [hr, hj, ↓reduceIte]
  case none.some =>
    constructor
    · rintro ⟨h1, h2, h4, h5, h6⟩
      refine ⟨h1, h2, h4, fun pr hpr => ?_, h6⟩
      have := h5 pr hpr; revert this; split <;> simp_all
    · rintro ⟨h1, h2, h4, h5, h6⟩
      refine ⟨h1, h2, h4, fun pr hpr => ?_, h6⟩
      obtain ⟨r, hr, hj⟩ := h5 pr hpr
      simp only [hr, hj, ↓reduceIte]
  all_goals exact Iff.rfl

theorem recordChecks_nil (c : Commitment) (e : Evidence) :
    OneStage.recordChecks c.record (Flock.hexOf c.digest) e = #[] ↔ UnitsProved c e := by
  unfold OneStage.recordChecks
  simp only [Array.map_eq_empty_iff, Array.filter_eq_empty_iff]
  have hf : ∀ cs : Array (String × Check),
      (∀ x ∈ cs, ¬(match x with | (n, c) => n != "draw" && !c.codes.isEmpty) = true) ↔ ∀ x ∈ cs, Fails x = false := by
    intro cs; simp [Fails]
  rw [hf, checks_ok, checkProved_nil]
  unfold UnitsProved Receipt openedRoots
  simp only [Option.map_eq_some_iff, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, and_assoc]
  exact Iff.rfl

theorem verdictOf_accept (vs : Array (String × Json)) :
    OneStage.verdictOf vs = .accept () () ↔ VerdictsAccept vs := by
  unfold OneStage.verdictOf VerdictsAccept
  split
  · rename_i name x hf
    simp only [reduceCtorEq, false_iff]
    have hm := Array.mem_of_find?_eq_some hf
    have hp := Array.find?_some hf
    rintro ⟨h1, -, -⟩
    have := h1 _ hm
    revert hp this
    simp only
    cases getD x "accepted" <;> simp
  · rename_i hf
    rw [Array.find?_eq_none] at hf
    have h1 : ∀ x ∈ vs, getD x.2 "accepted" ≠ .null → jeq (getD x.2 "accepted") (.bool true) = true := by
      intro x hx hn
      have := hf x hx
      revert this hn
      simp only
      cases getD x.2 "accepted" <;> simp
    refine Iff.trans ?_ (and_iff_right h1).symm
    simp only []
    split
    · rename_i he
      simp only [reduceCtorEq, false_iff, not_and]
      rw [Array.isEmpty_iff, Array.filter_eq_empty_iff] at he
      intro ⟨x, hx, ht⟩
      exact absurd ht (by simpa using he _ hx)
    · rename_i he
      split
      · rename_i name x hg
        simp only [reduceCtorEq, false_iff, not_and]
        have hm := Array.mem_of_find?_eq_some hg
        have hp := Array.find?_some hg
        rw [Array.mem_filter] at hm
        intro _ h3
        have := h3 _ hm.1 hm.2
        revert hp this
        simp only
        cases getD x "accepted" <;> simp
      · rename_i hg
        rw [Array.find?_eq_none] at hg
        simp only [true_iff]
        refine ⟨?_, ?_⟩
        · rw [Array.isEmpty_iff, Array.filter_eq_empty_iff] at he
          simpa using he
        · intro x hx ht
          have := hg x (Array.mem_filter.2 ⟨hx, ht⟩)
          revert this
          cases getD x.2 "accepted" <;> simp

/-- A loop step's codes, and whether it continues. -/
def codesOf (x : Id (ForInStep (Array String))) : Array String :=
  match x.run with
  | .yield c => c
  | .done c => c
def yields (x : Id (ForInStep (Array String))) : Bool :=
  match x.run with
  | .yield _ => true
  | .done _ => false

theorem yields_eq (x : Id (ForInStep (Array String))) (h : yields x = true) : x = pure (.yield (codesOf x)) := by
  revert h
  cases x with
  | yield a => intro _; rfl
  | done a => intro h; simp [yields, Id.run] at h

theorem forIn_codes_nil {α : Type} (xs : Array α) (c0 : Array String)
    (f : α → Array String → Id (ForInStep (Array String))) (Q : α → Prop)
    (hc : ∀ x c, yields (f x c) = true ∧ (codesOf (f x c) = #[] ↔ c = #[] ∧ Q x)) :
    (forIn xs c0 f).run = #[] ↔ c0 = #[] ∧ ∀ x ∈ xs, Q x := by
  rw [← Array.forIn_toList]
  have : ∀ x, x ∈ xs ↔ x ∈ xs.toList := fun x => Array.mem_toList_iff.symm
  simp only [this]
  generalize xs.toList = l
  induction l generalizing c0 with
  | nil => simp
  | cons a as ih =>
    rw [List.forIn_cons, yields_eq _ (hc a c0).1]
    simp only [pure_bind]
    refine (ih _).trans ?_
    rw [(hc a c0).2]
    simp only [List.mem_cons, forall_eq_or_imp, and_assoc]

theorem step_pure (c : Array String) : yields (pure (.yield c)) = true ∧ codesOf (pure (.yield c)) = c := ⟨rfl, rfl⟩

theorem root_ok (j : Json) :
    ((!match j with | .str s => s.length == 128 && isHex s | _ => false) = false) ↔
      ∃ s, j = .str s ∧ s.length = 128 ∧ isHex s = true := by
  cases j <;> simp

theorem width_ok (e : Json) (want : Registered.Entry) :
    ((match field? e Registered.WIDTH_KEY with
      | none => want.wordBits != Registered.WORD_BITS
      | some j => match int? j with
        | some w => w != (want.wordBits : Int) || w == (Registered.WORD_BITS : Int)
        | none => true) = false) ↔
    (match field? e Registered.WIDTH_KEY with
      | none => want.wordBits = Registered.WORD_BITS
      | some j => int? j = some (want.wordBits : Int) ∧ (want.wordBits : Int) ≠ Registered.WORD_BITS) := by
  cases field? e Registered.WIDTH_KEY with
  | none => simp
  | some j =>
    simp only []
    generalize int? j = o
    cases o with
    | none => simp
    | some w =>
      simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq, beq_eq_false_iff_ne, Option.some.injEq]
      constructor
      · rintro ⟨rfl, h⟩; exact ⟨rfl, h⟩
      · rintro ⟨rfl, h⟩; exact ⟨rfl, h⟩

theorem checkRegistered_nil_supplied (c : Array String) (got : Json) (mine : Flock.V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → Flock.V String) :
    Registration.checkRegistered c got mine held supplied rootOf = #[] ↔ c = #[] ∧ ∃ mine' es, mine = .ok mine' ∧
      got = .arr es ∧
      (∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
        keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
      ValuesAre es mine' ∧
      (∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
        EntryIs e want (heldFor supplied held want.value)) ∧
      SuppliedIs es mine' supplied rootOf := by
  unfold Registration.checkRegistered
  extract_lets widthKeys
  rcases mine with err | mine
  · simp [addCode_ne_nil]
  rcases got with _ | _ | _ | _ | es | _
  all_goals try (simp [addCode_ne_nil]; done)
  simp (config := {zeta := false}) only []
  by_cases hk : (!es.all fun e => keys? e == some Registered.ENTRY_KEYS || keys? e == some widthKeys) = true
  · simp only [hk, ↓reduceIte, Id.run_pure, addCode_ne_nil, false_iff, not_and, not_exists]
    intro _ mine' es' hm he hkeys
    cases hm
    cases he
    intro _ _
    have hall : (es.all fun e => keys? e == some Registered.ENTRY_KEYS || keys? e == some widthKeys) = true := by
      apply Array.all_eq_true'.2
      intro e he
      rcases hkeys e he with h | h
      · simp only [h, beq_self_eq_true, Bool.true_or]
      · show (keys? e == some Registered.ENTRY_KEYS || keys? e == some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) = true
        simp only [h, beq_self_eq_true, Bool.or_true]
    rw [hall] at hk
    cases hk
  simp (config := {zeta := false}) only [hk, Bool.false_eq_true, ↓reduceIte]
  extract_lets values names distinct jp
  simp (config := {zeta := false, failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp (config := {zeta := false}) only [jp]
  show (forIn supplied (forIn mine _ _ : Id (Array String)).run _ : Id (Array String)).run = #[] ↔ _
  refine (forIn_codes_nil _ _ _ (fun x => ∃ want, mine.find? (·.value == x.1) = some want ∧
    ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str x.1)) = some e →
      ∃ r, rootOf x.1 x.2 want = .ok r ∧ jeq (getD e "root") (.str r)) ?_).trans ?_
  · intro x c
    rcases x with ⟨v, rs⟩
    dsimp only []
    split
    · rename_i hm
      refine ⟨rfl, ?_⟩
      show addCode c "R7-supplied" = #[] ↔ _
      simp only [addCode_ne_nil, false_iff, not_and, not_exists]
      intro _ want hw
      rw [hm] at hw
      cases hw
    · rename_i want hm
      split
      · rename_i e hfe
        split
        · rename_i err hr
          refine ⟨rfl, ?_⟩
          show addCode c "R7-supplied" = #[] ↔ _
          simp only [addCode_ne_nil, false_iff, not_and, not_exists]
          intro _ want' hw h
          rw [hm] at hw
          cases hw
          obtain ⟨r, hr', _⟩ := h e hfe
          rw [hr] at hr'
          cases hr'
        · rename_i r hr
          have hQ : (∃ want', mine.find? (·.value == v) = some want' ∧
              ∀ e', es.reverse.find? (fun e => jeq (getD e "value") (.str v)) = some e' →
                ∃ r', rootOf v rs want' = .ok r' ∧ jeq (getD e' "root") (.str r') = true) ↔
              jeq (getD e "root") (.str r) = true := by
            constructor
            · rintro ⟨want', hw, h⟩
              rw [hm] at hw
              cases hw
              obtain ⟨r', hr', hj⟩ := h e hfe
              rw [hr] at hr'
              cases hr'
              exact hj
            · intro hj
              exact ⟨want, hm, fun e' he' => by rw [hfe] at he'; cases he'; exact ⟨r, hr, hj⟩⟩
          rw [hQ]
          cases hj : jeq (getD e "root") (.str r)
          · refine ⟨rfl, ?_⟩
            show addCode c "R7-supplied" = #[] ↔ _
            simp [addCode_ne_nil]
          · refine ⟨rfl, ?_⟩
            show c = #[] ↔ _
            simp
      · rename_i hfe
        refine ⟨rfl, ?_⟩
        show c = #[] ↔ _
        exact ⟨fun h => ⟨h, want, hm, fun e he => (hfe e he).elim⟩, fun h => h.1⟩
  refine (and_congr (forIn_codes_nil _ _ _ (fun (want : Registered.Entry) => ∀ e,
    es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
      EntryIs e want (heldFor supplied held want.value)) ?_) Iff.rfl).trans ?_
  · intro want c
    split
    · rename_i e hfe
      extract_lets c1 bw jh jw jd jwo jl cr
      have hH : ∀ c, yields (jh () c) = true ∧ (codesOf (jh () c) = #[] ↔ c = #[] ∧
          ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jh, heldFor]
        cases hs : supplied.any (·.1 == want.value)
        · simp only [Bool.not_false, ↓reduceIte, Bool.false_eq_true]
          cases hf : held.find? (·.1 == want.value) with
          | none => simp [step_pure]
          | some x =>
            rcases x with ⟨fst, r⟩
            simp only []
            cases hj : jeq (getD e "root") (.str r) <;> simp [step_pure, hj, addCode_ne_nil]
        · simp [step_pure]
      have hW : ∀ c, yields (jw () c) = true ∧ (codesOf (jw () c) = #[] ↔ c = #[] ∧ bw = false ∧
          ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jw]
        by_cases hb : bw = true <;> simp [hb, hH, addCode_ne_nil]
      have hD : ∀ c, yields (jd () c) = true ∧ (codesOf (jd () c) = #[] ↔ c = #[] ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jd]
        by_cases hb : jeq (getD e "domain") (.str want.domain) = true <;> simp [hb, hW, addCode_ne_nil]
      have hWo : ∀ c, yields (jwo () c) = true ∧ (codesOf (jwo () c) = #[] ↔ c = #[] ∧
          int? (getD e "words") = some (want.words : Int) ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jwo]
        by_cases hb : int? (getD e "words") = some (want.words : Int) <;> simp [hb, hD, addCode_ne_nil]
      have hL : ∀ c, yields (jl () c) = true ∧ (codesOf (jl () c) = #[] ↔ c = #[] ∧
          int? (getD e "leaves") = some (want.leaves : Int) ∧
          int? (getD e "words") = some (want.words : Int) ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jl]
        by_cases hb : int? (getD e "leaves") = some (want.leaves : Int) <;> simp [hb, hWo, addCode_ne_nil]
      have hR := root_ok (getD e "root")
      have hBw : bw = false ↔ _ := width_ok e want
      have hQ : (∀ e', Array.find? (fun e => jeq (getD e "value") (Json.str want.value)) es.reverse = some e' →
          EntryIs e' want (heldFor supplied held want.value)) ↔ EntryIs e want (heldFor supplied held want.value) := by
        rw [hfe]
        constructor
        · intro h; exact h e rfl
        · intro h e' he; cases he; exact h
      rw [hQ]
      unfold EntryIs
      have key : ∀ b : Bool, (b = false ↔ ∃ s, getD e "root" = .str s ∧ s.length = 128 ∧ isHex s = true) →
          yields (if b = true then jl () cr else jl () c1) = true ∧
          (codesOf (if b = true then jl () cr else jl () c1) = #[] ↔ c = #[] ∧
            (∃ s, getD e "root" = .str s ∧ s.length = 128 ∧ isHex s = true) ∧
            int? (getD e "leaves") = some (want.leaves : Int) ∧ int? (getD e "words") = some (want.words : Int) ∧
            jeq (getD e "domain") (.str want.domain) = true ∧
            (match field? e Registered.WIDTH_KEY with
              | none => want.wordBits = Registered.WORD_BITS
              | some j => int? j = some (want.wordBits : Int) ∧ (want.wordBits : Int) ≠ Registered.WORD_BITS) ∧
            ∀ x ∈ (heldFor supplied held want.value).find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro b hb
        cases b
        · have hr' := hb.1 rfl
          show yields (jl () c1) = true ∧ (codesOf (jl () c1) = #[] ↔ _)
          refine ⟨(hL c1).1, ?_⟩
          rw [(hL c1).2, hBw]
          exact ⟨fun ⟨a, b, c, d, f, g⟩ => ⟨a, hr', b, c, d, f, g⟩, fun ⟨a, _, b, c, d, f, g⟩ => ⟨a, b, c, d, f, g⟩⟩
        · show yields (jl () cr) = true ∧ (codesOf (jl () cr) = #[] ↔ _)
          have hcr : cr ≠ #[] := addCode_ne_nil _ _
          refine ⟨(hL cr).1, ?_⟩
          rw [(hL cr).2]
          refine ⟨fun h => (hcr h.1).elim, fun ⟨_, h1, _⟩ => ?_⟩
          exact absurd (hb.2 h1) (by decide)
      exact key _ hR
    · rename_i hfe
      refine ⟨rfl, ?_⟩
      show c = #[] ↔ _
      exact ⟨fun h => ⟨h, fun e he => (hfe e he).elim⟩, fun h => h.1⟩
  · have hnames : names = namesOf es := by
      simp only [names, values, namesOf, Array.filterMap_map]; rfl
    have hValues : ((names.size != values.size || distinct.length != values.size || distinct.length != mine.size ||
        !distinct.all fun v => mine.any fun x => x.value == v) = false) ↔ ValuesAre es mine := by
      simp only [distinct, values]
      rw [hnames]
      simp only [Array.size_map, Bool.or_eq_false_iff, bne_eq_false_iff_eq, Bool.not_eq_false', List.all_eq_true,
        Array.any_eq_true', beq_iff_eq, ValuesAre, and_assoc]
    have hkeys : ∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
        keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY)) := by
      intro e he
      have := Array.all_eq_true'.1 (Bool.not_eq_false' _ ▸ (Bool.eq_false_iff.2 hk)) e he
      simp only [Bool.or_eq_true, beq_iff_eq] at this
      exact this
    have key : ∀ (b : Bool) (u v : Array String), v = c → u ≠ #[] → (b = false ↔ ValuesAre es mine) →
        ((((if b = true then u else v) = #[] ∧ ∀ x ∈ mine, ∀ e,
          es.reverse.find? (fun e => jeq (getD e "value") (.str x.value)) = some e →
            EntryIs e x (heldFor supplied held x.value)) ∧ SuppliedIs es mine supplied rootOf) ↔
        c = #[] ∧ ∃ mine' es', (Except.ok mine : Flock.V (Array Registered.Entry)) = .ok mine' ∧ Json.arr es = .arr es' ∧
          (∀ e ∈ es', keys? e = some Registered.ENTRY_KEYS ∨
            keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
          ValuesAre es' mine' ∧
          (∀ want ∈ mine', ∀ e, es'.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
            EntryIs e want (heldFor supplied held want.value)) ∧
          SuppliedIs es' mine' supplied rootOf) := by
      intro b u v hv hu hb
      subst hv
      constructor
      · rintro ⟨⟨h1, h2⟩, h3⟩
        cases b
        · exact ⟨h1, mine, es, rfl, rfl, hkeys, hb.1 rfl, h2, h3⟩
        · exact (hu h1).elim
      · rintro ⟨hc, mine', es', hm, he, _, hv, h2, h3⟩
        cases hm
        cases he
        refine ⟨⟨?_, h2⟩, h3⟩
        rw [hb.2 hv]
        exact hc
    exact key _ _ _ rfl (addCode_ne_nil _ _) hValues

/-- `checkRegistered` with nothing supplied: every value held to `held`, and `rootOf` never read. -/
theorem checkRegistered_nil (c : Array String) (got : Json) (mine : Flock.V (Array Registered.Entry))
    (held : Array (String × String)) (rootOf : String → Supplied.Rows → Registered.Entry → Flock.V String) :
    Registration.checkRegistered c got mine held #[] rootOf = #[] ↔ c = #[] ∧ ∃ mine' es, mine = .ok mine' ∧
      got = .arr es ∧
      (∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
        keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
      ValuesAre es mine' ∧
      ∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
        EntryIs e want held := by
  have hh : ∀ v, heldFor (#[] : Array (String × Supplied.Rows)) held v = held := fun _ => by simp [heldFor]
  have hs : ∀ es mine', SuppliedIs es mine' #[] rootOf := fun _ _ x hx => absurd hx (Array.not_mem_empty x)
  simp only [checkRegistered_nil_supplied, hh, hs, and_true]

/-- R7-supplied: `checkRegistered` holds each value it fills to the root `rootOf` recomputes, and to no root `held`
names: an entry with any other root, or a supplied value whose root doesn't compute, is refused. -/
theorem checkRegistered_supplied (c : Array String) (got : Json) (mine : Flock.V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → Flock.V String)
    (h : Registration.checkRegistered c got mine held supplied rootOf = #[]) :
    ∃ mine' es, mine = .ok mine' ∧ got = .arr es ∧ SuppliedIs es mine' supplied rootOf ∧
      ∀ x ∈ supplied, heldFor supplied held x.1 = #[] := by
  obtain ⟨-, mine', es, hm, hg, -, -, -, hs⟩ := (checkRegistered_nil_supplied _ _ _ _ _ _).1 h
  refine ⟨mine', es, hm, hg, hs, fun x hx => ?_⟩
  have : supplied.any (·.1 == x.1) = true := Array.any_eq_true'.2 ⟨x, hx, by simp⟩
  simp [heldFor, this]

theorem outputRoot_some (log : Array Json) (n : Nat) (d p : String) :
    (∃ x, Registration.outputRoot log n d p = some x) ↔ OutputPortIs log n d p := by
  unfold Registration.outputRoot OutputPortIs
  cases Registration.outputPort log d p with
  | none => simp
  | some r =>
    simp only [Option.bind_some, Option.some.injEq, exists_eq_left']
    have hb : (int? (getD r "leaves") == some (n : Int)) = true ↔ int? (getD r "leaves") = some (n : Int) :=
      beq_iff_eq
    rw [← hb]
    cases (int? (getD r "leaves") == some (n : Int)) <;> cases (getD r "domain") <;> cases (getD r "root") <;> simp

theorem valueRoot_some (log : Array Json) (n w b : Nat) (d u : String) :
    (∃ x, Registration.valueRoot log n w b d u = some x) ↔ OutputValueIs log n w b d u := by
  unfold Registration.valueRoot OutputValueIs
  cases Registration.outputValue log d u with
  | none => simp
  | some r =>
    simp only [Option.bind_some, Option.some.injEq, exists_eq_left']
    have hb : (int? (getD r "leaves") == some (n : Int) && int? (getD r "words") == some (w : Int)
        && Registration.widthOf r == some (b : Int)) = true ↔
        int? (getD r "leaves") = some (n : Int) ∧ int? (getD r "words") = some (w : Int) ∧
          Registration.widthOf r = some (b : Int) := by
      simp only [Bool.and_eq_true, beq_iff_eq, and_assoc]
    rw [← hb]
    generalize (int? (getD r "leaves") == some (n : Int) && int? (getD r "words") == some (w : Int)
        && Registration.widthOf r == some (b : Int)) = t
    cases t <;> cases (getD r "domain") <;> cases (getD r "root") <;> simp

theorem earlierRoot_some (log : Array Json) (want : Registered.Entry) (d : String) (e : Registration.Earlier) :
    (∃ x, Registration.earlierRoot log want d e = some x) ↔
      EarlierIs log want.leaves want.words want.wordBits d e := by
  cases e with
  | port p => exact outputRoot_some log want.leaves d p
  | value u => exact valueRoot_some log want.leaves want.words want.wordBits d u

/-- An entry's shape: its leaf count, words and word bits, all an output value's earlier entry is held to. -/
abbrev shape (e : Registered.Entry) : Nat × Nat × Nat := (e.leaves, e.words, e.wordBits)

/-- Rewriting the domain of `v`'s entries keeps every value's entry and its shape. -/
theorem find_map_domain (mine : Array Registered.Entry) (v dom u : String) :
    ((mine.map fun e => if e.value == v then { e with domain := dom } else e).find? (·.value == u)).map shape =
      (mine.find? (·.value == u)).map shape := by
  have hv : ((fun x : Registered.Entry => x.value == u) ∘
      fun e => if e.value == v then { e with domain := dom } else e) = fun x => x.value == u := by
    funext e; simp only [Function.comp]; split <;> rfl
  have hl : (shape ∘ fun e => if e.value == v then { e with domain := dom } else e) = shape := by
    funext e; simp only [Function.comp]; split <;> rfl
  rw [Array.find?_map, hv, Option.map_map, hl]

theorem find_leaves_congr {m m' : Array Registered.Entry} {u : String}
    (h : (m'.find? (·.value == u)).map shape = (m.find? (·.value == u)).map shape) (P : Nat → Nat → Nat → Prop) :
    (∀ want, m'.find? (·.value == u) = some want → P want.leaves want.words want.wordBits) ↔
      ∀ want, m.find? (·.value == u) = some want → P want.leaves want.words want.wordBits := by
  revert h
  cases h1 : m'.find? (·.value == u) with
  | none =>
    cases h2 : m.find? (·.value == u) with
    | none => intro _; exact Iff.rfl
    | some b => intro h; simp at h
  | some a =>
    cases h2 : m.find? (·.value == u) with
    | none => intro h; simp at h
    | some b =>
      intro h
      obtain ⟨h1, h2, h3⟩ : a.leaves = b.leaves ∧ a.words = b.words ∧ a.wordBits = b.wordBits := by
        simpa [shape] using h
      constructor
      · intro H w hw; cases hw; rw [← h1, ← h2, ← h3]; exact H a rfl
      · intro H w hw; cases hw; rw [h1, h2, h3]; exact H b rfl

theorem applyOutput_step (log : Array Json) (c : Array String) (mine : Array Registered.Entry)
    (held : Array (String × String)) (v d : String) (p : Registration.Earlier) :
    ((Registration.applyOutput log (c, mine, held) (v, d, p)).1 = #[] ↔ c = #[] ∧
      ∀ want, mine.find? (·.value == v) = some want → EarlierIs log want.leaves want.words want.wordBits d p) ∧
    ∀ u, ((Registration.applyOutput log (c, mine, held) (v, d, p)).2.1.find? (·.value == u)).map shape =
      (mine.find? (·.value == u)).map shape := by
  simp only [Registration.applyOutput]
  cases hf : mine.find? (·.value == v) with
  | none => simp
  | some want =>
    dsimp only
    cases ho : Registration.earlierRoot log want d p with
    | none =>
      dsimp only
      refine ⟨?_, fun _ => rfl⟩
      simp only [addCode_ne_nil, false_iff, not_and]
      intro _ h
      obtain ⟨x, hx⟩ := (earlierRoot_some log want d p).2 (h want rfl)
      rw [ho] at hx
      cases hx
    | some dr =>
      rcases dr with ⟨dom, root⟩
      dsimp only
      refine ⟨⟨fun hc => ⟨hc, fun w hw => ?_⟩, fun h => h.1⟩, fun u => find_map_domain mine v dom u⟩
      cases hw
      exact (earlierRoot_some log want d p).1 ⟨_, ho⟩

/-- `applyOutputs` adds no code exactly when each output value with an entry has its earlier port or value. -/
theorem applyOutputs_nil (log : Array Json) (outputs : Array (String × String × Registration.Earlier))
    (c : Array String) (mine : Array Registered.Entry) (held : Array (String × String)) :
    (Registration.applyOutputs c mine held outputs log).1 = #[] ↔ c = #[] ∧
      ∀ o ∈ outputs, ∀ want, mine.find? (·.value == o.1) = some want →
        EarlierIs log want.leaves want.words want.wordBits o.2.1 o.2.2 := by
  unfold Registration.applyOutputs
  rw [← Array.foldl_toList]
  have hm : ∀ o, o ∈ outputs ↔ o ∈ outputs.toList := fun o => Array.mem_toList_iff.symm
  simp only [hm]
  generalize outputs.toList = l
  induction l generalizing c mine held with
  | nil => simp
  | cons o os ih =>
    rcases o with ⟨v, d, p⟩
    rw [List.foldl_cons]
    obtain ⟨h1, h2⟩ := applyOutput_step log c mine held v d p
    revert h1 h2
    generalize Registration.applyOutput log (c, mine, held) (v, d, p) = r
    rcases r with ⟨c', mine', held'⟩
    intro h1 h2
    dsimp only at h1 h2
    refine (ih c' mine' held').trans ?_
    rw [h1]
    simp only [List.mem_cons, forall_eq_or_imp, and_assoc]
    exact and_congr Iff.rfl (and_congr Iff.rfl (forall_congr' fun o => forall_congr' fun _ =>
      find_leaves_congr (h2 o.1) (fun n w b => EarlierIs log n w b o.2.1 o.2.2)))

/-- R7 with output values (`checkOutputs`): every output value with an entry has its earlier port or value, and the rest of R7
holds against the entries and held roots `applyOutputs` leaves. -/
theorem checkOutputs_nil (c : Array String) (got : Json) (mine : Flock.V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → Flock.V String)
    (outputs : Array (String × String × Registration.Earlier)) (log : Array Json) :
    Registration.checkOutputs c got mine held supplied rootOf outputs log = #[] ↔ c = #[] ∧
      ∃ mine0 es mine' held', mine = .ok mine0 ∧ got = .arr es ∧
      (∀ o ∈ outputs, ∀ want, mine0.find? (·.value == o.1) = some want →
        EarlierIs log want.leaves want.words want.wordBits o.2.1 o.2.2) ∧
      (Registration.applyOutputs #[] mine0 held outputs log).2 = (mine', held') ∧
      (∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
        keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
      ValuesAre es mine' ∧
      (∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
        EntryIs e want (heldFor supplied held' want.value)) ∧
      SuppliedIs es mine' supplied rootOf := by
  unfold Registration.checkOutputs
  rcases mine with err | mine0
  · simp [addCode_ne_nil]
  dsimp only
  by_cases h0 : c = #[]
  · subst h0
    have hc := applyOutputs_nil log outputs #[] mine0 held
    revert hc
    generalize hr : Registration.applyOutputs #[] mine0 held outputs log = r
    rcases r with ⟨c', mine', held'⟩
    intro hc
    dsimp only at hc ⊢
    rw [checkRegistered_nil_supplied]
    constructor
    · rintro ⟨hc', mine'', es, hm, hg, hk, hv, he, hsu⟩
      cases hm
      exact ⟨rfl, mine0, es, mine', held', rfl, hg, (hc.1 hc').2, by rw [hr], hk, hv, he, hsu⟩
    · rintro ⟨-, m0, es, m, h, hm0, hg, hf, hmh, hk, hv, he, hsu⟩
      cases hm0
      rw [hr] at hmh
      simp only [Prod.mk.injEq] at hmh
      obtain ⟨rfl, rfl⟩ := hmh
      exact ⟨hc.2 ⟨rfl, hf⟩, _, es, rfl, hg, hk, hv, he, hsu⟩
  · have hc := applyOutputs_nil log outputs c mine0 held
    revert hc
    generalize Registration.applyOutputs c mine0 held outputs log = r
    rcases r with ⟨c', mine', held'⟩
    intro hc
    dsimp only at hc ⊢
    rw [checkRegistered_nil_supplied]
    constructor
    · rintro ⟨hc', -⟩; exact absurd (hc.1 hc').1 h0
    · rintro ⟨h, -⟩; exact absurd h h0

theorem step_ite {P : Prop} (Q : Prop) (j' j : Unit → Array String → Id (Array String)) (b : Bool) (x : String)
    (hj : ∀ c, j () c = #[] ↔ c = #[] ∧ P)
    (hdef : ∀ c, j' () c = if b = true then j () (addCode c x) else j () c)
    (hb : b = false ↔ Q) : ∀ c, j' () c = #[] ↔ c = #[] ∧ Q ∧ P := by
  intro c
  rw [hdef]
  cases b
  · simp only [Bool.false_eq_true, ↓reduceIte, hj]
    exact ⟨fun ⟨a, p⟩ => ⟨a, hb.1 rfl, p⟩, fun ⟨a, _, p⟩ => ⟨a, p⟩⟩
  · simp only [↓reduceIte, hj, addCode_ne_nil, false_and, false_iff]
    exact fun ⟨_, q, _⟩ => absurd (hb.2 q) (by decide)

theorem step_ite2 {P R : Prop} (Q1 Q2 : Prop) (j' j8 j7 : Unit → Array String → Id (Array String)) (b1 b2 : Bool)
    (x y : String)
    (h8 : ∀ c, j8 () c = #[] ↔ c = #[] ∧ P) (h7 : ∀ c, j7 () c = #[] ↔ c = #[] ∧ R ∧ P)
    (hdef : ∀ c, j' () c =
      if b1 = true then j8 () (addCode c x) else if b2 = true then j7 () (addCode c y) else j7 () c)
    (hb1 : b1 = false ↔ Q1) (hb2 : b2 = false ↔ Q2) :
    ∀ c, j' () c = #[] ↔ c = #[] ∧ Q1 ∧ Q2 ∧ R ∧ P := by
  intro c
  rw [hdef]
  cases b1
  · cases b2
    · simp only [Bool.false_eq_true, ↓reduceIte, h7]
      exact ⟨fun ⟨a, r, p⟩ => ⟨a, hb1.1 rfl, hb2.1 rfl, r, p⟩, fun ⟨a, _, _, r, p⟩ => ⟨a, r, p⟩⟩
    · simp only [Bool.false_eq_true, ↓reduceIte, h7, addCode_ne_nil, false_and, false_iff]
      exact fun ⟨_, _, q, _⟩ => absurd (hb2.2 q) (by decide)
  · simp only [↓reduceIte, h8, addCode_ne_nil, false_and, false_iff]
    exact fun ⟨_, q, _⟩ => absurd (hb1.2 q) (by decide)

theorem forIn_list_codes_nil {α : Type} (l : List α) (c0 : Array String)
    (f : α → Array String → Id (ForInStep (Array String))) (Q : α → Prop)
    (hc : ∀ x c, yields (f x c) = true ∧ (codesOf (f x c) = #[] ↔ c = #[] ∧ Q x)) :
    @Eq (Array String) (forIn (m := Id) l c0 f) #[] ↔ c0 = #[] ∧ ∀ x ∈ l, Q x := by
  induction l generalizing c0 with
  | nil => simp; rfl
  | cons a as ih =>
    rw [List.forIn_cons, yields_eq _ (hc a c0).1]
    simp only [pure_bind]
    refine (ih _).trans ?_
    rw [(hc a c0).2]
    simp only [List.mem_cons, forall_eq_or_imp, and_assoc]

theorem step_forIn {α : Type} {P : Prop} (Q : α → Prop) (j' j : Unit → Array String → Id (Array String))
    (xs : Array α) (f : α → Array String → Id (ForInStep (Array String)))
    (hj : ∀ c, j () c = #[] ↔ c = #[] ∧ P)
    (hdef : ∀ c, j' () c = j () (forIn xs c f).run)
    (hf : ∀ x c, yields (f x c) = true ∧ (codesOf (f x c) = #[] ↔ c = #[] ∧ Q x)) :
    ∀ c, j' () c = #[] ↔ c = #[] ∧ (∀ x ∈ xs, Q x) ∧ P := by
  intro c; rw [hdef, hj, forIn_codes_nil xs c f Q hf, and_assoc]

theorem reset {P : Prop} {j : Unit → Array String → Id (Array String)} (h : ∀ c, j () c = #[] ↔ c = #[] ∧ P) :
    ∀ c, j () c = #[] ↔ c = #[] ∧ j () #[] = #[] := by
  intro c; rw [h c, h #[]]; simp

theorem not_false_iff' (a : Bool) : ((!a) = false) ↔ a = true := by cases a <;> decide

theorem any_false_iff {α : Type} (xs : Array α) (p : α → Bool) : xs.any p = false ↔ ∀ x ∈ xs, p x = false := by
  rw [Bool.eq_false_iff, ne_eq, Array.any_eq_true']
  simp

theorem hex_ok (j : Json) :
    ((!match j with | .str s => isHex s | _ => false) = false) ↔ ∃ s, j = .str s ∧ isHex s = true := by
  cases j <;> simp

theorem isNone_false_iff {α : Type} (o : Option α) : o.isNone = false ↔ ∃ a, o = some a := by
  cases o <;> simp

theorem not_and_false_iff (x y : Bool) : (!(x && y)) = false ↔ x = true ∧ y = true := by
  cases x <;> cases y <;> decide

theorem r2loop (g : String → Json) (a : Json) (oq : Flock.V String) (od : Option String) :
    (∀ x ∈ [("program", some a), ("query", oq.toOption.map Json.str), ("partition", od.map Json.str)],
      ∀ m, x.2 = some m → jeq (g x.1) m = true) ↔
    (jeq (g "program") a = true ∧ (∀ q, oq = .ok q → jeq (g "query") (.str q) = true) ∧
      ∀ d, od = some d → jeq (g "partition") (.str d) = true) := by
  cases oq <;> cases od <;> simp [Except.toOption]

theorem id_bind {α β : Type} (x : Id α) (f : α → Id β) : x >>= f = f x := rfl

theorem registration_nil (rec : Json) (v : Registration.Verifier) :
    Registration.check rec v = #[] ↔ ∃ law, Registered v rec law := by
  rw [Registration.check_eq_checkRef]
  unfold Registration.checkRef
  extract_lets v2 get c0 w program part partDigest mine roots previous j14 j13 j12 j11 j10 j9 j8 distinct j7 j6 j5 j4 j3 j2 j1 j0
    c1
  have h14 : ∀ c, j14 () c = #[] ↔ c = #[] ∧ True := fun c => ⟨fun h => ⟨h, trivial⟩, fun h => h.1⟩
  have h13 : ∀ c, j13 () c = #[] ↔ c = #[] ∧ (v2 = true → R7 rec v) ∧ True := by
    intro c
    simp only [j13]
    by_cases hv : v2 = true
    · simp only [hv, ↓reduceIte, true_implies]
      rw [h14, checkOutputs_nil]
      exact ⟨fun ⟨⟨a, b⟩, _⟩ => ⟨a, b, trivial⟩, fun ⟨a, b, _⟩ => ⟨⟨a, b⟩, trivial⟩⟩
    · simp only [hv, Bool.false_eq_true, ↓reduceIte, false_implies, h14]
      exact ⟨fun ⟨a, _⟩ => ⟨a, trivial, trivial⟩, fun ⟨a, _⟩ => ⟨a, trivial⟩⟩
  have h12 : ∀ c, j12 () c = #[] ↔ c = #[] ∧ (∀ sp, v.binding = some sp →
      jeq (getD (get "window") "stream") (.str sp.1) = true ∧
      jeq (getD (get "window") "prover") (.str sp.2) = true) ∧ ((v2 = true → R7 rec v) ∧ True) := by
    intro c
    simp only [j12]
    rcases hb : v.binding with _ | ⟨stream, prover⟩
    · simp [h13]
    · by_cases hs : jeq (getD (get "window") "stream") (.str stream) = true <;>
        by_cases hp : jeq (getD (get "window") "prover") (.str prover) = true <;>
        simp [hs, hp, h13, addCode_ne_nil]
  have h11 := step_ite (∀ e ∈ v.log, jeq (getD e "window") (getD rec "window") = false) j11 j12 _ _ h12
    (fun _ => rfl) (any_false_iff _ _)
  have h10 := step_ite (jeq (getD rec "previous") previous = true) j10 j11 _ _ h11 (fun _ => rfl) (not_false_iff' _)
  have h9 := step_ite (jeq (getD rec "anchors") (.arr v.anchors) = true) j9 j10 _ _ h10 (fun _ => rfl)
    (not_false_iff' _)
  have h8 := step_ite (R4 rec v) j8 j9 _ _ h9 (fun _ => rfl) (not_false_iff' _)
  have h7 := step_forIn (fun pw => ∀ r, rootFor rec pw.1 = some r → RootIs r pw.2) j7 j8 v.ports _ h8 (fun _ => rfl)
    (by
      intro pw c
      rcases pw with ⟨p, want⟩
      dsimp (config := {zeta := false}) only []
      split
      · rename_i r hfr
        extract_lets c' jd jl cr
        have hD : ∀ c, yields (jd () c) = true ∧ (codesOf (jd () c) = #[] ↔ c = #[] ∧
            jeq (getD r "domain") (getD want "domain") = true) := by
          intro c
          simp only [jd]
          by_cases hb : jeq (getD r "domain") (getD want "domain") = true <;> simp [hb, step_pure, addCode_ne_nil]
        have hL : ∀ c, yields (jl () c) = true ∧ (codesOf (jl () c) = #[] ↔ c = #[] ∧
            jeq (getD r "leaves") (getD want "leaves") = true ∧ jeq (getD r "domain") (getD want "domain") = true) := by
          intro c
          simp only [jl]
          by_cases hb : jeq (getD r "leaves") (getD want "leaves") = true <;> simp [hb, hD, addCode_ne_nil]
        have hQ : (∀ r', rootFor rec p = some r' → RootIs r' want) ↔ RootIs r want := by
          rw [show rootFor rec p = some r from hfr]
          exact ⟨fun h => h r rfl, fun h r' he => by cases he; exact h⟩
        have key : ∀ b : Bool, (b = false ↔ ∃ s, getD r "root" = .str s ∧ isHex s = true) →
            yields (if b = true then jl () cr else jl () c') = true ∧
            (codesOf (if b = true then jl () cr else jl () c') = #[] ↔ c = #[] ∧ RootIs r want) := by
          intro b hb
          unfold RootIs
          cases b
          · have hr' := hb.1 rfl
            show yields (jl () c') = true ∧ (codesOf (jl () c') = #[] ↔ _)
            refine ⟨(hL c').1, ?_⟩
            rw [(hL c').2]
            exact ⟨fun ⟨a, b, d⟩ => ⟨a, hr', b, d⟩, fun ⟨a, _, b, d⟩ => ⟨a, b, d⟩⟩
          · show yields (jl () cr) = true ∧ (codesOf (jl () cr) = #[] ↔ _)
            have hcr : cr ≠ #[] := addCode_ne_nil _ _
            refine ⟨(hL cr).1, ?_⟩
            rw [(hL cr).2]
            refine ⟨fun h => (hcr h.1).elim, fun ⟨_, h1, _⟩ => ?_⟩
            exact absurd (hb.2 h1) (by decide)
        show _ ∧ (_ ↔ c = #[] ∧ ∀ r', rootFor rec p = some r' → RootIs r' want)
        rw [hQ]
        exact key _ (hex_ok (getD r "root"))
      · rename_i hfr
        refine ⟨rfl, ?_⟩
        show c = #[] ↔ _
        exact ⟨fun h => ⟨h, fun r hr => (hfr r hr).elim⟩, fun h => h.1⟩)
  have h6 := step_ite2 (rootsOf rec ≠ #[] ∧ ∀ r ∈ rootsOf rec, keys? r = some Registration.ROOT_KEYS)
    ((portsOf rec).size = (rootsOf rec).size ∧ (portsOf rec).size = v.ports.size ∧
      ∀ pw ∈ v.ports, ∃ q ∈ portsOf rec, jeq q (.str pw.1) = true) j6 j8 j7 _ _ _ _ h8 h7 (fun _ => rfl)
    (by
      simp only [Bool.or_eq_false_iff, Bool.not_eq_false', Array.all_eq_true', beq_iff_eq, Array.isEmpty_eq_false_iff]
      exact Iff.rfl)
    (by
      simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq, Bool.not_eq_false', Array.all_eq_true', and_assoc]
      refine and_congr Iff.rfl (and_congr Iff.rfl (forall_congr' fun x => imp_congr_right fun _ => ?_))
      rcases x with ⟨p, w⟩
      exact Array.any_eq_true')
  have h5 := step_ite (jeq (getD rec "scheme") (.str v.scheme) = true) j5 j6 _ _ h6 (fun _ => rfl) (not_false_iff' _)
  have h5r := reset h5
  have h4 : ∀ c, j4 () c = #[] ↔ c = #[] ∧ (∀ x ∈ mine, ∀ m, x.2 = some m → jeq (get x.1) m = true) ∧
      (∃ n, Partition.units v.program v.query = .ok n ∧ jeq (get "population") (Lean.toJson n) = true) ∧
      j5 () #[] = #[] := by
    intro c
    simp only [j4, id_bind]
    rcases hu : Partition.units v.program v.query with a | n
    · rw [h5r]
      simp only [addCode_ne_nil, false_and, false_iff]
      rintro ⟨_, _, ⟨n, hn, _⟩, _⟩
      cases hn
    · simp only []
      cases hp : jeq (get "population") (Lean.toJson n)
      · simp only [Bool.not_false, ↓reduceIte]
        rw [h5r]
        simp only [addCode_ne_nil, false_and, false_iff]
        rintro ⟨_, _, ⟨n', hn, hp'⟩, _⟩
        cases hn
        rw [hp] at hp'
        cases hp'
      · simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
        rw [h5r]
        refine (and_congr_left' (forIn_list_codes_nil mine c _
          (fun x => ∀ m, x.2 = some m → jeq (get x.1) m = true) ?_)).trans ?_
        · intro x c
          rcases x with ⟨name, m⟩
          cases m with
          | none => exact ⟨rfl, Iff.intro (fun h => ⟨h, fun m hm => by cases hm⟩) (fun h => h.1)⟩
          | some m =>
            by_cases hj : jeq (get name) m = true <;> simp [hj, step_pure, addCode_ne_nil]
        · exact ⟨fun ⟨⟨a, b⟩, d⟩ => ⟨a, b, ⟨n, rfl, hp⟩, d⟩, fun ⟨a, b, _, d⟩ => ⟨⟨a, b⟩, d⟩⟩
  have h3 := step_ite (∃ d, partDigest = some d) j3 j4 _ _ h4 (fun _ => rfl) (isNone_false_iff _)
  have h2 := step_ite (jeq (getD part "program") (.str program) = true) j2 j3 _ _ h3 (fun _ => rfl)
    (not_false_iff' _)
  have h1 := step_ite ((w matches .obj _) = true ∧ ∃ k ∈ Registration.KINDS, jeq (getD w "kind") (.str k) = true)
    j1 j2 _ _ h2 (fun _ => rfl) (by
      simp only [Bool.or_eq_false_iff, Bool.not_eq_false', List.any_eq_true]
      exact Iff.rfl)
  have h1r := reset h1
  have h0 : ∀ c, j0 () c = #[] ↔ c = #[] ∧
      (∃ law, Draw.derive v.program v.query v.law v.workTable = .ok law ∧ jeq (get "law") law = true) ∧
      j1 () #[] = #[] := by
    intro c
    simp only [j0]
    rcases hd : Draw.derive v.program v.query v.law v.workTable with a | law
    · rw [h1r]
      simp only [addCode_ne_nil, false_and, false_iff]
      rintro ⟨_, ⟨law, hl, _⟩, _⟩
      cases hl
    · simp only []
      have hs : Draw.sameLaw (get "law") law = jeq (get "law") law := rfl
      rw [hs]
      cases hl : jeq (get "law") law
      · simp only [Bool.not_false, ↓reduceIte]
        rw [h1r]
        simp only [addCode_ne_nil, false_and, false_iff]
        rintro ⟨_, ⟨law', hl', hj⟩, _⟩
        cases hl'
        rw [hl] at hj
        cases hj
      · simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
        rw [h1r]
        exact ⟨fun ⟨a, b⟩ => ⟨a, ⟨law, rfl, hl⟩, b⟩, fun ⟨a, _, b⟩ => ⟨a, b⟩⟩
  have top : ∀ (bk bf : Bool),
      (bk = false ↔ keys? rec = some (if v2 = true then Registration.KEYS_V2 else Registration.KEYS)) →
      (bf = false ↔ jeq (get "format") (if v2 = true then Json.str Registration.FORMAT_V2 else Json.str Registration.FORMAT) = true ∧
        jeq (get "protocol") (Json.str Registration.PROTOCOL) = true) →
      ((if bk = true then pure #["keys"] else if bf = true then j0 () c1 else j0 () c0 : Id (Array String)).run = #[] ↔
        keys? rec = some (if v2 = true then Registration.KEYS_V2 else Registration.KEYS) ∧
        (jeq (get "format") (if v2 = true then Json.str Registration.FORMAT_V2 else Json.str Registration.FORMAT) = true ∧
          jeq (get "protocol") (Json.str Registration.PROTOCOL) = true) ∧
        (∃ law, Draw.derive v.program v.query v.law v.workTable = .ok law ∧ jeq (get "law") law = true) ∧
        j1 () #[] = #[]) := by
    intro bk bf hk hf
    cases bk
    · cases bf
      · show j0 () c0 = #[] ↔ _
        rw [h0]
        exact ⟨fun ⟨_, a, b⟩ => ⟨hk.1 rfl, hf.1 rfl, a, b⟩, fun ⟨_, _, a, b⟩ => ⟨rfl, a, b⟩⟩
      · show j0 () c1 = #[] ↔ _
        rw [h0]
        have hc1 : c1 ≠ #[] := addCode_ne_nil _ _
        exact ⟨fun h => (hc1 h.1).elim, fun ⟨_, h, _⟩ => absurd (hf.2 h) (by decide)⟩
    · show #["keys"] = #[] ↔ _
      exact ⟨fun h => by simp at h, fun ⟨h, _⟩ => absurd (hk.2 h) (by decide)⟩
  refine (top _ _ bne_eq_false_iff_eq (not_and_false_iff _ _)).trans ?_
  rw [h1 #[], h5 #[]]
  have hv2 : v2 = !v.registered.isEmpty := rfl
  have ek : (keys? rec = some (if v2 = true then Registration.KEYS_V2 else Registration.KEYS)) ↔
      keys? rec = some (if v.registered.isEmpty then Registration.KEYS else Registration.KEYS_V2) := by
    rw [hv2]; generalize v.registered.isEmpty = e; cases e <;> exact Iff.rfl
  have ef : jeq (get "format")
        (if v2 = true then Json.str Registration.FORMAT_V2 else Json.str Registration.FORMAT) = true ↔
      jeq (getD rec "format") (.str (if v.registered.isEmpty then Registration.FORMAT else Registration.FORMAT_V2)) =
        true := by
    rw [hv2]; generalize v.registered.isEmpty = e; cases e <;> exact Iff.rfl
  have e7 : (v2 = true → R7 rec v) ↔ (ExpectsV2 v → R7 rec v) := by
    unfold ExpectsV2; rw [hv2]; generalize v.registered.isEmpty = e; cases e <;> simp
  rw [ek, ef, e7]
  constructor
  · rintro ⟨hk, ⟨hf, hp⟩, ⟨law, hd, hl⟩, _, ⟨hw1, hw2⟩, hprog, ⟨d, hd'⟩, hloop, hpop, _, hsch, hroots, hports, h7',
      h4', h5', h6a, h6b, h6c, h7r, _⟩
    have ⟨l1, l2, l3⟩ := (r2loop get _ _ _).1 hloop
    exact ⟨law, ⟨hk, hf, hp, hd, hl, hw1, hw2⟩, ⟨hprog, l1, l2, ⟨d, hd', l3 d hd'⟩, hpop⟩,
      ⟨hsch, hroots.1, hroots.2, hports.1, hports.2.1, hports.2.2, h7'⟩, h4', h5', ⟨h6a, h6b, h6c⟩, h7r⟩
  · rintro ⟨law, ⟨hk, hf, hp, hd, hl, hw1, hw2⟩, ⟨hprog, l1, l2, ⟨d, hd', l3⟩, hpop⟩,
      ⟨hsch, hr1, hr2, hp1, hp2, hp3, h7'⟩, h4', h5', ⟨h6a, h6b, h6c⟩, h7r⟩
    refine ⟨hk, ⟨hf, hp⟩, ⟨law, hd, hl⟩, rfl, ⟨hw1, hw2⟩, hprog, ⟨d, hd'⟩,
      (r2loop get _ _ _).2 ⟨l1, l2, fun d' hd'' => ?_⟩, hpop, rfl, hsch, ⟨hr1, hr2⟩, ⟨hp1, hp2, hp3⟩, h7', h4', h5',
      h6a, h6b, h6c, h7r, trivial⟩
    have : some d' = some d := hd''.symm.trans hd'
    cases this
    exact l3
theorem ascending_iff (xs : Array Int) :
    ((List.range (xs.size - 1)).all fun i => decide (xs[i]! < xs[i + 1]!)) = true ↔ Ascending xs := by
  simp only [List.all_eq_true, List.mem_range, decide_eq_true_eq, Ascending]
  constructor
  · intro h i hi
    have := h i (by omega)
    rwa [getElem!_pos xs i (by omega), getElem!_pos xs (i + 1) hi] at this
  · intro h i hi
    rw [getElem!_pos xs i (by omega), getElem!_pos xs (i + 1) (by omega)]
    exact h i (by omega)

theorem checkDrawFull_obj_nil (kvs : Std.TreeMap.Raw String Json compare) (pop law : Json) :
    (Draw.checkDrawFull (.obj kvs) pop law).codes = #[] ↔ Drawn (.obj kvs) pop law := by
  unfold Draw.checkDrawFull Drawn
  extract_lets kind closure want have_ c0 n us j2 raw cs bad problem ok j1 j0 c1 dk dcl
  have hw : want = drawKeys kind closure := rfl
  have hkd : dk = kind := rfl
  have hcl : dcl = closure := rfl
  have hkeys : ∀ x, keys? (Json.obj kvs) = some x → have_ = x := by intro x h; simp [have_, h]
  simp only []
  by_cases hc : ["subset", "bernoulli", "stratified", "work"].contains kind = true
  rotate_left
  · simp only [hc, Bool.not_false, Bool.or_true, ↓reduceIte, Id.run_pure]
    constructor
    · intro h; exact absurd h (fail_ne_nil _ _ _)
    · rintro ⟨_, hm', _⟩
      rw [hkd] at hm'
      exact absurd (List.contains_iff_mem.2 hm') hc
  simp only [hc, Bool.not_true, Bool.or_false, Bool.false_eq_true, ↓reduceIte]
  by_cases hk : (have_ != want) = true
  · simp only [hk, ↓reduceIte, Id.run_pure]
    constructor
    · intro h; exact absurd h (fail_ne_nil _ _ _)
    · rintro ⟨_, _, hks, _⟩
      rw [hkd, hcl, ← hw] at hks
      rw [hkeys _ hks] at hk; simp at hk
  simp only [hk, Bool.false_eq_true, ↓reduceIte]
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp only [j0]
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp (config := {zeta := false}) only [j1]
  split
  · rename_i units hunits
    obtain ⟨xs, hxs, hm⟩ : ∃ xs, us = .arr xs ∧ xs.mapM int? = some units := by
      generalize hu : us = u at hunits
      cases u <;> simp at hunits
      exact ⟨_, rfl, hunits⟩
    extract_lets ja jb jc jd je out jf cu
    have hjb : jb.isEmpty = true ↔ ∀ ss, getD (Json.obj kvs) "strata" = .arr ss → ∀ s ∈ ss, ∀ t,
        getD s "template" = .str t → int? (getD s "k") = some (countIn units s) := by
      simp only [jb, Array.isEmpty_iff, Array.filterMap_eq_empty_iff]
      generalize getD (Json.obj kvs) "strata" = st
      cases st <;> simp only [Json.arr.injEq, reduceCtorEq, false_implies, implies_true, forall_eq', Array.not_mem_empty,
        false_implies, imp_false, iff_true]
      rename_i ss
      constructor
      · intro h s hs t ht
        have h1 := h s hs
        rw [ht] at h1
        apply Classical.byContradiction; intro hne
        simp only [countIn] at hne
        split at h1
        · simp at h1
        · rename_i hb; simp only [Bool.not_eq_true, bne_eq_false_iff_eq] at hb; exact hne hb
      · intro h s hs
        split
        · rename_i hb
          split
          · rename_i t ht; exfalso; have := h s hs t ht; simp only [countIn] at this; simp [this] at hb
          · rfl
        · rfl
    have hmem : kind ∈ ["subset", "bernoulli", "stratified", "work"] := by
      simpa only [List.contains_iff_mem] using hc
    have hkeys' : keys? (Json.obj kvs) = some (drawKeys kind closure) := by
      have : have_ = want := by simpa only [Bool.not_eq_true, bne_eq_false_iff_eq] using hk
      rw [← hw, ← this]; simp [have_, keys?]
    have hex : ∀ P : Array Int → Prop, (∃ us' units', getD (Json.obj kvs) "units" = .arr us' ∧
        us'.mapM int? = some units' ∧ P units') ↔ P units := by
      intro P; constructor
      · rintro ⟨us', units', h1, h2, h3⟩
        have e : Json.arr xs = Json.arr us' := hxs.symm.trans h1
        cases e; rw [hm] at h2; cases h2; exact h3
      · intro h; exact ⟨xs, units, hxs, hm, h⟩
    have hpn : ∀ m, int? (getD (Json.obj kvs) "population") = some m → Draw.strataMessage (Json.obj kvs) m = none →
        problem = none := by
      intro m h1 h2; simp only [problem, n, h1]; exact h2
    have hpe : problem = none → ∃ m, int? (getD (Json.obj kvs) "population") = some m ∧
        Draw.strataMessage (Json.obj kvs) m = none := by
      intro h; simp only [problem, n] at h; revert h
      cases int? (getD (Json.obj kvs) "population") <;> simp
    have hcs_some : ∀ ys, cs = some ys → ∃ zs, getD (Json.obj kvs) "closure" = .arr zs ∧ zs.mapM int? = some ys := by
      intro ys h; simp only [cs, raw] at h; revert h
      cases getD (Json.obj kvs) "closure" <;> simp
    have hcs_of : ∀ zs ys, getD (Json.obj kvs) "closure" = .arr zs → zs.mapM int? = some ys → cs = some ys := by
      intro zs ys h1 h2; simp only [cs, raw, h1]; exact h2
    simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
    simp only [jf]
    simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite', ↓jp_dite]
    simp only [je]
    simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
    simp only [jd]
    simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
    rcases hp : problem with _ | why <;> simp only [jc, hp] <;>
      simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite'] <;>
      rcases hcs : cs with _ | ys <;> simp only [ja, hcs] <;>
      simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite'] <;>
      simp only [j2, Id.run_pure] <;>
      simp only [ite_fail_nil, ite_ite_fail_nil, ite_ite_ite_fail_nil, dite_ite_fail_nil, c1, c0, cu]
    all_goals simp only [hkd, hcl, hex, and_assoc, true_and, hmem, hkeys']
    all_goals
      refine and_congr ?_ (and_congr ?_ (and_congr ?_ (and_congr ?_ (and_congr ?_ (and_congr ?_ (and_congr ?_ ?_))))))
      · simp [Draw.sameLaw]
      · simp only [Bool.not_eq_false']; exact Iff.rfl
      · simp only [Bool.not_eq_false']; exact ascending_iff units
      · apply forall_congr'; intro h
        rw [getElem!_pos units (units.size - 1) (by omega)]
        simp only [n, Bool.not_eq_false', Bool.and_eq_true, decide_eq_true_eq]
        cases int? (getD (Json.obj kvs) "population") <;> simp
      · by_cases hs : kind = "subset" <;> simp [hs]
      · simp only [beq_iff_eq, Bool.not_eq_false']
        apply imp_congr_right; intro _
        simp only [ok]
        split <;> simp_all
    all_goals first
      | (simp only [Bool.or_eq_true, beq_iff_eq, Bool.not_eq_false']
         apply imp_congr_right; intro _
         rw [hjb]
         obtain ⟨m, h1, h2⟩ := hpe hp
         exact ⟨fun h => ⟨m, h1, h2, h⟩, fun ⟨_, _, _, h⟩ => h⟩)
      | (simp only [Bool.or_eq_false_iff, beq_eq_false_iff_ne]
         constructor
         · rintro ⟨h1, h2⟩ (h | h) <;> contradiction
         · intro h
           refine ⟨fun hs => ?_, fun hw => ?_⟩
           · obtain ⟨m, h1, h2, _⟩ := h (.inl hs); rw [hpn m h1 h2] at hp; cases hp
           · obtain ⟨m, h1, h2, _⟩ := h (.inr hw); rw [hpn m h1 h2] at hp; cases hp)
      | (constructor
         · intro h1 h2; rw [h1] at h2; cases h2
         · intro h
           by_cases hc' : closure = true
           · obtain ⟨zs, ys, h1, h2, _⟩ := h hc'
             rw [hcs_of zs ys h1 h2] at hcs; cases hcs
           · simpa using hc')
      | (apply imp_congr_right; intro _
         obtain ⟨zs, h1, h2⟩ := hcs_some ys hcs
         have hx : ∀ Q : Array Int → Prop, (∃ cs' ys', getD (Json.obj kvs) "closure" = .arr cs' ∧
             cs'.mapM int? = some ys' ∧ Q ys') ↔ Q ys := by
           intro Q; constructor
           · rintro ⟨cs', ys', h3, h4, h5⟩; rw [h1] at h3; cases h3; rw [h2] at h4; cases h4; exact h5
           · intro h; exact ⟨zs, ys, h1, h2, h⟩
         simp only [hx]
         simp only [Bool.or_eq_false_iff, Bool.not_eq_false', Array.isEmpty_iff, qsort_eq_empty,
           Array.filter_eq_empty_iff, Array.contains_iff_mem, Array.isEmpty_eq_false_iff]
         by_cases hpos : 0 < ys.size
         · rw [Array.back?_eq_getElem?, Array.getElem?_eq_getElem hpos,
             Array.getElem?_eq_getElem (xs := ys) (i := ys.size - 1) (by omega)]
           simp only [n]
           constructor
           · rintro ⟨⟨⟨_, hasc⟩, hin⟩, hdis⟩
             refine ⟨hpos, (ascending_iff ys).1 hasc, ?_⟩
             revert hin; cases int? (getD (Json.obj kvs) "population") <;> simp_all
           · rintro ⟨_, hasc, h0, ⟨m, hm1, hm2⟩, hdis⟩
             refine ⟨⟨⟨?_, (ascending_iff ys).2 hasc⟩, ?_⟩, hdis⟩
             · intro he; rw [he] at hpos; simp at hpos
             · rw [hm1]; simp_all
         · simp only [Nat.not_lt, Nat.le_zero, Array.size_eq_zero_iff] at hpos
           subst hpos; simp)
  · rename_i hnone
    simp only [Id.run_pure]
    constructor
    · intro h; exact absurd h (fail_ne_nil _ _ _)
    · rintro ⟨_, _, _, _, _, us', units', h1, h2, _⟩
      exact (hnone units' (by simp only [us, h1]; exact h2)).elim
theorem checkDraw_nil (draw pop law : Json) : Draw.checkDraw draw pop law = #[] ↔ Drawn draw pop law := by
  unfold Draw.checkDraw
  cases draw
  case obj kvs => exact checkDrawFull_obj_nil kvs pop law
  all_goals
    constructor
    · intro h
      unfold Draw.checkDrawFull at h
      simp only [Bool.not_false, Bool.true_or, ↓reduceIte, Id.run_pure] at h
      exact absurd h (fail_ne_nil _ _ _)
    · rintro ⟨h, _⟩; cases h

theorem registered_law {v : Registration.Verifier} {rec law : Json} (h : Registered v rec law) :
    Draw.derive v.program v.query v.law v.workTable = .ok law := h.1.2.2.2.1

/-- The check accepts exactly what `AcceptMeans` says it accepts. -/
theorem acceptMeans_iff (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity) :
    OneStage.check p c e = .accept pub prof ↔
      ∃ law d n, Registered p.verifier c.record law ∧ Drawn e.draw (getD c.record "population") law ∧
        UnitsProved c e ∧ VerdictsAccept e.verdicts ∧
        Record.profileDraw law = some d ∧ nat? (getD c.record "population") = some n ∧
        pub = publicOf p e law d n ∧ prof = profileOf p c e d n := by
  unfold OneStage.check
  simp only []
  by_cases h1 : Registration.check c.record p.verifier = #[]
  · obtain ⟨law, hreg⟩ := (registration_nil _ _).1 h1
    have hd := registered_law hreg
    have uniq : ∀ law', Registered p.verifier c.record law' → law' = law := fun law' h' => by
      have := registered_law h'
      rw [hd] at this
      exact (Except.ok.inj this).symm
    simp only [h1, Array.isEmpty_empty, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hd]
    by_cases h2 : Draw.checkDraw e.draw (getD c.record "population") law = #[]
    · simp only [h2, Array.isEmpty_empty, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
      by_cases h3 : OneStage.recordChecks c.record (Flock.hexOf c.digest) e = #[]
      · simp only [h3, Array.isEmpty_empty, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
        have hu := (recordChecks_nil c e).1 h3
        have hdr := (checkDraw_nil _ _ _).1 h2
        rcases hv : OneStage.verdictOf e.verdicts with ⟨⟨⟩, ⟨⟩⟩ | r | r
        · have hva := (verdictOf_accept _).1 hv
          simp only []
          rcases hpd : Record.profileDraw law with _ | d <;>
            rcases hn : nat? (getD c.record "population") with _ | n
          · simp only [reduceCtorEq, false_iff]
            rintro ⟨law', d', n', hreg', -, -, -, hd', -⟩
            rw [uniq law' hreg', hpd] at hd'; cases hd'
          · simp only [reduceCtorEq, false_iff]
            rintro ⟨law', d', n', hreg', -, -, -, hd', -⟩
            rw [uniq law' hreg', hpd] at hd'; cases hd'
          · simp only [reduceCtorEq, false_iff]
            rintro ⟨law', d', n', -, -, -, -, -, hn', -⟩
            cases hn'
          · simp only [Core.Outcome.accept.injEq]
            constructor
            · rintro ⟨rfl, rfl⟩
              exact ⟨law, d, n, hreg, hdr, hu, hva, hpd, rfl, rfl, rfl⟩
            · rintro ⟨law', d', n', hreg', -, -, -, hd', hn', rfl, rfl⟩
              obtain rfl := uniq law' hreg'
              rw [hpd] at hd'; cases hd'
              cases hn'
              exact ⟨rfl, rfl⟩
        · simp only [reduceCtorEq, false_iff]
          rintro ⟨law', d', n', -, -, -, hva, -⟩
          have := (verdictOf_accept _).2 hva
          rw [hv] at this; cases this
        · simp only [reduceCtorEq, false_iff]
          rintro ⟨law', d', n', -, -, -, hva, -⟩
          have := (verdictOf_accept _).2 hva
          rw [hv] at this; cases this
      · have : (OneStage.recordChecks c.record (Flock.hexOf c.digest) e).isEmpty = false := by
          simpa [Array.isEmpty_iff] using h3
        simp only [this, Bool.not_false, ↓reduceIte, reduceCtorEq, false_iff]
        rintro ⟨law', d', n', -, -, hu, -⟩
        exact h3 ((recordChecks_nil c e).2 hu)
    · have : (Draw.checkDraw e.draw (getD c.record "population") law).isEmpty = false := by
        simpa [Array.isEmpty_iff] using h2
      simp only [this, Bool.not_false, ↓reduceIte, reduceCtorEq, false_iff]
      rintro ⟨law', d', n', hreg', hdr, -⟩
      obtain rfl := uniq law' hreg'
      exact h2 ((checkDraw_nil _ _ _).2 hdr)
  · have : (Registration.check c.record p.verifier).isEmpty = false := by
      simpa [Array.isEmpty_iff] using h1
    simp only [this, Bool.not_false, ↓reduceIte, reduceCtorEq, false_iff]
    rintro ⟨law', d', n', hreg', -⟩
    exact h1 ((registration_nil _ _).2 ⟨law', hreg'⟩)

theorem acceptMeans : AcceptMeans := fun p c e pub prof => (acceptMeans_iff p c e pub prof).1

/-- Completeness of `AcceptMeans`: a commitment whose registration keeps R1–R7, with a draw of the agreed law over the
registered population, its units proved and served as drawn, and verdicts that accept, is accepted, publishing
`publicOf` and the profile `profileOf`. -/
theorem acceptMeans_complete (p : Params) (c : Commitment) (e : Evidence) (law : Json) (d : Core.SamplingDesign) (n : Nat)
    (hr : Registered p.verifier c.record law) (hd : Drawn e.draw (getD c.record "population") law)
    (hu : UnitsProved c e) (hv : VerdictsAccept e.verdicts) (hp : Record.profileDraw law = some d)
    (hn : nat? (getD c.record "population") = some n) :
    OneStage.check p c e = .accept (publicOf p e law d n) (profileOf p c e d n) :=
  (acceptMeans_iff p c e _ _).2 ⟨law, d, n, hr, hd, hu, hv, hp, hn, rfl, rfl⟩

theorem digestIsRecord's : DigestIsRecord's := by
  intro b c h
  unfold OneStage.commitmentOf at h
  rcases hj : Core.canonicalJson b with err | j
  · rw [hj] at h; cases h
  · rw [hj] at h
    rcases hg : Registration.digest j with err | d
    · cases j <;> simp [bind, Except.bind, hg, throw, throwThe, MonadExceptOf.throw, Except.pure, pure] at h
    · have : c = { record := j, digest := d } := by
        cases j <;> simp [bind, Except.bind, hg, throw, throwThe, MonadExceptOf.throw, Except.pure, pure] at h <;>
          exact h.symm
      subst this
      exact hg

theorem bernoulliBounds : BernoulliBounds := by
  intro p c e pub prof h r l hl hd
  obtain ⟨law, d, n, -, -, -, -, -, -, rfl, rfl⟩ := acceptMeans p c e pub prof h
  simp only [profileOf, List.mem_toArray, List.mem_singleton] at hl
  subst hl
  simp only [Option.some.injEq] at hd
  subst hd
  rfl

end Properties.SampledProofs
