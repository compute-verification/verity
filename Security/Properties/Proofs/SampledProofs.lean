import Properties.SampledProofs

/-! # Proofs of sampled proofs' one-stage property (`Properties.SampledProofs`)

Each check of `OneStage.check` is a pipeline of failure codes (`addCode`, `push`) written with `let mut` and `if`, which
`do` elaborates into join points. A proof lifts the join points out (`extract_lets`), moves each `if` into the argument of
the join point it calls (`jp_ite`, `jp_ite'`), unfolds the join points one at a time, and reads the codes' emptiness off
the pipeline one check at a time (`ite_addCode_nil`, `ite_push_nil`).
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

theorem checkRegistered_nil (c : Array String) (got : Json) (mine : Flock.V (Array Registered.Entry))
    (held : Array (String × String)) :
    Registration.checkRegistered c got mine held = #[] ↔ c = #[] ∧ ∃ mine' es, mine = .ok mine' ∧ got = .arr es ∧
      (∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
        keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
      ValuesAre es mine' ∧
      ∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
        EntryIs e want held := by
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
  show (forIn mine _ _ : Id (Array String)).run = #[] ↔ _
  refine (forIn_codes_nil _ _ _ (fun want => ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
    EntryIs e want held) ?_).trans ?_
  · intro want c
    split
    · rename_i e hfe
      extract_lets c1 bw jh jw jd jwo jl cr
      have hH : ∀ c, yields (jh () c) = true ∧ (codesOf (jh () c) = #[] ↔ c = #[] ∧
          ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jh]
        cases hf : held.find? (·.1 == want.value) with
        | none => simp [step_pure]
        | some x =>
          rcases x with ⟨fst, r⟩
          simp only []
          cases hj : jeq (getD e "root") (.str r) <;> simp [step_pure, hj, addCode_ne_nil]
      have hW : ∀ c, yields (jw () c) = true ∧ (codesOf (jw () c) = #[] ↔ c = #[] ∧ bw = false ∧
          ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jw]
        by_cases hb : bw = true <;> simp [hb, hH, addCode_ne_nil]
      have hD : ∀ c, yields (jd () c) = true ∧ (codesOf (jd () c) = #[] ↔ c = #[] ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jd]
        by_cases hb : jeq (getD e "domain") (.str want.domain) = true <;> simp [hb, hW, addCode_ne_nil]
      have hWo : ∀ c, yields (jwo () c) = true ∧ (codesOf (jwo () c) = #[] ↔ c = #[] ∧
          int? (getD e "words") = some (want.words : Int) ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jwo]
        by_cases hb : int? (getD e "words") = some (want.words : Int) <;> simp [hb, hD, addCode_ne_nil]
      have hL : ∀ c, yields (jl () c) = true ∧ (codesOf (jl () c) = #[] ↔ c = #[] ∧
          int? (getD e "leaves") = some (want.leaves : Int) ∧
          int? (getD e "words") = some (want.words : Int) ∧
          jeq (getD e "domain") (.str want.domain) = true ∧ bw = false ∧
          ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
        intro c
        simp only [jl]
        by_cases hb : int? (getD e "leaves") = some (want.leaves : Int) <;> simp [hb, hWo, addCode_ne_nil]
      have hR := root_ok (getD e "root")
      have hBw : bw = false ↔ _ := width_ok e want
      have hQ : (∀ e', Array.find? (fun e => jeq (getD e "value") (Json.str want.value)) es.reverse = some e' →
          EntryIs e' want held) ↔ EntryIs e want held := by
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
            ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)) := by
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
        (((if b = true then u else v) = #[] ∧ ∀ x ∈ mine, ∀ e,
          es.reverse.find? (fun e => jeq (getD e "value") (.str x.value)) = some e → EntryIs e x held) ↔
        c = #[] ∧ ∃ mine' es', (Except.ok mine : Flock.V (Array Registered.Entry)) = .ok mine' ∧ Json.arr es = .arr es' ∧
          (∀ e ∈ es', keys? e = some Registered.ENTRY_KEYS ∨
            keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
          ValuesAre es' mine' ∧
          ∀ want ∈ mine', ∀ e, es'.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
            EntryIs e want held) := by
      intro b u v hv hu hb
      subst hv
      constructor
      · rintro ⟨h1, h2⟩
        cases b
        · exact ⟨h1, mine, es, rfl, rfl, hkeys, hb.1 rfl, h2⟩
        · exact (hu h1).elim
      · rintro ⟨hc, mine', es', hm, he, _, hv, h2⟩
        cases hm
        cases he
        refine ⟨?_, h2⟩
        rw [hb.2 hv]
        exact hc
    exact key _ _ _ rfl (addCode_ne_nil _ _) hValues
theorem registration_nil (rec : Json) (v : Registration.Verifier) :
    Registration.check rec v = #[] ↔ ∃ law, Registered v rec law := sorry
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

theorem acceptMeans : AcceptMeans := by
  intro p c e pub prof
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

theorem bernoulliBoundsNothing : BernoulliBoundsNothing := by
  intro p c e pub prof r h hl
  obtain ⟨law, d, n, -, -, -, -, -, -, rfl, rfl⟩ := (acceptMeans p c e pub prof).1 h
  obtain ⟨l, hl, hd⟩ := hl
  simp only [profileOf, List.mem_toArray, List.mem_singleton] at hl
  subst hl
  simp only [Option.some.injEq] at hd
  subst hd
  rfl

end Properties.SampledProofs
