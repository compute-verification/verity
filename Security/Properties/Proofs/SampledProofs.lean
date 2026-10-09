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
theorem ite_push_nil (b : Bool) (c : Array String) (x : String) :
    ((if b = true then c.push x else c) = #[]) ↔ (c = #[] ∧ b = false) := by
  cases b <;> simp
theorem ite_ite_push_nil (a b : Bool) (c : Array String) (x : String) :
    ((if a = true then (if b = true then c.push x else c) else c) = #[]) ↔ (c = #[] ∧ (a = true → b = false)) := by
  cases a <;> cases b <;> simp

theorem addCode_ne_nil (c : Array String) (x : String) : addCode c x ≠ #[] := by
  unfold addCode
  split
  · rename_i h; intro hc; subst hc; simp at h
  · simp
theorem ite_addCode_nil (b : Bool) (c : Array String) (x : String) :
    ((if b = true then addCode c x else c) = #[]) ↔ (b = false ∧ c = #[]) := by
  cases b <;> simp [addCode_ne_nil]
theorem map_some_inj {α : Type} (xs ys : Array α) : xs.map some = ys.map some ↔ xs = ys := by
  rw [← Array.toList_inj, Array.toList_map, Array.toList_map, List.map_inj_right (fun _ _ h => Option.some.inj h),
    Array.toList_inj]
theorem toJson_nat_int (n : Nat) : int? (Lean.toJson n) = some (n : Int) := by
  simp [Lean.toJson, int?, Lean.JsonNumber.fromNat]

theorem checkProved_nil (draw : Json) (us : Array Nat) (n : Nat) :
    Draw.checkProved draw (us.map Lean.toJson) (Lean.toJson n) = #[] ↔
      n = (Draw.proved draw).size ∧ us.map (fun n : Nat => (n : Int)) = Draw.proved draw := by
  unfold Draw.checkProved
  extract_lets want c0 j1 j0 c1
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp only [j0]
  simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  simp only [j1, Id.run_pure, c1, c0]
  simp only [ite_addCode_nil, bne_eq_false_iff_eq, and_true, Array.map_map, Function.comp_def, toJson_nat_int,
    Option.some.injEq, Int.natCast_inj]
  rw [show (Array.map (fun (x : Nat) => some (x : Int)) us) = (us.map fun (x : Nat) => (x : Int)).map some by
    simp [Array.map_map, Function.comp_def], map_some_inj]
  exact And.comm

theorem recordChecks_nil (c : Commitment) (e : Evidence) :
    OneStage.recordChecks c.record (Flock.hexOf c.digest) e = #[] ↔ UnitsProved c e := by
  unfold OneStage.recordChecks
  extract_lets f0 j4 want opened j3 registered j2 units j1 j0 f1
  have hw : want = registeredRoots c.record := rfl
  have hu : units = (drawnInts e.draw).qsort (· < ·) := rfl
  have hreg : registered = rootsOf c.record := rfl
  unfold UnitsProved
  rcases hs : e.served with _ | served <;> rcases hr : e.sessionRoots with _ | roots
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j0]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j1, hs]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j2, hr]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j3]
  all_goals simp (config := {failIfUnchanged := false}) only [↓jp_ite, ↓jp_ite']
  all_goals simp only [j4, Id.run_pure, f1, f0]
  all_goals simp only [ite_ite_push_nil, ite_push_nil]
  all_goals simp only [and_assoc, opened, Bool.or_eq_true, Array.all_eq_true', Array.any_eq_true', Bool.and_eq_true,
    Array.any_eq_false', Bool.not_eq_true, Bool.not_eq_false', Array.isEmpty_eq_false_iff, 
    Array.isEmpty_iff, checkProved_nil, Bool.not_eq_true', bne_eq_false_iff_eq, Bool.or_eq_false_iff, Receipt,
    reduceCtorEq, false_implies, implies_true, true_and, and_true, Option.some.injEq, forall_eq', hu, hw, hreg]
  all_goals clear j0 j1 j2 j3 j4 f0 f1 opened want units registered hw hu hreg
  case some.some =>
    constructor
    · rintro ⟨h1, h2, h3, h4, h5, h6, h7⟩
      refine ⟨h1, h2, h3, h4, h5, fun x hx => ?_, h7⟩
      have := h6 x hx; revert this; unfold rootFor; split <;> simp_all
    · rintro ⟨h1, h2, h3, h4, h5, h6, h7⟩
      refine ⟨h1, h2, h3, h4, h5, fun x hx => ?_, h7⟩
      obtain ⟨reg, hreg, hj⟩ := h6 x hx
      unfold rootFor at hreg
      rw [hreg]; simp [hj]
  case none.some =>
    constructor
    · rintro ⟨h1, h2, h3, h5, h6, h7⟩
      refine ⟨h1, h2, h3, h5, fun x hx => ?_, h7⟩
      have := h6 x hx; revert this; unfold rootFor; split <;> simp_all
    · rintro ⟨h1, h2, h3, h5, h6, h7⟩
      refine ⟨h1, h2, h3, h5, fun x hx => ?_, h7⟩
      obtain ⟨reg, hreg, hj⟩ := h6 x hx
      unfold rootFor at hreg
      rw [hreg]; simp [hj]

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

theorem registration_nil (rec : Json) (v : Registration.Verifier) :
    Registration.check rec v = #[] ↔ ∃ law, Registered v rec law := sorry
theorem checkDraw_nil (draw pop law : Json) : Draw.checkDraw draw pop law = #[] ↔ Drawn draw pop law := sorry
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
          rcases hpd : OneStage.profileDraw law with _ | d <;>
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
