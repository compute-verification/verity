module

public import Proofs.Properties.InferenceOnly.BodiesOk

/-! Resolution from inside an activation stays inside it or leaves through one of its parameters. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.Resolve

open Flock.Program

def Under (U : Scope) : Scope → Prop
  | .top d off st => U = .top d off st
  | .child parent d off k m => U = .child parent d off k m ∨ Under U parent

theorem under_refl : ∀ U : Scope, Under U U
  | .top _ _ _ => rfl
  | .child _ _ _ _ _ => Or.inl rfl

theorem under_off (P : Prog) (U : Scope) : ∀ sc, Good P sc → Under U sc → U.off ≤ sc.off
  | .top d off st, _, hu => by
    have hu' : U = .top d off st := hu
    rw [hu']
  | .child parent d off k m, hg, hu => by
    have hu' : U = .child parent d off k m ∨ Under U parent := hu
    rcases hu' with hu' | hu'
    · rw [hu']
    · obtain ⟨hp, -, -, -, -, ho⟩ := hg
      have := under_off P U parent hp hu'
      show U.off ≤ off
      omega

theorem resolve_of_run (P : Prog) {n : ℕ} (hn : n ≤ Flock.Steps.FUEL) {sc : Scope} {r : Ref} {b : ℕ}
    (h : Flock.Steps.run "scope resolution" (resolveStep P) n (sc, r) = .ok b) : resolve P sc r = .ok b :=
  Flock.Steps.run_mono hn h

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

omit hroot hL in
theorem step_where (B : ℕ) (s : Scope × Ref) (x) (hs : Inv P B s) (h : resolveStep P s = .ok x) :
    (∀ a, x = .inl a → s.2.p = false ∧ s.1.off ≤ a) ∧
    (∀ s', x = .inr s' →
      (s.2.p = false ∧ ∃ d off k m, s'.1 = .child s.1 d off k m) ∨
      (s.2.p = true ∧ ∃ parent d off k m, s.1 = .child parent d off k m ∧
        (s'.1 = parent ∨ ∃ off' m', s'.1 = .child parent d off' k m'))) := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, -, -⟩ := hs
  dsimp only at hg hr
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj := (hr.2 rfl).2
    dsimp only at hi hj
    obtain ⟨-, -, -, hdown, -⟩ := (hB sc hg).1 i hi
    have hdn := hdown j hj
    rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form
    · rw [rs_prim P sc i j hf] at h
      cases h
      exact ⟨fun a ha => (by cases ha; exact ⟨rfl, Nat.le_add_right _ _⟩), fun s' hs' => by cases hs'⟩
    · rw [rs_call P sc i j hf] at h
      cases h
      exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact Or.inl ⟨rfl, _, _, _, _, rfl⟩⟩
    · rw [rs_batch P sc i j hf] at h
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true] at h
        cases h
        exact ⟨fun a ha => (by cases ha; exact ⟨rfl, Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _)⟩),
          fun s' hs' => by cases hs'⟩
      · simp only [hcp, Bool.false_eq_true, ite_false] at h
        cases h
        exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact Or.inl ⟨rfl, _, _, _, _, rfl⟩⟩
    · simp only [hf] at hdn
      obtain ⟨-, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · exact absurd h (rs_scan0 P sc i j x hf hn)
      obtain ⟨carry, y, hret, -, -⟩ := hdn hn
      rw [rs_scan P sc i j carry y hf hn hret] at h
      dsimp only at h
      cases h
      exact ⟨fun a ha => (by cases ha), fun s' hs' => by cases hs'; exact Or.inl ⟨rfl, _, _, _, _, rfl⟩⟩
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k, m⟩
    · unfold Good at hg
      obtain ⟨-, -, rfl⟩ := hg
      exact absurd h (rs_top P d off i j x)
    · rcases hf : ((P.defs[parent.d]!).body.nodes[k]!).form
      · rw [rs_up P parent d off k m i j (Or.inl hf)] at h
        cases h
        exact ⟨fun a ha => (by cases ha),
          fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inl rfl⟩⟩
      · rw [rs_up P parent d off k m i j (Or.inr hf)] at h
        cases h
        exact ⟨fun a ha => (by cases ha),
          fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inl rfl⟩⟩
      · rw [rs_up_batch P parent d off k m i j hf] at h
        cases h
        exact ⟨fun a ha => (by cases ha),
          fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inl rfl⟩⟩
      · by_cases hi0 : i = 0
        · subst hi0
          rw [rs_up_scan0 P parent d off k m j hf] at h
          by_cases hm0 : m = 0
          · simp only [hm0, ite_true] at h
            cases h
            exact ⟨fun a ha => (by cases ha),
              fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inl rfl⟩⟩
          · simp only [hm0, ite_false] at h
            cases h
            exact ⟨fun a ha => (by cases ha),
              fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inr ⟨_, _, rfl⟩⟩⟩
        · rw [rs_up_scan P parent d off k m i j hi0 hf] at h
          cases h
          exact ⟨fun a ha => (by cases ha),
            fun s' hs' => by cases hs'; exact Or.inr ⟨rfl, _, _, _, _, _, rfl, Or.inl rfl⟩⟩

theorem run_within (U : Scope) (B : ℕ) :
    ∀ (n : ℕ), n ≤ Flock.Steps.FUEL → ∀ (s : Scope × Ref) (b : ℕ), Inv P B s → Under U s.1 →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok b →
      (U.off ≤ b ∧ b < B) ∨ ∃ i j, i < (P.defs[U.d]!).body.params.size ∧
        j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok b
  | 0, _, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, hn, s, b, hs, hu, h => by
    have h' := h
    rw [Flock.Steps.run.eq_2] at h'
    rcases hx : resolveStep P s with e | (a | s')
    · simp [hx, bind, Except.bind] at h'
    · simp only [hx, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h'
      rw [← h']
      exact Or.inl ⟨Nat.le_trans (under_off P U s.1 hs.1 hu) ((step_where P hB B s _ hs hx).1 a rfl).2,
        (step_ok P hroot hB hL B s _ hs hx).1 a rfl⟩
    · simp only [hx, bind, Except.bind] at h'
      have hs' := (step_ok P hroot hB hL B s _ hs hx).2 s' rfl
      rcases (step_where P hB B s _ hs hx).2 s' rfl with
        ⟨-, d, off, k, m, hc⟩ | ⟨hp, parent, d, off, k, m, hsc, hc⟩
      · refine run_within U B n (by omega) s' b hs' ?_ h'
        rw [hc]
        exact Or.inr hu
      · by_cases hU : s.1 = U
        · obtain ⟨sc, ⟨p, i, j⟩⟩ := s
          have hp' : p = true := hp
          have hU' : sc = U := hU
          rw [hp', hU'] at h
          have hb : i < (P.defs[sc.d]!).body.params.size ∧ j < ((P.defs[sc.d]!).body.params[i]!).2.leaves :=
            hs.2.1.1 hp'
          rw [hU'] at hb
          exact Or.inr ⟨i, j, hb.1, hb.2, resolve_of_run P hn h⟩
        · have hup : Under U parent := by
            rw [hsc] at hu hU
            have hu' : U = .child parent d off k m ∨ Under U parent := hu
            rcases hu' with hu' | hu'
            · exact absurd hu'.symm hU
            · exact hu'
          rcases hc with hc | ⟨off', m', hc⟩
          · refine run_within U B n (by omega) s' b hs' ?_ h'
            rw [hc]
            exact hup
          · refine run_within U B n (by omega) s' b hs' ?_ h'
            rw [hc]
            exact Or.inr hup

theorem ret_within (U : Scope) (hU : Good P U) (t : ℕ) (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨ ∃ i j, i < (P.defs[U.d]!).body.params.size ∧
      j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok b :=
  run_within P hroot hB hL U _ Flock.Steps.FUEL (Nat.le_refl _) _ b
    (down_ok P hroot hB hL _ U hU (Nat.le_refl _) t ht) (under_refl U) h

theorem operands_within (U : Scope) (sc : Scope) (hg : Good P sc) (hu : Under U sc) (k : ℕ)
    (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hcp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true) (m : ℕ)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) {ops : Array ℕ}
    (h : operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops) :
    ∀ a ∈ ops, (U.off ≤ a ∧ a < sc.off + (P.defs[sc.d]!).body.offsets[k]!) ∨
      ∃ i j, i < (P.defs[U.d]!).body.params.size ∧ j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧
        resolve P U ⟨true, i, j⟩ = .ok a := by
  intro a ha
  obtain ⟨i, hi, t, ht, hres⟩ := operands_spec P sc _ _ ops h a ha
  obtain ⟨-, hargs, -, -, hop⟩ := (hB sc hg).1 k hk
  have hidx := hop hcp i hi m hm t ht
  have hr : (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
        (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).memberRef m t
      else (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get t) =
      (((P.defs[sc.d]!).body.nodes[k]!).args[i]!).seq.get
        (if (((P.defs[sc.d]!).body.nodes[k]!).form == Form.batch) = true then
          memberIdx (((P.defs[sc.d]!).body.nodes[k]!).args[i]!) m t
        else t) := by
    split <;> simp only [memberRef_eq]
  rw [hr] at hres
  exact run_within P hroot hB hL U _ Flock.Steps.FUEL (Nat.le_refl _) _ a
    (start_ok P hroot hB hL sc hg k hk _ (hargs i _ hidx)) hu hres

theorem walk_within (U : Scope) {g : ℕ} (n : ℕ) (sc : Scope) (x : ℕ × Array ℕ × Scope) (hq : WalkQ P g sc)
    (hu : Under U sc) (h : Flock.Steps.run "gate location" (gateStep P g) n sc = .ok x) :
    ∀ a ∈ x.2.1, (U.off ≤ a ∧ a < g) ∨ ∃ i j, i < (P.defs[U.d]!).body.params.size ∧
      j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok a := by
  refine DescAgree.run_inv "gate location" (gateStep P g) (fun sc => WalkQ P g sc ∧ Under U sc)
    (fun b => ∀ a ∈ b.2.1, (U.off ≤ a ∧ a < g) ∨ ∃ i j, i < (P.defs[U.d]!).body.params.size ∧
      j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok a) ?_ ?_ n sc x ⟨hq, hu⟩ h
  · intro sc b ⟨⟨hg, hoff, _⟩, hu⟩ hb
    obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
    obtain ⟨-, -, hnodes⟩ := hL sc.d hd hpc
    unfold gateStep at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · rename_i k hk
      obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
          ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
        have := hnodes k hkK
        omega
      obtain ⟨hm1, -, -⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
      by_cases hpc' : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
      · simp only [hpc', ite_true] at hb
        split at hb
        · cases hb
        · rename_i ops' hops
          simp only [pure, Except.pure, Except.ok.injEq, Sum.inl.injEq] at hb
          subst hb
          intro a ha
          rcases operands_within P hroot hB hL U sc hg hu k hkK hpc' _ hm1 hops a ha with ⟨h1, h2⟩ | h1
          · exact Or.inl ⟨h1, by omega⟩
          · exact Or.inr h1
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure] at hb
        cases hb
    · simp [throw, throwThe, MonadExceptOf.throw] at hb
  · intro sc sc' ⟨⟨hg, hoff, _⟩, hu⟩ hb
    obtain ⟨hd, hpc⟩ := good_d P hroot hB sc hg
    obtain ⟨-, -, hnodes⟩ := hL sc.d hd hpc
    unfold gateStep at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · rename_i k hk
      obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
      have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
          ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
        have := hnodes k hkK
        omega
      obtain ⟨hm1, hm2, hm3⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
      by_cases hpc' : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
      · simp only [hpc', ite_true] at hb
        split at hb
        · cases hb
        · simp only [pure, Except.pure, Except.ok.injEq] at hb
          cases hb
      · simp only [hpc', Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at hb
        subst hb
        refine ⟨⟨good_child P hg hkK rfl (by simpa using hpc') hm1 rfl, ?_, ?_⟩, Or.inr hu⟩
        · simp only [Prog.child, Scope.off.eq_2]
          omega
        · simp only [Prog.child, Scope.off.eq_2, Scope.d.eq_2]
          omega
    · simp [throw, throwThe, MonadExceptOf.throw] at hb

omit hroot hB in
theorem gateStep_callee (sc : Scope) (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) {k m g : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hlo : sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g)
    (hhi : g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) {x : (ℕ × Array ℕ × Scope) ⊕ Scope}
    (h : gateStep P g sc = .ok x) :
    ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧
        ∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc) ∧
          operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops) ∨
      ((P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧ x = .inr (P.child sc k m)) := by
  have hmono := offsets_mono P sc.d (hL sc.d hd) hp
  obtain ⟨-, -, hnodes⟩ := hL sc.d hd hp
  have hstep := hnodes k hk
  have hle : (m + 1) * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤
      ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates :=
    Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at hle
  unfold gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k' hk'
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hkk : k' = k := by
      rcases Nat.lt_trichotomy k' k with hlt | heq | hgt
      · have := hmono (k' + 1) k (by omega) (by omega)
        omega
      · exact heq
      · have := hmono (k + 1) k' (by omega) (by omega)
        omega
    rw [hkk] at h
    have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
        ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
      omega
    obtain ⟨-, hm2, hm3⟩ := Walk.member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
    have e1 : (g - sc.off - (P.defs[sc.d]!).body.offsets[k]!) /
        (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates = m :=
      Nat.div_eq_of_lt_le (by omega) (by rw [Nat.add_mul, Nat.one_mul]; omega)
    have hM := (Nat.div_eq_of_lt_le hm2 (by rw [Nat.add_mul, Nat.one_mul]; omega)).symm.trans e1
    rcases Bool.eq_false_or_eq_true (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim with hc | hc
    · simp only [hc, ite_true] at h
      split at h
      · cases h
      · rename_i ops hops
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact Or.inl ⟨hc, ops, rfl,
          (congrArg (fun M => operands P sc ((P.defs[sc.d]!).body.nodes[k]!) M = Except.ok ops) hM).mp hops⟩
    · simp only [hc, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact Or.inr ⟨hc, congrArg (fun M => Sum.inr (P.child sc k M)) hM⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

omit hroot hB in
theorem gateStep_member (sc : Scope) (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) {k m g : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false)
    (hlo : sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g)
    (hhi : g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) {x : (ℕ × Array ℕ × Scope) ⊕ Scope}
    (h : gateStep P g sc = .ok x) : x = .inr (P.child sc k m) := by
  rcases gateStep_callee P hL sc hd hp hk hm hlo hhi h with ⟨hc', -⟩ | ⟨-, hx⟩
  · rw [hc] at hc'
    cases hc'
  · exact hx

omit hroot hB in
theorem gateStep_at_prim (sc : Scope) (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) {k m g : ℕ}
    (hk : k < (P.defs[sc.d]!).body.nodes.size) (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hlo : sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates ≤ g)
    (hhi : g < sc.off + (P.defs[sc.d]!).body.offsets[k]! +
      m * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates +
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) {x : (ℕ × Array ℕ × Scope) ⊕ Scope}
    (h : gateStep P g sc = .ok x) :
    ∃ ops, x = .inl (((P.defs[sc.d]!).body.nodes[k]!).fn, ops, sc) ∧
      operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops := by
  rcases gateStep_callee P hL sc hd hp hk hm hlo hhi h with ⟨-, hx⟩ | ⟨hc', -⟩
  · exact hx
  · rw [hc] at hc'
    cases hc'

theorem gate_within {k m : ℕ} (hk : k < (P.defs[P.root]!).body.nodes.size)
    (hm : m < ((P.defs[P.root]!).body.nodes[k]!).members)
    (hc : (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).prim = false) {g p : ℕ} {ops : Array ℕ} {sc' : Scope}
    (hlo : (P.child P.rootScope k m).off ≤ g)
    (hhi : g < (P.child P.rootScope k m).off + (P.defs[(P.child P.rootScope k m).d]!).gates)
    (h : P.gate g = .ok (p, ops, sc')) :
    ∀ a ∈ ops, ((P.child P.rootScope k m).off ≤ a ∧ a < g) ∨
      ∃ i j, i < (P.defs[(P.child P.rootScope k m).d]!).body.params.size ∧
        j < ((P.defs[(P.child P.rootScope k m).d]!).body.params[i]!).2.leaves ∧
        resolve P (P.child P.rootScope k m) ⟨true, i, j⟩ = .ok a := by
  have hgood : Good P (P.child P.rootScope k m) :=
    good_child P (parent := P.rootScope) (by show Good P (.top P.root 0 none); unfold Good; exact ⟨rfl, rfl, rfl⟩)
      hk rfl hc hm rfl
  unfold Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  obtain ⟨n, hn⟩ : ∃ n, Flock.Steps.FUEL = n + 1 :=
    ⟨Flock.Steps.FUEL - 1, by unfold Flock.Steps.FUEL; omega⟩
  rw [hn, Flock.Steps.run.eq_2] at h
  rcases hs : gateStep P g P.rootScope with e | x
  · simp [hs, bind, Except.bind] at h
  have hx := gateStep_member P hL P.rootScope hroot.1 hroot.2 hk hm hc hlo hhi hs
  subst hx
  simp only [hs, bind, Except.bind] at h
  exact walk_within P hroot hB hL (P.child P.rootScope k m) (g := g) n (P.child P.rootScope k m) (p, ops, sc')
    ⟨hgood, hlo, by omega⟩ (under_refl _) h

end

theorem ret_within_of (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (U : Scope) (hU : Good P U) (t : ℕ)
    (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨ ∃ i j, i < (P.defs[U.d]!).body.params.size ∧
      j < ((P.defs[U.d]!).body.params[i]!).2.leaves ∧ resolve P U ⟨true, i, j⟩ = .ok b := by
  obtain ⟨hr, hrp, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨-, hL, -, -⟩ := Walk.bodiesLaidOut_spec P hlay
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hB : ∀ sc, Good P sc → BodyOk P sc.d := by
    intro sc hg
    obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrp (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1
  exact ret_within P ⟨hr, hrp⟩ hB hL U hU t ht h

end InferenceOnlyCheck.Properties.SecurityProofs.Resolve
