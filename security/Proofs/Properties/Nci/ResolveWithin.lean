module

public import Proofs.Properties.Nci.BodiesOk

/-!
# Resolution from inside an activation stays inside it, or leaves through one of its parameters

`Resolve.Inv` bounds the gate a reference resolves to from above (`Resolve.resolve_lt`). This module bounds it from
below, by the activation it started in. `Under U sc`: the activation `sc` is `U` or lies inside it. One step of
resolution from an activation ends at one of its gates, moves down into a member of one of its nodes, or moves up
through a parameter, to its parent or to the member before it (`step_where`). So a walk that starts under `U` stays
under `U` until it reads a parameter of `U` itself, and the gate it ends at is one of `U`'s own or the gate a leaf of
one of `U`'s parameters resolves to (`run_within`).

Of a body's return that is `ret_within`, and of a primitive node's operands `operands_within`: every leaf an
activation returns, and every operand of a gate inside it, is a gate of the activation or a leaf of one of its
parameters. `ret_within_of` states the first on what a decoded Program with no order violation and bodies laid out
gives, with no hypothesis on its bodies.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program

/-- The activation `sc` is `U`, or lies inside it. -/
def Under (U : Scope) : Scope → Prop
  | .top d off st => U = .top d off st
  | .child parent d off k m => U = .child parent d off k m ∨ Under U parent

theorem under_refl : ∀ U : Scope, Under U U
  | .top _ _ _ => rfl
  | .child _ _ _ _ _ => Or.inl rfl

/-- An activation inside `U` starts no earlier than `U`. -/
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

/-- A walk of resolution that ends within its fuel is the resolution. -/
theorem resolve_of_run (P : Prog) {n : ℕ} (hn : n ≤ Flock.Steps.FUEL) {sc : Scope} {r : Ref} {b : ℕ}
    (h : Flock.Steps.run "scope resolution" (resolveStep P) n (sc, r) = .ok b) : resolve P sc r = .ok b :=
  Flock.Steps.run_mono hn h

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

omit hroot hL in
/-- **Where one step of resolution goes.** From a state that resolves below a bound, a step ends at a gate of its
activation, no earlier than the activation starts, and only from a node's reference; or it moves down from a node's
reference into a member of a node of the activation; or it moves up from a parameter's reference, to the activation's
parent or to another member of the node the activation is a member of. -/
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

/-- **A walk of resolution that starts under `U` ends in `U` or leaves through a parameter of `U`.** From a state under
`U` that resolves below `B`, the gate the walk ends at is no earlier than `U` starts and before `B`, or it is the gate a
leaf of one of `U`'s parameters resolves to. -/
theorem run_within (U : Scope) (B : ℕ) :
    ∀ (n : ℕ), n ≤ Flock.Steps.FUEL → ∀ (s : Scope × Ref) (b : ℕ), Inv P B s → Under U s.1 →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok b →
      (U.off ≤ b ∧ b < B) ∨ ∃ i j, resolve P U ⟨true, i, j⟩ = .ok b
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
          exact Or.inr ⟨i, j, resolve_of_run P hn h⟩
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

/-- **A leaf an activation returns is one of its gates or a leaf of one of its parameters.** -/
theorem ret_within (U : Scope) (hU : Good P U) (t : ℕ) (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨ ∃ i j, resolve P U ⟨true, i, j⟩ = .ok b :=
  run_within P hroot hB hL U _ Flock.Steps.FUEL (Nat.le_refl _) _ b
    (down_ok P hroot hB hL _ U hU (Nat.le_refl _) t ht) (under_refl U) h

/-- **An operand of a gate inside an activation is an earlier gate of the activation or a leaf of one of its
parameters.** Node `k` of an activation `sc` under `U`, calling a primitive: every operand `operands` gives its member
`m` is no earlier than `U` starts and before the node, or the gate a leaf of one of `U`'s parameters resolves to. -/
theorem operands_within (U : Scope) (sc : Scope) (hg : Good P sc) (hu : Under U sc) (k : ℕ)
    (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hcp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true) (m : ℕ)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) {ops : Array ℕ}
    (h : operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok ops) :
    ∀ a ∈ ops, (U.off ≤ a ∧ a < sc.off + (P.defs[sc.d]!).body.offsets[k]!) ∨
      ∃ i j, resolve P U ⟨true, i, j⟩ = .ok a := by
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

end

/-- **`ret_within`, on a Program read**: one that decodes, with no order violation and its bodies laid out
(`Desc.bodiesLaidOut`, which `Desc.walkable` holds). -/
theorem ret_within_of (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (U : Scope) (hU : Good P U) (t : ℕ)
    (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨ ∃ i j, resolve P U ⟨true, i, j⟩ = .ok b := by
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

end NciCheck.Properties.SecurityProofs.Resolve
