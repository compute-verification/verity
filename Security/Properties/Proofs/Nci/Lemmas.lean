import Properties.Nci
import Mathlib.Tactic

/-!
# Lemmas about `NciCheck`'s functions

Each step of the check, characterized: the folds over a range in `Except`, `gateOf` and `circuitOf`, `thetaStep`,
`ownerOf` and `divisionOf`, `tally` and `ius`, `failure`, `parse` and `evaluate`.
-/

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySeqFocus false

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)

/-! ## Folds over a range in `Except` -/

theorem foldlM_range_succ {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (n : ℕ) :
    (List.range (n + 1)).foldlM f b = ((List.range n).foldlM f b >>= fun c => f c n) := by
  rw [List.range_succ, List.foldlM_append]
  simp

theorem foldlM_range_error {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (k : ε) : ∀ n,
    (List.range n).foldlM f b = .error k ↔ ∃ i < n, ∃ c, (List.range i).foldlM f b = .ok c ∧ f c i = .error k
  | 0 => by simp [pure, Except.pure]
  | n + 1 => by
    have ih := foldlM_range_error f b k n
    rw [foldlM_range_succ]
    rcases h : (List.range n).foldlM f b with e | c'
    · simp only [bind, Except.bind]
      constructor
      · intro he
        cases he
        obtain ⟨i, hi, c, hc, hf⟩ := ih.1 h
        exact ⟨i, by omega, c, hc, hf⟩
      · rintro ⟨i, hi, c, hc, hf⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
        · rw [ih.2 ⟨i, hi, c, hc, hf⟩] at h
          cases h
          rfl
        · rw [hc] at h
          cases h
    · simp only [bind, Except.bind]
      constructor
      · intro hf
        exact ⟨n, by omega, c', h, hf⟩
      · rintro ⟨i, hi, c, hc, hf⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
        · rw [ih.2 ⟨i, hi, c, hc, hf⟩] at h
          cases h
        · rw [hc] at h
          cases h
          exact hf

theorem foldlM_range_ok {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (Q : ℕ → β → Prop) (h0 : Q 0 b)
    (hs : ∀ n c c', Q n c → f c n = .ok c' → Q (n + 1) c') : ∀ n c, (List.range n).foldlM f b = .ok c → Q n c
  | 0, c, h => by
    simp only [List.range_zero, List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    exact h ▸ h0
  | n + 1, c, h => by
    rw [foldlM_range_succ] at h
    rcases hn : (List.range n).foldlM f b with e | c'
    · rw [hn] at h
      cases h
    · rw [hn] at h
      exact hs n c' c (foldlM_range_ok f b Q h0 hs n c' hn) h

/-- A fold over a range succeeds exactly when no step fails. -/
theorem foldlM_range_isOk {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (n : ℕ) :
    (∃ c, (List.range n).foldlM f b = .ok c) ↔ ¬ ∃ k, (List.range n).foldlM f b = .error k := by
  rcases (List.range n).foldlM f b with e | c <;> simp

/-! ## A gate -/

@[simp] theorem error_eq_error_iff {α : Type} (s k : String) : (Except.error s : Except String α) = .error k ↔ k = s :=
  ⟨fun h => by cases h; rfl, fun h => h ▸ rfl⟩

theorem any_ge_iff (ops : Array ℕ) (g : ℕ) : (ops.any fun x => decide (x ≥ g)) = true ↔ ∃ a ∈ ops, g ≤ a := by
  rw [Array.any_eq_true]
  constructor
  · rintro ⟨i, hi, h⟩
    exact ⟨ops[i], Array.getElem_mem hi, by simpa using h⟩
  · rintro ⟨a, ha, h⟩
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ha
    exact ⟨i, hi, by simpa using h⟩

theorem gateOf_ok (P : Prog) (g : ℕ) (x : ℕ × Bool × Array ℕ) :
    gateOf P g = .ok x ↔ ∃ p ops w, Located P g p ops w ∧ (∀ a ∈ ops, a < g) ∧
      x = (w, Flock.Program.isInputId (P.defs[p]!).key, ops) := by
  unfold gateOf Located
  rcases h : P.gate g with e | ⟨p, ops, sc⟩
  · simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
  · simp only [bind, Except.bind, pure, Except.pure]
    rcases hr : (P.defs[p]!).ret with w | _ | _ | _
    · have ht := any_ge_iff ops g
      have hlt : (∀ a ∈ ops, a < g) ↔ ¬ ∃ a ∈ ops, g ≤ a := by
        constructor
        · rintro h ⟨a, ha, hga⟩
          exact absurd (h a ha) (by omega)
        · intro h a ha
          by_contra hc
          exact h ⟨a, ha, by omega⟩
      constructor
      · intro hx
        split_ifs at hx with h1 h2
        simp only [Except.ok.injEq] at hx
        refine ⟨p, ops, w, ⟨⟨sc, rfl⟩, hr, ?_⟩, hlt.2 fun hc => h2 (ht.2 hc), hx.symm⟩
        intro hk
        simpa [hk, Array.isEmpty_iff] using h1
      · rintro ⟨p', ops', w', ⟨⟨sc', hs⟩, hr', hin⟩, hlt', rfl⟩
        simp only [Except.ok.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl, rfl⟩ := hs
        rw [hr, Flock.Program.Ty.value.injEq] at hr'
        subst hr'
        have h1 : (Flock.Program.isInputId (P.defs[p]!).key && !ops.isEmpty) = false := by
          cases hk : Flock.Program.isInputId (P.defs[p]!).key
          · rfl
          · simp [hin hk]
        have h2 : (ops.any fun x => decide (x ≥ g)) = false := by
          cases hc : ops.any fun x => decide (x ≥ g)
          · rfl
          · exact absurd (ht.1 hc) (hlt.1 hlt')
        simp [h1, h2]
    all_goals simp [throw, throwThe, MonadExceptOf.throw, hr]


theorem gateOk_iff (P : Prog) (g : ℕ) : GateOk P g ↔ ∃ x, gateOf P g = .ok x := by
  simp only [gateOf_ok, GateOk]
  constructor
  · rintro ⟨p, ops, w, hl, hlt⟩
    exact ⟨_, p, ops, w, hl, hlt, rfl⟩
  · rintro ⟨_, p, ops, w, hl, hlt, rfl⟩
    exact ⟨p, ops, w, hl, hlt⟩

theorem gateOf_error (P : Prog) (g : ℕ) (k : String) : gateOf P g = .error k ↔ GateFault P g k := by
  unfold gateOf GateFault Located
  rcases h : P.gate g with e | ⟨p, ops, sc⟩
  · simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
    exact ⟨Eq.symm, Eq.symm⟩
  · simp only [bind, Except.bind, pure, Except.pure]
    have ht := any_ge_iff ops g
    rcases hr : (P.defs[p]!).ret with w | _ | _ | _
    · simp only [Except.ok.injEq, Prod.mk.injEq]
      by_cases h1 : (Flock.Program.isInputId (P.defs[p]!).key && !ops.isEmpty) = true
      · have hn : ¬ (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[]) := by
          simp only [Bool.and_eq_true, Bool.not_eq_true'] at h1
          intro hk
          exact absurd (hk h1.1) (by simpa [Array.isEmpty_iff] using h1.2)
        simp [h1, throw, throwThe, MonadExceptOf.throw, hn]
        exact ⟨Eq.symm, Eq.symm⟩
      · have hy : Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[] := by
          intro hk
          simpa [hk, Array.isEmpty_iff] using h1
        by_cases h2 : (ops.any fun x => decide (x ≥ g)) = true
        · simp [h1, h2, throw, throwThe, MonadExceptOf.throw, ht.1 h2]
          constructor
          · rintro rfl
            exact Or.inr ⟨rfl, ⟨w, hr⟩, hy⟩
          · rintro (⟨rfl, hx⟩ | ⟨rfl, _⟩)
            · exact absurd (hy (hx w hr).1) (hx w hr).2
            · rfl
        · have : ¬ ∃ a ∈ ops, g ≤ a := fun hc => h2 (ht.2 hc)
          simp only [h1, h2, Bool.false_eq_true, ite_false, reduceCtorEq, false_iff]
          simp_all
    all_goals simp [throw, throwThe, MonadExceptOf.throw, hr]; exact ⟨Eq.symm, Eq.symm⟩

/-! ## The circuit -/

theorem getElem!_push_lt {α : Type} [Inhabited α] (xs : Array α) (x : α) {i : ℕ} (h : i < xs.size) :
    (xs.push x)[i]! = xs[i]! := by
  rw [getElem!_pos (xs.push x) i (by simp; omega), getElem!_pos xs i h, Array.getElem_push_lt]

theorem getElem!_push_size {α : Type} [Inhabited α] (xs : Array α) (x : α) : (xs.push x)[xs.size]! = x := by
  rw [getElem!_pos (xs.push x) xs.size (by simp), Array.getElem_push_eq]

theorem circuitOf_ok (P : Prog) (c : Circuit) (h : circuitOf P = .ok c) :
    c.N = gates P ∧ c.isInput.size = gates P ∧ c.reads.size = gates P ∧
      ∀ g < gates P, gateOf P g = .ok (c.w[g]!, c.isInput[g]!, c.reads[g]!) := by
  refine foldlM_range_ok _ _ (fun n (c : Circuit) => c.w.size = n ∧ c.isInput.size = n ∧ c.reads.size = n ∧
    ∀ g < n, gateOf P g = .ok (c.w[g]!, c.isInput[g]!, c.reads[g]!)) ⟨rfl, rfl, rfl, by simp⟩ ?_ _ c h
  rintro n c c' ⟨hw, hi, hr, hg⟩ hs
  rcases hx : gateOf P n with e | ⟨w, inp, ops⟩
  · simp [hx, Functor.map, Except.map] at hs
  · simp only [hx, Functor.map, Except.map, Except.ok.injEq] at hs
    subst hs
    refine ⟨by simp [Circuit.push, hw], by simp [Circuit.push, hi], by simp [Circuit.push, hr], fun g hg' => ?_⟩
    simp only [Circuit.push]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hg' with hlt | rfl
    · rw [getElem!_push_lt _ _ (by omega), getElem!_push_lt _ _ (by omega), getElem!_push_lt _ _ (by omega)]
      exact hg g hlt
    · have e1 : (c.w.push w)[c.w.size]! = w := getElem!_push_size _ _
      have e2 : (c.isInput.push inp)[c.isInput.size]! = inp := getElem!_push_size _ _
      have e3 : (c.reads.push ops)[c.reads.size]! = ops := getElem!_push_size _ _
      rw [hw] at e1
      rw [hi] at e2
      rw [hr] at e3
      rw [e1, e2, e3]
      exact hx

/-- A fold over a range succeeds exactly when every step does, each step's success read off the state through an
invariant `Q`. -/
theorem foldlM_range_isOk_inv {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (Q : ℕ → β → Prop) (h0 : Q 0 b)
    (hs : ∀ n c c', Q n c → f c n = .ok c' → Q (n + 1) c') (ok : ℕ → Prop)
    (hf : ∀ c i, Q i c → ((∃ c', f c i = .ok c') ↔ ok i)) :
    ∀ n, (∃ c, (List.range n).foldlM f b = .ok c) ↔ ∀ g < n, ok g
  | 0 => by simp [pure, Except.pure]
  | n + 1 => by
    have ih := foldlM_range_isOk_inv f b Q h0 hs ok hf n
    rw [foldlM_range_succ]
    rcases h : (List.range n).foldlM f b with e | c
    · simp only [bind, Except.bind, reduceCtorEq, exists_false, false_iff]
      have : ¬ ∀ g < n, ok g := fun hc => by simpa [h] using ih.2 hc
      exact fun hall => this fun g hg => hall g (by omega)
    · simp only [bind, Except.bind]
      rw [hf c n (foldlM_range_ok f b Q h0 hs n c h)]
      have := ih.1 ⟨c, h⟩
      constructor
      · intro hn g hg
        rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
        · exact this g hg
        · exact hn
      · intro hall
        exact hall n (by omega)

/-- A fold over a range fails with `k` exactly at its first failing step, through an invariant `Q`. -/
theorem foldlM_range_error_inv {β : Type} (f : β → ℕ → Except String β) (b : β) (Q : ℕ → β → Prop) (h0 : Q 0 b)
    (hs : ∀ n c c', Q n c → f c n = .ok c' → Q (n + 1) c') (ok : ℕ → Prop) (fault : ℕ → String → Prop)
    (hf : ∀ c i, Q i c → ((∃ c', f c i = .ok c') ↔ ok i))
    (he : ∀ c i k, Q i c → (f c i = .error k ↔ fault i k)) (n : ℕ) (k : String) :
    (List.range n).foldlM f b = .error k ↔ FirstFault ok fault n k := by
  rw [foldlM_range_error, FirstFault]
  constructor
  · rintro ⟨i, hi, c, hc, hk⟩
    exact ⟨i, hi, (foldlM_range_isOk_inv f b Q h0 hs ok hf i).1 ⟨c, hc⟩,
      (he c i k (foldlM_range_ok f b Q h0 hs i c hc)).1 hk⟩
  · rintro ⟨i, hi, hok, hk⟩
    obtain ⟨c, hc⟩ := (foldlM_range_isOk_inv f b Q h0 hs ok hf i).2 hok
    exact ⟨i, hi, c, hc, (he c i k (foldlM_range_ok f b Q h0 hs i c hc)).2 hk⟩

theorem foldlM_range_isOk_indep {β ε : Type} (f : β → ℕ → Except ε β) (b : β) (ok : ℕ → Prop)
    (hf : ∀ c i, (∃ c', f c i = .ok c') ↔ ok i) : ∀ n, (∃ c, (List.range n).foldlM f b = .ok c) ↔ ∀ g < n, ok g :=
  foldlM_range_isOk_inv f b (fun _ _ => True) trivial (fun _ _ _ _ _ => trivial) ok (fun c i _ => hf c i)

theorem foldlM_range_error_indep {β : Type} (f : β → ℕ → Except String β) (b : β) (ok : ℕ → Prop)
    (fault : ℕ → String → Prop) (hf : ∀ c i, (∃ c', f c i = .ok c') ↔ ok i)
    (he : ∀ c i k, f c i = .error k ↔ fault i k) (n : ℕ) (k : String) :
    (List.range n).foldlM f b = .error k ↔ FirstFault ok fault n k :=
  foldlM_range_error_inv f b (fun _ _ => True) trivial (fun _ _ _ _ _ => trivial) ok fault (fun c i _ => hf c i)
    (fun c i k _ => he c i k) n k

theorem push_map_ok (P : Prog) (c : Circuit) (g : ℕ) :
    (∃ c', (c.push <$> gateOf P g) = .ok c') ↔ GateOk P g := by
  rw [gateOk_iff]
  rcases gateOf P g with e | x <;> simp [Functor.map, Except.map]

theorem push_map_error (P : Prog) (c : Circuit) (g : ℕ) (k : String) :
    (c.push <$> gateOf P g) = .error k ↔ GateFault P g k := by
  rw [← gateOf_error]
  rcases gateOf P g with e | x <;> simp [Functor.map, Except.map]

theorem circuitOf_error (P : Prog) (k : String) :
    circuitOf P = .error k ↔ FirstFault (GateOk P) (GateFault P) (gates P) k :=
  foldlM_range_error_indep _ _ _ _ (push_map_ok P) (push_map_error P) _ k

theorem circuitOf_isOk (P : Prog) : (∃ c, circuitOf P = .ok c) ↔ ∀ g < gates P, GateOk P g :=
  foldlM_range_isOk_indep _ _ _ (push_map_ok P) _

/-- What `circuitOf` read of gate `g`, from the decoder. -/
theorem circuitOf_gate {P : Prog} {c : Circuit} (h : circuitOf P = .ok c) {g : ℕ} (hg : g < gates P) :
    ∃ p sc, P.gate g = .ok (p, c.reads[g]!, sc) ∧ (P.defs[p]!).ret = .value c.w[g]! ∧
      c.isInput[g]! = Flock.Program.isInputId (P.defs[p]!).key ∧
      (c.isInput[g]! = true → c.reads[g]! = #[]) ∧ ∀ a ∈ c.reads[g]!, a < g := by
  obtain ⟨p, ops, w, ⟨⟨sc, hs⟩, hr, hin⟩, hlt, hx⟩ := (gateOf_ok P g _).1 ((circuitOf_ok P c h).2.2.2 g hg)
  simp only [Prod.mk.injEq] at hx
  obtain ⟨hw, hi, ho⟩ := hx
  refine ⟨p, sc, ho ▸ hs, hw ▸ hr, hi, fun h' => ho ▸ hin (hi ▸ h'), ho ▸ hlt⟩

theorem circuitOf_isInput {P : Prog} {c : Circuit} (h : circuitOf P = .ok c) {g : ℕ} (hg : g < gates P) :
    c.isInput[g]! = true ↔ IsInput P g := by
  obtain ⟨p, sc, hs, -, hi, -, -⟩ := circuitOf_gate h hg
  constructor
  · intro h'
    exact ⟨p, _, sc, hs, hi ▸ h'⟩
  · rintro ⟨p', ops', sc', hs', h'⟩
    rw [hs] at hs'
    simp only [Except.ok.injEq, Prod.mk.injEq] at hs'
    obtain ⟨rfl, -, -⟩ := hs'
    exact hi ▸ h'

/-! ## θ -/

theorem natOf?_str (s : String) : natOf? (.str s) = none := rfl

theorem thetaStep_ok (c : Circuit) (s s' : Std.HashSet ℕ) (t : Json) :
    thetaStep c s t = .ok s' ↔ ∃ g, natOf? t = some g ∧ g < c.N ∧ c.isInput[g]! = true ∧ g ∉ s ∧ s' = s.insert g := by
  unfold thetaStep
  rcases hn : natOf? t with _ | g
  · simp [throw, throwThe, MonadExceptOf.throw]
  · simp only [bind, Except.bind, pure, Except.pure, Option.some.injEq, exists_eq_left']
    rcases Nat.lt_or_ge g c.N with h1 | h1 <;>
      [have h1' : ¬ c.N ≤ g := Nat.not_le.2 h1; have h1' : ¬ g < c.N := Nat.not_lt.2 h1] <;>
      by_cases h2 : c.isInput[g]! = true <;> by_cases h3 : g ∈ s <;>
      simp [h1, h1', h2, h3, Std.HashSet.contains_iff_mem, throw, throwThe, MonadExceptOf.throw, eq_comm (a := s')]

theorem thetaStep_error (c : Circuit) (s : Std.HashSet ℕ) (t : Json) (k : String) :
    thetaStep c s t = .error k ↔
      (k = "theta-unknown" ∧ ∀ g, natOf? t = some g → c.N ≤ g) ∨
      (k = "theta-not-input" ∧ ∃ g, natOf? t = some g ∧ g < c.N ∧ c.isInput[g]! = false) ∨
      (k = "theta-duplicate" ∧ ∃ g, natOf? t = some g ∧ g < c.N ∧ c.isInput[g]! = true ∧ g ∈ s) := by
  unfold thetaStep
  rcases hn : natOf? t with _ | g
  · simp [throw, throwThe, MonadExceptOf.throw]
    exact ⟨Eq.symm, Eq.symm⟩
  · simp only [bind, Except.bind, pure, Except.pure, Option.some.injEq, exists_eq_left', forall_eq']
    rcases Nat.lt_or_ge g c.N with h1 | h1 <;>
      [have h1' : ¬ c.N ≤ g := Nat.not_le.2 h1; have h1' : ¬ g < c.N := Nat.not_lt.2 h1] <;>
      by_cases h2 : c.isInput[g]! = true <;> by_cases h3 : g ∈ s <;>
      simp [h1, h1', h2, h3, Std.HashSet.contains_iff_mem, throw, throwThe, MonadExceptOf.throw] <;>
      exact ⟨Eq.symm, Eq.symm⟩

/-- The set after entry `n`: the gates the entries up to `n` name. -/
theorem thetaStep_mem (c : Circuit) (theta : Array Json) (n : ℕ) (s s' : Std.HashSet ℕ)
    (hQ : ∀ g, g ∈ s ↔ ∃ t < n, natOf? theta[t]! = some g) (hs : thetaStep c s theta[n]! = .ok s') :
    ∀ g, g ∈ s' ↔ ∃ t < n + 1, natOf? theta[t]! = some g := by
  intro g
  obtain ⟨g0, hg0, -, -, -, rfl⟩ := (thetaStep_ok c s s' _).1 hs
  rw [Std.HashSet.mem_insert, hQ g]
  simp only [beq_iff_eq]
  constructor
  · rintro (rfl | ⟨t, ht, h⟩)
    · exact ⟨n, by omega, hg0⟩
    · exact ⟨t, by omega, h⟩
  · rintro ⟨t, ht, h⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 ht with ht | rfl
    · exact Or.inr ⟨t, ht, h⟩
    · rw [hg0] at h
      exact Or.inl (Option.some.inj h)

/-- The entries before `n` that name a gate, as a set. -/
theorem thetaFold_mem (c : Circuit) (theta : Array Json) : ∀ n s,
    (List.range n).foldlM (fun s t => thetaStep c s theta[t]!) {} = .ok s →
      ∀ g, g ∈ s ↔ ∃ t < n, natOf? theta[t]! = some g :=
  foldlM_range_ok _ _ _ (by simp) (thetaStep_mem c theta)

theorem thetaStep_isOk {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta : Array Json) (t : ℕ)
    (s : Std.HashSet ℕ) (hs : ∀ g, g ∈ s ↔ ∃ t' < t, natOf? theta[t']! = some g) :
    (∃ s', thetaStep c s theta[t]! = .ok s') ↔ ThetaOk P theta t := by
  have hN := (circuitOf_ok P c hc).1
  simp only [thetaStep_ok, ThetaOk]
  constructor
  · rintro ⟨_, g, hg, hlt, hin, hns, -⟩
    refine ⟨g, hg, hN ▸ hlt, (circuitOf_isInput hc (hN ▸ hlt)).1 hin, fun t' ht' h => hns ((hs g).2 ⟨t', ht', h⟩)⟩
  · rintro ⟨g, hg, hlt, hin, hne⟩
    refine ⟨_, g, hg, hN ▸ hlt, (circuitOf_isInput hc hlt).2 hin, fun hm => ?_, rfl⟩
    obtain ⟨t', ht', h⟩ := (hs g).1 hm
    exact hne t' ht' h

theorem thetaStep_error' {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta : Array Json) (t : ℕ)
    (s : Std.HashSet ℕ) (hs : ∀ g, g ∈ s ↔ ∃ t' < t, natOf? theta[t']! = some g) (k : String) :
    thetaStep c s theta[t]! = .error k ↔ ThetaFault P theta t k := by
  have hN := (circuitOf_ok P c hc).1
  rw [thetaStep_error, ThetaFault, hN]
  have hb : ∀ g < gates P, (c.isInput[g]! = false ↔ ¬ IsInput P g) := fun g hg => by
    rw [← circuitOf_isInput hc hg]; simp
  constructor
  · rintro (h | ⟨rfl, g, hg, hlt, hin⟩ | ⟨rfl, g, hg, hlt, hin, hm⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl ⟨rfl, g, hg, hlt, (hb g hlt).1 hin⟩)
    · exact Or.inr (Or.inr ⟨rfl, g, hg, hlt, (circuitOf_isInput hc hlt).1 hin, (hs g).1 hm⟩)
  · rintro (h | ⟨rfl, g, hg, hlt, hin⟩ | ⟨rfl, g, hg, hlt, hin, hm⟩)
    · exact Or.inl h
    · exact Or.inr (Or.inl ⟨rfl, g, hg, hlt, (hb g hlt).2 hin⟩)
    · exact Or.inr (Or.inr ⟨rfl, g, hg, hlt, (circuitOf_isInput hc hlt).2 hin, (hs g).2 hm⟩)

theorem thetaFold_isOk {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta : Array Json) :
    (∃ s, (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {} = .ok s) ↔
      ∀ t < theta.size, ThetaOk P theta t :=
  foldlM_range_isOk_inv _ _ (fun n s => ∀ g, g ∈ s ↔ ∃ t < n, natOf? theta[t]! = some g) (by simp)
    (thetaStep_mem c theta) _ (fun s t hQ => thetaStep_isOk hc theta t s hQ) _

theorem thetaFold_error {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta : Array Json) (k : String) :
    (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {} = .error k ↔
      FirstFault (ThetaOk P theta) (ThetaFault P theta) theta.size k :=
  foldlM_range_error_inv _ _ (fun n s => ∀ g, g ∈ s ↔ ∃ t < n, natOf? theta[t]! = some g) (by simp)
    (thetaStep_mem c theta) _ _ (fun s t hQ => thetaStep_isOk hc theta t s hQ)
    (fun s t k hQ => thetaStep_error' hc theta t s hQ k) _ k

/-! ## The owners -/

theorem isInputTag_iff (j : Json) : isInputTag j = true ↔ j = .str "input" := by
  cases j <;> simp [isInputTag]

theorem ownerOf_ok (c : Circuit) (g : ℕ) (j : Json) (o : Option ℕ) :
    ownerOf c g j = .ok o ↔ o = natOf? j ∧
      ((j = .str "input" ∧ c.isInput[g]! = true) ∨ ((∃ u, natOf? j = some u) ∧ c.isInput[g]! = false)) := by
  unfold ownerOf
  by_cases ht : isInputTag j = true
  · obtain rfl := (isInputTag_iff j).1 ht
    cases hi : c.isInput[g]! <;> simp only [ht, natOf?_str, ite_true] <;>
      simp [hi, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] <;>
      exact ⟨Eq.symm, Eq.symm⟩
  · have hj : j ≠ .str "input" := fun h => ht ((isInputTag_iff j).2 h)
    rcases hn : natOf? j with _ | u <;> cases hi : c.isInput[g]! <;>
      simp only [ht, hn, hi, Bool.false_eq_true, ite_false] <;>
      simp [hj, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] <;>
      exact ⟨Eq.symm, Eq.symm⟩

theorem ownerOf_error (c : Circuit) (g : ℕ) (j : Json) (k : String) :
    ownerOf c g j = .error k ↔
      (k = "owner-entry" ∧ j ≠ .str "input" ∧ natOf? j = none) ∨
      (k = "input-in-iu" ∧ (∃ u, natOf? j = some u) ∧ c.isInput[g]! = true) ∨
      (k = "gate-in-no-iu" ∧ j = .str "input" ∧ c.isInput[g]! = false) := by
  unfold ownerOf
  by_cases ht : isInputTag j = true
  · obtain rfl := (isInputTag_iff j).1 ht
    cases hi : c.isInput[g]! <;> simp only [ht, natOf?_str, ite_true] <;>
      simp [hi, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] <;>
      exact ⟨Eq.symm, Eq.symm⟩
  · have hj : j ≠ .str "input" := fun h => ht ((isInputTag_iff j).2 h)
    rcases hn : natOf? j with _ | u <;> cases hi : c.isInput[g]! <;>
      simp only [ht, hn, hi, Bool.false_eq_true, ite_false] <;>
      simp [hj, pure, Except.pure, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] <;>
      exact ⟨Eq.symm, Eq.symm⟩

theorem FirstFault_congr {ok ok' : ℕ → Prop} {fault fault' : ℕ → String → Prop} {n : ℕ} {k : String}
    (hok : ∀ i < n, ok i ↔ ok' i) (hf : ∀ i < n, fault i k ↔ fault' i k) :
    FirstFault ok fault n k ↔ FirstFault ok' fault' n k := by
  unfold FirstFault
  constructor
  · rintro ⟨i, hi, ho, hk⟩
    exact ⟨i, hi, fun i' hi' => (hok i' (by omega)).1 (ho i' hi'), (hf i hi).1 hk⟩
  · rintro ⟨i, hi, ho, hk⟩
    exact ⟨i, hi, fun i' hi' => (hok i' (by omega)).2 (ho i' hi'), (hf i hi).2 hk⟩

/-- An owner step, as the gate's own condition read off the circuit. -/
def OwnerOkC (c : Circuit) (owner : Array Json) (g : ℕ) : Prop :=
  (owner[g]! = .str "input" ∧ c.isInput[g]! = true) ∨ ((∃ u, natOf? owner[g]! = some u) ∧ c.isInput[g]! = false)

def OwnerFaultC (c : Circuit) (owner : Array Json) (g : ℕ) (k : String) : Prop :=
  (k = "owner-entry" ∧ owner[g]! ≠ .str "input" ∧ natOf? owner[g]! = none) ∨
    (k = "input-in-iu" ∧ (∃ u, natOf? owner[g]! = some u) ∧ c.isInput[g]! = true) ∨
    (k = "gate-in-no-iu" ∧ owner[g]! = .str "input" ∧ c.isInput[g]! = false)

theorem ownerOkC_iff {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (owner : Array Json) {g : ℕ}
    (hg : g < gates P) : OwnerOkC c owner g ↔ OwnerOk P owner g := by
  have h := circuitOf_isInput hc hg
  unfold OwnerOkC OwnerOk
  rw [← h]
  simp

theorem ownerFaultC_iff {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (owner : Array Json) {g : ℕ}
    (hg : g < gates P) (k : String) : OwnerFaultC c owner g k ↔ OwnerFault P owner g k := by
  have h := circuitOf_isInput hc hg
  unfold OwnerFaultC OwnerFault
  rw [← h]
  simp

theorem push_ownerOf_ok (c : Circuit) (owner : Array Json) (own : Array (Option ℕ)) (g : ℕ) :
    (∃ own', (own.push <$> ownerOf c g owner[g]!) = .ok own') ↔ OwnerOkC c owner g := by
  rcases h : ownerOf c g owner[g]! with e | o
  · simp only [Functor.map, Except.map, reduceCtorEq, exists_false, false_iff]
    intro hc
    rcases hc with ⟨hj, hi⟩ | ⟨⟨u, hu⟩, hi⟩
    · have := (ownerOf_ok c g _ none).2 ⟨by rw [hj]; rfl, Or.inl ⟨hj, hi⟩⟩
      rw [h] at this; cases this
    · have := (ownerOf_ok c g _ (some u)).2 ⟨hu.symm, Or.inr ⟨⟨u, hu⟩, hi⟩⟩
      rw [h] at this; cases this
  · simp only [Functor.map, Except.map, Except.ok.injEq, exists_eq', true_iff]
    exact ((ownerOf_ok c g _ o).1 h).2

theorem push_ownerOf_error (c : Circuit) (owner : Array Json) (own : Array (Option ℕ)) (g : ℕ) (k : String) :
    (own.push <$> ownerOf c g owner[g]!) = .error k ↔ OwnerFaultC c owner g k := by
  rw [OwnerFaultC, ← ownerOf_error]
  rcases ownerOf c g owner[g]! with e | o <;> simp [Functor.map, Except.map]

/-- The owner fold: one entry per gate so far, gate `g`'s the IU its owner names. -/
theorem ownerFold_ok (c : Circuit) (owner : Array Json) : ∀ n own,
    (List.range n).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[] = .ok own →
      own.size = n ∧ ∀ g < n, own[g]! = natOf? owner[g]! := by
  refine foldlM_range_ok _ _ _ ⟨rfl, by simp⟩ ?_
  rintro n own own' ⟨hs, hg⟩ h
  rcases hx : ownerOf c n owner[n]! with e | o
  · simp [hx, Functor.map, Except.map] at h
  · simp only [hx, Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    refine ⟨by simp [hs], fun g hg' => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hg' with hlt | rfl
    · rw [getElem!_push_lt _ _ (by omega)]
      exact hg g hlt
    · have e1 := getElem!_push_size own o
      rw [hs] at e1
      rw [e1]
      exact ((ownerOf_ok c _ _ o).1 hx).1

theorem ownerFold_isOk {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (owner : Array Json) :
    (∃ own, (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[] = .ok own) ↔
      ∀ g < gates P, OwnerOk P owner g := by
  rw [foldlM_range_isOk_indep _ _ _ (push_ownerOf_ok c owner) c.N, (circuitOf_ok P c hc).1]
  exact forall₂_congr fun g hg => ownerOkC_iff hc owner hg

theorem ownerFold_error {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (owner : Array Json) (k : String) :
    (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[] = .error k ↔
      FirstFault (OwnerOk P owner) (OwnerFault P owner) (gates P) k := by
  rw [foldlM_range_error_indep _ _ _ _ (push_ownerOf_ok c owner) (push_ownerOf_error c owner) c.N k,
    (circuitOf_ok P c hc).1]
  exact FirstFault_congr (fun g hg => ownerOkC_iff hc owner hg) (fun g hg => ownerFaultC_iff hc owner hg k)

/-! ## The IUs -/

theorem getElem!_fin {α : Type} [Inhabited α] (xs : Array α) (i : Fin xs.size) : xs[i] = xs[i.val]! :=
  (getElem!_pos xs i.val i.isLt).symm

/-- `iuCount` bounds every IU an owner names, and is `0` or one more than an IU an owner names. -/
theorem iuCount_spec (own : Array (Option ℕ)) :
    (∀ g < own.size, ∀ u, own[g]! = some u → u < iuCount own) ∧
      (iuCount own = 0 ∨ ∃ g < own.size, own[g]! = some (iuCount own - 1)) := by
  unfold iuCount
  refine Array.foldl_induction (as := own)
    (fun i m => (∀ g < i, ∀ u, own[g]! = some u → u < m) ∧ (m = 0 ∨ ∃ g < i, own[g]! = some (m - 1)))
    ⟨by simp, Or.inl rfl⟩ ?_
  rintro ⟨i, hi⟩ m ⟨hb, hm⟩
  simp only [getElem!_fin]
  dsimp only at hb hm ⊢
  rcases ho : own[i]! with _ | u <;> dsimp only
  · refine ⟨fun g hg u' hu => ?_, ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
      · exact hb g hg u' hu
      · rw [ho] at hu; cases hu
    · rcases hm with hm | ⟨g, hg, h⟩
      · exact Or.inl hm
      · exact Or.inr ⟨g, by omega, h⟩
  · refine ⟨fun g hg u' hu => ?_, Or.inr ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
      · exact lt_of_lt_of_le (hb g hg u' hu) (le_max_left _ _)
      · rw [ho] at hu; cases hu; simp
    · rcases le_total m (u + 1) with h | h
      · rw [max_eq_right h]
        exact ⟨i, by omega, by simpa using ho⟩
      · rw [max_eq_left h]
        rcases hm with rfl | ⟨g, hg, hg'⟩
        · exact ⟨i, by omega, by simp at h⟩
        · exact ⟨g, by omega, hg'⟩

theorem usedOf_mem (own : Array (Option ℕ)) (u : ℕ) : u ∈ usedOf own ↔ ∃ g < own.size, own[g]! = some u := by
  unfold usedOf
  refine Array.foldl_induction (as := own) (fun i s => ∀ u, u ∈ s ↔ ∃ g < i, own[g]! = some u)
    (by simp) ?_ u
  rintro ⟨i, hi⟩ s hs u
  simp only [getElem!_fin]
  dsimp only at hs ⊢
  rcases ho : own[i]! with _ | v <;> dsimp only
  · rw [hs u]
    constructor
    · rintro ⟨g, hg, h⟩; exact ⟨g, by omega, h⟩
    · rintro ⟨g, hg, h⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
      · exact ⟨g, hg, h⟩
      · rw [ho] at h; cases h
  · rw [Std.HashSet.mem_insert, hs u, beq_iff_eq]
    constructor
    · rintro (rfl | ⟨g, hg, h⟩)
      · exact ⟨i, by omega, ho⟩
      · exact ⟨g, by omega, h⟩
    · rintro ⟨g, hg, h⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
      · exact Or.inr ⟨g, hg, h⟩
      · rw [ho] at h; exact Or.inl (Option.some.inj h)

/-- The IUs are `0 … iuCount - 1`, each named by an owner and no more of them than gates, exactly when no IU below one
an owner names is empty. -/
theorem iuCheck_iff (owner : Array Json) (own : Array (Option ℕ)) (hs : own.size = owner.size)
    (ho : ∀ g < owner.size, own[g]! = natOf? owner[g]!) :
    (iuCount own ≤ owner.size ∧ ∀ u < iuCount own, u ∈ usedOf own) ↔ ¬ EmptyIU owner := by
  obtain ⟨hb, hm⟩ := iuCount_spec own
  have hu : ∀ u, u ∈ usedOf own ↔ ∃ g < owner.size, natOf? owner[g]! = some u := fun u => by
    rw [usedOf_mem, hs]
    exact exists_congr fun g => and_congr_right fun hg => by rw [ho g hg]
  constructor
  · rintro ⟨-, hall⟩ ⟨g, u, hg, hgu, u', hu', hno⟩
    have := hb g (hs ▸ hg) u (by rw [ho g hg]; exact hgu)
    obtain ⟨g', hg', h'⟩ := (hu u').1 (hall u' (by omega))
    exact hno g' hg' h'
  · intro hne
    have hall : ∀ u < iuCount own, u ∈ usedOf own := by
      intro u hlt
      rcases hm with h0 | ⟨g0, hg0, h0⟩
      · omega
      rw [hs] at hg0
      rw [ho g0 hg0] at h0
      rcases Nat.lt_or_ge u (iuCount own - 1) with h | h
      · by_contra hc
        exact hne ⟨g0, _, hg0, h0, u, h, fun g' hg' h' => hc ((hu u).2 ⟨g', hg', h'⟩)⟩
      · have : u = iuCount own - 1 := by omega
        subst this
        exact (hu _).2 ⟨g0, hg0, h0⟩
    refine ⟨?_, hall⟩
    have hc : ∀ u : Fin (iuCount own), ∃ g < owner.size, natOf? owner[g]! = some u.val :=
      fun u => (hu u.val).1 (hall u.val u.isLt)
    let f : Fin (iuCount own) → Fin owner.size := fun u => ⟨Classical.choose (hc u), (Classical.choose_spec (hc u)).1⟩
    have hf : Function.Injective f := by
      intro u v huv
      have h1 := (Classical.choose_spec (hc u)).2
      have h2 := (Classical.choose_spec (hc v)).2
      have : Classical.choose (hc u) = Classical.choose (hc v) := congrArg Fin.val huv
      rw [this, h2] at h1
      exact Fin.ext (Option.some.inj h1).symm
    simpa using Fintype.card_le_of_injective f hf

/-! ## The division -/

/-- θ's fold. -/
abbrev thetaFold (c : Circuit) (theta : Array Json) : Except String (Std.HashSet ℕ) :=
  (List.range theta.size).foldlM (fun s t => thetaStep c s theta[t]!) {}

/-- The owners' fold. -/
abbrev ownerFold (c : Circuit) (owner : Array Json) : Except String (Array (Option ℕ)) :=
  (List.range c.N).foldlM (fun own g => own.push <$> ownerOf c g owner[g]!) #[]

open Classical in
theorem divisionOf_eq (c : Circuit) (theta owner : Array Json) :
    divisionOf c theta owner = match thetaFold c theta with
      | .error e => .error e
      | .ok mask => if owner.size ≠ c.N then .error "owner-length" else match ownerFold c owner with
        | .error e => .error e
        | .ok own => if iuCount own ≤ c.N ∧ ∀ u < iuCount own, u ∈ usedOf own
            then .ok { theta := mask, owner := own, n := iuCount own } else .error "empty-iu" := by
  unfold divisionOf
  rcases h1 : thetaFold c theta with e | mask
  · simp only [h1]; rfl
  · simp only [h1]
    by_cases h2 : owner.size = c.N
    · rcases h3 : ownerFold c owner with e | own
      · simp [h2, h3, bind, Except.bind]
      · by_cases hA : c.N < iuCount own <;> by_cases hB : ∃ x < iuCount own, x ∉ usedOf own <;>
          simp [h2, h3, hA, hB, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
    · simp [h2, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-- What a checked division holds: θ as the set its entries name, and per gate the IU its owner names. -/
theorem divisionOf_ok {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {theta owner : Array Json} {d : Division}
    (h : divisionOf c theta owner = .ok d) :
    (∀ t < theta.size, ThetaOk P theta t) ∧ (∀ g, g ∈ d.theta ↔ ∃ t < theta.size, natOf? theta[t]! = some g) ∧
      owner.size = gates P ∧ (∀ g < gates P, OwnerOk P owner g) ∧ d.owner.size = gates P ∧
      (∀ g < gates P, d.owner[g]! = natOf? owner[g]!) ∧ d.n = iuCount d.owner ∧ ¬ EmptyIU owner := by
  have hN := (circuitOf_ok P c hc).1
  rw [divisionOf_eq] at h
  rcases h1 : thetaFold c theta with e | mask <;> rw [h1] at h
  · cases h
  dsimp only at h
  by_cases h2 : owner.size = c.N
  · rw [ite_eq_right (not_not.2 h2)] at h
    rcases h3 : ownerFold c owner with e | own <;> rw [h3] at h
    · cases h
    dsimp only at h
    obtain ⟨hs, ho⟩ := ownerFold_ok c owner c.N own h3
    split_ifs at h with h4
    cases h
    refine ⟨(thetaFold_isOk hc theta).1 ⟨mask, h1⟩, thetaFold_mem c theta _ mask h1, hN ▸ h2,
      (ownerFold_isOk hc owner).1 ⟨own, h3⟩, hN ▸ hs, hN ▸ ho, rfl, ?_⟩
    exact (iuCheck_iff owner own (by omega) (fun g hg => ho g (by omega))).1 (h2 ▸ h4)
  · rw [ite_eq_left h2] at h
    cases h

theorem divisionOf_isOk {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta owner : Array Json) :
    (∃ d, divisionOf c theta owner = .ok d) ↔ (∀ t < theta.size, ThetaOk P theta t) ∧ owner.size = gates P ∧
      (∀ g < gates P, OwnerOk P owner g) ∧ ¬ EmptyIU owner := by
  constructor
  · rintro ⟨d, h⟩
    obtain ⟨h1, -, h2, h3, -, -, -, h4⟩ := divisionOf_ok hc h
    exact ⟨h1, h2, h3, h4⟩
  · rintro ⟨h1, h2, h3, h4⟩
    have hN := (circuitOf_ok P c hc).1
    obtain ⟨mask, hm⟩ := (thetaFold_isOk hc theta).2 h1
    obtain ⟨own, ho⟩ := (ownerFold_isOk hc owner).2 h3
    obtain ⟨hs, ho'⟩ := ownerFold_ok c owner c.N own ho
    have hi := (iuCheck_iff owner own (by omega) (fun g hg => ho' g (by omega))).2 h4
    have hm' : thetaFold c theta = .ok mask := hm
    have ho2 : ownerFold c owner = .ok own := ho
    rw [divisionOf_eq, hm']
    dsimp only
    rw [ite_eq_right (by omega), ho2]
    dsimp only
    rw [ite_eq_left (by rw [hN, ← h2]; exact hi)]
    exact ⟨_, rfl⟩

theorem divisionOf_error {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) (theta owner : Array Json)
    (k : String) : divisionOf c theta owner = .error k ↔
      FirstFault (ThetaOk P theta) (ThetaFault P theta) theta.size k ∨
        (∀ t < theta.size, ThetaOk P theta t) ∧
          ((owner.size ≠ gates P ∧ k = "owner-length") ∨
            owner.size = gates P ∧
              (FirstFault (OwnerOk P owner) (OwnerFault P owner) (gates P) k ∨
                (∀ g < gates P, OwnerOk P owner g) ∧ EmptyIU owner ∧ k = "empty-iu")) := by
  have hN := (circuitOf_ok P c hc).1
  have hθe : ∀ k, thetaFold c theta = .error k ↔ FirstFault (ThetaOk P theta) (ThetaFault P theta) theta.size k :=
    thetaFold_error hc theta
  have hθo : (∃ s, thetaFold c theta = .ok s) ↔ ∀ t < theta.size, ThetaOk P theta t := thetaFold_isOk hc theta
  have hoe : ∀ k, ownerFold c owner = .error k ↔ FirstFault (OwnerOk P owner) (OwnerFault P owner) (gates P) k :=
    ownerFold_error hc owner
  have hoo : (∃ s, ownerFold c owner = .ok s) ↔ ∀ g < gates P, OwnerOk P owner g := ownerFold_isOk hc owner
  rw [divisionOf_eq, ← hθe, ← hθo, ← hoe, ← hoo, ← hN]
  rcases h1 : thetaFold c theta with e | mask
  · simp
  dsimp only
  by_cases h2 : owner.size = c.N
  · rw [ite_eq_right (not_not.2 h2)]
    rcases h3 : ownerFold c owner with e | own
    · simp [h2]
    dsimp only
    obtain ⟨hs, ho⟩ := ownerFold_ok c owner c.N own h3
    have hi := iuCheck_iff owner own (by omega) (fun g hg => ho g (by omega))
    rw [h2] at hi
    split_ifs with h4
    · simp [h2, hi.1 h4]
    · simp [h2, not_not.1 (mt hi.2 h4), eq_comm]
  · rw [ite_eq_left h2]
    simp [h2, eq_comm]

/-! ## The tally -/

theorem getElem!_modify {α : Type} [Inhabited α] (xs : Array α) (j : ℕ) (f : α → α) {i : ℕ} (hi : i < xs.size) :
    (xs.modify j f)[i]! = if j = i then f xs[i]! else xs[i]! := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_modify, getElem!_pos xs i hi]

/-- The reads of one gate, added to an IU's set: those outside the IU. -/
theorem readsFold_mem (reads : Array ℕ) (own : Array (Option ℕ)) (u : ℕ) (into : Std.HashSet ℕ) (x : ℕ) :
    x ∈ reads.foldl (fun into a => if own[a]! = some u then into else into.insert a) into ↔
      x ∈ into ∨ (x ∈ reads ∧ own[x]! ≠ some u) := by
  refine Array.foldl_induction (as := reads)
    (fun i S => ∀ x, x ∈ S ↔ x ∈ into ∨ ∃ j < i, reads[j]! = x ∧ own[x]! ≠ some u) (by simp) ?_ x |>.trans ?_
  · intro i S hS x
    simp only [getElem!_fin]
    split_ifs with h
    · rw [hS x]
      constructor
      · rintro (h' | ⟨j, hj, h'⟩)
        · exact Or.inl h'
        · exact Or.inr ⟨j, by omega, h'⟩
      · rintro (h' | ⟨j, hj, hx, hne⟩)
        · exact Or.inl h'
        · rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
          · exact Or.inr ⟨j, hj, hx, hne⟩
          · subst hx; exact absurd h hne
    · rw [Std.HashSet.mem_insert, hS x, beq_iff_eq]
      constructor
      · rintro (rfl | h' | ⟨j, hj, h'⟩)
        · exact Or.inr ⟨i, by omega, rfl, h⟩
        · exact Or.inl h'
        · exact Or.inr ⟨j, by omega, h'⟩
      · rintro (h' | ⟨j, hj, hx, hne⟩)
        · exact Or.inr (Or.inl h')
        · rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
          · exact Or.inr (Or.inr ⟨j, hj, hx, hne⟩)
          · exact Or.inl hx
  · refine or_congr_right ⟨?_, ?_⟩
    · rintro ⟨j, hj, rfl, hne⟩
      exact ⟨by rw [getElem!_pos reads j hj]; exact Array.getElem_mem hj, hne⟩
    · rintro ⟨hx, hne⟩
      obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hx
      exact ⟨j, hj, by rw [getElem!_pos reads j hj], hne⟩

/-- One step of `tally`. -/
def tallyStep (c : Circuit) (d : Division) (acc : Array (ℕ × Std.HashSet ℕ)) (g : ℕ) : Array (ℕ × Std.HashSet ℕ) :=
  match d.owner[g]! with
  | some u => acc.modify u fun (s, into) =>
      (s + 1, c.reads[g]!.foldl (fun into a => if d.owner[a]! = some u then into else into.insert a) into)
  | none => acc

theorem tally_eq (c : Circuit) (d : Division) :
    tally c d = (List.range c.N).foldl (tallyStep c d) (Array.replicate d.n (0, {})) := rfl

/-- `tally`'s first `m` steps: per IU, its gates so far and the gates outside it they read. -/
theorem tally_inv (c : Circuit) (d : Division) : ∀ m,
    let T := (List.range m).foldl (tallyStep c d) (Array.replicate d.n (0, {}))
    T.size = d.n ∧ ∀ u < d.n, T[u]!.1 = ((Finset.range m).filter fun g => d.owner[g]! = some u).card ∧
      ∀ x, x ∈ T[u]!.2 ↔ ∃ g < m, d.owner[g]! = some u ∧ x ∈ c.reads[g]! ∧ d.owner[x]! ≠ some u
  | 0 => by
    refine ⟨by simp, fun u hu => ?_⟩
    rw [List.range_zero, List.foldl_nil, getElem!_pos _ u (by simpa using hu), Array.getElem_replicate]
    simp
  | m + 1 => by
    obtain ⟨hs, hT⟩ := tally_inv c d m
    intro T
    simp only [T, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range m).foldl _ _ = A at hs hT
    unfold tallyStep
    rcases ho : d.owner[m]! with _ | v
    · refine ⟨hs, fun u hu => ⟨?_, fun x => ?_⟩⟩
      · rw [(hT u hu).1, Finset.range_add_one, Finset.filter_insert]
        simp [ho]
      · rw [(hT u hu).2 x]
        constructor
        · rintro ⟨g, hg, h⟩; exact ⟨g, by omega, h⟩
        · rintro ⟨g, hg, h1, h⟩
          rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
          · exact ⟨g, hg, h1, h⟩
          · rw [ho] at h1; cases h1
    · refine ⟨by simp [hs], fun u hu => ?_⟩
      rw [getElem!_modify _ _ _ (by omega)]
      by_cases hvu : v = u
      · subst hvu
        rw [ite_eq_left rfl]
        obtain ⟨h1, h2⟩ := hT v hu
        rcases hp : A[v]! with ⟨s0, into0⟩
        rw [hp] at h1 h2
        dsimp only at h1 h2
        refine ⟨?_, fun x => ?_⟩
        · dsimp only
          rw [h1, Finset.range_add_one, Finset.filter_insert, ite_eq_left ho,
            Finset.card_insert_of_notMem (by simp)]
        · dsimp only
          rw [readsFold_mem, h2 x]
          constructor
          · rintro (⟨g, hg, h⟩ | h)
            · exact ⟨g, by omega, h⟩
            · exact ⟨m, by omega, ho, h⟩
          · rintro ⟨g, hg, h1, h⟩
            rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
            · exact Or.inl ⟨g, hg, h1, h⟩
            · exact Or.inr h
      · rw [ite_eq_right hvu]
        refine ⟨?_, fun x => ?_⟩
        · rw [(hT u hu).1, Finset.range_add_one, Finset.filter_insert]
          simp [ho, hvu]
        · rw [(hT u hu).2 x]
          constructor
          · rintro ⟨g, hg, h⟩; exact ⟨g, by omega, h⟩
          · rintro ⟨g, hg, h1, h⟩
            rcases Nat.lt_succ_iff_lt_or_eq.1 hg with hg | rfl
            · exact ⟨g, hg, h1, h⟩
            · rw [ho] at h1; exact absurd (Option.some.inj h1) hvu

theorem tally_spec (c : Circuit) (d : Division) :
    (tally c d).size = d.n ∧ ∀ u < d.n,
      (tally c d)[u]!.1 = ((Finset.range c.N).filter fun g => d.owner[g]! = some u).card ∧
      ∀ x, x ∈ (tally c d)[u]!.2 ↔ ∃ g < c.N, d.owner[g]! = some u ∧ x ∈ c.reads[g]! ∧ d.owner[x]! ≠ some u := by
  rw [tally_eq]
  exact tally_inv c d c.N

/-! ## Sums over a hash set -/

theorem foldl_ite_add (p : ℕ → Prop) [DecidablePred p] (f : ℕ → ℕ) : ∀ (l : List ℕ) (init : ℕ),
    l.foldl (fun x a => if p a then x else x + f a) init = init + (l.map fun a => if p a then 0 else f a).sum
  | [], init => by simp
  | a :: l, init => by
    rw [List.foldl_cons, foldl_ite_add p f l, List.map_cons, List.sum_cons]
    split_ifs <;> omega

/-- A sum over a hash set of naturals, as a sum over the gates it holds. -/
theorem hashSet_fold_sum {N : ℕ} (S : Std.HashSet ℕ) (F : Finset (Fin N)) (hS : ∀ x, x ∈ S ↔ ∃ a ∈ F, a.val = x)
    (p : ℕ → Prop) [DecidablePred p] (f : ℕ → ℕ) :
    S.fold (fun x a => if p a then x else x + f a) 0 = ∑ a ∈ F, if p a.val then 0 else f a.val := by
  rw [Std.HashSet.fold_eq_foldl_toList, foldl_ite_add, zero_add]
  have hnd : S.toList.Nodup := Std.HashSet.distinct_toList.imp fun h => by simpa using h
  rw [← List.sum_toFinset _ hnd]
  have : S.toList.toFinset = F.map Fin.valEmbedding := by
    ext x
    simp only [List.mem_toFinset, Std.HashSet.mem_toList, hS, Finset.mem_map, Fin.valEmbedding_apply]
  rw [this, Finset.sum_map]
  rfl

/-! ## The check's arrays are the definition's circuit, baseline and partition -/

/-- What `LaysOut` and `circuitOf` together say of gate `g`: the same width, input flag and reads. -/
theorem laysOut_gate {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {D : Verity.Circuit} (hL : LaysOut P D)
    (g : Fin D.N) : D.w g = c.w[g.val]! ∧ D.isInput g = c.isInput[g.val]! ∧
      ∀ a : Fin D.N, a ∈ D.reads g ↔ a.val ∈ c.reads[g.val]! := by
  obtain ⟨p, ops, sc, hs, hr, hi, hreads⟩ := hL.2 g
  obtain ⟨p', sc', hs', hr', hi', -, -⟩ := circuitOf_gate hc (by rw [← hL.1]; exact g.isLt)
  rw [hs] at hs'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs'
  obtain ⟨rfl, rfl, -⟩ := hs'
  rw [hr] at hr'
  exact ⟨Flock.Program.Ty.value.inj hr', by rw [hi, hi'], hreads⟩

theorem mem_iff_index (theta : Array Json) (x : ℕ) :
    (∃ t ∈ theta, natOf? t = some x) ↔ ∃ t < theta.size, natOf? theta[t]! = some x := by
  constructor
  · rintro ⟨t, ht, h⟩
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ht
    exact ⟨i, hi, by rw [getElem!_pos theta i hi]; exact h⟩
  · rintro ⟨i, hi, h⟩
    exact ⟨theta[i], Array.getElem_mem hi, by rw [← getElem!_pos theta i hi]; exact h⟩

theorem names_mem {theta : Array Json} {D : Verity.Circuit} {θ : Finset (Fin D.N)} (hθ : Names theta θ)
    (g : Fin D.N) : g ∈ θ ↔ ∃ t < theta.size, natOf? theta[t]! = some g.val := by
  rw [hθ g]
  constructor
  · rintro ⟨t, ht, h⟩
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ht
    exact ⟨i, hi, by rw [getElem!_pos theta i hi]; exact h⟩
  · rintro ⟨i, hi, h⟩
    exact ⟨theta[i], Array.getElem_mem hi, by rw [← getElem!_pos theta i hi]; exact h⟩

theorem card_filter_fin (N : ℕ) (p : ℕ → Prop) [DecidablePred p] :
    (Finset.univ.filter fun g : Fin N => p g.val).card = ((Finset.range N).filter p).card := by
  rw [← Finset.card_map Fin.valEmbedding]
  congr 1
  ext x
  simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply,
    Finset.mem_range]
  constructor
  · rintro ⟨g, hg, rfl⟩; exact ⟨g.isLt, hg⟩
  · rintro ⟨hx, h⟩; exact ⟨⟨x, hx⟩, h, rfl⟩

/-- The setting of a verdict: the Program's circuit `c`, the checked division `dv`, a declaration `d` the Program lays
out with the baseline θ names, and a partition `U` that the owners are. -/
structure Setting (P : Prog) (theta owner : Array Json) (c : Circuit) (dv : Division) (d : Verity.Declaration)
    (n : ℕ) (U : Verity.Partition d.D n) : Prop where
  hc : circuitOf P = .ok c
  hd : divisionOf c theta owner = .ok dv
  hL : LaysOut P d.D
  hθ : Names theta d.θ
  hU : Divides owner U

namespace Setting

variable {P : Prog} {theta owner : Array Json} {c : Circuit} {dv : Division} {d : Verity.Declaration} {n : ℕ}
  {U : Verity.Partition d.D n} (S : Setting P theta owner c dv d n U)
include S

theorem N_eq : d.D.N = c.N := S.hL.1.trans (circuitOf_ok P c S.hc).1.symm

theorem owner_eq (g : Fin d.D.N) : dv.owner[g.val]! = natOf? owner[g.val]! :=
  (divisionOf_ok S.hc S.hd).2.2.2.2.2.1 g (by rw [← S.hL.1]; exact g.isLt)

theorem unit_some (g : Fin d.D.N) (u : Fin n) : U.unit g = some u ↔ dv.owner[g.val]! = some u.val := by
  rw [S.owner_eq g]
  exact (S.hU.2.1 g).2 u

theorem n_eq : n = dv.n := by
  obtain ⟨-, -, hsz, -, hos, ho, hn, -⟩ := divisionOf_ok S.hc S.hd
  obtain ⟨hb, hm⟩ := iuCount_spec dv.owner
  rw [← hn] at hb hm
  apply le_antisymm
  · rcases Nat.eq_zero_or_pos n with h0 | hpos
    · omega
    obtain ⟨g, hg⟩ := S.hU.2.2 ⟨n - 1, by omega⟩
    have := (S.unit_some g _).1 (Finset.mem_filter.1 hg).2
    have := hb g.val (by rw [hos, ← S.hL.1]; exact g.isLt) _ this
    simp at this
    omega
  · rcases hm with h0 | ⟨g, hg, h⟩
    · omega
    rw [hos, ← S.hL.1] at hg
    rcases hU : U.unit ⟨g, hg⟩ with _ | u
    · have h1 := (S.hU.2.1 ⟨g, hg⟩).1.1 hU
      have h2 := S.owner_eq ⟨g, hg⟩
      rw [h, h1, natOf?_str] at h2
      cases h2
    · have h1 := (S.unit_some ⟨g, hg⟩ u).1 hU
      rw [h, Option.some.injEq] at h1
      have := u.isLt
      omega

theorem theta_mem (a : Fin d.D.N) : a ∈ d.θ ↔ a.val ∈ dv.theta := by
  rw [names_mem S.hθ, (divisionOf_ok S.hc S.hd).2.1]

theorem size_eq (u : Fin n) : d.D.size (U.gates u) = (tally c dv)[u.val]!.1 := by
  have hu : u.val < dv.n := S.n_eq ▸ u.isLt
  rw [(tally_spec c dv).2 u.val hu |>.1, Verity.Circuit.size, Verity.Partition.gates, ← S.N_eq,
    ← card_filter_fin]
  exact congrArg Finset.card (Finset.filter_congr fun g _ => S.unit_some g u)

theorem into_mem (u : Fin n) (x : ℕ) :
    x ∈ (tally c dv)[u.val]!.2 ↔ ∃ a ∈ d.D.into (U.gates u), a.val = x := by
  have hu : u.val < dv.n := S.n_eq ▸ u.isLt
  rw [(tally_spec c dv).2 u.val hu |>.2 x]
  simp only [Verity.Circuit.into, Verity.Partition.gates, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨g, hg, hgu, hx, hxu⟩
    have hgP : g < gates P := (circuitOf_ok P c S.hc).1 ▸ hg
    obtain ⟨-, -, -, -, -, -, hlt⟩ := circuitOf_gate S.hc hgP
    have hxg := hlt x hx
    have hgD : g < d.D.N := S.N_eq ▸ hg
    have hxD : x < d.D.N := by omega
    refine ⟨⟨x, hxD⟩, ⟨fun h => hxu ((S.unit_some _ u).1 h), ⟨g, hgD⟩, (S.unit_some _ u).2 hgu,
      ((laysOut_gate S.hc S.hL ⟨g, hgD⟩).2.2 ⟨x, hxD⟩).2 hx⟩, rfl⟩
  · rintro ⟨a, ⟨hau, g, hgu, hag⟩, rfl⟩
    exact ⟨g.val, S.N_eq ▸ g.isLt, (S.unit_some g u).1 hgu, ((laysOut_gate S.hc S.hL g).2.2 a).1 hag,
      fun h => hau ((S.unit_some a u).2 h)⟩

theorem novel_eq (u : Fin n) : Nci.novelReads d (U.gates u) =
    (tally c dv)[u.val]!.2.fold (fun x a => if dv.theta.contains a then x else x + c.w[a]!) 0 := by
  rw [hashSet_fold_sum _ _ (S.into_mem u), Nci.novelReads, Verity.Circuit.width, Finset.sdiff_eq_filter,
    Finset.sum_filter]
  refine Finset.sum_congr rfl fun a _ => ?_
  have h1 := S.theta_mem a
  have h2 := (laysOut_gate S.hc S.hL a).1
  by_cases ha : a ∈ d.θ
  · simp [ha, Std.HashSet.contains_iff_mem.2 (h1.1 ha)]
  · have : ¬ dv.theta.contains a.val = true := fun h => ha (h1.2 (Std.HashSet.contains_iff_mem.1 h))
    simp [ha, this, h2]

/-- Entry `u` of `ius`: the size and novel reads of IU `u`. -/
theorem ius_get (u : Fin n) : (ius c dv)[u.val]! = (d.D.size (U.gates u), Nci.novelReads d (U.gates u)) := by
  have hu : u.val < (tally c dv).size := by rw [(tally_spec c dv).1, ← S.n_eq]; exact u.isLt
  rw [S.size_eq u, S.novel_eq u, ius, getElem!_pos _ u.val (by simpa using hu), Array.getElem_map,
    ← getElem!_pos _ u.val hu]

theorem ius_size : (ius c dv).size = n := by
  rw [ius, Array.size_map, (tally_spec c dv).1, S.n_eq]

end Setting

/-! ## The instance -/

theorem pairOf?_eq (v : Json) (a b : Json) : pairOf? v = some (a, b) ↔ v = .arr #[a, b] := by
  cases v with
  | arr xs =>
    simp only [pairOf?, Json.arr.injEq]
    constructor
    · intro h
      split_ifs at h with hs
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      apply Array.ext
      · simp [hs]
      · intro i h1 h2
        have : i = 0 ∨ i = 1 := by omega
        rcases this with rfl | rfl <;> simp <;> exact (getElem!_pos xs _ (by omega)).symm
    · rintro rfl
      simp
  | _ => simp [pairOf?]

/-- An instance missing a field, or with one of the wrong JSON kind, is refused with `instance`. -/
theorem parse_not_shaped (j : Json) (h : ¬ Shaped j) : parse j = .error "instance" := by
  unfold Shaped at h
  unfold parse field
  by_cases hA : ∃ s, j.getObjVal? "program" = .ok (.str s)
  · obtain ⟨s, h1⟩ := hA
    simp only [h1]
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
    by_cases hB : ∃ a, j.getObjVal? "theta" = .ok (.arr a)
    · obtain ⟨t, h2⟩ := hB
      simp only [h2]
      by_cases hC : ∃ a, j.getObjVal? "owner" = .ok (.arr a)
      · obtain ⟨o, h3⟩ := hC
        simp only [h3]
        by_cases hD : ∃ v, j.getObjVal? "gamma" = .ok v
        · obtain ⟨v, h4⟩ := hD
          simp only [h4]
          rcases h5 : j.getObjVal? "alpha" with e | w
          · rfl
          · exact absurd ⟨⟨s, h1⟩, ⟨t, h2⟩, ⟨o, h3⟩, ⟨v, h4⟩, ⟨w, h5⟩⟩ h
        · rcases h4 : j.getObjVal? "gamma" with e | v
          · rfl
          · exact absurd ⟨v, h4⟩ hD
      · rcases h3 : j.getObjVal? "owner" with e | (_ | _ | _ | _ | o | _) <;> try rfl
        exact absurd ⟨o, h3⟩ hC
    · rcases h2 : j.getObjVal? "theta" with e | (_ | _ | _ | _ | t | _) <;> try rfl
      exact absurd ⟨t, h2⟩ hB
  · rcases h1 : j.getObjVal? "program" with e | (_ | _ | _ | s | _ | _) <;> try rfl
    exact absurd ⟨s, h1⟩ hA

/-- A well-shaped instance: its fields parse, or it is refused with `params`. -/
theorem parse_shaped (j : Json) {s : String} {t o : Array Json} {v w : Json}
    (h1 : j.getObjVal? "program" = .ok (.str s)) (h2 : j.getObjVal? "theta" = .ok (.arr t))
    (h3 : j.getObjVal? "owner" = .ok (.arr o)) (h4 : j.getObjVal? "gamma" = .ok v)
    (h5 : j.getObjVal? "alpha" = .ok w) :
    (∀ i, parse j = .ok i ↔ Parses j i) ∧ ∀ k, parse j = .error k → k = "params" := by
  have fail : parse j = .error "params" → (∀ i, ¬ Parses j i) →
      (∀ i, parse j = .ok i ↔ Parses j i) ∧ ∀ k, parse j = .error k → k = "params" := fun he hn => by
    rw [he]
    refine ⟨fun i => ⟨fun h => ?_, fun h => absurd h (hn i)⟩, fun k hk => ?_⟩
    · cases h
    · cases hk; rfl
  have pre : ∀ i, Parses j i → natOf? v = some i.gamma ∧ ∃ a b, w = .arr #[a, b] ∧ natOf? a = some i.num ∧
      natOf? b = some i.den ∧ 0 < i.den := by
    rintro i ⟨-, -, -, ⟨v', hv', hg'⟩, ⟨a, b, ha, hna, hnb⟩, hpos⟩
    rw [h4, Except.ok.injEq] at hv'
    rw [h5, Except.ok.injEq] at ha
    exact ⟨hv' ▸ hg', a, b, ha, hna, hnb, hpos⟩
  rcases hg : natOf? v with _ | gamma
  · refine fail (by unfold parse field; simp only [h1, h2, h3, h4, h5, hg, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; try rfl) fun i hP => ?_
    have := (pre i hP).1
    rw [hg] at this; cases this
  rcases hp : pairOf? w with _ | ⟨a, b⟩
  · refine fail (by unfold parse field; simp only [h1, h2, h3, h4, h5, hg, hp, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; try rfl) fun i hP => ?_
    obtain ⟨-, a, b, hw, -⟩ := pre i hP
    rw [← pairOf?_eq] at hw
    rw [hw] at hp; cases hp
  have hw := (pairOf?_eq w a b).1 hp
  have pre' : ∀ i, Parses j i → natOf? a = some i.num ∧ natOf? b = some i.den ∧ 0 < i.den := by
    intro i hP
    obtain ⟨-, a', b', hw', h', h'', hpos⟩ := pre i hP
    rw [hw, Json.arr.injEq] at hw'
    simp at hw'
    obtain ⟨e1, e2⟩ := hw'
    subst e1 e2
    exact ⟨h', h'', hpos⟩
  rcases ha : natOf? a with _ | num
  · refine fail (by unfold parse field; simp only [h1, h2, h3, h4, h5, hg, hp, ha, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; try rfl) fun i hP => ?_
    have := (pre' i hP).1
    rw [ha] at this; cases this
  rcases hb : natOf? b with _ | den
  · refine fail (by unfold parse field; simp only [h1, h2, h3, h4, h5, hg, hp, ha, hb, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; try rfl) fun i hP => ?_
    have := (pre' i hP).2.1
    rw [hb] at this; cases this
  by_cases hd : den = 0
  · refine fail (by unfold parse field; simp only [h1, h2, h3, h4, h5, hg, hp, ha, hb, hd, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; try rfl) fun i hP => ?_
    obtain ⟨-, h', hpos⟩ := pre' i hP
    rw [hb, Option.some.injEq] at h'
    omega
  have hok : parse j = .ok { program := s, theta := t, owner := o, gamma, num, den } := by
    unfold parse field
    simp only [h1, h2, h3, h4, h5, hg, hp, ha, hb, hd, bind, Except.bind, pure, Except.pure, throw, throwThe,
      MonadExceptOf.throw]
    try rfl
  rw [hok]
  refine ⟨fun i => ⟨?_, ?_⟩, fun k hk => by cases hk⟩
  · rintro ⟨⟩
    exact ⟨h1, h2, h3, ⟨v, h4, hg⟩, ⟨a, b, by rw [h5, hw], ha, hb⟩, Nat.pos_of_ne_zero hd⟩
  · intro hP
    obtain ⟨hg', -⟩ := pre i hP
    obtain ⟨ha', hb', -⟩ := pre' i hP
    obtain ⟨hs, ht, ho, -⟩ := hP
    rw [h1, Except.ok.injEq, Json.str.injEq] at hs
    rw [h2, Except.ok.injEq, Json.arr.injEq] at ht
    rw [h3, Except.ok.injEq, Json.arr.injEq] at ho
    rw [hg, Option.some.injEq] at hg'
    rw [ha, Option.some.injEq] at ha'
    rw [hb, Option.some.injEq] at hb'
    cases i
    simp_all

theorem parse_ok (j : Json) (i : Instance) : parse j = .ok i ↔ Parses j i := by
  by_cases h : Shaped j
  · obtain ⟨⟨s, h1⟩, ⟨t, h2⟩, ⟨o, h3⟩, ⟨v, h4⟩, ⟨w, h5⟩⟩ := h
    exact (parse_shaped j h1 h2 h3 h4 h5).1 i
  · rw [parse_not_shaped j h]
    simp only [reduceCtorEq, false_iff]
    rintro ⟨h1, h2, h3, ⟨v, h4, -⟩, ⟨a, b, h5, -⟩, -⟩
    exact h ⟨⟨_, h1⟩, ⟨_, h2⟩, ⟨_, h3⟩, ⟨_, h4⟩, ⟨_, h5⟩⟩

theorem parse_error (j : Json) (k : String) : parse j = .error k ↔
    (¬ Shaped j ∧ k = "instance") ∨ (Shaped j ∧ (¬ ∃ i, Parses j i) ∧ k = "params") := by
  by_cases h : Shaped j
  · obtain ⟨⟨s, h1⟩, ⟨t, h2⟩, ⟨o, h3⟩, ⟨v, h4⟩, ⟨w, h5⟩⟩ := h
    have hs : Shaped j := ⟨⟨s, h1⟩, ⟨t, h2⟩, ⟨o, h3⟩, ⟨v, h4⟩, ⟨w, h5⟩⟩
    obtain ⟨hok, herr⟩ := parse_shaped j h1 h2 h3 h4 h5
    simp only [hs, not_true_eq_false, false_and, true_and, false_or]
    constructor
    · intro he
      refine ⟨fun ⟨i, hi⟩ => ?_, herr k he⟩
      rw [← hok, he] at hi; cases hi
    · rintro ⟨hn, rfl⟩
      rcases hp : parse j with e | i
      · rw [herr e hp]
      · exact absurd ⟨i, (hok i).1 hp⟩ hn
  · rw [parse_not_shaped j h]
    simp [h, eq_comm]

/-! ## The definition's objects, built from the check's -/

/-- The circuit the Program lays out, built from `circuitOf`'s arrays: no outputs and every operation `0`, which
`LaysOut` leaves free. -/
def circuitFrom {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) : Verity.Circuit where
  N := c.N
  w g := c.w[g.val]!
  isInput g := c.isInput[g.val]!
  reads g := Finset.univ.filter fun a => a.val ∈ c.reads[g.val]!
  reads_lt g a ha := by
    have hg : g.val < gates P := (circuitOf_ok P c hc).1 ▸ g.isLt
    obtain ⟨-, -, -, -, -, -, hlt⟩ := circuitOf_gate hc hg
    exact hlt a.val (Finset.mem_filter.1 ha).2
  input_reads g hi := by
    have hg : g.val < gates P := (circuitOf_ok P c hc).1 ▸ g.isLt
    obtain ⟨-, -, -, -, -, hin, -⟩ := circuitOf_gate hc hg
    ext a
    simp [hin hi]
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := ∅

theorem laysOut_circuitFrom {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) : LaysOut P (circuitFrom hc) := by
  refine ⟨(circuitOf_ok P c hc).1, fun g => ?_⟩
  have hg : g.val < gates P := (circuitOf_ok P c hc).1 ▸ g.isLt
  obtain ⟨p, sc, hs, hr, hi, -, -⟩ := circuitOf_gate hc hg
  exact ⟨p, c.reads[g.val]!, sc, hs, hr, hi, fun a =>
    ⟨fun h => (Finset.mem_filter.1 h).2, fun h => Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩⟩⟩

/-- θ as the entries name it is a set of input gates of every circuit the Program lays out. -/
theorem theta_inputs {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {theta owner : Array Json}
    {dv : Division} (hd : divisionOf c theta owner = .ok dv) {D : Verity.Circuit} (hL : LaysOut P D) :
    ∀ g : Fin D.N, (∃ t ∈ theta, natOf? t = some g.val) → D.isInput g = true := by
  intro g hg
  obtain ⟨t, ht, h⟩ := (mem_iff_index theta g.val).1 hg
  obtain ⟨g', hg', -, hin, -⟩ := (divisionOf_ok hc hd).1 t ht
  rw [h, Option.some.injEq] at hg'
  subst hg'
  rw [(laysOut_gate hc hL g).2.1]
  exact (circuitOf_isInput hc (hL.1 ▸ g.isLt)).2 hin

/-- The partition the owners are, of every circuit the Program lays out. -/
theorem exists_partition {P : Prog} {c : Circuit} (hc : circuitOf P = .ok c) {theta owner : Array Json}
    {dv : Division} (hd : divisionOf c theta owner = .ok dv) {D : Verity.Circuit} (hL : LaysOut P D) :
    ∃ U : Verity.Partition D dv.n, Divides owner U := by
  obtain ⟨-, -, hsz, hok, hos, ho, hn, hne⟩ := divisionOf_ok hc hd
  obtain ⟨hb, -⟩ := iuCount_spec dv.owner
  rw [← hn] at hb
  have hlt : ∀ g : Fin D.N, g.val < gates P := fun g => hL.1 ▸ g.isLt
  have hbound : ∀ g : Fin D.N, ∀ u, dv.owner[g.val]! = some u → u < dv.n :=
    fun g u h => hb g.val (by rw [hos]; exact hlt g) u h
  let unit : Fin D.N → Option (Fin dv.n) := fun g =>
    match h : dv.owner[g.val]! with
    | some u => some ⟨u, hbound g u h⟩
    | none => none
  have hunit : ∀ g (u : Fin dv.n), unit g = some u ↔ dv.owner[g.val]! = some u.val := by
    intro g u
    simp only [unit]
    split
    · rename_i v hv
      constructor
      · rintro ⟨⟩; exact hv
      · intro h
        rw [hv, Option.some.injEq] at h
        exact congrArg some (Fin.ext h)
    · rename_i hv
      rw [hv]
      simp
  have hnone : ∀ g, unit g = none ↔ dv.owner[g.val]! = none := by
    intro g
    simp only [unit]
    split
    · rename_i v hv; simp [hv]
    · rename_i hv; simp [hv]
  have hinput : ∀ g : Fin D.N, (dv.owner[g.val]! = none ↔ owner[g.val]! = .str "input") ∧
      (dv.owner[g.val]! = none ↔ D.isInput g = true) := by
    intro g
    rw [ho g.val (hlt g), (laysOut_gate hc hL g).2.1, circuitOf_isInput hc (hlt g)]
    rcases hok g.val (hlt g) with ⟨h1, h2⟩ | ⟨⟨u, h1⟩, h2⟩
    · rw [h1, natOf?_str]
      exact ⟨by simp, by simp [h2]⟩
    · rw [h1]
      refine ⟨⟨fun h => (by cases h), fun h => ?_⟩, by simp [h2]⟩
      rw [h, natOf?_str] at h1
      cases h1
  refine ⟨⟨unit, fun g => (hnone g).trans (hinput g).2⟩, ?_, fun g => ⟨?_, fun u => ?_⟩, fun u => ?_⟩
  · rw [hsz, hL.1]
  · exact (hnone g).trans (hinput g).1
  · exact (hunit g u).trans (by rw [ho g.val (hlt g)])
  · have hall := ((iuCheck_iff owner dv.owner (by rw [hos, hsz]) (fun g hg => ho g (hsz ▸ hg))).2 hne).2
    rw [← hn] at hall
    obtain ⟨g, hg, h⟩ := (usedOf_mem dv.owner u.val).1 (hall u.val u.isLt)
    rw [hos, ← hL.1] at hg
    exact ⟨⟨g, hg⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, (hunit ⟨g, hg⟩ u).2 h⟩⟩

/-! ## The first failing IU -/

theorem findSome_range_none {β : Type} (f : ℕ → Option β) (n : ℕ) :
    (List.range n).findSome? f = none ↔ ∀ i < n, f i = none := by
  simp [List.findSome?_eq_none_iff]

theorem findSome_range_some {β : Type} (f : ℕ → Option β) (b : β) : ∀ n,
    (List.range n).findSome? f = some b ↔ ∃ i < n, f i = some b ∧ ∀ j < i, f j = none
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.findSome?_append]
    rcases h : (List.range n).findSome? f with _ | b'
    · rw [findSome_range_none] at h
      simp only [Option.none_or, List.findSome?_cons, List.findSome?_nil]
      constructor
      · intro hb
        refine ⟨n, by omega, ?_, h⟩
        rcases hf : f n with _ | x <;> rw [hf] at hb <;> simp_all
      · rintro ⟨i, hi, hfi, hj⟩
        rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
        · rw [h i hi] at hfi; cases hfi
        · rw [hfi]
    · simp only [Option.some_or, Option.some.injEq]
      have hs := (findSome_range_some f b' n).1 h
      obtain ⟨i, hi, hfi, hj⟩ := hs
      constructor
      · rintro rfl
        exact ⟨i, by omega, hfi, hj⟩
      · rintro ⟨i', hi', hfi', hj'⟩
        rcases lt_trichotomy i i' with hlt | rfl | hgt
        · rw [hj' i hlt] at hfi; cases hfi
        · rw [hfi] at hfi'; exact Option.some.inj hfi'
        · rw [hj i' hgt] at hfi'; cases hfi'

/-- `novel ≤ (num / den) · size` in ℝ is `novel · den ≤ num · size` in ℕ. -/
theorem le_ratio_iff (num den size novel : ℕ) (hden : 0 < den) :
    (novel : ℝ) ≤ (num : ℝ) / den * size ↔ novel * den ≤ num * size := by
  rw [div_mul_eq_mul_div, le_div_iff₀ (by exact_mod_cast hden)]
  exact_mod_cast Iff.rfl

theorem ratio_lt_iff (num den size novel : ℕ) (hden : 0 < den) :
    (num : ℝ) / den * size < novel ↔ num * size < novel * den := by
  rw [div_mul_eq_mul_div, div_lt_iff₀ (by exact_mod_cast hden)]
  exact_mod_cast Iff.rfl

/-! ## `evaluate` -/

theorem evaluate_ok (j out : Json) : evaluate j = .ok out ↔ ∃ i P c dv, parse j = .ok i ∧
    Flock.Program.decode i.program.toUTF8 = .ok P ∧ circuitOf P = .ok c ∧ divisionOf c i.theta i.owner = .ok dv ∧
    out = verdict c.N (ius c dv) (failure i.gamma i.num i.den (ius c dv)) := by
  unfold evaluate
  rcases h1 : parse j with e | i
  · simp only [h1]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp
  rcases h2 : Flock.Program.decode i.program.toUTF8 with e | P <;>
    have h2' : Flock.Program.decode i.program.toByteArray = _ := h2
  · simp only [h1, h2, h2']; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2']
  rcases h3 : circuitOf P with e | c
  · simp only [h1, h2, h2', h3]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2', h3]
  rcases h4 : divisionOf c i.theta i.owner with e | dv
  · simp only [h1, h2, h2', h3, h4]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2', h3, h4]
  simp only [h1, h2, h2', h3, h4]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
  simp [h2', h3, h4, eq_comm]

theorem evaluate_error (j : Json) (k : String) : evaluate j = .error k ↔ parse j = .error k ∨
    ∃ i, parse j = .ok i ∧ (((∃ e, Flock.Program.decode i.program.toUTF8 = .error e) ∧ k = "program") ∨
      ∃ P, Flock.Program.decode i.program.toUTF8 = .ok P ∧
        (circuitOf P = .error k ∨ ∃ c, circuitOf P = .ok c ∧ divisionOf c i.theta i.owner = .error k)) := by
  unfold evaluate
  rcases h1 : parse j with e | i
  · simp only [h1]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp
  rcases h2 : Flock.Program.decode i.program.toUTF8 with e | P <;>
    have h2' : Flock.Program.decode i.program.toByteArray = _ := h2
  · simp only [h1, h2, h2']; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2']; exact ⟨Eq.symm, Eq.symm⟩
  rcases h3 : circuitOf P with e | c
  · simp only [h1, h2, h2', h3]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2', h3]
  rcases h4 : divisionOf c i.theta i.owner with e | dv
  · simp only [h1, h2, h2', h3, h4]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]; simp [h2', h3, h4]
  simp only [h1, h2, h2', h3, h4]; simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
  simp [h2', h3, h4]

end NciCheck.Properties.SecurityProofs
