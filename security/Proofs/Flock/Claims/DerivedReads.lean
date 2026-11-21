module

public import Flock.Registered

/-!
# The reads `Registered.withPositions` derives from C

`withPositions_reads`: each port `withPositions` derives is the verifier's port with its reads alone replaced, so its
name, value, root, leaves, domain and schema are the verifier's own; `withPositions_some`: it derives ports from ports.
-/

@[expose] public section

namespace Flock.Registered

private theorem ebind_ok' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β}
    (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩


private theorem mapM_ok_mem {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → ∀ a ∈ l, ∃ b ∈ l', f a = .ok b
  | [], _, _, a, ha => absurd ha List.not_mem_nil
  | a :: l, l', h, x, hx => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := ebind_ok' h
    obtain ⟨bs, hbs, h⟩ := ebind_ok' h
    cases h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨b, List.mem_cons_self, hb⟩
    · obtain ⟨y, hy, hfy⟩ := mapM_ok_mem hbs x hx
      exact ⟨y, List.mem_cons_of_mem _ hy, hfy⟩

private theorem array_mapM_ok_mem {ε α β : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ∀ a ∈ xs, ∃ b ∈ ys, f a = .ok b := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => simp [hl, Functor.map, Except.map] at h
  | ok l =>
    simp [hl, Functor.map, Except.map] at h
    subst h
    intro a ha
    obtain ⟨b, hb, hfb⟩ := mapM_ok_mem hl a (Array.mem_toList_iff.2 ha)
    exact ⟨b, by simpa using hb, hfb⟩

private theorem map_some_ok {ε α : Type} {x : Except ε α} {y : α} (h : (some <$> x) = .ok (some y)) :
    x = .ok y := by
  cases x <;> simp_all [Functor.map, Except.map]

private theorem map_some_some {ε α : Type} {x : Except ε α} {o : Option α} (h : (some <$> x) = .ok o) :
    ∃ y, o = some y := by
  cases x with
  | error e => simp [Functor.map, Except.map] at h
  | ok a =>
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    exact ⟨a, h.symm⟩

theorem pos_reads {P : _root_.Flock.Program.Prog} {indices : Array Nat} {reg r : Port}
    (h : _root_.Flock.OneCircuit.pos P indices reg = .ok r) :
    r = { reg with readsRel := r.readsRel, readsFrom := r.readsFrom } := by
  unfold _root_.Flock.OneCircuit.pos at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

theorem treeReads_reads {P : _root_.Flock.Program.Prog} {indices : Array Nat} {lo hi : Nat}
    {reg r : Port} (h : treeReads P indices lo hi reg = .ok r) :
    r = { reg with readsRel := r.readsRel, readsFrom := r.readsFrom } := by
  unfold treeReads at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

theorem outReads_reads {c : _root_.Flock.Circuit} {P : _root_.Flock.Program.Prog} {indices : Array Nat} {j : Nat}
    {reg r : Port} (h : outReads c P indices j reg = .ok r) :
    r = { reg with readsRel := r.readsRel, readsFrom := r.readsFrom } := by
  unfold outReads at h
  cases ho : outReadsAt c P indices j reg with
  | error e => simp [ho, bind, Except.bind] at h
  | ok x =>
    simp only [ho, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

theorem ins_reads {c : _root_.Flock.Circuit} {P : _root_.Flock.Program.Prog} {indices : Array Nat}
    {rows : Array (Nat × Nat × Nat)} {reg r : Port}
    (h : withPositions.ins c P indices rows reg = .ok r) :
    r = { reg with readsRel := r.readsRel, readsFrom := r.readsFrom } := by
  unfold withPositions.ins at h
  split at h
  · obtain ⟨r0, h0, h⟩ := ebind_ok' h
    split at h
    · exact pos_reads h
    · cases h
      exact treeReads_reads h0
  · exact pos_reads h

/-- **A port `withPositions` derives is the verifier's port with its reads alone derived from C.** -/
theorem withPositions_reads {c : _root_.Flock.Circuit} {pub : _root_.Flock.Public} {program : Option ByteArray}
    {own own' : Array Port}
    (h : withPositions c pub program (some own) = .ok (some own')) :
    ∀ reg ∈ own, ∃ reg' ∈ own', reg' = { reg with readsRel := reg'.readsRel, readsFrom := reg'.readsFrom } := by
  unfold withPositions at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    intro reg hreg
    obtain ⟨reg', hreg', hf⟩ := array_mapM_ok_mem (map_some_ok h) reg hreg
    refine ⟨reg', hreg', ?_⟩
    split at hf
    · exact outReads_reads hf
    · exact ins_reads hf

/-- **`withPositions` derives some ports from some ports.** -/
theorem withPositions_some {c : _root_.Flock.Circuit} {pub : _root_.Flock.Public} {program : Option ByteArray}
    {own : Array Port} {o : Option (Array Port)}
    (h : withPositions c pub program (some own) = .ok o) : ∃ own', o = some own' := by
  unfold withPositions at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals exact map_some_some h

end Flock.Registered
