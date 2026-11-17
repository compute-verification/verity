module

public import Proofs.Flock.Verifier.Registered

/-! The recursive audit's hidden statement read from the inner statement's own description rows, which are the registered ones or SHA-512 collides. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Flock.Registered

def AcceptsDescription (rt : Port) (st : Flock.Stmt) : Prop :=
  ∃ own : Array Port, rt ∈ own ∧ checkDescription st.c st.pub own = .ok ()

theorem description_rows {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st) :
    (∀ r < rt.leaves, opens rt (rowsOf st.c st.pub rt r) r (pathOf st.pub rt r) = true) ∧
    ∃ t p, st.pub.tables = some t ∧ portOf st.c rt.name = some p ∧
      (∀ i < st.pub.n, ref t st.c.ports.size p i < rt.leaves ∧
        st.pub.digest i p st.c.ports.size = rowsOf st.c st.pub rt (ref t st.c.ports.size p i)) ∧
      (∀ r < rt.leaves, ∃ i < st.pub.n, ref t st.c.ports.size p i = r) := by
  obtain ⟨own, hrt, hck⟩ := h
  obtain ⟨t, p, ht, hp, -, -, hopen, hinst, hread⟩ := checkDescription_ok hck hrt
  exact ⟨hopen, t, p, ht, hp, hinst, hread⟩

def ReadsBelow {Ω H : Type} (n : Nat) (stmt : (Nat → ByteArray) → Ω → H) : Prop :=
  ∀ q q' : Nat → ByteArray, (∀ r < n, q r = q' r) → ∀ ω, stmt q ω = stmt q' ω

theorem rows_eq_or_conflict {Ω H : Type} {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st)
    {stmt : (Nat → ByteArray) → Ω → H} (hst : ReadsBelow rt.leaves stmt) (p path₀ : Nat → ByteArray)
    (hp : ∀ r < rt.leaves, opens rt (p r) r (path₀ r) = true) :
    (∀ ω, stmt (rowsOf st.c st.pub rt) ω = stmt p ω) ∨
      ∃ r < rt.leaves, rowsOf st.c st.pub rt r ≠ p r ∧
        opens rt (rowsOf st.c st.pub rt r) r (pathOf st.pub rt r) = true ∧ opens rt (p r) r (path₀ r) = true := by
  rcases Classical.em (∀ r < rt.leaves, rowsOf st.c st.pub rt r = p r) with he | he
  · exact Or.inl (hst _ _ he)
  · exact Or.inr (Classical.byContradiction fun hno => he fun r hr =>
      Classical.byContradiction fun hne => hno ⟨r, hr, hne, (description_rows h).1 r hr, hp r hr⟩)

theorem rows_eq_or_collides {Ω H : Type} {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st)
    {stmt : (Nat → ByteArray) → Ω → H} (hst : ReadsBelow rt.leaves stmt) (p path₀ : Nat → ByteArray)
    (hp : ∀ r < rt.leaves, opens rt (p r) r (path₀ r) = true) :
    (∀ ω, stmt (rowsOf st.c st.pub rt) ω = stmt p ω) ∨
      ∃ r < rt.leaves, rowsOf st.c st.pub rt r ≠ p r ∧
        Collides (bcPair rt r (rowsOf st.c st.pub rt r) (pathOf st.pub rt r) (p r) (path₀ r)) := by
  rcases rows_eq_or_conflict h hst p path₀ hp with he | ⟨r, hr, hne, h1, h2⟩
  · exact Or.inl he
  · exact Or.inr ⟨r, hr, hne, opens_binding rt r _ _ _ _ hne h1 h2⟩

def ofRows {Ω H : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port) :
    (Nat → ByteArray) → Ω → H :=
  fun _ ω => stmt (rowsOf st.c st.pub rt) ω

theorem hx_ofRows {Ω H O Reg : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port)
    (R : O → Reg) (ω : O → Ω) (verifies : O → Prop) (reads : O → Nat → Prop) (row : O → Nat → ByteArray) :
    ∀ o, verifies o → ∀ q : Nat → ByteArray, (∀ i, reads o i → q i = row o i) →
      (fun (_ : Reg) ω' => stmt (rowsOf st.c st.pub rt) ω') (R o) (ω o) = ofRows stmt st rt q (ω o) :=
  fun _ _ _ _ => rfl

theorem readsCover_ofRows {Ω H : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port)
    (reads : Nat → Prop) :
    ∀ q q' : Nat → ByteArray, (∀ i, reads i → q i = q' i) → ∀ ω, ofRows stmt st rt q ω = ofRows stmt st rt q' ω :=
  fun _ _ _ _ => rfl

theorem ofRows_eq_or_collides {Ω H : Type} {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st)
    {stmt : (Nat → ByteArray) → Ω → H} (hst : ReadsBelow rt.leaves stmt) (p path₀ : Nat → ByteArray)
    (hp : ∀ r < rt.leaves, opens rt (p r) r (path₀ r) = true) :
    (∀ q ω, ofRows stmt st rt q ω = stmt p ω) ∨
      ∃ r < rt.leaves, rowsOf st.c st.pub rt r ≠ p r ∧
        Collides (bcPair rt r (rowsOf st.c st.pub rt r) (pathOf st.pub rt r) (p r) (path₀ r)) :=
  (rows_eq_or_collides h hst p path₀ hp).imp (fun he _ ω => he ω) id

end FlockSoundness.Discharge.Recursion
