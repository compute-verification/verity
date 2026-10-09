import Flock.Statement
import Proofs.Flock.Verifier.Registered

/-!
# Recursion: the inner statement's description, read from its own rows

The recursive audit's hidden statement, read from the inner statement `st` itself rather than from V*'s sessions: the
registered private circuit's statement at `st`'s description rows `rowsOf st.c st.pub rt` (the `b ‖ c` of the table of
`st`'s circuit's port named `rt`'s name), which Lean's verifier opens against the developer's registered root `rt`
(`Flock.Registered.checkDescription`, PROTOCOL.md §16.16). No session's reads are on this path.

* `AcceptsDescription rt st`: the description check accepts `st` against registered reads holding `rt`.
* `description_rows`: then each of `st`'s registered rows `r < rt.leaves` opens `rt` at `r`, through the path `st`'s
  header gives it (`pathOf`); every instance of `st` reads one of them, and each is read
  (`Flock.Registered.checkDescription_ok`).
* `rows_eq_or_conflict`, `rows_eq_or_collides`: for the developer's commit strings `p`, each opening `rt` at its row, and
  a statement that reads rows `[0, rt.leaves)` only (`ReadsBelow`), the statement at `st`'s rows is the statement at `p`,
  or at some row both open `rt` with different strings, which gives an explicit SHA-512 collision
  (`Flock.Registered.opens_binding`, `bcPair`).
* `ofRows stmt st rt`: `stmt` at `st`'s rows, whatever strings it is given. The read strings' hidden statement (`_hx`'s
  form, `hx_ofRows`) and ReadsCover (`readsCover_ofRows`) hold of it by `rfl`, whatever the reads; at the developer's
  strings it is `stmt p` or SHA-512 collides (`ofRows_eq_or_collides`).
* A hypothesis that the developer's strings open `rt` at every `i` holds of none: the positions past `rt.leaves` open
  nothing (`Flock.Registered.not_opens_all`).
-/

namespace FlockSoundness.Discharge.Recursion

open Flock.Registered

/-- **Lean's verifier accepts `st`'s description against `rt`**: its registered reads hold `rt`, and
`Flock.Registered.checkDescription` accepts `st` against them. -/
def AcceptsDescription (rt : Port) (st : Flock.Stmt) : Prop :=
  ∃ own : Array Port, rt ∈ own ∧ checkDescription st.c st.pub own = .ok ()

/-- **The inner statement's description rows open the registration, and every one is read.** When `st`'s description
is accepted against `rt`, each registered row `r < rt.leaves` of `st` opens `rt` at `r` through the path `st`'s header
gives it; `st` is a shared-row file whose circuit's port `p` is `rt`'s, every instance's row of it is the registered row
its ref names, and every registered row is some instance's. -/
theorem description_rows {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st) :
    (∀ r < rt.leaves, opens rt (rowsOf st.c st.pub rt r) r (pathOf st.pub rt r) = true) ∧
    ∃ t p, st.pub.tables = some t ∧ portOf st.c rt.name = some p ∧
      (∀ i < st.pub.n, ref t st.c.ports.size p i < rt.leaves ∧
        st.pub.digest i p st.c.ports.size = rowsOf st.c st.pub rt (ref t st.c.ports.size p i)) ∧
      (∀ r < rt.leaves, ∃ i < st.pub.n, ref t st.c.ports.size p i = r) := by
  obtain ⟨own, hrt, hck⟩ := h
  obtain ⟨t, p, ht, hp, -, -, hopen, hinst, hread⟩ := checkDescription_ok hck hrt
  exact ⟨hopen, t, p, ht, hp, hinst, hread⟩

/-- **A statement of commit strings that reads rows `[0, n)` only**: strings that agree there give one statement. -/
def ReadsBelow {Ω H : Type} (n : Nat) (stmt : (Nat → ByteArray) → Ω → H) : Prop :=
  ∀ q q' : Nat → ByteArray, (∀ r < n, q r = q' r) → ∀ ω, stmt q ω = stmt q' ω

/-- **`st`'s rows are the developer's, or two strings open one row.** When `st`'s description is accepted against `rt`
and the developer's commit strings `p` open `rt` at each of its rows (through `path₀`), a statement that reads rows
`[0, rt.leaves)` only is one at `st`'s rows and at `p`, or at some row `st`'s string and the developer's differ and
both open `rt` there. -/
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

/-- **`st`'s rows are the developer's, or SHA-512 collides**: `rows_eq_or_conflict`, with the two openings' explicit
collision (`bcPair`, `Flock.Registered.opens_binding`). -/
theorem rows_eq_or_collides {Ω H : Type} {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st)
    {stmt : (Nat → ByteArray) → Ω → H} (hst : ReadsBelow rt.leaves stmt) (p path₀ : Nat → ByteArray)
    (hp : ∀ r < rt.leaves, opens rt (p r) r (path₀ r) = true) :
    (∀ ω, stmt (rowsOf st.c st.pub rt) ω = stmt p ω) ∨
      ∃ r < rt.leaves, rowsOf st.c st.pub rt r ≠ p r ∧
        Collides (bcPair rt r (rowsOf st.c st.pub rt r) (pathOf st.pub rt r) (p r) (path₀ r)) := by
  rcases rows_eq_or_conflict h hst p path₀ hp with he | ⟨r, hr, hne, h1, h2⟩
  · exact Or.inl he
  · exact Or.inr ⟨r, hr, hne, opens_binding rt r _ _ _ _ hne h1 h2⟩

/-- **The hidden statement read from `st`'s own rows**: `stmt` at `st`'s description rows, whatever commit strings it is
given. -/
def ofRows {Ω H : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port) :
    (Nat → ByteArray) → Ω → H :=
  fun _ ω => stmt (rowsOf st.c st.pub rt) ω

/-- **`_hx` by `rfl`**: the hidden statement `x R ω := stmt (rowsOf st) ω` is `ofRows stmt st rt q` at every outcome
`o` and every `q`, whatever `o` verifies and whichever rows its sessions read (`verifies`, `reads` and `row` are
`VerifiesV`, `readsZ` and `rowZ` in `RecursiveAudit`'s `_hx`). -/
theorem hx_ofRows {Ω H O Reg : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port)
    (R : O → Reg) (ω : O → Ω) (verifies : O → Prop) (reads : O → Nat → Prop) (row : O → Nat → ByteArray) :
    ∀ o, verifies o → ∀ q : Nat → ByteArray, (∀ i, reads o i → q i = row o i) →
      (fun (_ : Reg) ω' => stmt (rowsOf st.c st.pub rt) ω') (R o) (ω o) = ofRows stmt st rt q (ω o) :=
  fun _ _ _ _ => rfl

/-- **ReadsCover by `rfl`**: two lists of commit strings that agree wherever a session reads give one statement,
whatever the reads. -/
theorem readsCover_ofRows {Ω H : Type} (stmt : (Nat → ByteArray) → Ω → H) (st : Flock.Stmt) (rt : Port)
    (reads : Nat → Prop) :
    ∀ q q' : Nat → ByteArray, (∀ i, reads i → q i = q' i) → ∀ ω, ofRows stmt st rt q ω = ofRows stmt st rt q' ω :=
  fun _ _ _ _ => rfl

/-- **The developer's statement at an accepted inner statement**: wherever `st`'s description is accepted against `rt`
and the developer's strings `p` open `rt` at each of its rows, `ofRows stmt st rt` at any strings is `stmt p`, or
SHA-512 collides at some row. -/
theorem ofRows_eq_or_collides {Ω H : Type} {rt : Port} {st : Flock.Stmt} (h : AcceptsDescription rt st)
    {stmt : (Nat → ByteArray) → Ω → H} (hst : ReadsBelow rt.leaves stmt) (p path₀ : Nat → ByteArray)
    (hp : ∀ r < rt.leaves, opens rt (p r) r (path₀ r) = true) :
    (∀ q ω, ofRows stmt st rt q ω = stmt p ω) ∨
      ∃ r < rt.leaves, rowsOf st.c st.pub rt r ≠ p r ∧
        Collides (bcPair rt r (rowsOf st.c st.pub rt r) (pathOf st.pub rt r) (p r) (path₀ r)) :=
  (rows_eq_or_collides h hst p path₀ hp).imp (fun he _ ω => he ω) id

end FlockSoundness.Discharge.Recursion
