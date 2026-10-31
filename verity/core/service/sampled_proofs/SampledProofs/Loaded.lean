import SampledProofs.Program
import SampledProofs.Json

/-!
# The verifier's programs, loaded once per process

`sampled-proofs-verifier` answers a whole session's calls in one process, and they read one program or a few: a window's
population program, and a negative's. The process loads each program once (`Store.load`, on the descriptor a piece
would read), and a call names a loaded program by its digest and annotations, `{"annotations": A, "loaded": HEX}`
(`ref?`), where it would send the descriptor. Where a piece reads its program (`Store.parse`) it reads the loaded one,
so a call neither sends the program nor parses it again, and every piece reads the program it read before:

* `parse_empty`: with nothing loaded, `Store.parse` is the parse every piece made (`parseDescriptor`: `Program.parse` of
  the descriptor's canonical bytes), so each piece at the empty store is the piece as it was (`Pieces.verifier` is
  `Pieces.verifierAt #[]`), and a reference to a program not loaded is refused as the descriptor it isn't;
* `parse_ref`: in a store `Store.load` built (`valid_empty`, `load_valid`), the program a reference reads is
  `parseDescriptor` of a descriptor whose digest (`Program.sha512`, the SHA-512 of its canonical bytes without
  `annotations`) and canonical annotations are the reference's: under SHA-512's collision resistance, the descriptor
  the caller holds, which it would have sent.
-/

namespace SampledProofs.Loaded

open Lean (Json)
open Flock (V)

/-- A loaded program: its digest (`Program.sha512`) in hex, its descriptor's annotations in canonical JSON, and the
program. -/
structure Entry where
  sha512 : String
  annotations : String
  program : Program
  deriving Inhabited

/-- The programs one verifier process has loaded. -/
abbrev Store := Array Entry

/-- How a piece reads a program from its descriptor: `Program.parse` of the descriptor's canonical bytes. -/
def parseDescriptor (j : Json) : V Program := do Program.parse (← Flock.canon j).toUTF8

/-- A reference's digest and canonical annotations: an object of exactly `annotations` and `loaded`, a string. -/
def ref? (j : Json) : Option (String × String) :=
  if keys? j != some #["annotations", "loaded"] then none else
  match getD j "loaded", Flock.canon (getD j "annotations") with
  | .str h, .ok a => some (h, a)
  | _, _ => none

/-- The loaded program a reference names. -/
def Store.find (S : Store) (j : Json) : Option Program :=
  (ref? j).bind fun (h, a) => (S.find? fun e => e.sha512 == h && e.annotations == a).map (·.program)

/-- A piece's program from `j`: the loaded program `j` names, else `parseDescriptor j`. -/
def Store.parse (S : Store) (j : Json) : V Program :=
  match S.find j with
  | some p => pure p
  | none => parseDescriptor j

/-- `S` holding the program of the descriptor `j`, an object, as `parseDescriptor` reads it (refused where that refuses),
and its digest in hex; a program loaded already is held once. -/
def Store.load (S : Store) (j : Json) : V (Store × String) := do
  let .obj _ := j | throw "program is not an object"
  let p ← parseDescriptor j
  let e : Entry := { sha512 := Flock.hexOf p.sha512, annotations := ← Flock.canon (getD j "annotations"), program := p }
  return (if S.any (fun x => x.sha512 == e.sha512 && x.annotations == e.annotations) then S else S.push e, e.sha512)

/-- Every loaded program is the parse of a descriptor with its digest and annotations. -/
def Store.Valid (S : Store) : Prop :=
  ∀ e ∈ S, ∃ d, parseDescriptor d = .ok e.program ∧ Flock.hexOf e.program.sha512 = e.sha512 ∧
    Flock.canon (getD d "annotations") = .ok e.annotations

theorem valid_empty : Store.Valid #[] := by
  intro e he
  simp at he

theorem parse_empty (j : Json) : Store.parse #[] j = parseDescriptor j := by
  unfold Store.parse Store.find
  cases ref? j <;> rfl

theorem load_valid {S S' : Store} {j : Json} {h : String} (hS : S.Valid) (hl : S.load j = .ok (S', h)) :
    S'.Valid := by
  unfold Store.load at hl
  match j, hl with
  | .obj kvs, hl =>
    simp only [bind, Except.bind] at hl
    split at hl
    · cases hl
    · rename_i p hp
      split at hl
      · cases hl
      · rename_i a ha
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hl
        obtain ⟨rfl, -⟩ := hl
        split
        · exact hS
        · intro x hx
          rcases Array.mem_push.1 hx with hx | rfl
          · exact hS x hx
          · exact ⟨.obj kvs, hp, rfl, ha⟩

theorem parse_ref {S : Store} {j : Json} {h a : String} {p : Program} (hS : S.Valid) (hr : ref? j = some (h, a))
    (hp : S.find j = some p) :
    ∃ d, parseDescriptor d = .ok p ∧ Flock.hexOf p.sha512 = h ∧ Flock.canon (getD d "annotations") = .ok a := by
  unfold Store.find at hp
  rw [hr] at hp
  simp only [Option.bind_some, Option.map_eq_some_iff] at hp
  obtain ⟨e, he, rfl⟩ := hp
  have hm := Array.mem_of_find?_eq_some he
  have hq := Array.find?_some he
  simp only [Bool.and_eq_true, beq_iff_eq] at hq
  obtain ⟨d, hd, hs, ha⟩ := hS e hm
  exact ⟨d, hd, hs.trans hq.1, ha.trans (congrArg _ hq.2)⟩

theorem parse_of_find {S : Store} {j : Json} {p : Program} (hp : S.find j = some p) : S.parse j = .ok p := by
  unfold Store.parse
  rw [hp]
  rfl

end SampledProofs.Loaded
