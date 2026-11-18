module

public import Flock.Program

/-!
# What `decode` guarantees of the Boolean map (PROTOCOL.md §4.9)

On a descriptor `decode` accepts, every word primitive (`isWordPrim`: a primitive that is no gate, constant bit or input)
has one Boolean Definition, `P.bodyOf?` of its key, and the map names nothing else:

* the body is **Boolean** (`Prog.BoolDef`): a §9 primitive, or a composite each of whose nodes calls a §9 primitive or a
  Boolean composite placed before it, as `checkBody` reads `reach`;
* a composite body's parameters and return are bits (`Ty.bits`);
* the body has the word primitive's **bit signature**: its parameters' types are the word's, each read as bits
  (`Ty.bitType`), and so is its return;
* every key of `P.boolean` is a word primitive's, at its place in `P.defs`.

Which refusal a descriptor gets first is the difference test's against `codec._decode_boolean`, not stated here.
-/

@[expose] public section

namespace Flock.Program

/-- Definition `d` of `P` is Boolean (§9): a primitive that is a gate, a constant bit or a 1-bit input, or a composite
each of whose nodes calls such a primitive or a Boolean composite placed before it. -/
inductive Prog.BoolDef (P : Prog) : Nat → Prop
  | prim {d : Nat} : (P.defs[d]!).prim = true → isBooleanPrim (P.defs[d]!).key = true → Prog.BoolDef P d
  | comp {d : Nat} : (P.defs[d]!).prim = false →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = true → isBooleanPrim (P.defs[nd.fn]!).key = true) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → nd.fn < d) →
      (∀ nd ∈ (P.defs[d]!).body.nodes, (P.defs[nd.fn]!).prim = false → Prog.BoolDef P nd.fn) →
      Prog.BoolDef P d

/-- Definition `b` of `P` is a Boolean Definition of word primitive `w`'s bit signature (`checkBody`). -/
def Prog.BodyOf (P : Prog) (w b : Nat) : Prop :=
  Prog.BoolDef P b ∧
    ((P.defs[b]!).prim = false → (P.defs[b]!).params.all (·.2.bits) = true ∧ (P.defs[b]!).ret.bits = true) ∧
    (P.defs[b]!).params.map (·.2) = (P.defs[w]!).params.map (·.2.bitType) ∧
    (P.defs[b]!).ret = (P.defs[w]!).ret.bitType

/-- **`decode`'s Boolean map (§4.9).** On an accepted descriptor, each word primitive's key leads (`P.bodyOf?`) to a
Definition of `P` that is a Boolean Definition of its bit signature, and every key of the map is a word primitive's. -/
def DecodeBoolean : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), decode bytes = .ok P →
    (∀ w < P.defs.size, isWordPrim (P.defs[w]!).prim (P.defs[w]!).key = true →
      ∃ b < P.defs.size, P.bodyOf? (P.defs[w]!).key = some b ∧ P.BodyOf w b) ∧
    (∀ k v, P.boolean[k]? = some v →
      ∃ w < P.defs.size, P.index[k]? = some w ∧ (P.defs[w]!).key = k ∧ isWordPrim (P.defs[w]!).prim k = true)

end Flock.Program
