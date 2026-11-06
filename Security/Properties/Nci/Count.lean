module

public import Definitions.Nci
public import NciCheck.Count
public import Properties.Nci

/-!
# The count of `(c, k)`-inference blocks: what an accepted count means

`NciCheck.Count.evaluate` (`verity/core/protocols/network/nci/lean/NciCheck/Count.lean`, package `nci_check`) counts the
blocks of a declaration's word-level Program: each activation of a block Definition that no other block holds, with its
Boolean gates from the gate table and its fresh bits from `NciCheck.Desc`'s summaries. `DescCountsCK` states what an
accepted count means for the Boolean declaration the Program lowers to: it complies with `Nci.CompliesAtMost c k n`,
at most `n` proof blocks of a `(c, k)`-partition (`Verity.Partition.IsCK`).

The Boolean declaration is related to the Program by `Lowering`: each word gate `g` of the canonical layout is a set
`L g` of the Boolean circuit's gates, the sets covering it once each; an input word gate's are input gates, at most its
width of them, and all of them its output bits; a computed word gate's are computed gates, as many as the gate table
gives its primitive, of which at most its width are output bits; and a gate of `L g` reads only gates of `L g` and the
output bits of `g`'s operands. θ's bits are the gates of the word gates of θ's root nodes (`InTheta`). The count's
instance is read off its JSON (`CountParses`), never off the check's state.

`DescCountsCK` is stated and not yet proved: the lock lists it as an open piece (`shape.frozen`, `unstated`). Its proof
needs `Verity.Partition.IsCK` to bound a proof block's gates by `c` (`≤ c`): on a block of fewer than `c` gates, an
`IsCK` that asks for exactly `c` doesn't hold of the blocks the count names, and the statement doesn't hold as written.
It also needs `NciCheck.Desc`'s summaries to cover the parameter leaves a Definition's gates read (`ReadsExact`, which
`NciCheck.DescFuzz` decides on generated Programs), and the decoder's widths (`E_REINTERPRET`): a leaf a block reads is
as wide as the gate it names.
-/

@[expose] public section

namespace NciCheck.Properties

open Lean (Json toJson)
open Flock.Program (Prog)

/-- `j`'s fields are the count instance `i`'s: the program's text, θ as an array, the blocks as an array of their keys,
the gate table an object whose entries are `i.gatesOf`'s, and `c`, `k` and `n` naturals. -/
def CountParses (j : Json) (i : Count.Instance) : Prop :=
  j.getObjVal? "program" = .ok (.str i.program) ∧ j.getObjVal? "theta" = .ok (.arr i.theta) ∧
    j.getObjVal? "blocks" = .ok (.arr (i.blocks.map .str)) ∧
    (∃ t, j.getObjVal? "table" = .ok t ∧
      ∀ key g, i.gatesOf key = some g ↔ ∃ v, t.getObjVal? key = .ok v ∧ natOf? v = some g) ∧
    (∃ v, j.getObjVal? "c" = .ok v ∧ natOf? v = some i.c) ∧ (∃ v, j.getObjVal? "k" = .ok v ∧ natOf? v = some i.k) ∧
    ∃ v, j.getObjVal? "n" = .ok v ∧ natOf? v = some i.n

/-- Word gate `g` lies in a root node that an entry of `theta` names: between the root body's offsets of that node and
the next. -/
def InTheta (P : Prog) (theta : Array Json) (g : ℕ) : Prop :=
  ∃ t ∈ theta, ∃ k, natOf? t = some k ∧
    ((P.defs[P.root]!).body.offsets[k]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[k + 1]!)

/-- **The Boolean circuit `D` lowers the Program `P` by the gate table `gatesOf`**: word gate `g` of the canonical layout
is the set `L g` of `D`'s gates, of which `out g` are its output bits, the bits a gate outside `L g` may read. -/
structure Lowering (P : Prog) (gatesOf : String → Option ℕ) (D : Verity.Circuit) where
  /-- word gate `g`'s Boolean gates -/
  L : ℕ → Finset (Fin D.N)
  /-- its output bits -/
  out : ℕ → Finset (Fin D.N)
  /-- every Boolean gate lowers exactly one word gate -/
  cover : ∀ b : Fin D.N, ∃! g, g < gates P ∧ b ∈ L g
  /-- each word gate is located (primitive `p`, operands `ops`, width `w`), at most `w` of its Boolean gates are output
  bits, an input word gate's are input gates and all output bits, a computed word gate's are computed gates as many as
  the table gives `p`, and they read only gates of `L g` and the output bits of `g`'s operands -/
  located : ∀ g < gates P, ∃ p ops sc w, P.gate g = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value w ∧
    out g ⊆ L g ∧ (out g).card ≤ w ∧
    (if Flock.Program.isInputId (P.defs[p]!).key then (∀ b ∈ L g, D.isInput b = true) ∧ out g = L g
     else (∀ b ∈ L g, D.isInput b = false) ∧ gatesOf (P.defs[p]!).key = some (L g).card) ∧
    ∀ b ∈ L g, ∀ a ∈ D.reads b, a ∈ L g ∨ ∃ o ∈ ops, a ∈ out o

/-- When the count `out` on `j` has `complies` `true`, every Boolean declaration that lowers the Program by the instance's
gate table, with θ the bits of θ's root nodes, has at most `n` `(c, k)`-inference blocks. -/
def CountWitnessed (j out : Json) : Prop :=
  ∀ (i : Count.Instance) (P : Prog) (d : Verity.RegisteredCircuit) (lw : Lowering P i.gatesOf d.D),
    CountParses j i → Flock.Program.decode i.program.toUTF8 = .ok P → d.D.Boolean →
    (∀ b, b ∈ d.θ ↔ ∃ g, b ∈ lw.L g ∧ InTheta P i.theta g) →
    out.getObjVal? "complies" = .ok (.bool true) → Nci.CompliesAtMost i.c i.k i.n d

/-- **An accepted count witnesses at most `n` `(c, k)`-inference blocks.** On every instance the count gives a verdict
on, the verdict is `CountWitnessed`. -/
def DescCountsCK : Prop :=
  ∀ (j out : Json), Count.evaluate j = .ok out → CountWitnessed j out

end NciCheck.Properties
