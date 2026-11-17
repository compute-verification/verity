module

public import Definitions.InferenceOnly
public import InferenceOnlyCheck.Count
public import Properties.InferenceOnly

/-! What an accepted count of `(c, k)`-inference blocks means; `DescCountsCK` is stated and not yet proved. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties

open Lean (Json toJson)
open Flock.Program (Prog)

def CountParses (j : Json) (i : Count.Instance) : Prop :=
  j.getObjVal? "program" = .ok (.str i.program) ∧ j.getObjVal? "theta" = .ok (.arr i.theta) ∧
    j.getObjVal? "blocks" = .ok (.arr (i.blocks.map .str)) ∧
    (∃ t, j.getObjVal? "table" = .ok t ∧
      ∀ key g, i.gatesOf key = some g ↔ ∃ v, t.getObjVal? key = .ok v ∧ natOf? v = some g) ∧
    (∃ v, j.getObjVal? "c" = .ok v ∧ natOf? v = some i.c) ∧ (∃ v, j.getObjVal? "k" = .ok v ∧ natOf? v = some i.k) ∧
    ∃ v, j.getObjVal? "n" = .ok v ∧ natOf? v = some i.n

def InTheta (P : Prog) (theta : Array Json) (g : ℕ) : Prop :=
  ∃ t ∈ theta, ∃ k, natOf? t = some k ∧
    ((P.defs[P.root]!).body.offsets[k]! ≤ g ∧ g < (P.defs[P.root]!).body.offsets[k + 1]!)

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

def CountWitnessed (j : Json) : Prop :=
  ∀ (i : Count.Instance) (P : Prog) (d : Verity.RegisteredCircuit) (lw : Lowering P i.gatesOf d.D),
    CountParses j i → Flock.Program.decode i.program.toUTF8 = .ok P → d.D.Boolean →
    (∀ b, b ∈ d.θ ↔ ∃ g, b ∈ lw.L g ∧ InTheta P i.theta g) → InferenceOnly.CompliesAtMost i.c i.k i.n d

def DescCountsCK : Prop :=
  ∀ j : Json, (Count.check j).getObjVal? "complies" = .ok (.bool true) → CountWitnessed j

end InferenceOnlyCheck.Properties
