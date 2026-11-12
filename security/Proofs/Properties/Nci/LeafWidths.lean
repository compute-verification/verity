module

public import Proofs.Properties.Nci.BodiesOk.Decode

/-!
# A leaf read at its width, reference by reference

The decoder refuses a body one of whose references names a leaf of another width than the type at its position
gives it (`Flock.Program.Widths.bodyBad`). `BodyWide` says what that leaves of a body: every reference of a node's
argument, and of the return, names a leaf as wide as the leaf of the type it is read as (`SeqWide`), among the
sources the check reads (`srcOf`: the parameters, and the nodes before the one reading; every node, for the return).
The check decides a strided part by a closed form over its pattern, which says something only where the pattern has
a positive length (`PartInner`, `BodyInner`: the decoder's invariant's own, named here too), as the parser leaves
every strided part. `tyOf` is the type a reference of an activation's body names.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-- The leaf sources of body `B` as `Widths.bodyBad` reads them: its parameters' types, and its nodes' outputs below
node `below`. -/
def srcOf (B : Body) (below : ℕ) : Widths.Src :=
  { params := B.params.map (·.2), nodes := B.nodes.map (·.out), hp := B.params.map (Widths.homo ·.2),
    hn := (B.nodes.map (·.out)).map Widths.homo, below }

/-- The references `seq`, read as a `t`, name leaves as wide as `t`'s: reference `j` names a leaf of the width of
`t`'s leaf `j`, wherever it names a source of `s`. -/
def SeqWide (s : Widths.Src) (t : Ty) (seq : Seq) : Prop :=
  ∀ j < seq.count, ∀ S, s.ty? (seq.get j).p (seq.get j).i = some S → Widths.wAt S (seq.get j).j = Widths.wAt t j

/-- **Every reference of body `B` names a leaf as wide as its position**: each argument of node `k`, read as the
argument's type among the parameters and the nodes before `k`, and the return, read as the return type among the
parameters and every node. -/
def BodyWide (B : Body) : Prop :=
  (∀ k < B.nodes.size, ∀ a < (B.nodes[k]!).args.size,
      SeqWide (srcOf B k) ((B.nodes[k]!).args[a]!).ty ((B.nodes[k]!).args[a]!).seq) ∧
    SeqWide (srcOf B B.nodes.size) B.retType B.ret

export NciCheck.Properties.SecurityProofs.BodiesOk (PartInner BodyInner)

/-- The type reference `r` names in the body of activation `sc`: a parameter's, or a node's output. -/
def tyOf (P : Prog) (sc : Scope) (r : Ref) : Ty :=
  if r.p then ((P.defs[sc.d]!).body.params[r.i]!).2 else ((P.defs[sc.d]!).body.nodes[r.i]!).out

end NciCheck.Properties.SecurityProofs.LeafWidths
