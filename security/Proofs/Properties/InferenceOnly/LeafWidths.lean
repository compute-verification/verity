module

public import Proofs.Properties.InferenceOnly.BodiesOk.Decode

/-! `BodyWide`, what the decoder's width check leaves of a body: every reference names a leaf as wide as the type it is
read as. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

def srcOf (B : Body) (below : ℕ) : Widths.Src :=
  { params := B.params.map (·.2), nodes := B.nodes.map (·.out), hp := B.params.map (Widths.homo ·.2),
    hn := (B.nodes.map (·.out)).map Widths.homo, below }

def SeqWide (s : Widths.Src) (t : Ty) (seq : Seq) : Prop :=
  ∀ j < seq.count, ∀ S, s.ty? (seq.get j).p (seq.get j).i = some S → Widths.wAt S (seq.get j).j = Widths.wAt t j

def BodyWide (B : Body) : Prop :=
  (∀ k < B.nodes.size, ∀ a < (B.nodes[k]!).args.size,
      SeqWide (srcOf B k) ((B.nodes[k]!).args[a]!).ty ((B.nodes[k]!).args[a]!).seq) ∧
    SeqWide (srcOf B B.nodes.size) B.retType B.ret

export InferenceOnlyCheck.Properties.SecurityProofs.BodiesOk (PartInner BodyInner)

def tyOf (P : Prog) (sc : Scope) (r : Ref) : Ty :=
  if r.p then ((P.defs[sc.d]!).body.params[r.i]!).2 else ((P.defs[sc.d]!).body.nodes[r.i]!).out

end InferenceOnlyCheck.Properties.SecurityProofs.LeafWidths
