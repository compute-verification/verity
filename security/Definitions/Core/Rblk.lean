module

public import Definitions.Core.Partition
public import Definitions.Core.Hash

/-!
# The base argument's relation for a proof block

`Partition.Rblk` is what a proof block's base argument establishes: its in(S) and out(S) open under the string
commitment at their positions (`Partition.pos`), and out(S) holds what the block's circuit computes. It is the one
reader of the string commitment among the partition's definitions, so it sits apart from `Partition`, whose readers
do not take the hash's vocabulary.
-/

@[expose] public section

namespace Verity

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- **`Rblk`, the base argument's relation for proof block `u`**, with `S` its gates and `C_S = C.restrict S`: the trace
`X` opens, under the string commitment `Sc` at the root `c r` of each gate's position `pos L g = (r, i)`, at leaf `i`,
for each gate of in(S) and out(S), each value
encoded by `enc`, with opening `π g`; and out(S) holds what `C_S` computes with every gate outside `S` held at `X`. It
constrains openings at in(S) and out(S) only, and reads `X` only on `S` and in(S) (`Partition.rblk_congr`): a gate
outside `S` that `S` never reads is a free input of `C_S`, which no opening and no bound reads. -/
def Rblk {α : Type} (Sc : Hash.StringCommitment α) (H : Hash.Msg → Digest) (enc : (g : Fin C.N) → BitVec (C.w g) → α)
    (L : Layout C) (c : Root → Sc.Com) (u : Fin n) (X : C.Trace) (π : Fin C.N → Sc.Op) : Prop :=
  (∀ g ∈ C.into (P.gates u) ∪ C.exits (P.gates u),
      ∃ p, P.pos L g = some p ∧ Sc.Opens H (c p.1) p.2 (enc g (X g)) (π g)) ∧
    ∀ g ∈ C.exits (P.gates u),
      X g = (C.restrict (P.gates u)).run (fun a => if P.unit a = some u then none else some (X a)) g

end Partition

end Verity
