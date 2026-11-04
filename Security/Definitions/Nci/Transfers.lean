import Definitions.Core.Partition
import Definitions.Nci.Covering

/-!
# Transfer gates: what one machine sends another

The whole computation, clients, the auditor's warden and the servers alike, is one circuit `C`, and a division of it
into machines is a `Verity.Partition C n`: machine `m` computes the gates `R_m = P.gates m`, every computed gate on
exactly one machine, and the input gates of `C` on none. A gate of `R_a` that a gate of `R_b` reads, for machines
`a ≠ b`, is a **transfer gate** from `a` to `b` (`transfers`), in NCI's words a gate of `out(R_a) ∩ in(R_b)`: `a`
computes it and `b` takes it as an input. Everything else `R_b` reads from outside itself, the rest of `in(R_b)`, is an
input gate of `C` (`inputsRead`), since a gate outside every machine is an input gate. So
`in(R_b) = transfersInto b ∪ inputsRead b`, two disjoint sets: what reaches `b` over the wire, and the inputs it reads.
A transfer gate's value reaches `b` once, in `w g` bits, so `width (transfersInto b)` is the bits on the wire into `b`,
and in the Boolean profile (`w = 1`, `FlockSoundness.Audit.Circuit.toCircuit`) `|transfersInto b|` is.

The machines are a partition rather than a family of `Covering.Subcircuit`s for two reasons. A machine's gates are what
compute credit counts, each computed gate credited to the one machine that computes it, so the credits add up to
`size(C)`. And a partition is `Complies`'s division into IUs, so when each machine (one whole-node session) is one IU, the
machines are that division. Subcircuits may overlap and recompute, so neither holds of them. Each machine is still a
subcircuit (`machine`): `R_m ∪ in(R_m)`, with inputs `in(R_m)` and the gates of `R_m` that have no predecessors, which a
subcircuit holds as inputs (`Covering.sources`). In the subcircuits' terms, the transfer gates from `a` to `b` are
`(machine a).inner ∩ (machine b).ins`, apart from one kind of gate. A gate of `R_a` with no predecessors (a constant:
a computed gate that reads nothing) is an input of `machine a`, not an inner gate, but `transfers` counts it, since `b`
reads it from `a` and `novelReads` charges `b` for it.
-/

namespace Nci

open Verity Finset

variable {C : Circuit} {n : ℕ}

/-- **The transfer gates from machine `a` to machine `b`**: the gates of `R_a` that gates of `R_b` read,
`in(R_b) ∩ R_a`, which for `a ≠ b` is `out(R_a) ∩ in(R_b)`, and empty for `a = b`. `a` computes them and `b` takes them
as inputs; in the Boolean profile, `|transfers P a b|` is the bits `a` sends `b`. -/
def transfers (P : Partition C n) (a b : Fin n) : Finset (Fin C.N) := C.into (P.gates b) ∩ P.gates a

/-- **What machine `b` receives over the wire**: the transfer gates into `b` from every other machine,
`⋃_{a ≠ b} transfers a b`, the gates of `in(R_b)` that another machine computes. Their width is the bits on the wire
into `b`, and in the Boolean profile their number. -/
def transfersInto (P : Partition C n) (b : Fin n) : Finset (Fin C.N) :=
  (univ.filter (· ≠ b)).biUnion fun a => transfers P a b

/-- **The input gates of `C` that machine `b` reads**: the gates of `in(R_b)` that no machine computes. Those in the
baseline `θ` are free; the rest, such as a prompt `b` reads directly, are novel. -/
def inputsRead (P : Partition C n) (b : Fin n) : Finset (Fin C.N) :=
  (C.into (P.gates b)).filter fun g => C.isInput g = true

/-- **Machine `b` as a subcircuit** of `C`: its gates and what they read, `R_b ∪ in(R_b)`, with inputs `in(R_b)` and
the gates of `R_b` that have no predecessors. -/
def machine (P : Partition C n) (b : Fin n) : Covering.Subcircuit C where
  gates := P.gates b ∪ C.into (P.gates b)
  ins := C.into (P.gates b) ∪ (P.gates b ∩ Covering.sources C)
  ins_sub := union_subset subset_union_right (inter_subset_left.trans subset_union_left)
  closed g hg hin := by
    have hgb : g ∈ P.gates b := by
      rcases mem_union.1 hg with h | h
      · exact h
      · exact absurd (mem_union_left _ h) hin
    intro x hx
    by_cases hxb : x ∈ P.gates b
    · exact mem_union_left _ hxb
    · exact mem_union_right _ (mem_filter.2 ⟨mem_univ _, hxb, g, hgb, hx⟩)
  sources_ins g hg hs := by
    rcases mem_union.1 hg with h | h
    · exact mem_union_right _ (mem_inter.2 ⟨h, hs⟩)
    · exact mem_union_left _ h

end Nci
