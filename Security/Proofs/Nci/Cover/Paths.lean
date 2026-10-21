import Models.Nci.Guarantees.Cover

/-!
# The cover theorem: paths into a block

`Entry` walks a path back from its end in `I`: a gate `I` computes reads only gates of `I`, so the walk stays in `I`
until it meets an input of `I`, or reaches the path's first gate, which `I` holds and does not compute, an input.
`BackwardClosure` is `Entry` on a path inside `A`, which meets no input of `I`. `reachesIn_output` follows readers
forward from any gate of a set with one output gate: indices increase, so the walk ends, at a gate no gate of the set
reads, the output.
-/

namespace Nci.Cover.SecurityProofs

theorem Entry : Guarantees.Entry := by
  sorry

theorem BackwardClosure : Guarantees.BackwardClosure := by
  sorry

/-- Every gate of a set `I` whose only output gate is `o` reaches `o` inside `I`. -/
theorem reachesIn_output (C : Circuit) (I : Finset (Fin C.N)) (o : Fin C.N) (ho : C.outputs I = {o}) :
    ∀ x ∈ I, C.ReachesIn I x o := by
  sorry

end Nci.Cover.SecurityProofs
