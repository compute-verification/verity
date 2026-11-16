module

public import Definitions.Slowdown.Cut

/-!
# Untied words: the vocabulary

A word is **untied** when nothing ties its value to registered data any more. `RegisteredCircuit.untie` is the circuit
untied at a set of gates: the same circuit and blocks, with those gates taken out of `θ`, so a block that reads one of
them reads it as fresh input. `wordsRead` is the words of a set that a block reads, `OneReader` says each word is read
by at most one block, and `UntiedWords` puts together what the reads bound counts: words of at most `z` gates, each read by
one block, at most `w` of them read by each block of a set. `Proofs.Slowdown.Untied` proves what a block reads once a
word is untied.
-/

@[expose] public section

namespace Verity

open Finset

/-- `d` **untied at `E`**: `d`'s circuit, inference blocks, proof blocks and positions, with `E`'s gates taken out of
`θ`. -/
@[reducible] def RegisteredCircuit.untie (d : RegisteredCircuit) (E : Finset (Fin d.D.N)) : RegisteredCircuit :=
  .ofOutput ⟨d.D, d.θ \ E, fun g hg => d.θ_input g (mem_sdiff.1 hg).1, d.n, d.𝓘, d.m, d.P, d.refines, d.pos⟩

end Verity

namespace Slowdown.Cut

open Verity Finset

/-- The words of `𝒲` that `R` reads: those with a gate in `in(R)`. -/
def wordsRead (C : Circuit) (𝒲 : Finset (Finset (Fin C.N))) (R : Finset (Fin C.N)) : Finset (Finset (Fin C.N)) :=
  𝒲.filter fun W => ¬ Disjoint W (C.into R)

/-- Each word of `𝒲` is read by at most one block of `P`. -/
def OneReader {C : Circuit} {m : ℕ} (P : Partition C m) (𝒲 : Finset (Finset (Fin C.N))) : Prop :=
  ∀ W ∈ 𝒲, ∀ u v, ¬ Disjoint W (C.into (P.gates u)) → ¬ Disjoint W (C.into (P.gates v)) → u = v

/-- **The untied words, as the reads bound counts them**: each word of `𝒲` has at most `z` gates, each is read by at
most one block of `P` (`OneReader`), and each block of `G` reads at most `w` of them. -/
def UntiedWords {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (G : Finset (Fin m))
    (𝒲 : Finset (Finset (Fin d.D.N))) (z w : ℕ) : Prop :=
  (∀ W ∈ 𝒲, W.card ≤ z) ∧ OneReader P 𝒲 ∧ ∀ u ∈ G, (wordsRead d.D 𝒲 (P.gates u)).card ≤ w

end Slowdown.Cut
