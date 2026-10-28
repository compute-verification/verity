import Definitions.Core.Compiler

/-!
# The registered circuit

Setup fixes the registered compiler `B` (`Verity.Compiler`) with its advice cap `ℓa`, and at the start of each
verification phase the developer reports its advice `a_j`, at most `ℓa` bits. **The registered circuit** (`Registered`)
is the compiler with the advice list: its circuit `D`, its registered inputs `θ`, its partition `𝓘` into inference
blocks and its partition `P` into proof blocks, nested in `𝓘`, are the compiler's output on the advice,
`D = B.init ‖ B(a_1) ‖ … ‖ B(a_n)` with the blocks' partitions side by side (`Compiler.run`). `θ` are the input gates
whose values the developer committed beforehand (its models, code and data in storage), the gates of `B.init`. The values
of every other gate are fresh input: the remaining input gates hold data such as prompts, and every computed gate is
computed after the commitment. Per deployment, the workload's Program is the one circuit of record.

A circuit derived from the registered one (its bit-level expansion, a padding, a cut) is registered by the one-block
compiler of what it derives (`Registered.ofPartition`, `Registered.ofCircuit`), so a statement about it reads the same
`D`, `θ` and `𝓘`.

Implementation carries over to registered circuits when it keeps fresh input fresh: an input gate of `t` outside `t.θ` is
an input gate of `d` outside `d.θ`. Containment carries over as it is between the circuits, and says nothing about `θ`:
that a containment keeps fresh input fresh is a premise about the protocol's rule for `θ`
(`Nci.Assumptions.ThetaRegistered`), which each guarantee that needs it takes for its containment.
-/

namespace Verity

/-- **The registered circuit**: the registered compiler `B` and the advice `a = a_1, …, a_n` reported at the start of
each verification phase, each of at most `B.ℓa` bits. Its circuit, registered inputs, inference blocks and proof blocks
are `B`'s output on `a` (`out`). -/
structure Registered where
  B : Compiler
  a : List (List Bool)
  a_le : ∀ x ∈ a, x.length ≤ B.ℓa

namespace Registered

variable (d : Registered)

/-- The compiler's output on the advice: `B.init ‖ B(a_1) ‖ … ‖ B(a_n)`. -/
@[reducible] def out : Compiler.Output := d.B.run d.a

/-- The registered circuit `D`. -/
@[reducible] def D : Circuit := d.out.D

/-- The registered inputs `θ`, a set of `D`'s input gates. -/
@[reducible] def θ : Finset (Fin d.D.N) := d.out.θ

theorem θ_input : ∀ g ∈ d.θ, d.D.isInput g = true := d.out.θ_input

/-- The number of inference blocks. -/
@[reducible] def n : ℕ := d.out.n

/-- **`𝓘`, the partition into inference blocks**, the compiler's. -/
@[reducible] def 𝓘 : Partition d.D d.n := d.out.𝓘

/-- The number of proof blocks. -/
@[reducible] def m : ℕ := d.out.m

/-- **`P`, the partition into proof blocks**, the compiler's, each inside one inference block (`refines`). -/
@[reducible] def P : Partition d.D d.m := d.out.P

/-- Each proof block lies inside one inference block. -/
@[reducible] def refines : d.P.Refines d.𝓘 := d.out.refines

/-- Each gate's position in the committed trace, the compiler's. -/
@[reducible] def pos : Fin d.D.N → Option ℕ := d.out.pos

/-- The compiler's output `o`, registered by its one-block compiler with no advice. -/
@[reducible] def ofOutput (o : Compiler.Output) : Registered := ⟨.const o, [], by simp⟩

/-- The circuit `C` with registered inputs `θ` and inference blocks `I`, each its own proof block, registered by its
one-block compiler with no advice and no positions in a trace. -/
@[reducible] def ofPartition (C : Circuit) (θ : Finset (Fin C.N)) (h : ∀ g ∈ θ, C.isInput g = true) (n : ℕ)
    (I : Partition C n) : Registered :=
  ofOutput ⟨C, θ, h, n, I, n, I, ⟨id, fun g => by simp⟩, fun _ => none⟩

/-- The circuit `C` with registered inputs `θ`, each computed gate its own block, registered by its one-block compiler
with no advice. -/
@[reducible] def ofCircuit (C : Circuit) (θ : Finset (Fin C.N)) (h : ∀ g ∈ θ, C.isInput g = true) : Registered :=
  ofPartition C θ h C.N (.singletons C)

/-- `t ≼ d`: `d`'s circuit implements `t`'s, through an implementation that sends no input gate of `t` outside `t.θ`
into `d.θ`. -/
def Implements (t d : Registered) : Prop :=
  ∃ I : Implementation t.D d.D, ∀ g, t.D.isInput g = true → I.inp g ∈ d.θ → g ∈ t.θ

/-- `d` contains `t`: `d`'s circuit contains `t`'s. -/
def Contains (t d : Registered) : Prop := Nonempty (Containment t.D d.D)

end Registered

end Verity
