import Definitions.Core.Site

/-!
# The registered circuit

Setup registers θ's values and the builder `B` in the registration (`Verity.Registration`, its information entry), and
`B` names the registered compiler (`Verity.Compiler`) with its advice cap `ℓa`. In each execution phase the developer
reports its advice `a_j`, at most `ℓa` bits, and at the start of the verification phase after it `B` runs on it.
**The registered circuit** (`RegisteredCircuit`) is the registration with the advice list: its circuit `D`, its
registered inputs `θ`, its partition `𝓘` into inference blocks and its partition `P` into proof blocks, nested in `𝓘`,
are the output of the log's compiler on the log's θ and the advice, `D = B.init θ ‖ B(a_1) ‖ … ‖ B(a_n)` with the
blocks' partitions side by side (`Compiler.run`). `θ` are the input gates whose values the developer committed
beforehand (its models, code and data in storage), the gates `B.init θ` registers. The values of every other gate are
fresh input: the remaining input gates hold data such as prompts, and every computed gate is computed after the
commitment. Per deployment, the workload's Program is the one circuit of record.

A circuit derived from the registered one (its bit-level expansion, a padding, a cut) is registered by the one-entry log
of the one-block compiler of what it derives (`RegisteredCircuit.ofPartition`, `RegisteredCircuit.ofCircuit`), so a
statement about it reads the same `D`, `θ` and `𝓘`.

Implementation carries over to registered circuits when it keeps fresh input fresh: an input gate of `t` outside `t.θ` is
an input gate of `d` outside `d.θ`. Containment carries over as it is between the circuits, and says nothing about `θ`:
that a containment keeps fresh input fresh is a premise about the protocol's rule for `θ`
(`Nci.Assumptions.ThetaRegistered`), which each guarantee that needs it takes for its containment.
-/

namespace Verity

/-- **The registered circuit**: the registration `log`, its builder `b` (`Registration.builder?`), and the advice
`a = a_1, …, a_n` reported in the execution phases, each of at most `B.ℓa` bits for the compiler `B` the builder names.
Its circuit, registered inputs, inference blocks and proof blocks are `B`'s output on the log's θ and `a` (`out`). -/
structure RegisteredCircuit where
  log : Registration
  b : Registration.Builder
  builder : log.builder? = some b
  a : List (List Bool)
  a_le : ∀ x ∈ a, x.length ≤ b.B.ℓa

namespace RegisteredCircuit

variable (d : RegisteredCircuit)

/-- The registered compiler, the one the log's builder names. -/
@[reducible] def B : Compiler := d.b.B

/-- The compiler's output on the log's θ and the advice: `B.init θ ‖ B(a_1) ‖ … ‖ B(a_n)`. -/
@[reducible] def out : Compiler.Output := d.B.run d.log.theta d.a

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

/-- The builder of the one-block compiler of `o`, with no digest or advice format. -/
@[reducible] def constBuilder (o : Compiler.Output) : Registration.Builder := ⟨"", "", .const o⟩

/-- The compiler's output `o`, registered by the one-entry log of its one-block compiler, with no advice. -/
@[reducible] def ofOutput (o : Compiler.Output) : RegisteredCircuit :=
  ⟨[.information "" [.builder (constBuilder o)]], constBuilder o, rfl, [], by simp⟩

/-- The circuit `C` with registered inputs `θ` and inference blocks `I`, each its own proof block, registered by its
one-block compiler with no advice and no positions in a trace. -/
@[reducible] def ofPartition (C : Circuit) (θ : Finset (Fin C.N)) (h : ∀ g ∈ θ, C.isInput g = true) (n : ℕ)
    (I : Partition C n) : RegisteredCircuit :=
  ofOutput ⟨C, θ, h, n, I, n, I, ⟨id, fun g => by simp⟩, fun _ => none⟩

/-- The circuit `C` with registered inputs `θ`, each computed gate its own block, registered by its one-block compiler
with no advice. -/
@[reducible] def ofCircuit (C : Circuit) (θ : Finset (Fin C.N)) (h : ∀ g ∈ θ, C.isInput g = true) : RegisteredCircuit :=
  ofPartition C θ h C.N (.singletons C)

/-- `t ≼ d`: `d`'s circuit implements `t`'s, through an implementation that sends no input gate of `t` outside `t.θ`
into `d.θ`. -/
def Implements (t d : RegisteredCircuit) : Prop :=
  ∃ I : Implementation t.D d.D, ∀ g, t.D.isInput g = true → I.inp g ∈ d.θ → g ∈ t.θ

/-- `d` contains `t`: `d`'s circuit contains `t`'s. -/
def Contains (t d : RegisteredCircuit) : Prop := Nonempty (Containment t.D d.D)

end RegisteredCircuit

end Verity
