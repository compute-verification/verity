import Definitions.Accounting.Guarantees.ServingBlocks
import Definitions.Accounting.ServingBlocks.Instances

/-!
# The serving-block bound: honest and nontrivial lemmas

Each guarantee of `Accounting.Guarantees.ServingBlocks` has two statements here, at the instances of
`Security/Definitions/Accounting/ServingBlocks/Instances.lean`. `<Guarantee>_honest`: its premises hold together at
`Honest`. `<Guarantee>_nontrivial`: at an explicit circuit every premise but one named premise holds, that one fails, and
the counted weight is above the bound's right-hand side, so the guarantee is false without it. `ServingBlockBound` has
two more, one for each premise that bounds its boundary term: `ServingBlockBound_nontrivial_bodies` (`OperationBodies`)
and `ServingBlockBound_nontrivial_overlap` (`Overlap`, and so the boundary term itself).

`Honest` is a small faithful instance of a serving-block layout: a Boolean declaration with a registered input, two
serving blocks around one wipe, an operation split across the wipe and a dark remainder. It is accepted as the honest
instance of this structural bound, which is about one circuit: Layer 2's envelopes supply the deployed premises, each
with its probability.

The proofs are in `Security/Proofs/Accounting/ServingBlocks/Witnesses.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The serving-block bound's premises hold** at `Honest`, at `(c_S, k_S) = (c, k) = (1, 1)`, the serving blocks as
their own inference blocks, `ω_max = μ = 1`, `q_U(u) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBound_honest : Prop :=
  (∀ u ∈ Honest.S, (Honest.P.gates u).card ≤ 1 ∧ (Honest.d.unregistered (Honest.P.gates u)).card ≤ 1) ∧
  (∀ u ∈ Honest.S, PerBlockCap Honest.d Honest.J Honest.q 1 1) ∧ Honest.P.IsCK Honest.d 1 1 ∧
  PerBlockCap Honest.d Honest.J Honest.q 1 1 ∧ OperationBodies Honest.d Honest.J ∧ Overlap Honest.J 1 ∧
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧
  (∀ u ∈ Honest.S, UncertifiedCap Honest.J (Honest.P.gates u)
    (fun i : {i // Honest.r.parent i = u} => Honest.P.gates i) 1) ∧
  UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) 1

/-- **The serving-block bound is false without the per-block cap at the serving blocks' size**: at `Counter`, with
`(c_S, k_S) = (1, 0)`, `(c, k) = (1, 1)`, the serving block as its own inference block, `ω_max = μ = 1` and `q_U = 0`,
every other premise holds, the cap at `(1, 0)` fails, and the counted weight `1` is above the right-hand side `0`. -/
def ServingBlockBound_nontrivial : Prop :=
  (∀ u ∈ Counter.S, (Counter.P.gates u).card ≤ 1 ∧ (Counter.d.unregistered (Counter.P.gates u)).card ≤ 0) ∧
  ¬ (∀ u ∈ Counter.S, PerBlockCap Counter.d Counter.J Counter.q 1 0) ∧ Counter.P.IsCK Counter.d 1 1 ∧
  PerBlockCap Counter.d Counter.J Counter.q 1 1 ∧ OperationBodies Counter.d Counter.J ∧ Overlap Counter.J 1 ∧
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
  (∀ u ∈ Counter.S, UncertifiedCap Counter.J (Counter.P.gates u)
    (fun i : {i // Counter.r.parent i = u} => Counter.P.gates i) 0) ∧
  UncertifiedCap Counter.J ((univ \ Counter.S).biUnion Counter.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N)) 0 ∧
  ∑ u ∈ Counter.S, min (Counter.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ))
    (((univ.filter fun i => Counter.r.parent i = u).card : ℝ) * Counter.q 1 1 + 0) + 0 < Counter.J.total

/-- **The serving-block bound is false without operation bodies**: at `Detached`, whose operation takes in gate 1 though
no gate of it reads gate 1, with `(c_S, k_S) = (c, k) = (1, 0)`, the serving blocks as their own inference blocks,
`ω_max = μ = 1`, `q_U(u) = 1` and `q_U(L_dark) = 0`, every other premise holds and the counted weight `1` is above the
right-hand side `0`: the serving block after the wipe reads no unregistered bit, yet the operation is assigned to it. -/
def ServingBlockBound_nontrivial_bodies : Prop :=
  (∀ u ∈ Detached.S, (Detached.P.gates u).card ≤ 1 ∧ (Detached.d.unregistered (Detached.P.gates u)).card ≤ 0) ∧
  (∀ u ∈ Detached.S, PerBlockCap Detached.d Detached.J Detached.q 1 0) ∧ Detached.P.IsCK Detached.d 1 0 ∧
  PerBlockCap Detached.d Detached.J Detached.q 1 0 ∧ ¬ OperationBodies Detached.d Detached.J ∧
  Overlap Detached.J 1 ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
  (∀ u ∈ Detached.S, UncertifiedCap Detached.J (Detached.P.gates u)
    (fun i : {i // Detached.r.parent i = u} => Detached.P.gates i) 1) ∧
  UncertifiedCap Detached.J ((univ \ Detached.S).biUnion Detached.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Detached.d.D.N)) 0 ∧
  ∑ u ∈ Detached.S, min (Detached.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ))
    (((univ.filter fun i => Detached.r.parent i = u).card : ℝ) * Detached.q 1 0 + 1) + 0 < Detached.J.total

/-- **The serving-block bound is false without overlap, and so without its boundary term**: at `Straddle`, whose
operation is split across the wipe, with `(c_S, k_S) = (c, k) = (1, 1)`, the serving blocks as their own inference
blocks, `ω_max = 1`, `μ = 0`, `q_U(u) = 1` and `q_U(L_dark) = 0`, every other premise holds, no gate is in at most `0`
bodies, and the counted weight `1` is above the right-hand side `0`. At `μ = 0` the boundary term `ω_max μ k_S`
vanishes: the bound without it is false where every premise holds at `μ = 1`. -/
def ServingBlockBound_nontrivial_overlap : Prop :=
  (∀ u ∈ Straddle.S, (Straddle.P.gates u).card ≤ 1 ∧ (Straddle.d.unregistered (Straddle.P.gates u)).card ≤ 1) ∧
  (∀ u ∈ Straddle.S, PerBlockCap Straddle.d Straddle.J Straddle.q 1 1) ∧ Straddle.P.IsCK Straddle.d 1 1 ∧
  PerBlockCap Straddle.d Straddle.J Straddle.q 1 1 ∧ OperationBodies Straddle.d Straddle.J ∧
  Overlap Straddle.J 1 ∧ ¬ Overlap Straddle.J 0 ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧
  (∀ u ∈ Straddle.S, UncertifiedCap Straddle.J (Straddle.P.gates u)
    (fun i : {i // Straddle.r.parent i = u} => Straddle.P.gates i) 1) ∧
  UncertifiedCap Straddle.J ((univ \ Straddle.S).biUnion Straddle.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Straddle.d.D.N)) 0 ∧
  ∑ u ∈ Straddle.S, min (Straddle.q 1 1 + 1 * (0 : ℕ) * (1 : ℕ))
    (((univ.filter fun i => Straddle.r.parent i = u).card : ℝ) * Straddle.q 1 1 + 1) + 0 < Straddle.J.total

/-- **The remote bound's premises hold** at `Honest`, at `(c, k) = (1, 1)`, the serving blocks as their own inference
blocks, `q_U(u) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBoundRemote_honest : Prop :=
  Honest.P.IsCK Honest.d 1 1 ∧ PerBlockCap Honest.d Honest.J Honest.q 1 1 ∧
  (∀ u ∈ Honest.S, UncertifiedCap Honest.J (Honest.P.gates u)
    (fun i : {i // Honest.r.parent i = u} => Honest.P.gates i) 1) ∧
  UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) 1

/-- **The remote bound is false without the per-block cap at the inference size**: at `Counter`, with `(c, k) = (1, 0)`,
the serving block as its own inference block and `q_U = 0`, every other premise holds, the cap at `(1, 0)` fails, and
the counted weight `1` is above the right-hand side `0`. -/
def ServingBlockBoundRemote_nontrivial : Prop :=
  Counter.P.IsCK Counter.d 1 0 ∧ ¬ PerBlockCap Counter.d Counter.J Counter.q 1 0 ∧
  (∀ u ∈ Counter.S, UncertifiedCap Counter.J (Counter.P.gates u)
    (fun i : {i // Counter.r.parent i = u} => Counter.P.gates i) 0) ∧
  UncertifiedCap Counter.J ((univ \ Counter.S).biUnion Counter.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N)) 0 ∧
  ∑ u ∈ Counter.S, (((univ.filter fun i => Counter.r.parent i = u).card : ℝ) * Counter.q 1 0 + 0) + 0 <
    Counter.J.total

/-- **The elementary uncertified cap's premise holds** at `Honest`: its serving blocks and dark remainder refine
themselves. -/
def UncertifiedElementary_honest : Prop :=
  Nonempty (Honest.P.Refines Honest.P)

/-- **The elementary uncertified cap is false without refinement**: at `Counter`, the parent map `away` sends the one
inference block, which holds the counted operation's gate, to the unit of `P₂` that holds no gate, so it is no
refinement, and the operation's weight `1`, in `P₂`'s first unit with no inference block mapped there, is above the cap
`split = 0`. -/
def UncertifiedElementary_nontrivial : Prop :=
  ¬ (∀ g, Counter.P₂.unit g = (Counter.P.unit g).map Counter.away) ∧
  ¬ UncertifiedCap Counter.J (Counter.P₂.gates 0) (fun i : {i // Counter.away i = 0} => Counter.P.gates i)
    (Counter.J.split (Counter.P₂.gates 0) (fun i : {i // Counter.away i = 0} => Counter.P.gates i))

end Accounting.Guarantees.ServingBlocks
