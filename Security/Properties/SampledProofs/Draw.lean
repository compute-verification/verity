import Definitions.Claims.Sampling
import SampledProofs.Slots

/-!
# Sampled proofs' draw in the claims' slots: it misses proof units no more often than `P_a` (`SampledProofs.Slots.select`)

The property of the challenger's draw as the claims read it (`Deploy.checked`, `Claims.Sampling.KS`'s `DrawLaw`): the
draw `Slots.select` makes from `L` uniform bytes under a Spec's selection rule, over the verifier's own program and query,
checks no proof unit at a slot of `B` with probability at most the claims' acceptance probability
`P_a(B)` of the design it draws, on the partition it draws over. It takes no premise: the draw is a function of its inputs
and its bytes, and every input is in it, the `L` bytes included, however few.

* **The partition** (`slotPartition`) is `Slots.layout`'s, its classes the population's strata and the input class, each
  proof block its gates.
* **The design** (`slotDesign`) is the one the draw runs: the first class's design the C-Flock law it samples
  (`Select.select`'s, `oneStageOf`), `all` for the `All` rule; the input class drawn `all`. The bound is `0` where the
  draw is refused before it samples (`missBound`).
* **The bytes** are `src`'s, in order (`bytesOf`): a draw that would read past them is refused, so it misses nothing.

`DesignWellFormed`: that design is well formed on that partition (the claims' `Design.WellFormed`), so its `P_a` is a
probability, wherever both are given; the draw's own refusals make it so, and it takes no premise either.
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

/-- `L` bytes as `Slots.select` reads its coins. -/
def bytesOf {L : ℕ} (src : Fin L → Fin 256) : ByteArray := ⟨Array.ofFn fun i => UInt8.ofFin (src i)⟩

/-- A layout's proof block, its gates `[lo, hi)`, as the claims' set of gates. -/
def puGates (b : Slots.ProofBlock) : List ℕ := List.range' b.gates.1 (b.gates.2 - b.gates.1)

/-- The claims' partition of a layout: its classes, each its strata, each its proof blocks' gates. -/
def slotPartition (l : Slots.Layout) : Claims.Sampling.Partition :=
  ⟨l.classes.toList.map fun c => c.toList.map fun s => s.toList.map puGates⟩

/-- The claims' one-stage design of a C-Flock law. -/
def oneStageOf : Flock.Draw.Law → Claims.Sampling.OneStage
  | .subset k => .subset k
  | .bernoulli num den => .bernoulli num den
  | .stratified k _ => .stratified k
  | .work k ws => .work k (ws.toList.map (·.work)) (ws.toList.map (·.floor))

/-- **The design `Slots.select` draws** under `rule` on `d` and `query` with the work `table`: the population's design,
the law `Select.select` samples (`all` for the `All` rule), and the input class drawn `all`. An error where
`Select.select` refuses before it samples. -/
def slotDesign (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) : Flock.V Claims.Sampling.Design := do
  let n ← Partition.units d query
  let named ← rule.lawOver n
  let derived ← Draw.deriveLaw d query named table
  let first ← match rule with
    | .all => pure Claims.Sampling.OneStage.all
    | _ => do
      let parsed ← Draw.flockLaw ((derived.map (·.toJson)).getD named) n
      pure (oneStageOf (derived.getD parsed))
  return ⟨[first, .all]⟩

/-- The bound: `P_a(B)` of the design on the partition the draw runs over, and `0` where the draw is refused before it
samples. -/
noncomputable def missBound (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (B : Finset ℕ) : ℝ :=
  match Slots.layout d query, slotDesign rule d query table with
  | .ok l, .ok D => D.Pa (slotPartition l) B
  | _, _ => 0

/-- The slots a draw checks, none where it is refused. -/
def checkedOf : Flock.V (Slots.Layout × Array ℕ) → Array ℕ
  | .ok (_, c) => c
  | .error _ => #[]

/-- The checked slots `c` miss `B`: none is a slot of `B`. -/
def Misses (B : Finset ℕ) (c : Array ℕ) : Prop := ∀ v ∈ B, v ∉ c

/-- **The draw misses `B` with probability at most `P_a(B)`**: for every selection rule, program, query, work table,
closure map and number `L` of uniform bytes, and every set `B` of slots, the draw `Slots.select` makes from the bytes is
accepted and checks no proof unit at a slot of `B` with probability at most `missBound`. -/
def DrawMisses : Prop :=
  ∀ (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int))) (L : ℕ)
    (B : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Slots.select rule d query table cm (bytesOf src)).isOk = true ∧
        Misses B (checkedOf (Slots.select rule d query table cm (bytesOf src)))) ≤
      missBound rule d query table B

/-- **The design the draw runs is well formed on the partition it runs over**: wherever `Slots.layout` lays out `d`
under `query` and `slotDesign` gives the design, it is the claims' `Design.WellFormed` there. One rule per class, and
the first stage `subset K` with `1 ≤ K ≤ N`, Bernoulli at `NUM/DEN` with `0 < DEN` and `NUM ≤ DEN`, `stratified K` with
`1 ≤ K ≤ N` over strata of at least one unit each, or `work K` over such strata with one work and one floor of at least
one per stratum, and some work. -/
def DesignWellFormed : Prop :=
  ∀ (rule : Select.Rule) (d : Program) (query : Json) (table : Option Json) (l : Slots.Layout)
    (D : Claims.Sampling.Design),
    Slots.layout d query = .ok l → slotDesign rule d query table = .ok D → D.WellFormed (slotPartition l)

end Properties.SampledProofs
