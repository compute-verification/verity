import Definitions.Claims.Sampling
import SampledProofs.Slots

/-!
# Sampled proofs' draw in the claims' slots: it misses replay units no more often than `P_a` (`SampledProofs.Slots.select`)

The property of the challenger's draw as the claims read it (`Deploy.checked`, `Claims.Sampling.KS`'s `DrawLaw`): the
draw `Slots.select` makes from `L` uniform bytes under a Spec's selection rule, over the verifier's own program and query,
checks no proof unit of any replay unit at a slot of `B` with probability at most the claims' acceptance probability
`P_a(B)` of the design it draws, on the partition it draws over. It takes no premise: the draw is a function of its inputs
and its bytes, and every input is in it, the `L` bytes included, however few.

* **The partition** (`slotPartition`) is `Slots.layout`'s, its classes the population's strata and the input class, each
  replay unit its proof units' gates.
* **The design** (`slotDesign`) is the one the draw runs: the first class's first stage the C-Flock law it samples
  (`Select.select`'s, `oneStageOf`), `all` for the `All` rule; the input class drawn `all`; every proof unit of a drawn
  replay unit checked. The bound is `0` where the draw is refused before it samples (`missBound`).
* **The bytes** are `src`'s, in order (`bytesOf`): a draw that would read past them is refused, so it misses nothing.
-/

open Lean (Json)
open SampledProofs
open Flock.TemplateQuery (Desc)

namespace Properties.SampledProofs

/-- `L` bytes as `Slots.select` reads its coins. -/
def bytesOf {L : ℕ} (src : Fin L → Fin 256) : ByteArray := ⟨Array.ofFn fun i => UInt8.ofFin (src i)⟩

/-- The claims' partition of a layout: its classes, each its strata, each its replay units' proof units. -/
def slotPartition (l : Slots.Layout) : Claims.Sampling.Partition :=
  ⟨l.classes.toList.map fun c => c.toList.map fun s => s.toList.map fun r => r.pus.toList⟩

/-- The claims' one-stage design of a C-Flock law. -/
def oneStageOf : Flock.Draw.Law → Claims.Sampling.OneStage
  | .subset k => .subset k
  | .bernoulli num den => .bernoulli num den
  | .stratified k _ => .stratified k
  | .work k ws => .work k (ws.toList.map (·.work)) (ws.toList.map (·.floor))

/-- **The design `Slots.select` draws** under `rule` on `d` and `query` with the work `table`: the population's first
stage, the law `Select.select` samples (`all` for the `All` rule), and the input class drawn `all`, each class checking
every proof unit of a drawn replay unit. An error where `Select.select` refuses before it samples. -/
def slotDesign (rule : Select.Rule) (d : Desc) (query : Json) (table : Option Json) : Flock.V Claims.Sampling.Design := do
  let n ← Partition.units d query
  let named ← rule.lawOver n
  let derived ← Draw.deriveLaw d query named table
  let first ← match rule with
    | .all => pure Claims.Sampling.OneStage.all
    | _ => do
      let parsed ← Draw.flockLaw ((derived.map (·.toJson)).getD named) n
      pure (oneStageOf (derived.getD parsed))
  return ⟨[⟨first, none⟩, ⟨.all, none⟩]⟩

/-- The bound: `P_a(B)` of the design on the partition the draw runs over, and `0` where the draw is refused before it
samples. -/
noncomputable def missBound (rule : Select.Rule) (d : Desc) (query : Json) (table : Option Json) (B : Finset ℕ) : ℝ :=
  match Slots.layout d query, slotDesign rule d query table with
  | .ok l, .ok D => D.Pa (slotPartition l) B
  | _, _ => 0

/-- The proof units a draw checks, none where it is refused. -/
def checkedOf : Flock.V (Slots.Layout × Array (ℕ × ℕ)) → Array (ℕ × ℕ)
  | .ok (_, c) => c
  | .error _ => #[]

/-- The checked proof units `c` miss `B`: none is a proof unit of a replay unit at a slot of `B`. -/
def Misses (B : Finset ℕ) (c : Array (ℕ × ℕ)) : Prop := ∀ v ∈ B, ∀ i, (v, i) ∉ c

/-- **The draw misses `B` with probability at most `P_a(B)`**: for every selection rule, program, query, work table,
closure map and number `L` of uniform bytes, and every set `B` of slots, the draw `Slots.select` makes from the bytes is
accepted and checks no proof unit of a replay unit in `B` with probability at most `missBound`. -/
def DrawMisses : Prop :=
  ∀ (rule : Select.Rule) (d : Desc) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int))) (L : ℕ)
    (B : Finset ℕ),
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Slots.select rule d query table cm (bytesOf src)).isOk = true ∧
        Misses B (checkedOf (Slots.select rule d query table cm (bytesOf src)))) ≤
      missBound rule d query table B

end Properties.SampledProofs
