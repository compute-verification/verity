module

public import Definitions.ResourceAccounting.Requirement
public import CircuitRequirement

/-!
# The requirement's code: each function is the definition it computes

`CircuitRequirement` (`verity/circuits/requirement/`, package `core_requirement`) is code, so it requires nothing
of `security/`: on one instance it computes `ResourceAccounting.Requirement`'s definitions, a set as a list. These are those
equalities, each a `Prop` stated against the code's function and the definition it computes, on the wiring, execution,
layout, store and placement the instance denotes (`wiring`, `execution`, `layout`, `store`, `placement`), and
`Proofs.Properties.ResourceAccounting.Requirement` proves every one. The code's difftest holds it to
`verity.circuits.requirement`, so these carry the Python requirement to the definitions that
`Definitions/ResourceAccounting/Guarantees/Requirement.lean`'s guarantees are stated over.

An instance's values are the natural numbers, of which it describes `0 … n - 1`, and a gate uses, at any circuit's
values, the operands it uses at the execution's: the definitions read `uses` only at `Execution.value`.
-/

@[expose] public section

namespace Properties.Requirement

open CircuitRequirement

/-- The wiring an instance denotes. -/
def wiring (i : Instance) : ResourceAccounting.Requirement.Wiring ℕ where
  width := i.widthOf
  registered := i.registered.toFinset
  inputs := i.inputs.toFinset
  ops g := (i.opsOf g).toFinset
  uses _ g := (i.usesOf g).toFinset

/-- The execution an instance denotes. -/
def execution (i : Instance) : ResourceAccounting.Requirement.Execution ℕ where
  ran := i.ran.toFinset
  step := i.stepOf
  value := i.valueOf

/-- The layout an instance denotes. -/
def layout (i : Instance) : ResourceAccounting.Requirement.Store.Placement ℕ where
  block := i.blockOf
  offset := i.offsetOf
  encoded := i.encodedOf

/-- The store a code store denotes. -/
def store (s : CircuitRequirement.Store) : ResourceAccounting.Requirement.Store where
  registered := s.registered
  written t := (s.writtenAt t).toFinset
  freed t := (s.freedAt t).toFinset
  payload := s.payloadOf

/-- The placement a code placement denotes, of the execution an instance denotes. -/
def placement (i : Instance) (p : CircuitRequirement.Placement) : ResourceAccounting.Requirement.Placement ℕ ℕ where
  toExecution := execution i
  leaf := p.leafOf
  ticks := p.ticksOf

/-- `topological` decides `Wiring.Topological`. -/
def Topological : Prop := ∀ i : Instance, i.topological = true ↔ (wiring i).Topological

/-- `follows` decides `Execution.Follows`. -/
def Follows : Prop := ∀ i : Instance, i.follows = true ↔ (execution i).Follows (wiring i)

def Live : Prop := ∀ (i : Instance) (t : ℕ), (i.live t).toFinset = ResourceAccounting.Requirement.live (wiring i) (execution i) t

def Reads : Prop :=
  ∀ (i : Instance) (t : ℕ), (i.reads t).toFinset = ResourceAccounting.Requirement.reads (wiring i) (execution i) t

def LiveBits : Prop :=
  ∀ (i : Instance) (t : ℕ), i.liveBits t = ResourceAccounting.Requirement.liveBits (wiring i) (execution i) t

def ReadBits : Prop :=
  ∀ (i : Instance) (t : ℕ), i.readBits t = ResourceAccounting.Requirement.readBits (wiring i) (execution i) t

def Held : Prop :=
  ∀ (i : Instance) (start len : ℕ), i.held start len = ResourceAccounting.Requirement.held (wiring i) (execution i) ⟨start, len⟩

def Read : Prop :=
  ∀ (i : Instance) (start len : ℕ), i.read start len = ResourceAccounting.Requirement.read (wiring i) (execution i) ⟨start, len⟩

def LiveBlocks : Prop :=
  ∀ (i : Instance) (t : ℕ),
    (i.liveBlocks t).toFinset = ResourceAccounting.Requirement.liveBlocks (wiring i) (execution i) (layout i) t

def LiveIn : Prop :=
  ∀ (i : Instance) (t j : ℕ), i.liveIn t j = ResourceAccounting.Requirement.liveIn (wiring i) (execution i) (layout i) t j

def ReadBlocks : Prop :=
  ∀ (i : Instance) (t : ℕ),
    (i.readBlocks t).toFinset = ResourceAccounting.Requirement.readBlocks (wiring i) (execution i) (layout i) t

def ReadIn : Prop :=
  ∀ (i : Instance) (t j : ℕ), i.readIn t j = ResourceAccounting.Requirement.readIn (wiring i) (execution i) (layout i) t j

def Resident : Prop := ∀ (s : CircuitRequirement.Store) (t : ℕ), (s.resident t).toFinset = (store s).resident t

def Incomplete : Prop :=
  ∀ (i : Instance) (t : ℕ),
    (i.incomplete t).toFinset = ResourceAccounting.Requirement.incomplete (wiring i) (execution i) (layout i) t

def Credited : Prop :=
  ∀ (i : Instance) (t : ℕ),
    (i.credited t).toFinset = ResourceAccounting.Requirement.credited (wiring i) (execution i) (layout i) t

def CreditedReads : Prop :=
  ∀ (i : Instance) (t : ℕ),
    (i.creditedReads t).toFinset = ResourceAccounting.Requirement.creditedReads (wiring i) (execution i) (layout i) t

def CreditedBits : Prop :=
  ∀ (i : Instance) (t : ℕ),
    i.creditedBits t = ResourceAccounting.Requirement.creditedBits (wiring i) (execution i) (layout i) t

def CreditedReadBits : Prop :=
  ∀ (i : Instance) (t : ℕ),
    i.creditedReadBits t = ResourceAccounting.Requirement.creditedReadBits (wiring i) (execution i) (layout i) t

/-- `storeMatches` decides `StoreMatches`. -/
def StoreMatches : Prop :=
  ∀ (i : Instance) (s : CircuitRequirement.Store) (b t : ℕ),
    i.storeMatches s b t = true ↔ ResourceAccounting.Requirement.StoreMatches (wiring i) (execution i) (layout i) (store s) b t

/-- `planMatches` decides `PlanMatches`, of the plan's blocks. -/
def PlanMatches : Prop :=
  ∀ (i : Instance) (plan : List ℕ) (b t : ℕ),
    i.planMatches plan b t = true ↔
      ResourceAccounting.Requirement.PlanMatches (wiring i) (execution i) (layout i) plan.toFinset b t

/-- `region` is `W`'s region at the ticks `[lo, hi)`. -/
def Region : Prop :=
  ∀ (i : Instance) (p : CircuitRequirement.Placement) (X : List ℕ) (lo hi : ℕ),
    (i.region p X lo hi).toFinset =
      ResourceAccounting.Requirement.region (wiring i) (placement i p) X.toFinset (Set.Ico lo hi)

/-- `stepRegion` is `W`'s gates on `X` at the engine steps `[start, start + len)`. -/
def StepRegion : Prop :=
  ∀ (i : Instance) (p : CircuitRequirement.Placement) (X : List ℕ) (start len : ℕ),
    (i.stepRegion p X start len).toFinset =
      ResourceAccounting.Requirement.stepRegion (wiring i) (placement i p) X.toFinset ⟨start, len⟩

/-- `ingress` is `W`'s ingress at the ticks `[lo, hi)`. -/
def Ingress : Prop :=
  ∀ (i : Instance) (p : CircuitRequirement.Placement) (X : List ℕ) (lo hi : ℕ),
    i.ingress p X lo hi = ResourceAccounting.Requirement.ingress (wiring i) (placement i p) X.toFinset (Set.Ico lo hi)

/-- `credits` is the work, at the pricing `price`, of `W`'s region at the ticks `[lo, hi)` less the marked values. Over
the wiring on `Fin n` with these values, `RequirementCredits` makes that work what PoUW credits a node that runs it as
placed (`PouwWork.credits`); this equality is over `ℕ`, and transfers to it only through that restriction. -/
def Credits : Prop :=
  ∀ (i : Instance) (p : CircuitRequirement.Placement) (price : Array ℕ) (X marked : List ℕ) (lo hi : ℕ),
    i.credits p price X marked lo hi =
      ResourceAccounting.Requirement.work (fun g => price.getD g 0)
        (ResourceAccounting.Requirement.region (wiring i) (placement i p) X.toFinset (Set.Ico lo hi) \ marked.toFinset)

end Properties.Requirement
