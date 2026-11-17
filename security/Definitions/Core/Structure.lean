module

public import Mathlib.Logic.Relation
public import Definitions.Core.Formalism

/-! The structural vocabulary: methods, realizations, decompositions, crossings and attainable profiles. -/

@[expose] public section

namespace Verity.Structure

open Finset

structure Method where
  d : RegisteredCircuit
  /-- the registered values, read at `θ` only -/
  registered : d.D.Trace
  coins : Finset (Fin d.D.N)
  coins_open : ∀ g ∈ coins, d.D.isInput g = true ∧ g ∉ d.θ

namespace Method

variable (M : Method)

def external : Finset (Fin M.d.D.N) :=
  univ.filter fun g => M.d.D.isInput g = true ∧ g ∉ M.d.θ ∧ g ∉ M.coins

def assign (x ω : M.d.D.Trace) : M.d.D.Trace := fun g =>
  if g ∈ M.d.θ then M.registered g else if g ∈ M.coins then ω g else x g

end Method

structure Evaluation where
  M : Method
  x : M.d.D.Trace
  ω : M.d.D.Trace

def Evaluation.trace (e : Evaluation) : e.M.d.D.Trace := e.M.d.D.eval (e.M.assign e.x e.ω)

/-- An execution: a declaration and the values its gates held. `ResourceAccounting.Requirement.Execution` is a
schedule, not this. -/
structure Execution where
  d : RegisteredCircuit
  val : d.D.Trace
  faithful : ∀ g, d.D.isInput g = false → val g = d.D.op g val

/-- `X` contains an evaluation equivalent to `e`, registering exactly the images of its registered gates, so an
execution cannot register precomputed tables. -/
def Realizes (E : Setoid Evaluation) (e : Evaluation) (X : Execution) : Prop :=
  ∃ e', E.r e e' ∧ ∃ ι : Containment e'.M.d.D X.d.D,
    X.d.θ = e'.M.d.θ.map ⟨ι.emb, ι.emb_inj⟩ ∧ ∀ g, (X.val (ι.emb g)).cast (ι.emb_w g) = e'.trace g

def circuitPreserving : Setoid Evaluation where
  r := Eq
  iseqv := ⟨fun _ => rfl, fun h => h.symm, fun h h' => h.trans h'⟩

def IsBlock (C : Circuit) (B : Finset (Fin C.N)) : Prop := B ⊆ C.computed

def IsPrefix (C : Circuit) (T : Finset (Fin C.N)) : Prop :=
  IsBlock C T ∧ ∀ a ∈ C.into T, C.isInput a = true

open Classical in
noncomputable def downClosure (C : Circuit) (S : Finset (Fin C.N)) : Finset (Fin C.N) :=
  C.computed.filter fun g => ∃ t ∈ S, C.Feeds g t

/-- The most gates of `B` on a path inside `B` that ends at `g`, and `0` when `g ∉ B`. -/
def depthTo (C : Circuit) (B : Finset (Fin C.N)) (g : Fin C.N) : ℕ :=
  if g ∈ B then (C.reads g).sup (fun a => if _h : a < g then depthTo C B a else 0) + 1 else 0
termination_by g.val
decreasing_by all_goals assumption

def depth (C : Circuit) (B : Finset (Fin C.N)) : ℕ := B.sup (depthTo C B)

inductive Provenance (Λ L : Type) where
  | registered
  | external (source : Λ)
  | produced (producer : L)

def provenance (d : RegisteredCircuit) {Λ : Type} {n : ℕ} (source : Fin d.D.N → Λ) (P : Partition d.D n)
    (a : Fin d.D.N) : Provenance Λ (Fin n) :=
  if a ∈ d.θ then .registered else
    match P.unit a with
    | some u => .produced u
    | none => .external (source a)

def registeredIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into B ∩ d.θ

def externalIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => d.D.isInput a = true

def producedIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => ¬ d.D.isInput a = true

open Classical in
noncomputable def externalFrom (d : RegisteredCircuit) {Λ : Type} (source : Fin d.D.N → Λ) (s : Λ)
    (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (externalIn d B).filter fun a => source a = s

def producedFrom (C : Circuit) (B' B : Finset (Fin C.N)) : Finset (Fin C.N) := C.into B ∩ B'

def AtScale {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop := ∀ u, C.size (P.gates u) ≤ c

def ExactScale {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop := ∀ u, C.size (P.gates u) = c

def total {C : Circuit} {n : ℕ} (P : Partition C n) (q : Finset (Fin C.N) → ℕ) : ℕ := ∑ u, q (P.gates u)

def Dep {C : Circuit} {n : ℕ} (P : Partition C n) (v u : Fin n) : Prop :=
  v ≠ u ∧ ∃ g, ∃ a ∈ C.reads g, P.unit g = some u ∧ P.unit a = some v

/-- **An ordered (schedulable) partition**: its block-level dependency graph `Dep` is acyclic. Every partition is a
decomposition, but on `a → b → c` the blocks `{a, c}` and `{b}` are not ordered. -/
def Ordered {C : Circuit} {n : ℕ} (P : Partition C n) : Prop := ∀ u, ¬ Relation.TransGen (Dep P) u u

inductive Destination (n : ℕ) where
  | block (u : Fin n)
  | output

inductive Timing where
  | same
  | later

def Crosses {C : Circuit} {n : ℕ} (P : Partition C n) (a : Fin C.N) : Destination n → Prop
  | .block u => a ∈ C.into (P.gates u)
  | .output => a ∈ C.outputs

structure Crossing {C : Circuit} {n : ℕ} (P : Partition C n) where
  gate : Fin C.N
  dest : Destination n
  crosses : Crosses P gate dest

/-- **The information in `S`** under the description-length function `I`: `I` of `S`'s values in `X`, in gate order.
`I = LDL M t r` is the local description length; `I = length` is the wire bits. -/
def info (I : List Bool → ℕ∞) (C : Circuit) (X : C.Trace) (S : Finset (Fin C.N)) : ℕ∞ :=
  I (C.bits X (S.sort (· ≤ ·)))

structure TimeOrdered (C : Circuit) (n : ℕ) (Loc : Type) (m : ℕ) extends Partition C n where
  loc : Fin n → Loc
  period : Fin n → Fin m
  ordered : ∀ g, ∀ a ∈ C.reads g, ∀ u v, unit g = some u → unit a = some v → period v ≤ period u

namespace TimeOrdered

variable {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m)

open Classical in
noncomputable def upto (j : ℕ) : Finset (Fin C.N) :=
  univ.filter fun g => ∃ u, F.unit g = some u ∧ (F.period u : ℕ) < j

/-- **When gate `a` reaches `δ`**: a produced value entering block `u` in its producer's period or a later one;
registered and external values, and final outputs, have no timing. -/
def timing (a : Fin C.N) : Destination n → Option Timing
  | .block u => (F.unit a).map fun v => if F.period v = F.period u then Timing.same else Timing.later
  | .output => none

open Classical in
/-- **What block `v` hands on with timing `t`**: its gates another block reads with that timing, sent (`same`) or
carried (`later`). A gate two blocks read can be both. -/
noncomputable def handsOn (t : Timing) (v : Fin n) : Finset (Fin C.N) :=
  (F.toPartition.gates v).filter fun a => ∃ u, a ∈ C.into (F.toPartition.gates u) ∧ F.timing a (.block u) = some t

def outputsOf (v : Fin n) : Finset (Fin C.N) := (F.toPartition.gates v).filter fun g => g ∈ C.outputs

noncomputable def infoHandsOn (I : List Bool → ℕ∞) (X : C.Trace) (t : Timing) (v : Fin n) : ℕ∞ :=
  info I C X (F.handsOn t v)

end TimeOrdered

def labels (d : RegisteredCircuit) {n m : ℕ} {Loc Λ : Type} (F : TimeOrdered d.D n Loc m) (source : Fin d.D.N → Λ)
    (x : Crossing F.toPartition) : Provenance Λ (Fin n) × Destination n × Option Timing :=
  (provenance d source F.toPartition x.gate, x.dest, F.timing x.gate x.dest)

structure Resources where
  blocks : ℕ
  work : ℕ
  /-- the fresh input bits over all blocks -/
  fresh : ℕ
  /-- the fresh input bits of the block that reads the most -/
  peakFresh : ℕ
  /-- the bits handed on to later periods -/
  carried : ℕ
  /-- the bits handed on within a period -/
  sent : ℕ

instance instLEResources : LE Resources where
  le r r' := r.blocks ≤ r'.blocks ∧ r.work ≤ r'.work ∧ r.fresh ≤ r'.fresh ∧ r.peakFresh ≤ r'.peakFresh ∧
    r.carried ≤ r'.carried ∧ r.sent ≤ r'.sent

/-- **The resource vector** of the time-ordered partition `F` of `d`, in wire bits. -/
noncomputable def resources (d : RegisteredCircuit) {n m : ℕ} {Loc : Type} (F : TimeOrdered d.D n Loc m) :
    Resources where
  blocks := n
  work := d.D.size d.D.computed
  fresh := total F.toPartition fun B => d.D.width (d.unregistered B)
  peakFresh := univ.sup fun u => d.D.width (d.unregistered (F.toPartition.gates u))
  carried := ∑ v, d.D.width (F.handsOn Timing.later v)
  sent := ∑ v, d.D.width (F.handsOn Timing.same v)

structure Family (Loc : Type) where
  admits : (d : RegisteredCircuit) → (n m : ℕ) → TimeOrdered d.D n Loc m → Prop

namespace Family

def atMost (Loc : Type) (c : ℕ) : Family Loc := ⟨fun _ _ _ F => AtScale F.toPartition c⟩

def exactly (Loc : Type) (c : ℕ) : Family Loc := ⟨fun _ _ _ F => ExactScale F.toPartition c⟩

def ordered {Loc : Type} (𝔉 : Family Loc) : Family Loc := ⟨fun d n m F => 𝔉.admits d n m F ∧ Ordered F.toPartition⟩

def onePeriod {Loc : Type} (𝔉 : Family Loc) : Family Loc := ⟨fun d n m F => 𝔉.admits d n m F ∧ m = 1⟩

end Family

noncomputable def profile {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (e : Evaluation) :
    Set Resources :=
  {r | ∃ X : Execution, Realizes E e X ∧
    ∃ (n m : ℕ) (F : TimeOrdered X.d.D n Loc m), 𝔉.admits X.d n m F ∧ resources X.d F = r}

noncomputable def leastFresh (𝓡 : Set Resources) : ℕ∞ := ⨅ r ∈ 𝓡, (r.fresh : ℕ∞)

noncomputable def fewestBlocks (𝓡 : Set Resources) (k : ℕ) : ℕ∞ :=
  ⨅ r ∈ 𝓡, ⨅ (_ : r.peakFresh ≤ k), (r.blocks : ℕ∞)

def everywhere (p : Evaluation → Prop) : Set Method := {M | ∀ x ω : M.d.D.Trace, p ⟨M, x, ω⟩}

noncomputable def worst (q : Evaluation → ℕ∞) (M : Method) : ℕ∞ :=
  ⨆ (x : M.d.D.Trace) (ω : M.d.D.Trace), q ⟨M, x, ω⟩

/-- The methods that attain a vector at most `r` on each external input and coin separately: the decomposition, and
even the executing circuit, may depend on the input. -/
noncomputable def within {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (r : Resources) :
    Set Method :=
  everywhere fun e => ∃ r' ∈ profile E 𝔉 e, r' ≤ r

/-- **A method's profile**: the vectors whose cell it lies in, read on no one input. -/
noncomputable def Method.profile {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (M : Method) :
    Set Resources :=
  {r | M ∈ within E 𝔉 r}

end Verity.Structure
