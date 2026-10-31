import Mathlib.Logic.Relation
import Definitions.Core.Formalism

/-!
# Structure: methods, realizations, decompositions and attainable profiles

The structural vocabulary, stated on core's objects and adding only what they lack. Definitions only: the short facts
named below are proved in `Proofs.Core.Structure`.

* **Method, evaluation, execution, realization.** A method (`Method`) is a declaration with its input interface: its
  registered gates `θ` fixed to their values, its coin gates and its other (external) input gates left open. An
  evaluation (`Evaluation`) is a method on one external input and one coin, with its trace (`Evaluation.trace`). An
  execution (`Execution`) is a declaration with the values its gates held, each computed gate's value its operation's.
  An execution realizes an evaluation up to an equivalence `E` (`Realizes E`, `E` a `Setoid` on evaluations) when it
  contains, by a `Containment`, an evaluation equivalent to it, registering exactly its registered gates and holding its
  values. The circuit-preserving equivalence is equality (`circuitPreserving`); a re-encoding is another method, related
  by another `E`, and a theorem names the class it covers by the `E` it takes.
* **Decompositions.** Any `Partition` of the computed gates is a decomposition; an ordered (schedulable) one (`Ordered`)
  has an acyclic block-level dependency graph (`Dep`). Blocks have at most `c` gates (`AtScale`) or exactly `c`
  (`ExactScale`, padded as `Pad` pads), and then `|C| = c · n` (`size_computed_of_exact`). A time-ordered partition
  (`TimeOrdered`) gives each block a location and a period, no block reading a later period; the blocks before each
  period form a prefix (`TimeOrdered.upto_isPrefix`), and distinct periods make it ordered
  (`TimeOrdered.ordered_of_injective`).
* **Crossings.** A crossing (`Crossing`) is a gate read by a block outside it, or a final output. Its three labels
  (`labels`) are independent: provenance (`provenance`: registered, external tagged with its source, produced tagged
  with its producer block), destination (`Destination`: a block, or the output) and timing (`TimeOrdered.timing`: its
  producer's period, or a later one). A gate read by two blocks can be both sent and carried (`TimeOrdered.handsOn`).
  Crossings are counted in wire bits (`Circuit.width`) and, separately, in information under a description-length
  function `I` passed as a parameter (`info`, `TimeOrdered.infoHandsOn`; `I = LDL M t r` is the local description
  length), so wire-width and reconstruction-hardness claims stay apart.
* **Profiles.** A resource vector (`Resources`) is blocks, work, fresh input (total, and at the largest reader), carried
  bits and sent bits. The attainable profile `𝓡(e)` (`profile E 𝔉 e`) is the set of vectors of the decompositions a
  family `𝔉` admits (`Family`: `atMost`, `exactly`, `ordered`, `onePeriod`) of the executions realizing `e` up to `E`.
  `K*` and `n*` are its projections (`leastFresh`, `fewestBlocks`).
* **Classes** are sets of methods, so membership never depends on an input or a coin: `everywhere p` holds `p` on every
  external input and coin, `worst q` is `q`'s supremum over them (as `Slowdown.cost` takes the worst input), the cell of
  a vector `r` (`within`) is the methods attaining a vector at most `r` on every input and coin, and a method's profile
  (`Method.profile`) is the vectors whose cell it lies in.

**Where it meets Accounting.** Core cannot import Accounting, so Accounting reads these. An `Actual.Run` is an
`Execution` (`A.toCircuit`, `val`, `faithful`) once its protocol's baseline is given. `Slowdown.Task`'s coin draws the
task's input, a method's external input here; a method's `coins` are its own randomness, neither the task's nor the
verifier's (`Verity.Coins`). A class of methods `𝔐` gives the `Slowdown.Class` of pairs `(C, x)` whose runs on every
input realize evaluations of one method of `𝔐`, the same for every `x`.

Memory is not a block quantity: what wipes and bandwidth bound is what a time-ordered partition's blocks carry and send.

**Stated limits.**
* A realization may differ per input: `within` asks for one on each external input and coin separately, so the
  decomposition, and even the executing circuit, may depend on the input, as `Slowdown`'s `In → Actual.Run` allows. A
  data-independent schedule would fix the execution's declaration and its decomposition across inputs.
* `Realizes` forbids registering anything extra: the execution's registered gates are exactly the images of the
  evaluation's, so an execution cannot register precomputed tables. Registration happens once, at the start.
-/

namespace Verity.Structure

open Finset

/-! ### Methods, evaluations, executions and realizations -/

/-- **A method**: a declaration with its input interface, its registered gates `θ` fixed to `registered` and its coin
gates and other (external) input gates left open. -/
structure Method where
  /-- the circuit and its registered gates -/
  d : RegisteredCircuit
  /-- the registered values, read at `θ` only -/
  registered : d.D.Transcript
  /-- the coin gates -/
  coins : Finset (Fin d.D.N)
  /-- coin gates are input gates outside `θ` -/
  coins_open : ∀ g ∈ coins, d.D.isInput g = true ∧ g ∉ d.θ

namespace Method

variable (M : Method)

/-- **The external input gates**: the input gates neither registered nor coins. -/
def external : Finset (Fin M.d.D.N) :=
  univ.filter fun g => M.d.D.isInput g = true ∧ g ∉ M.d.θ ∧ g ∉ M.coins

/-- The input assignment on external input `x` and coin `ω`: registered gates take their registered values, coin gates
`ω`'s, and the rest `x`'s. -/
def assign (x ω : M.d.D.Transcript) : M.d.D.Transcript := fun g =>
  if g ∈ M.d.θ then M.registered g else if g ∈ M.coins then ω g else x g

end Method

/-- **An evaluated computation**: a method on one external input `x` and one coin `ω`, each read only at its gates. -/
structure Evaluation where
  /-- the method -/
  M : Method
  /-- the external input -/
  x : M.d.D.Transcript
  /-- the coin -/
  ω : M.d.D.Transcript

/-- An evaluation's trace: its method's circuit evaluated on its input assignment. -/
def Evaluation.trace (e : Evaluation) : e.M.d.D.Transcript := e.M.d.D.eval (e.M.assign e.x e.ω)

/-- **An execution**: a declaration and the values its gates held, each computed gate's value its operation on them.
`Actual.Run` is one once its baseline is given; `Accounting.Requirement.Execution` is a schedule, not this. -/
structure Execution where
  /-- the circuit that ran and its registered gates -/
  d : RegisteredCircuit
  /-- the values its gates held -/
  val : d.D.Transcript
  /-- each computed gate's value is its operation on them (`Placed.Faithful`) -/
  faithful : ∀ g, d.D.isInput g = false → val g = d.D.op g val

/-- **`X` realizes `e` up to the equivalence `E`**: `X` contains an evaluation `e'` equivalent to `e`, registering
exactly the images of `e'`'s registered gates and holding `e'`'s values at the images of its gates. -/
def Realizes (E : Setoid Evaluation) (e : Evaluation) (X : Execution) : Prop :=
  ∃ e', E.r e e' ∧ ∃ ι : Containment e'.M.d.D X.d.D,
    X.d.θ = e'.M.d.θ.map ⟨ι.emb, ι.emb_inj⟩ ∧ ∀ g, (X.val (ι.emb g)).cast (ι.emb_w g) = e'.trace g

/-- **The circuit-preserving equivalence**: equality, so a realization contains the evaluation's own circuit. -/
def circuitPreserving : Setoid Evaluation where
  r := Eq
  iseqv := ⟨fun _ => rfl, fun h => h.symm, fun h h' => h.trans h'⟩

/-! ### Blocks and prefixes -/

/-- **A block** of `C`: a set of its computed gates. -/
def IsBlock (C : Circuit) (B : Finset (Fin C.N)) : Prop := B ⊆ C.computed

/-- **A prefix** of `C`: a block that reads no computed gate outside itself. -/
def IsPrefix (C : Circuit) (T : Finset (Fin C.N)) : Prop :=
  IsBlock C T ∧ ∀ a ∈ C.into T, C.isInput a = true

open Classical in
/-- **The down-closure** of `S`: the computed gates that feed a gate of `S`. -/
noncomputable def downClosure (C : Circuit) (S : Finset (Fin C.N)) : Finset (Fin C.N) :=
  C.computed.filter fun g => ∃ t ∈ S, C.Feeds g t

/-- The most gates of `B` on a path inside `B` that ends at `g`, and `0` when `g ∉ B`. -/
def depthTo (C : Circuit) (B : Finset (Fin C.N)) (g : Fin C.N) : ℕ :=
  if g ∈ B then (C.reads g).sup (fun a => if _h : a < g then depthTo C B a else 0) + 1 else 0
termination_by g.val
decreasing_by all_goals assumption

/-- **The depth** of `B`: the most gates on a path inside `B`. -/
def depth (C : Circuit) (B : Finset (Fin C.N)) : ℕ := B.sup (depthTo C B)

/-! ### Provenance and reads -/

/-- **A read bit's provenance**: registered, external from a source in `Λ`, or produced by a block labelled in `L`. -/
inductive Provenance (Λ L : Type) where
  /-- in the baseline `θ`, registered once at the start -/
  | registered
  /-- an input gate outside `θ`, from the source `source` -/
  | external (source : Λ)
  /-- a computed gate, produced by the block `producer` -/
  | produced (producer : L)

/-- **The provenance of gate `a`** in the partition `P` of `d`: registered in `θ`, else produced by its block, else
external from `source a`. -/
def provenance (d : RegisteredCircuit) {Λ : Type} {n : ℕ} (source : Fin d.D.N → Λ) (P : Partition d.D n)
    (a : Fin d.D.N) : Provenance Λ (Fin n) :=
  if a ∈ d.θ then .registered else
    match P.unit a with
    | some u => .produced u
    | none => .external (source a)

/-- **The registered reads** of `B`: the gates of `in(B)` in `θ`. -/
def registeredIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) := d.D.into B ∩ d.θ

/-- **The external reads** of `B`: the input gates of `in(B)` outside `θ`. -/
def externalIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => d.D.isInput a = true

/-- **The produced reads** of `B`: the computed gates of `in(B)`. -/
def producedIn (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (d.unregistered B).filter fun a => ¬ d.D.isInput a = true

open Classical in
/-- **The reads of `B` from the source `s`**: its external reads that `source` tags `s`. -/
noncomputable def externalFrom (d : RegisteredCircuit) {Λ : Type} (source : Fin d.D.N → Λ) (s : Λ)
    (B : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  (externalIn d B).filter fun a => source a = s

/-- **The reads of `B` from the block `B'`**, `r(B' → B)`: the gates of `B'` with a wire into `B`. -/
def producedFrom (C : Circuit) (B' B : Finset (Fin C.N)) : Finset (Fin C.N) := C.into B ∩ B'

/-! ### Partitions: arbitrary, ordered, at most or exactly `c` -/

/-- **Scale `c`**: every block of `P` has at most `c` gates. -/
def AtScale {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop := ∀ u, C.size (P.gates u) ≤ c

/-- **Exact scale `c`**: every block of `P` has exactly `c` gates, padded where it would have fewer (`Pad`). -/
def ExactScale {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop := ∀ u, C.size (P.gates u) = c

/-- **The total** of a per-block quantity `q` over the blocks of `P`. -/
def total {C : Circuit} {n : ℕ} (P : Partition C n) (q : Finset (Fin C.N) → ℕ) : ℕ := ∑ u, q (P.gates u)

/-- **Block `v` feeds block `u`** in `P`: they differ, and a gate of `u` reads a gate of `v`. -/
def Dep {C : Circuit} {n : ℕ} (P : Partition C n) (v u : Fin n) : Prop :=
  v ≠ u ∧ ∃ g, ∃ a ∈ C.reads g, P.unit g = some u ∧ P.unit a = some v

/-- **An ordered (schedulable) partition**: its block-level dependency graph `Dep` is acyclic. Every partition is a
decomposition, but on `a → b → c` the blocks `{a, c}` and `{b}` are not ordered. -/
def Ordered {C : Circuit} {n : ℕ} (P : Partition C n) : Prop := ∀ u, ¬ Relation.TransGen (Dep P) u u

/-! ### Crossings -/

/-- **Where a crossing goes**: into a block, or out as a final output. -/
inductive Destination (n : ℕ) where
  /-- read by the block `u` -/
  | block (u : Fin n)
  /-- a final output -/
  | output

/-- **When a crossing lands**: in its producer's period (sent) or a later one (carried). -/
inductive Timing where
  /-- the producer's period -/
  | same
  /-- a later period -/
  | later

/-- **`a` crosses to `δ`** in `P`: block `u` reads `a` from outside itself (`δ = block u`), or `a` is a final output. -/
def Crosses {C : Circuit} {n : ℕ} (P : Partition C n) (a : Fin C.N) : Destination n → Prop
  | .block u => a ∈ C.into (P.gates u)
  | .output => a ∈ C.outputs

/-- **A crossing** of `P`: a gate and a destination it crosses to. -/
structure Crossing {C : Circuit} {n : ℕ} (P : Partition C n) where
  /-- the gate whose value crosses -/
  gate : Fin C.N
  /-- where it goes -/
  dest : Destination n
  /-- it does cross there -/
  crosses : Crosses P gate dest

/-- **The information in `S`** under the description-length function `I`: `I` of `S`'s values in `X`, in gate order.
`I = LDL M t r` is the local description length; `I = length` is the wire bits. -/
def info (I : List Bool → ℕ∞) (C : Circuit) (X : C.Transcript) (S : Finset (Fin C.N)) : ℕ∞ :=
  I (C.bits X (S.sort (· ≤ ·)))

/-! ### Time-ordered partitions -/

/-- **A time-ordered partition** of `C` into `n` blocks, each tagged with a location in `Loc` and one of `m` periods: no
block reads a computed gate of a later period. -/
structure TimeOrdered (C : Circuit) (n : ℕ) (Loc : Type) (m : ℕ) extends Partition C n where
  /-- the location each block runs at -/
  loc : Fin n → Loc
  /-- the period each block runs in -/
  period : Fin n → Fin m
  /-- a block reads computed gates of its own period or earlier ones only -/
  ordered : ∀ g, ∀ a ∈ C.reads g, ∀ u v, unit g = some u → unit a = some v → period v ≤ period u

namespace TimeOrdered

variable {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m)

open Classical in
/-- The gates of the blocks of the periods before `j`. -/
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

/-- **What block `v` outputs**: its final output gates. -/
def outputsOf (v : Fin n) : Finset (Fin C.N) := (F.toPartition.gates v).filter fun g => g ∈ C.outputs

/-- **The information block `v` hands on with timing `t`** under `I`, on the values `X`: `handsOn`'s crossings measured
by description length rather than wire bits. -/
noncomputable def infoHandsOn (I : List Bool → ℕ∞) (X : C.Transcript) (t : Timing) (v : Fin n) : ℕ∞ :=
  info I C X (F.handsOn t v)

end TimeOrdered

/-- **A crossing's three labels**, independent of each other: provenance, destination and timing. -/
def labels (d : RegisteredCircuit) {n m : ℕ} {Loc Λ : Type} (F : TimeOrdered d.D n Loc m) (source : Fin d.D.N → Λ)
    (x : Crossing F.toPartition) : Provenance Λ (Fin n) × Destination n × Option Timing :=
  (provenance d source F.toPartition x.gate, x.dest, F.timing x.gate x.dest)

/-! ### Attainable profiles -/

/-- **A joint resource vector**: blocks, work, fresh input in total and at its largest reader, carried bits and sent
bits. -/
structure Resources where
  /-- the number of blocks -/
  blocks : ℕ
  /-- the gates computed -/
  work : ℕ
  /-- the fresh input bits over all blocks -/
  fresh : ℕ
  /-- the fresh input bits of the block that reads the most -/
  peakFresh : ℕ
  /-- the bits handed on to later periods -/
  carried : ℕ
  /-- the bits handed on within a period -/
  sent : ℕ

/-- At most in every resource. -/
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

/-- **A family of decompositions**: the time-ordered partitions, at locations `Loc`, it admits of each declaration. -/
structure Family (Loc : Type) where
  /-- the decompositions it admits -/
  admits : (d : RegisteredCircuit) → (n m : ℕ) → TimeOrdered d.D n Loc m → Prop

namespace Family

/-- Blocks of at most `c` gates. -/
def atMost (Loc : Type) (c : ℕ) : Family Loc := ⟨fun _ _ _ F => AtScale F.toPartition c⟩

/-- Blocks of exactly `c` gates. -/
def exactly (Loc : Type) (c : ℕ) : Family Loc := ⟨fun _ _ _ F => ExactScale F.toPartition c⟩

/-- The ordered decompositions of `𝔉`. -/
def ordered {Loc : Type} (𝔉 : Family Loc) : Family Loc := ⟨fun d n m F => 𝔉.admits d n m F ∧ Ordered F.toPartition⟩

/-- The decompositions of `𝔉` in one period: plain partitions, nothing carried. -/
def onePeriod {Loc : Type} (𝔉 : Family Loc) : Family Loc := ⟨fun d n m F => 𝔉.admits d n m F ∧ m = 1⟩

end Family

/-- **The attainable profile** `𝓡(e)`: the resource vectors of the decompositions in `𝔉` of the executions that realize
`e` up to `E`. -/
noncomputable def profile {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (e : Evaluation) :
    Set Resources :=
  {r | ∃ X : Execution, Realizes E e X ∧
    ∃ (n m : ℕ) (F : TimeOrdered X.d.D n Loc m), 𝔉.admits X.d n m F ∧ resources X.d F = r}

/-- **`K*`**, a projection of a profile: the least fresh input over its vectors. -/
noncomputable def leastFresh (𝓡 : Set Resources) : ℕ∞ := ⨅ r ∈ 𝓡, (r.fresh : ℕ∞)

/-- **`n*` at `k`**, a projection of a profile: the fewest blocks over its vectors whose largest reader reads at most
`k` fresh bits. -/
noncomputable def fewestBlocks (𝓡 : Set Resources) (k : ℕ) : ℕ∞ :=
  ⨅ r ∈ 𝓡, ⨅ (_ : r.peakFresh ≤ k), (r.blocks : ℕ∞)

/-! ### Classes of methods -/

/-- **A class fixed across inputs and coins**: the methods whose every evaluation has the property `p`. -/
def everywhere (p : Evaluation → Prop) : Set Method := {M | ∀ x ω : M.d.D.Transcript, p ⟨M, x, ω⟩}

/-- **The worst case** of `q` over a method's external inputs and coins. -/
noncomputable def worst (q : Evaluation → ℕ∞) (M : Method) : ℕ∞ :=
  ⨆ (x : M.d.D.Transcript) (ω : M.d.D.Transcript), q ⟨M, x, ω⟩

/-- **The cell of `r`**: the methods that attain a vector at most `r` on every external input and coin. -/
noncomputable def within {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (r : Resources) :
    Set Method :=
  everywhere fun e => ∃ r' ∈ profile E 𝔉 e, r' ≤ r

/-- **A method's profile**: the vectors whose cell it lies in, read on no one input. -/
noncomputable def Method.profile {Loc : Type} (E : Setoid Evaluation) (𝔉 : Family Loc) (M : Method) :
    Set Resources :=
  {r | M ∈ within E 𝔉 r}

end Verity.Structure
