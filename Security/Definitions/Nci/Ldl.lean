import Definitions.Nci
import Mathlib.Data.ENat.Lattice
import Mathlib.Data.Nat.Log

/-!
# Local description length on a `(c,k)`-partition: the vocabulary

The working draft's model ("Distinguishing inference from frontier training", §3–§5.1), on core's circuit and its one
partition. The registered inputs `θ` are a declaration's baseline (`Verity.Declaration`), the blocks are a
`Verity.Partition` of the operation gates, and a block's unregistered input bits `|in(I) \ θ|` are its novel reads
(`Nci.novelReads`, which on a Boolean circuit, one bit per gate, is the number of those gates). Blocks carry no order:
each inherits the circuit's topological order.

* `Decoder`, `LDL`: a fixed decoder and `LDL_{t,R}(y) = K^t(y | R)`, the least program length producing `y` within `t`.
* `Registration`: `R = (θ, C, 𝓘)` with the registered values, fixed once; `Registration.Consistent`: a run takes them.
* `Exact`, `ReadCap`, `CKPartition`: exactly `c` operation gates per block, at most `k` unregistered input bits per
  block, and both.
* `Products`, `Products.inBlock`, `Products.acts`, `Products.FanOut`: counted product occurrences, `𝓙_I`, `V_I`, and
  activation fan-out at most `e`.
-/

namespace Nci.Ldl

open Verity

/-- **A decoder**: `D p t` is what the program `p` outputs within `t` decoder steps, `none` when it outputs nothing
within `t`.

The draft's `K^t(y | R)` runs a fixed self-delimiting decoder with indexed access to the registration `R`. That decoder,
with `R` built in, is one such function, fixed before any string it describes, so a statement about every decoder is one
about it, and a premise on `D` is a premise on it. Nothing here models the machine: the one upper bound (Lemma 1,
`Guarantees.BlockDescription`) needs one capability, that the decoder evaluates a block from its index and its
unregistered input bits, and that is the named premise `Assumptions.EvaluatesBlocks`; the lower bounds are premises
(`Assumptions.ActivationRate`). Prefix-freeness and monotonicity in `t` hold of the draft's decoder, and no result here
uses them. -/
abbrev Decoder : Type := List Bool → ℕ → Option (List Bool)

/-- **Local description length** `LDL_{t,R}(y) = K^t(y | R)`: the least length of a program from which `D` outputs `y`
within `t` steps, `⊤` when there is none. -/
noncomputable def LDL (D : Decoder) (t : ℕ) (y : List Bool) : ℕ∞ :=
  ⨅ (p : List Bool) (_ : D p t = some y), (p.length : ℕ∞)

/-- **A registration** `R = (θ, C, 𝓘)` of the declaration `d = (C, θ)`, fixed once, before the monitored computation:
the registered values `val` (read only at `θ`) and the partition `P` of `C`'s operation gates into the blocks
`0, …, n − 1`. Nothing computed later is added to it. -/
structure Registration (d : Declaration) (n : ℕ) where
  val : d.D.Transcript
  P : Partition d.D n

/-- The run `X` takes the registered values at `θ`. -/
def Registration.Consistent {d : Declaration} {n : ℕ} (R : Registration d n) (X : d.D.Transcript) : Prop :=
  ∀ g ∈ d.θ, X g = R.val g

/-- **Exactly `c` operation gates in every block**, so that `|C| = c B`: padding counts as gates. -/
def Exact {C : Circuit} {n : ℕ} (P : Partition C n) (c : ℕ) : Prop :=
  ∀ u, C.size (P.gates u) = c

/-- **At most `k` unregistered input bits into every block**: `|in(I) \ θ| ≤ k`, in bits. -/
def ReadCap (d : Declaration) {n : ℕ} (P : Partition d.D n) (k : ℕ) : Prop :=
  ∀ u, novelReads d (P.gates u) ≤ k

/-- **A `(c,k)`-partition**: exactly `c` operation gates and at most `k` unregistered input bits per block. -/
def CKPartition (d : Declaration) {n : ℕ} (P : Partition d.D n) (c k : ℕ) : Prop :=
  Exact P c ∧ ReadCap d P k

/-- **Counted product occurrences** (§5.1): the occurrences `J`, `M = #J` of them, each with a distinct representative
operation gate `rep p` and the activation occurrence `act p` it reads. An occurrence is assigned to the block that holds
its representative. -/
structure Products (C : Circuit) (ι A : Type) where
  J : Finset ι
  rep : ι → Fin C.N
  rep_inj : Set.InjOn rep J
  rep_op : ∀ p ∈ J, C.isInput (rep p) = false
  act : ι → A

namespace Products

variable {C : Circuit} {ι A : Type} (M : Products C ι A)

/-- `𝓙_I`: the occurrences assigned to block `u`. -/
def inBlock {n : ℕ} (P : Partition C n) (u : Fin n) : Finset ι :=
  M.J.filter fun p => P.unit (M.rep p) = some u

/-- `V_I`: the activation occurrences that block `u`'s occurrences read. -/
def acts [DecidableEq A] {n : ℕ} (P : Partition C n) (u : Fin n) : Finset A :=
  (M.inBlock P u).image M.act

/-- **Activation fan-out at most `e`**: each activation occurrence is read by at most `e` counted occurrences. -/
def FanOut [DecidableEq A] (e : ℕ) : Prop :=
  ∀ j, (M.J.filter fun p => M.act p = j).card ≤ e

end Products

end Nci.Ldl
