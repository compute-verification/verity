module

public import Mathlib.Data.ENat.Lattice
public import Mathlib.Data.Finset.Sort
public import Mathlib.Data.Nat.Log
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.ZMod.Defs
public import Definitions.Core.Partition
public import Definitions.Core.RegisteredCircuit

/-! The working draft's model (§3 and §4): Boolean circuits, `(c, k)`-partitions, the committed registered circuit and
local description length. -/

@[expose] public section

namespace Verity

open Finset

namespace Circuit

/-- **A Boolean circuit**: every gate holds one bit, and every operation gate reads at least one gate. -/
structure Boolean (C : Circuit) : Prop where
  one_bit : ∀ g, C.w g = 1
  fanin_pos : ∀ g, C.isInput g = false → (C.reads g).Nonempty

/-- **Fan-in at most `a`**: every gate reads at most `a` gates. Sizes are compared at a fixed basis of bounded fan-in:
one gate of unbounded fan-in can compute a whole product. -/
def FanIn (C : Circuit) (a : ℕ) : Prop := ∀ g, (C.reads g).card ≤ a

/-- The bits of a bundle of gates `l` in `X`, gate by gate in `l`'s order, each gate's value least significant bit
first. On a Boolean circuit, one bit per gate. -/
def bits (C : Circuit) (X : C.Trace) (l : List (Fin C.N)) : List Bool :=
  l.flatMap fun g => List.ofFn fun i : Fin (C.w g) => (X g).getLsbD i

/-- A value's least significant bit, in `ZMod 2`. -/
def bitZ {n : ℕ} (x : BitVec n) : ZMod 2 := if x.getLsbD 0 then 1 else 0

/-- Gate `g` computes an **affine** function over GF(2) of the gates it reads: its bit is a constant plus the XOR of the
bits of some of them. On a Boolean circuit these are the gates of an XOR/NOT basis, which Flock proves for free. -/
def Affine (C : Circuit) (g : Fin C.N) : Prop :=
  ∃ (S : Finset (Fin C.N)) (c : ZMod 2), S ⊆ C.reads g ∧ ∀ v, bitZ (C.op g v) = c + ∑ a ∈ S, bitZ (v a)

/-- **An AND gate** of a Boolean circuit: an operation gate whose function of the gates it reads is not affine over
GF(2). `Verity.Circuit`'s gate functions carry no tag, so AND is this predicate on the function. Flock's `and a b`
(`a ≠ b`) and its rows `(⊕ a) ∧ (⊕ b)` whose two XORs differ are AND gates, and its XOR, NOT and input gates are not
(`Verity.Guarantees.ToCircuitAnd`). -/
def IsAnd (C : Circuit) (g : Fin C.N) : Prop := C.isInput g = false ∧ ¬ C.Affine g

open Classical in
/-- **The AND count** of the gates `S`, the number of AND gates among them: on `C.computed`, the circuit's
multiplicative complexity, the gates Flock pays for. Credit (`creditSize`) and every reported count are stated in
`andSize`. Exclusion results are not converted to ANDs: they are reported as `N` counted operations (FP8 MACs, say),
each labelled with that operation's canonical AND count, since converting them to ANDs of hidden work would need a
ceiling on ANDs per resource unit over all programs, which is vacuous. All-gate counts (the `(c, k)`-partition, the
draft's `Φ`, `c_S`, `size`) stay in `size`, linked to `andSize` by `andSize_le_size`. -/
noncomputable def andSize (C : Circuit) (S : Finset (Fin C.N)) : ℕ := (S.filter fun g => C.IsAnd g).card

end Circuit

/-- **Credit**: the AND count of the registered circuit `d` under the one public lowering `lower`, pinned at
registration, never the prover's own lowering. Each credit record carries `lower`'s digest. -/
noncomputable def creditSize (lower : RegisteredCircuit → Circuit) (d : RegisteredCircuit) : ℕ :=
  (lower d).andSize Finset.univ

/-- `in(S) \ θ`: the gates outside `S` with a wire into `S` that are not registered. Every operation gate outside `S`
that `S` reads is among them. -/
def RegisteredCircuit.unregistered (d : RegisteredCircuit) (S : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  d.D.into S \ d.θ

/-- **A `(c, k)`-partition** of a Boolean declaration's operation gates: every inference block has at most `c` gates and at
most `k` unregistered input bits. -/
structure Partition.IsCK (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (c k : ℕ) : Prop where
  boolean : d.D.Boolean
  card : ∀ u, (P.gates u).card ≤ c
  reads : ∀ u, (d.unregistered (P.gates u)).card ≤ k

/-- **The size ratio** of `B` inference blocks of budget `c` relative to a reference training algorithm of size `T₀`: `c · B / T₀`,
each inference block priced at `c` gates. -/
noncomputable def slowdown (c B : ℕ) (T₀ : ℝ) : ℝ := ((c * B : ℕ) : ℝ) / T₀

/-- **The committed registered circuit**: the committed trace (`trace`, the values committed in the execution phase,
in commitment order), the registered circuit `d`, whose circuit `C`, registered inputs `θ`, partition `𝓘` into `d.n`
inference blocks and partition into proof blocks are the registered compiler's output on the registered θ and the
advice reported in the execution phases, and the committed values of `θ`'s gates (`val`, read only there), fixed before
the monitored computation. -/
structure Committed where
  trace : List (List Bool)
  d : RegisteredCircuit
  val : d.D.Trace

namespace Committed

variable (R : Committed)

/-- A run of the registered circuit: its evaluation on inputs that agree with the registered values on `θ`. -/
def Honest (a : R.d.D.Trace) : Prop := ∀ g ∈ R.d.θ, a g = R.val g

def Reads (X : R.d.D.Trace) : Prop :=
  R.Honest X ∧ ∀ g ∈ R.d.P.committed, g ∉ R.d.θ → ∃ i, R.d.pos g = some i ∧ R.trace[i]? = some (R.d.D.bits X [g])

/-- The compiler's output explains the committed trace. The developer proves it of `B`'s run on the committed advice
before the challenger's coins; a trace too coarse for the partition `B` outputs has no such transcript. -/
def Explains : Prop := Set.InjOn R.d.pos ↑(R.d.P.committed \ R.d.θ) ∧ ∃ X, R.Reads X

/-- **Inference block `u` recomputed** from `R` and `X`: a registered gate takes its registered value, every
other gate outside the inference block takes `X`'s value, and the inference block's gates are computed in order. On the
inference block and its inputs, only `X`'s values on `in(I) \ θ` matter. -/
def recompute (X : R.d.D.Trace) (u : Fin R.d.n) : R.d.D.Trace :=
  R.d.D.run fun g => if g ∈ R.d.θ then some (R.val g) else if R.d.𝓘.unit g = some u then none else some (X g)

end Committed

/-- Inference block `u`'s index in `⌈log₂ B⌉` bits, least significant first. -/
def indexCode {B : ℕ} (u : Fin B) : List Bool :=
  List.ofFn fun i : Fin (Nat.clog 2 B) => u.val.testBit i

/-- **A decoder** with indexed access to registered information of type `ρ`: on a program and a time allowance it outputs
a string, or nothing. More time never changes an output, and its programs are self-delimiting: of two programs it
accepts at one allowance, neither is a proper prefix of the other. -/
structure Decoder (ρ : Type) where
  run : ρ → List Bool → ℕ → Option (List Bool)
  mono : ∀ r p t t' y, t ≤ t' → run r p t = some y → run r p t' = some y
  prefixFree : ∀ r t p p' y y', run r p t = some y → run r p' t = some y' → p <+: p' → p = p'

/-- **Local description length** `LDL_{t,r}(y) = K^t(y | r)`: the length of a shortest program on which `M` outputs `y`
within `t` steps given `r`, and `⊤` when there is none. -/
noncomputable def LDL {ρ : Type} (M : Decoder ρ) (t : ℕ) (r : ρ) (y : List Bool) : ℕ∞ :=
  ⨅ (p : List Bool) (_ : M.run r p t = some y), (p.length : ℕ∞)

/-- The program Lemma 1 hands a decoder for inference block `u` of a run `X`: the evaluator `e`, the inference block's index,
and the bits of its unregistered inputs in gate order. -/
def blockProgram (R : Committed) (e : List Bool) (X : R.d.D.Trace) (u : Fin R.d.n) : List Bool :=
  e ++ indexCode u ++ R.d.D.bits X ((R.d.unregistered (R.d.𝓘.gates u)).sort (· ≤ ·))

/-- **A fixed evaluator** `e` performs the extraction rule `y` within time `t` for every inference block: on inference block
`u`'s program of any run `X`, `M` outputs the string `y u` extracts from inference block `u` recomputed from that program's
contents. -/
def Evaluates {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (R : Committed) (e : List Bool)
    (y : Fin R.d.n → R.d.D.Trace → List Bool) : Prop :=
  ∀ u X, M.run r (blockProgram R e X u) t = some (y u (R.recompute X u))

end Verity
