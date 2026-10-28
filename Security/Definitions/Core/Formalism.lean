import Mathlib.Data.ENat.Lattice
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Nat.Log
import Mathlib.Basic.Real.Basic
import Mathlib.Data.ZMod.Defs
import Definitions.Core.Partition
import Definitions.Core.Registered

/-!
# The working draft's model: Boolean circuits, `(c, k)`-partitions and local description length

The model of the working draft "Distinguishing inference from frontier training" (§3 and §4), on core's one circuit
type `Verity.Circuit` and its one `Partition`. Every statement about training, NCI and the accounting envelopes is to be
stated on these objects.

* A **Boolean circuit** (`Circuit.Boolean`) is a circuit whose every gate holds one bit and whose every operation gate
  reads at least one gate: the draft's positive fan-in, with constants as registered input gates. Its gate basis is
  whatever `op` computes. The basis's bounded fan-in is `Circuit.FanIn C a`, which a statement comparing sizes across
  circuits takes and none of Lemma 1, Theorem 3 and Lemma A.1 needs. The circuit's **size**
  `|C|` is its number of operation gates, `C.size C.computed`. A multi-bit value is a bundle of gates, read in order as
  the string `bits X l`.
* The **AND count** `C.andSize S` (`andSize`) is the number of AND gates (`IsAnd`: an operation gate whose function is
  not affine over GF(2)) among `S`, and `C.andSize C.computed` is the circuit's multiplicative complexity, the gates
  C-Flock pays for. Credit and reporting count `andSize`; Theorem 3 and the `(c, k)`-partition count `size`.
* **Credit** (`creditSize lower d`) is the AND count of the registered declared circuit under one public lowering,
  pinned at registration.
* The **committed registered circuit** `R` (`Committed`) is the committed trace, the values the developer committed in
  the execution phase, with the registered circuit (`Verity.Registered`) and the committed values of `θ`'s gates. Its
  circuit `C`, `θ`, its partition `𝓘` of the operation gates into inference blocks and their partition into proof
  blocks are the registered compiler's output on the advice (`Verity.Compiler`), which runs after the trace is
  committed, so the partition the protocols prove blocks of is the one this model reads. The output **explains** the
  committed trace (`Committed.Explains`) when some transcript holds the registered values on `θ` and, at each proof
  block's committed inputs and outputs, the committed value at the gate's position in the trace: what the developer
  proves of `B`'s run before any coin. `R` is not the registration (`Verity.Registration`), which holds only the Site,
  `θ` and the compiler `B`: the advice, and with it the partition, is private. An inference block is a unit of a
  `Partition`; training's instance calls them serving blocks. No ordering of the inference blocks is required: Lemma 1
  and Theorem 3 read none, so the draft's "inter-block edges point forward" is dropped.
* A **`(c, k)`-partition** (`Partition.IsCK`) of a Boolean circuit: every inference block has exactly `c` gates, padding
  included, and at most `k` unregistered input bits, `|in(I) \ θ| ≤ k` (`unregistered`). Its circuit has size `c · B` for `B` inference blocks.
  Inference blocks of at most `c` gates become exactly `c` by a `Pad`: registered padding gates in each inference block, read by
  nothing of `d`.
* The **slowdown** relative to a reference of size `T₀` is `|C| / T₀` (`slowdown`), a ratio of circuit sizes.
* **Local description length** (§4): `LDL M t r y` is the length of a shortest program on which the decoder `M`, with
  indexed access to the registered information `r`, outputs `y` within `t` steps, and `⊤` when none does. Side
  information `z` is registered information paired with it, `LDL M t (r, z) y`. The decoder is a parameter (`Decoder`):
  its programs are self-delimiting and more time never changes an output.
* Lemma 1's inference-block evaluation: `Committed.recompute X u` evaluates inference block `u`'s gates from the registered
  values and `X`'s values at its unregistered inputs, which a decoder is handed as `blockProgram`: an evaluator `e`, the
  inference block's index in `⌈log₂ B⌉` bits (`indexCode`) and the bits of `in(I) \ θ` in gate order.

A multi-bit circuit reaches this model through its bit-level expansion (`Definitions.Core.Expand`).
-/

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
def bits (C : Circuit) (X : C.Transcript) (l : List (Fin C.N)) : List Bool :=
  l.flatMap fun g => List.ofFn fun i : Fin (C.w g) => (X g).getLsbD i

/-- A value's least significant bit, in `ZMod 2`. -/
def bitZ {n : ℕ} (x : BitVec n) : ZMod 2 := if x.getLsbD 0 then 1 else 0

/-- Gate `g` computes an **affine** function over GF(2) of the gates it reads: its bit is a constant plus the XOR of the
bits of some of them. On a Boolean circuit these are the gates of an XOR/NOT basis, which C-Flock proves for free. -/
def Affine (C : Circuit) (g : Fin C.N) : Prop :=
  ∃ (S : Finset (Fin C.N)) (c : ZMod 2), S ⊆ C.reads g ∧ ∀ v, bitZ (C.op g v) = c + ∑ a ∈ S, bitZ (v a)

/-- **An AND gate** of a Boolean circuit: an operation gate whose function of the gates it reads is not affine over
GF(2). `Verity.Circuit`'s gate functions carry no tag, so AND is this predicate on the function. C-Flock's `and a b`
(`a ≠ b`) and its rows `(⊕ a) ∧ (⊕ b)` whose two XORs differ are AND gates, and its XOR, NOT and input gates are not
(`Verity.Guarantees.ToCircuitAnd`). -/
def IsAnd (C : Circuit) (g : Fin C.N) : Prop := C.isInput g = false ∧ ¬ C.Affine g

open Classical in
/-- **The AND count** of the gates `S`, the number of AND gates among them: on `C.computed`, the circuit's
multiplicative complexity, the gates C-Flock pays for. Credit (`creditSize`) and every reported count are stated in
`andSize`. Exclusion results are not converted to ANDs: they are reported as `N` counted operations (FP8 MACs, say),
each labelled with that operation's canonical AND count, since converting them to ANDs of hidden work would need a
ceiling on ANDs per resource unit over all programs, which is vacuous. All-gate counts (the `(c, k)`-partition, the
draft's `Φ`, `c_S`, `size`) stay in `size`, linked to `andSize` by `andSize_le_size`. -/
noncomputable def andSize (C : Circuit) (S : Finset (Fin C.N)) : ℕ := (S.filter fun g => C.IsAnd g).card

end Circuit

/-- **Credit**: the AND count of the registered declared circuit `d` under the one public lowering `lower`, pinned at
registration, never the prover's own lowering. Each credit record carries `lower`'s digest. -/
noncomputable def creditSize (lower : Registered → Circuit) (d : Registered) : ℕ :=
  (lower d).andSize Finset.univ

/-- `in(S) \ θ`: the gates outside `S` with a wire into `S` that are not registered. Every operation gate outside `S`
that `S` reads is among them. -/
def Registered.unregistered (d : Registered) (S : Finset (Fin d.D.N)) : Finset (Fin d.D.N) :=
  d.D.into S \ d.θ

/-- **A `(c, k)`-partition** of a Boolean declaration's operation gates: every inference block has exactly `c` gates and at
most `k` unregistered input bits. -/
structure Partition.IsCK (d : Registered) {B : ℕ} (P : Partition d.D B) (c k : ℕ) : Prop where
  boolean : d.D.Boolean
  card : ∀ u, (P.gates u).card = c
  reads : ∀ u, (d.unregistered (P.gates u)).card ≤ k

/-- **A padding** of the partition `P` of `d` to the partition `P'` of `d'`, with as many inference blocks: `d'`'s circuit
contains `d`'s (`M`), with the same registered gates among `d`'s, every gate of `d` stays in its inference block, and every
other gate of `d'`, the padding, is a registered input gate or reads registered gates only. Padding is fixed and
input-independent, so each inference block reads the unregistered gates it read before (`Verity.Pad.unregistered`).
It completes the accounting; it is not evidence that hardware evaluated anything. -/
structure Pad (d : Registered) {B : ℕ} (P : Partition d.D B) (d' : Registered) (P' : Partition d'.D B) where
  M : Containment d.D d'.D
  θ_emb : ∀ g, M.emb g ∈ d'.θ ↔ g ∈ d.θ
  unit_emb : ∀ g, P'.unit (M.emb g) = P.unit g
  pad_input : ∀ g', g' ∉ Set.range M.emb → d'.D.isInput g' = true → g' ∈ d'.θ
  pad_reads : ∀ g', g' ∉ Set.range M.emb → ∀ a ∈ d'.D.reads g', a ∈ d'.θ

/-- **The slowdown** of `C` relative to a reference circuit of size `T₀`: `|C| / T₀`. -/
noncomputable def slowdown (C : Circuit) (T₀ : ℝ) : ℝ := ((C.size C.computed : ℕ) : ℝ) / T₀

/-- **The committed registered circuit**: the committed trace (`trace`, the values committed in the execution phase,
in commitment order), the registered circuit `d`, whose circuit `C`, registered inputs `θ`, partition `𝓘` into `d.n`
inference blocks and partition into proof blocks are the registered compiler's output on the advice, and the committed
values of `θ`'s gates (`val`, read only there), fixed before the monitored computation. -/
structure Committed where
  trace : List (List Bool)
  d : Registered
  val : d.D.Transcript

namespace Committed

variable (R : Committed)

/-- A run of the registered circuit: its evaluation on inputs that agree with the registered values on `θ`. -/
def Honest (a : R.d.D.Transcript) : Prop := ∀ g ∈ R.d.θ, a g = R.val g

/-- **`X` reads the commitments**: it holds the registered values on `θ`, and at each proof block's committed inputs
and outputs outside `θ` (`Partition.committed` of the proof blocks) the committed value at the gate's position in the
trace, as its bits. -/
def Reads (X : R.d.D.Transcript) : Prop :=
  R.Honest X ∧ ∀ g ∈ R.d.P.committed, g ∉ R.d.θ → ∃ i, R.d.pos g = some i ∧ R.trace[i]? = some (R.d.D.bits X [g])

/-- **The compiler's output explains the committed trace**: some transcript reads the commitments, so each proof block's
inputs and outputs are committed values. The developer proves it of `B`'s run on the committed advice before the
challenger's coins; a trace too coarse for the partition `B` outputs has no such transcript. -/
def Explains : Prop := ∃ X, R.Reads X

/-- **Inference block `u` recomputed** from `R` and `X`: a registered gate takes its registered value, every
other gate outside the inference block takes `X`'s value, and the inference block's gates are computed in order. On the
inference block and its inputs, only `X`'s values on `in(I) \ θ` matter. -/
def recompute (X : R.d.D.Transcript) (u : Fin R.d.n) : R.d.D.Transcript :=
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
def blockProgram (R : Committed) (e : List Bool) (X : R.d.D.Transcript) (u : Fin R.d.n) : List Bool :=
  e ++ indexCode u ++ R.d.D.bits X ((R.d.unregistered (R.d.𝓘.gates u)).sort (· ≤ ·))

/-- **A fixed evaluator** `e` performs the extraction rule `y` within time `t` for every inference block: on inference block
`u`'s program of any run `X`, `M` outputs the string `y u` extracts from inference block `u` recomputed from that program's
contents. -/
def Evaluates {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (R : Committed) (e : List Bool)
    (y : Fin R.d.n → R.d.D.Transcript → List Bool) : Prop :=
  ∀ u X, M.run r (blockProgram R e X u) t = some (y u (R.recompute X u))

end Verity
