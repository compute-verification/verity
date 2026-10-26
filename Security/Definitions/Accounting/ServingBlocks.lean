import Definitions.Accounting.Actual
import Mathlib.Basic.Real.Basic

/-!
# Serving blocks and the training class's counts (trusted)

Layer 3 of `note:training/20261026T1840Z-draft-formal-properties` (locked 26 Oct, TOP-95; the working draft's §3 and §8).
A training class's counted operations sit in the actual circuit `A` (`Accounting.Placed`). Its per-block cap
`q(c, k)` bounds the weight of the operations computed wholly inside any block of at most `c` operation gates that reads
at most `k` unregistered input bits. A **serving block** is the region one server evaluates between two consecutive
wipes, and each counted operation is assigned to the serving block holding its representative gate. The serving-block
bound (`Accounting.Guarantees.ServingBlocks`) is Theorem 3 (counting) applied to each serving block and summed.

For a circuit `C`, its registered input gates `θ` and counted operations `J` (`Counted`):
* `J.total`: `Σ_{v ∈ 𝒥} ω_v`;
* `J.assigned R`: the local count `Ω_R`, the weight of the operations whose representative gate is in `R`;
* `J.inside I`: the weight of the operations computed wholly inside `I` (`body v ⊆ I`);
* `J.outside R B`: the weight of `R`'s operations computed wholly inside none of the blocks `B i`;
* `J.split R B`: the part of `J.outside R B` whose representative is in one of the blocks, the operations split between a
  block and gates outside it.

A block's unregistered input bits are `width(in(I) \ θ)`. Blocks here have at most `c` gates: the partition convention
pads each block to exactly `c` with gates that read only registered constants, which adds no unregistered input bit and
no counted operation, so a cap at exactly `c` on the padded circuit is the cap at most `c` on the circuit as placed.
A block needs no ordering: it inherits the circuit's topological order.

A serving block `block A X κ (s, j)` is `A[X s, [τ_s j, τ_s (j + 1))]`, the region of server `s` (location `X s`) between
its wipe ticks, the cuts `κ s` of `[0, t)`.

The named hypotheses: `PerBlockCap` (a), `Alignment` (b), `UncertifiedCap` (c), `InferenceBlocks` (each inference block
counted for a serving block is a `(c, k)`-block) and `ServingBlocksPartition` (the serving blocks hold every counted
operation off the dark leaves, `ServingBlocksCover`, each with at most `c_S` gates and `k_S` unregistered input bits).

Everything here is on `Accounting.Placed` and `Verity.Circuit` as they are on main, whose gates may be wider than one
bit; at width 1 (the Boolean profile, `Verity.Boolean.Circuit.toCircuit`) input bits are input gates. LEAN-63 moves every
statement onto one Boolean circuit type and one partition (`Verity.Partition`, no ordering, exact size `c`, at most `k`
unregistered inputs); this file moves with it.
-/

namespace Accounting.ServingBlocks

open Verity Finset

/-- **Counted operations** of a circuit `C`: a training class's `𝒥`, Theorem 3's `𝒱`. Each is named by its
representative gate `v ∈ V`, an operation gate of `C`, so distinct operations have distinct representatives. `body v` is
the set of gates that compute it, and it is computed wholly inside a set `I` of gates when `body v ⊆ I`. `ω v ≥ 0` is
its counting weight. -/
structure Counted (C : Circuit) where
  V : Finset (Fin C.N)
  body : Fin C.N → Finset (Fin C.N)
  ω : Fin C.N → ℝ
  V_computed : ∀ v ∈ V, C.isInput v = false
  ω_nonneg : ∀ v ∈ V, 0 ≤ ω v

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- `Σ_{v ∈ 𝒥} ω_v`: the counted weight. -/
def total : ℝ := ∑ v ∈ J.V, J.ω v

/-- **The local count** `Ω_R`: the weight of the counted operations whose representative gate is in `R`. -/
def assigned (R : Finset (Fin C.N)) : ℝ := ∑ v ∈ J.V with v ∈ R, J.ω v

/-- The weight of the counted operations computed wholly inside `I`. -/
def inside (I : Finset (Fin C.N)) : ℝ := ∑ v ∈ J.V with J.body v ⊆ I, J.ω v

/-- The weight of the counted operations assigned to `R` that are computed wholly inside none of the blocks `B i`. -/
def outside (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) : ℝ :=
  ∑ v ∈ J.V with v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i, J.ω v

/-- The weight of the counted operations assigned to `R` whose representative is in one of the blocks `B i` but that are
computed wholly inside none of them: the operations split between a block and gates outside it. -/
def split (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) : ℝ :=
  ∑ v ∈ J.V with v ∈ R ∧ (∃ i, v ∈ B i) ∧ ∀ i, ¬ J.body v ⊆ B i, J.ω v

end Counted

variable {L σ : Type*}

/-- The serving blocks' index: a server `s` and one of the windows `j` its cuts `κ s` make. -/
abbrev Block {t : ℕ} (κ : σ → Cuts t) : Type _ := Σ s, Fin (κ s).J

/-- **Serving block** `S = (s, j)`: the region `A[X s, [τ_s j, τ_s (j + 1))]` server `s` evaluates between its wipes
`τ_s j` and `τ_s (j + 1)`. -/
noncomputable def block (A : Placed L) (X : σ → Finset L) {t : ℕ} (κ : σ → Cuts t) (S : Block κ) :
    Finset (Fin A.N) :=
  A.region (X S.1) (Set.Ico ((κ S.1).τ S.2.castSucc) ((κ S.1).τ S.2.succ))

/-- **(a) A per-block cap** `q(c, k)`: every block `I` of at most `c` operation gates of `C` that reads at most `k`
unregistered input bits (`width(in(I) \ θ)`) holds counted weight at most `q c k` computed wholly inside it. A property of
the training class at the block size `(c, k)` (the working draft's Lemma 1 and §5), read on the circuit at hand. -/
def PerBlockCap (C : Circuit) (θ : Finset (Fin C.N)) (J : Counted C) (q : ℕ → ℕ → ℝ) (c k : ℕ) : Prop :=
  ∀ I ⊆ C.computed, C.size I ≤ c → C.width (C.into I \ θ) ≤ k → J.inside I ≤ q c k

/-- **The inference blocks are `(c, k)`-blocks**: the `n S` inference blocks `inf S i` counted for each serving block `S`
are sets of operation gates of `C`, each of at most `c` gates reading at most `k` unregistered input bits (the
registered `(c, k)`-partition of the declared circuit, as the actual circuit evaluates it). -/
def InferenceBlocks (C : Circuit) (θ : Finset (Fin C.N)) {β : Type*} (n : β → ℕ)
    (inf : (S : β) → Fin (n S) → Finset (Fin C.N)) (c k : ℕ) : Prop :=
  ∀ S i, inf S i ⊆ C.computed ∧ C.size (inf S i) ≤ c ∧ C.width (C.into (inf S i) \ θ) ≤ k

/-- **(b) Alignment**: the blocks `B i` counted for `R` hold every certified gate (`W`) of `R`. At a serving block `S`
with the inference blocks counted for it: when each registered inference block lies wholly inside one serving block
(sessions drained at wipe ticks), the blocks inside `S` hold every certified gate of `S`. A block that straddles serving
blocks may instead be counted for each serving block it meets, at one more `q(c, k)` each. -/
def Alignment {C : Circuit} (R : Finset (Fin C.N)) {ι : Type*} (B : ι → Finset (Fin C.N)) (W : Finset (Fin C.N)) :
    Prop :=
  ∀ g ∈ R, g ∈ W → ∃ i, g ∈ B i

/-- **(c) An uncertified cap** `q_U`: the weight of `R`'s counted operations computed wholly inside none of the blocks
`B i` is at most `qU`. At a serving block with its inference blocks this is `q_U(S)`; with the one block `S` itself it is
the boundary term `q_B(S)`; at the dark leaves, with no blocks, `q_U(L_dark)`. -/
def UncertifiedCap {C : Circuit} (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) (qU : ℝ) : Prop :=
  J.outside R B ≤ qU

/-- **The serving blocks cover the counted operations**: every counted operation's representative gate is placed at a
dark leaf or on a server `X s` before `t`, so it lies in one of the serving blocks or at the dark leaves. -/
def ServingBlocksCover (A : Placed L) (J : Counted A.toCircuit) (dark : Finset L) (X : σ → Finset L) (t : ℕ) :
    Prop :=
  ∀ v ∈ J.V, A.x v ∈ dark ∨ ∃ s, A.x v ∈ X s ∧ A.t v < t

/-- **The serving blocks are a `(c_S, k_S)`-partition**: they cover the counted operations off the dark leaves
(`ServingBlocksCover`), and each serving block `S` has at most `c_S` gates and reads at most `k_S` unregistered input
bits. Padding each to exactly `c_S` gates makes them a `(c_S, k_S)`-partition of the padded circuit; the regions of
disjoint servers between consecutive wipes are disjoint, which the bound does not need. -/
def ServingBlocksPartition (A : Placed L) (θ : Finset (Fin A.N)) (J : Counted A.toCircuit) (dark : Finset L)
    (X : σ → Finset L) {t : ℕ} (κ : σ → Cuts t) (cS kS : Block κ → ℕ) : Prop :=
  ServingBlocksCover A J dark X t ∧
    ∀ S, A.size (block A X κ S) ≤ cS S ∧ A.width (A.into (block A X κ S) \ θ) ≤ kS S

end Accounting.ServingBlocks
