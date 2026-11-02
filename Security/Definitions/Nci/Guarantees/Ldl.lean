import Definitions.Nci.Assumptions.Ldl

/-!
# Local description length on a `(c,k)`-partition: the guarantees

The working draft's Lemma 1, Theorem 3 and Corollary 4b, on core's circuit and `Verity.Partition`.

* `BlockDescription` (Lemma 1): under `EvaluatesBlocks`, every block's extraction has
  `LDL_{t,R}(y_I) ≤ k' = k + ⌈log₂ B⌉ + a_blk`.
* `Counting` (Theorem 3): blocks of exactly `c` gates, each of counted weight at most `q`, give
  `|C| ≥ (c / q) Σ_v ω_v`.
* `FanOutBound` (Corollary 4b): with activation fan-out at most `e` and `ActivationRate`,
  `ρ ≥ c β_x M / (T₀ e h_x)`, `h_x = k' + λ_x`.

`ρ = |C| / T₀` is written out: `|C|` is `size computed`, the number of operation gates, and `T₀ > 0` the size of the
reference circuit. The statements are about an arbitrary circuit, not the NCI check's code.
-/

namespace Nci.Ldl.Guarantees

open Verity

/-- **Lemma 1 (block description)**: let the committed declaration `R` (`Verity.Committed`) have at most `k`
unregistered input bits per block, and let the run `X = eval inp` take the registered values. If the decoder evaluates
blocks for the extraction rule `ext` within `t` at overhead `aBlk` (`EvaluatesBlocks`), every block's extraction
`y_I = ext u X` has `LDL_{t,R}(y_I) ≤ k + ⌈log₂ B⌉ + aBlk`.

The description is the block index and its unregistered input bits: evaluated from them, the block holds the run's
values, since the run is the circuit's evaluation. The block size `c` is not used. -/
def BlockDescription : Prop :=
  ∀ (R : Committed) (k : ℕ) (D : Decoder) (t : ℕ) (ext : Fin R.B → R.d.D.Transcript → List Bool) (aBlk : ℕ)
    (inp : R.d.D.Transcript),
    ReadCap R.d R.P k → R.Honest (R.d.D.eval inp) → Assumptions.EvaluatesBlocks D t R ext aBlk →
      ∀ u, LDL D t (ext u (R.d.D.eval inp)) ≤ (k + Nat.clog 2 R.B + aBlk : ℕ)

/-- **Theorem 3 (counting)**: let `P` have exactly `c` operation gates per block, and give each gate `v` of a set `V`
of operation gates a counting weight `ω v`. If `q > 0` and every block's counted weight `Σ_{v ∈ V ∩ I} ω v` is at most
`q`, then `|C| ≥ (c / q) Σ_{v ∈ V} ω v`, and for a reference size `T₀ > 0`, `ρ = |C| / T₀ ≥ (c / (T₀ q)) Σ_{v ∈ V} ω v`.

Each gate of `V` lies in one block, so `Σ_{v ∈ V} ω v ≤ q B`, and `|C| = c B`. The weights need not be nonnegative. -/
def Counting : Prop :=
  ∀ (C : Circuit) (n c : ℕ) (P : Partition C n) (V : Finset (Fin C.N)) (ω : Fin C.N → ℝ) (q T₀ : ℝ),
    Exact P c → V ⊆ C.computed → 0 < q → 0 < T₀ → (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q) →
      c / q * ∑ v ∈ V, ω v ≤ C.size C.computed ∧ c / (T₀ * q) * ∑ v ∈ V, ω v ≤ C.size C.computed / T₀

/-- **Corollary 4b (activation-only fan-out bound)**: let `R` be a `(c,k)`-partition of `R.d` into `B = R.B` blocks, and
`M` counted product occurrences `J`, each assigned to the block of its representative gate, each activation occurrence
read by at most `e` of them. Let the run `X = eval inp` take the registered values. If the decoder evaluates blocks for
the activation tuples `a` within `t` at overhead `aBlk` (`EvaluatesBlocks`, Lemma 1's allowance) and they meet
`ActivationRate β Λ t`, `LDL_{t,R}(a_I) ≥ β |V_I| − Λ` with `β > 0`, then for `h = k + ⌈log₂ B⌉ + aBlk + Λ` and a
reference size `T₀ > 0`, `ρ = |C| / T₀ ≥ c β M / (T₀ e h)`.

Lemma 1 gives `β |V_I| − Λ ≤ k'`, so `|V_I| ≤ h / β`; fan-out gives `|𝓙_I| ≤ e |V_I| ≤ e h / β`; Theorem 3 with unit
weights on the representatives gives the bound. No weight-price premise is used, nor that a product's multiplication
subcircuit lies whole in its block: that is what makes `ActivationRate` plausible, and the counting doesn't read it.
When `e h ≤ 0` the bound is at most `0`. -/
def FanOutBound : Prop :=
  ∀ (R : Committed) (c k : ℕ) {ι A : Type} [DecidableEq A] (M : Products R.d.D ι A)
    (e : ℕ) (D : Decoder) (t aBlk : ℕ) (a : Fin R.B → R.d.D.Transcript → List Bool) (β Λ T₀ : ℝ)
    (inp : R.d.D.Transcript),
    CKPartition R.d R.P c k → R.Honest (R.d.D.eval inp) → M.FanOut e → 0 < β → 0 < T₀ →
    Assumptions.EvaluatesBlocks D t R a aBlk → Assumptions.ActivationRate β Λ t D R M a (R.d.D.eval inp) →
      c * β * M.J.card / (T₀ * e * (k + Nat.clog 2 R.B + aBlk + Λ)) ≤ R.d.D.size R.d.D.computed / T₀

end Nci.Ldl.Guarantees
