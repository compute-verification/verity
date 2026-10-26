import Definitions.Nci.Assumptions.Hardness
import Definitions.Nci.Assumptions.Ldl

/-!
# Priced reads: the guarantees

* `FanOutBoundPriced`: Corollary 4b under `Priced` at `β` per activation occurrence, with no decoder and no Lemma 1.
* `PricedOfActivationRate`: the local-description-length route, `ActivationRate` with `EvaluatesBlocks`, gives
  `Priced`'s inequality at every block of the registered partition, at deficit `Λ + ⌈log₂ B⌉ + aBlk`, so Corollary 4b
  as `Nci.Ldl.Guarantees.FanOutBound` states it is a corollary of the priced bound.
-/

namespace Nci.Hardness.Guarantees

open Verity Nci.Ldl

/-- **Corollary 4b under `Priced`**: let `P` be a `(c,k)`-partition of `d` with `B = n` blocks, and `J` counted
products, each reading one activation occurrence `act p` and lying whole in one block, each activation occurrence read
by at most `e` of them. If the products are `Priced` at `β` per activation occurrence (`Operations.ofActs`), deficit
`Λ` and block size `c`, with no side information, then for `h = k + Λ` and a reference size `T₀ > 0`,
`ρ = |C| / T₀ ≥ c β M / (T₀ e h)`, `M = #J`.

A block is a set of `c` operation gates handed at most `k` bits, so `β |V_I| − Λ ≤ k`, `|V_I| ≤ h / β`; fan-out gives
`|𝓙_I| ≤ e |V_I|`; summing over the blocks with `|C| = c B` (Theorem 3's count) gives the bound. No decoder, no Lemma 1
and no extraction premise. When `e h ≤ 0` the bound is at most `0`. -/
def FanOutBoundPriced : Prop :=
  ∀ (d : Declaration) (n c k : ℕ) (P : Partition d.D n) {ι V : Type} [DecidableEq V] (J : Finset ι)
    (gates : ι → Finset (Fin d.D.N)) (act : ι → V) (e : ℕ) (β Λ T₀ : ℝ),
    CKPartition d P c k → (Operations.ofActs d.D J gates act).WholeIn P →
    (Operations.ofActs d.D J gates act).FanOut e → 0 < β → 0 < T₀ →
    Assumptions.Priced d (Operations.ofActs d.D J gates act) (fun _ => β) Λ c (fun _ => ∅) →
      c * β * J.card / (T₀ * e * (k + Λ)) ≤ d.D.size d.D.computed / T₀

/-- **The LDL route gives `Priced`** (`ActivationRate` ⇒ `Priced`): let the run `X = eval inp` take the registered
values of `R`, and give each counted product of `M` gates holding its representative. If the decoder evaluates blocks
for the activation tuples `a` within `t` at overhead `aBlk` (`EvaluatesBlocks`) and they meet `ActivationRate β Λ t`
with `β ≥ 0`, then at every block of `R`'s partition `Priced`'s inequality holds at `β` per activation occurrence and
deficit `Λ' = Λ + ⌈log₂ B⌉ + aBlk`, with no side information: `β |Ops(I)| − Λ' ≤ |in(I) \ θ|`.

Lemma 1 without the read cap: the block's description is its index and its unregistered input bits, so
`β |V_I| − Λ ≤ LDL(a_I) ≤ ⌈log₂ B⌉ + |in(I) \ θ| + aBlk`; the products the block computes whole have their
representative in it, so their activation occurrences are among `V_I`. It gives the inequality at the partition's blocks
only, which is all `FanOutBoundPriced`'s count uses. -/
def PricedOfActivationRate : Prop :=
  ∀ (d : Declaration) (n : ℕ) (R : Registration d n) {ι A : Type} [DecidableEq A] (M : Products d.D ι A)
    (gates : ι → Finset (Fin d.D.N)) (D : Decoder) (t aBlk : ℕ) (a : Fin n → d.D.Transcript → List Bool) (β Λ : ℝ)
    (inp : d.D.Transcript),
    (∀ p ∈ M.J, M.rep p ∈ gates p) → 0 ≤ β → R.Consistent (d.D.eval inp) →
    Ldl.Assumptions.EvaluatesBlocks D t R a aBlk → Ldl.Assumptions.ActivationRate β Λ t D R M a (d.D.eval inp) →
      ∀ u, ∑ _v ∈ (Operations.ofActs d.D M.J gates M.act).ops (R.P.gates u), β - (Λ + Nat.clog 2 n + aBlk) ≤
        novelReads d (R.P.gates u)

end Nci.Hardness.Guarantees
