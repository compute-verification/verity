import Definitions.Nci.Assumptions.Hardness
import Definitions.Nci.Assumptions.Ldl

/-!
# Priced reads: the guarantees

* `FanOutBlockCap`: Corollary 4b per block, the per-block cap `q(c, k) = e (k + Λ) / β` on the products a block of
  at most `c` gates reading at most `k` unregistered input bits computes whole, under `Priced` at `β` per activation
  occurrence: the cap the serving-block bound sums over serving blocks.
* `FanOutBoundPriced`: Corollary 4b under `Priced` at `β` per activation occurrence, with no decoder and no Lemma 1.
* `PricedOfActivationRate`: the local-description-length route, `ActivationRate` with `EvaluatesBlocks`, gives
  `Priced`'s inequality at the blocks of `R`'s partition, at deficit `Λ + ⌈log₂ B⌉ + aBlk`. It does not give `Priced`,
  which is an inequality at every set of at most `c` gates. Corollary 4b's count reads the inequality at the blocks
  only, so the bound follows from it through `bound_of_blocks`, as from `Priced`.

The overhead `|C| / T₀` is written out: `|C|` is `size computed`, the number of operation gates, and `T₀ > 0` a
reference construction's size, which bounds the threshold.
-/

namespace Nci.Hardness.Guarantees

open Verity Nci.Ldl

/-- **Corollary 4b per block** (the per-block cap): let `J` be counted products, each reading one activation
occurrence `act p`, each activation occurrence read by at most `e` of them. If they are `Priced` at `β > 0` per
activation occurrence (`Operations.ofActs`), deficit `Λ` and block size `c`, with no side information, then every
block `I` of at most `c` operation gates reading at most `k` unregistered input bits computes at most
`q(c, k) = e (k + Λ) / β` of them whole.

`Priced` at `I` gives `β |Ops(I)| − Λ ≤ |in(I) \ θ| ≤ k`, so `I`'s products read at most `(k + Λ) / β` activation
occurrences, and fan-out gives at most `e` products for each. It is the serving-block bound's per-block cap (a) at unit
weights: a cap on every block of the size, not only on a partition's, so it needs no partition, no decoder and no
Lemma 1. -/
def FanOutBlockCap : Prop :=
  ∀ (d : Declaration) {ι V : Type} [DecidableEq V] (J : Finset ι) (gates : ι → Finset (Fin d.D.N)) (act : ι → V)
    (e c k : ℕ) (β Λ : ℝ),
    (Operations.ofActs d.D J gates act).FanOut e → 0 < β →
    Assumptions.Priced d (Operations.ofActs d.D J gates act) (fun _ => β) Λ c (fun _ => ∅) →
      ∀ I ⊆ d.D.computed, d.D.size I ≤ c → d.D.width (d.D.into I \ d.θ) ≤ k →
        (((Operations.ofActs d.D J gates act).whole I).card : ℝ) ≤ e * (k + Λ) / β

/-- **Corollary 4b under `Priced`**: let `P` be a `(c,k)`-partition of `d` with `B = n` blocks, and `J` counted
products, each reading one activation occurrence `act p` and lying whole in one block, each activation occurrence read
by at most `e` of them. If the products are `Priced` at `β` per activation occurrence (`Operations.ofActs`), deficit
`Λ` and block size `c`, with no side information, then for `h = k + Λ` and a reference size `T₀ > 0`, the overhead
`|C| / T₀ ≥ c β M / (T₀ e h)`, `M = #J`.

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

/-- **The LDL route gives `Priced`'s inequality at `R`'s blocks**: let the run `X = eval inp` take the registered
values of `R`, and give each counted product of `M` gates holding its representative. If the decoder evaluates blocks
for the activation tuples `a` within `t` at overhead `aBlk` (`EvaluatesBlocks`) and they meet `ActivationRate β Λ t`
with `β ≥ 0`, then at each block `I` of `R`'s partition `Priced`'s inequality holds at `β` per activation occurrence
and deficit `Λ' = Λ + ⌈log₂ B⌉ + aBlk`, with no side information: `β |Ops(I)| − Λ' ≤ |in(I) \ θ|`.

Lemma 1 without the read cap: the block's description is its index and its unregistered input bits, so
`β |V_I| − Λ ≤ LDL(a_I) ≤ ⌈log₂ B⌉ + |in(I) \ θ| + aBlk`; the products the block computes whole have their
representative in it, so their activation occurrences are among `V_I`. It gives the inequality at `R`'s blocks, not at
every set of at most `c` gates, so it is not `Priced`. Corollary 4b's count reads it at the blocks only:
`FanOutBoundPriced`'s bound follows from it through `bound_of_blocks`, at `h = k + Λ'`. -/
def PricedOfActivationRate : Prop :=
  ∀ (d : Declaration) (n : ℕ) (R : Registration d n) {ι A : Type} [DecidableEq A] (M : Products d.D ι A)
    (gates : ι → Finset (Fin d.D.N)) (D : Decoder) (t aBlk : ℕ) (a : Fin n → d.D.Transcript → List Bool) (β Λ : ℝ)
    (inp : d.D.Transcript),
    (∀ p ∈ M.J, M.rep p ∈ gates p) → 0 ≤ β → R.Consistent (d.D.eval inp) →
    Ldl.Assumptions.EvaluatesBlocks D t R a aBlk → Ldl.Assumptions.ActivationRate β Λ t D R M a (d.D.eval inp) →
      ∀ u, ∑ _v ∈ (Operations.ofActs d.D M.J gates M.act).ops (R.P.gates u), β - (Λ + Nat.clog 2 n + aBlk) ≤
        novelReads d (R.P.gates u)

end Nci.Hardness.Guarantees
