module

public import Proofs.ResourceAccounting.Lemmas.EncodeChain

/-! Steps of proofs that the encode chain's conjecture fails in the circuit model, where a Boolean circuit of any
size costs no unit. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Gates

open ResourceAccounting.Gates

/-- Every guarantee that takes `EncodeChainHardness` as a hypothesis at such an `F` and `k` (p1024 at `k = 3` among
them) holds vacuously. -/
def EncodeChainRefuted : Prop :=
  ∀ (F : Field) (k B E εnum εden : ℕ), εden ≤ 2 ^ 64 * εnum → εnum < εden → 0 < k → 2 ^ 66 ≤ F.p →
    F.p % 4 = 3 → F.n % 8 = 0 → 72 * ((k + 2) * F.bytes) < F.p → 0 < E →
    ¬ ResourceAccounting.Assumptions.EncodeChainHardness F k B E εnum εden

end ResourceAccounting.Guarantees.Gates
