module

public import Definitions.ResourceAccounting.Gates.Codec

/-! The encode bound's falsified conjecture, kept beside its refutation (`Gates.EncodeChainRefuted`); no guarantee
rests on it. -/

@[expose] public section

namespace ResourceAccounting.Assumptions

open ResourceAccounting.Gates

/-- `encode-chain-hardness`: computing a word of the codec's encoding costs `E` budgeted units; falsified
(`ResourceAccounting.SecurityProofs.Gates.EncodeChainRefuted`). It is claimed only at `ε ≥ 2^−64`, `k ≥ 1`,
`p ≥ 2^66` and `p ≡ 3 mod 4`, and says nothing outside them. -/
def EncodeChainHardness (F : Field) (k B E εnum εden : ℕ) : Prop :=
  εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ F.p → F.p % 4 = 3 →
  ∀ {ι : Type} (C : IntDag ι) (m : ℕ → ℕ) (x : Sample k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    Restricted B C → CodecInputs F C (samples F k) m x → words.Nodup → (∀ i ∈ words, i < F.p) →
    ComputesWords F k C x m words out εnum εden →
    words.length * E * (F.n - 1) ≤ budgeted C * (F.n - 1) + runBits C * E

end ResourceAccounting.Assumptions
