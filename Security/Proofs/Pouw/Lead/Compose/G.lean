import Definitions.Pouw.Guarantees.Lead
import Proofs.Pouw.Lead.MH100W.Floor
import Proofs.Pouw.Lead.CoinBranch.Writes
import Proofs.Pouw.Lead.Tags.Distinct
import Proofs.Pouw.Lead.PStar.Table
import Proofs.Pouw.Lead.Commit.Credit
import Proofs.Pouw.Lead.Network.Term

/-!
# The lead's guarantee from its pieces (COMPOSE; owner N-COMPOSE)

`G`'s proof, at each accepted coin value with no break, on the egress-free twin (`OutsourceWithinEgressLead`), which
is accepted where the original is (it holds what it holds, registers its root and exchanges nothing) and hashed no
collision where the original hashed none (beyond the original's, it hashes only the honest commitment's messages):
1. the twin's run holds every committed tile (`TileGrain`), so its whole-tile writes are at least the distinct
   committed tiles less the table hits (`CoinBranch.distinct_le_writes`), and the distinct tiles are the committed
   tiles less the repeats; the lead commits whole tiles, so their words are `m n S` (`perCoin`);
2. its hashing credit is at least the honest commitment's (`HonestHashRuled` at egress 0, `Commit.hashed_credit_ge`),
   at least the honest compressions' (`CommitL1w`), and its compressions write that many registers (`ShaWrites`);
3. `OneInstructionMinimal` and `MH100W.cost_ge_regs` (`W1`, `H32`) price those registers at 32 each (`perCoin`);
4. summed over `O`, the repeats and table hits are at most `|Ω|` times `coinBranchGap` and `tableGap`
   (`Tags.repeats_le`, `PStar.table_le`); the twin's saving is at most `betaLead` a bit the original exchanged, at
   most `callBits` at an accepted value (`Accepted`), so `|O|` times `32 · networkGap`; and the head, hiding and `p*`
   terms are nonnegative.
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-- Owner: N-COMPOSE. **One coin value's floor**: at a coin value where a realizable strategy exchanges nothing, is
accepted and hashed no collision, its priced W1 work is at least 32 times the honest registers less the repeated
and table-held tiles' words. -/
theorem perCoin (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (Real : Strategy (lead T m n K) → Prop)
    (hcommit : CommitL1w H C isA κ) (hpr : H32 pr) (hw1 : W1 pr Real) (hgrain : TileGrain stream Real)
    (hmin : OneInstructionMinimal Real) (hsha : ShaWrites isA Real) (hhash : HonestHashRuled H C κ Real)
    (σ : Strategy (lead T m n K)) (hσ : Real σ) (ω : Coins (lead T m n K)) (he : (σ.exec ω).egress = 0)
    (hacc : Accepted (lead T m n K) H C κ σ ω) (hnb : ¬ Break (lead T m n K) H C κ σ ω) :
    32 * (honestRegs (lead T m n K) C -
        tileWords * ((repeats (lead T m n K) κ ω : ℚ) + tableHits (lead T m n K) κ σ stream ω)) ≤
      (σ.work pr ω : ℚ) := by
  sorry

/-- Owner: N-COMPOSE. **The lead's guarantee** (`Pouw.Guarantees.Lead.G`), from `perCoin` on the twin and the
averaged bounds. -/
theorem G : Pouw.Guarantees.Lead.G := by
  sorry

/-- Owner: N-COMPOSE. **The honest lemma**: the nominal W1 table meets `H32` (`H32Sat`), and the honest strategy
(write each committed tile once, hash the honest commitment) with the protocol's coin stream and commitment meets
every premise and is accepted at every coin value. -/
theorem honest : Pouw.Guarantees.Lead.Honest := by
  sorry

end Pouw.SecurityProofs.Lead.Compose
