import Models.Accounting.Guarantees.Compose
import Proofs.Accounting.TS.Stages
import Proofs.Accounting.HMu

/-!
# PoUW's credit on an arbitrary declared circuit: proofs (open)

Every proof here is open (note:20261018T0346Z-draft-theorem-compose gives the order), and this file doesn't land until
none is a `sorry`:

1. `WorkWeighted`: `prob (t.1 = u)` counts `u`'s `Fin (wordsOf u)` tickets, so it is `workWeight u`.
2. `StageLuckOfTemplates`: induction on the window and its stages, `luckOf u · Pr[skip] ≤ p · Pr[skip]`.
3. `TemplateLocal`: `ReceiptLuckOfCoins` on each stage's `CoinRound`, carried to the stage's coin by its equivalence
   (`prob` is a count, so an equivalence keeps it).
4. `CompilerFloor`: `CompilerC`'s proof (`budget_tail`, `fixed_of_tail`) with `DIdealFloor`'s `floor_of_price` in place
   of `honest_of_price`.
5. `StageIdealFloor`: `StageBudget` at `r = 1`, then `CompilerFloor` at `Λ = 1/ln(1/p)`, where `ε_sample = 0`.
6. `CircuitCreditLive`: `SheetDIdealFloor`'s reduction of (I1)–(I5) to A1–A7, then `StageIdealFloor` at
   `δ = δ_s + S·(δ_bind + s·δ_ZK)` with steps 1–3, and `T_cred > 0` to clear the division.
7. `CircuitCredit`: step 6 and `SeedGrind`.
-/

namespace Accounting.SecurityProofs.Compose

theorem WorkWeighted : Accounting.Guarantees.Compose.WorkWeighted := by
  sorry

theorem StageLuckOfTemplates : Accounting.Guarantees.Compose.StageLuckOfTemplates := by
  sorry

theorem TemplateLocal : Accounting.Guarantees.Compose.TemplateLocal := by
  sorry

theorem CompilerFloor : Accounting.Guarantees.Compose.CompilerFloor := by
  sorry

theorem StageIdealFloor : Accounting.Guarantees.Compose.StageIdealFloor := by
  sorry

theorem CircuitCreditLive : Accounting.Guarantees.Compose.CircuitCreditLive := by
  sorry

theorem CircuitCredit : Accounting.Guarantees.Compose.CircuitCredit := by
  sorry

end Accounting.SecurityProofs.Compose
