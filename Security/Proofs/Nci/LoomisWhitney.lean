import Definitions.Nci.Guarantees
import Proofs.Nci.LoomisWhitneyBase

/-!
# The discrete Loomis–Whitney inequality

`Nci.Guarantees.LoomisWhitney` is `loomis_whitney`, which `Proofs.Nci.LoomisWhitneyBase` proves on Mathlib alone, with
the forms the main theorem's proof uses (`card_products_le_sqrt`, `sum_card_products_le_sqrt`).
-/

theorem Nci.SecurityProofs.LoomisWhitney : Nci.Guarantees.LoomisWhitney := @Nci.SecurityProofs.loomis_whitney
