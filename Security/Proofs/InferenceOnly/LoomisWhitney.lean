import Definitions.InferenceOnly.Guarantees
import Proofs.InferenceOnly.LoomisWhitneyBase

/-!
# The discrete Loomis–Whitney inequality

`InferenceOnly.Guarantees.LoomisWhitney` is `loomis_whitney`, which `Proofs.InferenceOnly.LoomisWhitneyBase` proves on Mathlib alone, with
the forms the main theorem's proof uses (`card_products_le_sqrt`, `sum_card_products_le_sqrt`).
-/

theorem InferenceOnly.SecurityProofs.LoomisWhitney : InferenceOnly.Guarantees.LoomisWhitney := @InferenceOnly.SecurityProofs.loomis_whitney
