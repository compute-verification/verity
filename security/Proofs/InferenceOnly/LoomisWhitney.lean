module

public import Proofs.InferenceOnly.LoomisWhitneyBase
public import Proofs.InferenceOnly.Lemmas

/-!
# The discrete Loomis–Whitney inequality

`InferenceOnly.Guarantees.LoomisWhitney` is `loomis_whitney`, which `Proofs.InferenceOnly.LoomisWhitneyBase` proves on Mathlib alone, with
the forms the main theorem's proof uses (`card_products_le_sqrt`, `sum_card_products_le_sqrt`).
-/

@[expose] public section

theorem InferenceOnly.SecurityProofs.LoomisWhitney : InferenceOnly.Guarantees.LoomisWhitney := @InferenceOnly.SecurityProofs.loomis_whitney
