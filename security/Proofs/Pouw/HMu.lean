module

public import Proofs.Pouw.HMu.Luck
public import Proofs.Pouw.HMu.Persist
public import Proofs.Pouw.HMu.Gaps
public import Proofs.ResourceAccounting.Lemmas.HMu

/-! Proofs of (H-μ⁺)'s guarantees. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.HMu

theorem LemmaB : ResourceAccounting.Guarantees.HMu.LemmaB := ResourceAccounting.HMuProofs.lemmaB
theorem CoinPointMass : ResourceAccounting.Guarantees.HMu.CoinPointMass := ResourceAccounting.HMuProofs.coinPointMass
theorem CoinPointMassHalf : ResourceAccounting.Guarantees.HMu.CoinPointMassHalf := ResourceAccounting.HMuProofs.coinPointMassHalf
theorem ReceiptLuckOfCoins : ResourceAccounting.Guarantees.HMu.ReceiptLuckOfCoins := ResourceAccounting.HMuProofs.receiptLuckOfCoins
theorem ReceiptLuckOfCoin : ResourceAccounting.Guarantees.HMu.ReceiptLuckOfCoin := ResourceAccounting.HMuProofs.receiptLuckOfCoin
theorem PersistentOneWord : ResourceAccounting.Guarantees.HMu.PersistentOneWord := ResourceAccounting.HMuProofs.persistentOneWord

end ResourceAccounting.SecurityProofs.HMu
