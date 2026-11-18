module

public import Proofs.Pouw.HMu.Luck
public import Proofs.Pouw.HMu.Persist
public import Proofs.Pouw.HMu.Gaps
public import Proofs.ResourceAccounting.Lemmas.HMu

/-!
# (H-μ⁺)'s guarantees

Each statement of `Definitions/Pouw/Guarantees/HMu.lean`, proved:

* `HMu/Luck.lean`: Lemma B, the point mass of fresh coins (Erdős's bound over any finite index, and `1/2` in any
  additive group), and the receipt luck of a round whose cheater doesn't read them;
* `HMu/Persist.lean`: Lemma B′ for a persistent pattern with one word before the checkpoint;
* `HMu/Gaps.lean`: why the rest stays a premise: a view that reads the coin is lucky on every deficient round, and two
  words in one unit cancel under a step injective in both arguments.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.HMu

theorem LemmaB : ResourceAccounting.Guarantees.HMu.LemmaB := ResourceAccounting.HMuProofs.lemmaB
theorem CoinPointMass : ResourceAccounting.Guarantees.HMu.CoinPointMass := ResourceAccounting.HMuProofs.coinPointMass
theorem CoinPointMassHalf : ResourceAccounting.Guarantees.HMu.CoinPointMassHalf := ResourceAccounting.HMuProofs.coinPointMassHalf
theorem ReceiptLuckOfCoins : ResourceAccounting.Guarantees.HMu.ReceiptLuckOfCoins := ResourceAccounting.HMuProofs.receiptLuckOfCoins
theorem ReceiptLuckOfCoin : ResourceAccounting.Guarantees.HMu.ReceiptLuckOfCoin := ResourceAccounting.HMuProofs.receiptLuckOfCoin
theorem PersistentOneWord : ResourceAccounting.Guarantees.HMu.PersistentOneWord := ResourceAccounting.HMuProofs.persistentOneWord

end ResourceAccounting.SecurityProofs.HMu
