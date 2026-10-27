import Proofs.Accounting.HMu.Luck
import Proofs.Accounting.HMu.Persist
import Proofs.Accounting.HMu.Gaps

/-!
# (H-μ⁺)'s guarantees

Each statement of `Definitions/Accounting/Guarantees/HMu.lean`, proved:

* `HMu/Luck.lean`: Lemma B, the point mass of fresh coins (Erdős's bound over any finite index, and `1/2` in any
  additive group), and the receipt luck of a round whose cheater doesn't read them;
* `HMu/Persist.lean`: Lemma B′ for a persistent pattern with one word before the checkpoint;
* `HMu/Gaps.lean`: why the rest stays a premise: a view that reads the coin is lucky on every deficient round, and two
  words in one unit cancel under a step injective in both arguments.
-/

namespace Accounting.SecurityProofs.HMu

theorem LemmaB : Accounting.Guarantees.HMu.LemmaB := Accounting.HMuProofs.lemmaB
theorem CoinPointMass : Accounting.Guarantees.HMu.CoinPointMass := Accounting.HMuProofs.coinPointMass
theorem CoinPointMassHalf : Accounting.Guarantees.HMu.CoinPointMassHalf := Accounting.HMuProofs.coinPointMassHalf
theorem ReceiptLuckOfCoins : Accounting.Guarantees.HMu.ReceiptLuckOfCoins := Accounting.HMuProofs.receiptLuckOfCoins
theorem ReceiptLuckOfCoin : Accounting.Guarantees.HMu.ReceiptLuckOfCoin := Accounting.HMuProofs.receiptLuckOfCoin
theorem PersistentOneWord : Accounting.Guarantees.HMu.PersistentOneWord := Accounting.HMuProofs.persistentOneWord

end Accounting.SecurityProofs.HMu
