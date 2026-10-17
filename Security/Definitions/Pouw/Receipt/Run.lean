import Definitions.Pouw.Receipt.Duo

/-!
# duo and fx1 over a run of words (trusted)

`duoRun` absorbs `N` words into duo's state, word `i` with its own keys `(k i, a i, r i)`. fx1, sm_120's fold, is one
keyed IMAD a word in front of duo: each word is multiplied by its key `m₁ i` first (`fx1Run`).
-/

namespace Pouw.Receipt

/-- duo over `N` words `d` from the state `s`, word `i` with XOR key `k i`, multiplier `a i` and rotation `r i`. -/
def duoRun {n N : ℕ} (k a : Fin N → BitVec n) (r : Fin N → ℕ) (s : BitVec n × BitVec n) (d : Fin N → BitVec n) :
    BitVec n × BitVec n :=
  Fin.foldl N (fun t i => duoWord (k i) (a i) (r i) t (d i)) s

/-- fx1 over `N` words `d` from the state `s`: duo on the words `d i · m₁ i`. -/
def fx1Run {n N : ℕ} (m₁ k a : Fin N → BitVec n) (r : Fin N → ℕ) (s : BitVec n × BitVec n) (d : Fin N → BitVec n) :
    BitVec n × BitVec n :=
  duoRun k a r s fun i => d i * m₁ i

end Pouw.Receipt
