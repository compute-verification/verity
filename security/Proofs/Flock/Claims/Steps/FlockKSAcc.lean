module

public import Proofs.Flock.Claims.Bridge.Coins
public import Proofs.Flock.Claims.KOf
public import Proofs.Flock.Claims.KS.Joint

/-! # What the plain KS proofs read of `Claims.Recursion.FlockKSAcc`, without its recursion imports. -/

@[expose] public section

namespace Claims.Recursion

open Claims Claims.Flock

section CodeFacts

variable {Coin : Type}

/-- **The verifier accepts only at its last round** (a code fact): wherever it accepts, with one coin per message, the
prover sent `k(x)` messages. -/
def AccLast (H : Msg → Msg) (V : Verifier Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin), cs.length = as.length → ((V.next x as cs).eval H).1 = .acc →
    as.length = V.k x

end CodeFacts

end Claims.Recursion
