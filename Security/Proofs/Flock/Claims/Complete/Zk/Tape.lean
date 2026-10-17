import Proofs.Flock.Claims.Complete.Prover.Transcript
import Flock.Zk

/-!
# The masked prover's tape and pads

What `flock-circuit --zk` draws besides the plain prover's salts and nonces (`Tape`), as tape entries: per table `i`
and rep `r`, the pads `h` and the dot-product mask `μ` (index `k < s`, the layout's length), each one's RS padding
(`k < K_PAD`), `τ`'s salt and the pads' commitment's leaf salts; per table, the four extra lanes' entries and each
level-0 lane's padding coefficients (lane `l < 2^lanes + 4`, `k < t_pad`).

* **A rep's pads** (`padsOf`, upstream's `Pads::draw`): `h` from the tape with `h_ab = h_a·h_b` and `w = x·y` set, so
  the zerocheck's product and the triple hold at every tape.
-/

namespace FlockComplete

open Flock

/-- **The masked prover's tape**: the plain prover's, and every value `flock-circuit --zk` draws. -/
structure TapeZk extends Tape where
  h : Nat → Nat → Nat → F128
  mu : Nat → Nat → Nat → F128
  padH : Nat → Nat → Nat → F128
  padMu : Nat → Nat → Nat → F128
  tauSalt : Nat → Nat → ByteArray
  innerSalt : Nat → Nat → Nat → ByteArray
  extra : Nat → Nat → Nat → F128
  lanePad : Nat → Nat → Nat → F128

instance instInhabitedTapeZk : Inhabited TapeZk :=
  ⟨{ toTape := default, h := fun _ _ _ => 0, mu := fun _ _ _ => 0, padH := fun _ _ _ => 0, padMu := fun _ _ _ => 0,
     tauSalt := fun _ _ => ByteArray.empty, innerSalt := fun _ _ _ => ByteArray.empty, extra := fun _ _ _ => 0,
     lanePad := fun _ _ _ => 0 }⟩

/-- **One rep's pads** (upstream's `Pads`): `h`, `μ`, their RS padding, and `τ`'s salt. -/
structure PadsZk where
  h : Array F128
  mu : Array F128
  padH : Array F128
  padMu : Array F128
  tauSalt : ByteArray
  deriving Inhabited

/-- **Table `i`'s rep `r`'s pads** from the tape (`Pads::draw`): `h_ab = h_a·h_b` and `w = x·y`. -/
def padsOf (tape : TapeZk) (i r : Nat) (lay : Zk.PadLayout) : PadsZk :=
  let h0 := (Array.range lay.len).map (tape.h i r)
  let h1 := h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)
  { h := h1.set! lay.tw (h1[lay.tx]! * h1[lay.ty]!), mu := (Array.range lay.len).map (tape.mu i r),
    padH := (Array.range Zk.K_PAD).map (tape.padH i r), padMu := (Array.range Zk.K_PAD).map (tape.padMu i r),
    tauSalt := fitBytes Zk.TAU_SALT_BYTES (tape.tauSalt i r) }

end FlockComplete
