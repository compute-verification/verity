module

public import Proofs.Flock.Claims.Complete.Prover.Transcript
public import Flock.Zk

/-! The masked prover's tape: what `flock-circuit --zk` samples besides the plain prover's salts and nonces. -/

@[expose] public section

namespace FlockComplete

open Flock

/-- **The masked prover's tape**: the plain prover's, and every value `flock-circuit --zk` samples. -/
structure TapeZk extends Tape where
  h : Nat → Nat → Nat → F128
  mu : Nat → Nat → Nat → F128
  padH : Nat → Nat → Nat → F128
  padMu : Nat → Nat → Nat → F128
  tauSalt : Nat → Nat → ByteArray
  innerSalt : Nat → Nat → Nat → ByteArray
  extra : Nat → Nat → Nat → F128
  lanePad : Nat → Nat → Nat → F128
  mask : Nat → Nat → F128

instance instInhabitedTapeZk : Inhabited TapeZk :=
  ⟨{ toTape := default, h := fun _ _ _ => 0, mu := fun _ _ _ => 0, padH := fun _ _ _ => 0, padMu := fun _ _ _ => 0,
     tauSalt := fun _ _ => ByteArray.empty, innerSalt := fun _ _ _ => ByteArray.empty, extra := fun _ _ _ => 0,
     lanePad := fun _ _ _ => 0, mask := fun _ _ => 0 }⟩

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
