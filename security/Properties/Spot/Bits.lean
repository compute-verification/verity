module

public import Definitions.Claims.Sampling
public import Spot.Bits

/-! The map between `Spot.Bits`' layout of a registered circuit file and the claims' program (`bitsGate`,
`bitsProg`). -/

@[expose] public section

namespace Properties.Spot

open _root_.Spot

/-- **The claims' gate an `LGate` is**, constructor for constructor. -/
def bitsGate : Bits.LGate → Claims.Sampling.BGate
  | .input => .input
  | .const b => .const b
  | .not i => .not i
  | .and ws => .and ws
  | .xor ws => .xor ws

def ofBGate : Claims.Sampling.BGate → Bits.LGate
  | .input => .input
  | .const b => .const b
  | .not i => .not i
  | .and ws => .and ws
  | .xor ws => .xor ws

/-- **The claims' program Bits lays out**: `Bits.prog kp κ`'s gates under `bitsGate`, returning its outputs'
positions. -/
def bitsProg (kp : Bits.KeyProg) (κ : ℕ → Option Bool) : Claims.Sampling.Prog where
  gates := (Bits.prog kp κ).1.map bitsGate
  outs := (Bits.prog kp κ).2

end Properties.Spot
