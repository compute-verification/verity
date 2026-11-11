module

public import Definitions.Claims.Sampling
public import Spot.Bits

/-!
# A sampled statement's bits as the claims' program (`Spot.Bits`)

`Spot.Bits` lays a registered circuit file out as a Boolean-profile program of `Bits.LGate`s; the claims read
a program as `Claims.Sampling.Prog`, of `Claims.Sampling.BGate`s, constructor for constructor the same gates. This is
the map between them, so a statement's program is Bits' layout read as the claims read it.

* `bitsGate g`: the claims' gate an `LGate` is, and `ofBGate` back.
* `bitsProg kp κ`: the claims' program `Bits.prog kp κ` is, its gates under `bitsGate` and its outputs' positions.

`Proofs.Properties.Spot.Bits` proves that the two maps are inverse and keep the gates read, that `bitsProg`'s
gates are `Bits.off kp kp.size`, and that gate `g` of `kp` sits at `Bits.posOf kp g`, the last of its lowered gates.
-/

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

/-- The `LGate` a claims' gate is. -/
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
