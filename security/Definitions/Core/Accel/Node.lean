module

public import Definitions.Core.Accel.Card
public import Definitions.Core.Hardware

/-!
# A card as nodes of the hardware model (trusted)

A card's table gives two nodes of a machine (`Hardware.Network`), at a cycle of `τ` seconds. Its die (`Card.die`)
computes: a gate's kind is an ACCEL op or a branch, its resources are the card's (`Res`), each with the table's rate
times `τ` a cycle, and it holds the registers', shared memory's and L2's bits. Its device memory (`Card.mem`) holds its
bits and computes nothing. A server of `n` such cards with a host is `Card.server`: a die reads and writes its device
memory at the table's device-memory rate.
-/

@[expose] public section

open scoped NNRat

namespace Verity.Accel

namespace Card

variable (c : Card)

/-- **A card's die**, at the semantics `x` and a cycle of `τ` seconds. -/
def die (x : Open) (τ : ℚ≥0) : Hardware.Component where
  Kind := Option (Opcode x)
  Res := Res
  use
    | some o => c.use o
    | none => slot (1 / 32)
  cap r := c.ρ r * τ
  state := 32 * (c.size .reg + c.size .smem + c.size .l2)

/-- **A card's device memory**: its bits, and no gates. -/
def mem : Hardware.Component where
  Kind := Empty
  Res := Empty
  use := nofun
  cap := nofun
  state := 32 * c.size .dram

/-- **A server of `n` cards** with a host and the world outside. -/
def server (n : ℕ) (x : Open) (τ : ℚ≥0) (host outside : Hardware.Component) (bus peer net : ℕ) :
    Hardware.Network (Hardware.Loc n) :=
  Hardware.server n (c.die x τ) c.mem host outside ⌈32 * c.ρ .dram * τ⌉₊ ⌈32 * c.ρ .dram * τ⌉₊ bus peer net

end Card

end Verity.Accel
