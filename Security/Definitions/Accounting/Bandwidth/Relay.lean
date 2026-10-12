import Definitions.Accounting.Bandwidth.Game

/-!
# The bandwidth audit v0 with relayed tags

`Game.lean`'s game, except that the side sources may depend on the oracle as well as the store. A peer that holds a
copy of `E` and computes tags after the seed can relay them over a side channel; each bit the chip reads that way is a
side read, counted against `c` as before. The on-chip state and device memory are still laid out from `E` before the
seed, so neither can depend on the oracle, and the sample is still drawn after the output.
-/

namespace Bandwidth

open Pous (Bits)

/-- A device whose side sources may depend on the oracle as well as the store, so a peer can relay tags it computed
after the seed. Its other fields and its bounds are `Device`'s: in every world the program reads at most `c` bits of
side sources, regenerates at most `G` blocks and makes at most `q` oracle queries. -/
structure RelayDevice (N b h S c G q : ℕ) where
  state : Store N b → Bits S
  mem : Store N b → ℕ → Bool
  side : Store N b → Oracle N b h → ℕ → Bool
  chip : Bits S → Chip N b h (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : World N b h),
    (chip st).sideBits w ≤ c ∧ (chip st).regens w ≤ G ∧ ((chip st).queries w).length ≤ q

namespace RelayDevice

variable {N b h S c G q : ℕ}

/-- The world a relay device's chip runs in, for store `E` and oracle `O`. -/
def world (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : World N b h :=
  ⟨O, A.mem E, A.side E O, E⟩

/-- The tags the device commits to. -/
def tags (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E O)

/-- The device's ingress in the window, in bits. -/
def ingress (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) : ℕ :=
  (A.chip (A.state E)).ingressBits (A.world E O)

/-- The verifier accepts: every sampled block's committed tag is the oracle's tag of that block. -/
def Accepts {s : ℕ} (A : RelayDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) (I : Fin s → Fin N) : Prop :=
  ∀ i, A.tags E O (I i) = O (I i, E (I i))

end RelayDevice

end Bandwidth
