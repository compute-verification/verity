import Definitions.Accounting.Bandwidth.Game

/-!
# The bandwidth audit v0 with a second secret the chip's view determines

`Game.lean`'s game for a device that also holds `z`, `n` bits uniform and independent of the store, in the sense of
bandwidth additivity (Mbw, section 5 of `note:console/20261012T0645Z-draft-actual-circuit-model`): the chip's view in
the window (its state at the seed and every answer it read) determines `z`. The device lays out its on-chip state,
device memory and side sources from the store and `z`, and its chip outputs `z` along with its tags. That is the same
as the view determining `z`, since a chip program's output is any function of its state and of the answers it read.
-/

namespace Bandwidth

open Pous (Bits)

/-- A device whose layout is any function of the store and of `n` more bits `z`, and whose chip outputs its tags and a
value of `n` bits. Its bounds are `Device`'s: in every world the program reads at most `c` bits of side sources,
regenerates at most `G` blocks and makes at most `q` oracle queries. -/
structure JointDevice (N b h S c G q n : ℕ) where
  state : Store N b → Bits n → Bits S
  mem : Store N b → Bits n → ℕ → Bool
  side : Store N b → Bits n → ℕ → Bool
  chip : Bits S → Chip N b h ((Fin N → Bits h) × Bits n)
  bounded : ∀ (st : Bits S) (w : World N b h),
    (chip st).sideBits w ≤ c ∧ (chip st).regens w ≤ G ∧ ((chip st).queries w).length ≤ q

namespace JointDevice

variable {N b h S c G q n : ℕ}

/-- The world the chip runs in, for store `E`, the bits `z` and oracle `O`. -/
def world (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (O : Oracle N b h) : World N b h :=
  ⟨O, A.mem E z, A.side E z, E⟩

/-- What the chip outputs: its tags and its value. -/
def out (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (O : Oracle N b h) :
    (Fin N → Bits h) × Bits n :=
  (A.chip (A.state E z)).run (A.world E z O)

/-- The device's ingress in the window, in bits. -/
def ingress (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (O : Oracle N b h) : ℕ :=
  (A.chip (A.state E z)).ingressBits (A.world E z O)

/-- The chip's view determines `z`: in every world of a store, `z` and an oracle, its value is `z`. -/
def Recovers (A : JointDevice N b h S c G q n) : Prop :=
  ∀ E z O, (A.out E z O).2 = z

/-- The verifier accepts: every sampled block's committed tag is the oracle's tag of that block. -/
def Accepts {s : ℕ} (A : JointDevice N b h S c G q n) (E : Store N b) (z : Bits n) (O : Oracle N b h)
    (I : Fin s → Fin N) : Prop :=
  ∀ i, (A.out E z O).1 (I i) = O (I i, E (I i))

end JointDevice

end Bandwidth
