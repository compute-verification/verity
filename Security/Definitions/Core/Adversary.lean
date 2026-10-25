import Mathlib.Data.Nat.Notation

/-!
# An audit window's adversary (trusted)

The strategy a store audit's game plays against: the device of the bandwidth audit (`Bandwidth.Device`) and of the rate
audit (`Accounting.Rate.Device`), which are the same game up to the rate audit's certified stream. Before the window's
coins it lays out, from the audit's input `X` (the store), an on-chip state of `S` bits, which is all of the input the
program sees directly. In the window its program runs from that state in a world `W`, built from the input and the
coins `C`, which answers the program's oracle queries (of type `Q`), its device-memory reads and its side-link reads,
and outputs `Y`, the commitment the verifier checks. In every world the program makes at most `q` oracle queries and
takes in at most `L` bits over the side link.

An `InteractiveAdversary` is an adversary whose side sources answer what its program sent: each read's answer is any
function of the input, the coins and every bit sent so far, and the program runs from nothing sent.

These are interfaces: each game's device is an instance, and keeps the steps its own program takes.
-/

namespace Verity

/-- **An audit window's adversary**: an on-chip state of `S` bits laid out from the input `X` before the coins `C`, and
a program run from it in the world `W` the input and the coins give, making at most `q` oracle queries (of `Q`) and
taking in at most `L` bits over the side link in every world, and ending in an output `Y`. -/
class Adversary (A : Type) (X C : outParam Type) (S : outParam ℕ) (W Q Y : outParam Type) (L q : outParam ℕ) where
  /-- the on-chip state laid out from the input before the coins -/
  state : A → X → Fin S → Bool
  /-- the world the program runs in, from the input and the coins -/
  world : A → X → C → W
  /-- the program's output from a state in a world -/
  run : A → (Fin S → Bool) → W → Y
  /-- the oracle queries the program makes from a state in a world, in order -/
  queries : A → (Fin S → Bool) → W → List Q
  /-- the bits the program reads from device memory from a state in a world -/
  ingressBits : A → (Fin S → Bool) → W → ℕ
  /-- the bits the program takes in over the side link from a state in a world -/
  linkBits : A → (Fin S → Bool) → W → ℕ
  queries_le : ∀ (a : A) (st : Fin S → Bool) (w : W), (queries a st w).length ≤ q
  linkBits_le : ∀ (a : A) (st : Fin S → Bool) (w : W), linkBits a st w ≤ L

/-- **An interactive adversary**: an adversary whose side sources answer a read at an address as any function of the
input, the coins and the bits its program has sent so far (newest first). -/
class InteractiveAdversary (A : Type) (X C : outParam Type) (S : outParam ℕ) (W Q Y : outParam Type)
    (L q : outParam ℕ) extends Adversary A X C S W Q Y L q where
  /-- the side sources' answer to a read at an address, after the bits sent so far -/
  side : A → X → C → List Bool → ℕ → Bool

namespace Adversary

variable {A X C W Q Y : Type} {S L q : ℕ} [Adversary A X C S W Q Y L q]

/-- The adversary's output on input `x` at coins `c`: its program run from the state it laid out from `x`. -/
def output (a : A) (x : X) (c : C) : Y := run a (state a x) (world a x c)

/-- The adversary's ingress on input `x` at coins `c`: every bit its program read from device memory. -/
def ingress (a : A) (x : X) (c : C) : ℕ := ingressBits a (state a x) (world a x c)

end Adversary

end Verity
