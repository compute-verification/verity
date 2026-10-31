import Flock.With.Verify

/-!
# The honest prover's transcript

The plain prover's side of the replay transcript (`Flock.Transcript`). Its state is the verifier's own `Transcript`: what the
prover absorbs it absorbs with the verifier's actions (`label`, `absorbF`, `absorbSlice`, `absorbBytes`, `pow`, lifted by
`liftTM`), so the pending bytes evolve as the verifier's do. Only the squeeze differs: where the verifier reads round `k`
from the record, the prover sends the framed bytes as the round's message (`Asks.round`), takes the live coin it is
answered with, and records the round (its count, coins and framed bytes) in `rounds`, which is what the session record
lists.

The prover runs in any verdict monad `M` with the two sends (`Asks`): the link message, answered by the link's coin, and
a round's framed bytes, answered by the round's. A coin is its elements (`Nat → F128`); a round of `n` coins reads the
first `n`.
-/

namespace FlockComplete

open Flock Transcript

/-- **What the prover sends**: the link message (each table's `root_B` and public words) and a round's framed bytes, each
answered by its live coin. -/
structure Asks (M : Type → Type) where
  link : Array ByteArray → Array ByteArray → M (Nat → F128)
  round : ByteArray → M (Nat → F128)

/-- **The prover's tape**: the salt of each leaf of each Merkle tree (table, rep, level, leaf; level `0` is the table's
commitment, which both reps share) and each grinding nonce (table, rep, list, index; the lists are query, consistency
and claim). -/
structure Tape where
  salt : Nat → Nat → Nat → Nat → ByteArray
  nonce : Nat → Nat → Nat → Nat → UInt64

instance instInhabitedTape : Inhabited Tape := ⟨⟨fun _ _ _ _ => ByteArray.empty, fun _ _ _ _ => 0⟩⟩

/-- `n` bytes: `b`'s first `n`, zero-padded. -/
def fitBytes (n : Nat) (b : ByteArray) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - min n b.size) 0)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The prover's squeeze** (`Flock.Transcript` from the prover's side): a squeeze of `n = 0` frames its header; otherwise the framed
bytes are sent as round `k`, the round's first `n` coins returned, `pending` cleared, and the round recorded. -/
def squeezeP (io : Asks M) (kind : UInt8) (n : Nat) : TMW M (Array F128) := fun t =>
  let c := framed t kind n
  if n = 0 then pure (#[], { t with pending := c })
  else do
    let coin ← io.round c
    let v := (Array.range n).map coin
    pure (v, { t with pending := ByteArray.empty, k := t.k + 1, rounds := t.rounds.push ⟨n, ByteArray.empty, v, some c⟩ })

def sampleP (io : Asks M) : TMW M F128 := do
  let c ← squeezeP io 1 1
  return c.getD 0 0

def sampleVecP (io : Asks M) (n : Nat) : TMW M (Array F128) := squeezeP io 2 n

def sampleKP (io : Asks M) : TMW M F256 := do
  let c ← squeezeP io 2 2
  return ⟨c.getD 0 0, c.getD 1 0⟩

end FlockComplete
