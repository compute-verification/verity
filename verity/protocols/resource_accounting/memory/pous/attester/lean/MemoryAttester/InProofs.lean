module

public import MemoryAttester.Verdict

/-!
# The session's clear checks

`pous.verifier.session_proofs`: what the verifier of PoUS's session checks in the clear beside its proofs, so that the
memory challenger's signed record never reaches it: the store the session audits is the registered one (its binding the
store's root, its blocks the store's rows), the scheme accepted, the key trusted, every round's index below the store's
blocks and every round's coin opening (`clearReasons`, which takes the registered store's root and number of blocks). The
reasons are the Python's, word for word.
-/

@[expose] public section

namespace MemoryAttester.InProofs

open Flock

def byJ (openings : Array CoinOpening) (j : Nat) : Option CoinOpening := openings.findRev? (·.j == j)

/-- The session audits another store than the registered one (`session_proofs.OTHER_STORE`). -/
def OTHER_STORE : String := "the session audits another store than the registered one: "

/-- A round whose index is no block of the store (`session_proofs.PAST_STORE`). -/
def PAST_STORE : String := "an index past the store: "

/-- `clear_reasons`, in its order: the session's signed binding and block count are the registered store's root
`storeRoot` and `storeBlocks` rows (`OTHER_STORE`), the scheme accepted, the key trusted, and every round's last opening
is an index below the store's blocks (`PAST_STORE`) that opens its committed coin. -/
def clearReasons (schemes : Schemes) (trusted : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey : ByteArray) (storeRoot : ByteArray) (storeBlocks : Nat) : Array String := Id.run do
  let mut reasons : Array String := #[]
  if c.binding != storeRoot then
    reasons := reasons.push s!"{OTHER_STORE}its binding {(hexOf c.binding).take 16}... is not the store's root {(hexOf storeRoot).take 16}..."
  if c.blocks != storeBlocks then
    reasons := reasons.push s!"{OTHER_STORE}it audits {c.blocks} blocks, the store has {storeBlocks}"
  match schemes.lookup scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr scheme} is not accepted"
  | some _ =>
    if !trusted.contains publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
  for j in [0:c.k] do
    match byJ openings j with
    | some o =>
      if o.index ≥ storeBlocks then
        reasons := reasons.push s!"{PAST_STORE}round {j}'s index {o.index} is not below the store's {storeBlocks} blocks"
      else if !verifyCoin c o then reasons := reasons.push s!"round {j}'s index is not its committed coin"
    | none => reasons := reasons.push s!"round {j}'s index is not its committed coin"
  return reasons

def clearOk (schemes : Schemes) (trusted : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey : ByteArray) (storeRoot : ByteArray) (storeBlocks : Nat) : Bool :=
  (clearReasons schemes trusted c openings scheme publicKey storeRoot storeBlocks).isEmpty

end MemoryAttester.InProofs
