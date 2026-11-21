module

public import MemoryAttester.Verdict

/-!
# The session's clear checks

`pous.verifier.session_proofs`: what the verifier of PoUS's session checks in the clear beside its proofs, so that the
memory attester's signed record never reaches it, only its message `m` (a hiding commitment to the record, salted) and
the signature over it: the store the session audits is the registered one (its binding the store's root, its blocks the
store's rows), the scheme accepted, the key's SHA-512 one the configuration pins for the memory attester (`pinned`,
never a key the session carries on its own), the signature over `m` under that key, every round's index below the
store's blocks and every round's coin opening (`clearReasons`, which takes the registered store's root and number of
blocks). The reasons are the Python's, word for word.
-/

@[expose] public section

namespace MemoryAttester.InProofs

open Flock

def byJ (openings : Array CoinOpening) (j : Nat) : Option CoinOpening := openings.findRev? (·.j == j)

/-- The session audits another store than the registered one (`session_proofs.OTHER_STORE`). -/
def OTHER_STORE : String := "the session audits another store than the registered one: "

/-- A round whose index is no block of the store (`session_proofs.PAST_STORE`). -/
def PAST_STORE : String := "an index past the store: "

/-- A session signed under a key whose SHA-512 the configuration's pins don't give the memory attester
(`session_proofs.UNPINNED`). -/
def UNPINNED : String := "the session is signed by a key the configuration does not pin for the memory attester"

/-- A signature that doesn't verify over the message under the pinned key (`session_proofs.BAD_SIGNATURE`). -/
def BAD_SIGNATURE : String := "the signature does not verify"

/-- `clear_reasons`, in its order: the session's signed binding and block count are the registered store's root
`storeRoot` and `storeBlocks` rows (`OTHER_STORE`), the scheme accepted, the key's SHA-512 among `pinned` (`UNPINNED`),
`sig` an LMS signature of `m` under that key (`BAD_SIGNATURE`), and every round's last opening is an index below the
store's blocks (`PAST_STORE`) that opens its committed coin. -/
def clearReasons (schemes : Schemes) (pinned : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey m sig : ByteArray) (storeRoot : ByteArray) (storeBlocks : Nat) : Array String := Id.run do
  let mut reasons : Array String := #[]
  if c.binding != storeRoot then
    reasons := reasons.push s!"{OTHER_STORE}its binding {(hexOf c.binding).take 16}... is not the store's root {(hexOf storeRoot).take 16}..."
  if c.blocks != storeBlocks then
    reasons := reasons.push s!"{OTHER_STORE}it audits {c.blocks} blocks, the store has {storeBlocks}"
  match schemes.lookup scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr scheme} is not accepted"
  | some p =>
    if !pinned.contains (Sha512.hash publicKey) then reasons := reasons.push UNPINNED
    else
      let ok := match lmsVerify p publicKey c.session m sig with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push BAD_SIGNATURE
  for j in [0:c.k] do
    match byJ openings j with
    | some o =>
      if o.index ≥ storeBlocks then
        reasons := reasons.push s!"{PAST_STORE}round {j}'s index {o.index} is not below the store's {storeBlocks} blocks"
      else if !verifyCoin c o then reasons := reasons.push s!"round {j}'s index is not its committed coin"
    | none => reasons := reasons.push s!"round {j}'s index is not its committed coin"
  return reasons

def clearOk (schemes : Schemes) (pinned : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey m sig : ByteArray) (storeRoot : ByteArray) (storeBlocks : Nat) : Bool :=
  (clearReasons schemes pinned c openings scheme publicKey m sig storeRoot storeBlocks).isEmpty

end MemoryAttester.InProofs
