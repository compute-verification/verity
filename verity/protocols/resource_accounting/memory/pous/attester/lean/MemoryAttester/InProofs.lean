module

public import MemoryAttester.Verdict

/-!
# The session's clear checks

`pous.verifier.session_proofs`: what the verifier of PoUS's session checks in the clear beside its proofs, so that the
memory challenger's signed record never reaches it: the scheme accepted, the key trusted and every round's coin opening
(`clearReasons`). The reasons are the Python's, word for word.
-/

@[expose] public section

namespace MemoryAttester.InProofs

open Flock

def byJ (openings : Array CoinOpening) (j : Nat) : Option CoinOpening := openings.findRev? (·.j == j)

/-- `clear_reasons`: the scheme accepted, the key trusted, and every round's last opening opens its committed coin. -/
def clearReasons (schemes : Schemes) (trusted : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey : ByteArray) : Array String := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr scheme} is not accepted"
  | some _ =>
    if !trusted.contains publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
  for j in [0:c.k] do
    let ok := match byJ openings j with
      | some o => verifyCoin c o
      | none => false
    if !ok then reasons := reasons.push s!"round {j}'s index is not its committed coin"
  return reasons

def clearOk (schemes : Schemes) (trusted : List ByteArray) (c : ChallengePhaseCommitment) (openings : Array CoinOpening)
    (scheme : String) (publicKey : ByteArray) : Bool :=
  (clearReasons schemes trusted c openings scheme publicKey).isEmpty

end MemoryAttester.InProofs
