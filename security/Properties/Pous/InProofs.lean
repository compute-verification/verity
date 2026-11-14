module

public import Properties.Pous.MemoryChallenger
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512

/-!
# PoUS's session inside proofs: accepted proofs give a session `verdict` accepts

In PoUS's session checked inside proofs, the memory challenger's signed record never reaches the verifier. The session of
`c.k` rounds under the commitment `c` is `c.k + 2` proof blocks: the signature's (`0`: the LMS signature verifies on the
record digest `m` under the pinned key), the signed digest's (`1`: `m` is the record digest of the session's rounds and how
it ended, the session ended complete after `c.k` rounds, every round on time), and one a round (`j + 2`: round `j`'s
index is its public coin `idx j`, below `c.blocks`, and its digest is the expected digest of that block of the store). The
digest `m` is one registered row the first two read, and each round's record one the second and that round's read.

The verifier accepts the session when every statement is accepted, each proof block's registered output is `1`, and its
clear checks pass on what is public (`ClearOk`: the scheme is accepted, the key trusted, and each round's coin opening
verifies to its index).

`SessionGives` states what that acceptance gives, in `Verity.InProofs`'s shape: the session the extraction opens
(`sessionOf`) is one `MemoryChallenger.verdict` accepts, or the run exhibits a SHA-512 collision, except with the proof
system's probability `ε` that an accepted session has a proof block whose values its circuit rejects. It takes three
premises, all named and all open:

* the proof system's half (`Verity.InProofs.Gives` at `K₀ = 1`): `IntegrityPeriod` at `K₀ = 1` over the session's
  statements, its sampling term `0` under `All`, so `ε` is the statements' `ε_ks + δ_link` summed;
* each proof block refines its piece of today's check (`SessionUnitsRefine`);
* the carried rows (`Carried`): an accepted run's two extractions open the digest `m`, and each round's record, to one
  value, or the run's SHA-512 inputs hold a collision (proofs' per-session form with carried leaves).

`VerdictSigned` and the PoUS certificate read the verdict's acceptance and stand on this unchanged.
-/

@[expose] public section

namespace MemoryChallenger.Properties.InProofs

open Verity.InProofs Verity.Game
open scoped ENNReal

/-- **The values a session's proof block opens**, each kind reading its own: the signature's the digest `m` and the
signature `sig`; the signed digest's `m`, how the session ended (`ended`) and its rounds; a round's its record
(`round`). -/
structure Vals where
  m : ByteArray
  sig : ByteArray
  ended : String
  rounds : Array Round
  round : Round

/-- **The session the extraction opens**: the commitment, scheme and key the verifier holds, how it ended and its rounds
as the signed digest's proof block opens them, and the signature as the signature's opens it. -/
def sessionOf (c : SessionCommitment) (scheme : String) (pk : ByteArray) (sd sg : Vals) : SignedSession :=
  { commitment := c, ended := sd.ended, rounds := sd.rounds, scheme := scheme, publicKey := pk, signature := sg.sig }

/-- **The verifier's clear checks**: the scheme is accepted, the key trusted, and each round `j`'s coin opening (the last
for `j`, as `verdict` reads it) opens its coin to the public index `idx j`. -/
def ClearOk (schemes : Schemes) (trusted : List ByteArray) (scheme : String) (pk : ByteArray) (c : SessionCommitment)
    (openings : Array CoinOpening) (idx : Nat → Nat) : Prop :=
  (∃ p, schemes.lookup scheme = some p) ∧ pk ∈ trusted ∧
    ∀ j < c.k, ∃ o, openings.findRev? (·.j == j) = some o ∧ o.index = idx j ∧ verifyCoin c o = true

/-- **Each proof block's piece of today's check**: the signature's, `lmsVerify` on `m` under the scheme's parameters; the
signed digest's, `m` the record digest of its rounds, ended complete after `c.k` rounds, each on time; round `j`'s, its
record is round `j` at the index `idx j`, below `c.blocks`, with the expected digest. -/
def PieceOk (schemes : Schemes) (scheme : String) (pk : ByteArray) (c : SessionCommitment)
    (expected : Nat → Option ByteArray) (idx : Nat → Nat) (u : Fin (c.k + 2)) (v : Vals) : Prop :=
  if u.val = 0 then ∀ p, schemes.lookup scheme = some p → lmsVerify p pk c.session v.m v.sig = .ok true
  else if u.val = 1 then
    message c v.ended v.rounds = .ok v.m ∧ v.ended = COMPLETE ∧ v.rounds.size = c.k ∧
      ∀ j (hj : j < v.rounds.size), 0 ≤ v.rounds[j].elapsedNs ∧ v.rounds[j].elapsedNs ≤ c.limitNs
  else
    v.round.j = u.val - 2 ∧ v.round.index = idx (u.val - 2) ∧ v.round.index < c.blocks ∧
      expected v.round.index = some v.round.digest

/-- **The session's proof blocks refine today's check**: every value a proof block's circuit accepts (`right u`) meets
its piece of today's check. -/
def SessionUnitsRefine (schemes : Schemes) (scheme : String) (pk : ByteArray) (c : SessionCommitment)
    (expected : Nat → Option ByteArray) (idx : Nat → Nat) (right : Fin (c.k + 2) → Vals → Prop) : Prop :=
  Refines (V := fun _ => Vals) right (PieceOk schemes scheme pk c expected idx)

/-- **The carried rows open to one value, or the run carries a collision**: whenever the verifier accepts, the
signature's and the signed digest's proof blocks open `m` to one value and each round's proof block opens the record the
signed digest's holds at its round, or the run's SHA-512 inputs (`hashInputs`) hold a collision. -/
def Carried {Ω : Type} (c : SessionCommitment) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals)
    (hashInputs : Ω → List Hash.Msg) : Prop :=
  ∀ ω, r.accepts ω →
    ((r.ports ⟨0, by omega⟩ ω).m = (r.ports ⟨1, by omega⟩ ω).m ∧
      ∀ j (hj : j < c.k), (r.ports ⟨1, by omega⟩ ω).rounds[j]? = some (r.ports ⟨j + 2, by omega⟩ ω).round) ∨
    Hash.CollisionIn Verity.sha512 (hashInputs ω)

/-- **An accepted session inside proofs gives a session `verdict` accepts, except `ε`.** For the commitment `c`, the
coin openings `openings`, the scheme and key the verifier holds, the public indices `idx` and the expected digests
`expected`, and a run `r` of the session's `c.k + 2` proof blocks: if the proof system gives that an accepted run leaves
no proof block whose values its circuit (`right`) rejects, except with probability `ε`, the proof blocks refine today's
check, and the carried rows open to one value or the run carries a collision, then the verifier accepts the run and its
clear checks pass while `verdict` rejects the session the extraction opens and the run's SHA-512 inputs hold no
collision, with probability at most `ε`. -/
def SessionGives : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (c : SessionCommitment)
    (openings : Array CoinOpening) (scheme : String) (pk : ByteArray) (idx : Nat → Nat)
    (Ω : Type) (μ : PMF Ω) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals) (right : Fin (c.k + 2) → Vals → Prop)
    (hashInputs : Ω → List Hash.Msg) (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    SessionUnitsRefine schemes scheme pk c expected idx right →
    Carried c r hashInputs →
    prPMF μ (fun ω => r.accepts ω ∧ ClearOk schemes trusted scheme pk c openings idx ∧
        (verdict schemes trusted expected
          (sessionOf c scheme pk (r.ports ⟨1, by omega⟩ ω) (r.ports ⟨0, by omega⟩ ω)) c openings).accepted = false ∧
        ¬ Hash.CollisionIn Verity.sha512 (hashInputs ω)) ≤ ε

end MemoryChallenger.Properties.InProofs
