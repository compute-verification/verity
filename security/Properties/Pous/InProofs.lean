module

public import Properties.Pous.MemoryChallenger
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Mathlib.Probability.ProbabilityMassFunction.Monad
public import Lms.Sign

/-!
# PoUS's session inside proofs: an accepted run gives a session `verdict` accepts

`SessionGives` holds for any run of a session's proof blocks that meets its three hypotheses (`Gives` at the relation
`right`, `right ≤ PieceOk` and `Carried`); it says nothing of the deployed run until proofs' period form of V5 and their
per-session form with carried leaves give `Gives` and `Carried` there.

In PoUS's session checked inside proofs, the memory challenger's signed record never reaches the verifier. The session of
`c.k` rounds under the commitment `c` is `c.k + 2` proof blocks: the signature's (`0`: the LMS signature verifies on the
record digest `m` under the pinned key), the signed digest's (`1`: `m` is the record digest of the session's rounds and how
it ended, the session ended complete after `c.k` rounds, every round on time), and one a round (`j + 2`: round `j`'s
index is its public coin `idx j`, below `c.blocks`, and its digest is the expected digest of that block of the store). The
digest `m` is one registered row the first two read, and each round's record one the second and that round's read. The
verifier's clear checks run on what is public (`ClearOk`: the scheme is accepted, the key trusted, and each round's coin
opening verifies to its index).

`SessionGives`, in `Verity.InProofs`'s shape at `K₀ = 1`: if an accepted run leaves a proof block whose value `right`
rejects with probability at most `ε`, `right ≤ PieceOk`, and an accepted run's proof blocks open the carried rows to one
value or its SHA-512 inputs hold a collision (`Carried`), then the event `r.accepts ∧ ClearOk ∧ verdict rejects the
session the run opens (sessionOf) ∧ no collision in r.hashInputs` has probability at most `ε`. The guarantee that
instantiates it builds `r.ports` from the counted extractor `IntegrityPeriod` reads and `r.hashInputs` from the run, sets
`right` to each proof block's circuit, takes `Gives` from `IntegrityPeriod` at `K₀ = 1` over the session's statements
(sampling term `0` under `All`, so `ε` is the statements' `ε_ks + δ_link` summed) and `Carried` from the carried form;
`cr/sha-512` then bounds the collision at a finder built from the prover, outside this lemma.
-/

@[expose] public section

namespace MemoryChallenger.Properties.InProofs

open Verity.InProofs Verity.Game
open scoped ENNReal

/-- **The values at a session's proof block's ports**, each kind reading its own: the signature's the digest `m` and the
signature `sig`; the signed digest's `m`, how the session ended (`ended`) and its rounds; a round's its record
(`round`). -/
structure Vals where
  m : ByteArray
  sig : ByteArray
  ended : String
  rounds : Array Round
  round : Round

/-- **The session a run opens**: the commitment, scheme and key the verifier holds, how it ended and its rounds at the
signed digest's proof block, and the signature at the signature's. -/
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

/-- **The carried rows open to one value, or the run carries a collision**: whenever the verifier accepts, the
signature's and the signed digest's proof blocks hold one `m` and each round's proof block holds the record the signed
digest's holds at its round, or the run's SHA-512 inputs hold a collision. -/
def Carried {Ω : Type} (c : SessionCommitment) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals) : Prop :=
  ∀ ω, r.accepts ω →
    ((r.ports ⟨0, by omega⟩ ω).m = (r.ports ⟨1, by omega⟩ ω).m ∧
      ∀ j (hj : j < c.k), (r.ports ⟨1, by omega⟩ ω).rounds[j]? = some (r.ports ⟨j + 2, by omega⟩ ω).round) ∨
    Hash.CollisionIn Verity.sha512 (r.hashInputs ω)

/-- **An accepted session gives a session `verdict` accepts, except `ε`, for any run meeting its hypotheses**; it says
nothing of the deployed run until proofs' period form of V5 and carried form give `Gives` and `Carried` there. For the
commitment `c`, the coin openings `openings`, the scheme and key the verifier holds, the public indices `idx` and the
expected digests `expected`, and a run `r` of the session's `c.k + 2` proof blocks: if an accepted run leaves a proof block
whose value `right` rejects with probability at most `ε`, every value `right` accepts meets its piece of today's check
(`PieceOk`), and `Carried` holds, then `r.accepts ∧ ClearOk ∧` `verdict` rejects the session the run opens `∧` no
collision in `r.hashInputs` has probability at most `ε`. -/
def SessionGives : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (c : SessionCommitment)
    (openings : Array CoinOpening) (scheme : String) (pk : ByteArray) (idx : Nat → Nat)
    (Ω : Type) (μ : PMF Ω) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals) (right : Fin (c.k + 2) → Vals → Prop)
    (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    right ≤ PieceOk schemes scheme pk c expected idx →
    Carried c r →
    prPMF μ (fun ω => r.accepts ω ∧ ClearOk schemes trusted scheme pk c openings idx ∧
        (verdict schemes trusted expected
          (sessionOf c scheme pk (r.ports ⟨1, by omega⟩ ω) (r.ports ⟨0, by omega⟩ ω)) c openings).accepted = false ∧
        ¬ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)) ≤ ε

/-- The run that always accepts, holds `v u` at proof block `u` and hashes nothing. -/
def constRun {n : ℕ} (v : Fin n → Vals) : Verity.InProofs.Run Unit n fun _ => Vals :=
  ⟨fun _ => True, fun u _ => v u, fun _ => []⟩

/-- **`SessionGives`' hypotheses hold, with `verdict`'s acceptance, at an honest session**: a key `Lms.keygen` derives at
S1's deployed parameters, trusted alone, signs a session of one round that ended complete on time, its coin opened from a
coin tree of one leaf. The run that always accepts holds that session's values at each of its three proof blocks and hashes
nothing: no proof block fails `PieceOk` (`Gives` at `K₀ = 1`, `ε = 0`), `Carried` holds, the clear checks pass, and
`verdict` accepts the session the run opens. -/
def SessionGivesHonest : Prop :=
  ∃ (seed : ByteArray) (k : Lms.PrivateKey) (expected : Nat → Option ByteArray) (c : SessionCommitment)
    (openings : Array CoinOpening) (idx : Nat → Nat) (v : Fin (c.k + 2) → Vals),
    Lms.keygen Lms.deployed seed = .ok k ∧ c.k = 1 ∧
      Gives (PMF.pure ()) (constRun v) (PieceOk SCHEMES LMS k.publicKey c expected idx) 1 0 ∧
      Carried c (constRun v) ∧ ClearOk SCHEMES [k.publicKey] LMS k.publicKey c openings idx ∧
      (verdict SCHEMES [k.publicKey] expected
        (sessionOf c LMS k.publicKey ((constRun v).ports ⟨1, by omega⟩ ()) ((constRun v).ports ⟨0, by omega⟩ ())) c
        openings).accepted = true

/-- **`SessionGives`' conclusion fails without its hypotheses**: under S1's scheme (`SCHEMES`, `LMS`) and a trusted key,
a session of no rounds that did not end complete passes the clear checks and `verdict` rejects it, so a run that always
accepts it and hashes nothing gives the event probability `1`, and the bound fails at `ε = 0`. -/
def SessionGivesNontrivial : Prop :=
  ∃ (c : SessionCommitment) (pk : ByteArray) (v : Fin (c.k + 2) → Vals),
    ¬ prPMF (PMF.pure ()) (fun ω => (constRun v).accepts ω ∧ ClearOk SCHEMES [pk] LMS pk c #[] (fun _ => 0) ∧
        (verdict SCHEMES [pk] (fun _ => none)
          (sessionOf c LMS pk ((constRun v).ports ⟨1, by omega⟩ ω) ((constRun v).ports ⟨0, by omega⟩ ω)) c
          #[]).accepted = false ∧
        ¬ Hash.CollisionIn Verity.sha512 ((constRun v).hashInputs ω)) ≤ 0

end MemoryChallenger.Properties.InProofs
