import MemoryChallenger.Records
import MemoryChallenger.SignedMessage

/-!
# The verdict on a signed session

`memory_challenger.verdict`: the check of a signed session against its commitment, released with it. It checks the
signature under a trusted key and an accepted scheme, the header against the commitment, every revealed index against
its coin opening, every `H(block)` against the expected digest of that block of `C`, and every round on time within
`Δ + RTT`. It fails closed: an unknown scheme, an untrusted key or a bad signature rejects, whatever the rounds. The
reasons are the Python's, word for word.

A scheme is an LMS parameter set under its id (`SCHEMES`: S1's, `LMS` at `Lms.deployed`, and nothing else). The test
signers' schemes (`TEST_SCHEMES`, `pous_test_signer.SCHEMES`) are accepted only where a caller passes them.
-/

namespace MemoryChallenger

open Flock

/-- A closed session: its commitment, how it ended, its round records, and the one signature over them. -/
structure SignedSession where
  commitment : SessionCommitment
  ended : String
  rounds : Array Round
  scheme : String
  publicKey : ByteArray
  signature : ByteArray
  deriving BEq, Inhabited

def SignedSession.message (s : SignedSession) : E ByteArray := MemoryChallenger.message s.commitment s.ended s.rounds

/-- `lms_verify(params)(public_key, session, record, signature)`: false for a key of the wrong size; a signature of the
wrong size is refused (Python's `ValueError`, which `verdict` reads as no signature). -/
def lmsVerify (p : Lms.Params) (publicKey session record signature : ByteArray) : E Bool := do
  if publicKey.size ≠ 16 + p.N then return false
  let sig ← Lms.Signature.decode p signature
  Lms.verify p publicKey (← memoryChallengerMessage (publicKey.extract 0 16) session record) sig

/-- `LMS`: S1's scheme at its deployed parameters. -/
def LMS : String := "lms-sha512/n64-w1-h20/v0"

abbrev Schemes := List (String × Lms.Params)

/-- `SCHEMES`: the signature schemes a session is accepted under, S1's and nothing else. -/
def SCHEMES : Schemes := [(LMS, Lms.deployed)]

/-- `pous_test_signer.scheme(height)`. -/
def testScheme (height : Nat) : String := TEST_PREFIX ++ s!"lms-sha512/n64-w1-h{height}/v0"

/-- `pous_test_signer.SCHEMES`: the test signers' schemes, heights 1 to 10, which `SCHEMES` never holds. -/
def TEST_SCHEMES : Schemes := (List.range 10).map fun i => (testScheme (i + 1), ⟨64, 1, i + 1⟩)

/-- Every scheme this port verifies, by id: a caller's mapping of scheme ids may name these and no others. -/
def KNOWN_SCHEMES : Schemes := SCHEMES ++ TEST_SCHEMES

/-- Python's `repr` of a string, for the scheme in a reason: single quotes unless the string holds `'` and no `"`; `\`,
the quote, `\t \n \r` escaped, other ASCII controls and DEL as `\xhh`. Characters past ASCII are written as they are,
which is Python's `repr` for the printable ones (all a difftest draws). -/
def pyRepr (s : String) : String := Id.run do
  let q := if s.contains '\'' && !s.contains '"' then '"' else '\''
  let hex2 := fun (n : Nat) =>
    let d := fun (x : Nat) => if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
    String.ofList [d (n / 16), d (n % 16)]
  let mut out := String.singleton q
  for c in s.toList do
    let n := c.toNat
    out := if c == q || c == '\\' then (out.push '\\').push c
      else if c == '\t' then out ++ "\\t" else if c == '\n' then out ++ "\\n" else if c == '\r' then out ++ "\\r"
      else if n < 0x20 || n == 0x7f then out ++ "\\x" ++ hex2 n
      else out.push c
  return out.push q

/-- Whether the signed session passes. `reasons` (empty when it passes), `late` and `wrong` are witness: only
`public` is published. -/
structure Verdict where
  accepted : Bool
  reasons : Array String
  late : Nat
  wrong : Nat
  deriving BEq, Inhabited

/-- `verdict(signed, commitment, openings, expected, trusted=trusted, schemes=schemes)`: `expected i` is `block_digest` of
block `i` of `C`, or none when the caller has no digest for it (no answer is then that block). Openings are looked up by
`j`, a later one for the same `j` winning, as the Python's dict does. -/
def verdict (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening) : Verdict := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup s.scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr s.scheme} is not accepted"
  | some p =>
    if !trusted.contains s.publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
    else
      let ok := match s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push "the signature does not verify"
  if !(s.commitment == c) then reasons := reasons.push "the signed header is not the session's commitment"
  if s.ended != COMPLETE || s.rounds.size != c.k then
    reasons := reasons.push s!"the session ended {s.ended} after {s.rounds.size} of {c.k} rounds"
  let byJ := fun (j : Nat) => openings.findRev? (·.j == j)
  let mut late := 0
  let mut wrong := 0
  for h : j in [0:s.rounds.size] do
    let r := s.rounds[j]
    let committed := match byJ j with
      | some o => r.j == j && o.index == r.index && verifyCoin c o
      | none => false
    if !committed then reasons := reasons.push s!"round {j}'s index is not its committed coin"
    if !(r.index < c.blocks && expected r.index == some r.digest) then wrong := wrong + 1
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ c.limitNs) then late := late + 1
  if wrong > 0 then reasons := reasons.push s!"{wrong} answers are not their blocks of C"
  if late > 0 then reasons := reasons.push s!"{late} answers are past Δ + RTT = {c.limitNs} ns"
  return ⟨reasons.isEmpty, reasons, late, wrong⟩

end MemoryChallenger
