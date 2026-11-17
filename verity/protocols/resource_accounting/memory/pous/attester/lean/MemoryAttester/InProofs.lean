module

public import MemoryAttester.Verdict

/-!
# The session checked inside proofs

`pous.verifier.session_proofs`: the session's verdict from the outcomes of `k + 2` proofs, linked only by their committed
rows, so that the memory challenger's signed record never reaches the verifier in the clear. Unit 0 is the signature's
(`PousSessionSig_v1{N, W, H}`: key, m, sig; ok), unit 1 the signed digest's (`PousSessionDigest_v1{k, HEAD, LIMIT}`:
head, ctx, the `T = ⌈k/2⌉` records rows; ok, m, the records rows) and unit `j + 2` round `j`'s
(`PousRound_v1{blockBytes, j % 2}`: row, records, j, i; ok). A unit's rows are its parameters' committed rows, then its
outputs'. The verifier checks in the clear the scheme, the key and every round's coin opening (`clearReasons`), each
proof accepted and of its program, each public input against the value it computes (`knownReasons`), and the links,
which the window's wires make: unit 0's `m` is unit 1's output m (row `3 + T`), and round `j`'s records row is unit 1's
records output `j / 2` (row `4 + T + j / 2`) (`linkReasons`). The reasons are the Python's, word for word.
-/

@[expose] public section

namespace MemoryAttester.InProofs

open Flock

/-- One unit's proof as its verifier left it: its program, whether it was accepted, each parameter's committed row, and
the parameters public in the proof with their values, in parameter order. -/
structure UnitOutcome where
  program : String
  accepted : Bool
  rows : Array ByteArray
  known : Array (Nat × ByteArray)
  deriving Inhabited

/-- Unit 0's, unit 1's and each round's outcome. -/
structure RunProofs where
  signature : UnitOutcome
  digest : UnitOutcome
  rounds : Array UnitOutcome
  deriving Inhabited

/-- The signed header of a session that ended complete after its `k` rounds. -/
def headOf (c : ChallengePhaseCommitment) : ByteArray := (header c COMPLETE c.k).toUTF8

/-- The header row: the header, then a zero byte when its length is odd (a row is whole u16 words). -/
def headRow (c : ChallengePhaseCommitment) : ByteArray :=
  let h := headOf c
  if h.size % 2 == 0 then h else h.push 0

/-- What `memory_challenger_message` hashes before the record: the role's tag, `I` and the session id's field. -/
def ctxOf (c : ChallengePhaseCommitment) (publicKey : ByteArray) : ByteArray :=
  field MEMORY_CHALLENGER_TAG.toUTF8 ++ publicKey.extract 0 16 ++ field (field c.session)

/-- `n` bytes of `x`, least significant first: an integer row's bytes. -/
def le (n x : Nat) : ByteArray := ByteArray.mk ((Array.range n).map fun i => (x / 256 ^ i % 256).toUInt8)

def recordsRows (k : Nat) : Nat := (k + 1) / 2

/-- Unit 0's, unit 1's and each round's program. -/
def programs (c : ChallengePhaseCommitment) (p : Lms.Params) : String × String × Array String :=
  (s!"PousSessionSig_v1\{N={p.N},W={p.W},H={p.H}}",
   s!"PousSessionDigest_v1\{K={c.k},HEAD={(headOf c).size},LIMIT={c.limitNs}}",
   (Array.range c.k).map fun j => s!"PousRound_v1\{N={c.blockBytes},SLOT={j % 2}}")

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

/-- `link_reasons`: the shape, then unit 0's `m` is unit 1's output m and round `j`'s records row is unit 1's records
output `j / 2`. -/
def linkReasons (c : ChallengePhaseCommitment) (proofs : RunProofs) : Array String := Id.run do
  let d := proofs.digest.rows
  let t := recordsRows c.k
  if proofs.rounds.size != c.k || d.size != 4 + 2 * t || proofs.signature.rows.size != 4 then
    return #[s!"the session's proofs are not one signature's, one signed digest's over {recordsRows c.k} records rows " ++
      s!"and {c.k} rounds'"]
  let mut reasons : Array String := #[]
  if proofs.signature.rows[1]! != d[3 + t]! then
    reasons := reasons.push "the signed message is not the signed digest's"
  for j in [0:proofs.rounds.size] do
    let out := proofs.rounds[j]!
    if out.rows.size != 5 || out.rows[1]! != d[4 + t + j / 2]! then
      reasons := reasons.push s!"round {j}'s record is not the signed digest's"
  return reasons

def linksOk (c : ChallengePhaseCommitment) (proofs : RunProofs) : Bool := (linkReasons c proofs).isEmpty

/-- `known_reasons`: each public input is the value the verifier computes, and each round's row the registered row at
its opened index (`registered i`, none past C's blocks). -/
def knownReasons (c : ChallengePhaseCommitment) (openings : Array CoinOpening) (publicKey : ByteArray) (proofs : RunProofs)
    (registered : Nat → Option ByteArray) : Array String := Id.run do
  let mut reasons : Array String := #[]
  let mut want : Array (UnitOutcome × String × Array (Nat × ByteArray)) :=
    #[(proofs.signature, "the signature's", #[(0, publicKey)]),
      (proofs.digest, "the signed digest's",
        #[(0, headRow c), (1, ctxOf c publicKey)])]
  for j in [0:proofs.rounds.size] do
    let out := proofs.rounds[j]!
    match byJ openings j with
    | none => pure ()
    | some o =>
      want := want.push (out, s!"round {j}'s", #[(2, le 4 j), (3, le 8 o.index)])
      let row := if o.index < c.blocks then registered o.index else none
      let same := match row with
        | some r => out.rows.size ≥ 1 && out.rows[0]! == r
        | none => false
      if !same then reasons := reasons.push s!"round {j}'s row is not the registered row at its index"
  for (out, name, values) in want do
    if out.known != values then reasons := reasons.push s!"{name} public inputs are not the session's"
  return reasons

/-- `check_session`: the session's verdict from its proofs. It fails closed; which rounds were late or wrong stays inside
the proofs. -/
def runInProofs (schemes : Schemes) (trusted : List ByteArray) (registered : Nat → Option ByteArray)
    (c : ChallengePhaseCommitment) (openings : Array CoinOpening) (scheme : String) (publicKey : ByteArray)
    (proofs : RunProofs) : Verdict := Id.run do
  let mut reasons := clearReasons schemes trusted c openings scheme publicKey
  match schemes.lookup scheme with
  | none => pure ()
  | some p =>
    let (sig, dig, rounds) := programs c p
    let mut units : Array (UnitOutcome × String × String) :=
      #[(proofs.signature, sig, "the signature's"), (proofs.digest, dig, "the signed digest's")]
    for j in [0:min proofs.rounds.size c.k] do
      units := units.push (proofs.rounds[j]!, rounds[j]!, s!"round {j}'s")
    for (out, program, name) in units do
      if out.program != program || !out.accepted then
        reasons := reasons.push s!"{name} proof is not an accepted proof of {program}"
  let links := linkReasons c proofs
  reasons := reasons ++ links
  if links.isEmpty then reasons := reasons ++ knownReasons c openings publicKey proofs registered
  return (Findings.mk reasons 0 0, acceptIf reasons.isEmpty)

end MemoryAttester.InProofs
