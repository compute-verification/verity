/-
PROTOCOL.md §3: the verifier as one total function of its inputs, `Accept` or `Reject reason`.
`verifyRep` is §9–§14 on one rep's replay transcript; `verify` adds the session checks S1–S18.
-/
import Flock.Statement
import Flock.Ligerito

namespace Flock

open Transcript

/-- One table's statement as the pipeline sees it (§16's `setup` output). -/
structure Setup where
  m : Nat
  kLog : Nat
  digest : ByteArray
  fold : CircuitFold
  /-- the statement's extra claims at the link points -/
  extra : Array (Array F128) → V (Array Claim)
  /-- S8: the committed public words -/
  publics : ByteArray → V Unit
  /-- the proof's Merkle trees (§14) -/
  merkle : MerkleScheme
  /-- every opening carries `opened_salts`, empty for an unsalted scheme, as the circuit prover writes it from `2908d078` on
  (§16.16); its earlier unsalted openings carry none -/
  openedSalts : Bool := false
  deriving Inhabited

def Setup.ofCircuit (st : Stmt) : Setup :=
  { m := st.m, kLog := st.c.kLog, digest := st.digest.extract 0 32, fold := st.circuitFold,
    extra := st.extraClaims, merkle := st.merkle, openedSalts := st.tags.hm96Rows,
    publics := fun p => if p.size == 0 then pure () else throw "S8: the circuit statement commits no public words" }

def pcsParams (ms : MerkleScheme) (sch : Schedule) : PcsParams := ⟨sch.m.toUInt64, 1, sch.k0.toUInt64, 1, none, ms.pcsIndex⟩

/-- §9–§14 for table `i`'s rep `r`; returns the proof's flattened cap. -/
def verifyRep (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session) (i r : Nat) (bytes : ByteArray) : V ByteArray := do
  let t := spec.tables[i]!
  let name := s!"{t}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes
  let shape : Shape := { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let p ← decodeProof shape bytes
  if p.params != pcsParams st.merkle sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
  let prog : TM Unit := do
    let zc ← bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let lc ← lincheck st.fold zc p.lincheck
    let (T, sws) ← ringSwitch st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    ligerito st.merkle sch T p.cap p.ligerito (basis sws)
  let ((), tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- The verdict on the files the server received, each rep's proof decoded from `dec` of its file: S17 binds the
record's `proof_sha` to the file itself, never to `dec` of it. `.ok ()` is Accept. `tabs[i]` is table `i`'s statement,
and `proofs[2i + r]` the file the server received for its rep `r`. -/
def verifyFiles (dec : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : V Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decode spec record
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (spec.dg b)) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRep spec st sch sess i 0 (dec proofs[2 * i]!)
    let c1 ← verifyRep spec st sch sess i 1 (dec proofs[2 * i + 1]!)
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

/-- The verdict: `.ok ()` is Accept. `tabs[i]` is table `i`'s statement, and `proofs[2i + r]` the file the server received
for its rep `r`, decoded as it is (the deployed form, `plainBytes false`). -/
def verify (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) : V Unit :=
  verifyFiles (fun b => b) spec tabs record proofs

/-- Under the deployed form the files are decoded as they are: `verifyFiles (plainBytes false)` is `verify`. -/
theorem verifyFiles_plainBytes_false (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : verifyFiles (plainBytes false) spec tabs record proofs = verify spec tabs record proofs :=
  rfl

end Flock
