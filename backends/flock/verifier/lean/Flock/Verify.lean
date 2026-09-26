/-
PROTOCOL.md §3: the verifier as one total function of its inputs, `Accept` or `Reject reason`.
`verifyRep` is §9–§14 on one rep's replay transcript; `verify` adds the session checks S1–S18.
-/
import Flock.Statement
import Flock.Ligerito

namespace Flock

open Transcript

/-- A statement as the pipeline sees it (§16's `setup` output). -/
structure Setup where
  spec : SessionSpec
  m : Nat
  kLog : Nat
  digest : ByteArray
  fold : CircuitFold
  /-- the statement's extra claims at the link points -/
  extra : Array (Array F128) → V (Array Claim)
  /-- S8: the committed public words -/
  publics : ByteArray → V Unit
  deriving Inhabited

def Setup.ofCircuit (st : Stmt) : Setup :=
  { spec := st.spec, m := st.m, kLog := st.c.kLog, digest := st.digest, fold := st.circuitFold,
    extra := st.extraClaims,
    publics := fun p => if p.size == 0 then pure () else throw "S8: the circuit statement commits no public words" }

def pcsParams (sch : Schedule) : PcsParams := ⟨sch.m.toUInt64, 1, sch.k0.toUInt64, 1, none, 0⟩

/-- §9–§14 for rep `r`; returns the proof's flattened cap. -/
def verifyRep (st : Setup) (sch : Schedule) (sess : Session) (r : Nat) (bytes : ByteArray) : V ByteArray := do
  let name := s!"{st.spec.table}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  let p ← decodeProof { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch } bytes
  if p.params != pcsParams sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
  let prog : TM Unit := do
    let zc ← bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let lc ← lincheck st.fold zc p.lincheck
    let (T, sws) ← ringSwitch st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    ligerito sch T p.cap p.ligerito (basis sws)
  let ((), t) ← prog.run (Transcript.init stream.rounds (st.spec.domain r).toUTF8)
  if t.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - t.k} recorded rounds were never consumed"
  let cap := p.cap.foldl (· ++ ·) ByteArray.empty
  if cap != sess.rootB then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
  return cap

/-- The verdict: `.ok ()` is Accept. `proofs[r]` is the file the server received for rep `r`. -/
def verify (st : Setup) (record : String) (proofs : Array ByteArray) : V Unit := do
  let sess ← Record.decode st.spec record
  st.publics sess.publics
  let some sch := fast100 st.m | throw s!"m = {st.m}: no fast100 schedule"
  if proofs.size != 2 then throw "two proofs expected"
  let want := [s!"{st.spec.table}/rep0", s!"{st.spec.table}/rep1"]
  if sess.proofSha.size != 2 || !want.all (fun w => sess.proofSha.any (·.1 == w)) then
    throw "S17: proof_sha256 does not list exactly the two rep streams"
  for r in [0:2] do
    let sha := (sess.proofSha.find? (·.1 == want[r]!)).map (·.2)
    if sha != some (hexOf (Sha256.hash proofs[r]!)) then throw s!"S17: {want[r]!}: the proof is not the one the server received"
  let c0 ← verifyRep st sch sess 0 proofs[0]!
  let c1 ← verifyRep st sch sess 1 proofs[1]!
  if c0 != c1 then throw "S18/R1: the reps' caps differ"

end Flock
