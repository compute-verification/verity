import Flock

/-!
What the inner verifier's algebra reads from a statement, per rep of a recorded session, from the Lean verifier of record:
the extra claims (`Stmt.extraClaims` at the session's link points), the lincheck's circuit fold `comb`
(`CircuitFold.folded` at the rep's coins), and the pipeline's values the Python reference is held to (`ZcOut`, `LcOut`,
the batched target `T` and the batching coins `γ`). It runs each rep's zerocheck, lincheck, ring switch and Ligerito as
`verifyRep` does, Merkle openings included, and writes nothing unless every check passes. For
`tests/fixtures/inner_claims.json` (`verity_flock.rec_algebra`'s tests); regenerate with `tests/lean/generate_inner_claims.sh`.

    lean --run InnerClaims.lean [--partition FILE] STATEMENT CIRCUIT PUBLIC SESSION_DIR [os]

`os`: a session on the verifier's live OS coins (`verify --coins os`, recursion's inner session), else seeded coins.
`--partition`: the verifier's copy of the `verity/partition/v1` object a circuit binds (`verify --partition`).
-/

open Lean Flock Transcript

def hx (a : F128) : Json := .str a.toHex

def hxs (a : Array F128) : Json := .arr (a.map hx)

/-- `lincheck`, returning the circuit fold it checks `z_partial` against beside its output. -/
def lincheckComb (c : CircuitFold) (zc : ZcOut) (p : LincheckProof) : TM (LcOut × Array F128) := do
  let kLog := c.kLog
  let rhoIn := zc.rho.extract 0 (kLog - 6)
  label "flock-lincheck-v0"
  let alpha ← sample
  let beta ← sample
  let mut claim := alpha * zc.vA + zc.vB + beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (q1, qinf) := p.rounds[j]!
    absorbF q1
    absorbF qinf
    let t ← sample
    ts := ts.push t
    let e0 := claim + q1
    let c1 := e0 + q1 + qinf
    claim := qinf * t * t + c1 * t + e0
  let comb ← match c.folded alpha beta (skipWeights zc.z) rhoIn ts with
    | .ok v => pure v
    | .error err => throw err
  check (comb.size == 64) "lincheck: the circuit fold has the wrong width"
  absorbSlice p.zPartial
  check (dot comb p.zPartial == claim) "lincheck: ConsistencyFailed (sumcheck-final)"
  let rPrime ← sample
  let w := dot (skipWeights rPrime) p.zPartial
  return ({ w, rPrime, rIn := ts.reverse }, comb)

def claimJson (c : Claim) : Json := Json.mkObj [("value", hx c.value), ("skip", hxs c.skip), ("x", hxs c.x)]

def dumpRep (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session) (r : Nat) (bytes : ByteArray) : V Json := do
  let t := spec.tables[0]!
  let name := s!"{t}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"no stream {name}"
  let extras ← st.extra sess.points
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes
  let shape : Shape := { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let p ← decodeProof shape bytes
  let prog : TM Json := do
    let zc ← bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let (lc, comb) ← lincheckComb st.fold zc p.lincheck
    let (T, sws) ← ringSwitch st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    ligerito st.merkle sch T p.cap p.ligerito (basis sws)
    return Json.mkObj [
      ("stream", .str name),
      ("zerocheck", Json.mkObj [("z", hx zc.z), ("rho", hxs zc.rho), ("r", hxs zc.rRest), ("v_a", hx zc.vA),
        ("v_b", hx zc.vB), ("v_c", hx zc.vC)]),
      ("lincheck", Json.mkObj [("w", hx lc.w), ("r_prime", hx lc.rPrime), ("r_in", hxs lc.rIn)]),
      ("comb", hxs comb),
      ("extras", .arr (extras.map claimJson)),
      ("T", hx T),
      ("gamma", hxs (sws.map (·.gamma)))]
  let (out, tr) ← prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8)
  if tr.k != stream.rounds.size then throw s!"{name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
  return out

def main (args : List String) : IO UInt32 := do
  let (args, partFile) := match args with
    | "--partition" :: f :: rest => (rest, some f)
    | _ => (args, none)
  let (args, coins) := match args with
    | [sid, circuit, pub, dir, "os"] => ([sid, circuit, pub, dir], Coins.os)
    | _ => (args, Coins.seed)
  let [sid, circuit, pub, dir] := args
    | IO.eprintln "usage: lean --run InnerClaims.lean [--partition FILE] STATEMENT CIRCUIT PUBLIC SESSION_DIR [os]"; return 2
  let some tags := Tags.find? sid | IO.eprintln s!"unknown statement {sid}"; return 2
  let tags := tags.withCoins coins
  let circuitB ← IO.FS.readBinFile circuit
  let pubB ← IO.FS.readBinFile pub
  let record ← IO.FS.readFile (dir ++ "/session.json")
  let partition ← match partFile with
    | none => pure none
    | some f => match Json.parse (← IO.FS.readFile f) with
      | .ok j => pure (some j)
      | .error e => IO.eprintln s!"--partition {f}: not JSON: {e}"; return 2
  let (spec, sts) ← match Stmt.setupTables tags circuitB pubB coins 1 (partition := partition) with
    | .ok x => pure x
    | .error e => IO.eprintln s!"setup: {e}"; return 1
  let proofs ← spec.repStreams.mapM fun (_, _, w) => IO.FS.readBinFile s!"{dir}/{w.replace "/" "."}.bin"
  let res : V Json := do
    let st := Setup.ofCircuit sts[0]!
    let some sch := fast100 st.m | throw s!"m = {st.m}: no fast100 schedule"
    let sess ← Record.decode spec record
    let mut reps := #[]
    for r in [0:2] do
      reps := reps.push (← dumpRep spec st sch sess r proofs[r]!)
    return Json.mkObj [("m", .num st.m), ("k_log", .num st.kLog), ("reps", .arr reps)]
  match res with
  | .ok j => IO.println j.compress; return 0
  | .error e => IO.eprintln e; return 1
