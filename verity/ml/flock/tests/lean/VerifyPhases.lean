import FlockVerifyLib

/-!
Where `flock-verify verify --zk` spends a session's time. The statement is set up as `verifyCmd` sets it up (the same input
readers and `buildSession`), then the session is verified by `Zk.verify`'s steps one at a time on one transcript, each step
timed: reading the session's files, decoding its record, the proofs' digests, and per rep the extra claims, the proof's
decoding, the masked zerocheck and lincheck rounds, the lincheck's circuit fold, the ring switch, Ligerito, level 0's lane
phase, the inner proof and the caps. The verdict must be `verify --zk`'s on the same session.

    verify-phases --statement ID --circuit F --public F [--partition F] [--program F] [--registered F] --session DIR...

`FlockVerifyLib` is `FlockVerify.lean` with its `main` renamed, built beside this file (`85-rec-reprice.sh STEP=lean PHASES=1`).
One `PHASES` line per session: `total_s` from reading its files to its verdict, as `verify`'s per-session time less setup, and
`phases` (name, seconds; both reps summed), whose sum is `total_s` less the timer's own overhead.
-/

namespace Flock.Zk

open Transcript
open Lean (Json)

/-- Each phase's nanoseconds, in the order first seen, and the first refusal. -/
structure Clock where
  phases : Array (String × Nat) := #[]
  why : Option String := none

/-- `f ()`, timed under `name`. The result is matched in IO (a refusal is recorded) before the clock is read again, so the
computation cannot move past the second reading. -/
def timed {α : Type} (clk : IO.Ref Clock) (name : String) (f : Unit → V α) : IO (Option α) := do
  let t0 ← IO.monoNanosNow
  let r ← IO.lazyPure f
  let out ← match r with
    | .ok a => pure (some a)
    | .error e => clk.modify (fun c => { c with why := c.why <|> some s!"{name}: {e}" }); pure none
  let dt := (← IO.monoNanosNow) - t0
  clk.modify fun c => match c.phases.findIdx? (·.1 == name) with
    | some i => { c with phases := c.phases.modify i fun (n, t) => (n, t + dt) }
    | none => { c with phases := c.phases.push (name, dt) }
  return out

/-- A transcript step, timed under `name`: its value and the transcript after it. -/
def step {α : Type} (clk : IO.Ref Clock) (name : String) (tr : Transcript) (m : TM α) : IO (Option (α × Transcript)) :=
  timed clk name fun _ => m.run tr

/-- `replay`'s state where the lincheck's circuit fold is computed. -/
structure Mid where
  s : Nat
  m : Nat
  kLog : Nat
  z : F128
  rho : Array F128
  r : Array F128
  ts : Array F128
  alpha : F128
  beta : F128
  running : Lin
  fc : Lin
  cons : Array (String × Lin)
  deriving Inhabited

/-- `replay` up to its circuit fold. -/
def replayRounds (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) : TM Mid := do
  let s := lay.len
  let m := st.m
  let zc := p.zerocheck
  label "flock-r1cs-v0"
  absorbBytes st.digest
  absorbBytes (p.cap.foldl (· ++ ·) ByteArray.empty)
  check (m ≥ 13) "zerocheck: m < 13"
  label "flock-zerocheck-v0"
  label LABEL_PADS
  absorbBytes root
  let _rSkip ← sampleVec 6
  let rOuter ← sampleVec (m - 13)
  let r := pinned ++ rOuter
  absorbSlice zc.round1ab
  absorbSlice zc.round1c
  let z ← sample
  let r1c := (Array.range 64).map fun i => Lin.masked zc.round1c[i]! (lay.r1c i) s
  let r1ab := (Array.range 64).map fun i => Lin.masked zc.round1ab[i]! (lay.r1ab i) s
  let computedC := Lin.combo (lagrange nodesL den6Inv z) r1c s
  let fc := Lin.masked zc.finalC lay.fc s
  let mut cons : Array (String × Lin) := #[("zerocheck c_eval", computedC.add fc)]
  let comb := (Array.range 64).map fun i => r1ab[i]!.add r1c[i]!
  let mut running := (Lin.combo (combWeights z) comb s).add computedC
  let mut rho : Array F128 := #[]
  for j in [0:m - 6] do
    let (m1, minf) := zc.rounds[j]!
    let req := r[j]!
    let g1 := Lin.masked m1 (lay.ml j 0) s
    let ginf := Lin.masked minf (lay.ml j 1) s
    let g0 := (running.add (g1.scale req)).scale (1 + req).inv
    absorbF m1
    absorbF minf
    let rr ← sample
    rho := rho.push rr
    running := ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))
  let a := Lin.masked zc.finalA lay.fa s
  let b := Lin.masked zc.finalB lay.fb s
  let prod : Lin := ⟨zc.finalA * zc.finalB, (((Array.replicate s 0).set! lay.fb zc.finalA).set! lay.fa zc.finalB).set! lay.hab 1⟩
  cons := cons.push ("zerocheck final a·b", running.add prod)
  absorbF zc.finalA
  absorbF zc.finalB
  let kLog := st.kLog
  label "flock-lincheck-v0"
  let alpha ← sample
  let beta ← sample
  running := ((a.scale alpha).add b).addConst beta
  let mut ts : Array F128 := #[]
  for j in [0:kLog - 6] do
    let (e1v, einfv) := p.lincheck.rounds[j]!
    absorbF e1v
    absorbF einfv
    let rr ← sample
    let e1 := Lin.masked e1v (lay.lc j 0) s
    let einf := Lin.masked einfv (lay.lc j 1) s
    let e0 := running.add e1
    let c1 := (e0.add e1).add einf
    running := ((einf.scale (rr * rr)).add (c1.scale rr)).add e0
    ts := ts.push rr
  return { s, m, kLog, z, rho, r, ts, alpha, beta, running, fc, cons }

/-- `replay` from its circuit fold `combv` on. -/
def replayTail (lay : PadLayout) (p : Proof) (md : Mid) (combv : Array F128) : TM Replayed := do
  let s := md.s
  check (combv.size == 64) "lincheck: the circuit fold has the wrong width"
  absorbSlice p.lincheck.zPartial
  let zp := (Array.range 64).map fun i => Lin.masked p.lincheck.zPartial[i]! (lay.zp i) s
  let cons := md.cons.push ("lincheck final", md.running.add (Lin.combo combv zp s))
  let rPrime ← sample
  let w := Lin.combo (skipWeights rPrime) zp s
  let zcOut : ZcOut := { z := md.z, rho := md.rho, rRest := md.r, vA := 0, vB := 0, vC := 0 }
  let lcOut : LcOut := { w := 0, rPrime, rIn := md.ts.reverse }
  return { claims := abcClaims md.m md.kLog zcOut lcOut, abValue := w, cValue := md.fc, cons }

/-- `verifyRep`, step by step; the proof's flattened cap. -/
def repPhases (clk : IO.Ref Clock) (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : Nat) (sess : Session) (i r : Nat)
    (bytes : ByteArray) : IO (Option ByteArray) := do
  let t := spec.tables[i]!
  let name := s!"{t}/rep{r}"
  let some stream ← timed clk "record" (fun _ => (sess.streams.find? (·.name == name)).elim
      (Except.error s!"S11: no stream {name}") Except.ok) | return none
  let some extras ← timed clk "extra claims" fun _ => st.extra sess.points | return none
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes
  let shape : Shape := { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let some zp ← timed clk "proof decode" fun _ => do
    let zp ← decode shape bytes
    if zp.proof.params != pcsParams st.merkle sch then throw "R7/F2: the proof's PCS parameters are not the pinned fast100 parameters"
    if zp.inner.root.size != 64 then throw "zk: the pads' root is not 64 bytes"
    pure zp | return none
  let p := zp.proof
  let lay := PadLayout.new st.m st.kLog sch.k0
  let tr := Transcript.init stream.rounds (spec.domain t r).toUTF8
  let some (md, tr) ← step clk "zerocheck and lincheck rounds" tr (replayRounds st lay p zp.inner.root) | return none
  let some combv ← timed clk "circuit fold" fun _ =>
    st.fold.folded md.alpha md.beta (skipWeights md.z) (md.rho.extract 0 (md.kLog - 6)) md.ts | return none
  let some (rp, tr) ← step clk "zerocheck and lincheck rounds" tr (replayTail lay p md combv) | return none
  let some ((ccAb, ccC, T, sws), tr) ← step clk "ring switch" tr (do
    let ab := rp.claims[0]!
    let c := rp.claims[1]!
    let ccAb := claimCheck ab p.ringSwitches[0]!
    let ccC := claimCheck c p.ringSwitches[1]!
    let (T, sws) ← ringSwitch st.m (#[{ ab with value := ccAb }, { c with value := ccC }] ++ extras) p.ringSwitches
    pure (ccAb, ccC, T, sws)) | return none
  let some (rec, tr) ← step clk "ligerito" tr (ligerito st.merkle sch T p.cap p.ligerito (basis sws) ⟨tPad, r, zp.level0⟩)
    | return none
  let some lg ← timed clk "lane phase" fun _ => ligeritoConstraints lay rec p.ligerito zp.level0 | return none
  let some ((), tr) ← step clk "inner proof" tr (verifyInner lay zp.inner p.zerocheck.finalC
      (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lg))
    | return none
  timed clk "caps" fun _ => do
    if tr.k != stream.rounds.size then throw s!"S15/R2: {name}: {stream.rounds.size - tr.k} recorded rounds were never consumed"
    let cap := p.cap.foldl (· ++ ·) ByteArray.empty
    if cap != sess.rootB[i]! then throw s!"S16/R1: {name}: the proof's cap is not the committed root_B"
    pure cap

/-- `Zk.verify` on session `dir`, step by step, its files read under the clock. -/
def sessionPhases (clk : IO.Ref Clock) (spec : SessionSpec) (tabs : Array Setup) (dir : String) : IO Unit := do
  let t0 ← IO.monoNanosNow
  let read ← try pure (Except.ok (← IO.FS.readFile (dir ++ "/session.json"),
      ← spec.repStreams.mapM fun (_, _, w) => IO.FS.readBinFile s!"{dir}/{w.replace "/" "."}.bin"))
    catch e => pure (.error s!"session files: {e}")
  let dt := (← IO.monoNanosNow) - t0
  clk.modify fun c => { c with phases := c.phases.push ("read", dt) }
  let (record, proofs) ← match read with
    | .ok x => pure x
    | .error e => clk.modify (fun c => { c with why := some e }); return
  let some (sess, schs) ← timed clk "record" fun _ => do
    if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
    if spec.coinTree.isNone then throw "zk: the session's coins are not the keyed coin tree"
    let sess ← Record.decode spec record
    for (st, p) in tabs.zip sess.publics do st.publics p
    let schs ← tabs.mapM fun st => schedule st.m
    let want := spec.repStreams
    if proofs.size != want.size then throw s!"{want.size} proofs expected"
    if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
      throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
    if let some k := (Lean.Json.parse record).toOption.bind fun j =>
        (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption then
      CoinTree.checkFresh #[] k
    pure (sess, schs) | return
  let some () ← timed clk "proof digests" fun _ => do
    for ((_, _, w), b) in spec.repStreams.zip proofs do
      if (sess.proofSha.find? (·.1 == w)).map (·.2) != some (hexOf (spec.dg b)) then
        throw s!"S17: {w}: the proof is not the one the server received"
    | return
  for i in [0:tabs.size] do
    let (st, (sch, tPad)) := (tabs[i]!, schs[i]!)
    let some c0 ← repPhases clk spec st sch tPad sess i 0 proofs[2 * i]! | return
    let some c1 ← repPhases clk spec st sch tPad sess i 1 proofs[2 * i + 1]! | return
    let some () ← timed clk "caps" fun _ => if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ" else pure ()
      | return

def secs (ns : Nat) : Json := (Lean.JsonNumber.fromFloat? (ns.toFloat / 1e9)).elim (fun _ => .null) (.num ·)

def main (args : List String) : IO UInt32 := do
  let usage := "usage: verify-phases --statement ID --circuit F --public F [--partition F] [--program F] [--registered F] --session DIR..."
  let (some _, some _) := (argOf args "--circuit", argOf args "--public") | IO.eprintln usage; return 2
  for f in ["--comp", "--archive", "--work", "--work-table", "--stratified", "--closure", "--public-inputs", "--identifier",
      "--session-tables", "--coins"] do
    if args.contains f then IO.eprintln s!"verify-phases: {f} is not taken (a --zk session of one table)"; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  if tags.plainLeaves then IO.eprintln s!"verify-phases: refused: {tags.name}"; return 2
  let dirs := (args.dropWhile (· != "--session")).drop 1
  let coins := Coins.tree (coinSpec (tags.tableNames 1))
  let tags := tags.withCoins coins
  let t0 ← IO.monoNanosNow
  let (circuitB, pubB, compB, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"statement inputs: {e}"; return 2
  let tags := Zk.tags tags (maskLogOf circuitB) 1
  let partition ← match ← partitionInput args with | .ok p => pure p | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with | .ok p => pure p | .error e => IO.eprintln e; return 2
  let registered ← match ← registeredInput args with | .ok r => pure r | .error e => IO.eprintln e; return 2
  let t1 ← IO.monoNanosNow
  let setup := buildSession tags circuitB pubB compB coins tables 1 none partition program none none none registered none none
  let (spec, tabs) ← match (do let (spec, sts) ← setup; sts.forM shapeOk; pure (spec, sts.map Setup.ofCircuit)) with
    | .ok r => pure r
    | .error e => IO.eprintln s!"setup: {e}"; return 2
  let t2 ← IO.monoNanosNow
  let mut all := true
  for dir in dirs do
    let record ← IO.FS.readFile (dir ++ "/session.json")
    if tags.hm96Rows && (recordDraw record).isSome then
      IO.eprintln s!"verify-phases: {dir}: a drawn session (its statement is the draw's)"; return 2
    let clk ← IO.mkRef ({} : Clock)
    let s0 ← IO.monoNanosNow
    sessionPhases clk spec tabs dir
    let total := (← IO.monoNanosNow) - s0
    let c ← clk.get
    all := all && c.why.isNone
    let line := Json.mkObj [("session", .str dir), ("accepted", .bool c.why.isNone), ("why", .str (c.why.getD "")),
      ("inputs_s", secs (t1 - t0)), ("setup_s", secs (t2 - t1)), ("total_s", secs total),
      ("phase_sum_s", secs (c.phases.foldl (· + ·.2) 0)), ("phases", .arr (c.phases.map fun (n, t) => .arr #[.str n, secs t]))]
    IO.println s!"PHASES {line.compress}"
  return (if all then 0 else 1)

end Flock.Zk

def main (args : List String) : IO UInt32 := Flock.Zk.main args
