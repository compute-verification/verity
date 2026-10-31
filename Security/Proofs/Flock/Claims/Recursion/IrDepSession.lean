import Proofs.Flock.Claims.Recursion.IrDep
import Proofs.Flock.Claims.Recursion.HonestHalfFixture

/-!
# `IrDep` at a deployed inner session: the difftest's Lean side

`IrDep x`'s rows at every inner round and its rest after the last, evaluated at a deployed run's inner session, for
`test_irdep_difftest.py` to hold to what the deployed builder (`rec_live`, `rec_vstar`, `rec_vstage`) produced there.
A probe run by hand, not a proof (`exempt` in `lean-audit.json`); what it evaluates is `IrDep`'s by the theorems of
`IrDep.lean`:

* **The inner statement** `xI`, as `85-rec-reprice.sh`'s inner `flock-verify verify` reads it (`Raw`:
  `verity/flock-circuit+plain-leaves`, `--coins os`, `--stratified K`, the partition and program, the record's
  `unit_draw`), and its setup `v`: `setupB false xI`, which is `setupOf false xI` (`setupB_eq`).
* **The tape**: each rep's C-Flock proof file decoded at its shape as `verifyRep` decodes it, with its batched value `T`
  from the verifier's own transcript run on the record's stream (`bindAndZerocheck`, `lincheck`, `ringSwitch`); the
  running sums' salts (`v/acc-salts.bin`).
* **The coins**: the link coin `c₁`, of which `IrDep` reads only the link points (`pointsOf`, held to the record's), then
  each rep stream's rounds' coins, rep by rep (held to `canonRounds`).
* **Rows**: round `i`'s at the first `i` coins, `IrDep.rows`' own at `v` (`irDep_rows`); with rep 0's proof's first cap
  node's first byte flipped too (`planted_rows`), the control.
* **Rest** at every coin: `depRestV v`, which is `IrDep.rest` (`irDep_rest`).

Prints one `IRDEP ` line of JSON on stdout and its progress on stderr:

    cd Security/Proofs && lake env lean --run Flock/Claims/Recursion/IrDepSession.lean TREE K

or on a node's warm slot tree, `VERITY_IRDEP_ARGS='TREE K' python3 tools/verity/lean/lean_changed.py --try FILE`, whose
`lake env lean FILE` runs the `#eval` below. `TREE` is the run's tree (`stage/`: `circuit.txt`, `pub-K.bin`,
`partition.json`, `program.json`; `inner/s0/session/`: `session.json` and `circuit.rep<r>.bin`; `v/acc-salts.bin`).
It imports `HonestHalfFixture`, the fixture whose `IrDep x` this run witnesses, so that one `--try` elaborates both.
-/

namespace IrDepSession

open Flock
open Lean (Json)
open RecDeploy.VStarBuild
open FlockDeploy (Coin tabsOf pointsOf)

def die {α : Type} (msg : String) : IO α := throw (IO.userError msg)

def ofE {α : Type} (what : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => die s!"{what}: {e}"

def say (msg : String) : IO Unit := do
  IO.eprintln s!"irdep: {msg}"
  (← IO.getStderr).flush

def jf (x : F128) : Json := Json.str x.toHex

def jb (b : ByteArray) : Json := Json.str (hexOf b)

def jarr {α : Type} (g : α → Json) (a : Array α) : Json := Json.arr (a.map g)

/-- The inner statement's bytes as the inner `verify` reads them, at `K` units, and the session's record. -/
def readTree (dir : String) (k : ℕ) : IO (FlockDeploy.Raw × String) := do
  let rd := fun (p : String) => IO.FS.readBinFile s!"{dir}/{p}"
  let record ← IO.FS.readFile s!"{dir}/inner/s0/session/session.json"
  let circuit ← rd "stage/circuit.txt"
  let publicFile ← rd s!"stage/pub-{k}.bin"
  let partition ← rd "stage/partition.json"
  let program ← rd "stage/program.json"
  pure ({ statement := "verity/flock-circuit+plain-leaves", circuit, publicFile, tables := [], zk := false,
          osCoins := true, count := 1, draw := (Flock.RecCoins.recordDraw record).map (·.compress.toUTF8),
          partition := some partition, program := some program, work := none, workTable := none,
          stratified := some k, closure := none, registered := none, publicInputs := none }, record)

/-- Rep `r`'s proof, decoded as `verifyRep` decodes it, and its batched value `T` from the verifier's transcript run on
the record's stream (the decoded record's `streams` and `points`). -/
def repProof (spec : SessionSpec) (st : Setup) (sch : Schedule) (streams : Array StreamRec)
    (points : Array (Array F128)) (r : ℕ) (bytes : ByteArray) : IO (Proof × F128) := do
  let t := spec.tables[0]!
  let name := s!"{t}/rep{r}"
  let some stream := streams.find? (·.name == name) | die s!"no stream {name}"
  let extras ← ofE "extras" (st.extra points)
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes
  let shape : Flock.Shape :=
    { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen, saltLen }
  let p ← ofE name (decodeProof shape bytes)
  let prog : Transcript.TM F128 := do
    let zc ← bindAndZerocheck st.m st.digest (p.cap.foldl (· ++ ·) ByteArray.empty) p.zerocheck
    let lc ← Flock.lincheck st.fold zc p.lincheck
    let (T, _) ← Flock.ringSwitch st.m (abcClaims st.m st.kLog zc lc ++ extras) p.ringSwitches
    pure T
  let (T, _) ← ofE s!"{name}: transcript" (prog.run (Transcript.init stream.rounds (spec.domain t r).toUTF8))
  pure (p, T)

def rowsJson (rounds : List (List (Array ℕ))) : Json :=
  Json.arr (rounds.toArray.map fun rs => Json.arr (rs.toArray.map fun w => Json.arr (w.map fun n => Lean.toJson n)))

def runTree (dir : String) (k : ℕ) : IO UInt32 := do
  let (raw, record) ← readTree dir k
  say s!"read the tree: circuit {raw.circuit.size} bytes"
  let some v := setupB false raw.encode | die "the inner statement does not set up (setupOf false)"
  let (spec, sts) := (v.2.1, v.2.2)
  say s!"set up: {sts.size} table(s) {spec.tables}"
  if sts.size != 1 then die s!"{sts.size} tables: the dense inner statement has one"
  let sess ← ofE "record" (Flock.Record.decode spec record)
  let st := (tabsOf sts)[0]!
  let some sch := fast100 st.m | die s!"m = {st.m}: no fast100 schedule"
  let mut ps : Array (Proof × F128) := #[]
  for r in [0:2] do
    ps := ps.push (← repProof spec st sch sess.streams sess.points r
      (← IO.FS.readBinFile s!"{dir}/inner/s0/session/{spec.tables[0]!}.rep{r}.bin"))
  say s!"decoded both reps' proofs, T {ps.map (·.2.toHex)}"
  let c₁ : Coin := fun i => (sess.points.getD (i.val / spec.mPts) #[]).getD (i.val % spec.mPts) 0
  if pointsOf spec c₁ != sess.points then die "the link coin's points are not the record's"
  let stream := fun (r : ℕ) =>
    ((sess.streams.find? (·.name == s!"{spec.tables[0]!}/rep{r}")).map (·.rounds)).getD #[]
  let repCoins : List Coin := (List.range 2).flatMap fun r =>
    (stream r).toList.map fun rd => (fun i => rd.coins.getD i.val 0 : Coin)
  for r in [0:2] do
    if RecDeploy.Rows.canonRounds spec sts repCoins r != (stream r).map (·.coins) then
      die s!"rep {r}: canonRounds is not the record's coins"
  let cs := c₁ :: repCoins
  let acc ← IO.FS.readBinFile s!"{dir}/v/acc-salts.bin"
  let n := MerkleScheme.hm96Sha512.saltBytes
  let τ : DepTape :=
    { prove := fun _ => ps, accSalts := (Array.range (acc.size / n)).map fun i => acc.extract (i * n) (i * n + n),
      fresh := fun _ _ _ => ByteArray.empty }
  let rowsAt := fun (qs : Array (Proof × F128)) (i : ℕ) =>
    ((depRows spec sts qs (cs.take i)).getD i []).map _root_.VStar.In.wordsOf
  let rounds := (List.range (cs.length + 1)).map (rowsAt ps)
  say s!"rows of {rounds.length} rounds, {(rounds.map List.length).sum} rows"
  let (p0, T0) := ps[0]!
  let node := p0.cap[0]!
  let planted := ps.set! 0 ({ p0 with cap := p0.cap.set! 0 (node.set! 0 (node.get! 0 ^^^ 1)) }, T0)
  let plantedRounds := (List.range (cs.length + 1)).map (rowsAt planted)
  let src := depRestV v τ cs
  say s!"rest: {src.openings.size} openings, {src.accSalts.size} running-sum salts"
  let out := Json.mkObj [
    ("m", Lean.toJson st.m), ("k_log", Lean.toJson st.kLog), ("levels", Lean.toJson sch.levels.size),
    ("rep_lens", Json.arr ((FlockDeploy.repLens spec sts).toArray.map fun n => Lean.toJson n)),
    ("T", jarr (fun p => jf p.2) ps),
    ("rows", rowsJson rounds), ("planted_rows", rowsJson plantedRounds),
    ("openings", jarr (fun o => Json.mkObj [("rep", Lean.toJson o.rep), ("level", Lean.toJson o.level), ("j", Lean.toJson o.j),
      ("pos", Lean.toJson o.pos), ("row", jarr jf o.row), ("sibs", jarr jb o.sibs)]) src.openings),
    ("acc_salts", jarr jb src.accSalts),
    ("kappa", jarr (jarr (jarr (jarr jf))) src.kappa),
    ("eq_a", jarr (jarr (jarr jf)) src.eqA),
    ("vs", jarr (jarr jf) src.vs)]
  IO.println s!"IRDEP {out.compress}"
  (← IO.getStdout).flush
  pure 0

def main (args : List String) : IO UInt32 := do
  match args with
  | [dir, k] =>
    match k.toNat? with
    | some k => runTree dir k
    | none => IO.eprintln s!"K: not a number: {k}" *> pure 2
  | _ => IO.eprintln "usage: IrDepSession.lean TREE K" *> pure 2

end IrDepSession

#eval show IO Unit from do
  if let some a := (← IO.getEnv "VERITY_IRDEP_ARGS") then
    let rc ← IrDepSession.main ((a.splitOn " ").filter (· ≠ ""))
    if rc != 0 then throw (IO.userError s!"IrDepSession: exit {rc}")
