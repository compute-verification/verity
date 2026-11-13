module

public import Proofs.Properties.Verity.SampledTwo.SessionMsgs
meta import Proofs.Properties.Verity.SampledTwo.SessionMsgs

/-!
# `repRounds` at a recorded two-stage play, and the masked verifier on an uneven split

A probe run by hand, not a proof. On a play's form case (`params.json`, `commitment.json`, `evidence.json`, `blobs/`
with `blobs.txt`, as `sampled_two_small.sh` keeps it under `$OUT/form`), for each template whose session statement the
case gives (`SessionMsgs.stmtAtCase`): the statement's setup (`FlockDeploy.setupOf true`), each table's
`FlockDeploy.repRounds true`, the session's `FlockDeploy.rounds true` and `(FlockDeploy.V true).k`, the record's streams'
sizes (`FlockDeploy.liveOf`) and `SessionMsgs.msgsAtCase`'s length; then `Flock.Zk.verify` on the record as kept, on it
re-serialized, and on two uneven splits with the same total: one round moved from rep 1's front to rep 0's end, and one
from rep 0's end to rep 1's front.

    VERITY_REPROUNDS_DIR=FORM_DIR python3 tools/verity/lean/lean_changed.py --try FILE

The blobs are taken under the digests `blobs.txt` lists, not hashed here.
-/

@[expose] public section

namespace Verity.SampledTwo.RepRoundsProbe

open Lean (Json)
open SessionMsgs

def die {α : Type} (msg : String) : IO α := throw (IO.userError msg)

def ofO {α : Type} (what : String) : Option α → IO α
  | some a => pure a
  | none => die s!"{what}: none"

def ofV {α : Type} (what : String) : Flock.V α → IO α
  | .ok a => pure a
  | .error e => die s!"{what}: {e}"

def say (t0 : Nat) (msg : String) : IO Unit := do
  let t ← IO.monoMsNow
  IO.println s!"REPROUNDS {(t - t0) / 1000} s: {msg}"
  (← IO.getStdout).flush

def verdict (v : Flock.V Unit) : String :=
  match v with
  | .ok () => "accepted"
  | .error e => s!"refused: {e}"

def readCase (dir : String) : IO Case := do
  let rd := fun (p : String) => IO.FS.readBinFile s!"{dir}/{p}"
  let lines := ((← IO.FS.readFile s!"{dir}/blobs.txt").splitOn "\n").filter (· ≠ "")
  let names := ((← System.FilePath.readDir s!"{dir}/blobs").map (·.fileName)).qsort (· < ·)
  if names.size != lines.length then die s!"{names.size} blobs, {lines.length} lines in blobs.txt"
  let entries ← (names.toList.zip lines).toArray.mapM fun (n, l) => do
    let d ← ofV "blobs.txt digest" (Flock.unhex ((l.splitOn " ").headD ""))
    pure (d, ← rd s!"blobs/{n}")
  pure ⟨← rd "params.json", ← rd "commitment.json", ← rd "evidence.json", ⟨entries⟩⟩

/-- The record with its streams `a` and `b` set to `f` of their rounds. -/
def reshape (record : String) (a b : String) (f : Array Json → Array Json → Array Json × Array Json) : IO String := do
  let j ← ofV "record" (Flock.JsonParse.parse record)
  let streams ← ofV "streams" (j.getObjValAs? (Array Json) "streams")
  let find := fun (name : String) => streams.findIdx? fun s => (s.getObjValAs? String "stream").toOption == some name
  let ia ← ofO s!"stream {a}" (find a)
  let ib ← ofO s!"stream {b}" (find b)
  let ra ← ofV a (streams[ia]!.getObjValAs? (Array Json) "rounds")
  let rb ← ofV b (streams[ib]!.getObjValAs? (Array Json) "rounds")
  let (ra', rb') := f ra rb
  let streams' := (streams.set! ia (streams[ia]!.setObjVal! "rounds" (Json.arr ra'))).set! ib
    (streams[ib]!.setObjVal! "rounds" (Json.arr rb'))
  pure (j.setObjVal! "streams" (Json.arr streams')).compress

def template (t0 : Nat) (k : Case) (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence)
    (t : Nat) : IO Unit := do
  let some x := stmtAtCase k t | say t0 s!"template {t}: the case gives no session statement"
  say t0 s!"template {t}: session statement of {x.length} bytes"
  let (_, spec, sts) ← ofO s!"template {t}: setupOf true" (FlockDeploy.setupOf true x)
  say t0 s!"template {t}: set up: {sts.size} table(s) {spec.tables}, points {spec.points}, rep streams {spec.repStreams.map (·.2.2)}"
  for st in sts do
    say t0 s!"template {t}: table: m {st.m}, kLog {st.c.kLog}, regions {st.regions.size}, repRounds true = {FlockDeploy.repRounds true spec st}"
  say t0 s!"template {t}: rounds true = {FlockDeploy.rounds true spec sts}, (V true).k x = {(FlockDeploy.V true).k x}"
  let s ← ofO s!"template {t}: filesAt" (filesAt p e t)
  let record ← ofO s!"template {t}: recordAt" (recordAt k.blobs s)
  let lv ← ofV s!"template {t}: liveOf" (FlockDeploy.liveOf spec record)
  say t0 s!"template {t}: liveOf: stream sizes {lv.streams.map (·.size)}, total {(lv.streams.map (·.size)).foldl (· + ·) 0}"
  match msgsAtCase k t with
  | some ms => say t0 s!"template {t}: msgsAtCase: {ms.length} messages"
  | none => say t0 s!"template {t}: msgsAtCase: none"
  let proofs ← ofV s!"template {t}: filesOf" (Spot.OneStageZk.filesOf k.blobs s (.ok (spec, #[], Json.null, Json.null)))
  let tabs := FlockDeploy.tabsOf sts
  say t0 s!"template {t}: Zk.verify, the record as kept: {verdict (Flock.Zk.verify spec tabs record proofs)}"
  let names := spec.repStreams.map (·.2.2)
  if names.size != 2 then die s!"template {t}: {names.size} rep streams, not two"
  let (a, b) := (names[0]!, names[1]!)
  let same ← reshape record a b fun ra rb => (ra, rb)
  say t0 s!"template {t}: Zk.verify, re-serialized: {verdict (Flock.Zk.verify spec tabs same proofs)}"
  let cases : List (String × (Array Json → Array Json → Array Json × Array Json)) :=
    [("one round from rep 1's front to rep 0's end", fun ra rb => (ra.push rb[0]!, rb.extract 1 rb.size)),
     ("one round from rep 0's end to rep 1's front", fun ra rb => (ra.pop, #[ra.back!] ++ rb))]
  for (what, f) in cases do
    let r ← reshape record a b f
    match FlockDeploy.liveOf spec r with
    | .ok lv' => say t0 s!"template {t}: {what}: liveOf: stream sizes {lv'.streams.map (·.size)}, total {(lv'.streams.map (·.size)).foldl (· + ·) 0}"
    | .error err => say t0 s!"template {t}: {what}: liveOf refused: {err}"
    say t0 s!"template {t}: {what}: Zk.verify: {verdict (Flock.Zk.verify spec tabs r proofs)}"

def main (args : List String) : IO UInt32 := do
  let [dir] := args | IO.eprintln "usage: RepRoundsProbe.lean FORM_DIR" *> pure 2
  let t0 ← IO.monoMsNow
  let k ← readCase dir
  say t0 s!"read the case: {k.blobs.entries.size} blobs"
  let (p, e) ← ofO "formOf" (formOf k)
  say t0 s!"templates {p.circuits.size}, stmts {e.stmts.size}, statements {e.statements.size}"
  for t in List.range p.circuits.size do
    template t0 k p e t
  pure 0

end Verity.SampledTwo.RepRoundsProbe

#eval show IO Unit from do
  if let some a := (← IO.getEnv "VERITY_REPROUNDS_DIR") then
    let rc ← Verity.SampledTwo.RepRoundsProbe.main [a]
    if rc != 0 then throw (IO.userError s!"RepRoundsProbe: exit {rc}")
