import CertifierDevice.Device
import Lean.Data.Json

/-!
# The difftest: `step` against `active.Proxy`, on generated traces

`scripts/difftest_vectors.py` writes traces (a configuration and the events of one run) and what the checkout's real
`active.Proxy` does on each, driven event by event in a world the script holds. `CertifierDevice/DifftestMain.lean`
reads both files and fails unless `evaluate`, which replays each trace through `step` from `init`, gives every result
exactly: each action in order (a link write's bytes, a connection, a server write's bytes and whether it waits for its
drain, a record's link, window, grid and status with its grid's parameters), and at the end the declared syncs, the
shaper's hand times, the log of egress frames read and the two counts. On an honest trace, which carries the Program's
frames, it also fails unless every record is complete and each egress record's grid is `NetworkCertifier.expectedRow`'s
under the syncs `step` declared. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and
not a proof.
-/

namespace CertifierDevice.Difftest

open Lean Flock
open NetworkCertifier (ClockSync SyncSet LogicalFrame Status)

def hexJson (b : ByteArray) : Json := .str (hexOf b)

def bytesOf (j : Json) : Except String ByteArray := do unhex (← j.getStr?)

def arrOf (j : Json) : Except String (Array Json) := j.getArr?

def paramsOf (j : Json) : Except String Params := do
  pure { buckets := ← j.getObjValAs? Nat "T", allocation := ← j.getObjValAs? Nat "r",
         frameBytes := ← j.getObjValAs? Nat "B", bucketMs := ← j.getObjValAs? Nat "tau",
         queueCapacity := ← j.getObjValAs? Nat "Q" }

def rhoOf (n : Int) (d : Nat) : Except String {ρ : Rat // 0 < ρ} :=
  let ρ := mkRat n d
  if h : d ≠ 0 ∧ 0 < ρ then pure ⟨ρ, h.2⟩ else throw s!"not a positive rate: {n}/{d}"

def syncOf (j : Json) : Except String ClockSync := do
  let #[d, n, den] ← arrOf j | throw s!"not [delta, num, den]: {j.compress}"
  let ρ ← rhoOf (← n.getInt?) (← den.getNat?)
  pure ⟨← d.getNat?, ρ.1, ρ.2⟩

def syncJson (s : ClockSync) : Json := .arr #[toJson s.delta, toJson s.rho.num, toJson s.rho.den]

def syncSetOf (j : Json) : Except String SyncSet := do
  let deltas := (← j.getObjValAs? (Array Nat) "deltas").toList
  let rhos ← (← arrOf (← j.getObjVal? "rhos")).toList.mapM fun r => do
    let #[n, d] ← arrOf r | throw s!"not [num, den]: {r.compress}"
    pure (mkRat (← n.getInt?) (← d.getNat?))
  if h : deltas ≠ [] ∧ rhos ≠ [] ∧ (∀ ρ ∈ rhos, 0 < ρ) ∧ deltas.Nodup ∧ rhos.Nodup then
    pure ⟨deltas, rhos, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩
  else throw s!"not a sync set: {j.compress}"

def pastOf (j : Json) : Except String Past := do
  let frontier ← (← arrOf (← j.getObjVal? "frontier")).mapM fun
    | .null => pure none
    | f => do
      let #[d, c] ← arrOf f | throw s!"not [delta, cost]: {f.compress}"
      pure (some ((← d.getNat?), (← c.getNat?)))
  pure { window := ← j.getObjValAs? Nat "window", sync := ← syncOf (← j.getObjVal? "sync"),
         frames := ← j.getObjValAs? Nat "frames", late := ← j.getObjValAs? Bool "late", frontier }

def proverOf : Json → Except String (Option ProverConfig)
  | .null => pure none
  | j => do
    let #[k, margin, fallback] ← arrOf (← j.getObjVal? "rule") | throw s!"not [k, margin, fallback]: {j.compress}"
    pure (some { rule := ⟨← k.getNat?, ← margin.getNat?, ← fallback.getBool?⟩,
                 envelope := ← syncOf (← j.getObjVal? "envelope"), lag := ← j.getObjValAs? Nat "lag",
                 prior := ← (← arrOf (← j.getObjVal? "prior")).mapM pastOf })

def saltOf : Json → Except String (Option SaltSource)
  | .null => pure none
  | j => do
    pure (some { values := ← (← arrOf (← j.getObjVal? "values")).mapM bytesOf, bits := ← j.getObjValAs? Nat "bits",
                 pouwWindow := ← j.getObjValAs? Nat "pouw_window" })

def servedOf : Json → Except String (Option ServedProfile)
  | .null => pure none
  | j => do
    pure (some { responseFrames := ← j.getObjValAs? Nat "F", requestFrames := ← j.getObjValAs? Nat "FIN",
                 windows := ← j.getObjValAs? Nat "windows", own := ← j.getObjValAs? Nat "own" })

/-- The configuration and the window and position the proxy starts at. -/
def configOf (j : Json) : Except String (Config × Nat × Nat) := do
  let c : Config := {
    egress := ← paramsOf (← j.getObjVal? "egress"), ingress := ← paramsOf (← j.getObjVal? "ingress"),
    syncSet := ← syncSetOf (← j.getObjVal? "sync_set"), syncs := ← (← arrOf (← j.getObjVal? "syncs")).mapM syncOf,
    windows := ← j.getObjValAs? Nat "windows", origin := ← j.getObjValAs? Nat "origin",
    slack := ← j.getObjValAs? Nat "slack", prover := ← proverOf (← j.getObjVal? "prover"),
    egressLink := ← j.getObjValAs? String "egress_link", ingressLink := ← j.getObjValAs? String "ingress_link",
    salt := ← saltOf (← j.getObjVal? "salt"), served := ← servedOf (← j.getObjVal? "served"),
    ordinals := ← j.getObjValAs? Bool "ordinals", compact := ← j.getObjValAs? Bool "compact" }
  let start ← j.getObjValAs? Nat "start"
  c.check start
  pure (c, start, ← j.getObjValAs? Nat "position")

def gridOf : String → Except String GridKind
  | "egress" => pure .egress
  | "ingress" => pure .ingress
  | g => throw s!"no grid {g}"

def eventOf (j : Json) : Except String Event := do
  let a ← arrOf j
  match a.toList with
  | [.str "tick", g, now, backlog] => pure (.tick (← gridOf (← g.getStr?)) (← now.getNat?) (← backlog.getNat?))
  | [.str "link", now, b] => pure (.link (← now.getNat?) (← bytesOf b))
  | [.str "connected", s, now] => pure (.connected (← s.getNat?) (← now.getNat?))
  | [.str "refused", s, now] => pure (.refused (← s.getNat?) (← now.getNat?))
  | [.str "delivered", s, now] => pure (.delivered (← s.getNat?) (← now.getNat?))
  | [.str "upstream", s, now, b] => pure (.upstream (← s.getNat?) (← now.getNat?) (← bytesOf b))
  | [.str "salt-connected", now] => pure (.saltConnected (← now.getNat?))
  | [.str "salt-refused", now] => pure (.saltRefused (← now.getNat?))
  | [.str "salt-delivered", now] => pure (.saltDelivered (← now.getNat?))
  | _ => throw s!"no event {j.compress}"

def paramsJson (p : Params) : Json :=
  Json.mkObj [("T", toJson p.buckets), ("r", toJson p.allocation), ("B", toJson p.frameBytes),
              ("tau", toJson p.bucketMs), ("Q", toJson p.queueCapacity)]

def slotJson : Option ByteArray → Json
  | none => .null
  | some b => hexJson b

def recordJson (r : Record) : Json :=
  Json.mkObj [("link", .str r.link), ("window", toJson r.window),
              ("grid", .arr (r.grid.map fun row => .arr (row.map slotJson))),
              ("status", match Status.reason? r.status with | none => .null | some x => .str (reasonValue x))]

def actionJson : Action → Json
  | .writeLink b => .arr #[.str "link", hexJson b]
  | .connect s => .arr #[.str "connect", toJson s]
  | .writeUpstream s b drain => .arr #[.str "upstream", toJson s, hexJson b, toJson drain]
  | .publish r p => .arr #[.str "publish", recordJson r, paramsJson p]
  | .connectSalt => .arr #[.str "connect-salt"]
  | .writeSalt b => .arr #[.str "salt", hexJson b]

/-- An honest run's check: every record complete, and egress window w's row i is `expectedRow` at bucket w·T + i of the
Program's frames under the syncs `step` declared. -/
def honest (c : Config) (syncs : Array ClockSync) (program : Array (ByteArray × Nat × Nat)) (records : Array Record) :
    Except String Unit := do
  for r in records do
    if let some x := Status.reason? r.status then throw s!"an honest run faulted: {r.link} window {r.window}: {reasonValue x}"
  let p := c.egress
  let frames : List LogicalFrame := program.toList.map fun (b, anchor, index) => ⟨bitsOf b, anchor, index⟩
  let W := syncs.size
  let σ : Fin W → ClockSync := fun w => syncs[w]
  for r in records.filter (·.link == c.egressLink) do
    for i in [0:r.grid.size] do
      let want := NetworkCertifier.expectedRow p.allocation p.buckets W σ frames (r.window * p.buckets + i)
      let got := r.grid[i]!
      if got.size != p.allocation then throw s!"window {r.window} row {i} has {got.size} slots"
      for h : j in [0:p.allocation] do
        if want ⟨j, h.upper⟩ != (got[j]!).map bitsOf then
          throw s!"window {r.window} row {i} slot {j} isn't expectedRow's"

/-- What `step` does on an instance, in the shape of the Python's results, and its path. -/
def evaluate (j : Json) : Except String (String × Json) := do
  let path ← j.getObjValAs? String "path"
  let (c, start, position) ← configOf (← j.getObjVal? "config")
  let events ← (← arrOf (← j.getObjVal? "events")).mapM eventOf
  let mut st := initAt c start position
  let mut actions : Array Action := #[]
  for ev in events do
    let (st', acts) := step st ev
    st := st'
    actions := actions ++ acts.toArray
  if !st.done then throw "the trace leaves ticks unrun"
  match ← j.getObjVal? "program" with
  | .null => pure ()
  | prog =>
    let program ← (← arrOf prog).mapM fun f => do
      let #[b, a, i] ← arrOf f | throw s!"not [payload, anchor, index]: {f.compress}"
      pure (← bytesOf b, ← a.getNat?, ← i.getNat?)
    honest c st.shaper.syncs program (actions.filterMap fun | .publish r _ => some r | _ => none)
  pure (path, Json.mkObj [
    ("actions", .arr (actions.map actionJson)),
    ("syncs", .arr (st.shaper.syncs.map syncJson)),
    ("hands", toJson st.shaper.hands),
    ("frames", .arr (st.frames.map fun (pos, s, k, a, r) => .arr #[toJson pos, toJson s, toJson k, toJson a, toJson r])),
    ("link_frames_read", toJson st.linkFramesRead),
    ("requests_released", toJson st.requestsReleased),
    ("salts_released", toJson st.saltsReleased),
    ("salts_delivered", toJson st.saltsDelivered)])

/-- The results' fields, in the order they are compared. -/
def fields : List String :=
  ["actions", "syncs", "hands", "frames", "link_frames_read", "requests_released", "salts_released", "salts_delivered"]

/-- A run's outcome: accepted, or the reasons its records name. -/
def outcome (results : Json) : String :=
  let reasons := ((results.getObjValD "actions").getArr?.toOption.getD #[]).toList.filterMap fun a =>
    match a with
    | .arr #[.str "publish", r, _] => match r.getObjValD "status" with | .str x => some x | _ => none
    | _ => none
  let distinct := reasons.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []
  if distinct.isEmpty then "accepted" else ",".intercalate (distinct.mergeSort (· ≤ ·))

/-- How the Lean side's results for instance line `i` differ from Python's, and the instance's path and outcome. -/
def differences (i : Nat) (inst python : String) : List String × String :=
  match Json.parse inst >>= evaluate, Json.parse python with
  | .error e, _ => ([s!"instance {i}: the Lean side rejects it: {e}"], "")
  | _, .error e => ([s!"instance {i}: Python's results don't parse: {e}"], "")
  | .ok (path, lean), .ok py =>
    let found := fields.filterMap fun k =>
      let a := lean.getObjValD k
      let b := py.getObjValD k
      if a == b then none else some s!"instance {i}: {k}: Lean {a.compress.take 400}, Python {b.compress.take 400}"
    let found := if found.isEmpty && lean != py then [s!"instance {i}: Python's results have fields Lean doesn't give"]
      else found
    (found, s!"{path} {outcome lean}")

/-- `certifier-device-difftest INSTANCES PYTHON N`: every trace of `INSTANCES` replayed through `step`, each compared
with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs; prints the count of each path
and outcome. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: certifier-device-difftest INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let results := tasks.map Task.get
  let found := results.flatMap (·.1)
  let counts := results.foldl (fun (m : Std.HashMap String Nat) (_, k) => m.insert k (m.getD k 0 + 1)) {}
  for (k, v) in counts.toList.mergeSort (fun x y => x.1 ≤ y.1) do IO.println s!"  {v} {k}"
  for d in found.take 50 do IO.eprintln d
  IO.println s!"difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end CertifierDevice.Difftest
