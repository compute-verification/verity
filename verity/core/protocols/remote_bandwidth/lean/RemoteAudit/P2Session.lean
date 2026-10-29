import Std.Data.HashMap
import Flock.Canon
import RemoteAudit.P2
import RemoteAudit.Pooled
import RemoteAudit.Session

/-!
# A session over a P2 store, of record: `p2_party.py`

The verifier's side of a remote bandwidth session over PoUS's P2 store (`p2.Session`, as `p2_party.session` drives
it), read from the bytes it holds and the prover sends, and decided here:

* the auditor's registration file (`heldOf`, `p2_party.registration_json`'s bytes): its fields' registration, its
  digest recomputed from them (a `digest` it states must be theirs), and the plan every pass reads over the store's U
  units;
* the prover's request line (`paramsOf`), refused unless it is the registration's node's;
* each root's line (`receive`) and each window's openings line (`verify`), every opening checked on PoUS's vk root
  (`checkOpenings`, `P2.checkOpeningWith`);
* the answer (`answer`, `p2_party.answer`'s): the credited reads under the pooled bound at δ = 2^-40
  (`Pooled.pooled`).

The bookkeeping (seeds, plans, steps, deadlines, order, draws) is `Session`'s over the store's units: the registration
stands in as one of U blocks (`standIn`) and the session's digest is the P2 registration's (`init`), so only an
opening's check is P2's. A P2 store is fixed at setup and its sessions take no appends or releases: the verifier's
process (`Verifier`) takes a start, a boundary or a close through `Session.step`, a root or an openings line here. A
line out of form rejects the session `malformed` at the window filling (`fail`), as `p2_party` does when it can't
read the prover's message. The registration's C* is too large to read whole: what is checked of it is each opened
unit's four blocks, on the registered vk root.
-/

namespace RemoteAudit.P2Session

open Lean (Json ToJson toJson)
open Flock

def PROTOCOL : String := "verity/remote-bandwidth/p2/v0"
def REQUEST_KEYS : List String := ["deadline_ns", "node", "passes", "samples", "window_ns", "windows"]
/-- `audit.LIFETIME_NS`. -/
def LIFETIME_NS : Nat := 30 * 10 ^ 9
/-- `audit.lifetime_windows(window_ns)`: the session's lifetime cap, which nothing in a P2 session reaches (it appends
nothing), and which its record states. -/
def lifetime (windowNs : Nat) : Nat := max 1 (LIFETIME_NS / windowNs)
/-- `audit.DELTA`, 2^-40: the pooled bound's δ. -/
def DELTA : Rat := 1 / ((2 ^ 40 : Nat) : Rat)

/-- What the verifier holds of the store: its registration and the plan every pass reads. -/
structure Held where
  reg : P2.Registration
  plan : Plan

private def jsonOf (b : ByteArray) : Option Json := do
  let s ← String.fromUTF8? b
  (Json.parse s).toOption

private def hexAt (j : Json) (k : String) : V ByteArray := do
  match unhex (← J.str j k "registration") with
  | .ok b => pure b
  | .error e => throw s!"registration.{k}: {e}"

private def runsOf (j : Json) : V (Array (Int × Int)) := do
  (← J.asArr j "registration.plan").mapM fun r => do
    let xs ← J.asArr r "registration.plan"
    if xs.size ≠ 2 then throw "a registration's plan is a list of runs [first, count]"
    pure (← J.asInt xs[0]! "registration.plan", ← J.asInt xs[1]! "registration.plan")

/-- `p2_party.registration_of` and `plan_of` on the registration file's bytes: its fields' registration
(`P2.Registration.make?`), refused if the `digest` it states isn't theirs, and its `plan` of runs over the store's
units, every unit without one. -/
def heldOf (bytes : ByteArray) : V Held := do
  let some j := jsonOf bytes | throw "a registration is a UTF-8 JSON object"
  let reg ← P2.Registration.make? (← hexAt j "node") (← J.nat j "w" "registration") (← J.nat j "c0" "registration")
    (← J.str j "expander" "registration") (← J.nat j "blocks_per_segment" "registration")
    (← J.nat j "segments" "registration") (← hexAt j "commitment") (← hexAt j "salt") (← hexAt j "root")
    (← J.nat j "rows" "registration")
  if J.has j "digest" then
    if (← hexAt j "digest") != reg.digest then throw "the registration's digest is not its fields'"
  let plan ← match j.getObjValD "plan" with
    | .null => pure (Plan.whole reg.units)
    | v => do
      let some pl := Plan.make? (← runsOf v) reg.units | throw "a registration's plan is a canonical plan of its units"
      pure pl
  pure { reg, plan }

/-- What the verifier's process answers on holding it: the digest, the store's units, and the plan's runs and units. -/
def Held.json (h : Held) : Json :=
  Json.mkObj [("digest", .str (hexOf h.reg.digest)),
    ("plan", .arr (h.plan.runs.map fun (f, c) => .arr #[toJson f, toJson c])), ("plan_units", toJson h.plan.length),
    ("units", toJson h.reg.units)]

/-- `p2_party.message(line, key)`: the value of a line's one key `key`; `none` for a line its sender didn't end (a
newline, its last byte), one not UTF-8 JSON, or not an object of that one key. -/
def messageOf (line : ByteArray) (key : String) : Option Json := do
  if line.size == 0 || line.get! (line.size - 1) != 10 then none
  let j ← jsonOf (line.extract 0 (line.size - 1))
  let .obj kvs := j | none
  if kvs.toArray.size != 1 then none
  (j.getObjVal? key).toOption

private def positive (j : Json) (k : String) : V Nat := do
  let n ← J.nat j k "request"
  if n = 0 then throw s!"request.{k} is a positive integer"
  pure n

/-- `p2_party.params_of` on the prover's request R: the session's parameters, refused unless R's keys are
`REQUEST_KEYS`, its node is the registration's, and the rest are positive integers. -/
def requestOf (held : Held) (r : Json) : V Session.Params := do
  let .obj kvs := r | throw s!"a request is {REQUEST_KEYS}"
  if !(kvs.toArray.size == REQUEST_KEYS.length && REQUEST_KEYS.all (J.has r ·)) then throw s!"a request is {REQUEST_KEYS}"
  let node ← match unhex (← J.str r "node" "request") with
    | .ok b => pure b
    | .error e => throw s!"request.node: {e}"
  if node != held.reg.node then throw "the request's node is not the registration's"
  let windowNs ← positive r "window_ns"
  pure { deadlineNs := ← positive r "deadline_ns", windows := ← positive r "windows", samples := ← positive r "samples",
         minPasses := ← positive r "passes", windowNs, lifetime := lifetime windowNs }

/-- `requestOf` on the prover's first line, `{"request": R}`. -/
def paramsOf (held : Held) (line : ByteArray) : V Session.Params := do
  let some r := messageOf line "request" | throw "the prover's first line is {\"request\": R}"
  requestOf held r

/-- The registration `Session` keeps a P2 session's books on: the store's U units as its blocks, which is all of it
`Session` reads here (the session's digest is the P2 registration's, `init`, and an opening is checked by
`checkOpenings`, never `RemoteAudit.checkWindow`). -/
def standIn (r : P2.Registration) : Registration :=
  let codec : Codec.Codec := ⟨Codec.P512, 2, false⟩
  { node := r.node, blocks := r.units, payloadRoot := zeros DIGEST_BYTES, codecSeed := zeros Codec.SEED_BYTES, codec,
    receipt := .sketch, rows := r.rows,
    key := { codec, c := #[], tau := 1, mask := none, seed := zeros Codec.SEED_BYTES } }

/-- `p2.Session.__init__`: a session over the held store under `params` with the 2W `coins` (empty ones are the
challenger's to set as it releases them), before its first event. -/
def init (held : Held) (params : Session.Params) (coins : Array ByteArray) (notBeforeNs : Nat) : Session.State :=
  { Session.State.init params (standIn held.reg) coins #[] notBeforeNs with digest := held.reg.digest }

/-- `p2.Session.verify` on window `w`'s draws: each opening checked on PoUS's vk root in the order drawn
(`P2.checkOpeningWith`, each pass's seed and sign rows derived once), the first that fails with its draw, or accept;
`malformed` for another number of openings than draws. -/
def checkOpenings (reg : P2.Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (openings : Array P2.Opening) :
    Verdict := Id.run do
  if openings.size ≠ drawn.size then return .reject .malformed none
  let mut keys : Std.HashMap Nat (ByteArray × Array ByteArray) := {}
  for k in [0:drawn.size] do
    let (i, p) := drawn[k]!
    let (seed, signs) := match keys.get? i with
      | some x => x
      | none => let s := w.seedOf i; (s, P2.unitSigns s reg.rows)
    keys := keys.insert i (seed, signs)
    let pr := w.passes[i]!
    if let some f := P2.checkOpeningWith signs reg seed pr.plan pr.root i p openings[k]! then
      return .reject f (some (i, p))
  return .accept

/-- `checkOpenings` over contiguous chunks of `size` draws, each chunk on a thread of its own: the first chunk's
rejection in draw order, which is `checkOpenings`'s on the whole window, or accept. -/
def checkOpeningsChunked (size : Nat) (reg : P2.Registration) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (openings : Array P2.Opening) : Verdict := Id.run do
  if openings.size ≠ drawn.size then return .reject .malformed none
  let size := max 1 size
  let tasks := (List.range ((drawn.size + size - 1) / size)).map fun c =>
    Task.spawn (prio := .dedicated) fun _ =>
      checkOpenings reg w (drawn.extract (c * size) ((c + 1) * size)) (openings.extract (c * size) ((c + 1) * size))
  for t in tasks do
    if t.get ≠ .accept then return t.get
  return .accept

/-- A line out of form, or a prover gone: the session rejected `malformed` at the window filling, as `p2_party`
rejects it, or refused once it has its outcome. -/
def fail (s : Session.State) : Session.Result × Session.State :=
  if s.outcome.isNone then s.fail s.filling .malformed else (.refused "the session has its outcome", s)

/-- Root `i` of window `t` from its line, `{"root": hex}` (a value not lowercase hex is no root: `malformed`), the
pass's plan the held one and its step the one after the session's last, arriving at `now` (`Session.receive`); a line
out of form is `fail`'s. -/
def receive (held : Held) (s : Session.State) (t i : Int) (line : ByteArray) (now : Nat) :
    Session.Result × Session.State :=
  match messageOf line "root" with
  | none => fail s
  | some v =>
    let root := match v with
      | .str x => (unhex x).toOption.getD ByteArray.empty
      | _ => ByteArray.empty
    Session.receive s t i root (some (held.plan.runs.map fun (f, c) => ((f : Int), (c : Int)))) none
      (some (s.step + 1)) now

private def hexesOf (j : Json) : Option (Array ByteArray) := do
  let .arr xs := j | none
  xs.mapM fun x => match x with
    | .str s => (unhex s).toOption
    | _ => none

/-- `p2_party.opening_of`: an opening from its JSON (`opening_json`), `none` for anything else. -/
def openingOf (j : Json) : Option P2.Opening := do
  let get := fun (k : String) => (j.getObjVal? k).toOption
  let int := fun (k : String) => (get k).bind fun v => v.getInt?.toOption
  let hex := fun (k : String) => (get k).bind fun v => match v with
    | .str s => (unhex s).toOption
    | _ => none
  let .arr paths ← get "paths" | none
  pure { pass := ← int "pass", position := ← int "position", blocks := ← (get "blocks").bind hexesOf,
         paths := ← paths.mapM hexesOf, leaf := ← hex "leaf", leafSalt := ← hex "leaf_salt",
         receiptPath := ← (get "receipt_path").bind hexesOf }

/-- A window's openings from its line's value: none (so `malformed`, another number than its draws) unless it is a
list of openings each in form. -/
def openingsOf : Json → Array P2.Opening
  | .arr xs => (xs.mapM openingOf).getD #[]
  | _ => #[]

/-- Window t's openings from their line, `{"openings": [...]}`, checked `size` draws a thread
(`checkOpeningsChunked`), and the window's verdict applied (`Session.verifyBy`); a line out of form is `fail`'s. -/
def verify (size : Nat) (held : Held) (s : Session.State) (t : Int) (line : ByteArray) :
    Session.Result × Session.State :=
  match messageOf line "openings" with
  | none => fail s
  | some v =>
    let openings := openingsOf v
    Session.verifyBy s t fun drawn => match s.awaiting t with
      | some (w, _) => checkOpeningsChunked size held.reg w drawn openings
      | none => .reject .malformed none

private def ratJ (q : Rat) : Json := .str (toString q)

private def atJ : Option (Nat × Nat) → Json
  | some (i, p) => .arr #[toJson i, toJson p]
  | none => .null

/-- What an accepted session credits: its receipts, the wrong ones the pooled bound at δ leaves (`Pooled.pooled`,
rounded up; all of them where it has none) and that bound, each window's draws, the credited reads `q_bytes`, the
receipts less the wrong ones at 8,224 bytes each, and the span they are over, from the first challenge's release to
the last window's deadline. -/
structure Credit where
  receipts : Nat
  wrong : Nat
  bound : Json
  draws : Array Nat
  qBytes : Nat
  spanNs : Nat

def credit (sm : Session.Summary) : Credit :=
  let ws := sm.windows.map fun w => (w.lengths.toList.sum, w.drawn)
  let us := (ws.map (·.1)).toList.sum
  let (wrong, bound) := match Pooled.pooled ws DELTA with
    | .ok b => ((b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den,
        Json.mkObj [("budget", ratJ b.budget), ("floor", ratJ b.floor), ("floors", .arr (b.floors.map ratJ)),
          ("wrong", ratJ b.wrong)])
    | .error why => (us, Json.mkObj [("refused", .str why)])
  { receipts := us, wrong, bound, draws := ws.map (·.2), qBytes := (us - wrong) * P2.UNIT_BYTES,
    spanNs := ((sm.windows.back?.map (·.deadlineNs)).getD 0) - ((sm.windows[0]?.map (·.releasedNs)).getD 0) }

/-- `p2_party.answer` less its `reason`: `accept` with the credited reads (`credit`), `reject` with the failure, its
window and the draw it failed at, or `undecided`; `units` are the store's and `plan_units` those each pass reads. -/
def answer (held : Held) (s : Session.State) : Json :=
  let windows := s.windows.map fun w => Json.mkObj [("passes", toJson w.roots.size),
    ("draws", match w.drawn with | some d => toJson d.size | none => .null),
    ("verdict", match w.verdict with
      | some v => Json.mkObj [("accepted", .bool (v == .accept)), ("at", atJ (v.1.bind (·.at?))),
          ("failure", match v.1 with | some r => .str r.failure.name | none => .null)]
      | none => .null)]
  let base := [("protocol", Json.str PROTOCOL), ("registration", .str (hexOf held.reg.digest)),
    ("unit_bytes", toJson P2.UNIT_BYTES), ("units", toJson held.reg.units), ("plan_units", toJson held.plan.length),
    ("windows", .arr windows)]
  match s.failed, s.summary with
  | some f, _ =>
    Json.mkObj (base ++ [("verdict", .str "reject"), ("failure", .str f.failure.name), ("at", atJ f.at?),
      ("window", toJson f.window)])
  | none, none => Json.mkObj (base ++ [("verdict", .str "undecided")])
  | none, some sm =>
    let c := credit sm
    Json.mkObj (base ++ [("verdict", .str "accept"), ("receipts", toJson c.receipts), ("wrong_at_most", toJson c.wrong),
      ("delta", ratJ DELTA), ("bound", c.bound), ("draws", toJson c.draws), ("q_bytes", toJson c.qBytes),
      ("span_ns", toJson c.spanNs),
      ("credited_bytes_per_s", if c.spanNs > 0 then toJson (c.qBytes * 10 ^ 9 / c.spanNs) else .null)])

end RemoteAudit.P2Session
