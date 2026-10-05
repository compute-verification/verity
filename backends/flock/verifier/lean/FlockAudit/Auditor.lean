import Flock
import FlockAudit.Prog

/-!
# The one-stage audit's auditor, as a program

The auditor's run, in the order the protocol fixes:

1. **Register.** It asks the developer for its registered public file, by content key, and reads the circuit (its own
   pin), that public file and the lookup tables from its content store, each held to its key. The population is the public
   file's instances.
2. **Draw, once.** `drawP` is the verifier's sampler `Draw.drawWith` with every byte a coin node of this program; under
   `flock-audit` each is `IO.getRandomBytes`, the bytes `Draw.drawOS` reads.
3. **Session.** It sets up the drawn statement at its own draw (`Stmt.setupTables … (some D.toJson)`), asks the serving
   side for a session at that draw, reads the record and the proofs from its content store, refuses a record whose draw
   is not its own (U1), and runs `Zk.verify`.
4. **Report.** It hands the verdict to the audit worker, whose answer is a report and never changes the verdict.

Registration and everything after the draw are `Talk`s, which have no coin node (`Talk.noCoin_lift`): a worker can ask
the program for nothing, a second draw included.
-/

namespace FlockAudit

open Flock
open Lean (Json)

/-- The longest JSON answer a worker may give. -/
def ANSWER_MAX : Nat := 2 ^ 20

/-- The longest object of the content store: a circuit, a public file, a lookup table, a record or a proof. -/
def OBJECT_MAX : Nat := 2 ^ 31

/-- What the auditor holds before the run: the statement's tags, the circuit's SHA-512 (its pin), the law's `k` (a
uniformly random `k`-subset of the registered population) and the tables of the session. -/
structure Own where
  tags : Tags
  circuit : String
  k : Nat
  count : Nat

/-- What registration fixed before the draw: the `--zk` tags and coins, the circuit and the registered public file (and
its key), the lookup tables the store holds, and the population. -/
structure Held where
  tags : Tags
  coins : Coins
  circuitB : ByteArray
  pubB : ByteArray
  publicKey : String
  tables : List (String × ByteArray)
  population : Nat

structure Verdict where
  accepted : Bool
  why : String
  draw : Option Json
  /-- the audit worker's answer: a report on the verdict, never a part of it -/
  report : Option Json := none

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj ([("accepted", Json.bool v.accepted), ("why", Json.str v.why), ("draw", v.draw.getD Json.null)] ++
    (match v.report with | some r => [("report", r)] | none => []))

def refused (draw : Option Json) (why : String) : Verdict := { accepted := false, why, draw }

/-- A part of the run that may refuse. -/
abbrev Step := ExceptT String Talk

def liftV {α : Type} (v : V α) : Step α := ExceptT.mk (pure v)

def ask (chan : String) (body : ByteArray) (max : Nat) : Step ByteArray := ExceptT.lift (Talk.req chan body max)

/-- A worker's answer: canonical JSON, or refused. -/
def answerJson (chan : String) (reply : ByteArray) : V Json := do
  let some t := String.fromUTF8? reply | throw s!"{chan}: the answer is not UTF-8"
  let j ← match Json.parse t with
    | .ok j => pure j
    | .error e => throw s!"{chan}: the answer is not JSON: {e}"
  if (← canon j) != t then throw s!"{chan}: the answer is not canonical JSON"
  return j

/-- A request whose body is canonical JSON, and its answer. -/
def askJson (chan : String) (body : Json) : Step Json := do
  let b ← liftV (canon body)
  liftV (answerJson chan (← ask chan b.toUTF8 ANSWER_MAX))

/-- The object the content store holds under `key`, held to it; `none` when it holds none. -/
def fetch? (key : String) : Step (Option ByteArray) := do
  let reply ← ask "archive" key.toUTF8 OBJECT_MAX
  if reply.size == 1 && reply.get! 0 == 0 then return none
  if reply.size == 0 || reply.get! 0 != 1 then throw s!"archive: a malformed answer for {key}"
  let b := reply.extract 1 reply.size
  if hexOf (Sha512.hash b) != key then throw s!"archive: the object held under {key} does not hash to it"
  return some b

def fetch (what key : String) : Step ByteArray := do
  let some b ← fetch? key | throw s!"{what}: the content store does not hold {key}"
  return b

/-- The registered population: the public file's instances under the circuit, read as `Stmt.setupTables` reads it. -/
def populationOf (tags : Tags) (circuitB pubB : ByteArray) (tables : List (String × ByteArray)) : V Nat :=
  if tags.hiddenOutputs then do
    let c ← HmOut.parse tags circuitB tables
    pure (← HmIn.loadPublic tags c (← HmIn.parse c) pubB).n
  else do pure (← HmRow.loadPublic tags (← HmRow.parse tags circuitB tables) pubB).n

/-- `flock-verify verify`'s checks of a table's statement without the verifier's registered reads or identifier: a
statement that reads registered values, or whose header names an identifier, is refused. -/
def ownChecks (st : Stmt) : V Unit := do
  Registered.check st.c st.pub none (Registered.derived st.partition)
  if (st.pub.header.getObjVal? "identifier").toBool then
    throw "identifier: the public header names an identifier, and the auditor holds none"

/-- A record's `unit_draw`, when it is a draw (not absent, not null). -/
def recordDraw (record : String) : Option Json :=
  match Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some d
    | .error _ => none
  | .error _ => none

/-- Registration, before any coin. -/
def register (own : Own) : Step Held := do
  let ans ← askJson "developer"
    (Json.mkObj [("ask", Json.str "register"), ("circuit", Json.str own.circuit), ("statement", Json.str own.tags.name)])
  let publicKey ← liftV (J.str ans "public" "developer")
  let circuitB ← fetch "circuit" own.circuit
  let pubB ← fetch "public file" publicKey
  let mut tables := []
  for t in MUFU_TABLES do
    if let some b ← fetch? t.sha512 then tables := tables ++ [(t.name, b)]
  let coins := Coins.tree (Zk.coinSpec (own.tags.tableNames own.count))
  let tags := Zk.tags (own.tags.withCoins coins) (Zk.maskLogOf circuitB) own.count
  let population ← liftV (populationOf tags circuitB pubB tables)
  if own.k > population then throw s!"law: k = {own.k} is over the registered population of {population}"
  return { tags, coins, circuitB, pubB, publicKey, tables, population }

/-- The auditor's one draw: the verifier's sampler with every byte a coin node of this program. -/
def drawP (k population : Nat) : Prog (Option Draw.UnitDraw) :=
  Draw.drawWith (m := Prog) (fun _ need => Prog.coins need) (.subset k) population

/-- The session at the auditor's own draw `D`. -/
def session (own : Own) (h : Held) (D : Draw.UnitDraw) : Step Unit := do
  let dj := D.toJson
  let (spec, sts) ← liftV (Stmt.setupTables h.tags h.circuitB h.pubB h.coins own.count h.tables (some dj))
  for st in sts do
    liftV (ownChecks st)
    liftV (Zk.shapeOk st)
  let tabs := sts.map Setup.ofCircuit
  let ans ← askJson "serving" (Json.mkObj [("ask", Json.str "session"), ("circuit", Json.str own.circuit), ("draw", dj),
    ("public", Json.str h.publicKey), ("tables", Lean.toJson own.count)])
  let recordKey ← liftV (J.str ans "record" "serving")
  let files ← liftV (J.get ans "files" "serving")
  let some record := String.fromUTF8? (← fetch "record" recordKey) | throw "serving: the record is not UTF-8"
  if (recordDraw record).bind (fun d => (canon d).toOption) != (canon dj).toOption then
    throw "U1: the record's unit draw is not the auditor's own"
  let proofs ← spec.repStreams.mapM fun (_, _, w) => do
    let name := s!"{w.replace "/" "."}.bin"
    fetch name (← liftV (J.str files name "serving: files"))
  liftV (Zk.verify spec tabs record proofs)

/-- Everything after the draw: the session check `sess` at `D`, and its verdict. -/
def afterDraw {H : Type} (sess : H → Draw.UnitDraw → Step Unit) (h : H) (D : Draw.UnitDraw) : Talk Verdict :=
  (fun
    | .ok () => { accepted := true, why := "", draw := some D.toJson }
    | .error e => refused (some D.toJson) e) <$> (sess h D).run

/-- The verdict: registration `reg`, then the one draw of `k` of the population it fixed, then `sess` at that draw. -/
def decide {H : Type} (k : Nat) (reg : Step H) (pop : H → Nat) (sess : H → Draw.UnitDraw → Step Unit) : Prog Verdict :=
  reg.run.lift.bind fun
    | .error e => .ret (refused none e)
    | .ok h => (drawP k (pop h)).bind fun
      | none => .ret (refused none "draw: the sampler ended")
      | some D => (afterDraw sess h D).lift

/-- The audit worker's report on the verdict; its answer never changes it. -/
def report (v : Verdict) : Talk Verdict := do
  match ← (askJson "audit" (Json.mkObj [("ask", Json.str "audit"), ("verdict", v.toJson)])).run with
  | .ok j => return { v with report := some j }
  | .error e => return { v with report := some (Json.mkObj [("error", Json.str e)]) }

/-- An auditor's run from its parts: registration, the one draw, the session check at it, and the report. -/
def auditorOf {H : Type} (k : Nat) (reg : Step H) (pop : H → Nat) (sess : H → Draw.UnitDraw → Step Unit) :
    Prog Verdict :=
  (decide k reg pop sess).bind fun v => (report v).lift

/-- The one-stage audit's auditor: `auditorOf` with this file's registration and session check. -/
def auditor (own : Own) : Prog Verdict := auditorOf own.k (register own) Held.population (session own)

/-- **The auditor draws once.** Whatever its registration and session check, an `auditorOf` run is registration, `drawP`,
and then the session and the report, and registration and everything after the draw have no coin node, whatever the
workers answer: the program's coins are its one draw's. `auditor` is `auditorOf` by definition. -/
theorem auditor_draws_once {H : Type} (k : Nat) (reg : Step H) (pop : H → Nat) (sess : H → Draw.UnitDraw → Step Unit) :
    Prog.NoCoin reg.run.lift ∧ (∀ h D, Prog.NoCoin ((afterDraw sess h D).bind report).lift) ∧
      auditorOf k reg pop sess = reg.run.lift.bind (fun
        | .error e => (report (refused none e)).lift
        | .ok h => (drawP k (pop h)).bind fun
          | none => (report (refused none "draw: the sampler ended")).lift
          | some D => ((afterDraw sess h D).bind report).lift) := by
  refine ⟨Talk.noCoin_lift _, fun _ _ => Talk.noCoin_lift _, ?_⟩
  unfold auditorOf decide
  rw [Prog.bind_assoc]
  congr 1
  funext x
  cases x with
  | error e => rfl
  | ok h =>
    simp only
    rw [Prog.bind_assoc]
    congr 1
    funext y
    cases y with
    | none => rfl
    | some D => exact (Talk.lift_bind _ _).symm

end FlockAudit
