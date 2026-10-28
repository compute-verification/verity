/-
`flock-verify rec-compose`: a recursive statement's verdict from its V* parts' verdicts. Each part is one
`flock-verify verify --zk` process at one V* statement (`--rec-part J`), with one session. The compiled verifier the
recursion's guarantee reads (`RecDeploy.V`, `Proofs.Flock.Claims.Recursion.Deploy`) runs part `s` at the statement
`RecDeploy.partRaw … s p`: of what `verify` reads, only the part's Flock statement (its circuit, partition, program and
registered reads, with its staged public file), its index and the verifier's own bindings of it vary by part, and every
other input is one value for every part (`Shared`, with the shared rows `--rec-rows`): the verifier's inner shape, the
inner statement's files and how it is set up, the inner session's record, the coin server's transcript and the circuit
registration. No part sees another, so only the composition can hold every part to one set. Each part's `VERDICT` names
the SHA-512 of each (`rec_shared`, `rec_rows_sha512`), whether it ran `--zk` (`zk`), and its session's coin key
(`coin_key`).

It accepts exactly when there is one log per V* statement, each with one `VERDICT`; `rec_part` `0` to `n − 1` each once;
every `VERDICT` accepts, ran `--zk`, names no identifier and names the composition's own rows file's digest; the first
part's `rec_shared` holds every input the compiled verifier's parts read, at the composition's own inner shape
(`Shared.Complete`), and every part's is that one; and no two sessions share a coin key (`verify` refuses a key reused
within one run, `CoinTree.checkFresh`, and this across runs). `Accepts` states it, and `check_accepts` proves it of
`check`; `Proofs.Flock.Claims.Recursion.ComposeParts` carries it to the parts' statements.

**One run** (C10 `OneRun`). The keys alone don't show that the parts' sessions are one run of the compiled verifier: a
prover could retry a part and hand over only the session that accepted. So `check` also reads the challenger's
interactive-run record of the run (`verity/rec-run/v1`, `--rec-run`, written by
`verity.core.protocols.sampled_proofs.challenger.RunRecord`, read from the challenger's host, never from the prover's
bundle): the staging of V*'s statements (the rows file's SHA-512 and the shared set), then every session the
challenger opened for the run's parts, each logged at its `Hello` before the answer that carries its fresh coin key,
then the close. It accepts only when the staging is the record's first event, at the composition's rows and the shared
set every part read; each of V*'s statements has exactly one session opened and no other index has one; every opened
session is a composed part's (its part and coin key) and every composed part's session is among them; and the close is
the last event (`checkRun`). `OneRun` states it, and `check_oneRun` proves it of `check`. The record holds no coin:
each session's coins are its own keyed coin tree's, drawn fresh at its `Hello`, never committed before the run. The
record is read once, in comparisons quadratic in V*'s statement count (at most 4096, `RecBudget`), never per gate.
-/
import Lean.Data.Json
import Flock.Bytes
import Flock.Hash

namespace Flock
namespace RecCompose

open Lean (Json)

/-- A file's SHA-512, in lowercase hex. -/
def digest (b : ByteArray) : String := hexOf (Sha512.hash b)

/-- **What every part of one recursive statement reads alike**, as its `VERDICT` names it (`rec_shared`): the SHA-512 of
the verifier's own inner shape (`--rec-inner`), of the inner statement's files (`--rec-inner-circuit`,
`--rec-inner-public`, `--rec-inner-partition`), of a record of check 5's coins (`--rec-inner-coins`, which `verify`
refuses), of the inner session's record (`--rec-inner-session`'s `session.json`), of the coin server's transcript
(`--rec-inner-transcript`) and of the circuit registration (`--rec-inner-registration`); how the inner statement is set
up (`--rec-inner-statement`, its coins `os` as `true`, each lookup table's name and SHA-512, and the SHA-512 of
`--rec-inner-draw`). `none` where `verify` held no such input. -/
structure Shared where
  recInner : Option String
  innerCircuit : Option String
  innerPublic : Option String
  innerPartition : Option String
  innerCoins : Option String
  innerSession : Option String
  transcript : Option String
  registration : Option String
  innerStatement : Option String
  innerStatementCoins : Option Bool
  innerTables : Option (List (String × String))
  innerDraw : Option String
  deriving DecidableEq, Repr, Inhabited

/-- **The shared inputs' digests**, from the bytes `verify` read. -/
def Shared.ofFiles (recInner innerCircuit innerPublic innerPartition innerCoins innerSession transcript registration :
    Option ByteArray) (innerStatement : Option String) (innerStatementCoins : Option Bool)
    (innerTables : Option (List (String × ByteArray))) (innerDraw : Option ByteArray) : Shared :=
  ⟨recInner.map digest, innerCircuit.map digest, innerPublic.map digest, innerPartition.map digest,
    innerCoins.map digest, innerSession.map digest, transcript.map digest, registration.map digest, innerStatement,
    innerStatementCoins, innerTables.map (·.map fun t => (t.1, digest t.2)), innerDraw.map digest⟩

/-- A string or `null`. -/
def optJson : Option String → Json
  | none => .null
  | some s => .str s

def Shared.toJson (s : Shared) : Json :=
  Json.mkObj [("rec_inner", optJson s.recInner), ("inner_circuit", optJson s.innerCircuit),
    ("inner_public", optJson s.innerPublic), ("inner_partition", optJson s.innerPartition),
    ("inner_coins", optJson s.innerCoins), ("inner_session", optJson s.innerSession),
    ("inner_transcript", optJson s.transcript), ("inner_registration", optJson s.registration),
    ("inner_statement", optJson s.innerStatement),
    ("inner_statement_coins", optJson (s.innerStatementCoins.map fun os => if os then "os" else "seed")),
    ("inner_tables", match s.innerTables with
      | none => .null
      | some ts => .arr (ts.map fun t => Json.arr #[.str t.1, .str t.2]).toArray),
    ("inner_draw", optJson s.innerDraw)]

/-- Field `k` of `j`: a string, or `null` as `none`; nothing when absent or another value. -/
def readOpt (j : Json) (k : String) : Option (Option String) :=
  match j.getObjVal? k with
  | .ok .null => some none
  | .ok (.str s) => some (some s)
  | _ => none

/-- A lookup table as `Shared.toJson` writes it: its name and SHA-512. -/
def readTable : Json → Option (String × String)
  | .arr a => match a.toList with
    | [.str n, .str d] => some (n, d)
    | _ => none
  | _ => none

/-- `Shared.toJson` read back; nothing where a field is absent or not in its form. -/
def Shared.ofJson? (j : Json) : Option Shared := do
  let coins ← match j.getObjVal? "inner_statement_coins" with
    | .ok .null => some none
    | .ok (.str "os") => some (some true)
    | .ok (.str "seed") => some (some false)
    | _ => none
  let tables ← match j.getObjVal? "inner_tables" with
    | .ok .null => some none
    | .ok (.arr ts) => some <$> ts.toList.mapM readTable
    | _ => none
  pure ⟨← readOpt j "rec_inner", ← readOpt j "inner_circuit", ← readOpt j "inner_public", ← readOpt j "inner_partition",
    ← readOpt j "inner_coins", ← readOpt j "inner_session", ← readOpt j "inner_transcript",
    ← readOpt j "inner_registration", ← readOpt j "inner_statement", coins, tables, ← readOpt j "inner_draw"⟩

/-- The first field in which `s` differs from `t`, the first part's, as a refusal's reason. -/
def Shared.diff (s t : Shared) : String :=
  let show? := fun (o : Option String) => o.getD "none"
  let one := fun (key what : String) (a b : Option String) =>
    if a != b then some s!"its {key} ({what}) is {show? a}, not {show? b} as the first part's" else none
  (one "rec_inner" "the verifier's inner shape, --rec-inner" s.recInner t.recInner <|>
    one "inner_circuit" "the inner statement's circuit, --rec-inner-circuit" s.innerCircuit t.innerCircuit <|>
    one "inner_public" "the inner statement's public file, --rec-inner-public" s.innerPublic t.innerPublic <|>
    one "inner_partition" "the inner statement's partition, --rec-inner-partition" s.innerPartition t.innerPartition <|>
    one "inner_coins" "a record of check 5's coins, --rec-inner-coins" s.innerCoins t.innerCoins <|>
    one "inner_session" "the inner session's record, --rec-inner-session" s.innerSession t.innerSession <|>
    one "inner_transcript" "the coin server's transcript, --rec-inner-transcript" s.transcript t.transcript <|>
    one "inner_registration" "the circuit registration, --rec-inner-registration" s.registration t.registration <|>
    one "inner_statement" "how the inner statement is proved, --rec-inner-statement" s.innerStatement t.innerStatement <|>
    one "inner_statement_coins" "the inner statement's coins, --rec-inner-statement-coins"
      (s.innerStatementCoins.map toString) (t.innerStatementCoins.map toString) <|>
    one "inner_tables" "the inner statement's lookup tables, --rec-inner-tables"
      (s.innerTables.map toString) (t.innerTables.map toString) <|>
    one "inner_draw" "the inner statement's draw, --rec-inner-draw" s.innerDraw t.innerDraw).getD
    "its rec_shared is not the first part's"

/-- **A part's verdict as `rec-compose` reads it** off its `VERDICT` line. -/
structure Verdict where
  accepted : Bool
  why : String
  part : Option Nat
  rows : Option String
  shared : Option Shared
  zk : Option Bool
  coinKey : Option String
  identifier : Bool
  deriving Repr, Inhabited

def Verdict.ofJson (j : Json) : Verdict where
  accepted := (j.getObjValAs? Bool "accepted").toOption == some true
  why := (j.getObjValAs? String "why").toOption.getD ""
  part := (j.getObjValAs? Nat "rec_part").toOption
  rows := (j.getObjValAs? String "rec_rows_sha512").toOption
  shared := (j.getObjVal? "rec_shared").toOption.bind Shared.ofJson?
  zk := (j.getObjValAs? Bool "zk").toOption
  coinKey := (j.getObjValAs? String "coin_key").toOption
  identifier := (j.getObjVal? "identifier").toOption.isSome

/-- **The shared inputs in the compiled verifier's form**: every part of `RecDeploy.V` reads the verifier's own inner
shape, here the composition's (`inner`, its SHA-512), the inner statement's three files, the inner session's record,
the coin server's transcript and how the inner statement is proved, and no record of check 5's coins. -/
def Shared.Complete (inner : String) (s : Shared) : Prop :=
  s.recInner = some inner ∧ s.innerCircuit.isSome ∧ s.innerPublic.isSome ∧ s.innerPartition.isSome ∧
    s.innerCoins = none ∧ s.innerSession.isSome ∧ s.transcript.isSome ∧ s.innerStatement.isSome ∧
    s.innerStatementCoins.isSome ∧ s.innerTables.isSome

/-- **`rec-compose`'s acceptance** over the parts' verdicts, at V*'s `n` statements, the composition's own rows file's
SHA-512 `rows` and inner shape's `inner`: one verdict per statement, `rec_part` `j` once for each `j < n`; each accepted
at one of them,
`--zk`, with no identifier, at `rows`; one `Shared` in the compiled verifier's form at `inner` that every part names; and
every session's coin key named, no two alike. -/
def Accepts (n : Nat) (rows inner : String) (vs : List Verdict) : Prop :=
  0 < n ∧ vs.length = n ∧ (∀ j, j < n → (vs.map (·.part)).count (some j) = 1) ∧
    (∀ v ∈ vs, v.accepted = true ∧ v.zk = some true ∧ v.identifier = false ∧ v.rows = some rows ∧
      ∃ j, v.part = some j ∧ j < n) ∧
    (∃ sh, sh.Complete inner ∧ ∀ v ∈ vs, v.shared = some sh) ∧
    (∀ v ∈ vs, v.coinKey.isSome) ∧ (vs.map (·.coinKey)).Nodup

/-- A log's `VERDICT` lines, each a JSON object, in order. -/
def verdicts (log : String) : Except String (Array Json) :=
  (log.splitOn "\n").foldlM (init := #[]) fun out line =>
    if line.startsWith "VERDICT " then
      match Json.parse (line.drop 8).toString with
      | .ok j => pure (out.push j)
      | .error e => throw s!"a VERDICT line that is not JSON: {e}"
    else pure out

/-- **One part's log**: it has exactly one `VERDICT` (a V* part is one session). Its verdict. -/
def partOf (name log : String) : Except String (String × Verdict) := do
  let vs ← (verdicts log).mapError (s!"{name}: " ++ ·)
  match vs.toList with
  | [v] => pure (name, Verdict.ofJson v)
  | [] => throw s!"{name}: no VERDICT"
  | _ => throw s!"{name}: {vs.size} VERDICTs: a V* part is one session (RecDeploy.V runs one per part)"

/-- **One part's verdict on its own**: accepted, at one of V*'s `n` statements, `--zk`, without an identifier, and at
the composition's rows file of SHA-512 `rows`. -/
def checkOne (n : Nat) (rows : String) (name : String) (v : Verdict) : Except String Unit :=
  if v.accepted = false then .error s!"{name}: a session is not accepted ({v.why})"
  else match v.part with
  | none => .error s!"{name}: its VERDICT names no rec_part: not one of V*'s parts (--rec-part J)"
  | some j =>
    if n ≤ j then .error s!"rec_part {j}: V* has {n} statements"
    else if v.zk ≠ some true then
      .error s!"{name}: not a --zk session: V*'s parts are masked (RecDeploy.insO), the only form the recursion's proof reads"
    else if v.identifier = true then
      .error s!"{name}: its VERDICT names an identifier (--identifier): V*'s parts set up without one (RecIns.setup), the only form the recursion's proof reads"
    else match v.rows with
    | none => .error s!"{name}: its VERDICT names no rec_rows_sha512: it was not checked against the shared rows (--rec-rows)"
    | some h =>
      if h = rows then .ok () else .error s!"{name}: checked against the rows file {h}, not the one staged rows file {rows}"

/-- The refusal of a part whose verify held no `what`. -/
def held (name what : String) : String :=
  s!"{name}: its verify held no {what}, which every part of the compiled verifier reads"

/-- **The first part's shared inputs are the compiled verifier's** at the composition's inner shape of SHA-512 `inner`
(`Shared.Complete`), each refusal naming its input. -/
def checkForm (name inner : String) (s : Shared) : Except String Unit :=
  match s.recInner with
  | none => .error s!"{name}: its VERDICT names no rec_inner: it was not verified at V*'s inner shape (--rec-inner)"
  | some h =>
    if h ≠ inner then .error s!"{name}: verified at the inner shape {h}, not the composition's own --rec-inner {inner}"
    else if s.innerCircuit.isNone then .error (held name "inner statement's circuit (--rec-inner-circuit)")
    else if s.innerPublic.isNone then .error (held name "inner statement's public file (--rec-inner-public)")
    else if s.innerPartition.isNone then .error (held name "inner statement's partition (--rec-inner-partition)")
    else if s.innerCoins.isSome then
      .error s!"{name}: it read a record of check 5's coins (--rec-inner-coins), which no part of the compiled verifier reads"
    else if s.innerSession.isNone then .error (held name "inner session's record (--rec-inner-session)")
    else if s.transcript.isNone then .error (held name "coin server's transcript (--rec-inner-transcript)")
    else if s.innerStatement.isNone then .error (held name "word on how the inner statement is proved (--rec-inner-statement)")
    else if s.innerStatementCoins.isNone then .error (held name "inner statement's coins (--rec-inner-statement)")
    else if s.innerTables.isNone then .error (held name "inner statement's lookup tables (--rec-inner-statement)")
    else .ok ()

/-- **Every part names the one shared set** `sh`. -/
def checkShared (sh : Shared) (name : String) (v : Verdict) : Except String Unit :=
  match v.shared with
  | none => .error s!"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named"
  | some s => if s = sh then .ok () else .error s!"{name}: {s.diff sh}"

/-- **No two sessions share a coin key**, each named. -/
def checkKeys : List (String × Verdict) → Except String Unit
  | [] => .ok ()
  | (name, v) :: rest =>
    match v.coinKey with
    | none => .error s!"{name}: its VERDICT names no coin_key: its session's coins are not named"
    | some k =>
      if rest.any (·.2.coinKey == some k) then
        .error s!"{name}: its session's coin key is another part's: a session key is never reused (CoinTree.checkFresh)"
      else checkKeys rest

/-- `rec_part j` has one part. -/
def checkIndex (vs : List (String × Verdict)) (j : Nat) : Except String Unit :=
  if (vs.map (·.2.part)).count (some j) = 1 then .ok ()
  else .error s!"rec_part {j}: {(vs.map (·.2.part)).count (some j)} logs, not one"

/-- **The composition on the parts' verdicts** (name, verdict): `checkOne` on each, the first part's shared inputs in
the compiled verifier's form at `inner` (`checkForm`) and every part's the same (`checkShared`), one part per statement
(`checkIndex`, and as many parts as statements) and fresh coin keys (`checkKeys`). -/
def checkVerdicts (n : Nat) (rows inner : String) (vs : List (String × Verdict)) : Except String Unit :=
  if n = 0 then .error "no V* statement" else
  (vs.forM fun p => checkOne n rows p.1 p.2) >>= fun _ =>
  match vs with
  | [] => .error s!"0 parts, not V*'s {n} statements"
  | (name, v) :: _ =>
    match v.shared with
    | none => .error s!"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named"
    | some sh =>
      checkForm name inner sh >>= fun _ =>
      (vs.forM fun p => checkShared sh p.1 p.2) >>= fun _ =>
      ((List.range n).forM (checkIndex vs)) >>= fun _ =>
      (if vs.length = n then .ok () else .error s!"{vs.length} parts, not V*'s {n} statements") >>= fun _ =>
      checkKeys vs

/-! ## The challenger's interactive-run record -/

/-- **A session the challenger opened** for one of the run's parts, at the prover's `Hello`, before it answered: the
session's id, its part (`--rec-part`) and the fresh coin key the answer carries (the session's `coin_tree.key`,
`Flock.CoinTree`). -/
structure Opened where
  session : String
  part : Nat
  coinKey : String
  deriving DecidableEq, Repr, Inhabited

/-- **An event of the interactive-run record** (`verity/rec-run/v1`, `challenger.RunRecord`): V*'s statements staged
(the shared rows file's SHA-512 and the shared set every part reads, as `rec_shared` names it), a session opened, or
the run closed. -/
inductive Event where
  | stage (rows : String) (shared : Shared)
  | opened (o : Opened)
  | close
  deriving DecidableEq, Repr, Inhabited

def RUN_KIND : String := "verity/rec-run/v1"

/-- One event as the challenger writes it: `{"event": "stage", "rec_rows_sha512", "rec_shared"}`,
`{"event": "open", "session", "rec_part", "coin_key"}` or `{"event": "close"}`. -/
def Event.ofJson? (j : Json) : Option Event :=
  match j.getObjValAs? String "event" with
  | .ok "stage" => do
    let rows ← (j.getObjValAs? String "rec_rows_sha512").toOption
    pure (.stage rows (← (j.getObjVal? "rec_shared").toOption.bind Shared.ofJson?))
  | .ok "open" => do
    pure (.opened ⟨← (j.getObjValAs? String "session").toOption, ← (j.getObjValAs? Nat "rec_part").toOption,
      ← (j.getObjValAs? String "coin_key").toOption⟩)
  | .ok "close" => some .close
  | _ => none

/-- The events from the `i`-th on, each in its form. -/
def eventsOf : Nat → List Json → Except String (List Event)
  | _, [] => .ok []
  | i, e :: es => match Event.ofJson? e with
    | some ev => (eventsOf (i + 1) es).map (ev :: ·)
    | none => .error s!"run record: its event {i} is not a stage, an open or a close in its form"

/-- **The record** `{"kind": "verity/rec-run/v1", "events": [...]}`: its events, in the order the challenger logged
them. -/
def runOf (text : String) : Except String (List Event) :=
  match Json.parse text with
  | .error _ => .error "run record: not JSON"
  | .ok j =>
    if (j.getObjValAs? String "kind").toOption != some RUN_KIND then .error s!"run record: not a {RUN_KIND} record"
    else match (j.getObjValAs? (Array Json) "events").toOption with
      | none => .error "run record: no list of events"
      | some evs => eventsOf 0 evs.toList

/-- The sessions opened after the staging, up to the close, which is the record's last event. -/
def opensOf : List Event → Except String (List Opened)
  | [] => .error "run record: not closed: the challenger closes a run's record after its last session, and an open record may gain sessions"
  | [.close] => .ok []
  | .close :: _ :: _ => .error "run record: an event after its close"
  | .stage .. :: _ =>
    .error "run record: a second staging: V*'s statements are staged once, before the first part's Hello"
  | .opened o :: rest => (opensOf rest).map (o :: ·)

/-- The verdict `v` is the session `o`'s: at its part, on its coin key. -/
def isOf (o : Opened) (v : Verdict) : Bool := v.part == some o.part && v.coinKey == some o.coinKey

/-- Every part read the shared set the record staged. -/
def checkStaged (sh : Shared) (name : String) (v : Verdict) : Except String Unit :=
  if v.shared = some sh then .ok ()
  else .error s!"{name}: its rec_shared is not the one the run record staged before the first Hello"

/-- An opened session is for one of V*'s `n` statements. -/
def checkBound (n : Nat) (o : Opened) : Except String Unit :=
  if o.part < n then .ok ()
  else .error s!"run record: session {o.session} was opened for part {o.part}: V* has {n} statements"

/-- **Part `j` had exactly one session opened.** -/
def checkOnce (opens : List Opened) (j : Nat) : Except String Unit :=
  if (opens.map (·.part)).count j = 1 then .ok ()
  else .error s!"part {j}: {(opens.map (·.part)).count j} sessions opened in the run record, not one: a part is never retried"

/-- **Every opened session is a composed part's.** -/
def checkOpened (vs : List (String × Verdict)) (o : Opened) : Except String Unit :=
  if vs.any fun p => isOf o p.2 then .ok ()
  else .error s!"run record: session {o.session} was opened for part {o.part}, and it is not the session composed for that part"

/-- **Every composed part's session was opened** in the run. -/
def checkRecorded (opens : List Opened) (name : String) (v : Verdict) : Except String Unit :=
  if opens.any fun o => isOf o v then .ok ()
  else .error s!"{name}: its session is not one the run record opened for its part"

/-- **The parts are one run** (C10 `OneRun`) of the record `evs`: staging first, at the composition's rows `rows` and
the shared set every part read; then the sessions opened, each for one of V*'s `n` statements and exactly one for each
(`checkOnce`), every one a composed part's (`checkOpened`) and every composed part's among them (`checkRecorded`); then
the close. -/
def checkRun (n : Nat) (rows : String) (vs : List (String × Verdict)) (evs : List Event) : Except String Unit :=
  match evs with
  | .stage r sh :: rest =>
    if r ≠ rows then .error s!"run record: it staged the rows file {r}, not the composition's {rows}" else
    (vs.forM fun p => checkStaged sh p.1 p.2) >>= fun _ =>
    opensOf rest >>= fun opens =>
    (opens.forM (checkBound n)) >>= fun _ =>
    ((List.range n).forM (checkOnce opens)) >>= fun _ =>
    (opens.forM (checkOpened vs)) >>= fun _ =>
    vs.forM fun p => checkRecorded opens p.1 p.2
  | _ => .error "run record: its first event is not the staging: V*'s statements are staged before the first part's Hello"

/-- **The composition**: `parts`' logs (name, text), one `VERDICT` each (`partOf`), composed at V*'s `n` statements,
the composition's own rows file's SHA-512 `rows` and inner shape's `inner` (`checkVerdicts`), and one run of the
challenger's interactive-run record `run` (`runOf`, `checkRun`). -/
def check (n : Nat) (rows inner : String) (run : String) (parts : List (String × String)) : Except String Unit :=
  (parts.mapM fun p => partOf p.1 p.2) >>= fun vs =>
  checkVerdicts n rows inner vs >>= fun _ =>
  runOf run >>= checkRun n rows vs

/-! ## `check` accepts only what `Accepts` states -/

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem forM_ok {α : Type} {f : α → Except String Unit} :
    ∀ {l : List α} {u : Unit}, l.forM f = .ok u → ∀ a ∈ l, f a = .ok ()
  | [], _, _, a, ha => by cases ha
  | b :: l, _, h, a, ha => by
    dsimp only [List.forM] at h
    obtain ⟨⟨⟩, hb, h⟩ := bind_ok h
    cases ha with
    | head => exact hb
    | tail _ hm => exact forM_ok h a hm

theorem checkOne_ok {n : Nat} {rows name : String} {v : Verdict} (h : checkOne n rows name v = .ok ()) :
    v.accepted = true ∧ v.zk = some true ∧ v.identifier = false ∧ v.rows = some rows ∧
      ∃ j, v.part = some j ∧ j < n := by
  unfold checkOne at h
  split at h
  · cases h
  rename_i ha
  split at h
  · cases h
  rename_i j hj
  split at h
  · cases h
  rename_i hn
  split at h
  · cases h
  rename_i hz
  split at h
  · cases h
  rename_i hi
  split at h
  · cases h
  rename_i hr
  split at h
  · rename_i hh
    exact ⟨by simpa using ha, by simpa using hz, by simpa using hi, by rw [hr, hh], j, hj, by omega⟩
  · cases h

theorem checkForm_ok {name inner : String} {s : Shared} (h : checkForm name inner s = .ok ()) : s.Complete inner := by
  unfold checkForm at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i hr hi h1 h2 h3 h4 h5 h6 h7 h8 h9
  refine ⟨by rw [hr]; simpa using hi, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp_all [Option.isNone_iff_eq_none, Option.isSome_iff_ne_none]

theorem checkShared_ok {sh : Shared} {name : String} {v : Verdict} (h : checkShared sh name v = .ok ()) :
    v.shared = some sh := by
  unfold checkShared at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i hs e
  rw [hs, e]

theorem checkKeys_ok : ∀ {vs : List (String × Verdict)}, checkKeys vs = .ok () →
    (∀ p ∈ vs, p.2.coinKey.isSome) ∧ (vs.map (·.2.coinKey)).Nodup
  | [], _ => ⟨by simp, List.nodup_nil⟩
  | (name, v) :: rest, h => by
    unfold checkKeys at h
    split at h
    · cases h
    rename_i k hk
    split at h
    · cases h
    rename_i hany
    obtain ⟨h1, h2⟩ := checkKeys_ok h
    refine ⟨?_, ?_⟩
    · intro p hp
      cases hp with
      | head => simp [hk]
      | tail _ hm => exact h1 p hm
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨?_, h2⟩
      intro hm
      obtain ⟨q, hq, he⟩ := List.mem_map.1 hm
      apply hany
      rw [List.any_eq_true]
      exact ⟨q, hq, by simp [he, hk]⟩

theorem checkIndex_ok {vs : List (String × Verdict)} {j : Nat} (h : checkIndex vs j = .ok ()) :
    (vs.map (·.2.part)).count (some j) = 1 := by
  unfold checkIndex at h
  split at h
  · assumption
  · cases h

/-- **`checkVerdicts` accepts only what `Accepts` states.** -/
theorem checkVerdicts_accepts {n : Nat} {rows inner : String} {vs : List (String × Verdict)}
    (h : checkVerdicts n rows inner vs = .ok ()) : Accepts n rows inner (vs.map (·.2)) := by
  unfold checkVerdicts at h
  split at h
  · cases h
  rename_i hn
  obtain ⟨_, hone, h⟩ := bind_ok h
  obtain _ | ⟨⟨name, v⟩, rest⟩ := vs
  · cases h
  dsimp only at h
  split at h
  · cases h
  rename_i sh hsh
  obtain ⟨_, hform, h⟩ := bind_ok h
  obtain ⟨_, hshared, h⟩ := bind_ok h
  obtain ⟨_, hidx, h⟩ := bind_ok h
  obtain ⟨_, hlen, h⟩ := bind_ok h
  have hl : ((name, v) :: rest).length = n := by
    split at hlen
    · assumption
    · cases hlen
  obtain ⟨hk1, hk2⟩ := checkKeys_ok h
  refine ⟨Nat.pos_of_ne_zero hn, by simpa using hl, ?_, ?_, ⟨sh, checkForm_ok hform, ?_⟩, ?_, ?_⟩
  · intro j hj
    rw [List.map_map]
    exact checkIndex_ok (forM_ok hidx j (List.mem_range.2 hj))
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact checkOne_ok (forM_ok hone p hp)
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact checkShared_ok (forM_ok hshared p hp)
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact hk1 p hp
  · rw [List.map_map]
    exact hk2

/-- **C10 `OneRun`**, what `checkRun` holds the challenger's record `evs` to against the parts' verdicts `vs`: the
staging is its first event, at the composition's rows `rows` and the one shared set `sh` every part read; then come the
sessions it opened, `opens`, and then its close. Every opened session is for one of V*'s `n` statements, each
statement has exactly one, and the opened sessions are the composed parts' (each at its part on its coin key, both
ways). -/
def OneRun (n : Nat) (rows : String) (vs : List Verdict) (evs : List Event) : Prop :=
  ∃ (sh : Shared) (opens : List Opened), evs = Event.stage rows sh :: (opens.map Event.opened ++ [Event.close]) ∧
    (∀ v ∈ vs, v.shared = some sh) ∧ (∀ o ∈ opens, o.part < n) ∧
    (∀ j, j < n → (opens.map (·.part)).count j = 1) ∧
    (∀ o ∈ opens, ∃ v ∈ vs, v.part = some o.part ∧ v.coinKey = some o.coinKey) ∧
    (∀ v ∈ vs, ∃ o ∈ opens, v.part = some o.part ∧ v.coinKey = some o.coinKey)

theorem opensOf_ok : ∀ {evs : List Event} {opens : List Opened}, opensOf evs = .ok opens →
    evs = opens.map Event.opened ++ [Event.close]
  | [], _, h => by simp [opensOf] at h
  | [.close], _, h => by
    simp only [opensOf, Except.ok.injEq] at h
    subst h
    rfl
  | .close :: _ :: _, _, h => by simp [opensOf] at h
  | .stage .. :: _, _, h => by simp [opensOf] at h
  | .opened o :: rest, opens, h => by
    simp only [opensOf] at h
    cases hr : opensOf rest with
    | error e => rw [hr] at h; cases h
    | ok os =>
      rw [hr] at h
      cases h
      rw [opensOf_ok hr]
      rfl

theorem isOf_eq {o : Opened} {v : Verdict} (h : isOf o v = true) :
    v.part = some o.part ∧ v.coinKey = some o.coinKey := by
  simpa [isOf] using h

/-- **`checkRun` accepts only what `OneRun` states.** -/
theorem checkRun_oneRun {n : Nat} {rows : String} {vs : List (String × Verdict)} {evs : List Event}
    (h : checkRun n rows vs evs = .ok ()) : OneRun n rows (vs.map (·.2)) evs := by
  unfold checkRun at h
  split at h
  · rename_i r sh rest
    split at h
    · cases h
    rename_i hr
    obtain ⟨_, hst, h⟩ := bind_ok h
    obtain ⟨opens, hops, h⟩ := bind_ok h
    obtain ⟨_, hb, h⟩ := bind_ok h
    obtain ⟨_, honce, h⟩ := bind_ok h
    obtain ⟨_, hop, hrec⟩ := bind_ok h
    have hr : r = rows := by simpa using hr
    subst hr
    refine ⟨sh, opens, by rw [opensOf_ok hops], ?_, ?_, ?_, ?_, ?_⟩
    · intro w hw
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
      have e := forM_ok hst p hp
      unfold checkStaged at e
      split at e
      · assumption
      · cases e
    · intro o ho
      have e := forM_ok hb o ho
      unfold checkBound at e
      split at e
      · assumption
      · cases e
    · intro j hj
      have e := forM_ok honce j (List.mem_range.2 hj)
      unfold checkOnce at e
      split at e
      · assumption
      · cases e
    · intro o ho
      have e := forM_ok hop o ho
      unfold checkOpened at e
      split at e
      · rename_i ha
        obtain ⟨p, hp, hx⟩ := List.any_eq_true.1 ha
        exact ⟨p.2, List.mem_map_of_mem hp, isOf_eq hx⟩
      · cases e
    · intro w hw
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
      have e := forM_ok hrec p hp
      unfold checkRecorded at e
      split at e
      · rename_i ha
        obtain ⟨o, ho, hx⟩ := List.any_eq_true.1 ha
        exact ⟨o, ho, isOf_eq hx⟩
      · cases e
  · cases h

/-- **`check` accepts only what `Accepts` states**: where it accepts the logs, each holds one `VERDICT`, and their
verdicts meet `Accepts`. -/
theorem check_accepts {n : Nat} {rows inner run : String} {parts : List (String × String)}
    (h : check n rows inner run parts = .ok ()) :
    ∃ vs, parts.mapM (fun p => partOf p.1 p.2) = .ok vs ∧ Accepts n rows inner (vs.map (·.2)) := by
  obtain ⟨vs, hvs, h⟩ := bind_ok h
  obtain ⟨_, hv, _⟩ := bind_ok h
  exact ⟨vs, hvs, checkVerdicts_accepts hv⟩

/-- **`check` accepts only one run** (C10 `OneRun`): where it accepts the logs and the challenger's record, the
record reads as events, and they are one run of the parts' verdicts: the staging first, at the composition's rows and
the shared set every part read, exactly one session opened for each of V*'s statements and none for another, each the
composed part's session, and the close last. A part retried, a session opened and not composed, a part composed whose
session the record doesn't hold, a staging after a Hello or a second one, and a record not closed are refused. -/
theorem check_oneRun {n : Nat} {rows inner run : String} {parts : List (String × String)}
    (h : check n rows inner run parts = .ok ()) :
    ∃ vs evs, parts.mapM (fun p => partOf p.1 p.2) = .ok vs ∧ runOf run = .ok evs ∧
      Accepts n rows inner (vs.map (·.2)) ∧ OneRun n rows (vs.map (·.2)) evs := by
  obtain ⟨vs, hvs, h⟩ := bind_ok h
  obtain ⟨_, hv, h⟩ := bind_ok h
  obtain ⟨evs, hevs, h⟩ := bind_ok h
  exact ⟨vs, evs, hvs, hevs, checkVerdicts_accepts hv, checkRun_oneRun h⟩

end RecCompose
end Flock
