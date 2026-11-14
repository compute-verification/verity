module

public import CoreProtocol.Account
public import CoreProtocol.Cli
public import PouwCheck.Credit
meta import CoreProtocol.Account
meta import CoreProtocol.Cli
meta import PouwCheck.Credit

/-!
# A PoUW period as a `Core.Protocol`: `verity/pouw/period/v1`

Onsite's work component (`Core.Onsite.components`). A period is the windows the auditor's ledger lists between the
period's open and its close, and it earns all or nothing: `Credit.periodCredit` credits it only when every one of its
windows is credited (`Credit.window`), and a refused window refuses the period. `pouw-period` runs it through
`Core.Protocol.main`, and `tests/test_period_difftest.py` holds it to its Python, `verity/core/protocols/pouw/period_check.py`.

**Params**, canonical JSON with exactly these keys, the verifier's own:

    {"key": challenger key, "main": [commit, …], "name": period, "sessions": [session, …]}

the challenger key the verifier pins (`Spot.Challenger.ChallengerKey`), its own copy of main's first-parent history
(newest first), the name of the period it credits (a string) and the ordered ids of the sessions the challenger's
ledger opened for the period's windows (each window's and its sample's, `sessionId?`; the empty list names none).
Parameters without `sessions`, `{"key", "main", "name"}` (`aloneKeys`), run the period with no session list to check
its close against; any other key set is refused (`KEYS_FORM`). They name no device: the accounting layer binds the
challenger key to a registered device.

**Commitment**, canonical JSON: the auditor's ledger, the list of the lines the challenger signed (`ledgerEntry?`). Its
period's open commits the period's circuit, its sample K, its table rows T, its weights' root and model and its fixed
length before any coin of the period, and each window's session its roots before their coins. Its digest is
SHA-512(`commitmentTag` ‖ 0 ‖ the commitment's bytes).

**Evidence**, canonical JSON `{"pins": pins, "windows": [window, …]}`: the challenger host's published pins log, and each
window's evidence in the close's order, `Credit.PERIOD_KEYS` (its identity, session, salt, manifest, served rows, Commit
record, round log, sample, descriptor, registered entries, each sampled proof block's proof outcome and its live evidence).

**The check** first refuses, by name (`form: why`, `form`), every window in a form the period's proved guarantee
(`Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod`) doesn't cover: a Pearl-C pass, a circuit that is not an H1
window (a per-tile protocol's tile, a Pearl-C leaf), a template over per-unit noise (`PouwH1Tile` v1, v3 or v4), a
J10-R unit over a tile other than `PouwH1Tile_v5` (v6), or units at more than one K (more than one stratum). Then it is
`Credit.periodCredit` on them: it accepts with the period's credit, or rejects with the first refusal, `code: why`.
Last it reads the challenge phases (sessions) the period's close lists (`ledgerChallengePhases`) and refuses
(`sessions: why`) a ledger it can't read them from, and, where the parameters list the sessions, a session they
don't list or that repeats one before it (`challengePhasesRefusal`). It never leaves a period undecided.

**Public**, on acceptance (`Published.toJson`), what `account` reads: `Credit.PeriodCredit.toJson`, each window's
credited calls, its credited served work, the K the period's open committed, the δ (`Credit.DELTA`) and each window's
ε at it; beside it the parameters' challenger `key`, and from the ledger the close's open `roots` in its order and the
period's span: the period-open's `start` and its `length` to its close inclusive, in ledger entries, as the ledger
carries no clock reading on either. It names no device, registration head or place: the accounting layer binds the
challenger `key` to a registered device.

**Profile**: the period's name and its number of windows.

What it doesn't check: each sampled proof block's statement and each round's live statement, whose proof outcomes the evidence
carries as the verifier's verdicts (`PouwCheck.Verified` and `PouwCheck.LiveVerified` judge them, with
Flock's Lean verifier run outside).
-/

@[expose] public section

namespace PouwCheck.Period

open Lean (Json toJson)
open Flock (V)

def protocolId : String := "verity/pouw/period/v1"
def commitmentTag : String := "verity/pouw/period/ledger/v1"
def paramsKeys : List String := ["key", "main", "name", "sessions"]
/-- The keys of a period's parameters with no session list: `paramsKeys` without `sessions`. -/
def aloneKeys : List String := ["key", "main", "name"]
def evidenceKeys : List String := ["pins", "windows"]

/-- A period's parameters: the challenger key, main's history, the period's name and, where they list them, the
challenge phases (sessions, under the key `sessions`) the challenger's ledger opened for the period's windows. -/
structure Params where
  key : Json
  main : Json
  name : String
  challengePhases : Option (List String)

structure Commitment where
  ledger : Json
  bytes : ByteArray

structure Evidence where
  pins : Json
  windows : Json

structure Profile where
  period : String
  windows : Nat

def CHALLENGE_PHASES_FORM : String :=
  "sessions is not a list of the ids of the sessions the challenger's ledger opened for the period (1 to 64 of " ++
  "[A-Za-z0-9._:-] each, the first alphanumeric)"

def KEYS_FORM : String :=
  s!"keys are not {", ".intercalate paramsKeys}, or {", ".intercalate aloneKeys} alone"

def challengePhasesOf (j : Json) : V (List String) := do
  let some sessions := (match get j "sessions" with
      | .arr a => a.toList.mapM fun s => (str? s).filter sessionId?
      | _ => none) | throw CHALLENGE_PHASES_FORM
  pure sessions

def paramsOf (j : Json) : V Params := do
  let ks := keys j
  unless ks == some paramsKeys || ks == some aloneKeys do throw KEYS_FORM
  let some name := str? (get j "name") | throw "name is not a string"
  let challengePhases ← if ks == some paramsKeys then some <$> challengePhasesOf j else pure none
  pure { key := get j "key", main := get j "main", name, challengePhases }

def commitmentOf (b : ByteArray) : V Commitment := do
  let ledger ← Core.canonicalJson b
  let .arr _ := ledger | throw "the auditor's ledger is not a list of its lines"
  pure { ledger, bytes := b }

def Commitment.digest (c : Commitment) : ByteArray :=
  Flock.Sha512.hash (commitmentTag.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys j == some evidenceKeys do throw s!"keys are not {", ".intercalate evidenceKeys}"
  pure { pins := get j "pins", windows := get j "windows" }

/-- The guarantee a credited period is proved under. -/
def GUARANTEE : String := "Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod"
/-- A Pearl-C pass's evidence (`credit.PEARL_C_KEYS`). -/
def PASS_KEYS : List String := ["context", "descriptor", "manifest", "session"]
def TILE_ID : String := "PouwH1Tile_v"
/-- The one tile the guarantee covers: a sampled per-call unit over it is proved as its 32 lanes and their combine
(`h1zy.PouwH1LaneZy_v2`, `PouwH1Combine_v2`). -/
def COVERED_TILE : String := "PouwH1Tile_v5{"

def count (s pat : String) : Nat := (s.splitOn pat).length - 1

def passForm (nm : String) (i : Nat) : String :=
  s!"window {i} of period {nm} is a Pearl-C pass (\{context, descriptor, manifest, session}), which this protocol " ++
  s!"never credits: a Pearl-C pass earns only in a Pearl-C period, and {GUARANTEE} covers H1 windows at J10-R alone"

def notH1 (nm : String) (i : Nat) (fn : String) : String :=
  let name := (fn.splitOn "{").headD ""
  s!"window {i} of period {nm} proves {name}, not an H1 window ({Credit.WINDOW_FN}): a " ++
  "per-tile protocol's tile (PouwH1Tile_v1 at K = 64, 4 × 8, or a Pearl-C leaf) or any other circuit is outside " ++
  s!"{GUARANTEE}, which covers H1 windows at J10-R alone"

def perUnit (nm : String) (i k : Nat) : String :=
  s!"window {i} of period {nm} is an H1 window whose template at K = {k} takes per-unit noise (its parameters end " ++
  "in roots, not J10-R's call and blocks: PouwH1Tile_v1, v3 or v4), which " ++
  s!"{GUARANTEE} doesn't cover: serve it over J10-R's PouwH1Tile_v5"

def notV5 (nm : String) (i k : Nat) : String :=
  s!"window {i} of period {nm} is a J10-R window whose unit at K = {k} is not over PouwH1Tile_v5 alone (v6's " ++
  s!"one-stream z, or no named tile): {GUARANTEE} covers the per-call unit over PouwH1Tile_v5, proved as its 32 " ++
  "lanes and their combine"

def manyStrata (nm : String) (i : Nat) (ks : List Nat) : String :=
  s!"window {i} of period {nm} has units at {ks.length} K's ({Credit.listStr ks}), so its draw is over that many " ++
  s!"strata, and {GUARANTEE} draws from one tile stratum (TileSampleIsCoin): serve a window's linears at one K"

/-- Why window `i` of period `nm` is a form `GUARANTEE` doesn't cover, read from its evidence alone: a Pearl-C pass,
a circuit that is not an H1 window, a template over per-unit noise, a J10-R unit over a tile other than
`COVERED_TILE`, or units at more than one K. `none` for a covered window, and for evidence too malformed to read,
which `Credit.periodCredit` refuses. -/
def windowForm (nm : String) (i : Nat) (w : Json) : Option String := Id.run do
  if keys w == some PASS_KEYS then return some (passForm nm i)
  let some d := (match get w "descriptor" with
      | .str s => if Credit.isHexBytes s then hex? (.str s) else none
      | _ => none) | return none
  let some j := (String.fromUTF8? d).bind Credit.loads? | return none
  let .str fn := get j "program_fn" | return none
  unless fn.startsWith (Credit.WINDOW_FN ++ "{") do return some (notH1 nm i fn)
  let .ok P := Credit.programOf d | return none
  let some tiles := ((Flock.Program.recordPairs fn).find? (·.1 == "TILES")).bind (Credit.loads? ·.2) | return none
  let .arr ts := tiles | return none
  let mut ks : List Nat := []
  for t in ts do
    let .arr p := t | return none
    let [kj, f] := p.toList | return none
    let (some k, some tid) := (nat? kj, str? (get f "fn")) | return none
    let some dt := Credit.defn? P tid | return none
    let some (_, _, j10r) := Credit.tileOf dt k | return none
    unless j10r do return some (perUnit nm i k)
    let u := ((Flock.Program.recordPairs dt.key).toList.lookup "UNIT").getD ""
    unless 0 < count u TILE_ID && count u TILE_ID == count u COVERED_TILE do return some (notV5 nm i k)
    ks := ks ++ [k]
  if 1 < ks.length then return some (manyStrata nm i ks)
  return none

/-- The first window of `windows` in a form `GUARANTEE` doesn't cover (`windowForm`), else `none`. -/
def form (nm : String) (windows : Json) : Option String := Id.run do
  let .arr ws := windows | return none
  for (w, i) in ws.toList.zipIdx do
    if let some why := windowForm nm i w then return some why
  return none

/-- The ledger's entries, each line read (`Spot.Challenger.ledgerEntry?`); `none` when a line isn't one. -/
def entries? (ledger : Json) : Option (List Spot.Challenger.LedgerEntry) := do
  let .arr lines := ledger | none
  lines.toList.mapM Spot.Challenger.ledgerEntry?

def periodEntry? (es : List Spot.Challenger.LedgerEntry) (kind name : String) : Option Spot.Challenger.LedgerEntry :=
  es.find? fun e => e.kind == kind && get e.body "period" == Json.str name

/-- **The sessions period `name`'s close lists** in the ledger `ledger`, in its order (each window's session, then its
sample's): the empty list when the ledger closes no period `name`, and `none` when a line of the ledger isn't an entry
or a window of the close names no session, ledgers the check refuses. -/
def ledgerChallengePhases (ledger : Json) (name : String) : Option (List String) := do
  let es ← entries? ledger
  match periodEntry? es "period-close" name with
  | none => some []
  | some c => match get c.body "windows" with
    | .arr ws => ws.toList.mapM fun w => str? (get w "session")
    | _ => none

/-- Why the sessions `got` the period's close lists aren't sessions the parameters list (`listed`), each once: the
first that isn't listed or repeats one before it; `none` when they all are, once each. Linear in the sessions: both
sets are hash sets. -/
def challengePhasesRefusal (name : String) (listed got : List String) : Option String := Id.run do
  let listedSet : Std.HashMap String Unit := listed.foldl (fun m s => m.insert s ()) {}
  let mut seen : Std.HashMap String Unit := {}
  for (s, i) in got.zipIdx do
    unless listedSet.contains s do
      return some (s!"session {i} of period {name}'s close ({s}) is not one the verifier's parameters list for the " ++
        "period: only a session the challenger's ledger opened for one of its windows counts")
    if seen.contains s then
      return some s!"session {i} of period {name}'s close ({s}) repeats a session before it: each session counts once"
    seen := seen.insert s ()
  return none

/-- One value for a list of sessions, its canonical JSON: `[]` for none. -/
def challengePhasesValue (ss : List String) : ByteArray := (cs (Json.arr (ss.map Json.str).toArray)).toUTF8

/-- The challenge phases (sessions) a period's case names, as one value (`challengePhasesValue`): those its ledger's
close lists (`ledgerChallengePhases`), read with the check's own decoders from its parameters `params` and its
commitment `commitment`; `none` only for a case the check rejects. -/
def challengePhasesNamed (params : String) (commitment : ByteArray) : Option ByteArray := do
  let p ← (Core.canonicalJson params.toUTF8 >>= paramsOf).toOption
  let c ← (commitmentOf commitment).toOption
  challengePhasesValue <$> ledgerChallengePhases c.ledger p.name

/-- Period `name`'s span in the ledger: its open's and its close's seq, and the open roots its close lists in order. -/
structure Span where
  start : Nat
  stop : Nat
  roots : List String

def spanOf (ledger : Json) (name : String) : Option Span := do
  let es ← entries? ledger
  let o ← periodEntry? es "period-open" name
  let c ← periodEntry? es "period-close" name
  let .arr ws := get c.body "windows" | none
  let roots ← ws.toList.mapM fun w => str? (get w "root")
  pure ⟨o.seq, c.seq, roots⟩

/-- What an accepted period publishes: its credit, its parameters' challenger key, and from the ledger its close's open
roots and its span (the period-open's seq and the entries from it to its close). It names no device: the accounting
layer binds the challenger key to a registered device. -/
structure Published where
  credit : Credit.PeriodCredit
  key : Json
  roots : List String
  start : Nat
  length : Nat

def Published.toJson (p : Published) : Json :=
  (Credit.PeriodCredit.toJson p.credit).mergeObj <| Json.mkObj [("key", p.key),
    ("roots", Json.arr (p.roots.map Json.str).toArray), ("start", Lean.toJson p.start), ("length", Lean.toJson p.length)]

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Published Profile :=
  match form p.name e.windows with
  | some why => .reject s!"form: {why}"
  | none =>
    match Credit.periodCredit p.key c.ledger e.pins p.main (.str p.name) e.windows with
    | .error (code, why) => .reject s!"{code}: {why}"
    | .ok pc =>
      match ledgerChallengePhases c.ledger p.name, spanOf c.ledger p.name with
      | some got, some s =>
        match p.challengePhases.bind fun listed => challengePhasesRefusal p.name listed got with
        | some why => .reject s!"sessions: {why}"
        | none => .accept ⟨pc, p.key, s.roots, s.start, s.stop + 1 - s.start⟩
            { period := p.name, windows := pc.windows.length }
      | _, _ => .reject s!"sessions: period {p.name}'s close in the auditor's ledger lists no session ids to read"

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("period", .str pr.period), ("windows", toJson pr.windows)]

/-- **What the period accounts for** (`Core.Account`): the served work its windows' credited calls did, in MACs of
compute, from below, each window's ε at δ (once for the period), over the ledger entries of the period's span in the
challenger key's ledger (`series`), which carry no clock reading, so the span orders periods and covers no tick. It
names no registration head and no place: the accounting layer binds the challenger key to a registered device, so
until it does no registered account counts the answer. -/
def account : Core.Account :=
  { protocol := protocolId, head := none
    amounts := [{ metric := "work", part := .compute, key := .published "work",
                  unit := Core.Account.macs, side := .verified,
                  time := .steps (.run (.published "start") (.published "length")) ⟨0, 1⟩,
                  series := some (.published "key"), share := some (.published "eps"), error := .published "delta",
                  pool := .once, uncertified := .safe }] }

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Published
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest := Commitment.digest
  check := check
  publicJson := Published.toJson
  profileJson := profileJson

private def paramsWith (sessions name : String) : ByteArray :=
  s!"\{\"key\":null,\"main\":[],\"name\":{name},\"sessions\":{sessions}}".toUTF8
private def wellFormed : ByteArray := paramsWith "[\"s0\",\"s1\"]" "\"p\""
private def reason (ps : ByteArray) : Option String := (protocol.answer ps "[]".toUTF8 "{}".toUTF8).reason

-- every input gets a verdict: parameters, a commitment or evidence of another shape is refused, naming the input
#guard reason "{}".toUTF8 == some s!"params: {KEYS_FORM}"
#guard KEYS_FORM == "keys are not key, main, name, sessions, or key, main, name alone"
-- the parameters name no device, and without `sessions` the period runs with no session list
#guard reason ("{\"head\":\"" ++ "".pushn 'a' 128 ++
  "\",\"key\":null,\"main\":[],\"name\":\"p\",\"place\":\"gpu0\",\"sessions\":[]}").toUTF8 ==
  some s!"params: {KEYS_FORM}"
#guard reason "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 == some "evidence: keys are not pins, windows"
#guard reason (paramsWith "[]" "1") == some "params: name is not a string"
#guard reason (paramsWith "\"s0\"" "\"p\"") == some s!"params: {CHALLENGE_PHASES_FORM}"
#guard reason (paramsWith "[\"-s\"]" "\"p\"") == some s!"params: {CHALLENGE_PHASES_FORM}"
#guard (protocol.answer wellFormed "{}".toUTF8 "{}".toUTF8).reason ==
  some "commitment: the auditor's ledger is not a list of its lines"
#guard reason wellFormed == some "evidence: keys are not pins, windows"
-- well-formed inputs reach `periodCredit`, whose refusal is the period's
#guard (protocol.answer wellFormed "[]".toUTF8 "{\"pins\":[],\"windows\":7}".toUTF8).reason ==
  some s!"period: {Credit.WINDOWS_FORM}"

-- a session the parameters don't list, or one that repeats, is refused; the sessions named are one value, `[]` for a
-- ledger that closes no such period, and none for a case the check refuses
#guard challengePhasesRefusal "p" ["s0", "s1"] ["s0", "s1"] == none
#guard challengePhasesRefusal "p" ["s0", "s1"] ["s1"] == none
#guard (challengePhasesRefusal "p" ["s0"] ["s0", "s1"]).map
  (·.startsWith "session 1 of period p's close (s1) is not") == some true
#guard (challengePhasesRefusal "p" ["s0"] ["s0", "s0"]).map
  (·.startsWith "session 1 of period p's close (s0) repeats") == some true
#guard challengePhasesRefusal "p" ["s0", "s1"] ["s1", "s0", "s1"] ==
  some "session 2 of period p's close (s1) repeats a session before it: each session counts once"
#guard challengePhasesNamed (String.fromUTF8! wellFormed) "[]".toUTF8 == some "[]".toUTF8
#guard challengePhasesValue ["s0", "s1"] == "[\"s0\",\"s1\"]".toUTF8
#guard challengePhasesNamed (String.fromUTF8! wellFormed) "[\"x\"]".toUTF8 == none
#guard challengePhasesNamed (String.fromUTF8! wellFormed) "{}".toUTF8 == none
#guard challengePhasesNamed "{}" "[]".toUTF8 == none

-- the account reads what an accepted period publishes: its work with its ε, its challenger key, and its span's entries;
-- it names no device
private def credit : Credit.PeriodCredit := ⟨[[0], [1, 2]], [10, 20], 4, Credit.DELTA, [some (1 / 2), none]⟩
private def published : Json := Published.toJson ⟨credit, .str "k", ["r0", "r1"], 3, 5⟩
private def accepted : Core.Answer :=
  { protocol := protocolId, verdict := .accept, reason := none, digest := none, published := some published,
    profile := none }
#guard ["registration_head", "place", "part"].all fun k => (published.getObjVal? k).toOption.isNone
#guard account.Formed
#guard account.head.isNone && account.amounts.all (·.place.isNone)
#guard (account.amounts.map fun m => (m.key.read accepted, m.series.bind (·.read accepted))) ==
  [(some (Lean.toJson [10, 20]), some (.str "k"))]
#guard (account.amounts.map fun m => match m.time with
  | .steps s _ => s.read accepted
  | _ => none) == [some (3, 8)]

end PouwCheck.Period
