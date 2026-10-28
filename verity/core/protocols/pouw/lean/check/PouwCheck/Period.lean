import CoreProtocol.Cli
import PouwCheck.Credit
import PouwCheck.Fill

/-!
# A PoUW period as a `Core.Protocol`: `verity/pouw/period/v1`

Onsite's work component (`Core.Onsite.components`). A period is the windows the auditor's ledger lists between the
period's open and its close, and it earns all or nothing: `Credit.periodCredit` credits it only when every one of its
windows is credited (`Credit.window`), and a refused window refuses the period. `pouw-period` runs it through
`Core.Protocol.main`, and `tests/test_period_difftest.py` holds it to its Python, `verity/core/protocols/pouw/period_check.py`.

**Params**, canonical JSON with exactly these keys, the verifier's own:

    {"key": challenger key, "main": [commit, …], "name": period[, "capacity": C]}

the challenger key the verifier pins (`SampledProofs.Challenger.ChallengerKey`), its own copy of main's first-parent
history (newest first) and the name of the period it credits, a string; optionally the period's monitored capacity C
in MACs, a positive integer. It carries no policy parameter: the period surfaces what its draw
established, and whoever reads it decides whether it was idle.

**Commitment**, canonical JSON: the auditor's ledger, the list of the lines the challenger signed (`ledgerEntry?`). Its
period's open commits the period's circuit, its draw K, its table rows T, its weights' root and model and its fixed
length before any coin of the period, and each window's session its roots before their coins. Its digest is
SHA-512(`commitmentTag` ‖ 0 ‖ the commitment's bytes).

**Evidence**, canonical JSON `{"pins": pins, "windows": [window, …]}`: the challenger host's published pins log, and each
window's evidence in the close's order, `Credit.PERIOD_KEYS` (its identity, session, salt, manifest, served rows, Commit
record, round log, draw, descriptor, registered entries, each drawn unit's proof outcome and its live evidence). Or
`{"fills": [filler window, …], "pins": pins, "windows": [window, …]}`, with each filler window of the period's fill
period too, in its close's order (`Fill.FILL_KEYS`).

**The check** first refuses, by name (`form: why`, `form`), every served window in a form the period's proved guarantee
(`Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod`) doesn't cover: a Pearl-C pass, a circuit that is not an H1
window (a per-tile protocol's tile, a Pearl-C leaf), a template over per-unit noise (`PouwH1Tile` v1, v3 or v4), a
J10-R unit over a tile other than `PouwH1Tile_v5` (v6), or units at more than one K (more than one stratum). Then it is
`Credit.periodCredit` on them, all or nothing: it accepts with the period's credit, or rejects with the first refusal,
`code: why`. With filler windows it is `Fill.periodFill` on those too, also all or nothing, and reported, never decided
on: no guarantee covers the fill yet, and the verifier gains no form ahead of its proof. It never leaves a period
undecided.

**Public**, on acceptance: `Credit.PeriodCredit.toJson`, each window's credited calls, its credited served work, the K
the period's open committed, the δ (`Credit.DELTA`) and each window's ε at it; with filler windows, and `fill`
(`Fill.FillCredit.toJson`: each filler window's work, the fill period's K, the δ and each filler window's ε, or
`{"refused": "code: why"}`) and `gaps` (`GAPS`), so no reader takes the fill as proved; with a capacity, and
`utilization` (`{capacity, certified, share}`: C, the certified work, the served work alone, the work `GUARANTEE`
covers, and its share of C). Nothing named certified holds the fill. The verdict and the served credit are the same
with or without filler windows, and a guarantee over a period reads the served credit (`Credited.served`).

**Profile**: the period's name and its number of windows, and its number of filler windows when its fill is credited.

What it doesn't check: each drawn unit's statement and each round's live statement, whose proof outcomes the evidence
carries as the verifier of record's verdicts (`PouwCheck.Verified` and `PouwCheck.LiveVerified` judge them, with
C-Flock's Lean verifier run outside).
-/

namespace PouwCheck.Period

open Lean (Json toJson)
open Flock (V)

def protocolId : String := "verity/pouw/period/v1"
def commitmentTag : String := "verity/pouw/period/ledger/v1"
def paramsKeys : List String := ["key", "main", "name"]
/-- The parameters of a verifier that reads the period's utilization: its monitored capacity. -/
def capacityKeys : List String := ["capacity", "key", "main", "name"]
def evidenceKeys : List String := ["pins", "windows"]
/-- The evidence of a period that presents its filler windows too (`Fill`). -/
def fillEvidenceKeys : List String := ["fills", "pins", "windows"]
/-- What a period with a `fill` doesn't establish. -/
def GAPS : List String := ["fill: no guarantee covers the fill yet, so its work is reported and not certified"]

structure Params where
  key : Json
  main : Json
  name : String
  capacity : Option Nat := none

structure Commitment where
  ledger : Json
  bytes : ByteArray

structure Evidence where
  pins : Json
  windows : Json
  fills : Option Json := none

structure Profile where
  period : String
  windows : Nat
  fills : Option Nat := none

/-- A credited period's public output: its served credit and, when it presents filler windows, its fill (credited, or
the refusal `code: why`) with `GAPS`; with a capacity, its utilization (the capacity and the certified work, the served
work alone). -/
structure Credited where
  served : Credit.PeriodCredit
  fill : Option (Except String Fill.FillCredit) := none
  capacity : Option Nat := none

def Credited.toJson (c : Credited) : Json :=
  let j := match c.fill with
    | none => c.served.toJson
    | some f =>
      let fj := match f with
        | .ok f => f.toJson
        | .error why => Json.mkObj [("refused", .str why)]
      (c.served.toJson.setObjVal! "fill" fj).setObjVal! "gaps" (Lean.toJson GAPS)
  match c.capacity with
  | none => j
  | some cap =>
    let certified : Nat := c.served.work.foldl (· + ·) 0
    j.setObjVal! "utilization" (Json.mkObj [("capacity", Lean.toJson cap),
      ("certified", Lean.toJson certified), ("share", .str (toString (mkRat certified cap)))])

def paramsOf (j : Json) : V Params := do
  unless keys j == some paramsKeys || keys j == some capacityKeys do
    throw s!"keys are not {", ".intercalate paramsKeys}, with capacity optional"
  let some name := str? (get j "name") | throw "name is not a string"
  let capacity ← if keys j == some paramsKeys then pure none else
    match nat? (get j "capacity") with
    | some c => if 1 ≤ c then pure (some c) else throw capacityForm
    | none => throw capacityForm
  pure { key := get j "key", main := get j "main", name, capacity }
where
  capacityForm := "capacity is not a positive integer, the period's monitored capacity in MACs"

def commitmentOf (b : ByteArray) : V Commitment := do
  let ledger ← Core.canonicalJson b
  let .arr _ := ledger | throw "the auditor's ledger is not a list of its lines"
  pure { ledger, bytes := b }

def Commitment.digest (c : Commitment) : ByteArray :=
  Flock.Sha512.hash (commitmentTag.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys j == some evidenceKeys || keys j == some fillEvidenceKeys do
    throw s!"keys are not {", ".intercalate evidenceKeys}"
  pure { pins := get j "pins", windows := get j "windows",
         fills := if keys j == some fillEvidenceKeys then some (get j "fills") else none }

/-- The guarantee a credited period is proved under. -/
def GUARANTEE : String := "Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod"
/-- A Pearl-C pass's evidence (`credit.PEARL_C_KEYS`). -/
def PASS_KEYS : List String := ["context", "descriptor", "manifest", "session"]
def TILE_ID : String := "PouwH1Tile_v"
/-- The one tile the guarantee covers: a drawn per-call unit over it is proved as its 32 lanes and their combine
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
  s!"strata, and {GUARANTEE} draws from one tile stratum (TileDrawIsCoin): serve a window's linears at one K"

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

/-- The first served window in a form the period's guarantee doesn't cover (`form`), else the period's credit
(`Credit.periodCredit`) and, with filler windows, its fill (`Fill.periodFill`, reported and read by no decision), or the
first refusal; with a capacity, its utilization. -/
def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Credited Profile :=
  match form p.name e.windows with
  | some why => .reject s!"form: {why}"
  | none =>
    match Credit.periodCredit p.key c.ledger e.pins p.main (.str p.name) e.windows with
    | .error (code, why) => .reject s!"{code}: {why}"
    | .ok pc =>
      let fill := e.fills.map fun fills =>
        (Fill.periodFill p.key c.ledger e.pins p.main p.name fills).mapError fun (code, why) => s!"{code}: {why}"
      let credited := match fill with
        | some (.ok f) => some f
        | _ => none
      let profile : Profile :=
        { period := p.name, windows := pc.windows.length, fills := credited.map (·.work.length) }
      .accept { served := pc, fill, capacity := p.capacity } profile

def profileJson (pr : Profile) : Json :=
  Json.mkObj ([("period", .str pr.period), ("windows", toJson pr.windows)] ++
    (pr.fills.map fun n => [("fills", toJson n)]).getD [])

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Credited
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest := Commitment.digest
  check := check
  publicJson := Credited.toJson
  profileJson := profileJson

-- every input gets a verdict: parameters, a commitment or evidence of another shape is refused, naming the input
#guard (protocol.answer "{}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason ==
  some "params: keys are not key, main, name, with capacity optional"
#guard (protocol.answer "{\"capacity\":0,\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "[]".toUTF8
  "{}".toUTF8).reason == some "params: capacity is not a positive integer, the period's monitored capacity in MACs"
#guard (protocol.answer "{\"capacity\":1,\"floor\":\"1/2\",\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8
  "[]".toUTF8 "{}".toUTF8).reason == some "params: keys are not key, main, name, with capacity optional"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":1}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason ==
  some "params: name is not a string"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "{}".toUTF8 "{}".toUTF8).reason ==
  some "commitment: the auditor's ledger is not a list of its lines"
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "[]".toUTF8 "{}".toUTF8).reason ==
  some "evidence: keys are not pins, windows"
-- well-formed inputs reach `periodCredit`, whose refusal is the period's
#guard (protocol.answer "{\"key\":null,\"main\":[],\"name\":\"p\"}".toUTF8 "[]".toUTF8
  "{\"pins\":[],\"windows\":7}".toUTF8).reason == some s!"period: {Credit.WINDOWS_FORM}"

end PouwCheck.Period
