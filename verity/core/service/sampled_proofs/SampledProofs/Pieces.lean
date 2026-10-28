/-
The one-stage audit's pieces as JSON functions, which `sampled-proofs-difftest` serves (`Difftest.serve`) for the
difftest against `one_stage`'s Python, beside the whole protocol (`sampled-proofs-one-stage serve`):

* `decode`: a case's three decoders run independently (`answer` stops at the first refusal);
* `partition`: the partition's digest, population, strata and expanded law from the verifier's parameters;
* `committed_bits`: per Definition `Q_call` cuts, its units' committed bits read off its layout and by the walk that
  layout replaced (`Partition.committedBits`, `Partition.committedBitsRef`);
* `registration`: R1–R7's failure codes for a record against the parameters;
* `registered`: the verifier's own entries for the values its parameters expect registered;
* `domain`: a registered value's commitment domain, or a produced value's (`Produced.domain`);
* `produced`: a produced values' record's codes against the verifier's own entries, and the roots the verifiers of record
  opened them against, and the rows they opened against its openings and the verifier's own ports' (`Produced.check`,
  `Produced.opened`, `Produced.shared`);
* `known_port`: the root and opening of a port the verifier registers at known content (`Produced.known`);
* `check_draw`, `check_proved`: the draw's checks, each `{"ok", "codes", "detail"}`;
* `draw`: the verifier's draw from given bytes of its own randomness;
* `select`: the challenger's draw from a Spec's selection rule on, over a program and query, from given bytes of its
  coins, with the number of bytes it read (`Select.select`);
* `slots`: the claims' partition over the program and query, and the slots of that draw's checked proof blocks
  (`Slots.select`);
* `bound`: the exact bounds on wrong units at δ;
* `harm_bound`: the harm they hold at δ in closed form (`Bound.densityBound`);
* `max_harm`: a certificate's bracket around `Flock.Guarantees.JointStratified`'s exact supremum of that harm
  (`Bound.maxHarmCheck`), which `certify` is held to;
* `hidden_offsets`, `hidden_locate`, `hidden_witness`: the draw over hidden positions (`HiddenDraw`): the honest table's
  offsets and total, the honest prover's witnesses, and the witness for a named call;
* `verdict`: the verdicts' share of the outcome once the record's checks hold;
* `matches_public`: the verifier's own public file against the record (`Record.matchesPublic`);
* `session_registered`: the registered roots the verifiers of record report (`Record.sessionRegistered`), and with a
  registration only its own values' (`Record.ownRegistered`);
* `record`: the whole audit record (`Record.record`), as the text of Python's `canonical_json`;
* `check_public`: the proof service's public-items rule on a registration record (`Service.checkPublic`);
* `check_verdicts`: the verdicts' share of a proof service outcome (`Service.checkVerdicts`);
* `run`: the stream's run of one registration from its public check through its verdict (`Run.run`);
* `stream`: the proof service stream's own rules, one op's decision on what they hold of the stream (`Stream.run`);
* `fleet`: the fleet's rules over its streams' operators, one op's decision on what they hold of the fleet (`Fleet.run`);
* `hello`: the run's Hello on its record, its coin tree at run scope (`Hello.check`).
* `challenger_commitment`, `challenger_opening`, `challenger_reason`, `challenger_order`: the challenger's live coins
  (`Challenger`): a coin commitment and an opening as `from_json` reads them, `live_reason` and `order_reason`.
* `challenger_key`, `challenger_record`, `challenger_trusted`: the challenger's signed record (`ChallengerRecord`): the
  verifier's key and a record as `from_json` reads them, with `record_message`, and `trusted_reason`.
* `challenger_close`, `challenger_released`, `challenger_ledger`, `challenger_head`, `challenger_session`,
  `challenger_credited`, `challenger_pinned`: the challenger's session (`ChallengerSession`): its close as `from_json`
  reads it, with `close_message`; `log_chain` and `released_roots_of`; `audited_session`, the auditor's ledger;
  `ledger_head`, the ledger's published heads; `trusted_session_of` and `credited_session`, a window counted toward its
  period's circuit, each over `pinned_session`: a session from a commit the host's pins log records and the auditor's
  first-parent list of main holds.
* `challenger_read`, `challenger_draw_coins`: a draw's coins from one challenger family (`ChallengerCoins`):
  `read_reason`, the bytes a draw read against its segment, and `coins_reason`, against the segment and the session the
  auditor's ledger opens for the verifier's window (`trusted_session_of`).
* `wires`, `wires_unrolled`: a window circuit's wires checked at the size of their description (`Wires.check`) and by
  the unrolled reference (`Wires.checkRef`), each with its summary: wired rows and bits, spans, and drawn rows' writers.
* `wires_drawn`, `wires_drawn_unrolled`: the wire check on the drawn units alone (`Wires.Summary.drawnOk`, each drawn
  wired row's public string its writer's) and on the reference's entries whose unit is drawn.

`sampled-proofs-verifier` serves those the verifier of record runs (`verifier`): `registration`, `matches_public`,
`select` (the draw of record: `Stream.select` hands it the bytes of the challenger's coins), `check_draw`,
`check_proved`, `record`, `check_public`, `check_verdicts`, `run` (`Stream.outcome` holds its outcome to it), `hello`,
`stream` (each of `service.Stream`'s ops) and `fleet` (each of `service.Fleet`'s).
-/
import SampledProofs.OneStage
import SampledProofs.Slots
import SampledProofs.Service
import SampledProofs.Run
import SampledProofs.Stream
import SampledProofs.Fleet
import SampledProofs.Hello
import SampledProofs.ChallengerRecord
import SampledProofs.ChallengerSession
import SampledProofs.ChallengerCoins
import SampledProofs.Produced
import SampledProofs.HiddenDraw
import SampledProofs.PartitionRef
import SampledProofs.WiresDrawn

namespace SampledProofs.Pieces

open Lean (Json)
open Flock (V)

private def params (j : Json) : V OneStage.Params := do
  OneStage.paramsOfJson (← Core.canonicalJson (← Flock.canon j).toUTF8)

private def orNull {α} (f : α → Json) : V α → Json
  | .ok a => f a
  | .error _ => Json.null

private def codes (cs : Array String) : Json := Json.mkObj [("codes", Json.arr (cs.map Json.str))]

private def ints (j : Json) (what : String) : V (Array Int) := do
  let .arr xs := j | throw s!"{what} is not a list of integers"
  xs.mapM fun x => match int? x with
    | some n => pure n
    | none => throw s!"{what} is not a list of integers"

/-- `{"commitment": its digest or null, "evidence": decodes, "params": decodes}`; a malformed case is refused. -/
def decode (j : Json) : V Json := do
  let p ← Flock.J.get j "params" "the case"
  let c ← Flock.unhex (← Flock.J.str j "commitment" "the case")
  let e ← Flock.unhex (← Flock.J.str j "evidence" "the case")
  return Json.mkObj [("params", Json.bool (params p).toBool),
    ("commitment", orNull (fun c => Json.str (Flock.hexOf c.digest)) (OneStage.commitmentOf c)),
    ("evidence", Json.bool (OneStage.evidenceOf e).toBool)]

/-- `{"params"}` to `{"digest", "population", "strata", "law"}`, each null where Python raises. -/
def partition (j : Json) : V Json := do
  let v := (← params (getD j "params")).verifier
  let part := v.partition.getD (Partition.build (Flock.hexOf v.program.sha512) v.query)
  let range := fun ((lo, hi) : Nat × Nat) => Json.arr #[Lean.toJson lo, Lean.toJson hi]
  return Json.mkObj [("digest", (Partition.digest part).elim Json.null Json.str),
    ("population", orNull Lean.toJson (Partition.units v.program v.query)),
    ("strata", orNull (fun ss => Json.arr (ss.map fun (t, rs) => Json.arr #[Json.str t, Json.arr (rs.map range)]))
      (Partition.strata v.program v.query)),
    ("law", orNull id (Draw.derive v.program v.query v.law v.workTable))]

/-- `{"program", "X"}`, a program object and a natural number, to `[[definition, bits, bits by walk], …]`: per distinct
Definition `Q_call` v1 at X cuts (`Flock.Qcall.evaluate`), its units' committed bits read off its layout
(`Partition.committedBits`) and by the walk of its own that replaced (`Partition.committedBitsRef`); refused where
`Q_call` is. -/
def committedBits (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let some X := nat? (getD j "X") | throw "X is not a natural number"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  let mut seen : Array Nat := #[]
  let mut out : Array Json := #[]
  for c in ← Flock.Qcall.evaluate p.prog X do
    if seen.contains c.d then continue
    seen := seen.push c.d
    let units := c.c.units.map fun u => (u.lo, u.hi)
    out := out.push (Json.arr #[Json.str (p.prog.defs[c.d]!).key, Lean.toJson (Partition.committedBits c.c.gates units),
      Lean.toJson (← Partition.committedBitsRef p.prog c.d units)])
  return Json.arr out

/-- `{"params", "record"}` to R1–R7's codes. -/
def registration (j : Json) : V Json := do
  return codes (Registration.check (getD j "record") (← params (getD j "params")).verifier)

/-- `{"params"}` to `registered_entries` under the verifier's window root and fixed values (`Registration.ownEntries`):
`{value: {"leaves", "words", "word_bits", "domain"}}`, null where it raises. -/
def registered (j : Json) : V Json := do
  let v := (← params (getD j "params")).verifier
  return orNull (fun es => Json.mkObj (es.toList.map fun e => (e.value, e.toJson))) (Registration.ownEntries v)

/-- `{"program", "value", "words", "leaves", "word_bits"}`, two strings and three natural numbers, and an optional
`"window"`, to `registered.domain(...).domain_id` in hex; refused where it raises (a width but 16 or 32, a window that is
not lowercase hex of 64 bytes). -/
def domain (j : Json) : V Json := do
  let (.str program, .str value) := (getD j "program", getD j "value") | throw "program or value is not a string"
  let (some words, some leaves, some b) := (nat? (getD j "words"), nat? (getD j "leaves"), nat? (getD j "word_bits"))
    | throw "words, leaves or word_bits is not a natural number"
  let hex? := fun (k : String) => match field? j k with
    | none => pure none
    | some (.str r) => if Produced.hex64 r then pure (some r) else throw s!"the {k} is not 64 bytes of hex"
    | some _ => throw s!"the {k} is not 64 bytes of hex"
  let window ← hex? "window"
  match ← hex? "produced" with
  | none => return Json.str (Flock.hexOf (← Registered.domainAt program value words leaves b window))
  | some d =>
    if window.isSome then throw "a produced value's domain binds its registration, which binds the window: not both"
    return Json.str (Flock.hexOf (← Produced.domain program value words leaves b d))

/-- `known` (`{value: [[position, b ‖ c] ...]}`, natural positions distinct within a value and string openings), null as
none. -/
def knownOpenings (j : Json) : V (Array (String × Array (Nat × String))) := do
  match j with
  | .null => pure #[]
  | .obj _ => (entries j).mapM fun (v, rows) => do
      let .arr rs := rows | throw s!"known: {v} is not a list of openings"
      let ps ← rs.mapM fun r => match r with
        | .arr #[p, .str bc] => match nat? p with
          | some n => pure (n, bc)
          | none => throw s!"known: {v}'s position is not a natural number"
        | _ => throw s!"known: {v}'s opening is not [position, b ‖ c]"
      if (ps.toList.map (·.1)).eraseDups.length != ps.size then throw s!"known: {v} opens a position twice"
      pure (v, ps)
  | _ => throw "known is not an object"

/-- `{"record", "registration", "draw", "mine", "opened", "known", "verdicts"}` to `{"codes": produced.check's, "opened":
produced.opened's codes, "shared": produced.shared's over produced.wanted's, both null unless the record holds}`: the
registration and draw strings, `mine` the verifier's own entries (`{value: {"leaves", "words", "word_bits", "domain"}}`,
three natural numbers and a string each), `opened` null or `{value: root}`, `known` null or the verifier's own ports'
openings (`knownOpenings`) and `verdicts` null or an object; refused otherwise. -/
def produced (j : Json) : V Json := do
  let (.str registration, .str draw) := (getD j "registration", getD j "draw")
    | throw "registration or draw is not a string"
  let .obj _ := getD j "mine" | throw "mine is not an object"
  let mine ← (entries (getD j "mine")).mapM fun (value, e) =>
    match nat? (getD e "leaves"), nat? (getD e "words"), nat? (getD e "word_bits"), getD e "domain" with
    | some leaves, some words, some wordBits, .str domain => pure ({ value, leaves, words, wordBits, domain } : Produced.Mine)
    | _, _, _, _ => throw s!"mine: {value} is not leaves, words and word_bits and a string domain"
  let opened ← match getD j "opened" with
    | .null => pure none
    | o@(.obj _) => pure (some (entries o))
    | _ => throw "opened is not an object"
  let known ← knownOpenings (getD j "known")
  let verdicts := getD j "verdicts"
  if !(verdicts matches .null | .obj _) then throw "verdicts is not an object"
  let rec_ := getD j "record"
  let cs := Produced.check rec_ registration draw (mine.qsort (·.value < ·.value))
  let codes (xs : Array String) := if cs.isEmpty then Json.arr (xs.map Json.str) else Json.null
  return Json.mkObj [("codes", Json.arr (cs.map Json.str)), ("opened", codes (Produced.opened rec_ opened)),
    ("shared", codes (Produced.shared (Produced.wanted rec_ known) (entries verdicts)))]

/-- `{"program", "value", "row", "salt", "registration", "word_bits"}` to `{"root", "bc"}` in hex, `produced.known_port`'s
root and opening: four strings (the salt lowercase hex), a list of natural numbers and one; refused where it raises. -/
def knownPort (j : Json) : V Json := do
  let (.str program, .str value, .str salt, .str registration) :=
      (getD j "program", getD j "value", getD j "salt", getD j "registration")
    | throw "program, value, salt or registration is not a string"
  let .arr row := getD j "row" | throw "row is not a list"
  let some row := row.mapM nat? | throw "row is not natural numbers"
  let some wb := nat? (getD j "word_bits") | throw "word_bits is not a natural number"
  let (root, bc) ← Produced.known program value row (← Flock.unhex salt) registration wb
  return Json.mkObj [("root", Json.str (Flock.hexOf root)), ("bc", Json.str (Flock.hexOf bc))]

/-- `{"draw", "population", "law"}`, the population an integer and the law an object, to `check_draw`'s codes. -/
def checkDraw (j : Json) : V Json := do
  if (int? (getD j "population")).isNone then throw "population is not an integer"
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  return (Draw.checkDrawFull (getD j "draw") (getD j "population") (getD j "law")).toJson

/-- `{"law", "population", "source"}`, the law an object, the population a natural number and the source hex, to the draw
`coins.draw_first` samples from the source's bytes; refused where it raises. -/
def draw (j : Json) : V Json := do
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  let some n := nat? (getD j "population") | throw "population is not a natural number"
  Draw.drawFirst (getD j "law") n (← Flock.unhex (← Flock.J.str j "source" "the draw"))

/-- `{"rule", "program", "query", "work_table", "closure", "source"}` to `{"draw", "read"}`: the challenger's draw
(`Select.select`) under a Spec's selection rule over the units of the program under the query (`Select.resolve`, a rate
law's `subset:k`), from the source's bytes,
and how many of them it read. The program and the query are objects, the work table and the closure map
(`Select.closureMap?`) null when absent, the source hex; refused where the Python raises. -/
def select (j : Json) : V Json := do
  let rule ← Select.Rule.ofJson (getD j "rule")
  let .obj _ := getD j "program" | throw "program is not an object"
  let .obj _ := getD j "query" | throw "query is not an object"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  let table := match getD j "work_table" with
    | .null => none
    | t => some t
  let cm ← match getD j "closure" with
    | .null => pure none
    | c => some <$> Select.closureMap? c
  let rule ← Select.resolve rule p (getD j "query")
  let (draw, _, read) ← Select.select rule p (getD j "query") table cm (← Flock.unhex (← Flock.J.str j "source" "the select"))
  return Json.mkObj [("draw", draw), ("read", Lean.toJson read)]

/-- `select`'s input to `{"design"}`: the sampling design (`Core.SamplingDesign.ofFlock`) of C-Flock's law of the challenger's
draw, as the profile writes it (`Core.SamplingDesign.toJson`); refused where `select` is. -/
def design (j : Json) : V Json := do
  let d ← Flock.Draw.UnitDraw.ofJson (getD (← select j) "draw")
  let some design := Core.SamplingDesign.ofFlock d.law | throw "the draw's law has no sampling design"
  return Json.mkObj [("design", design.toJson)]

/-- `select`'s input to `{"classes", "checked"}`: the claims' partition over the program under the query (`Slots.layout`),
and the slots of the challenger's draw on the source's bytes, each as `[slot, 0]` (`Slots.checked`); refused where
`select` is. -/
def slots (j : Json) : V Json := do
  let rule ← Select.Rule.ofJson (getD j "rule")
  let .obj _ := getD j "program" | throw "program is not an object"
  let .obj _ := getD j "query" | throw "query is not an object"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  let table := match getD j "work_table" with
    | .null => none
    | t => some t
  let cm ← match getD j "closure" with
    | .null => pure none
    | c => some <$> Select.closureMap? c
  let rule ← Select.resolve rule p (getD j "query")
  let (l, checked) ← Slots.select rule p (getD j "query") table cm (← Flock.unhex (← Flock.J.str j "source" "the slots"))
  return Json.mkObj [("classes", l.toJson),
    ("checked", Json.arr (checked.map fun v => Json.arr #[Lean.toJson v, Lean.toJson (0 : Nat)]))]

/-- `{"draw", "proved_units", "instances"}`, a draw whose units (and closure) are integers, to `check_proved`'s codes. -/
def checkProved (j : Json) : V Json := do
  let draw := getD j "draw"
  if !(draw matches .obj _) then throw "draw is not an object"
  let _ ← ints (getD draw "units") "units"
  if let some c := field? draw "closure" then let _ ← ints c "closure"
  let us ← ints (getD j "proved_units") "proved_units"
  if (int? (getD j "instances")).isNone then throw "instances is not an integer"
  return (Draw.checkProvedFull draw (us.map Lean.toJson) (getD j "instances")).toJson

/-- `{"strata": [{"name", "size", "k"}], "delta"}` to `{"bound", "per_stratum"}` (`Stratified.count_bound` and
`stratum_bound`); refused where `Stratified` is: no strata, a name empty or twice, `k > size`, δ not in (0, 1). -/
def bound (j : Json) : V Json := do
  let .arr ss := getD j "strata" | throw "strata is not a list"
  let strata ← ss.mapM fun s => do
    if keys? s != some #["k", "name", "size"] then throw "a stratum is exactly {name, size, k}"
    let (.str name, some size, some k) := (getD s "name", nat? (getD s "size"), nat? (getD s "k"))
      | throw "a stratum's name is a string and its size and k natural numbers"
    if name.isEmpty || size < k then throw "a stratum needs a name and 0 <= k <= size"
    pure ({ name, size, k } : Core.Stratum)
  if strata.isEmpty then throw "no strata"
  if (sortStrings (strata.map (·.name))).toList.eraseDups.length != strata.size then throw "a name twice"
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let δ ← Core.Ratio.ofString d "delta"
  if δ.num == 0 || δ.den ≤ δ.num then throw "delta is not in (0, 1)"
  return Json.mkObj [("bound", Lean.toJson (Bound.countBound strata δ)),
    ("per_stratum", Json.mkObj (strata.toList.map fun s => (s.name, Lean.toJson (Bound.stratumBound s δ))))]

/-- `{"rows": [{"size", "k", "harm"}], "delta"}` to `{"budget", "value", "floor"}`: `Bound.lnUpper δ` and the greedy fill
at it (`Bound.densityBound`), ratios `n` or `n/d`, and the fill's floor; refused where a row is not exactly
`{size, k, harm}` of natural numbers `k ≤ size` and a ratio, or δ not in (0, 1). -/
def harmBound (j : Json) : V Json := do
  if keys? j != some #["delta", "rows"] then throw "exactly {rows, delta}"
  let .arr rs := getD j "rows" | throw "rows is not a list"
  let rows ← rs.mapM fun r => do
    if keys? r != some #["harm", "k", "size"] then throw "a row is exactly {size, k, harm}"
    let (some size, some k, .str h) := (nat? (getD r "size"), nat? (getD r "k"), getD r "harm")
      | throw "a row's size and k are natural numbers and its harm a ratio"
    if size < k then throw "a row needs k <= size"
    let h ← Core.Ratio.ofString h "harm"
    pure (size, k, (h.num : Rat) / (h.den : Rat))
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let δ ← Core.Ratio.ofString d "delta"
  if δ.num == 0 || δ.den ≤ δ.num then throw "delta is not in (0, 1)"
  let budget := Bound.lnUpper ((δ.num : Rat) / (δ.den : Rat))
  let value := Bound.fill rows budget
  return Json.mkObj [("budget", .str (toString budget)), ("floor", Lean.toJson value.floor.toNat),
    ("value", .str (toString value))]

/-- `{"rows": [{"size", "k", "harm", "take", "lo", "hi"}], "delta", "multiplier"}` to `{"lower", "upper"}`, ratios around
`StratifiedBlock.maxHarm` at δ (`Bound.maxHarmCheck`); refused where a row is not exactly those keys, natural numbers and
a ratio harm, `k ≤ size`, δ not in (0, 1), the multiplier not a ratio, or the certificate fails. -/
def maxHarm (j : Json) : V Json := do
  if keys? j != some #["delta", "multiplier", "rows"] then throw "exactly {rows, delta, multiplier}"
  let .arr rs := getD j "rows" | throw "rows is not a list"
  let rows ← rs.mapM fun r => do
    if keys? r != some #["harm", "hi", "k", "lo", "size", "take"] then
      throw "a row is exactly {size, k, harm, take, lo, hi}"
    let (some size, some k, .str h, some take, some lo, some hi) :=
        (nat? (getD r "size"), nat? (getD r "k"), getD r "harm", nat? (getD r "take"), nat? (getD r "lo"),
          nat? (getD r "hi"))
      | throw "a row's size, k, take, lo and hi are natural numbers and its harm a ratio"
    if size < k then throw "a row needs k <= size"
    let h ← Core.Ratio.ofString h "harm"
    pure ({ size, k, harm := (h.num : Rat) / (h.den : Rat), take, lo, hi } : Bound.HarmRow)
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let δ ← Core.Ratio.ofString d "delta"
  if δ.num == 0 || δ.den ≤ δ.num then throw "delta is not in (0, 1)"
  let .str m := getD j "multiplier" | throw "multiplier is not a ratio"
  let m ← Core.Ratio.ofString m "multiplier"
  let (lower, upper) ← Bound.maxHarmCheck rows ((δ.num : Rat) / (δ.den : Rat)) ((m.num : Rat) / (m.den : Rat))
  return Json.mkObj [("lower", .str (toString lower)), ("upper", .str (toString upper))]

/-- `{"population", "p", "delta"}` to `{"bound"}` (`Bernoulli.count_bound`); refused where `Bernoulli` is: `p` not a
ratio in [0, 1], δ not in (0, 1). -/
def bernoulliBound (j : Json) : V Json := do
  if keys? j != some #["delta", "p", "population"] then throw "exactly {population, p, delta}"
  let some population := nat? (getD j "population") | throw "population is not a natural number"
  let .str p := getD j "p" | throw "p is not a ratio"
  let p ← Core.Ratio.ofString p "p"
  if p.den < p.num then throw "p is not in [0, 1]"
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let δ ← Core.Ratio.ofString d "delta"
  if δ.num == 0 || δ.den ≤ δ.num then throw "delta is not in (0, 1)"
  return Json.mkObj [("bound", Lean.toJson (Bound.bernoulliBound population p δ))]

private def hiddenWidth (j : Json) : V Nat := do
  match nat? (getD j "width") with
  | some (w + 1) => pure (w + 1)
  | _ => throw "width is not a positive integer"

private def hiddenNats (j : Json) (what : String) : V (List Nat) := do
  let .arr xs := j | throw s!"{what} is not a list of natural numbers"
  xs.toList.mapM fun x => match nat? x with
    | some n => pure n
    | none => throw s!"{what} is not a list of natural numbers"

private def hiddenCalls (j : Json) : V (List HiddenDraw.Call) := do
  let .arr cs := j | throw "calls is not a list"
  cs.toList.mapM fun c => match c with
    | .arr #[o, m] => match nat? o, nat? m with
      | some offset, some rows => pure { offset, rows }
      | _, _ => throw "a call is not [offset, rows] of natural numbers"
    | _ => throw "a call is not [offset, rows] of natural numbers"

private def hiddenWitnessJson : Option (Nat × Nat × Nat) → Json
  | some (c, j, t) => Json.arr #[Lean.toJson c, Lean.toJson j, Lean.toJson t]
  | none => Json.null

/-- `{"rows", "width"}` to `{"offsets", "total"}` (`HiddenDraw.offsets`, `total`); refused where a row count is not a
natural number or the width not positive. -/
def hiddenOffsets (j : Json) : V Json := do
  if keys? j != some #["rows", "width"] then throw "exactly {rows, width}"
  let width ← hiddenWidth j
  let rows ← hiddenNats (getD j "rows") "rows"
  return Json.mkObj [("offsets", Lean.toJson (HiddenDraw.offsets width rows)),
    ("total", Lean.toJson (HiddenDraw.total width rows))]

/-- `{"calls", "positions", "width"}` to `{"located": [[c, j, t] or null]}` (`HiddenDraw.locate`), calls a list of
`[offset, rows]`. -/
def hiddenLocate (j : Json) : V Json := do
  if keys? j != some #["calls", "positions", "width"] then throw "exactly {calls, positions, width}"
  let width ← hiddenWidth j
  let calls ← hiddenCalls (getD j "calls")
  let positions ← hiddenNats (getD j "positions") "positions"
  return Json.mkObj [("located", Json.arr (positions.toArray.map (hiddenWitnessJson <| HiddenDraw.locate width calls ·)))]

/-- `{"call", "calls", "position", "width"}` to `{"witness": [c, j, t] or null}` (`HiddenDraw.witness`). -/
def hiddenWitness (j : Json) : V Json := do
  if keys? j != some #["call", "calls", "position", "width"] then throw "exactly {call, calls, position, width}"
  let width ← hiddenWidth j
  let calls ← hiddenCalls (getD j "calls")
  let (some c, some r) := (nat? (getD j "call"), nat? (getD j "position"))
    | throw "call or position is not a natural number"
  return Json.mkObj [("witness", hiddenWitnessJson (HiddenDraw.witness width calls c r))]

/-- `{"params", "state", "op"}` to `{"decision", "state"}`: the op's decision on the state the stream's rules hold of it
(`Stream.run`), and the state after it as the text Python's `canonical_json` writes. -/
def stream (j : Json) : V Json := do
  if keys? j != some #["op", "params", "state"] then throw "exactly {params, state, op}"
  let p ← Stream.Params.ofJson (getD j "params")
  let (d, s) ← Stream.run p (← Stream.State.ofJson (getD j "state")) (getD j "op")
  return Json.mkObj [("decision", d.toJson), ("state", Json.str (← PyJson.canonical s.toJson))]

/-- `{"state", "op"}` to `{"decision", "state"}`: the op's decision on the state the fleet's rules hold of it
(`Fleet.run`), and the state after it as the text Python's `canonical_json` writes. -/
def fleet (j : Json) : V Json := do
  if keys? j != some #["op", "state"] then throw "exactly {state, op}"
  let (d, s) ← Fleet.run (← Fleet.State.ofJson (getD j "state")) (getD j "op")
  return Json.mkObj [("decision", d.toJson), ("state", Json.str (← PyJson.canonical s.toJson))]

/-- `{"verdicts": {name: {...}}}` to `{"verdict"}`. -/
def verdict (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  let name : Core.Verdict := match OneStage.verdictOf vs with
    | .accept () () => .accept
    | .reject _ => .reject
    | .undecided _ => .undecided
  return Json.mkObj [("verdict", Json.str name.name)]

/-- `{"record", "scheme", "roots", "domains", "population"}` to `matches_public`'s `{"ok", "codes", "detail"}`: a record
whose roots are objects with a string port, root and domain, which has a scheme and an integer population; a string
scheme, roots and domains objects of strings, an integer population. (Python's `==` and `jeq` agree there: no boolean
meets an integer.) -/
def matchesPublic (j : Json) : V Json := do
  let rec_ := getD j "record"
  let .arr roots := getD rec_ "roots" | throw "the record's roots are not a list"
  if !roots.all fun r => ["port", "root", "domain"].all (getD r · matches .str _) then
    throw "a registered root is not an object with a string port, root and domain"
  if (field? rec_ "scheme").isNone || (int? (getD rec_ "population")).isNone then
    throw "the record has no scheme or no integer population"
  let strings := fun (k : String) => match getD j k with
    | .obj _ => (entries (getD j k)).all (·.2 matches .str _)
    | _ => false
  if !(getD j "scheme" matches .str _) || !strings "roots" || !strings "domains" || (int? (getD j "population")).isNone then
    throw "scheme is not a string, roots or domains not an object of strings, or population not an integer"
  return (Record.matchesPublic rec_ (getD j "scheme") (getD j "roots") (getD j "domains") (getD j "population")).toJson

/-- `{"verdicts", "registration"?}` to `{"registered": {value: root} or null}` (`Record.sessionRegistered`, with a
registration `Record.ownRegistered`); refused where `inputOf` refuses the verdicts, and a registration whose roots are not
a list of objects with a string port or whose registered values, if any, are not a list of objects with a string value. -/
def sessionRegistered (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  if !vs.all fun (_, v) =>
      !(truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _))
      || (entries (getD v "registered")).all fun (_, e) => (getD e "value" matches .str _) && (field? e "root").isSome then
    throw "a verifier of record reports registered reads that are not objects with a string value and a root"
  let opened := Record.sessionRegistered vs
  let opened ← match field? j "registration" with
    | none => pure opened
    | some reg => do
      let .arr roots := getD reg "roots" | throw "the registration's roots are not a list"
      if !roots.all (getD · "port" matches .str _) then throw "a registered root has no string port"
      if let some es := field? reg "registered" then
        let .arr es := es | throw "the registration's registered values are not a list"
        if !es.all (getD · "value" matches .str _) then throw "a registered value has no string value"
      pure (Record.ownRegistered reg opened)
  return Json.mkObj [("registered", match opened with
    | some kvs => Json.mkObj kvs.toList
    | none => Json.null)]

/-- `audit_record`'s arguments (`Record.inputOf`) to the record, as the text Python's `canonical_json` writes. -/
def record (j : Json) : V Json := do
  return Json.str (← PyJson.canonical (← Record.record (← Record.inputOf j)))

/-- `{"record", "scheme", "query", "values": [{"name", "registered"}], "public": [item]}` to the service's public-items
rule's `{"ok", "codes", "detail"}` (`Service.checkPublic`), the query's digest computed here: a string scheme, the check's query
an object, each value a string name with a boolean, each declared item a string. -/
def checkPublic (j : Json) : V Json := do
  if !(getD j "scheme" matches .str _) then throw "scheme is not a string"
  if !(getD j "query" matches .obj _) then throw "query is not an object"
  let .arr vs := getD j "values" | throw "values is not a list"
  let values ← vs.mapM fun v => match getD v "name", getD v "registered" with
    | .str name, .bool registered => pure ({ name, registered } : Service.Value)
    | _, _ => throw "a value is not a string name with a boolean"
  let .arr ps := getD j "public" | throw "public is not a list"
  let declared ← ps.mapM fun
    | .str s => pure s
    | _ => throw "a declared item is not a string"
  let query ← Registration.queryDigest (getD j "query")
  return (← Service.checkPublic (getD j "record") (getD j "scheme") query values declared).toJson

/-- `{"verdicts", "sessions", "outputs"}` to the outcome's verdict checks' `{"ok", "codes", "detail"}`
(`Service.checkVerdicts`): verdicts an object of objects, sessions a list of strings, outputs distinct strings. -/
def checkVerdicts (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  let .arr ss := getD j "sessions" | throw "sessions is not a list"
  if !ss.all (· matches .str _) then throw "a session is not a string"
  let .arr os := getD j "outputs" | throw "outputs is not a list"
  let outputs ← os.mapM fun
    | .str o => pure o
    | _ => throw "an output is not a string"
  if (sortStrings outputs).toList.eraseDups.length != outputs.size then throw "an output twice"
  return (← Service.checkVerdicts vs ss outputs).toJson

private def valuesOf (j : Json) (what : String) : V (Array Service.Value) := do
  let .arr vs := j | throw s!"{what} is not a list"
  vs.mapM fun v => match getD v "name", getD v "registered" with
    | .str name, .bool registered => pure ({ name, registered } : Service.Value)
    | _, _ => throw "a value is not a string name with a boolean"

/-- `{"params", "record", "rule", "values", "public", "challenger", "received", "source", "closure", "sessions",
"verdicts", "outputs", "statement", "delta", "proved_units", "instances", "served", "session_roots", "coin_source",
"coins"}` to the
stream's run of one registration (`Run.run`): `registration`'s parameters, the record, `select`'s rule, closure map and
hex source, `check_public`'s values and declared items, the challenger's name and the log's length at the receipt
(strings), `check_verdicts`' sessions, verdicts and outputs, and `record`'s statement, δ, served units and session roots,
with the proved units and instances null for the draw's own, where the coins came from (`coin_source`: `verifier` or
`challenger`, required) and the coins' evidence (`coins_reason`'s, which a draw from the challenger must carry and one
from the verifier's own coins must not). -/
def run (j : Json) : V Json := do
  let p ← params (getD j "params")
  let rule ← Select.Rule.ofJson (getD j "rule")
  let values ← valuesOf (getD j "values") "values"
  let .arr ps := getD j "public" | throw "public is not a list"
  let declared ← ps.mapM fun
    | .str s => pure s
    | _ => throw "a declared item is not a string"
  let (.str challenger, .str received) := (getD j "challenger", getD j "received")
    | throw "challenger or received is not a string"
  let closure ← match getD j "closure" with
    | .null => pure none
    | c => some <$> Select.closureMap? c
  let .arr sessions := getD j "sessions" | throw "sessions is not a list"
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let verdicts := entries (getD j "verdicts")
  if !verdicts.all (·.2 matches .obj _) then throw "a verdict is not an object"
  let .arr os := getD j "outputs" | throw "outputs is not a list"
  let outputs ← os.mapM fun
    | .str o => pure o
    | _ => throw "an output is not a string"
  if (sortStrings outputs).toList.eraseDups.length != outputs.size then throw "an output twice"
  let provedUnits ← match getD j "proved_units" with
    | .null => pure none
    | u => some <$> ints u "proved_units"
  let fromChallenger ← match getD j "coin_source" with
    | .str "challenger" => pure true
    | .str "verifier" => pure false
    | _ => throw "coin_source is \"verifier\" (the verifier's own coins) or \"challenger\" (the auditor's challenger's)"
  let instances ← match getD j "instances" with
    | .null => pure none
    | x => match int? x with
      | some n => pure (some n)
      | none => throw "instances is not an integer"
  Run.run { params := p, record := getD j "record", rule, values, declared, challenger, received,
            source := ← Flock.unhex (← Flock.J.str j "source" "the run"), closure, sessions, verdicts, outputs,
            statement := getD j "statement", delta := getD j "delta", provedUnits, instances,
            served := getD j "served", sessionRoots := getD j "session_roots", fromChallenger,
            coins := getD j "coins" }

def table : List (String × (Json → V Json)) :=
  [("decode", decode), ("partition", partition), ("committed_bits", committedBits), ("registration", registration),
   ("registered", registered),
   ("domain", domain), ("produced", produced), ("known_port", knownPort), ("check_draw", checkDraw), ("draw", draw), ("select", select), ("design", design),
   ("slots", slots), ("check_proved", checkProved),
   ("bound", bound), ("harm_bound", harmBound), ("max_harm", maxHarm), ("bernoulli_bound", bernoulliBound),
   ("hidden_offsets", hiddenOffsets), ("hidden_locate", hiddenLocate), ("hidden_witness", hiddenWitness),
   ("stream", stream), ("fleet", fleet), ("verdict", verdict), ("matches_public", matchesPublic),
   ("session_registered", sessionRegistered), ("record", record), ("check_public", checkPublic),
   ("check_verdicts", checkVerdicts), ("run", run), ("hello", Hello.check),
   ("challenger_commitment", Challenger.commitmentPiece),
   ("challenger_opening", Challenger.openingPiece), ("challenger_reason", Challenger.reasonPiece),
   ("challenger_order", Challenger.orderPiece), ("challenger_key", Challenger.keyPiece),
   ("challenger_record", Challenger.recordPiece), ("challenger_trusted", Challenger.trustedPiece),
   ("challenger_close", Challenger.closePiece), ("challenger_released", Challenger.releasedPiece),
   ("challenger_ledger", Challenger.ledgerPiece), ("challenger_head", Challenger.headPiece),
   ("challenger_session", Challenger.sessionPiece), ("challenger_credited", Challenger.creditedPiece),
   ("challenger_pinned", Challenger.pinnedPiece), ("challenger_read", Challenger.readPiece),
   ("challenger_draw_coins", Challenger.drawCoinsPiece),
   ("wires", Wires.piece), ("wires_unrolled", Wires.pieceRef),
   ("wires_drawn", Wires.pieceDrawn), ("wires_drawn_unrolled", Wires.pieceDrawnRef)]

def verifier : List (String × (Json → V Json)) :=
  [("registration", registration), ("matches_public", matchesPublic), ("select", select), ("check_draw", checkDraw),
   ("check_proved", checkProved), ("record", record), ("check_public", checkPublic),
   ("check_verdicts", checkVerdicts), ("run", run), ("hello", Hello.check), ("stream", stream), ("fleet", fleet)]

end SampledProofs.Pieces
