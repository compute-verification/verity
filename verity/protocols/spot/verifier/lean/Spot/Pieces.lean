/-
The one-stage audit's pieces as JSON functions, which `sampled-proofs-difftest` serves (`Difftest.serve`) for the
difftest against `one_stage`'s Python, beside the whole protocol (`sampled-proofs-one-stage serve`):

* `decode`: a case's three decoders run independently (`answer` stops at the first refusal), the parameters and the
  evidence as parsed (`OneStage.paramsParsed`, `evidenceParsed`), before the protocol's refusals of what soundness
  doesn't cover;
* `partition`: the partition's digest, population, strata and expanded law from the verifier's parameters;
* `committed_bits`: per Definition `Q_call` cuts, its units' committed bits read off its layout and by the walk that
  layout replaced (`Partition.committedBits`, `Partition.committedBitsRef`);
* `unit_gates`: each unit's gates over a program and query, a template query's instances a range at a time and by the
  walk unit by unit that replaced (`Partition.unitGates`, `Partition.unitGatesRef`);
* `registration`: R1–R8's failure codes for a record against the parameters, with the fan-out under a template query;
* `fanout`: a template query's fan-out and the run Input nodes its proof blocks don't each read alone;
* `registered`: the verifier's own entries for the values its parameters expect registered;
* `domain`: a registered value's commitment domain, or a produced value's (`Produced.domain`);
* `produced`: a produced values' record's codes against the verifier's own entries, and the roots the verifiers
  opened them against, and the rows they opened against its openings and the verifier's own ports' (`Produced.check`,
  `Produced.opened`, `Produced.shared`);
* `known_port`: the root and opening of a port the verifier registers at known content (`Produced.known`);
* `check_draw`, `check_proved`: the sample's checks, each `{"ok", "codes", "detail"}`;
* `draw`: the sample from given bytes of the challenger's coins;
* `select`: the challenger's sample from a Spec's selection rule on, over a program and query, from given bytes of its
  coins, with the number of bytes it read (`Select.select`);
* `select_second`: a two-stage rule's second stage, `ks[t]` proof blocks of each cell (a sampled replay block's
  instances of template `t`), from given bytes of the challenger's coins, with the number of bytes it read
  (`Select.selectSecond`);
* `second_statements`: round 2's statement check of second-stage sessions' statements against the interior
  registration and the second stage's cells, the two-stage form allowed or not (`SecondStage.check`);
* `second_interior_reads`: of that check's input, the statements' reads that rest on the interior's own values
  (`SecondStage.interiorReads`);
* `replay_links`: the design's links, from the replay block's Definition in a program (`SecondStage.replayLinks`);
* `replay_wires`: the design's wires, from the replay block's Definition in a program (`SecondStage.replayWires`);
* `replay_rows`: the rows the design's links' and wires' units read, from the replay block's Definition in a program
  (`SecondStage.replayRows`);
* `replay_returns`: the names the replay block's return has in the boundary registration's record, from its
  Definition in a program (`SecondStage.replayReturns`);
* `replay_returned`: the bits a replay block returns, from its Definition in a program
  (`SecondStage.replayReturned`);
* `second_schedule`: why round 2 refuses the boundary record's schedule, and each replay block's request and positions
  under it (`Schedule.failure`, `Schedule.at?`);
* `slots`: the claims' partition over the program and query, and the slots of that sample's checked proof blocks
  (`Slots.select`);
* `fits`: what of v2's inputs and of a session's draw a session's statement would hold at a given number of bytes or
  more (`OneStageZk.fitsWhy`, `OneStageZk.drawFitsWhy`), which v2 refuses at `2^64`;
* `file_population`: why the registration's population over a program and query is not a public file's units in order
  (`FilePart.why`);
* `tables_held`: v2's refusal of blobs whose pinned MUFU tables are not a circuit's lookups, on a META's `lookups` and
  the blobs' digests (`OneStageZk.tablesHeld`);
* `bound`: the exact bounds on incorrect proof blocks at δ;
* `harm_bound`: the harm they hold at δ in closed form (`Bound.densityBound`);
* `max_harm`: a certificate's bracket around `Flock.Guarantees.JointStratified`'s exact supremum of that harm
  (`Bound.maxHarmCheck`), which `certify` is held to;
* `hidden_offsets`, `hidden_locate`, `hidden_witness`: the sample over hidden positions (`HiddenDraw`): the honest table's
  offsets and total, the honest prover's witnesses, and the witness for a named call;
* `verdict`: the verdicts' share of the outcome once the record's checks hold;
* `matches_public`: the verifier's own public file against the record (`Record.matchesPublic`);
* `session_registered`: the registered roots the verifiers report (`Record.sessionRegistered`), and with a
  registration only its own values' (`Record.ownRegistered`);
* `record`: the whole audit record (`Record.record`), as the text of Python's `canonical_json`;
* `across`: the verdicts on one read across streams (`Record.across`): the later registration's previous record in the
  earlier stream's log, the order the challenger's signed log shows, and the earlier stream's audit record, whether it
  accepted and whether it is complete, as the record the caller hands says: nothing holds that record to the one the
  earlier stream's verifier wrote;
* `check_public`: the proof system's public-items rule on a registration record (`Service.checkPublic`);
* `check_verdicts`: the verdicts' share of a proof system outcome (`Service.checkVerdicts`);
* `run`: the stream's run of one registration from its public check through its verdict (`Run.run`);
* `template_circuits`: a two-stage run's params naming each template's circuit file by SHA-512 and its statements'
  reference header and registered reads, in the check's order (`Run.templateCircuits`);
* `stream`: the proof system stream's own rules, one op's decision on what they hold of the stream (`Stream.run`);
* `fleet`: the fleet's rules over its streams' operators, one op's decision on what they hold of the fleet (`Fleet.run`);
* `hello`: the run's Hello on its record, its coin tree at run scope (`Hello.check`).
* `challenger_commitment`, `challenger_opening`, `challenger_reason`, `challenger_order`: the challenger's live coins
  (`Challenger`): a coin commitment and an opening as `from_json` reads them, `live_reason` and `order_reason`.
* `challenger_key`, `challenger_record`, `challenger_trusted`: the challenger's signed record (`ChallengerRecord`): the
  verifier's key and a record as `from_json` reads them, with `record_message`, and `trusted_reason`.
* `challenger_close`, `challenger_released`, `challenger_ledger`, `challenger_day`, `challenger_head`,
  `challenger_session`, `challenger_credited`, `challenger_pinned`: the challenger's session (`ChallengerSession`): its
  close as `from_json` reads it, with `close_message`; `log_chain` and `released_roots_of`; `audited_session`, the
  auditor's ledger; `declared_day`, a window's day and its lane's declarations before it; `ledger_head`, the ledger's
  published heads; `trusted_session_of` and `credited_session`, a window counted toward its
  period's circuit, each over `pinned_session`: a session from a commit the host's pins log records and the auditor's
  first-parent list of main holds.
* `challenger_read`, `challenger_draw_coins`, `challenger_within`: a sample's coins from one challenger family
  (`ChallengerCoins`): `read_reason`, the bytes a sample read against its segment, `coins_reason`, against the segment
  and the session the auditor's ledger opens for the verifier's window (`trusted_session_of`), and `coins_within`, the
  sizes the evidence holds against v2's limits.
* `challenger_seed_salt`: a window's seed salt as the verifier reads it from the challenger's log (`seed_salt_of`,
  `Challenger.seedSaltOf`): the one coin of the seed-salt family released after the window's open root.
* `challenger_coin_trees`, `challenger_session_trees`, `challenger_sample_link_first`: a `--zk` session's coins
  (`ChallengerTrees`): the coin trees a release log shows the challenger sampled and opened (`coin_trees_of`), an
  audit's session records, each on its tables with their `root_B` lengths, held to them, one tree a session
  (`sessions_tree_reason`), and the log's order, the sample, then the link, then the other rounds
  (`sample_link_first`).
* `challenger_inner_trees`: a recursive session's inner coins (`ChallengerInnerTrees`): `rec_live`'s live-coin part
  records held to the trees a release log shows, one tree a session, every opening read (`rec_live.tree_held_reason`).
* `wires`, `wires_unrolled`: a window circuit's wires checked at the size of their description (`Wires.check`) and by
  the unrolled reference (`Wires.checkRef`), each with its summary: wired rows and bits, spans, and sampled rows' writers.
* `wires_drawn`, `wires_drawn_unrolled`: the wire check on the sampled proof blocks alone (`Wires.Summary.drawnOk`, each sampled
  wired row's public string its writer's) and on the reference's entries whose unit is sampled.
* `history_root`, `history_check`, `history_order`, `history_strings`, `history_roots`: a history table
  (`HistoryTable`): its domain, placeholder and root over its writers' `b ‖ c` at a capacity, the registered root held
  to that root, each position's source from the program's runs, the strings at those sources from the writer's members'
  public bodies, and each table's root from bodies the leaf layer binds, held to the root the window registers.

`sampled-proofs-verifier` serves those the verifier runs (`verifier`): `registration`, `matches_public`,
`select` (the sample of record: `Stream.select` hands it the bytes of the challenger's coins), `select_second` (a
two-stage rule's second stage, on the coins after the interior registration), `check_draw`,
`check_proved`, `record`, `across`, `check_public`, `check_verdicts`, `run` (`Stream.outcome` holds its
outcome to it),
`hello`,
`stream` (each of `service.Stream`'s ops) and `fleet` (each of `service.Fleet`'s).
-/
module

public import Spot.OneStage
public import Spot.Slots
public import Spot.FilePart
public import Spot.Service
public import Spot.Run
public import Spot.RecordAcross
public import Spot.Stream
public import Spot.Fleet
public import Spot.Hello
public import Spot.ChallengerRecord
public import Spot.ChallengerSession
public import Spot.ChallengerCoins
public import Spot.ChallengerTrees
public import Spot.ChallengerInnerTrees
public import Spot.Produced
public import Spot.HiddenDraw
public import Spot.PartitionRef
public import Spot.Returned
public import Spot.WiresDrawn
public import Spot.OneStageZk
public import Spot.HistoryTable

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Pieces

open Lean (Json)
open Flock (V)

private def params (j : Json) : V OneStage.Params := do
  OneStage.paramsParsed (← Core.canonicalJson (← Flock.canon j).toUTF8)

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
    ("evidence", Json.bool (OneStage.evidenceParsed e).toBool)]

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

/-- `{"program", "query"}`, a program object and a query, to `[gates, gates by walk]`: each unit's gates `[lo, hi)` in
root gates (`Partition.unitGates`, a template query's instances a range at a time) and the same unit by unit
(`Partition.unitGatesRef`, the walk it replaced); refused where the query is. -/
def unitGates (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  let q := getD j "query"
  let range := fun ((lo, hi) : Nat × Nat) => Json.arr #[Lean.toJson lo, Lean.toJson hi]
  return Json.arr #[Json.arr ((← Partition.unitGates p q).map range), Json.arr ((← Partition.unitGatesRef p q).map range)]

/-- `{"program", "query", "held"?}`, a program object, a template query and the gates `[[first, count], ...]` of the
parameters a verifier expects registered, to `{"fanout", "shared"}`: its fan-out and the run Input nodes outside `held`
its proof blocks don't each read alone (`Partition.fanout`, `Partition.sharedInputs`); refused for another query or one
that doesn't apply. -/
def fanout (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  let bad : V (Nat × Nat) := throw "held is not a list of [first, count]"
  let held ← match getD j "held" with
    | .null => pure #[]
    | .arr xs => xs.mapM fun x => match x with
      | .arr #[a, b] => match nat? a, nat? b with
        | some first, some count => pure (first, count)
        | _, _ => bad
      | _ => bad
    | _ => throw "held is not a list of [first, count]"
  let some rs := Partition.templateRanges? p (getD j "query") | throw "not a template query that applies"
  return Json.mkObj [("fanout", Lean.toJson (Partition.fanout p.prog rs)),
    ("shared", Lean.toJson (Partition.sharedInputs p.prog rs held))]

/-- `{"params", "record"}` to R1–R8's codes, and under a template query that applies the fan-out the verdict publishes
(`Partition.fanout?`) and, when any, the tied values R8 let through (`Registration.tiedThrough`), whatever the
record. -/
def registration (j : Json) : V Json := do
  let v := (← params (getD j "params")).verifier
  let cs := (Registration.check (getD j "record") v).map Json.str
  let through := Registration.tiedThrough v
  return Json.mkObj (("codes", Json.arr cs) :: ((Partition.fanout? v.program v.query).map ("fanout", Lean.toJson ·)).toList
    ++ (if through.isEmpty then [] else [("rests_on_period_check", Lean.toJson through)]))

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
registration and sample strings, `mine` the verifier's own entries (`{value: {"leaves", "words", "word_bits", "domain"}}`,
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

/-- `{"draw", "population", "law"}`, the population an integer and the sampling design an object, to `check_draw`'s codes. -/
def checkDraw (j : Json) : V Json := do
  if (int? (getD j "population")).isNone then throw "population is not an integer"
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  return (Draw.checkDrawFull (getD j "draw") (getD j "population") (getD j "law")).toJson

/-- `{"law", "population", "source"}`, the sampling design an object, the population a natural number and the source hex, to the sample
`coins.draw_first` samples from the source's bytes; refused where it raises. -/
def draw (j : Json) : V Json := do
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  let some n := nat? (getD j "population") | throw "population is not a natural number"
  Draw.drawFirst (getD j "law") n (← Flock.unhex (← Flock.J.str j "source" "the draw"))

/-- `{"rule", "program", "query", "work_table", "closure", "source"}` to `{"draw", "read"}`: the challenger's sample
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
  let table ← rule.tableOver p (getD j "query") table
  let rule ← Select.resolve rule p (getD j "query")
  let (draw, _, read) ← Select.select rule p (getD j "query") table cm (← Flock.unhex (← Flock.J.str j "source" "the select"))
  return Json.mkObj [("draw", draw), ("read", Lean.toJson read)]

/-- `{"sample", "ns", "ks", "source"}` to `{"draw", "read"}`: a two-stage rule's second stage (`Select.selectSecond`)
over the proof blocks of the replay sample `sample`, `ns[t]` of template `t` in each replay block, its cell, and `ks[t]`
of them sampled in each, from the source's bytes, and how many of them it read; refused where the Python raises. -/
def selectSecond (j : Json) : V Json := do
  let .arr ss := getD j "sample" | throw "sample is not a list"
  let some sample := ss.mapM nat? | throw "sample is not a list of replay blocks"
  let (.arr ns, .arr ks) := (getD j "ns", getD j "ks") | throw "ns and ks are lists of counts"
  let (some ns, some ks) := (ns.mapM nat?, ks.mapM nat?) | throw "ns and ks are lists of counts"
  let (draw, read) ← Select.selectSecond sample ns ks (← Flock.unhex (← Flock.J.str j "source" "the second stage"))
  return Json.mkObj [("draw", draw), ("read", Lean.toJson read)]

/-- `select`'s input to `{"classes", "checked"}`: the claims' partition over the program under the query (`Slots.layout`),
and the slots of the challenger's sample on the source's bytes, each as `[slot, 0]` (`Slots.checked`); refused where
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
  let table ← rule.tableOver p (getD j "query") table
  let rule ← Select.resolve rule p (getD j "query")
  let (l, checked) ← Slots.select rule p (getD j "query") table cm (← Flock.unhex (← Flock.J.str j "source" "the slots"))
  return Json.mkObj [("classes", l.toJson),
    ("checked", Json.arr (checked.map fun v => Json.arr #[Lean.toJson v, Lean.toJson (0 : Nat)]))]

/-- `{"draw", "proved_units", "instances"}`, a sample whose units (and closure) are integers, to `check_proved`'s codes. -/
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

/-- One read across streams from `{"window", "log", "audit", "later", "coins", "read"}`: the window's open root, the
earlier stream's log (a list of registrations), its audit record or null, the later stream's registration, and the later
sample's coin evidence and the bytes it read, in lowercase hex. -/
def acrossOf (j : Json) : V Record.Across := do
  if keys? j != some #["audit", "coins", "later", "log", "read", "window"] then
    throw "exactly {window, log, audit, later, coins, read}"
  let .str window := getD j "window" | throw "window is not a string"
  let .arr log := getD j "log" | throw "log is not a list"
  let .str read := getD j "read" | throw "read is not lowercase hex"
  return { window, log, audit := getD j "audit", later := getD j "later", coins := getD j "coins",
           read := ← Flock.unhex read }

/-- One read across streams (`acrossOf`) to `Record.across`'s verdicts on it. -/
def across (j : Json) : V Json := do Record.across (← acrossOf j)

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
"verdicts", "outputs", "statement", "delta", "proved_units", "instances", "served", "session_roots", "coins"}` to the
stream's run of one registration (`Run.run`): `registration`'s parameters, the record, `select`'s rule, closure map and
hex source, `check_public`'s values and declared items, the challenger's name and the log's length at the receipt
(strings), `check_verdicts`' sessions, verdicts and outputs, and `record`'s statement, δ, served units and session roots,
with the proved units and instances null for the sample's own, and the coins' evidence (`coins_reason`'s, which a sample
that read coins must carry); under a two-stage rule, `interior` (`Run.Interior`): `{"params", "replay", "values",
"record", "received", "source", "coins", "proved_units", "instances", "served", "session_roots", "statements"}` in the
same forms, `replay` one replay block's program object, its params' `proofs` the auditor's naming each template's
circuit file, reference header and registered reads (`Run.templateCircuits`), and no `links` or `wires`, which the
verifier derives; null otherwise.
The verifier flips no coins of its own: an input whose `coin_source` is not the challenger's is refused. -/
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
  match getD j "coin_source" with
    | .null | .str "challenger" => pure ()
    | _ => throw ("the verifier draws no coins of its own: a draw's coins are the challenger's, read from its log " ++
        "(coin_source, when given, is \"challenger\")")
  let instancesOf (x : Json) : V (Option Int) := match x with
    | .null => pure none
    | x => match int? x with
      | some n => pure (some n)
      | none => throw "instances is not an integer"
  let instances ← instancesOf (getD j "instances")
  let interior ← match getD j "interior" with
    | .null => pure none
    | it => do
      let .obj _ := getD it "replay" | throw "an interior's replay is a program object"
      let .str received := getD it "received" | throw "an interior's received is a string"
      let provedUnits ← match getD it "proved_units" with
        | .null => pure none
        | u => some <$> ints u "proved_units"
      let ipj := getD it "params"
      let ip ← params (Json.mkObj ((entries ipj).filter (·.1 != "proofs")).toList)
      let circuits ← Run.templateCircuits (getD ipj "proofs") ip.verifier.query
      let reads := Run.templateReads (getD ipj "proofs")
      let replay ← Program.parse (← Flock.canon (getD it "replay")).toUTF8
      let values ← valuesOf (getD it "values") "an interior's values"
      let source ← Flock.unhex (← Flock.J.str it "source" "the interior")
      let instances ← instancesOf (getD it "instances")
      let statements ← match getD it "statements" with
        | .null => pure none
        | .arr ss => pure (some ss)
        | _ => throw "an interior's statements is a list"
      if (field? it "links").isSome then
        throw "an interior names no links: the verifier derives them from the replay block's Definition"
      if (field? it "wires").isSome then
        throw "an interior names no wires: the verifier derives them from the replay block's Definition"
      if (field? it "rows").isSome then
        throw "an interior names no rows: the verifier derives them from the replay block's Definition"
      if (field? it "returns").isSome then
        throw "an interior names no returns: the verifier derives them from the replay block's Definition"
      pure (some ({ params := ip, replay, values, record := getD it "record", received, source, coins := getD it "coins",
                    provedUnits, instances, served := getD it "served", sessionRoots := getD it "session_roots",
                    statements, circuits, reads } : Run.Interior))
  Run.run { params := p, record := getD j "record", rule, values, declared, challenger, received,
            source := ← Flock.unhex (← Flock.J.str j "source" "the run"), closure, sessions, verdicts, outputs,
            statement := getD j "statement", delta := getD j "delta", provedUnits, instances,
            served := getD j "served", sessionRoots := getD j "session_roots", coins := getD j "coins", interior }

/-- `second_stage.check`'s input, read: the two-stage form allowed or not (`allow`), the first stage's replay sample,
the interior program and query, whose strata make the parts, the interior and boundary registrations' records, the
second stage's units (`drawn`), the design's links (`replayLinks`') and wires (`replayWires`'), each null for none, the
rows their units read (`replayRows`'), null for none, the names of what the replay block returns (`returns`,
`replayReturns`'), null for none, the verdicts, whose verifiers of record report the sessions'
reads (`Record.sessionRegistered`) and name the header, registered reads and sample each checked
(`SecondStage.verifiedOf`), the reference header of each template's statement (`refs`, `{template: header}`, null for
none) and the registered reads it reads (`ref_reads`, `{template: registered}`, each null for none, null for no
template's), the verifier's own copies of the interior registration's values it fills, with their sources
(`supplied`, `SecondStage.suppliedOf?`), and the names of those it pinned or holds to an earlier record (`fixed`), each
null for none, the templates the auditor's params record as table reads with their tables' rows (`reads`, `{template:
V}`, null for none), and the statements, null for none. -/
private def secondOwn (j : Json) : V (Bool × SecondStage.Own × Option (Array Json)) := do
  let .bool allow := getD j "allow" | throw "allow is not a boolean"
  let .arr ss := getD j "sample" | throw "sample is not a list"
  let some sample := ss.mapM nat? | throw "sample is not a list of replay blocks"
  let .arr ds := getD j "drawn" | throw "drawn is not a list"
  let some drawn := ds.mapM nat? | throw "drawn is not a list of slots"
  let .obj _ := getD j "program" | throw "program is not an object"
  let .obj _ := getD j "query" | throw "query is not an object"
  let links ← match getD j "links" with
    | .null => pure none
    | ls => match SecondStage.linksOf? ls with
      | some ls => pure (some ls)
      | none => throw "links is null or a list of {template, port, value}, strings"
  let wires ← match getD j "wires" with
    | .null => pure none
    | ws => match SecondStage.wiresOf? ws with
      | some ws => pure (some ws)
      | none => throw "wires is null or a list of {template, port, writer, wport}, strings"
  let rows ← match getD j "rows" with
    | .null => pure #[]
    | rs => match SecondStage.rowsOf? rs with
      | some rs => pure rs
      | none => throw "rows is null or a list of {template, port, per, runs}, runs [unit, count, row, step] naturals"
  let returns ← match getD j "returns" with
    | .null => pure #[]
    | .arr vs => match vs.mapM (fun v => match v with | .str s => some s | _ => none) with
      | some vs => pure vs
      | none => throw "returns is null or a list of strings"
    | _ => throw "returns is null or a list of strings"
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let verdicts := entries (getD j "verdicts")
  unless verdicts.all fun (_, v) => (v matches .obj _) && (!(truthy (getD v "of_record") &&
      jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _)) ||
      (entries (getD v "registered")).all fun (_, e) => (getD e "value" matches .str _) && (field? e "root").isSome) do
    throw "a verdict is not an object, or one of record reports reads that are not objects with a string value and a root"
  let strata ← Partition.strata (← Program.parse (← Flock.canon (getD j "program")).toUTF8) (getD j "query")
  let stmts ← match getD j "statements" with
    | .null => pure none
    | .arr xs => pure (some xs)
    | _ => throw "statements is not a list"
  let refs : Array (String × Json) ← match getD j "refs" with
    | .null => pure #[]
    | rs@(.obj _) => pure (entries rs)
    | _ => throw "refs is not an object"
  let refReads : Array (String × Json) ← match getD j "ref_reads" with
    | .null => pure #[]
    | rs@(.obj _) => pure (entries rs)
    | _ => throw "ref_reads is not an object"
  let supplied ← match SecondStage.suppliedOf? (getD j "supplied") with
    | some vs => pure vs
    | none => throw "supplied is null or an object of rows, each a list of sources, each a kind naming a string"
  let fixed ← match getD j "fixed" with
    | .null => pure #[]
    | .arr vs => match vs.mapM (fun v => match v with | .str x => some x | _ => none) with
      | some vs => pure vs
      | none => throw "fixed is null or a list of strings"
    | _ => throw "fixed is null or a list of strings"
  let reads ← match getD j "reads" with
    | .null => pure #[]
    | rs@(.obj _) => match (entries rs).mapM (fun (t, v) => (nat? v).map fun v => (t, v)) with
      | some rs => pure rs
      | none => throw "reads is null or an object of naturals"
    | _ => throw "reads is null or an object of naturals"
  let own : SecondStage.Own :=
    { sample, parts := SecondStage.parts sample.size strata, interior := getD j "interior",
      boundary := getD j "boundary", drawn, links, reported := Record.sessionRegistered verdicts,
      verified := SecondStage.verifiedOf verdicts, refs, refReads, wires, rows, returns, supplied, fixed, reads }
  return (allow, own, stmts)

/-- `second_stage.check`'s input (`secondOwn`) to its verdict (`SecondStage.check`). -/
def secondStatements (j : Json) : V Json := do
  let (allow, own, stmts) ← secondOwn j
  return (SecondStage.check allow own stmts).toJson

/-- `second_stage.check`'s input (`secondOwn`) to the reads that rest on the interior's own values
(`SecondStage.interiorReads`), each `{"template", "port", "value"}`. -/
def secondInteriorReads (j : Json) : V Json := do
  let (_, own, stmts) ← secondOwn j
  return .arr (SecondStage.interiorReads own ((stmts.getD #[]).toList))

/-- `second_stage.replay_links`' input to the design's links (`SecondStage.replayLinks`): a program object, the replay
block's descriptor id in it (`replay`) and the boundary registration's record; null for none. -/
def replayLinks (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let .str key := getD j "replay" | throw "replay is not a string"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  return match SecondStage.replayLinks p.prog key (getD j "boundary") with
    | some ls => Json.arr (ls.map SecondStage.Link.toJson)
    | none => .null

/-- `second_stage.replay_wires`' input to the design's wires (`SecondStage.replayWires`): a program object and the
replay block's descriptor id in it (`replay`); null for none. -/
def replayWires (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let .str key := getD j "replay" | throw "replay is not a string"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  return match SecondStage.replayWires p.prog key with
    | some ws => Json.arr (ws.map SecondStage.Wire.toJson)
    | none => .null

/-- `second_stage.replay_rows`' input to the rows the design's links' and wires' units read
(`SecondStage.replayRows`): a program object and the replay block's descriptor id in it (`replay`). -/
def replayRows (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let .str key := getD j "replay" | throw "replay is not a string"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  return Json.arr ((SecondStage.replayRows p.prog key).map SecondStage.Rows.toJson)

/-- `second_stage.replay_returns`' input to the names the replay block's return has in the boundary registration's
record (`SecondStage.replayReturns`): a program object, the replay block's descriptor id in it (`replay`) and the
boundary registration's record. -/
def replayReturns (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let .str key := getD j "replay" | throw "replay is not a string"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  return Json.arr ((SecondStage.replayReturns p.prog key (getD j "boundary")).map Json.str)

/-- `second_stage.replay_returned`'s input to the bits a replay block returns (`SecondStage.replayReturned`): a program
object and the replay block's descriptor id in it (`replay`); null for none. -/
def replayReturned (j : Json) : V Json := do
  let .obj _ := getD j "program" | throw "program is not an object"
  let .str key := getD j "replay" | throw "replay is not a string"
  let p ← Program.parse (← Flock.canon (getD j "program")).toUTF8
  return match SecondStage.replayReturned p.prog key with
    | some n => Lean.toJson n
    | none => .null

/-- `schedule.failure` and `schedule.at`'s input to the boundary record's schedule (`Schedule.failure`, `Schedule.at?`):
a record and replay blocks (`blocks`), to `{"failure", "at"}`: why round 2 refuses the record's schedule, null for
none, and each block's `[request, a, b]` under it, null where no run holds the block or the record commits no list of
runs. -/
def secondSchedule (j : Json) : V Json := do
  let .arr bs := getD j "blocks" | throw "blocks is not a list"
  let rec_ := getD j "record"
  let runs := Schedule.ofJson? (getD (getD rec_ "window") "schedule")
  let atJson (b : Json) : Json := match nat? b, runs with
    | some s, some rs => match Schedule.at? rs s with
      | some (r, a, e) => Json.arr #[.str r, Lean.toJson a, Lean.toJson e]
      | none => .null
    | _, _ => .null
  return Json.mkObj [("failure", (Schedule.failure rec_).elim Json.null Json.str), ("at", .arr (bs.map atJson))]

/-- `{"limit", "circuit", "public", "tables", "program", "partition", "registered", "draw"}` to `{"inputs", "draw"}`: what
of v2's inputs (`OneStageZk.fitsWhy`) and of a session's draw (`OneStageZk.drawFitsWhy`) a session's statement would hold
at `limit` bytes or more, each null for nothing. The files are hex, the tables `[name, hex]` pairs, the program hex or
null, and the partition object, the registered reads and the draw null for none. -/
def fits (j : Json) : V Json := do
  let some limit := nat? (getD j "limit") | throw "limit is not a natural number"
  let circuit ← Flock.unhex (← Flock.J.str j "circuit" "the lengths")
  let pub ← Flock.unhex (← Flock.J.str j "public" "the lengths")
  let .arr ts := getD j "tables" | throw "tables is not a list"
  let tables ← ts.toList.mapM fun
    | .arr #[.str n, .str b] => do pure (n, ← Flock.unhex b)
    | _ => throw "a table is not its name and its hex"
  let program ← match getD j "program" with
    | .null => pure none
    | .str b => some <$> Flock.unhex b
    | _ => throw "the program is not hex or null"
  let given (k : String) : Option Json := match getD j k with
    | .null => none
    | x => some x
  let why := fun (w : Option String) => w.elim Json.null Json.str
  return Json.mkObj [
    ("inputs", why (OneStageZk.fitsWhy limit circuit pub tables program (given "partition") (given "registered"))),
    ("draw", why (OneStageZk.drawFitsWhy limit (given "draw")))]

/-- `{"proofs", "query"}` to `{"circuits": [[template, circuit, header, registered], ...], "reads": [[template, V],
...]}`: a two-stage run's params naming each template's circuit file and its statements' reference header and
registered reads, in the order the check's template query names the templates (`Run.templateCircuits`), and the
templates they record as table reads, each with its table's rows (`Run.templateReads`); refused where the Python
raises. -/
def templateCircuits (j : Json) : V Json := do
  let named ← Run.templateCircuits (getD j "proofs") (getD j "query")
  let reads := Run.templateReads (getD j "proofs")
  return Json.mkObj [("circuits", Json.arr (named.map fun (t, c, h, r) => Json.arr #[Json.str t, Json.str c, h, r])),
    ("reads", Json.arr (reads.map fun (t, v) => Json.arr #[Json.str t, Lean.toJson v]))]

def table : List (String × (Json → V Json)) :=
  [("decode", decode), ("partition", partition), ("committed_bits", committedBits), ("unit_gates", unitGates),
   ("fanout", fanout), ("registration", registration),
   ("registered", registered),
   ("domain", domain), ("produced", produced), ("known_port", knownPort), ("check_draw", checkDraw), ("draw", draw), ("select", select), ("slots", slots),
   ("file_population", FilePart.piece), ("tables_held", OneStageZk.tablesPiece),
   ("check_proved", checkProved),
   ("select_second", selectSecond), ("second_statements", secondStatements),
   ("second_interior_reads", secondInteriorReads), ("replay_links", replayLinks),
   ("replay_wires", replayWires), ("replay_rows", replayRows), ("replay_returns", replayReturns),
   ("replay_returned", replayReturned), ("second_schedule", secondSchedule),
   ("bound", bound), ("harm_bound", harmBound), ("max_harm", maxHarm), ("bernoulli_bound", bernoulliBound),
   ("hidden_offsets", hiddenOffsets), ("hidden_locate", hiddenLocate), ("hidden_witness", hiddenWitness),
   ("stream", stream), ("fleet", fleet), ("verdict", verdict), ("matches_public", matchesPublic),
   ("session_registered", sessionRegistered), ("record", record), ("across", across),
   ("check_public", checkPublic),
   ("check_verdicts", checkVerdicts), ("run", run), ("template_circuits", templateCircuits), ("hello", Hello.check),
   ("fits", fits),
   ("challenger_commitment", Challenger.commitmentPiece),
   ("challenger_opening", Challenger.openingPiece), ("challenger_reason", Challenger.reasonPiece),
   ("challenger_order", Challenger.orderPiece), ("challenger_key", Challenger.keyPiece),
   ("challenger_record", Challenger.recordPiece), ("challenger_trusted", Challenger.trustedPiece),
   ("challenger_close", Challenger.closePiece), ("challenger_released", Challenger.releasedPiece),
   ("challenger_ledger", Challenger.ledgerPiece), ("challenger_day", Challenger.dayPiece),
   ("challenger_head", Challenger.headPiece),
   ("challenger_session", Challenger.sessionPiece), ("challenger_credited", Challenger.creditedPiece),
   ("challenger_pinned", Challenger.pinnedPiece), ("challenger_read", Challenger.readPiece),
   ("challenger_draw_coins", Challenger.drawCoinsPiece), ("challenger_within", Challenger.withinPiece),
   ("challenger_seed_salt", Challenger.seedSaltPiece),
   ("challenger_coin_trees", Challenger.coinTreesPiece), ("challenger_session_trees", Challenger.sessionTreesPiece),
   ("challenger_sample_link_first", Challenger.sampleLinkFirstPiece),
   ("challenger_inner_trees", Challenger.innerTreesPiece),
   ("wires", Wires.piece), ("wires_unrolled", Wires.pieceRef),
   ("wires_drawn", Wires.pieceDrawn), ("wires_drawn_unrolled", Wires.pieceDrawnRef),
   ("history_root", HistoryTable.rootPiece), ("history_check", HistoryTable.checkPiece),
   ("history_order", HistoryTable.orderPiece), ("history_strings", HistoryTable.stringsPiece),
   ("history_roots", HistoryTable.rootsPiece)]

/-- The verifier's pieces: `table`'s, by name. -/
def verifier : List (String × (Json → V Json)) :=
  table.filter fun p => ["registration", "matches_public", "select", "select_second", "check_draw", "check_proved",
    "record", "across", "check_public", "check_verdicts", "run", "hello", "stream", "fleet"].contains p.1

end Spot.Pieces
