import SampledProofs.OneStage

/-!
# Sampled proofs' one-stage audit: what its Lean verdict means (`SampledProofs.OneStage.check`)

The property of the executable audit (`verity/core/service/sampled_proofs/`, the port of `one_stage.audit.audit_record`
with `registration.check` before it), stated against `OneStage.check` as it runs, for every input of the types its
decoders produce. It takes no premise: the check is a function of its three inputs, and `AcceptMeans` says what it accepts.

`AcceptMeans`: what the check accepts is a commitment and evidence for which

* **the registration** keeps R1–R7 against the verifier's own copies (`Registered`): the record's format, protocol, law
  and window (`R1`); its program, query, partition and population (`R2`); its scheme, ports, roots, leaves and domains
  (`R3`); its leaf layer (`R4`), anchors (`R5`), previous record, window and the window's stream and prover (`R6`); and
  for a v2 record its registered values (`R7`);
* **the draw** is a draw of the registered law over the registered population (`Drawn`): its shape, its units and, for a
  stratified or work law, its strata and closure;
* **the units** were proved and served as drawn, under the registered roots, and the verifiers of record opened the
  registered values' reads against the registered roots (`UnitsProved`), and the receipt names the commitment's digest
  (`Receipt`);
* **the verdicts** accepted: one of record at least, every one of record ran, and none that ran refused
  (`VerdictsAccept`);
* the law has a profile's draw and the population is a natural number;

and it then publishes the drawn and proved units with the exact bounds at δ (`publicOf`), and the profile of one level
`unit`, the partition's units, drawn by the law at δ (`profileOf`). That the check accepts every such commitment and
evidence is completeness, `Properties.SampledProofs.acceptMeans_complete`.

The digest the receipt is held to is the commitment's own (`Commitment.digest`), which `OneStage.commitmentOf` computes
as the window record's digest of the record it decodes (`Flock.WindowRecord.digest`, `DigestIsRecord's`). A Bernoulli law is accepted as the code accepts
it: its draw is checked as any other, its profile's draw is `bernoulli p`, and it publishes the largest `M` of the
profile's units with `(1 − p)^M ≥ δ` (`BernoulliBounds`, `Bound.bernoulliBound`).
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

open OneStage (Params Commitment Evidence Public)

/-! ## The registration: R1–R7 against the verifier's own copies -/

/-- The verifier expects a v2 record: it expects values registered. -/
def ExpectsV2 (v : Registration.Verifier) : Prop := v.registered.isEmpty = false

/-- The partition object the verifier holds: its own, or the canonical one over its program's SHA-512 and its query. -/
def partitionOf (v : Registration.Verifier) : Json :=
  v.partition.getD (Partition.build (Flock.hexOf v.program.sha512) v.query)

/-- R1: the record has exactly a v1 record's keys, or a v2 record's when the verifier expects values registered; its format
is the matching one and its protocol one-stage's; its law is `law`, the verifier's agreed law as `Draw.derive` expands it
over the verifier's own program and query; and its window is an object of kind `served` or `stand-in`. -/
def R1 (rec : Json) (v : Registration.Verifier) (law : Json) : Prop :=
  keys? rec = some (if v.registered.isEmpty then Registration.KEYS else Registration.KEYS_V2) ∧
  jeq (getD rec "format") (.str (if v.registered.isEmpty then Registration.FORMAT else Registration.FORMAT_V2)) ∧
  jeq (getD rec "protocol") (.str Registration.PROTOCOL) ∧
  Draw.derive v.program v.query v.law v.workTable = .ok law ∧ jeq (getD rec "law") law ∧
  (getD rec "window" matches .obj _) ∧ ∃ k ∈ Registration.KINDS, jeq (getD (getD rec "window") "kind") (.str k)

/-- R2: the verifier's partition names its program's SHA-512, and the record names the same program; the record's query
is the SHA-512 of the partition's canonical query (whenever that query has a canonical form); the partition has a digest,
which the record names; and the record's population is the number of units the verifier's query derives from its own
program. -/
def R2 (rec : Json) (v : Registration.Verifier) : Prop :=
  jeq (getD (partitionOf v) "program") (.str (Flock.hexOf v.program.sha512)) ∧
  jeq (getD rec "program") (getD (partitionOf v) "program") ∧
  (∀ q, Registration.queryDigest (getD (partitionOf v) "query") = .ok q → jeq (getD rec "query") (.str q)) ∧
  (∃ d, Partition.digest (partitionOf v) = some d ∧ jeq (getD rec "partition") (.str d)) ∧
  ∃ n, Partition.units v.program v.query = .ok n ∧ jeq (getD rec "population") (Lean.toJson n)

/-- The record's roots: its `roots` list (none when it isn't a list). -/
def rootsOf (rec : Json) : Array Json :=
  match getD rec "roots" with
  | .arr rs => rs
  | _ => #[]

/-- The record's ports, each once, in the order of their first root. -/
def portsOf (rec : Json) : Array Json :=
  (rootsOf rec).foldl (fun acc r => if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]

/-- The record's root for port `p`: its last root naming `p`, as Python's dict reads it. -/
def rootFor (rec : Json) (p : String) : Option Json :=
  (rootsOf rec).reverse.find? fun r => jeq (getD r "port") (.str p)

/-- One registered root against the port the verifier's layout derives (`want`, its `{"leaves", "domain"}`): a hex root,
the port's leaves and its domain. -/
def RootIs (r want : Json) : Prop :=
  (∃ s, getD r "root" = .str s ∧ isHex s = true) ∧ jeq (getD r "leaves") (getD want "leaves") ∧
  jeq (getD r "domain") (getD want "domain")

/-- R3: the record's scheme is the verifier's; it registers at least one root, each exactly `{port, root, leaves,
domain}`; no port twice, and exactly the verifier's ports; and each of the verifier's ports has its root as derived. -/
def R3 (rec : Json) (v : Registration.Verifier) : Prop :=
  jeq (getD rec "scheme") (.str v.scheme) ∧
  rootsOf rec ≠ #[] ∧ (∀ r ∈ rootsOf rec, keys? r = some Registration.ROOT_KEYS) ∧
  (portsOf rec).size = (rootsOf rec).size ∧ (portsOf rec).size = v.ports.size ∧
  (∀ pw ∈ v.ports, ∃ q ∈ portsOf rec, jeq q (.str pw.1)) ∧
  ∀ pw ∈ v.ports, ∀ r, rootFor rec pw.1 = some r → RootIs r pw.2

/-- R4: the record names the verifier's leaf layer (the SHA-512 of its public file), or null when the verifier holds none. -/
def R4 (rec : Json) (v : Registration.Verifier) : Prop :=
  jeq (getD rec "leaf_layer") (match v.leafLayer with | some s => .str s | none => .null)

/-- R5: the record's anchors are the verifier's. -/
def R5 (rec : Json) (v : Registration.Verifier) : Prop := jeq (getD rec "anchors") (.arr v.anchors)

/-- R6: the record names the digest of the prover's last earlier record (null when the log is empty, or when that record
has no digest), no earlier record has its window, and a verifier bound to a stream and a prover (`binding`) sees its
window name both. -/
def R6 (rec : Json) (v : Registration.Verifier) : Prop :=
  jeq (getD rec "previous") (match v.log.back? with
    | some e => ((Registration.digestHex e).toOption.map Json.str).getD .null
    | none => .null) ∧
  (∀ e ∈ v.log, jeq (getD e "window") (getD rec "window") = false) ∧
  ∀ sp, v.binding = some sp → jeq (getD (getD rec "window") "stream") (.str sp.1) = true ∧
    jeq (getD (getD rec "window") "prover") (.str sp.2) = true

/-- A v2 record's registered values: the string values, each once (`names`), as many as the list's entries and as the
verifier's own entries, and each one of the verifier's. -/
def namesOf (es : Array Json) : Array String :=
  es.filterMap fun e => match getD e "value" with
    | .str s => some s
    | _ => none

def ValuesAre (es : Array Json) (mine : Array Registered.Entry) : Prop :=
  (namesOf es).size = es.size ∧ (namesOf es).toList.eraseDups.length = es.size ∧
  (namesOf es).toList.eraseDups.length = mine.size ∧ ∀ s ∈ (namesOf es).toList.eraseDups, ∃ m ∈ mine, m.value = s

/-- One registered value's entry against the verifier's own (`want`) and the roots it is held to (`held`): a 128-digit hex
root, the verifier's `leaves`, `words` and domain, its width (`word_bits` left out for 16 bits, else written and not 16),
and the first root `held` names for that value, when it names one. -/
def EntryIs (e : Json) (want : Registered.Entry) (held : Array (String × String)) : Prop :=
  (∃ s, getD e "root" = .str s ∧ s.length = 128 ∧ isHex s = true) ∧
  int? (getD e "leaves") = some (want.leaves : Int) ∧ int? (getD e "words") = some (want.words : Int) ∧
  jeq (getD e "domain") (.str want.domain) ∧
  (match field? e Registered.WIDTH_KEY with
    | none => want.wordBits = Registered.WORD_BITS
    | some j => int? j = some (want.wordBits : Int) ∧ (want.wordBits : Int) ≠ Registered.WORD_BITS) ∧
  ∀ x ∈ held.find? (·.1 == want.value), jeq (getD e "root") (.str x.2)

/-- The roots a registered value's entry is held to: none for a value the verifier fills itself (`supplied`), whose root
is the one it recomputes (`SuppliedIs`), and `held` for every other. -/
def heldFor (supplied : Array (String × Supplied.Rows)) (held : Array (String × String)) (value : String) :
    Array (String × String) :=
  if supplied.any (·.1 == value) then #[] else held

/-- R7-supplied: each value the verifier fills itself is one of its own entries (`mine`), and the record's entry for it
(the last naming it) has the root `rootOf` recomputes for it from its rows, which must compute. -/
def SuppliedIs (es : Array Json) (mine : Array Registered.Entry) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → Flock.V String) : Prop :=
  ∀ x ∈ supplied, ∃ want, mine.find? (·.value == x.1) = some want ∧
    ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str x.1)) = some e →
      ∃ r, rootOf x.1 x.2 want = .ok r ∧ jeq (getD e "root") (.str r)

/-- An output value's earlier port: the log's record with digest `record` has a `roots` entry for `port`
(`Registration.outputPort`) with `leaves` leaves, and a string domain and root. -/
def OutputPortIs (log : Array Json) (leaves : Nat) (record port : String) : Prop :=
  ∃ r, Registration.outputPort log record port = some r ∧ int? (getD r "leaves") = some (leaves : Int) ∧
    (∃ dom, getD r "domain" = .str dom) ∧ ∃ root, getD r "root" = .str root

/-- An output value's earlier registered value: the log's record with digest `record` has a `registered` entry for
`value` (`Registration.outputValue`) with `leaves` leaves, `words` words and `bits` word bits (`Registration.widthOf`:
16 when it leaves them out), and a string domain and root. -/
def OutputValueIs (log : Array Json) (leaves words bits : Nat) (record value : String) : Prop :=
  ∃ r, Registration.outputValue log record value = some r ∧
    (int? (getD r "leaves") = some (leaves : Int) ∧ int? (getD r "words") = some (words : Int) ∧
      Registration.widthOf r = some (bits : Int)) ∧
    (∃ dom, getD r "domain" = .str dom) ∧ ∃ root, getD r "root" = .str root

/-- What an output value takes from its earlier record, for a value of `leaves` leaves, `words` words and `bits` word
bits: its port (`OutputPortIs`) or its registered value (`OutputValueIs`). -/
def EarlierIs (log : Array Json) (leaves words bits : Nat) (record : String) : Registration.Earlier → Prop
  | .port p => OutputPortIs log leaves record p
  | .value u => OutputValueIs log leaves words bits record u

/-- R7, for a v2 record (`Registration.checkOutputs`): the verifier derives its own entries for the values it expects
registered (`Registration.ownEntries`: `Registered.entries`, counting each value's rows in the row format it holds for
it, `v.schemas`, with each value's domain under its window's open root but the values it holds across windows); each
output value it has an entry for has its earlier port, with the value's leaf count, or its earlier registered value,
with the value's leaves, words and word bits (`EarlierIs`), and takes that port's or entry's domain and root
(`Registration.applyOutputs`: `mine'` and `held'`); the
record's `registered` is a list of entries, each exactly `{value, root, leaves, words, domain}` with `word_bits` beside
them or not; its values are exactly the verifier's (`ValuesAre`); each of the verifier's values has its entry (the last
naming it, as Python's dict reads it) as derived and held (`EntryIs`), a value it doesn't fill held to its output's root,
else to the root it pinned before the record, else to the root the log first registered it under with the domain it
takes now (`Registration.heldRoots`), and one it fills to none (`heldFor`); and each value it fills has the root it
recomputes from the record, the log and its binding (`SuppliedIs`, `Supplied.root`). -/
def R7 (rec : Json) (v : Registration.Verifier) : Prop :=
  ∃ mine es mine' held', Registration.ownEntries v = .ok mine ∧
    getD rec "registered" = .arr es ∧
    (∀ o ∈ v.outputs, ∀ want, mine.find? (·.value == o.1) = some want →
      EarlierIs v.log want.leaves want.words want.wordBits o.2.1 o.2.2) ∧
    (Registration.applyOutputs #[] mine (v.pinned ++ Registration.heldRoots v.log (Registration.ownEntries v))
      v.outputs v.log).2 = (mine', held') ∧
    (∀ e ∈ es, keys? e = some Registered.ENTRY_KEYS ∨
      keys? e = some (sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY))) ∧
    ValuesAre es mine' ∧
    (∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
      EntryIs e want (heldFor v.supplied held' want.value)) ∧
    SuppliedIs es mine' v.supplied
      (Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1)))

/-- The registration keeps R1–R7 against the verifier's own copies, with `law` the agreed law as derived; R7 when the
verifier expects values registered. -/
def Registered (v : Registration.Verifier) (rec law : Json) : Prop :=
  R1 rec v law ∧ R2 rec v ∧ R3 rec v ∧ R4 rec v ∧ R5 rec v ∧ R6 rec v ∧ (ExpectsV2 v → R7 rec v)

/-! ## The draw: the registered law over the registered population -/

/-- The law a draw names, as a string (empty when it isn't one). -/
def kindOf (draw : Json) : String :=
  match getD draw "law" with
  | .str s => s
  | _ => ""

/-- A draw's keys: its law, population and units, with `k` (subset), `p` (bernoulli) or `k` and `strata` (stratified,
work), and a work draw's `closure` when it has one. -/
def drawKeys (kind : String) (closure : Bool) : Array String :=
  let keys := sortStrings (#["law", "population", "units"] ++
    if kind == "subset" then #["k"] else if kind == "bernoulli" then #["p"] else #["k", "strata"])
  if closure then sortStrings (keys.push "closure") else keys

/-- Strictly ascending. -/
def Ascending (xs : Array Int) : Prop := ∀ i (h : i + 1 < xs.size), xs[i] < xs[i + 1]

/-- The draw's units of stratum `s`: those in one of its ranges. -/
def countIn (units : Array Int) (s : Json) : Int :=
  let rs := (Draw.ranges? (getD s "units")).getD #[]
  units.foldl (fun acc u => if rs.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0

/-- A draw of the law `law` over the registered `population` (`draw.check_draw`): an object naming subset, bernoulli,
stratified or work, with exactly its law's keys; the law it names is `law` and its population the registered one; its
units integers, strictly ascending, in `[0, population)`; a subset's `k` its number of units; a Bernoulli `p` written
`NUM/DEN` with `0 ≤ NUM ≤ DEN`, `0 < DEN`; a stratified or work draw's strata well formed over its population
(`Draw.strataMessage` none: each stratum exactly its keys, a distinct string template, its ranges ascending, disjoint and
tiling `[0, population)`, its `k` the rule's), and each stratum's `k`, where its template is a string (as
`Draw.strataMessage` none has every one), the number of units drawn in it; and a work draw's closure, when it has one,
integers, non-empty, strictly ascending, in `[0, population)` and none of them drawn. -/
def Drawn (draw population law : Json) : Prop :=
  let kind := kindOf draw
  let closure := kind == "work" && (field? draw "closure").isSome
  (draw matches .obj _) ∧ kind ∈ ["subset", "bernoulli", "stratified", "work"] ∧
  keys? draw = some (drawKeys kind closure) ∧
  jeq (Draw.lawOf draw) law ∧ jeq (getD draw "population") population ∧
  ∃ us units, getD draw "units" = .arr us ∧ us.mapM int? = some units ∧ Ascending units ∧
    (∀ (h : 0 < units.size), 0 ≤ units[0] ∧ ∃ n, int? (getD draw "population") = some n ∧ units[units.size - 1] < n) ∧
    (kind = "subset" → int? (getD draw "k") = some (units.size : Int)) ∧
    (kind = "bernoulli" → ∃ p, getD draw "p" = .str p ∧ (Draw.probability? p).isSome) ∧
    ((kind = "stratified" ∨ kind = "work") →
      ∃ n, int? (getD draw "population") = some n ∧ Draw.strataMessage draw n = none ∧
        ∀ ss, getD draw "strata" = .arr ss → ∀ s ∈ ss, ∀ t, getD s "template" = .str t →
          int? (getD s "k") = some (countIn units s)) ∧
    (closure → ∃ cs ys, getD draw "closure" = .arr cs ∧ cs.mapM int? = some ys ∧ ∃ (h : 0 < ys.size),
      Ascending ys ∧ 0 ≤ ys[0] ∧ (∃ n, int? (getD draw "population") = some n ∧ ys[ys.size - 1] < n) ∧
      ∀ y ∈ ys, y ∉ units)

/-! ## The units, the receipt and the verdicts -/

/-- The drawn units, as integers. -/
def drawnInts (draw : Json) : Array Int :=
  match getD draw "units" with
  | .arr us => us.filterMap int?
  | _ => #[]

/-- The receipt names the commitment's digest, in hex. -/
def Receipt (c : Commitment) (e : Evidence) : Prop :=
  jeq (getD e.receipt "registration") (.str (Flock.hexOf c.digest))

/-- The roots the verifiers of record that accepted report opening the sessions' registered reads against, for the
record's own values (its roots' ports and its registered values): `{value: root}`, a value reported under two roots mapped
to null; none when no such verdict reports any (`audit.session_registered`, which `Record.sessionRegistered` and
`Record.ownRegistered` compute from the verdicts). -/
def openedRoots (rec : Json) (vs : Array (String × Json)) : Option (Array (String × Json)) :=
  Record.ownRegistered rec (Record.sessionRegistered vs)

/-- Two `{value: root}` dicts with distinct keys are equal (Python's `==`): the same size, and each entry of the first has
the same key and root in the second. -/
def SameRoots (a b : Array (String × Json)) : Prop :=
  a.size = b.size ∧ ∀ x ∈ a, ∃ y ∈ b, x.1 = y.1 ∧ jeq x.2 y.2

/-- The units, as the record holds them (`audit_record`'s checks after the draw's): the proved statement's instances are
exactly the units the draw proves (its units and a work draw's closure, ascending); the receipt names the commitment's
digest; the sessions were served exactly the drawn units, when they were handed the draw; the sessions' roots, when
given, name at least one port, each port's root the one the record registers for it (`Record.byPort`, the last root of
each port); and, when the record registers values or the verifiers of record report reads of the record's own values,
those reads were opened against exactly the record's `{value: root}` (`Record.registeredRoots`, each value's last
entry). -/
def UnitsProved (c : Commitment) (e : Evidence) : Prop :=
  e.instances = (Draw.proved e.draw).size ∧ e.provedUnits.map (fun n : Nat => (n : Int)) = Draw.proved e.draw ∧
  Receipt c e ∧
  (∀ served, e.served = some served →
    (served.map fun n : Nat => (n : Int)).qsort (· < ·) = (drawnInts e.draw).qsort (· < ·)) ∧
  (∀ roots, e.sessionRoots = some roots → entries roots ≠ #[] ∧
    ∀ pr ∈ entries roots, ∃ reg, (Record.byPort c.record).find? (·.1 == pr.1) = some reg ∧ jeq (getD reg.2 "root") pr.2) ∧
  ((field? c.record "registered").isSome ∨ (openedRoots c.record e.verdicts).isSome →
    SameRoots ((openedRoots c.record e.verdicts).getD #[]) (Record.registeredRoots c.record))

/-- The verdicts accept (`verdict` of `audit_record`): every verdict that ran (`accepted` not null) accepted; at least
one is of record (`of_record` truthy); and every verdict of record ran. -/
def VerdictsAccept (vs : Array (String × Json)) : Prop :=
  (∀ x ∈ vs, getD x.2 "accepted" ≠ .null → jeq (getD x.2 "accepted") (.bool true)) ∧
  (∃ x ∈ vs, truthy (getD x.2 "of_record")) ∧
  ∀ x ∈ vs, truthy (getD x.2 "of_record") → getD x.2 "accepted" ≠ .null

/-! ## What acceptance publishes -/

/-- The drawn units as natural numbers. -/
def drawnNats (draw : Json) : Array Nat :=
  match getD draw "units" with
  | .arr us => us.filterMap nat?
  | _ => #[]

/-- The public outputs at δ, for the law `law` with its profile's draw `d` over `n` units: the drawn units, the proved
ones, the exact bound on wrong units (`Bound.boundOf`: `Bound.countBound` over the draw's strata, `Bound.bernoulliBound`
for a Bernoulli draw), and for a stratified or work law each template's own (`Bound.stratumBound`). -/
def publicOf (p : Params) (e : Evidence) (law : Json) (d : Core.Draw) (n : Nat) : Public :=
  { drawn := drawnNats e.draw,
    proved := (Draw.proved e.draw).filterMap fun (i : Int) => if 0 ≤ i then some i.toNat else none,
    bound := Bound.boundOf n d p.delta,
    perTemplate := if jeq (getD law "law") "stratified" || jeq (getD law "law") "work" then
      (Bound.strataOf n d).map (·.map fun s => (s.name, Bound.stratumBound s p.delta)) else none }

/-- The integrity profile at δ: the record's program; one level `unit`, the record's partition, of `n` units, drawn by
`d`; and the number of units drawn. -/
def profileOf (p : Params) (c : Commitment) (e : Evidence) (d : Core.Draw) (n : Nat) : Core.Integrity :=
  let str := fun (k : String) => match getD c.record k with | .str s => s | _ => ""
  { programs := #[str "program"], prescribed := #[], linked := false, covered := #[],
    levels := #[{ name := Record.LEVEL, query := some (str "partition"), size := n, draw := some d }],
    delta := p.delta, drawn := #[(Record.LEVEL, (drawnNats e.draw).size)] }

/-- What the check accepts: the registration keeps R1–R7 with `law` the verifier's agreed law as derived, the draw is a
draw of `law` over the registered population, the units were proved and served as drawn under the registered roots and
values with the receipt naming the commitment's digest, and the verdicts accept; `law` has a profile's draw `d`
(`Record.profileDraw`) and the registered population is a natural number `n`, and the check publishes `publicOf` and
the profile `profileOf`. -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    OneStage.check p c e = .accept pub prof →
      ∃ law d n, Registered p.verifier c.record law ∧ Drawn e.draw (getD c.record "population") law ∧
        UnitsProved c e ∧ VerdictsAccept e.verdicts ∧
        Record.profileDraw law = some d ∧ nat? (getD c.record "population") = some n ∧
        pub = publicOf p e law d n ∧ prof = profileOf p c e d n

/-- The commitment's digest, which the receipt is held to, is the window record's digest of the record it decodes:
`SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag. -/
def DigestIsRecord's : Prop :=
  ∀ (b : ByteArray) (c : Commitment), OneStage.commitmentOf b = .ok c → Flock.WindowRecord.digest c.record = .ok c.digest

/-- An acceptance whose profile draws a level by a Bernoulli law at `r` bounds wrong units by `Bound.bernoulliBound` over
the level's units at δ: the largest `M` of them with `(1 − r)^M ≥ δ`. -/
def BernoulliBounds : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    OneStage.check p c e = .accept pub prof →
      ∀ (r : Core.Ratio) (l : Core.Level), l ∈ prof.levels → l.draw = some (.bernoulli r) →
        pub.bound = Bound.bernoulliBound l.size r p.delta

end Properties.SampledProofs
