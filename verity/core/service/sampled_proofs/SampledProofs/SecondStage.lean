/-
**Round 2's statement check**: a two-stage run's second-stage sessions held, each by its statement, to the interior
registration and to the second draw's cells (`check`). `Run.run` adds it to the interior registration's audit record as
`checks.statements` (`attach`), and `service.Stream` runs its mirror (`second_stage`), held to it by difftest
(`second_statements`).

* **The parts and their cells** are the verifier's own: one part per stratum of the interior program under the check's
  query (`Partition.strata`, which `population-order` holds to the population's units, template by template), each one
  range of `Q·n_t` slots from `off_t`, `Q` the replay sample's size (`parts`). Template `t`'s cell in the sample's
  `i`-th replay block `S[i]` is its `n_t` slots from `off_t + i·n_t` (`Part.span`). One template's cells are
  `Select.secondLaw`'s strata.
* **A statement** (`Stmt`) is one second-stage session's, as its files state it: the template its circuit is an
  instance of (its stratum's name), its public file's header, its session record's sample (`unit_draw`), and the
  registered reads it was verified against (`--registered`), null for none.
* **Its checks**, each a code (`stmtCodes`, `stmtFailures`):
  - `second-statement`: it is not one;
  - `second-template`: its template is none of the parts', or an earlier statement's;
  - `second-population`: its header's `units.indices` are not its part's slots in order, or its `instances` not their
    count;
  - `second-roots`: its scheme is not the interior record's, or its `frame_v3` roots and `domain_ids`, its registered
    reads aside, are not the record's for its template's ports (port `p` of template `t` is the record's `t/p` when the
    record names its ports by template, else `p`), each of which it roots or reads;
  - `second-sample`: its sample is not the verifier's own, the second draw's units in its part as a subset of its slots
    (`ownDraw`);
  - `second-read`: a port's registered read (`readOk`) is none of: its own port of the interior record, each slot `u`
    reading only its own rows (`[(u - off_t)·r, (u - off_t + 1)·r)`, `r` the port's leaves over the part's `Q·n_t`
    slots); the boundary port its link names (`Link`, the verifier's own), each slot reading only its cell's replay
    block's rows (`[S[i]·r, (S[i] + 1)·r)`, `r` the port's leaves over the boundary population); a value either
    registration registers. Each is read at its root, domain and leaves.
* **Over the statements**: `second-cells` when the interior program's strata make no parts, `second-statements` when
  no statements are given, `second-cover` for each part the second draw samples that no statement is for, and
  `second-reported` when the `{value: root}` the verifiers of record report the sessions read (`Record.sessionRegistered`)
  is not the statements' (`statementReads`). A passing check (`check_ok`) holds every session's sample to the second
  draw, cell by cell, and every read it reports to a statement's.
* **The form**: once every check passes, the run is refused last (`two-stage-form`) unless `FORM` allows it; while it
  doesn't, no run passes (`check_refused`).
-/
module

public import SampledProofs.Record
meta import SampledProofs.Record

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace SampledProofs.SecondStage

open Lean (Json)

/-- **The switch**: whether the verifier accepts a two-stage run whose every check of round 2 passes. Off, it refuses
each one last (`two-stage-form`): the two-stage guarantee (`Flock.SecurityProofs.SampledKSTwo`) takes its sessions'
openings (`SessionOpensTwo`) as a fact of the code, and SPOT[Flock] derives that fact only from its layout's `Embeds` and
`FrameIs`, both open. -/
def FORM : Bool := false

def FORM_WHY : String :=
  "refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the soundness " ++
  "proof does not cover"

def CELLS_WHY : String :=
  "the interior program's strata are not one range each, of a positive multiple of the replay sample's size"

def STATEMENTS_WHY : String :=
  "a two-stage outcome names its second-stage sessions' statements: each one's template, header, draw and registered reads"

def REPORTED_WHY : String :=
  "the registered reads the verifiers of record report are not the statements' registered reads, value by value"

/-- A template's part of the interior population: `Q·n` slots from `lo`, `n` in each sampled replay block. -/
structure Part where
  template : String
  lo : Nat
  n : Nat
  deriving Repr, Inhabited, DecidableEq

/-- Slot `u` is one of part `p`'s, in a sample of `q` replay blocks. -/
def Part.holds (q : Nat) (p : Part) (u : Nat) : Bool := decide (p.lo ≤ u) && decide (u < p.lo + q * p.n)

/-- Slot `u`'s place in the sample: the index of its cell's replay block. -/
def Part.span (p : Part) (u : Nat) : Nat := (u - p.lo) / p.n

/-- A stratum's part in a sample of `q` replay blocks: its one range, of a positive multiple of `q` slots. -/
def partOf (q : Nat) (s : String × Array (Nat × Nat)) : Option Part :=
  match s.2.toList with
  | [(lo, hi)] => if 0 < q ∧ lo < hi ∧ (hi - lo) % q = 0 then some ⟨s.1, lo, (hi - lo) / q⟩ else none
  | _ => none

/-- **The parts**: one per stratum, none when a stratum is not such a range. -/
def parts (q : Nat) (strata : Array (String × Array (Nat × Nat))) : Option (Array Part) := strata.mapM (partOf q)

/-- One second-stage session's statement (module docstring). -/
structure Stmt where
  template : String
  header : Json
  draw : Json
  registered : Json

def STMT_KEYS : Array String := #["draw", "header", "registered", "template"]

/-- A statement from its JSON: exactly `STMT_KEYS`, a string, two objects and an object or null. -/
def Stmt.ofJson? (j : Json) : Option Stmt :=
  if keys? j != some STMT_KEYS then none else
  match getD j "template", getD j "header", getD j "draw", getD j "registered" with
  | .str t, .obj h, .obj d, .null => some ⟨t, .obj h, .obj d, .null⟩
  | .str t, .obj h, .obj d, .obj r => some ⟨t, .obj h, .obj d, .obj r⟩
  | _, _, _, _ => none

/-- A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
`value`. -/
structure Link where
  template : String
  port : String
  value : String
  deriving Repr, Inhabited, DecidableEq

/-- The design's links from their JSON: none (null), or a list of objects of exactly `template`, `port` and `value`,
each a string. -/
def linksOf? : Json → Option (Array Link)
  | .null => some #[]
  | .arr ls => ls.mapM fun l =>
    if keys? l != some #["port", "template", "value"] then none else
    match getD l "template", getD l "port", getD l "value" with
    | .str t, .str p, .str v => some ⟨t, p, v⟩
    | _, _, _ => none
  | _ => none

/-- What the verifier holds of the run: the replay sample `S`, the parts (`parts`), the interior and boundary
registrations' records, the second draw's units, the design's links, and the `{value: root}` the verifiers of record
report the sessions read (`Record.sessionRegistered`). -/
structure Own where
  sample : Array Nat
  parts : Option (Array Part)
  interior : Json
  boundary : Json
  drawn : Array Nat
  links : Array Link := #[]
  reported : Option (Array (String × Json)) := none

/-- The record names its ports by template, `<template>/<port>`. -/
def byTemplate (rec : Json) : Bool := (Record.byPort rec).any fun (q, _) => q.contains '/'

/-- The interior record's port for port `p` of template `t`'s circuit. -/
def portOf (named : Bool) (t p : String) : String := if named then t ++ "/" ++ p else p

/-- The interior record's ports of template `t`: those under `t/` when it names its ports by template, else all. -/
def templatePorts (named : Bool) (t : String) (rec : Json) : Array String :=
  ((Record.byPort rec).map (·.1)).filter fun q => !named || q.startsWith (t ++ "/")

/-- The header's `units.indices` are part `p`'s slots in order, and its `instances` their count. -/
def populationOk (q : Nat) (p : Part) (h : Json) : Bool :=
  match getD (getD h "units") "indices", nat? (getD h "instances") with
  | .arr is, some m => m == q * p.n && is.toList.map nat? == (List.range (q * p.n)).map fun i => some (p.lo + i)
  | _, _ => false

/-- The header's scheme is the interior record's, its `frame_v3` roots outside the ports `reads` reads are the record's
for template `t`'s ports, each at the record's root, its `domain_ids` entry the record's domain, and each of those ports
is one it roots or reads. -/
def rootsOk (named : Bool) (rec : Json) (t : String) (h : Json) (reads : Array String) : Bool :=
  let fv := getD h "frame_v3"
  let roots := (entries (getD fv "roots")).filter fun (p, _) => !reads.contains p
  let reg := Record.byPort rec
  jeq (getD h "scheme") (getD rec "scheme") &&
    roots.all (fun (p, r) => match reg.find? (·.1 == portOf named t p) with
      | some (_, e) => jeq r (getD e "root") && jeq (getD (getD fv "domain_ids") p) (getD e "domain")
      | none => false) &&
    (templatePorts named t rec).all fun x =>
      roots.any (fun (p, _) => portOf named t p == x) || reads.any fun p => portOf named t p == x

/-- **The verifier's own sample** for part `p`'s session: the second draw's units among its slots, counted from its
first, as `unit_draw` writes a subset of its population. -/
def ownDraw (q : Nat) (p : Part) (drawn : Array Nat) : Json :=
  let units := (drawn.filter (p.holds q)).map (· - p.lo)
  Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson (q * p.n)), ("k", Lean.toJson units.size),
    ("units", natsJson units)]

/-- Value `v`'s entry among registration `rec`'s registered values. -/
def registeredEntry (rec : Json) (v : String) : Option Json :=
  (match getD rec "registered" with | .arr es => es | _ => #[]).find? fun e => jsonStrIs (getD e "value") v

/-- Entries `e` and `x` name one root, domain and leaf count. -/
def sameAt (e x : Json) : Bool :=
  jeq (getD e "root") (getD x "root") && jeq (getD e "domain") (getD x "domain") && jeq (getD e "leaves") (getD x "leaves")

/-- Every read of registered-reads entry `e` (slot `reads_from + k` reads row `reads[k]`, null for none) is by one of
part `p`'s slots `u`, of a row of the unit `unit u` names, in a tree of `units` units' rows, `leaves / units` each. -/
def rowsOk (q : Nat) (p : Part) (e : Json) (units : Nat) (unit : Nat → Option Nat) : Bool :=
  let first : Option Nat := match getD e "reads_from" with | .null => some 0 | x => nat? x
  match nat? (getD e "leaves"), getD e "reads", first with
  | some leaves, .arr rs, some f =>
    decide (0 < units) && leaves % units == 0 && (List.range rs.size).all fun k =>
      match rs.getD k .null with
      | .null => true
      | x => match nat? x, unit (f + k) with
        | some pos, some b =>
          p.holds q (f + k) && decide (b * (leaves / units) ≤ pos) && decide (pos < (b + 1) * (leaves / units))
        | _, _ => false
  | _, _, _ => false

/-- Template `p.template`'s session port `port` may read registered-reads entry `e`: its own port of the interior
record, each slot only its own rows; else the boundary port a link names for it, each slot only its cell's replay
block's rows; else a value either registration registers. Each at its root, domain and leaves. -/
def readOk (own : Own) (named : Bool) (p : Part) (port : String) (e : Json) : Bool :=
  let q := own.sample.size
  match getD e "value" with
  | .str v =>
    match (Record.byPort own.interior).find? (·.1 == v), (Record.byPort own.boundary).find? (·.1 == v) with
    | some (_, x), _ => v == portOf named p.template port && sameAt e x && rowsOk q p e (q * p.n) fun u => some (u - p.lo)
    | none, some (_, b) =>
      own.links.any (fun l => l.template == p.template && l.port == port && l.value == v) && sameAt e b &&
        match nat? (getD own.boundary "population") with
        | some m => rowsOk q p e m fun u => own.sample[p.span u]?
        | none => false
    | none, none => match registeredEntry own.boundary v <|> registeredEntry own.interior v with
      | some x => sameAt e x
      | none => false
  | _ => false

/-- Statement `s`'s failing checks against its part `p` (module docstring), in order. -/
def stmtCodes (own : Own) (named : Bool) (p : Part) (s : Stmt) : List String :=
  let q := own.sample.size
  let reads := entries (getD s.registered "ports")
  [("second-population", populationOk q p s.header),
   ("second-roots", rootsOk named own.interior s.template s.header (reads.map (·.1))),
   ("second-sample", jeq s.draw (ownDraw q p own.drawn)),
   ("second-read", reads.all fun (port, e) => readOk own named p port e)].filterMap fun (code, ok) =>
    if ok then none else some code

def detailOf (i : Nat) (t : Json) : Json := Json.mkObj [("statement", Lean.toJson i), ("template", t)]

/-- The templates statements `ss` are for, those that are statements. -/
def templates (ss : List Json) : List String := ss.filterMap fun j => (Stmt.ofJson? j).map (·.template)

/-- Statement `i` of `ss`'s failures, `(code, detail)` each: `second-statement` for one that is not a statement,
`second-template` for one for no part's template or an earlier statement's, else its failing checks. -/
def stmtFailures (own : Own) (ps : Array Part) (ss : List Json) (i : Nat) : List (String × Json) :=
  match Stmt.ofJson? (ss.getD i .null) with
  | none => [("second-statement", detailOf i .null)]
  | some s =>
    match ps.find? (·.template == s.template) with
    | some p =>
      if (templates (ss.take i)).contains s.template then [("second-template", detailOf i (.str s.template))]
      else (stmtCodes own (byTemplate own.interior) p s).map fun c => (c, detailOf i (.str s.template))
    | none => [("second-template", detailOf i (.str s.template))]

/-- `second-cover` for each part the second draw samples that no statement is for. -/
def coverFailures (own : Own) (ps : Array Part) (ss : List Json) : List (String × Json) :=
  ps.toList.filterMap fun p =>
    if !(templates ss).contains p.template && own.drawn.any (p.holds own.sample.size) then
      some ("second-cover", Json.mkObj [("template", Json.str p.template)])
    else none

/-- The `{value: root}` statements `ss` read, merged as `Record.sessionRegistered` merges the verdicts': a value read
under two roots maps to null. -/
def statementReads (ss : List Json) : Array (String × Json) :=
  ss.foldl (fun acc j => match Stmt.ofJson? j with
    | some s => (entries (getD s.registered "ports")).foldl (fun acc (_, e) => match getD e "value" with
      | .str v =>
        let root := getD e "root"
        match acc.find? (·.1 == v) with
        | some (_, r) => acc.map fun (k, x) => if k == v then (k, if jeq r root then r else .null) else (k, x)
        | none => acc.push (v, root)
      | _ => acc) acc
    | none => acc) #[]

/-- `second-reported` when the reads the verifiers of record report are not those statements `ss` state. -/
def reportedFailures (own : Own) (ss : List Json) : List (String × Json) :=
  if Record.sameDict (own.reported.getD #[]) (statementReads ss) then [] else [("second-reported", Json.str REPORTED_WHY)]

/-- Every failure of statements `ss` over parts `ps`, then, when there is none and the form isn't allowed, its
refusal. -/
def failures (allow : Bool) (own : Own) (ps : Array Part) (ss : List Json) : List (String × Json) :=
  let found := (List.range ss.length).flatMap (stmtFailures own ps ss) ++ coverFailures own ps ss ++
    reportedFailures own ss
  if found.isEmpty && !allow then [("two-stage-form", Json.str FORM_WHY)] else found

/-- **Round 2's statement check** (module docstring) of statements `stmts`, the two-stage form allowed when `allow`
(`FORM` in the run): each code once, with the detail it first failed with. -/
def check (allow : Bool) (own : Own) (stmts : Option (Array Json)) : Check :=
  let found := match own.parts, stmts with
    | none, _ => [("second-cells", Json.str CELLS_WHY)]
    | some _, none => [("second-statements", Json.str STATEMENTS_WHY)]
    | some ps, some ss => failures allow own ps ss.toList
  found.foldl (fun c x => c.fail x.1 x.2) {}

/-- The interior registration's audit record `rec` with round 2's check `c` as `checks.statements`, failing where `c`
does as a failed check of its own fails it (`Record.record`): not accepted, and no profile, integrity profile or wrong
units. -/
def attach (rec : Json) (c : Check) : Json :=
  let r := rec.setObjVal! "checks" ((getD rec "checks").setObjVal! "statements" c.toJson)
  if c.codes.isEmpty then r
  else (((r.setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal! "integrity_profile" .null).setObjVal!
    "wrong_units" .null

theorem mem_fail (c : Check) (code : String) (d : Json) (x : String) :
    x ∈ (c.fail code d).codes ↔ x ∈ c.codes ∨ x = code := by
  unfold Check.fail
  split
  · rename_i h
    constructor
    · exact Or.inl
    · rintro (hx | rfl)
      · exact hx
      · simpa using h
  · simp

theorem mem_fold (l : List (String × Json)) (c : Check) (x : String) :
    x ∈ (l.foldl (fun c x => c.fail x.1 x.2) c).codes ↔ x ∈ c.codes ∨ ∃ d, (x, d) ∈ l := by
  induction l generalizing c with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons, ih, mem_fail]
    constructor
    · rintro ((h | rfl) | ⟨d, hd⟩)
      · exact Or.inl h
      · exact Or.inr ⟨a.2, by simp⟩
      · exact Or.inr ⟨d, by simp [hd]⟩
    · rintro (h | ⟨d, hd⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hd with rfl | hd
        · exact Or.inl (Or.inr rfl)
        · exact Or.inr ⟨d, hd⟩

/-- A fold of `Check.fail` has no code only over no failure. -/
theorem fold_codes_empty (l : List (String × Json)) :
    (l.foldl (fun c x => c.fail x.1 x.2) ({} : Check)).codes = #[] ↔ l = [] := by
  constructor
  · intro h
    cases l with
    | nil => rfl
    | cons a l =>
      have hm := (mem_fold (a :: l) {} a.1).2 (Or.inr ⟨a.2, by simp⟩)
      rw [h] at hm
      simp at hm
  · rintro rfl
    rfl

theorem failures_ne_nil (own : Own) (ps : Array Part) (ss : List Json) : failures false own ps ss ≠ [] := by
  simp only [failures]
  split
  · simp
  · rename_i h
    intro he
    simp_all

/-- **Off, round 2 never passes**: with the two-stage form refused, every run's check fails. -/
theorem check_refused (own : Own) (stmts : Option (Array Json)) : (check false own stmts).codes ≠ #[] := by
  simp only [check, Ne, fold_codes_empty]
  split <;> simp [failures_ne_nil]

/-- **What a passing check says**: the form is allowed, the interior program's strata make parts, each statement is one
for a part's template, no earlier statement's, and passes every check (`stmtCodes_nil`), each part the second draw
samples has a statement, and the reads the verifiers of record report are the statements'. -/
theorem check_ok {allow : Bool} {own : Own} {ss : Array Json} (h : (check allow own (some ss)).codes = #[]) :
    allow = true ∧ ∃ ps, own.parts = some ps ∧
      (∀ i < ss.size, ∃ s p, Stmt.ofJson? (ss.toList.getD i .null) = some s ∧
        ps.find? (·.template == s.template) = some p ∧ (templates (ss.toList.take i)).contains s.template = false ∧
        stmtCodes own (byTemplate own.interior) p s = []) ∧
      (∀ p ∈ ps.toList, own.drawn.any (p.holds own.sample.size) = true → (templates ss.toList).contains p.template = true) ∧
      Record.sameDict (own.reported.getD #[]) (statementReads ss.toList) = true := by
  simp only [check, fold_codes_empty] at h
  split at h
  · simp at h
  · simp at h
  · rename_i ps _ hp hs
    cases hs
    simp only [failures] at h
    split at h
    · simp at h
    · rename_i hne
      simp only [h, List.isEmpty_nil, Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hne
      simp only [List.append_eq_nil_iff, List.flatMap_eq_nil_iff, List.mem_range, and_assoc] at h
      obtain ⟨hst, hcov, hrep⟩ := h
      refine ⟨by simpa using hne, ps, hp, ?_, ?_, ?_⟩
      · intro i hi
        have hi' : i < ss.toList.length := by simpa using hi
        have := hst i hi'
        simp only [stmtFailures] at this
        split at this
        · simp at this
        · rename_i s hs
          split at this
          · rename_i p hp'
            split at this
            · simp at this
            · rename_i hdup
              exact ⟨s, p, hs, hp', by simpa using hdup, by simpa using this⟩
          · simp at this
      · intro p hp' hd
        have := List.filterMap_eq_nil_iff.1 hcov p hp'
        split at this
        · simp at this
        · rename_i hc
          simp only [hd, Bool.and_true, Bool.not_eq_true', Bool.not_eq_false] at hc
          simpa using hc
      · unfold reportedFailures at hrep
        split at hrep
        · assumption
        · simp at hrep

/-- A passing statement's checks: its population, roots, sample and registered reads. -/
theorem stmtCodes_nil {own : Own} {named : Bool} {p : Part} {s : Stmt} (h : stmtCodes own named p s = []) :
    populationOk own.sample.size p s.header = true ∧
      rootsOk named own.interior s.template s.header ((entries (getD s.registered "ports")).map (·.1)) = true ∧
      jeq s.draw (ownDraw own.sample.size p own.drawn) = true ∧
      (entries (getD s.registered "ports")).all (fun (port, e) => readOk own named p port e) = true := by
  simpa [stmtCodes, List.filterMap_eq_nil_iff] using h

private def exRoot (i : Nat) : String := String.ofList (List.replicate 128 (Char.ofNat (48 + i)))

private def exRec (population : Nat) (roots : List (String × Nat × Nat)) : Json :=
  Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("population", Lean.toJson population),
    ("roots", Json.arr (roots.toArray.map fun (p, r, d) =>
      Json.mkObj [("port", p), ("root", exRoot r), ("domain", exRoot d), ("leaves", Lean.toJson population)]))]

/-- A replay sample of blocks 1 and 3 of 4, three proof blocks of one template in each, units 2 and 4 drawn. -/
private def exOwn : Own :=
  { sample := #[1, 3], parts := parts 2 #[("Pair_v0", #[(0, 6)])], interior := exRec 6 [("x", 1, 2), ("out", 3, 4)],
    boundary := exRec 4 [("bx", 5, 6)], drawn := #[2, 4] }

private def exStmt (indices units : List Nat) : Json :=
  Json.mkObj [("template", "Pair_v0"), ("registered", .null),
    ("header", Json.mkObj [("scheme", "frame-v3-sha512/hm96-sha512"), ("instances", Lean.toJson indices.length),
      ("units", Json.mkObj [("indices", natsJson indices.toArray)]),
      ("frame_v3", Json.mkObj [("roots", Json.mkObj [("x", exRoot 1), ("out", exRoot 3)]),
        ("domain_ids", Json.mkObj [("x", exRoot 2), ("out", exRoot 4)])])]),
    ("draw", Json.mkObj [("law", "subset"), ("population", (6 : Nat)), ("k", Lean.toJson units.length),
      ("units", natsJson units.toArray)])]

#guard exOwn.parts == some #[{ template := "Pair_v0", lo := 0, n := 3 }]
#guard (check false exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes == #["two-stage-form"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes.isEmpty
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 6] [2, 4]])).codes == #["second-population"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [1, 4]])).codes == #["second-sample"]
#guard (check true exOwn (some #[exStmt [0, 1, 2, 3, 4, 5] [2, 4], exStmt [0, 1, 2, 3, 4, 5] [2, 4]])).codes ==
  #["second-template"]
#guard (check true exOwn (some #[])).codes == #["second-cover"]
#guard (check true exOwn none).codes == #["second-statements"]
#guard (check true { exOwn with parts := parts 4 #[("Pair_v0", #[(0, 6)])] } none).codes == #["second-cells"]

/-- Port `x`'s statement reading `value`'s tree of `leaves` leaves at root `root`, domain `dom`, slot `k` its `reads[k]`. -/
private def exReading (value : String) (root dom leaves : Nat) (reads : List Nat) : Json :=
  (exStmt [0, 1, 2, 3, 4, 5] [2, 4]).setObjVal! "registered" (Json.mkObj [("ports", Json.mkObj [("x", Json.mkObj [
    ("value", value), ("root", exRoot root), ("domain", exRoot dom), ("leaves", Lean.toJson leaves),
    ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)])])])

private def exReported (value : String) (root : Nat) (own : Own := exOwn) : Own :=
  { own with reported := some #[(value, .str (exRoot root))] }

private def exLinked : Own := { exOwn with links := #[⟨"Pair_v0", "x", "bx"⟩] }

#guard (check true (exReported "x" 1) (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes.isEmpty
#guard (check true exOwn (some #[exReading "x" 1 2 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-reported"]
#guard (check true (exReported "x" 1) (some #[exReading "x" 1 2 6 [1, 1, 2, 3, 4, 5]])).codes == #["second-read"]
#guard (check true (exReported "out" 3) (some #[exReading "out" 3 4 6 [0, 1, 2, 3, 4, 5]])).codes == #["second-read"]
#guard (check true (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes.isEmpty
#guard (check true (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 2]])).codes == #["second-read"]
#guard (check true (exReported "bx" 5) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes == #["second-read"]
#guard (check false (exReported "bx" 5 exLinked) (some #[exReading "bx" 5 6 4 [1, 1, 1, 3, 3, 3]])).codes ==
  #["two-stage-form"]

end SampledProofs.SecondStage
