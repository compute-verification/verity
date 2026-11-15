import Proofs.Flock.SpotFlock.Code
import Proofs.Flock.SamplingGeneric.TwoAt

/-!
# SPOT[Flock]'s two-stage model on a run of the two-stage toy

`lake env lean --threads=1 --tstack=KB --run TwoGuard.lean DIR MODE [M2]` in `Security/Proofs`, with Flock's and
SPOT's code loaded native (`--load-dynlib`; `guard_node.py` links it and writes the command): the model's
round-2 check (`SpotFlock.stmtOk`) on the run `guard.export` wrote to `DIR`, at the auditor's copies (`Held`) built from
`DIR/held.json`, the reference statements `DIR/ref-<t>.bin` and their templates' reference reads
`DIR/ref-<t>.reads.json` alone, never from the run's messages or coins. `M2` replaces round 2's message `DIR/m2.bin` (a
negative's).

**What it evaluates.** `stmtOk` is its conjuncts on the registration alone and its conjuncts on the run (`stmtOk_split`):
`regOk`, which is `partition` and `two` (both ignore the statement they are given), and `runOkAt`. This evaluates
`runOkAt`, and of `regOk` only `two`'s conjuncts on the members' files (`twoFilesOk`). It never evaluates `partition`'s
`membersPart` or `two`'s `scopeOk` over `bitsProgs`: both read only `h`, so `regOk h = true` is a fact about the
registration, to be proved from the members' facts rather than decided on a run, and each unrolls the members' circuits
(`Spot.Bits.prog`, `Spot.Bits.blockOf`). The output prints each member's sizes and the time of the unrolling's pieces at
one gate: `Spot.Bits.off` past its last gate, `Spot.Bits.consts`, and `Spot.Bits.owner` at an input gate.

* `check`: `runOkAt` and `twoFilesOk` at the run, with their times and the members' sizes, and `runOkAt`'s records
  check (`recordsOk`) on its own; exit 0 when `runOkAt` and `twoFilesOk` accept, 1 when either refuses, 2 when the
  reference statements are not in Flock's masked scope (`Held.xS₀_ok`, decided before any `Held` exists).
* `diag`: each conjunct of `runOkAt` on its own, in the order it reads them, with its time, then `check`; where
  `recordsOk` refuses, `runOkAt` again with the names the run's records name held in place of the registrations'
  (`recordNames`), which is what the rest of round 2 decides of the run.
* `files`: the registration alone, with no record, statement or coin of a run: each conjunct of `twoFilesOk` on its
  own with its time, each template's rows the wiring's clauses read, then `twoFilesOk`; exit 0 when it accepts, 1 when
  it refuses. It reads `held.json` and the reference statements and reads only, and unrolls no member.

Every line starts `GUARD `; the last is `GUARD-RESULT {json}`. Its `twoFilesOk` is `filesFitOk` (the layout fits the
members' files) and every clause of `wiring`, each under its name (`wiringClauses`); `unnamed` and `unread` list each
template's name with a row `sourceRowsNamed` or `writtenRowsRead` refuses, and `unmoved` each with a wired row at
which some unit has no gate to go to, which `wiresMoveRows` decides, and why. Each member's circuit is read once
(`readAll`), and the clauses are evaluated at that array (`wiringClausesAt`, the model's by `wiringClausesAt_eq`). The
model's statement parsers recurse deeper than a default stack holds: give the interpreter a large `--tstack`.

**The coins.** `stmtOk` reads the coins only through their bytes (round 0's through `SpotFlock.src`, the second
stage's through `SpotFlock.stageSrc`), so `runOkAt` takes the bytes. `src` itself reads its `2^20` bytes one list lookup
at a time, which no evaluation finishes. The bytes here are the ones each stage's sample read of the challenger's coins
(`c0.bin`, `c1.bin`), and zeros past them: the stream's Lean `Select` read exactly those bytes, and reads its source in
order, so it reads none of the zeros. The toy's second stage reads one coin, so the `Held` here has no round after
round 1 (`more := 0`).
-/

open Lean (Json)

namespace TwoGuard

open SpotFlock

/-- `SpotFlock.secondDraw` at the stages' bytes `s₀` and `s₁`. -/
def secondDrawAt (h : Held) (s₀ s₁ : ByteArray) : Option Json :=
  match Spot.Select.selectSecond (h.reg.sample s₀) h.reg.ns h.reg.ks s₁ with
  | .ok (draw, _) => some draw
  | .error _ => none

/-- `SpotFlock.ownOf` at the first stage's bytes `s₀`. -/
def ownOfAt (h : Held) (st : Roots) (rec draw : Json) (s₀ : ByteArray) : Spot.SecondStage.Own :=
  { sample := h.reg.sample s₀, parts := some (partsAt h (h.reg.sample s₀).size), interior := rec,
    boundary := st.record, drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, wires := h.wires,
    rows := h.rows, refs := refsOf h, refReads := refRegisteredOf h }

/-- `SpotFlock.readsOk` at the stages' bytes. -/
def readsOkAt (h : Held) (t : ℕ) (x m₁ xS : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match decode x, jsonMsg m₁, secondDrawAt h s₀ s₁, FlockDeploy.decode xS with
  | some st, some rec, some draw, some w =>
    let own := ownOfAt h st rec draw s₀
    Spot.SecondStage.interiorOrderOk own && Spot.SecondStage.namesOk own &&
      (Spot.SecondStage.inputsFailure own).isNone &&
      Spot.SecondStage.designOk (partsAt h own.sample.size) own &&
      ((stmtJsonOf h t w).map (codesOk h own t)).getD false && pinOk h t w
  | _, _, _, _ => false

theorem readsOk_eq (h : Held) (t : ℕ) (x m₁ xS : Claims.Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    readsOk h t x m₁ xS c₀ c₁ = readsOkAt h t x m₁ xS (src c₀) (stageSrc c₁) := rfl

/-- **`stmtOk`'s conjuncts on the run**, at the stages' bytes: the run's statement decodes, `m₂` states one session
statement per template, the run's records name their registrations' names (`recordsOk`), and template `t`'s statement
has `k₀ t` rounds, is in scope (`sessionOk`) and reads the run's (`readsOk`). -/
def runOkAt (h : Held) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : Bool :=
  match decode x, stmtsOf m₂ with
  | some _, some xs =>
    xs.length == h.ms.length && recordsOk h x m₁ &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOkAt h t x m₁ (xs.getD t (h.xS₀ t)) s₀ s₁
  | _, _ => false

/-- **`stmtOk`'s conjuncts on the registration alone**: the partition and the two-stage design evaluate, neither
reading the statement `s`. -/
def regOk (h : Held) (s : Claims.Sampling.AuditStatement) : Bool := (partition h s).isSome && (two h s).isSome

/-- **`stmtOk` is its registration's conjuncts and its run's**, the run's read through the coins' bytes. -/
theorem stmtOk_split (h : Held) (x m₁ m₂ : Claims.Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) :
    stmtOk h x m₁ m₂ c₀ c₁ = ((parse h x).any (regOk h) && runOkAt h x m₁ m₂ (src c₀) (stageSrc c₁)) := by
  unfold stmtOk runOkAt parse
  cases decode x <;> cases stmtsOf m₂ <;> simp [regOk, readsOk_eq, Bool.and_assoc]

/-- **The registered wiring's clauses**, each under its name: every member's circuit is in the scope Flock's
hidden-output proof covers (`membersScoped`), and every clause of `wiringOk`, in its order: each name is one
template's, each wire's writer's part is before its reader's, every row a template's units read source bits at is a
port a link or a wire names, every wired source bit has a gate to go to (decided once per template, `wiresMoveRows`),
each wire's reader port is like its writer's output port, no wire names a row a template's units write that no link
names and some wire reads it, each reference read at a port no link or wire names reads a port of the interior
registration, the interior registration registers no value, each member's circuit names its ports once, and every
returned output bit of its unit sits on a computed row. Each reads the registration's description alone. -/
def wiringClauses : List (String × (Held → Bool)) :=
  [("membersScoped", membersScoped), ("namesOnce", namesOnce), ("writersFirst", writersFirst),
    ("sourceRowsNamed", sourceRowsNamed), ("wiresMoveRows", wiresMoveRows), ("wiresAlike", wiresAlike),
    ("writtenRowsUnwired", writtenRowsUnwired), ("writtenRowsRead", writtenRowsRead), ("refReadsOwn", refReadsOwn),
    ("interiorValuesEmpty", fun h => h.interiorNames.values.isEmpty), ("portsOnce", portsOnce), ("outsOk", outsOk)]

/-- **The registered wiring passes its clauses** (`wiringClauses`, every one): `membersScoped` and `wiringOk`. -/
def wiringSizedOk (h : Held) : Bool := wiringClauses.all fun kv => kv.2 h

/-- **`two`'s conjuncts that fit the layout to the members' files**, at the members `ms` (`h.ms`): the replay layout
and the first stage's design exist, the layout fits the members' files (`filesOk`), and each template's reference
statement reads as a member's (`templateOk`). -/
def filesFitOk (h : Held) (ms : Spot.Bits.Members) : Bool :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some _ => filesOk h.reg ms l h.k && (List.range ms.length).all (templateOk h)
  | _, _ => false

/-- **`two`'s conjuncts on the members' files**, at the members `ms` (`h.ms`): the layout fits them (`filesFitOk`),
and the registered wiring passes its clauses (`wiringSizedOk`). Not `scopeOk`, whose program is the members' circuits
unrolled (`bitsProgs`). -/
def twoFilesOk (h : Held) (ms : Spot.Bits.Members) : Bool := filesFitOk h ms && wiringSizedOk h

/-! ## The wiring's clauses at the members' circuits, each read once

A clause of the model reads a member's circuit as `memberCircuit h t`, which reads its template's reference statement
again at every use, and counts the members as `h.ms.length`, which reads every one. Below, each clause is stated at a
count `n` and at circuits `c` (`wiringClausesAt`). At `h.vus.length` and at an array that holds each member's circuit,
read once (`circuitsAt (readAll h) (memberCircuit h)`), they are the model's clauses (`wiringClausesAt_eq`): that is
what this evaluates. -/

/-- `f`, at an index of `cs` the circuit `cs` holds. -/
def circuitsAt (cs : Array _root_.Flock.Circuit) (f : ℕ → _root_.Flock.Circuit) (t : ℕ) : _root_.Flock.Circuit :=
  if ht : t < cs.size then cs[t] else f t

/-- **Each member's circuit**: `memberCircuit h` at every template, in order. -/
def readAll (h : Held) : Array _root_.Flock.Circuit := (Array.range h.vus.length).map (memberCircuit h)

theorem circuitsAt_read (h : Held) : circuitsAt (readAll h) (memberCircuit h) = memberCircuit h := by
  funext t
  unfold circuitsAt readAll
  split
  · simp
  · rfl

theorem membersCount (h : Held) : h.ms.length = h.vus.length := by simp [Held.ms]

/-- `Held.ms` at circuits `c`. -/
def msAt (h : Held) (c : ℕ → _root_.Flock.Circuit) : Spot.Bits.Members :=
  (List.range h.vus.length).map fun t => (c t, h.vus.getD t #[])

/-- **The members, their circuits read once**, are the model's. -/
theorem msAt_eq (h : Held) : msAt h (circuitsAt (readAll h) (memberCircuit h)) = h.ms := by
  rw [circuitsAt_read]; rfl

/-- `heldWriter` at a count. -/
def heldWriterAt (h : Held) (n : ℕ) (name : String) : Option ℕ :=
  (List.range n).find? fun t => h.names.getD t "" == name

/-- `heldPos` at a count. -/
def heldPosAt (h : Held) (n : ℕ) (name : String) : ℕ := (heldWriterAt h n name).getD n

/-- `portNamed` at circuits. -/
def portNamedAt (c : ℕ → _root_.Flock.Circuit) (t : ℕ) (name : String) : ℕ :=
  ((List.range (c t).ports.size).find? fun q => (c t).ports[q]!.name == name).getD 0

/-- `wireAlike` at a count and circuits. -/
def wireAlikeAt (h : Held) (n : ℕ) (c : ℕ → _root_.Flock.Circuit) (w : Spot.SecondStage.Wire) (t : ℕ) : Bool :=
  match heldWriterAt h n w.writer with
  | some tw =>
    ((c t).ports[portNamedAt c t w.port]!).words == ((c tw).ports[portNamedAt c tw w.wport]!).words &&
      _root_.Flock.PortRoots.schemaOf (c t) (portNamedAt c t w.port) ==
        _root_.Flock.PortRoots.schemaOf (c tw) (portNamedAt c tw w.wport)
  | none => true

/-- `wireGoes` at a count. -/
def wireGoesAt (h : Held) (n t : ℕ) (row : String) (w : Spot.SecondStage.Wire) : Bool :=
  match heldRows h (h.names.getD t "") row, heldWriterAt h n w.writer with
  | some r, some tw => cellGoes h t tw w r
  | _, _ => false

/-- `rowGoes` at a count. -/
def rowGoesAt (h : Held) (n t : ℕ) (row : String) : Bool :=
  match heldLink h (h.names.getD t "") row, heldWire h (h.names.getD t "") row with
  | none, some w => wireGoesAt h n t row w
  | _, _ => true

/-- `wireBit` at a count and circuits. -/
def wireBitAt (h : Held) (n : ℕ) (c : ℕ → _root_.Flock.Circuit) (w : Spot.SecondStage.Wire) (bit : ℕ) : Bool :=
  match heldWriterAt h n w.writer with
  | some tw => outBit (c tw) w.wport bit
  | none => false

/-- `bitGoes` at a count and circuits. -/
def bitGoesAt (h : Held) (n : ℕ) (c : ℕ → _root_.Flock.Circuit) (t : ℕ) (row : String) (bit : ℕ) : Bool :=
  match heldLink h (h.names.getD t "") row, heldWire h (h.names.getD t "") row with
  | none, some w => wireBitAt h n c w bit
  | _, _ => true

/-- **`wiringClauses`, each at a count `n` and circuits `c`**, in their order. -/
def wiringClausesAt : List (String × (Held → ℕ → (ℕ → _root_.Flock.Circuit) → Bool)) :=
  [("membersScoped", fun _ n c => (List.range n).all fun t => _root_.Flock.ProvedScope.scopeOkH (c t)),
    ("namesOnce", fun h n _ => decide ((List.range n).map fun t => h.names.getD t "").Nodup),
    ("writersFirst", fun h n _ =>
      (h.wires.getD #[]).all fun w => decide (heldPosAt h n w.writer < heldPosAt h n w.template)),
    ("sourceRowsNamed", fun h n c =>
      (List.range n).all fun t =>
        (sourceRows (c t)).all fun row =>
          (heldLink h (h.names.getD t "") row).isSome || (heldWire h (h.names.getD t "") row).isSome),
    ("wiresMoveRows", fun h n c =>
      (List.range n).all fun t =>
        ((sourceRows (c t)).all fun row => rowGoesAt h n t row) &&
          (sourceCells (c t)).all fun rb => bitGoesAt h n c t rb.1 rb.2),
    ("wiresAlike", fun h n c =>
      (h.wires.getD #[]).all fun w =>
        (List.range n).all fun t => !(w.template == h.names.getD t "") || wireAlikeAt h n c w t),
    ("writtenRowsUnwired", fun h n c =>
      (List.range n).all fun t =>
        (writtenRows (c t)).all fun row =>
          (heldLink h (h.names.getD t "") row).isSome || (heldWire h (h.names.getD t "") row).isNone),
    ("writtenRowsRead", fun h n c =>
      (List.range n).all fun t =>
        (writtenRows (c t)).all fun row =>
          (heldLink h (h.names.getD t "") row).isSome ||
            (h.wires.getD #[]).any fun w => w.writer == h.names.getD t "" && w.wport == row),
    ("refReadsOwn", fun h n _ =>
      (List.range n).all fun t =>
        (Spot.entries (Spot.getD (h.refReads t) "ports")).all fun pe =>
          (heldLink h (h.names.getD t "") pe.1).isSome || (heldWire h (h.names.getD t "") pe.1).isSome ||
            match Spot.getD pe.2 "value" with
            | .str v => h.interiorNames.ports.contains v
            | _ => false),
    ("interiorValuesEmpty", fun h _ _ => h.interiorNames.values.isEmpty),
    ("portsOnce", fun _ n c => (List.range n).all fun t => decide ((c t).ports.toList.map (·.name)).Nodup),
    ("outsOk", fun _ n c => (List.range n).all fun t => outsComputed (c t))]

/-- **The clauses at the members' circuits read once are the model's**: `wiringClausesAt` at the templates' count
(`h.vus.length`) and at the array of the members' circuits (`readAll`) is `wiringClauses`, name by name. -/
theorem wiringClausesAt_eq (h : Held) :
    (wiringClausesAt.map fun kv => (kv.1, kv.2 h h.vus.length (circuitsAt (readAll h) (memberCircuit h)))) =
      wiringClauses.map fun kv => (kv.1, kv.2 h) := by
  rw [circuitsAt_read, ← membersCount]; rfl

/-- **The rows the wiring's clauses read at one template**: its name, the rows one unit reads source bits at
(`sourceRows`) with those no link or wire names, which `sourceRowsNamed` refuses, and the rows it writes
(`writtenRows`) with those no link names and no wire reads, which `writtenRowsRead` refuses. -/
structure Rows where
  name : String
  source : List String
  unnamed : List String
  written : List String
  unread : List String

/-- Template `t`'s `Rows`, at its circuit `c`. -/
def rowsAt (h : Held) (c : _root_.Flock.Circuit) (t : ℕ) : Rows :=
  let name := h.names.getD t ""
  let source := (sourceRows c).eraseDups
  let written := writtenRows c
  { name, source, written,
    unnamed := source.filter fun row => !((heldLink h name row).isSome || (heldWire h name row).isSome),
    unread := written.filter fun row =>
      !((heldLink h name row).isSome || (h.wires.getD #[]).any fun w => w.writer == name && w.wport == row) }

/-! ## Why a wired source bit has no gate to go to

The model's clause is `wiresMoveRows`, one of `wiringClauses`: every unit of a template has a writer's unit to read
each wired row at, and that unit writes every bit read. Below, the same is asked of the rows one unit reads source
bits at and of how many bits of each it reads, and each row at which it fails is listed with why (`unmovedAt`). The
list is said beside the model's clause: `twoFilesOk` reads the clause and not the list, and no theorem here ties the
two. -/

/-- **How many bits of each row one unit of circuit `c` reads**: for each row an input column's source is a message
bit of (`Bits.srcKey`, `Bits.srcPos`), one more than the last bit of it any column reads. -/
def readBitsAt (c : _root_.Flock.Circuit) : List (String × ℕ) :=
  let ix := Spot.Bits.Index.of c
  (List.range ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn).foldl (fun acc b =>
    match Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c ix b) with
    | some x =>
      match acc.find? (·.1 == x.row) with
      | some e => if x.bit < e.2 then acc else acc.map fun e' => if e'.1 == x.row then (e'.1, x.bit + 1) else e'
      | none => acc ++ [(x.row, x.bit + 1)]
    | none => acc) []

/-- **How many bits of the row named `wport` one unit of circuit `c` writes**: sixteen a word of its first output
port of that name; none where no output row port has the name. An output row's port is padded to whole hash blocks,
so the port's own `words` would say more. -/
def writtenBitsAt (c : _root_.Flock.Circuit) (wport : String) : Option ℕ :=
  ((List.range c.outPorts.size).find? fun o =>
    (c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name == wport).map fun o =>
      16 * (c.outPorts.getD o default).2

/-- **Why a unit of template `t` has no gate to go to for some bit of its row `row`**, which wire `w` names, where
one unit reads `bits` bits of it; none where every unit has one for each. -/
def unmovedWhy (h : Held) (n : ℕ) (c : ℕ → _root_.Flock.Circuit) (t : ℕ) (row : String) (bits : ℕ)
    (w : Spot.SecondStage.Wire) : Option String :=
  match heldRows h (h.names.getD t "") row, heldWriterAt h n w.writer with
  | none, _ => some "the registration gives the row no rows"
  | _, none => some s!"the wire's writer {w.writer} names no template"
  | some r, some tw =>
    let nt := h.reg.ns.getD t 0
    let nw := h.reg.ns.getD tw 0
    let units := (h.vus.getD t #[]).size
    let wunits := (h.vus.getD tw #[]).size
    if (heldLink h (h.names.getD tw "") w.wport).isSome then some s!"a link names the writer's port {w.wport}"
    else if nt == 0 then some "the reader's cell is empty"
    else if units % nt != 0 then some s!"the reader's {units} units are not whole cells of {nt}"
    else if wunits < units / nt * nw then
      some s!"the writer's {wunits} units are fewer than {units / nt} cells of {nw}"
    else if nw < r.per then some s!"the rows a replay block holds ({r.per}) are more than the writer's cell ({nw})"
    else if !Spot.SecondStage.fits r.runs nt then some s!"the rows' runs do not hold the reader's cell of {nt}"
    else if !(r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)) then
      some s!"a run's last row is past the {r.per} rows a replay block holds"
    else match writtenBitsAt (c tw) w.wport with
      | none => some s!"the writer has no output row port {w.wport}"
      | some wbits =>
        if wbits < bits then some s!"a unit reads {bits} bits of the row and the writer's unit writes {wbits}"
        else none

/-- **Each template's rows a wire names and no link names at which some unit has no gate to go to**, with why
(`unmovedWhy`): the template's name, the row and the reason. Empty where every wired source bit has one. -/
def unmovedAt (h : Held) (n : ℕ) (c : ℕ → _root_.Flock.Circuit) : List (String × String × String) :=
  (List.range n).flatMap fun t =>
    (readBitsAt (c t)).filterMap fun rb =>
      match heldLink h (h.names.getD t "") rb.1, heldWire h (h.names.getD t "") rb.1 with
      | none, some w => (unmovedWhy h n c t rb.1 rb.2 w).map fun why => (h.names.getD t "", rb.1, why)
      | _, _ => none

/-! ## The auditor's copies -/

/-- Template `t`'s reference statement among `refs`, the last template's past them. -/
def refAt (refs : Array Claims.Msg) (t : ℕ) : Claims.Msg := refs.getD (min t (refs.size - 1)) []

theorem refAt_ok (refs : Array Claims.Msg)
    (hz : 0 < refs.size ∧ (List.range refs.size).all (fun t => zkOk (refs.getD t [])) = true) :
    ∀ t, zkOk (refAt refs t) = true := fun t =>
  List.all_eq_true.mp hz.2 (min t (refs.size - 1)) (List.mem_range.mpr (by have := hz.1; omega))

def ok {α : Type} (what : String) : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError s!"{what}: {e}")

def field (j : Json) (k : String) : Json := (j.getObjVal? k).toOption.getD .null

def natsOf (what : String) (j : Json) : IO (List ℕ) := do
  (← ok what j.getArr?).toList.mapM fun n => ok what n.getNat?

def hexOf (what : String) (j : Json) : IO Claims.Msg := do
  return (← ok what (_root_.Flock.unhex (← ok what j.getStr?))).data.toList

def strsOf (what : String) (j : Json) : IO (List String) := do
  (← ok what j.getArr?).toList.mapM fun n => ok what n.getStr?

/-- A registration's names (`Names`) as `held.json` gives them under `key`. -/
def namesOf (held : Json) (key : String) : IO Names := do
  let j := field held key
  if j == .null then
    throw (IO.userError s!"held.json: no {key}; an export from before the records' names (guard.export writes them)")
  return ⟨← strsOf key (field j "ports"), ← strsOf key (field j "values")⟩

/-- **The names record `rec` itself names** (`namesAt`), at which `recordOk rec` holds where it names each once. -/
def recordNames (rec : Json) : Names :=
  ⟨(namesAt rec "roots" "port").getD [], (namesAt rec "registered" "value").getD []⟩

/-- **The auditor's copies**, from `held.json` (`held`), the reference statements `refs` and their templates' reference
reads `reads` (`Held.refReads`, null past them), behind the decision that each reference statement is in Flock's masked
scope (`refAt_ok`); none where one isn't. The boundary and interior registrations' names are `held.json`'s
(`boundary_names`, `interior_names`: the design's ports, `expected_ports`, and no registered values). The first stage's
registration
is the boundary registration's program, query and work table with the stream's rule over them (`Select.resolve`, as
the `select` piece takes it), the fields of `Registration.Verifier` the model doesn't read empty; `blockOf` is
`filesBlockOf` at the members (`Held.ms`), and no proof block is checked whatever the stages sample (`always`): every
slot of the toy's boundary layout is a request's. -/
def heldOf (held : Json) (refs : Array Claims.Msg) (reads : Array Json) : IO (Option Held) := do
  let first := field held "first"
  let queryJ := field first "query"
  let P ← ok "program" do Spot.Program.parse (← _root_.Flock.canon (field first "program")).toUTF8
  let rule ← ok "rule" do Spot.Select.resolve (← Spot.Select.Rule.ofJson (field first "rule")) P queryJ
  let table : Option Json := match field first "work_table" with
    | .null => none
    | t => some t
  let ns ← natsOf "ns" (field held "ns")
  let ks ← natsOf "ks" (field held "ks")
  let vus ← (← ok "vus" (field held "vus").getArr?).toList.mapM fun v => return (← natsOf "vus" v).toArray
  let names ← (← ok "names" (field held "names").getArr?).toList.mapM fun n => ok "names" n.getStr?
  let q := field held "query"
  let query : Claims.Sampling.Query :=
    ⟨← ok "query" (field q "name").getStr?, ← ok "query" (field q "version").getNat?, ← hexOf "query" (field q "params")⟩
  let desc ← hexOf "desc" (field held "desc")
  let k ← natsOf "k" (field held "k")
  let links := Spot.SecondStage.linksOf? (field held "links")
  let wires := Spot.SecondStage.wiresOf? (field held "wires")
  let rows ← ok "rows" ((Spot.SecondStage.rowsOf? (field held "rows")).elim (.error "not a list of rows") .ok)
  let boundaryNames ← namesOf held "boundary_names"
  let interiorNames ← namesOf held "interior_names"
  if hz : 0 < refs.size ∧ (List.range refs.size).all (fun t => zkOk (refs.getD t [])) = true then
    let verifier : Spot.Registration.Verifier :=
      { program := P, query := queryJ, law := .null, workTable := table, scheme := "", ports := #[], leafLayer := none,
        anchors := #[], log := #[], partition := none }
    let h₀ : Held :=
      { desc, vus, query, k, names, links, wires, rows, boundaryNames, interiorNames, xS₀ := refAt refs,
        xS₀_ok := refAt_ok refs hz,
        refReads := fun t => reads.getD t .null,
        refHeaders := fun t =>
          ((FlockDeploy.decode (refAt refs t)).bind fun w₀ => headerOf w₀.publicFile).getD .null,
        frame := SamplingDeploy.toyCode, more := 0,
        reg := { first := { verifier, rule, closure := none }, ns := ns.toArray, ks := ks.toArray,
                 blockOf := fun _ _ _ => 0, always := fun _ => false } }
    let ms := h₀.ms
    return some { h₀ with reg := { h₀.reg with blockOf := Properties.Spot.filesBlockOf ms h₀.reg.ns } }
  else
    return none

/-! ## Running it -/

def say (line : String) : IO Unit := do
  IO.println s!"GUARD {line}"
  (← IO.getStdout).flush

/-- `f ()`, said with `fmt` and its time in ms, which it returns beside it. -/
def timed {α : Type} (label : String) (f : Unit → α) (fmt : α → String) : IO (α × ℕ) := do
  let t0 ← IO.monoMsNow
  let a ← IO.lazyPure f
  say s!"{label} = {fmt a}"
  let ms := (← IO.monoMsNow) - t0
  say s!"{label}: {ms} ms"
  return (a, ms)

def b (x : Bool) : String := toString x

def readMsg (p : System.FilePath) : IO Claims.Msg := return (← IO.FS.readBinFile p).data.toList

/-- **Why a reference statement is out of Flock's masked scope**: the first of `zkOk`'s steps it fails (`setupOf`:
decode, the verifier's parsers `Raw.ins`, `Ins.setup`; then `zkOk`'s bounds), with its error; none where `zkOk` holds. -/
def refWhy (xS : Claims.Msg) : Option String :=
  match FlockDeploy.decode xS with
  | none => some "decode: the bytes don't decode as a statement"
  | some w =>
    if !w.zk then some "the statement is not --zk" else
    match w.ins with
    | .error e => some s!"parse: {e}"
    | .ok s =>
      match s.setup with
      | .error e => some s!"setup: {e}"
      | .ok _ => if zkOk xS then none else some "zkOk: it sets up, outside the masked scope's bounds"

/-- A stage's bytes, zeros past `n` of them up to `L`. -/
def coinBytes (p : System.FilePath) : IO ByteArray := do
  let r ← IO.FS.readBinFile p
  return r ++ ⟨Array.replicate (L - r.size) 0⟩

/-- **Each member's key program** for its units (`Spot.Bits.keyProg`): its inputs and rows, its lowered gates (one an
input, three a row: `Spot.Bits.off` past its last gate), the walks of every gate its lowering makes (`Spot.Bits.lower`
calls `off` once per gate a row reads and twice more a row), and the time of one walk; and the time of the constants'
answer at one input gate (`Spot.Bits.consts`, which the lowering asks at each input gate) and of that gate's owner
(`Spot.Bits.owner`). -/
def members (ms : Spot.Bits.Members) : IO (List Json) :=
  (List.range ms.length).mapM fun t => do
    let (c, vs) := ms.getD t default
    let (kp, _) ← timed s!"m{t} keyProg" (fun _ => Spot.Bits.keyProg c vs.size) fun kp =>
      s!"{kp.inputs} inputs, {kp.rows.size} rows"
    let (lowered, walkMs) ← timed s!"m{t} off past its last gate" (fun _ => Spot.Bits.off kp kp.size) toString
    let walks := kp.rows.foldl (fun s r => s + r.1.length + r.2.length + 2) 0
    let κ := Spot.Bits.consts c vs.size
    let (_, constsMs) ← timed s!"m{t} consts at input gate 2" (fun _ => κ 2) toString
    let (_, ownerMs) ← timed s!"m{t} owner of input gate 2" (fun _ => Spot.Bits.owner kp 2) toString
    return Json.mkObj [("units", Lean.toJson vs.size), ("inputs", Lean.toJson kp.inputs),
      ("rows", Lean.toJson kp.rows.size), ("lowered", Lean.toJson lowered), ("lowering_walks", Lean.toJson walks),
      ("walk_ms", Lean.toJson walkMs), ("consts_ms", Lean.toJson constsMs), ("owner_ms", Lean.toJson ownerMs)]

/-- **`twoFilesOk`'s conjuncts, each evaluated once and said with its time**, at the members' circuits `c`
(`circuitsAt (readAll h) (memberCircuit h)`) and the members `ms` over them (`msAt h c`): the layout's fit (`filesOk`,
then each template's `templateOk`: `filesFitOk`), each clause of the wiring under its name (`wiringClausesAt`, the
model's by `wiringClausesAt_eq`) and each template's rows (`rowsAt`). Its result's fields: `twoFilesOk` with its time,
`filesFitOk`, the clauses (`wiring`), and each template's name with a row no link or wire names (`unnamed`) and with
a written row no link names and no wire reads (`unread`). Beside them, not read by `twoFilesOk`: each template's
name with a wired row at which some unit has no gate to go to and why (`unmoved`, `unmovedAt`), which the wiring's
clause `wiresMoveRows` decides. -/
def filesResult (h : Held) (c : ℕ → _root_.Flock.Circuit) (ms : Spot.Bits.Members) :
    IO (Bool × List (String × Json)) := do
  let mut total := 0
  let mut fit := false
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some _ =>
    let (v, vMs) ← timed "filesOk" (fun _ => filesOk h.reg ms l h.k) b
    fit := v
    total := total + vMs
  | _, _ => say "layout or replayDesign: none"
  for t in List.range ms.length do
    let (v, vMs) ← timed s!"t{t} templateOk" (fun _ => templateOk h t) b
    fit := fit && v
    total := total + vMs
  say s!"filesFitOk = {fit}"
  let mut wiring : List (String × Bool) := []
  for kv in wiringClausesAt do
    let (v, vMs) ← timed kv.1 (fun _ => kv.2 h h.vus.length c) b
    wiring := wiring ++ [(kv.1, v)]
    total := total + vMs
  let mut rows : List Rows := []
  for t in List.range ms.length do
    let (r, _) ← timed s!"t{t} rows" (fun _ => rowsAt h (c t) t) fun r =>
      s!"{r.name}: scopeOkH {_root_.Flock.ProvedScope.scopeOkH (c t)}; source rows {r.source}, \
        unnamed {r.unnamed}; written rows {r.written}, unread {r.unread}"
    rows := rows ++ [r]
  let (unmoved, _) ← timed "unmoved" (fun _ => unmovedAt h h.vus.length c) toString
  let files := fit && wiring.all (·.2)
  say s!"twoFilesOk = {files}"
  let pairs (f : Rows → List String) : Json :=
    Json.arr (rows.flatMap fun r => (f r).map fun row => Json.arr #[Json.str r.name, Json.str row]).toArray
  return (files, [("twoFilesOk", Lean.toJson files), ("twoFilesOk_ms", Lean.toJson total),
    ("filesFitOk", Lean.toJson fit), ("wiring", Json.mkObj (wiring.map fun kv => (kv.1, Lean.toJson kv.2))),
    ("unnamed", pairs (·.unnamed)), ("unread", pairs (·.unread)),
    ("unmoved", Json.arr (unmoved.map fun e => Json.arr #[Json.str e.1, Json.str e.2.1, Json.str e.2.2]).toArray)])

/-- Each conjunct of `runOkAt` on its own, in their order. -/
def diag (h : Held) (ms : Spot.Bits.Members) (x m₁ m₂ : Claims.Msg) (s₀ s₁ : ByteArray) : IO Unit := do
  let (st?, _) ← timed "decode x" (fun _ => decode x) (fun s => b s.isSome)
  let (xs?, _) ← timed "stmtsOf" (fun _ => stmtsOf m₂) fun xs => toString (xs.map (·.map List.length))
  let xs := xs?.getD []
  let (rec?, _) ← timed "jsonMsg m₁" (fun _ => jsonMsg m₁) (fun r => b r.isSome)
  let _ ← timed "recordsOk" (fun _ => recordsOk h x m₁) b
  let (sd, _) ← timed "secondDraw" (fun _ => secondDrawAt h s₀ s₁) fun d => (d.map (·.compress)).getD "none"
  let _ ← timed "replay sample" (fun _ => h.reg.sample s₀) toString
  let (refs, _) ← timed "refsOf" (fun _ => refsOf h) (fun r => toString r.size)
  let (refReads, _) ← timed "refRegisteredOf" (fun _ => refRegisteredOf h) fun r =>
    toString (r.toList.map fun e => (e.1, e.2 != Json.null))
  for t in List.range ms.length do
    let xS := xs.getD t (h.xS₀ t)
    let _ ← timed s!"t{t} rounds = k₀" (fun _ => ((FlockDeploy.V true).k xS, k₀ h t)) toString
    let _ ← timed s!"t{t} zkOk" (fun _ => zkOk xS) b
    let _ ← timed s!"t{t} shapes" (fun _ => (shapeOf xS, shapeOf (h.xS₀ t))) toString
    let _ ← timed s!"t{t} scopeCheck" (fun _ => FlockDeploy.scopeCheck true xS) b
    let _ ← timed s!"t{t} hiddenOf" (fun _ => hiddenOf xS) b
    let (w?, _) ← timed s!"t{t} decode" (fun _ => FlockDeploy.decode xS) (fun w => b w.isSome)
    match st?, rec?, sd, w? with
    | some st, some rec, some draw, some w =>
      let own : Spot.SecondStage.Own :=
        { sample := h.reg.sample s₀, parts := some (partsAt h (h.reg.sample s₀).size), interior := rec,
          boundary := st.record, drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, wires := h.wires,
          rows := h.rows, refs, refReads }
      let _ ← timed s!"t{t} interiorOrderOk" (fun _ => Spot.SecondStage.interiorOrderOk own) b
      let _ ← timed s!"t{t} namesOk" (fun _ => Spot.SecondStage.namesOk own) b
      let _ ← timed s!"t{t} inputsFailure" (fun _ => Spot.SecondStage.inputsFailure own) (·.getD "none")
      let _ ← timed s!"t{t} designOk" (fun _ =>
        Spot.SecondStage.designOk (partsAt h own.sample.size) own) b
      let (S?, _) ← timed s!"t{t} stmtJsonOf" (fun _ => stmtJsonOf h t w) (fun S => b S.isSome)
      if let some S := S? then
        let _ ← timed s!"t{t} codes, as codesOk reads them" (fun _ =>
          Spot.SecondStage.stmtCodes { own with verified := S.checked.toArray }
            (Spot.SecondStage.byTemplate own.interior)
            (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t ""))
            (partAt h own.sample.size t) S) toString
        let _ ← timed s!"t{t} codesOk" (fun _ => codesOk h own t S) b
      let _ ← timed s!"t{t} pinOk" (fun _ => pinOk h t w) b
    | _, _, _, _ => say s!"t{t}: readsOk has no input (decode x, jsonMsg m₁, secondDraw or decode xS is none)"

def main (args : List String) : IO UInt32 := do
  let (dir :: mode :: rest) := args
    | do
      IO.eprintln "usage: TwoGuard.lean DIR check|diag|files [M2]"
      return 2
  let dir : System.FilePath := dir
  let t0 ← IO.monoMsNow
  -- `files` reads the registration alone: no record, statement or coin of a run
  let ran := mode != "files"
  let x ← if ran then readMsg (dir / "x.bin") else pure []
  let m₁ ← if ran then readMsg (dir / "m1.bin") else pure []
  let m₂ ← if ran then readMsg (rest.head?.map (System.FilePath.mk ·) |>.getD (dir / "m2.bin")) else pure []
  let s₀ ← if ran then coinBytes (dir / "c0.bin") else pure ByteArray.empty
  let s₁ ← if ran then coinBytes (dir / "c1.bin") else pure ByteArray.empty
  let held ← ok "held.json" (Json.parse (← IO.FS.readFile (dir / "held.json")))
  let mut refs : Array Claims.Msg := #[]
  let mut reads : Array Json := #[]
  for t in [0:(← ok "names" (field held "names").getArr?).size] do
    refs := refs.push (← readMsg (dir / s!"ref-{t}.bin"))
    let r := dir / s!"ref-{t}.reads.json"
    unless (← r.pathExists) do
      throw (IO.userError s!"{r}: no reference reads; an export from before they left the reference statements \
        (python3 benchmarks/two_stage/guard.py refs {dir})")
    reads := reads.push (← ok r.toString (Json.parse (← IO.FS.readFile r)))
  say s!"read: x {x.length} B, m1 {m₁.length} B, m2 {m₂.length} B, {refs.size} references, {(← IO.monoMsNow) - t0} ms"
  let t1 ← IO.monoMsNow
  let some h ← heldOf held refs reads
    | do
      let whys := (List.range refs.size).filterMap fun t => (refWhy (refs.getD t [])).map (s!"ref-{t}: " ++ ·)
      for w in whys do say w
      let res := Json.mkObj [("held", Lean.toJson false),
        ("why", Lean.toJson "a reference statement is not in Flock's masked scope"), ("refs", Lean.toJson whys)]
      IO.println s!"GUARD-RESULT {res.compress}"
      return 2
  say s!"held: {(← IO.monoMsNow) - t1} ms"
  let (cs, _) ← timed "circuits (each reference statement read once)" (fun _ => readAll h) fun cs => toString cs.size
  let c := circuitsAt cs (memberCircuit h)
  let (ms, _) ← timed "ms" (fun _ => msAt h c) fun ms => toString (ms.map (·.2.size))
  if mode == "files" then
    let (files, fields) ← filesResult h c ms
    let res := Json.mkObj ([("held", Lean.toJson true),
      ("not_evaluated", Lean.toJson "the run; partition's membersPart and two's scopeOk, over bitsProgs")] ++
      fields)
    IO.println s!"GUARD-RESULT {res.compress}"
    return if files then 0 else 1
  let sizes ← members ms
  if mode == "diag" then diag h ms x m₁ m₂ s₀ s₁
  let (run, runMs) ← timed "runOk" (fun _ => runOkAt h x m₁ m₂ s₀ s₁) b
  let (records, _) ← timed "recordsOk" (fun _ => recordsOk h x m₁) b
  -- where the records' names refuse, the rest of round 2's verdict: `runOkAt` with the run's own names held, which
  -- takes several times `runOk`'s time, so only under `diag`
  let mut rest : Json := .null
  if mode == "diag" && !records then
    if let (some st, some rec) := (decode x, jsonMsg m₁) then
      let h' := { h with boundaryNames := recordNames st.record, interiorNames := recordNames rec }
      rest := Lean.toJson (← timed "runOk at the records' own names" (fun _ => runOkAt h' x m₁ m₂ s₀ s₁) b).1
  let (files, fields) ← filesResult h c ms
  let res := Json.mkObj ([("held", Lean.toJson true), ("runOk", Lean.toJson run), ("runOk_ms", Lean.toJson runMs),
    ("recordsOk", Lean.toJson records), ("runOk_at_own_names", rest), ("members", Json.arr sizes.toArray),
    ("not_evaluated", Lean.toJson "partition's membersPart and two's scopeOk, over bitsProgs")] ++ fields)
  IO.println s!"GUARD-RESULT {res.compress}"
  return if run && files then 0 else 1

end TwoGuard

def main (args : List String) : IO UInt32 := TwoGuard.main args
