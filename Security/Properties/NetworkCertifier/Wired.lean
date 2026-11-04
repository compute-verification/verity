import NetworkCheck.Circuit
import Definitions.NetworkCertifier.SampledCharge
import Properties.NetworkScan
import Properties.NetworkPeriod

/-!
# A compute audit's wired rows: what its acceptance gives the sampled audit

A compute audit's Program is a population of template instances whose inputs are all input gates, and its circuit
record (`NetworkCheck.Circuit.Stated`) states the dataflow between them, its wires, and the rows its registered reads
fill. The sampled audit reads the wires through its layout (`NetTiming.BoundaryGates.wires`): each wired input gate
reads its writer's value, so it is no free input, and the record's count (`Counts.free`) leaves it out.

**The circuit** is the audit's members' joint program (`Properties.NetworkScan.jointOf`), at the templates' circuit
files `fs`, each laid out by Bits as its template (`FilesOf`), and the members `ms` over them (`MembersOf`): every unit's
gates, returning every unit's output copy bits. **The wires** are Bits' (`SampledProofs.Bits.wiresOf`): each source
bit to the first claims position at its commitment position, compared at the partition's VUs, the writer's output bit
where a unit writes it and else the first copy.

**The registered rows** (`regGates`) are the reader rows of the registered reads of a held parameter and of the table
reads and openings of a table over one, the rows `countsOf` counts as `registered`, bit for bit: bit `β` of a template's
port at its instance `u` (`portGates`) is its port word's bit (`wordBit`), at the instance's unit and copy of its member
(`instPos`, `SampledProofs.Bits.inBit`).

* `CircuitCounts`: an accepted record's `free` is at least the circuit's free inputs outside its registered rows.
* `Properties.NetworkPeriod.RowsWiresChecked`: the period check (`NetworkCheck.Period`) charges each audit the free
  inputs it counted off its registered Program, and with a circuit record those `Circuit.countsOf` counted on the record
  `Circuit.checked` read, at its pins' parameters and at those and the scan's state (`stateFree`), the record its
  registration binds by digest, with each pin its registered root and none the scan's state.
* `Properties.NetworkPeriod.RowsHeld`: an accepted rows-form public's `held` names `circuit` and `free`, and each holds:
  `RowsCircuitHeldAt` and `RowsWiresCheckedAt`.

`NetTiming.Guarantees.PeriodLeakAccepted` takes each statement's layout and registered inputs (`reg`) as parameters and
counts `#(freeInputs \ reg)`. These compose with it at the record a statement's registration binds by digest
(`window.circuit`) and its pins' held parameters `held`: layouts whose wires are `wiresOf ms` and
`reg := regGates P fs ms r held`, so `CircuitCounts` bounds that count by the record's `free`, and the registration's
openings (`RegOpens`) are owed at exactly `regGates P fs ms r held`.

`CircuitCounts` says its conclusion (`CircuitCountsAt`) of every record `check` accepts, and so is its property of any
circuit check `chk` (`CircuitCountsOf`), which the nontrivial lemma states false at `acceptAllCircuit`, a check that
accepts every record; `RowsWiresChecked` and `RowsHeld` likewise of the period's decoder and check, false at
`acceptAllParams` and `acceptAllRows`, which decode every parameters to one audit's period and accept every period.
`Proofs.Properties.NetworkCertifier` proves them.
-/

namespace Properties.NetworkCertifier

open Lean (Json)
open Flock.Program (Prog Ty)
open NetworkCheck.Period (templateNodes bitsOf)
open NetworkCheck.Circuit (Stated Counts nodeOf templateDef recordOf heldOf)
open SampledProofs.Bits (Members inBit wiresOf)
open Properties.NetworkScan (jointOf FilesOf MembersOf instPos wordBit)

/-- Template node `k`'s parameter `port`: its index and type. -/
def portParam (P : Prog) (k : ℕ) (port : String) : Option (ℕ × Ty) :=
  ((templateDef P k).params.toList.zipIdx.find? (·.1.1 == port)).map fun ((_, t), i) => (i, t)

/-- Template `tid`'s port `port`'s bits; 0 where it has none. -/
def portWidth (P : Prog) (r : Stated) (tid port : String) : ℕ :=
  match nodeOf P r tid with
  | some k => ((portParam P k port).map (bitsOf ·.2)).getD 0
  | none => 0

/-- **The claims positions of bits `a … a + n − 1` of template `tid`'s port `port` at its instance `u`**, in the
members' joint program: each bit's port word and bit (`wordBit`, over the template's parameters) at the instance's
unit and copy (`instPos`, `SampledProofs.Bits.inBit`). None where the record names no template node `tid` or it has no
parameter `port`. -/
def portGates (P : Prog) (fs : List Flock.Circuit) (ms : Members) (r : Stated) (tid port : String) (u a n : ℕ) :
    List ℕ :=
  match nodeOf P r tid with
  | none => []
  | some k => match portParam P k port, (templateNodes P).findIdx? (· == k) with
    | some (π, _), some i =>
      let ps := (templateDef P k).params.toList.map (·.2)
      (List.range n).map fun β =>
        instPos P fs ms i u fun c m v q => inBit c m v q (wordBit ps π (a + β)).1 (wordBit ps π (a + β)).2
    | _, _ => []

/-- **The registered rows' gates** at the held parameters `held` (`heldOf`): each registered read of a held parameter's
reader rows (its pattern's reader indices, `Wire.readerAt`, each row its port's bits over its cut), and each port of
each table read and opening of a table over a held parameter, whole, at each of its units: the rows `countsOf` counts as
`registered`, bit for bit. -/
def regGates (P : Prog) (fs : List Flock.Circuit) (ms : Members) (r : Stated) (held : List String) : Finset ℕ :=
  let reg := r.registered
  let pinned := (reg.tables.filter (held.contains ·.param)).map (·.name)
  let whole := fun (tid p : String) (start n : ℕ) =>
    (List.range n).flatMap fun v => portGates P fs ms r tid p (start + v) 0 (portWidth P r tid p)
  ((reg.reads.filter (held.contains ·.param)).flatMap (fun x =>
      let row := portWidth P r x.pat.reader x.pat.port / r.wiring.cutsOf x.pat.reader x.pat.port
      (List.range x.pat.units).flatMap fun i =>
        portGates P fs ms r x.pat.reader x.pat.port (x.pat.readerAt i).1 ((x.pat.readerAt i).2 * row) row) ++
    (reg.roots.filter (pinned.contains ·.table)).flatMap (fun x => whole x.reader x.port x.start x.units) ++
    (reg.openings.filter (pinned.contains ·.table)).flatMap (fun o =>
      o.ports.flatMap fun p => whole o.reader p o.start o.units)).toFinset

/-- What `CircuitCounts` says of a record its check accepted against `P` with the count `c`: wherever `P` is what
C-Flock decodes from `bytes`, at the record `recordOf` reads, the held parameters `heldOf` reads off `pins`, `P`'s
templates' circuit files `fs` (`FilesOf`) and its audit's members `ms` over them (`MembersOf`), at a layout whose wires
are Bits' (`wiresOf`) and with no ingress before `k`, the members' joint program's free inputs outside the registered
rows (`regGates`) number at most `c.free`. -/
def CircuitCountsAt (P : Prog) (record pins : Json) (c : Counts) : Prop :=
  ∀ (bytes : ByteArray) (r : Stated) (held : List String) (fs : List Flock.Circuit) (ms : Members) {L S : ℕ}
    (lay : NetTiming.BoundaryGates L S) (k : ℕ),
    Flock.Program.decode bytes = .ok P → recordOf P record = .ok r → heldOf pins = .ok held → FilesOf P fs →
    MembersOf P fs ms → lay.wires = wiresOf ms → (∀ t < k, lay.ingress t = []) →
    (NetTiming.freeInputs (jointOf ms) lay k \ regGates P fs ms r held).card ≤ c.free

/-- `CircuitCounts` of the circuit check `chk`: `CircuitCountsAt` of every record `chk` accepts, at its count. -/
def CircuitCountsOf (chk : Prog → Json → Json → Json → Except String Counts) : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts), chk P record order pins = .ok c → CircuitCountsAt P record pins c

/-- **An accepted record's free count bounds its circuit's free inputs.** If `NetworkCheck.Circuit.check` accepts the
record `record` against the Program `P` C-Flock decodes, with the pins `pins` and the count `c`, then at `P`'s
templates' circuit files (`FilesOf`) and its audit's members over them (`MembersOf`), and every layout whose wires are
Bits' (`wiresOf`) with no ingress before `k`, the free inputs of the members' joint program (`NetTiming.freeInputs`: its
input gates, neither ingress nor wired) outside the rows the record's reads of held parameters fill (`regGates`)
number at most `c.free`, the bits of the input rows no wire or registered read fills. -/
def CircuitCounts : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts),
    NetworkCheck.Circuit.check P record order pins = .ok c → CircuitCountsAt P record pins c

/-- A circuit check that accepts every record, counting nothing free: what `CircuitCountsOf` must fail at for
`CircuitCounts` (its `…Of` at `check`) to say something. -/
def acceptAllCircuit (_ : Prog) (_ _ _ : Json) : Except String Counts := .ok ⟨0, 0, 0, 0⟩

end Properties.NetworkCertifier

namespace Properties.NetworkPeriod

open Lean (Json)
open NetworkCheck.Period

/-- What `RowsWiresChecked` says of a period its check accepted under the parameters `j` decoded to `p`: each compute
audit `i` is, in `j`'s `audits`, an audit whose `program` decodes through its canonical bytes to a Program `prog`, which
its registration names, and the free inputs the rows form charges it (`chargeOf`: `free`, or `stateFree` where its
`SCAN_STATE` is the scan's state) are what the check counted off `prog`. Where its parameters carry no circuit record,
both are its input gates (`inputGates`). Where they carry one (`Audit.record`: its digest `d` and its pins), the check
read their `circuit` and `order` against `prog` (`NetworkCheck.Circuit.checked`, the record `r` the audit keeps as its
`circuit`), and the two are `NetworkCheck.Circuit.countsOf` of `r` at the pins' parameters and at those and
`stateValue`; `d` is that record's digest and `pins` the pins `pins` names, its registration binds the record by its
digest (`window.circuit`) and records each pin as its parameter's registered root (`pinHeld`), and no pin is the scan's
state, whose reads `free` counts free. -/
def RowsWiresCheckedAt (j : Json) (p : Params) (c : Commitment) : Prop :=
  ∀ (i : ℕ) (au : Audit) (reg : Json), p.audits[i]? = some au → c.audits[i]? = some reg →
    ∃ (xs : Array Json) (a pj : Json) (s : String) (prog : Flock.Program.Prog),
      Flock.J.arr j "audits" "the object" = .ok xs ∧ xs[i]? = some a ∧
      a.getObjVal? "program" = .ok pj ∧ Flock.canon pj = .ok s ∧ Flock.Program.decode s.toUTF8 = .ok prog ∧
      reg.getObjValD "program" = .str (Flock.hexOf prog.sha512) ∧
      match au.record with
      | none => au.free = inputGates prog ∧ au.stateFree = inputGates prog ∧ au.circuit = none
      | some (d, pins) =>
        ∃ (record order pinsJ : Json) (r : NetworkCheck.Circuit.Stated) (sm : SampledProofs.Wires.Summary),
          a.getObjVal? "circuit" = .ok record ∧ a.getObjVal? "order" = .ok order ∧ a.getObjVal? "pins" = .ok pinsJ ∧
          NetworkCheck.Circuit.checked prog record order = .ok (r, sm) ∧
          NetworkCheck.Circuit.pinsOf pinsJ = .ok pins ∧ au.circuit = some r ∧
          au.free = (NetworkCheck.Circuit.countsOf r sm (pins.map (·.1))).free ∧
          au.stateFree = (NetworkCheck.Circuit.countsOf r sm (stateValue :: pins.map (·.1))).free ∧
          NetworkCheck.Circuit.digest record = .ok d ∧ (reg.getObjValD "window").getObjValD "circuit" = .str d ∧
          ∀ q ∈ pins, pinHeld reg q = true ∧ q.1 ≠ stateValue

/-- `RowsWiresChecked` of the parameters' decoder `dec` and the period check `chk`: `RowsWiresCheckedAt` of every
period `chk` accepts under parameters `dec` decoded. -/
def RowsWiresCheckedOf (dec : Json → Except String Params)
    (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    dec j = .ok p → chk p c e = .accept pub prof → RowsWiresCheckedAt j p c

/-- **The period check charges each audit the free inputs it counted off its registered Program.** If the parameters
`j` decode to `p` (`paramsOf`) and the check accepts a period under them, every compute audit's registration names the
Program its descriptor decodes to, and its free inputs, the number its term charges (`free`, or `stateFree` where its
`SCAN_STATE` is the scan's state), are that Program's input gates (`inputGates`), or, where its parameters carry its
window's circuit record, `NetworkCheck.Circuit.countsOf` of the record `NetworkCheck.Circuit.checked` read off that
record and its wire order against the Program, at its pins' parameters, and at those and the scan's state; its
registration then binds that record by its digest (`window.circuit`) and records each pin as its parameter's registered
root (`pinHeld`), and no pin is the scan's state (`stateValue`), so `free` leaves out only the rows its wires fill and
the reads of values the verifier fixed before the run, and `stateFree` those and its reads of the scan's state. -/
def RowsWiresChecked : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    paramsOf j = .ok p → check p c e = .accept pub prof → RowsWiresCheckedAt j p c

/-- What `RowsHeld` says of a period its check accepted under the parameters `j` decoded to `p`, publishing `pub`: in
the rows form, its public names the checks `heldNames` names (`held`), and each holds of it: `circuit`, each phase's tied
audits run the class circuit (`RowsCircuitHeldAt`), and `free`, each audit's free inputs are the count the check made
off its registered Program (`RowsWiresCheckedAt`). -/
def RowsHeldAt (j : Json) (p : Params) (c : Commitment) (pub : Public) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    (publicJson pub).getObjValD "held" = Lean.toJson heldNames ∧ heldNames = #["circuit", "free"] ∧
      RowsCircuitHeldAt p c ∧ RowsWiresCheckedAt j p c

/-- `RowsHeld` of the parameters' decoder `dec` and the period check `chk`: `RowsHeldAt` of every period `chk` accepts
under parameters `dec` decoded, at its public. -/
def RowsHeldOf (dec : Json → Except String Params)
    (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    dec j = .ok p → chk p c e = .accept pub prof → RowsHeldAt j p c pub

/-- **An accepted rows-form public names only checks the period check held.** If the parameters `j` decode to `p`
(`paramsOf`) and the check accepts a rows-form period under them, its public's `held` is `heldNames`, `circuit` and
`free`, and each holds of the period: every phase's tied audits run the class circuit (`RowsCircuitHeldAt`), and every
audit's free inputs are the count the check made off its registered Program, with its circuit record
`NetworkCheck.Circuit.countsOf`'s on the record `NetworkCheck.Circuit.checked` read (`RowsWiresCheckedAt`). A reader of
`K` charges a tied audit its counted free inputs on the strength of that `held`. -/
def RowsHeld : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    paramsOf j = .ok p → check p c e = .accept pub prof → RowsHeldAt j p c pub

/-- The rows-form period of one compute audit (`slotAudit`) and no link-window. -/
def slotPeriod : Params :=
  ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none, .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩

/-- A parameters' decoder that decodes everything to `slotPeriod`: with `acceptAllRows`, what `RowsWiresCheckedOf`
and `RowsHeldOf` must fail at for `RowsWiresChecked` and `RowsHeld` (each its `…Of` at `paramsOf` and `check`) to
say something. -/
def acceptAllParams (_ : Json) : Except String Params := .ok slotPeriod

/-- A compute audit's registration with a circuit record: it names the Program `x`, and its window binds the record
`d` and registers the root `r` for the parameter `k`. -/
def recordReg : Json :=
  Json.mkObj [("program", "x"), ("window", Json.mkObj [("circuit", "d"),
    ("registered_roots", Json.mkObj [("k", "r")])])]

end Properties.NetworkPeriod
