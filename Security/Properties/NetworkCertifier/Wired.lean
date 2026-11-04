import NetworkCheck.Circuit
import Properties.NetworkScan

/-!
# A compute audit's wired rows: what `NetworkCheck.Circuit.check`'s acceptance gives the sampled audit

A compute audit's Program is a population of template instances whose inputs are all input gates, and its circuit
record (`NetworkCheck.Circuit.Record`) states the dataflow between them, its wires, and the rows its registered reads
fill. The sampled audit reads the wires through its layout (`NetTiming.BoundaryGates.wires`): each wired input gate
reads its writer's value, so it is no free input, and the record's count (`Counts.free`) leaves it out.

**The circuit** (`copied`) is the Program flattened (`Flattens`) with one copy gate a leaf of its root's return, in
return order, which it returns. The root returns every template node's outputs whole and in order
(`NetworkCheck.Circuit.returnsAll`), so the copies are each unit's committed output rows, computed in that unit, and a
wired gate's writer is the copy of its writer's output leaf, whatever that leaf's gate is (a computed gate, a constant,
or an input the template passes through).

**The wires as gates** (`rowGates`): wire `w`'s reader index `i` is reader unit `u` and row `j` (`Wire.readerAt`), and
writer unit `v` (`Wire.writerAt`). Leaf `ℓ` of the row, of `F` leaves (its writer's output field's, `fieldOf`), is leaf
`u · Lₚ + j · F + ℓ` of the root input node the reader's template node reads as its port (`inputArg`, `Lₚ` the port's
leaves), resolved from the root (`Properties.NetworkScan.gateAt`); its writer is the copy of the writer's template node's
output leaf `v · L + off + ℓ` (`off` the field's first leaf, `L` the Definition's output leaves). The registered rows
(`regGates`) are the reader rows of the registered reads of a held parameter and of the table reads and openings of a
table over one, the rows `countsOf` counts as `registered`, leaf for leaf the same way.

* `CircuitWiresFit`: an accepted record's wires fit the circuit of its decoded Program (`BoundaryGates.WiresFit`) over
  any horizon with no ingress: each wired gate an input, each writer returned, and a rank falling along every wire.
* `CircuitCounts`: an accepted record's `free` is at least the circuit's free inputs outside its registered rows.
* `WiredScopeAcc`: `CircuitWiresFit` at the statements of a deployment whose statements are their descriptors' circuits
  (`StmtFlattensCopied`): the wires part of `NetTiming.ScopeAcc`'s `Fits`.
* `Properties.NetworkPeriod.RowsWiresChecked`: the period check (`NetworkCheck.Period`) charges each audit the free
  inputs it counted off its registered Program, and with a circuit record those `Circuit.check` counted on that record,
  which its registration binds by digest, with each pin its registered root.
* `Properties.NetworkPeriod.RowsHeld`: an accepted rows-form public's `held` names `circuit` and `free`, and each holds:
  `RowsCircuitHeldAt` and `RowsWiresCheckedAt`.

Each of `CircuitWiresFit` and `CircuitCounts` says its conclusion (`…At`) of every record `check` accepts, and so is its
property of any circuit check `chk` (`…Of`), which the nontrivial lemmas state false at `acceptAllCircuit`, a check that
accepts every record; `RowsWiresChecked` and `RowsHeld` likewise of the period's decoder and check.
`Proofs.Properties.NetworkCertifier` proves them.
-/

namespace Properties.NetworkCertifier

open Lean (Json)
open Flock.Program (Prog Ty)
open NetworkCheck.Circuit (Record Counts Wire nodeOf fieldOf templateDef recordOf heldOf)
open Properties.NetworkScan (gateAt Flattens)

/-- The leaves of the root's return before template node `k`'s: the root returns each template node's outputs whole and
in order (`returnsAll`), so member `u`'s output leaf `x` of template node `k` is return leaf `retStart P k + u · L + x`,
`L` its Definition's output leaves. -/
def retStart (P : Prog) (k : ℕ) : ℕ :=
  (((NetworkCheck.Period.templateNodes P).takeWhile (· != k)).map fun k' =>
    ((P.defs[P.root]!).body.nodes[k']!).n * (templateDef P k').outLeaves).sum

/-- **The circuit of `P` at its flattening `C`**: `C`'s gates, then one gate a leaf of `P`'s root's return, in return
order, reading that leaf's gate (`.and [g]`, whose value is `g`'s), and returning those copies. Return leaf `j`'s copy is
gate `C.gates.length + j`. -/
def copied (P : Prog) (C : Claims.Sampling.Prog) : Claims.Sampling.Prog :=
  let ret := (P.defs[P.root]!).body.ret
  ⟨C.gates ++ (List.range ret.count).map (fun j => .and [gateAt P P.rootScope (ret.get j)]),
    (List.range ret.count).map (C.gates.length + ·)⟩

/-- **The deployment's statements are their descriptors' circuits**: every statement `D` parses has a descriptor C-Flock
decodes (`Flock.Program.decode`), and its circuit is that Program's flattening with its returned leaves copied
(`copied`). `Properties.NetworkScan.StmtFlattens` with the copies: the deployed parser's premise for compute audits
whose units are wired. -/
def StmtFlattensCopied {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∃ P C, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ Flattens P C ∧ s.prog = copied P C

/-- Template node `k`'s parameter `port`: its index and type. -/
def portParam (P : Prog) (k : ℕ) (port : String) : Option (ℕ × Ty) :=
  ((templateDef P k).params.toList.zipIdx.find? (·.1.1 == port)).map fun ((_, t), i) => (i, t)

/-- Template `tid`'s port `port`'s leaves; 0 where it has none. -/
def portLeaves (P : Prog) (r : Record) (tid port : String) : ℕ :=
  match nodeOf P r tid with
  | some k => ((portParam P k port).map (·.2.leaves)).getD 0
  | none => 0

/-- **The gates of template `tid`'s port `port` at unit `u`**, its leaves `a … a + n − 1`: leaves `u · Lₚ + a …` of the
root input node its template node reads as `port` (`inputArg`; `argsOk` maps member `u` to leaves `[u · Lₚ, (u + 1) · Lₚ)`,
`Lₚ` the port's leaves), each resolved from the root. None where the port is no input node's. -/
def portGates (P : Prog) (r : Record) (tid port : String) (u a n : ℕ) : List ℕ :=
  match nodeOf P r tid with
  | none => []
  | some k => match portParam P k port with
    | none => []
    | some (i, t) => match (((P.defs[P.root]!).body.nodes[k]!).args[i]?).bind (NetworkCheck.Period.inputArg P) with
      | none => []
      | some q => (List.range n).map fun ℓ => gateAt P P.rootScope ⟨false, q, u * t.leaves + a + ℓ⟩

/-- **Wire `w`'s gates**, reader and writer: for each reader index `i` (`Wire.readerAt`: unit `u`, row `j`) and leaf `ℓ`
of the writer's output field `w.out` (`fieldOf`: `F` leaves from leaf `off`), leaf `j · F + ℓ` of the reader's port at
unit `u` (`portGates`), and the copy (`copied`) of return leaf `retStart + v · L + off + ℓ`, the writer's template node's
output leaf `off + ℓ` at unit `v` (`Wire.writerAt`), `L` its output leaves. -/
def wireGates (P : Prog) (r : Record) (w : Wire) : List (ℕ × ℕ) :=
  match nodeOf P r w.writer with
  | none => []
  | some kw => match fieldOf (templateDef P kw).ret w.out with
    | none => []
    | some (t, off, _) => (List.range w.units).flatMap fun i =>
      (portGates P r w.reader w.port (w.readerAt i).1 ((w.readerAt i).2 * t.leaves) t.leaves).zip
        ((List.range t.leaves).map fun ℓ => (P.defs[P.root]!).gates + retStart P kw +
          (w.writerAt i).toNat * (templateDef P kw).outLeaves + off + ℓ)

/-- **The record's wires as gates** of the circuit (`copied`): each gate a wire's reader leaf resolves to
(`wireGates`) ↦ its writer's copy, the first wire's that covers it; none at a gate no wire covers. -/
def rowGates (P : Prog) (r : Record) (g : ℕ) : Option ℕ :=
  ((r.wiring.wires.flatMap (wireGates P r)).find? (·.1 == g)).map (·.2)

/-- **The registered rows' gates** at the held parameters `held` (`heldOf`): each registered read of a held parameter's
reader rows (its pattern's reader indices, `Wire.readerAt`, each row its port's leaves over its cut), and each port of
each table read and opening of a table over a held parameter, whole, at each of its units: the rows `countsOf` counts as
`registered`. -/
def regGates (P : Prog) (r : Record) (held : List String) : Finset ℕ :=
  let reg := r.registered
  let pinned := (reg.tables.filter (held.contains ·.param)).map (·.name)
  let whole := fun (tid p : String) (start n : ℕ) =>
    (List.range n).flatMap fun v => portGates P r tid p (start + v) 0 (portLeaves P r tid p)
  ((reg.reads.filter (held.contains ·.param)).flatMap (fun x =>
      let row := portLeaves P r x.pat.reader x.pat.port / r.wiring.cutsOf x.pat.reader x.pat.port
      (List.range x.pat.units).flatMap fun i =>
        portGates P r x.pat.reader x.pat.port (x.pat.readerAt i).1 ((x.pat.readerAt i).2 * row) row) ++
    (reg.roots.filter (pinned.contains ·.table)).flatMap (fun x => whole x.reader x.port x.start x.units) ++
    (reg.openings.filter (pinned.contains ·.table)).flatMap (fun o =>
      o.ports.flatMap fun p => whole o.reader p o.start o.units)).toFinset

/-- What `CircuitWiresFit` says of a record its check accepted against the Program `P`: wherever `P` is what C-Flock
decodes from `bytes`, at the record `recordOf` reads and every flattening `C` of `P`, a layout whose wires are the
record's (`rowGates`) and with no ingress before `k` fits the circuit's wiring (`copied`, `BoundaryGates.WiresFit`). -/
def CircuitWiresFitAt (P : Prog) (record : Json) : Prop :=
  ∀ (bytes : ByteArray) (r : Record) (C : Claims.Sampling.Prog) {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (k : ℕ),
    Flock.Program.decode bytes = .ok P → recordOf P record = .ok r → Flattens P C → lay.wires = rowGates P r →
    (∀ t < k, lay.ingress t = []) → lay.WiresFit (copied P C) k

/-- `CircuitWiresFit` of the circuit check `chk`: `CircuitWiresFitAt` of every record `chk` accepts. -/
def CircuitWiresFitOf (chk : Prog → Json → Json → Json → Except String Counts) : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts), chk P record order pins = .ok c → CircuitWiresFitAt P record

/-- **An accepted record's wires fit its Program's circuit.** If `NetworkCheck.Circuit.check` accepts the record `record`
with the order `order` against the Program `P` C-Flock decodes, then at every flattening `C` of `P`, a layout whose
wires are the record's (`rowGates`) and with no ingress before `k` fits `copied P C` over `k` (`WiresFit`): each wired
gate is an input and no ingress gate, its writer is a copy the circuit returns, no writer depends on an ingress gate,
and some rank falls along every wire, a wired gate ranking above every wired gate its writer depends on. -/
def CircuitWiresFit : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts),
    NetworkCheck.Circuit.check P record order pins = .ok c → CircuitWiresFitAt P record

/-- What `CircuitCounts` says of a record its check accepted against `P` with the count `c`: wherever `P` is what
C-Flock decodes from `bytes`, at the record `recordOf` reads, the held parameters `heldOf` reads off `pins` and every
flattening `C` of `P`, at a layout whose wires are the record's (`rowGates`) and with no ingress before `k`, the
circuit's free inputs outside the registered rows (`regGates`) number at most `c.free`. -/
def CircuitCountsAt (P : Prog) (record pins : Json) (c : Counts) : Prop :=
  ∀ (bytes : ByteArray) (r : Record) (held : List String) (C : Claims.Sampling.Prog) {L S : ℕ}
    (lay : NetTiming.BoundaryGates L S) (k : ℕ),
    Flock.Program.decode bytes = .ok P → recordOf P record = .ok r → heldOf pins = .ok held → Flattens P C →
    lay.wires = rowGates P r → (∀ t < k, lay.ingress t = []) →
    (NetTiming.freeInputs (copied P C) lay k \ regGates P r held).card ≤ c.free

/-- `CircuitCounts` of the circuit check `chk`: `CircuitCountsAt` of every record `chk` accepts, at its count. -/
def CircuitCountsOf (chk : Prog → Json → Json → Json → Except String Counts) : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts), chk P record order pins = .ok c → CircuitCountsAt P record pins c

/-- **An accepted record's free count bounds its circuit's free inputs.** If `NetworkCheck.Circuit.check` accepts the
record `record` against the Program `P` C-Flock decodes, with the pins `pins` and the count `c`, then at every flattening
`C` of `P` and every layout whose wires are the record's (`rowGates`) with no ingress before `k`, the free inputs of
`copied P C` (`NetTiming.freeInputs`: its input gates, neither ingress nor wired) outside the rows the record's reads of
held parameters fill (`regGates`) number at most `c.free`, the bits of the input rows no wire or registered read fills.
-/
def CircuitCounts : Prop :=
  ∀ (P : Prog) (record order pins : Json) (c : Counts),
    NetworkCheck.Circuit.check P record order pins = .ok c → CircuitCountsAt P record pins c

/-- **An accepted record's wires fit at a deployment's statements.** If `NetworkCheck.Circuit.check` accepts the record
`record` against the Program `P` C-Flock decodes from `bytes`, then at every deployment `D` whose statements are their
descriptors' circuits (`StmtFlattensCopied`), every statement `D` parses from `x`, whose descriptor is `bytes`, whose
layout's wires are the record's (`rowGates`) and with no ingress before its horizon, has its layout's wiring fit its
circuit (`WiresFit`): the wires part of `NetTiming.ScopeAcc`'s `Fits`, `ScanScopeAcc`'s carry for the wires. -/
def WiredScopeAcc : Prop :=
  ∀ (bytes : ByteArray) (P : Prog) (record order pins : Json) (c : Counts) (r : Record),
    Flock.Program.decode bytes = .ok P → NetworkCheck.Circuit.check P record order pins = .ok c →
    recordOf P record = .ok r →
    ∀ {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) (x : Claims.Msg) {L S : ℕ}
      (lay : Claims.Sampling.AuditStatement → NetTiming.BoundaryGates L S) (k : Claims.Sampling.AuditStatement → ℕ),
      StmtFlattensCopied D →
      ∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = bytes → (lay st).wires = rowGates P r →
        (∀ t < k st, (lay st).ingress t = []) → (lay st).WiresFit st.prog (k st)

/-- A circuit check that accepts every record, counting nothing free: what `CircuitWiresFitOf` and `CircuitCountsOf`
must fail at for `CircuitWiresFit` and `CircuitCounts` (each its `…Of` at `check`) to say something. -/
def acceptAllCircuit (_ : Prog) (_ _ _ : Json) : Except String Counts := .ok ⟨0, 0, 0, 0⟩

end Properties.NetworkCertifier

namespace Properties.NetworkPeriod

open Lean (Json)
open NetworkCheck.Period

/-- What `RowsWiresChecked` says of a period its check accepted under the parameters `j` decoded to `p`: each compute
audit `i` is, in `j`'s `audits`, an audit whose `program` decodes through its canonical bytes to a Program `prog`, which
its registration names, and its free inputs (`free`) are what the check counted off `prog`: its input gates
(`inputGates`) where its parameters carry no circuit record, and where they carry one (`Audit.record`: its digest `d`
and its pins), the count `NetworkCheck.Circuit.check` made accepting their `circuit`, `order` and `pins` against `prog`,
`d` that record's digest and the pins `pins`', with its registration binding the record by its digest
(`window.circuit`) and recording each pin as its parameter's registered root (`pinHeld`). -/
def RowsWiresCheckedAt (j : Json) (p : Params) (c : Commitment) : Prop :=
  ∀ (i : ℕ) (au : Audit) (reg : Json), p.audits[i]? = some au → c.audits[i]? = some reg →
    ∃ (xs : Array Json) (a pj : Json) (s : String) (prog : Flock.Program.Prog),
      Flock.J.arr j "audits" "the object" = .ok xs ∧ xs[i]? = some a ∧
      a.getObjVal? "program" = .ok pj ∧ Flock.canon pj = .ok s ∧ Flock.Program.decode s.toUTF8 = .ok prog ∧
      reg.getObjValD "program" = .str (Flock.hexOf prog.sha512) ∧
      match au.record with
      | none => au.free = inputGates prog
      | some (d, pins) =>
        ∃ (record order pinsJ : Json) (cnt : NetworkCheck.Circuit.Counts),
          a.getObjVal? "circuit" = .ok record ∧ a.getObjVal? "order" = .ok order ∧ a.getObjVal? "pins" = .ok pinsJ ∧
          NetworkCheck.Circuit.check prog record order pinsJ = .ok cnt ∧ au.free = cnt.free ∧
          NetworkCheck.Circuit.digest record = .ok d ∧ NetworkCheck.Circuit.pinsOf pinsJ = .ok pins ∧
          (reg.getObjValD "window").getObjValD "circuit" = .str d ∧ ∀ q ∈ pins, pinHeld reg q = true

/-- `RowsWiresChecked` of the parameters' decoder `dec` and the period check `chk`: `RowsWiresCheckedAt` of every
period `chk` accepts under parameters `dec` decoded. -/
def RowsWiresCheckedOf (dec : Json → Except String Params)
    (chk : Params → Commitment → Evidence → Core.Outcome Public Profile) : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    dec j = .ok p → chk p c e = .accept pub prof → RowsWiresCheckedAt j p c

/-- **The period check charges each audit the free inputs it counted off its registered Program.** If the parameters
`j` decode to `p` (`paramsOf`) and the check accepts a period under them, every compute audit's registration names the
Program its descriptor decodes to, and its free inputs (`free`), the number its term charges, are that Program's input
gates (`inputGates`), or, where its parameters carry its window's circuit record, the count of
`NetworkCheck.Circuit.check`, which accepted that record, its wire order and its pins against the Program; its
registration then binds that record by its digest (`window.circuit`) and records each pin as its parameter's registered
root (`pinHeld`), so the count leaves out only the rows its wires fill and the reads of values the verifier fixed before
the run. -/
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
audit's free inputs are the count the check made off its registered Program, with its circuit record the count
`NetworkCheck.Circuit.check` made (`RowsWiresCheckedAt`). A reader of `K` charges a tied audit its counted free inputs
on the strength of that `held`. -/
def RowsHeld : Prop :=
  ∀ (j : Json) (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    paramsOf j = .ok p → check p c e = .accept pub prof → RowsHeldAt j p c pub

end Properties.NetworkPeriod
