module

public import Definitions.Accounting.ReadUnit
public import Definitions.Assumptions.Flock
public import Flock.ReadNet.Template
public import Properties.Verity.Game
public import Proofs.Flock.Bridge.UnitRows
public import Proofs.Flock.Soundness.ParseStmt

/-!
# A table read the read's template check accepts binds (trusted)

A served unit's table read is a copy of the read's template (`EmbeddingRead_v1{V, H}`, `Flock.ReadNet.readTmpl`) in a
templated circuit file: the verifier parses the file (`Flock.HmOut.parseTmpl`), and `Flock.ReadNet.check V H` accepts its
unit's template only if it is the read's. `TableReadBinding` is a property of that check: where it accepts, every read at
a copy of the parsed file's unit, its ports the copy's gates at the template's columns (`ReadUnit.portsAt`), returns its
table's committed row at the index its token picks, except with probability at most the sampling layer's bound plus the
collision finder's expected SHA-512 calls over `2^256` (`ReadUnit.BindsAt`). Nothing is assumed of the read's gates:
the check fixes them, and the gate refinement and the rows' run are proved of every template it accepts.

C-Flock's unit rows (`Flock.Bridge.copyGate`, `constGate`, `FlockSoundness.Rows.ofNet`) are still in its proofs package, so
this module imports them.
-/

@[expose] public section

namespace Accounting.ReadUnit

/-- Column `c` of the rows, or the constant's where the rows have none: a read's port columns are the read template's,
which every unit the check accepts has. -/
def colAt (R : FlockSoundness.Rows) (c : ℕ) : Fin R.w := if h : c < R.w then ⟨c, h⟩ else R.one

/-- **A table read's ports at copy `j` of unit `u`** of the circuit `D` describes, each the copy's gate at the read
template's column (`Flock.ReadNet`'s layout): the token's 32 bits from column 0, the root's 512 from `ROOT_AT`, the domain
id's 512 from `DOMAIN_AT`, the row's `16 H` from `ROW_AT`, the salt's 1536 from `saltAt H` and the path's 512 a level from
`pathAt H`; then the returned outputs, the row's `16 H` bits and the verdict, on the unit's output columns
(`Layout.outCol`). The table has `V` rows of `H` words, each at its own position. -/
noncomputable def portsAt (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n) {j : ℕ}
    (hj : j < D.unroll.upv) (V H : ℕ) : Ports (Flock.Bridge.circuitOf D hU n) :=
  let gate := fun c => Flock.Bridge.copyGate D hU n u hj (colAt (FlockSoundness.Rows.ofNet hU) c)
  { count := V
    words := H
    position := fun r => r
    tok := fun i => gate i
    root := fun i => gate (Flock.ReadNet.ROOT_AT + i)
    domain := fun i => gate (Flock.ReadNet.DOMAIN_AT + i)
    row := fun i => gate (Flock.ReadNet.ROW_AT + i)
    salt := fun i => gate (Flock.ReadNet.saltAt H + i)
    path := fun k i => gate (Flock.ReadNet.pathAt H + 512 * k + i)
    h := fun i => gate (FlockSoundness.Discharge.Layout.outCol D.unroll (i / 16) (i % 16))
    ok := gate (FlockSoundness.Discharge.Layout.outCol D.unroll H 0) }

/-- **The table reads at `D`'s copies bind.** In the key program of `n` units of `D`'s circuit, with every unit's
constant registered as `true`, take reads `R` whose ports are each at a copy of its unit (`portsAt`), each table
committed under its root and domain id with the served hm96 key (`Reads.Committed`), and any law `Ω` on the verifier's
acceptance `acc`, its drawn units and the trace. If an accepted run draws a read's unit that is incorrect or whose rows'
openings aren't the registered ones with probability at most `ε` (the sampling layer's bound), and SHA-512 is collision
resistant at the finder the reads build from the run (`SHA512CRExpected` at its one coin), then an accepted run draws a
read whose index is inside its table and whose returned row is not the table's row there with probability at most
`ε + E[cost] / 2^256`. -/
def BindsAt (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (V H : ℕ) : Prop :=
  ∀ (n : ℕ) (consts : Fin (Flock.Bridge.circuitOf D hU n).N → Option Bool),
    (∀ u, consts (Flock.Bridge.constGate D hU n u) = some true) →
    ∀ {ι : Type} (R : Reads (Flock.Bridge.circuitOf D hU n) n ι),
    (∀ i ∈ R.reads, ∃ j, ∃ hj : j < D.unroll.upv, R.port i = portsAt D hU n (R.unit i) hj V H) →
    ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (acc : Ω → Prop) (drawn : Ω → Finset (Fin n))
      (tr : Ω → (Flock.Bridge.circuitOf D hU n).Trace) (cost : Ω → ℕ) (ε : ℝ),
    R.Committed Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList →
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      ¬ ((Flock.Bridge.partsOf D hU n).Correct (tr ω) (R.unit i) ∧
        RegisteredRows consts (Flock.Bridge.partsOf D hU n) (R.unit i) (R.port i) (R.root i) (R.domain i) (tr ω))) ≤ ε →
    FlockSoundness.Assumptions.SHA512CRExpected (fun m => (Verity.sha512 m).bytes)
      (Verity.Game.draw (R.finder (Flock.Bridge.partsOf D hU n) Verity.sha512 Flock.Hm96.Default512.key.data.toList
        drawn tr cost).S) (fun _ => PUnit.unit)
      (R.finder (Flock.Bridge.partsOf D hU n) Verity.sha512 Flock.Hm96.Default512.key.data.toList drawn tr cost).out
      (fun s => ((R.finder (Flock.Bridge.partsOf D hU n) Verity.sha512 Flock.Hm96.Default512.key.data.toList
        drawn tr cost).cost s : ℝ)) →
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) ≠
        (R.table i (R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ)

end Accounting.ReadUnit

namespace Accounting.Guarantees

/-- **A table read the read's template check accepts binds** (`ReadUnit.BindsAt`): where `Flock.ReadNet.check V H`
accepts a template `t`, every circuit file the verifier parses (`Flock.HmOut.parseTmpl`) whose unit's template is `t` and
whose outputs are its unit's has its reads at the copies of its unit bind, at `V` rows of `H` words. -/
def TableReadBinding : Prop :=
  ∀ (V H : ℕ) (t : Flock.Tmpl.Tmpl), Flock.ReadNet.check V H t = true →
    ∀ (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
      (hparse : Flock.HmOut.parseTmpl tags bytes tables = .ok ct) (tu : Flock.TmplUnit),
      ct.tmpl = some tu → tu.t = t → ct.outNet = ct.unitNet →
      ReadUnit.BindsAt (.tmpl ct) (FlockSoundness.unrolled_built hparse) V H

end Accounting.Guarantees
