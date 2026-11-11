module

public import Proofs.Flock.SpotFlock.Format
public import Proofs.Flock.SpotFlock.Reads
import Proofs.Flock.Claims.Scope.InScope

/-!
# The session's hidden values at its tables' columns

The deployed relation (`Readable.relation`) runs the session statement's `J` tables as one system of `2^kLog`-bit
blocks (`Readable.sysOf`). In block `β`, the block circuit's input gate `j · 2^kLog + y` is table `j`'s bit `y`, and it
reads hidden value `hidAt J kLog j β y` (`Readable.Sys.idx`). So a layout whose wire for a gate is such an input
decodes the gate's value as that hidden value (`decOf_hid`), and a reveal that opens the hidden values at a row's
columns opens the values the decoder gives.

`parse_setupOf`: the statement the claims' parser reads is the readable form of the setup's own tables (the verifier's
parser returns them, `parsed_setupOf`), so a session statement's tables (`setupOf`, whose registered reads
`SpotFlock.Reads` reads) are the relation's.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **The hidden value at column `y` of table `j`'s block `β`**, among `J` tables of `2^kLog`-bit blocks: the one the
deployed relation's block `β` reads at table `j`'s bit `y` (`Readable.Sys.idx`). -/
def hidAt (J kLog j β y : ℕ) : ℕ := (β * 2 ^ kLog + y) * J + j

/-- **A system's value at an input gate**: at a statement with a system's block circuit and inputs, the value at input
gate `q` of block `β` is the hidden value the system reads there. -/
theorem valueAt_sys {Sy : _root_.Flock.Readable.Sys} {S : Claims.Flock.Stmt} (hC : S.C = Sy.circ)
    (hinp : S.inp = Sy.rel.inp) (β : ℕ) (w : Claims.Flock.Wit) {q : ℕ} (hq : q < Sy.J * Sy.N) :
    valueAt S β w q = w.hid.getD (Sy.idx β q) false := by
  obtain ⟨m, kLog, C, inp, out, pub, form⟩ := S
  obtain rfl : C = Sy.circ := hC
  obtain rfl : inp = Sy.rel.inp := hinp
  have hqN : q < Sy.circ.N := by
    show q < 2 * (Sy.J * Sy.N) + Sy.J + Sy.slots.length
    omega
  have hin : (Sy.circ.op ⟨q, hqN⟩).isInput = true := by
    show (Sy.gate ⟨q, hqN⟩).isInput = true
    simp only [_root_.Flock.Readable.Sys.gate, hq, ↓reduceIte]
    rfl
  exact valueAt_input w hqN hin

/-- In block `β`, table `j`'s bit `y` is the system's hidden value `(β · N + y) · J + j`. -/
theorem idx_table (Sy : _root_.Flock.Readable.Sys) (β : ℕ) {j y : ℕ} (hy : y < Sy.N) :
    Sy.idx β (j * Sy.N + y) = (β * Sy.N + y) * Sy.J + j := by
  unfold _root_.Flock.Readable.Sys.idx
  rw [Nat.mul_comm j, Nat.mul_add_mod, Nat.mod_eq_of_lt hy, Nat.mul_add_div (Nat.zero_lt_of_lt hy),
    Nat.div_eq_of_lt hy, Nat.add_zero]

/-- **A system's value at a table's bit**: at a statement with a system's block circuit and inputs, of `J` tables of
`2^kLog`-bit blocks, the value at table `j`'s bit `y` in block `β` is the hidden value at `hidAt`. -/
theorem valueAt_table {Sy : _root_.Flock.Readable.Sys} {S : Claims.Flock.Stmt} {J kLog : ℕ} (hC : S.C = Sy.circ)
    (hinp : S.inp = Sy.rel.inp) (hJ : Sy.J = J) (hN : Sy.N = 2 ^ kLog) (β : ℕ) (w : Claims.Flock.Wit) {j y : ℕ}
    (hj : j < J) (hy : y < 2 ^ kLog) :
    valueAt S β w (j * 2 ^ kLog + y) = w.hid.getD (hidAt J kLog j β y) false := by
  subst hJ
  unfold hidAt
  rw [← hN] at hy ⊢
  have hq : j * Sy.N + y < Sy.J * Sy.N :=
    calc j * Sy.N + y < j * Sy.N + Sy.N := Nat.add_lt_add_left hy _
      _ = (j + 1) * Sy.N := by rw [Nat.add_mul, Nat.one_mul]
      _ ≤ Sy.J * Sy.N := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hj)
  rw [valueAt_sys hC hinp β w hq, idx_table Sy β hy]

/-- `valueAt_table` at the readable form of a relation that is a system's. -/
theorem valueAt_toReadable {rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation}
    {p : _root_.Flock.Readable.Parsed} {Sy : _root_.Flock.Readable.Sys} {J kLog : ℕ} (hrel : rel p = Sy.rel)
    (hJ : Sy.J = J) (hN : Sy.N = 2 ^ kLog) (β : ℕ) (w : Claims.Flock.Wit) {j y : ℕ} (hj : j < J)
    (hy : y < 2 ^ kLog) :
    valueAt (_root_.Flock.Readable.toReadable rel p) β w (j * 2 ^ kLog + y) =
      w.hid.getD (hidAt J kLog j β y) false :=
  valueAt_table (S := _root_.Flock.Readable.toReadable rel p)
    (by show (rel p).C = Sy.circ; rw [hrel]; rfl) (by show (rel p).inp = Sy.rel.inp; rw [hrel]) hJ hN β w hj hy

/-- The deployed relation's system: its tables' blocks (`Readable.relation`). -/
def relationSys (p : _root_.Flock.Readable.Parsed) : _root_.Flock.Readable.Sys :=
  _root_.Flock.Readable.sysOf ((p.sts[0]?.map (·.m)).getD 0) ((p.sts[0]?.map (·.c.kLog)).getD 0)
    (p.sts.toList.map _root_.Flock.Readable.Tab.ofStmt)

theorem relation_sys (p : _root_.Flock.Readable.Parsed) :
    _root_.Flock.Readable.relation p = (relationSys p).rel := by
  unfold _root_.Flock.Readable.relation
  rfl

theorem relation_J (p : _root_.Flock.Readable.Parsed) : (relationSys p).J = p.sts.size := by
  show (p.sts.toList.map _root_.Flock.Readable.Tab.ofStmt).length = p.sts.size
  rw [List.length_map, Array.length_toList]

theorem relation_N (p : _root_.Flock.Readable.Parsed) :
    (relationSys p).N = 2 ^ (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).kLog :=
  rfl

/-- **The deployed relation's value at an input gate**: in block `β`, gate `j · 2^kLog + y` of the block circuit, table
`j`'s bit `y`, holds the hidden value at `hidAt`. -/
theorem valueAt_relation {p : _root_.Flock.Readable.Parsed} (β : ℕ) (w : Claims.Flock.Wit) {j y : ℕ}
    (hj : j < p.sts.size)
    (hy : y < 2 ^ (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).kLog) :
    valueAt (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) β w
        (j * 2 ^ (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).kLog + y) =
      w.hid.getD
        (hidAt p.sts.size (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).kLog j β y) false :=
  valueAt_toReadable (relation_sys p) (relation_J p) (relation_N p) β w hj hy

/-- **The verifier's parser returns the setup's tables**: at a session statement that sets up as a masked Flock
statement, the statement the verifier's parser returns is the setup's reading with the setup's own tables, and it is
the parser's own. -/
theorem parsed_setupOf {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {p : _root_.Flock.Readable.Parsed}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hp : _root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) xS = some p) :
    p = { ins.read with sts } ∧ _root_.Flock.Readable.Parses p := by
  obtain ⟨hins, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hread := FlockDeploy.Ins.setup_read hsetup
  unfold _root_.Flock.Readable.parseBytes FlockDeploy.readOf at hp
  rw [hins] at hp
  obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
  obtain rfl : ins.read = r := Option.some.inj hr
  refine ⟨?_, _root_.Flock.Readable.parses_of_parse hrp⟩
  unfold _root_.Flock.Readable.Read.parse at hrp
  obtain ⟨sts', hsts', rfl⟩ := Option.map_eq_some_iff.1 hrp
  simp only [_root_.Flock.Readable.Read.tablesOf, hread] at hsts'
  have e : sts = sts' := by
    split_ifs at hsts'
    · split at hsts'
      · exact Option.some.inj hsts'
      · cases hsts'
    · exact Option.some.inj hsts'
  rw [e]

/-- **The claims' parser reads the setup's tables**: at a session statement that sets up as a masked Flock statement,
the statement the claims' parser reads is the readable form of the setup's own tables. -/
theorem parse_setupOf {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {S : Claims.Flock.Stmt} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) :
    S = _root_.Flock.Readable.toReadable _root_.Flock.Readable.relation { ins.read with sts } := by
  unfold FlockDeploy.parse _root_.Flock.Readable.parseOf at hS
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hS
  rw [(parsed_setupOf hset hp).1]

/-- **The decoder at a table's bit**: where a layout's wire for gate `g` of unit `u` is table `j`'s bit `y` in the
session's block circuit, the decoded value is the hidden value at `hidAt` in the unit's block. -/
theorem decOf_hid {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ} {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) {g j y : ℕ} (hj : j < sts.size) (hy : y < 2 ^ S.kLog)
    (hw : L.wire x m₁ xS u g = j * 2 ^ S.kLog + y) :
    decOf L x m₁ xS w u g = w.hid.getD (hidAt sts.size S.kLog j β y) false := by
  rw [decOf_eq hS hβ, hw]
  obtain rfl := parse_setupOf hset hS
  exact valueAt_relation (p := { ins.read with sts }) β w hj hy

end SpotFlock
