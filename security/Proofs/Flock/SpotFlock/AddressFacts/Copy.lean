module

public import Proofs.Flock.SpotFlock.AddressFacts.Width
public import Proofs.Flock.SpotFlock.UnitRows
public import Properties.Flock.Circuit.Keys
public import Properties.Flock.Circuit.Outs
public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH

/-!
# The copy fact of an address

`RowAt`'s last fact (`CheckedRun.RowAt`): the bit the address's port's row holds at the place, at the unit's instance in
its session's first table, is the bit the layout reads for the gate. The first is a message column of the row
(`Typed.msgCol`), the second a unit column (`layoutFor`'s `wire`: the column `pickAny` chooses for the gate in the unit's
stacked rows, `CheckedRun.pick`), and they are equal only through the session circuit's copy rows. `Addresses` asks it
at every accepted run's checked unit that needs the gate, at one port a unit and one place for all (`CopyHolds`).

`rowAt_of_holds`: the width fact (`WidthHolds`) and the copy fact give `RowAt`.

**What the copy rows give** (`CheckedRun.copy_of_copies`), at a checked unit of an accepted run whose instance's block
is one of its table's (`InBlock`): the session's witness satisfies its first table's accepted statement
(`accepted_check`, `relS_setupOf`), whose copy rows carry into the column the layout reads a message bit of one of the
template instance's rows (`Copies`):

* an input column whose Δ key is message bit `i` of row `q` (`Aliased.srcKey`): Flock's `HiddenExec.copy_siteH`;
* copy `u'`'s computed column for its output `j`, bit `t`, whose output word is word `k` of output port `jo`, at output
  row `nIn + jo` and bit `16 k + t` (`OutCopies`, over the member circuit, the session's unrolled one): Flock's
  `HiddenExec.out_copyH`.

The copy fact then holds where every accepted run's checked unit that needs the gate reads it at a column that copies
the address's port and place (`copyHolds_of_copies`).

The address's place is one for every unit that needs the gate, and the column the layout reads is `pickAny`'s, one of
the unit's columns on the gate's wire, chosen once. Which columns copy which row bit is fixed by the session circuit, so
by its template's (`CheckedRun.ports_member` for the ports); that every column on a gate's wire copies one row bit is
the registration's to state and is not decided here.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **Gate `g`'s copy fact at address `S`**: at every accepted run's checked unit that needs `g`, the bit the address's
port's row holds at its place, at the unit's instance, is the one the layout reads (`CheckedRun.RowAt`'s last fact). -/
def CopyHolds (S : Address) (g : ℕ) : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g →
    _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g)
        (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) (S.port r.u) (S.place.2 / 8)
          (S.place.2 % 8)) =
      _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h g)

/-- **`RowAt` from the width and the copy facts.** -/
theorem rowAt_of_holds {S : Address} {g : ℕ} (hw : WidthHolds h S g) (hc : CopyHolds h S g) :
    ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.RowAt h S g :=
  fun r hr hg => ⟨hw r hr hg, hc r hr hg⟩

/-- **Copy `u'`'s computed column `x` holds output row `q`'s bit `i`**, in the stacked rows of the session statement
`st` whose unrolled circuit is `c`: `x` is the column of copy `u'`'s output `j`, bit `t` (`Layout.outCol`), the copy's
output word is word `k` of output port `jo` (`Hidden.outAt`), and `q = nIn + jo`, `i = 16 k + t`. The unit returns
the outputs (`outNet = unitNet`). -/
def OutCopies (st : _root_.Flock.Stmt) (c : _root_.Flock.Circuit) (x q i : ℕ) : Prop :=
  c.outNet = c.unitNet ∧
    ∃ (hb : Built (FlockSoundness.lunit st.c)) (u' j t jo k : ℕ),
      u' < c.upv ∧ j < (c.leavesOut.getD u' #[]).size ∧ t < 16 ∧
      (FlockSoundness.unitRows st hb).nIn ≤ FlockSoundness.Discharge.Layout.outCol c j t ∧
      FlockSoundness.Discharge.Layout.outCol c j t <
        (FlockSoundness.unitRows st hb).nIn + (FlockSoundness.unitRows st hb).nComp ∧
      x = c.upv * (FlockSoundness.unitRows st hb).nIn +
        (u' * ((FlockSoundness.unitRows st hb).nComp + 1) +
          (1 + (FlockSoundness.Discharge.Layout.outCol c j t - (FlockSoundness.unitRows st hb).nIn))) ∧
      jo < c.outPorts.size ∧ k < (c.outPorts.getD jo default).2 ∧
      FlockSoundness.Discharge.Hidden.outAt c jo + k = (c.leavesOut.getD u' #[])[j]! ∧
      q = _root_.Flock.HmOut.nIn c + jo ∧ i = 16 * k + t

/-- **Column `x` of session statement `st`'s stacked rows copies bit `i` of row `q`** through the copy rows, `c` being
`st`'s unrolled circuit: an input column whose Δ key is message bit `i` of row `q` (`Aliased.srcKey`), or a computed
output column (`OutCopies`). -/
def Copies (st : _root_.Flock.Stmt) (c : _root_.Flock.Circuit) (x q i : ℕ) : Prop :=
  (x < st.c.upv * (st.c.unit.inWords * _root_.Flock.WORD) ∧
      FlockSoundness.Discharge.Aliased.srcKey st.c x = .msg q (i / 8) (i % 8)) ∨
    OutCopies st c x q i

namespace CheckedRun

variable (r : CheckedRun h.stage)

/-- **The column of the unit's stacked rows the layout reads gate `g` at**: `pickAny`'s, for the unit's template and
its index there (`layoutAt`). -/
noncomputable def pick (g : ℕ) : ℕ :=
  pickAny h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) g

/-- **The unit's instance's block is one of its session's first table's blocks**: the block `rankAt / g` the row hash
reads, unreduced, is the layout's block (`β`, reduced mod the table's blocks). -/
def InBlock : Prop := r.n h / r.sts[0]!.c.g < 2 ^ (r.sts[0]!.m - r.sts[0]!.c.kLog)

variable {h r}

/-- **The copy rows at an accepted run's checked unit** (the module's docstring): where the unit's instance's block is
one of its table's, and the column the layout reads gate `g` at copies bit `i` of row `q` (`Copies`, at the unit's
template's member circuit, the session's unrolled one), the row's bit `i` at the unit's instance is the one the layout
reads. -/
theorem copy_of_copies (hr : r.Ok h) (hblk : r.InBlock h) {g q i : ℕ} {m : _root_.Flock.Circuit × Array ℕ}
    (hm : h.ms[templateOf h.ms r.u]? = some m) (hx : Copies r.sts[0]! m.1 (r.pick h g) q i) :
    _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g)
        (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) q (i / 8) (i % 8)) =
      _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h g) := by
  sorry

end CheckedRun

/-- **The copy fact from the copy rows**: where every accepted run's checked unit that needs `g` has its instance's block
in its table, and reads `g` at a column that copies the address's port's bit at its place (`Copies`), the copy fact
holds (`CheckedRun.copy_of_copies`). -/
theorem copyHolds_of_copies {S : Address} {g : ℕ}
    (hc : ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g →
      r.InBlock h ∧ ∃ m, h.ms[templateOf h.ms r.u]? = some m ∧ Copies r.sts[0]! m.1 (r.pick h g) (S.port r.u) S.place.2) :
    CopyHolds h S g := by
  intro r hr hg
  obtain ⟨hblk, m, hm, hx⟩ := hc r hr hg
  exact CheckedRun.copy_of_copies hr hblk hm hx

end SpotFlock
