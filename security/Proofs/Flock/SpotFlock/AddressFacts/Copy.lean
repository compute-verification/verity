module

public import Proofs.Flock.SpotFlock.AddressFacts.Width
public import Proofs.Flock.SpotFlock.UnitRows
public import Properties.Flock.Circuit.Keys
public import Properties.Flock.Circuit.Outs
public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.InScope
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Committed
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
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
    ∃ (hb : FlockSoundness.Built (FlockSoundness.lunit st.c)) (u' j t jo k : ℕ),
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
  (x < st.c.upv * (st.c.unit.inWords * _root_.Flock.WORD) ∧ q < st.c.ports.size ∧
      FlockSoundness.Discharge.Aliased.srcKey st.c x = .msg q (i / 8) (i % 8)) ∨
    OutCopies st c x q i

section RowsFit

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **Each table of a parsed, checked, hidden-output statement has rows that fit** (`Hidden.Any.RowsFitS`): its circuit
is the one its setup commits, whose rows the scope check holds to `v1` ports or bit rows (`accepted_check`'s steps). -/
theorem rowsFitS_check {p : Parsed} (hP : Parses p) (hc : p.check = .ok ()) (hh : p.tags.hiddenOutputs = true)
    {j : ℕ} (hj : j < p.sts.size) : Hidden.Any.RowsFitS p.sts[j].c := by
  obtain ⟨hRu, c0, -, hc0, -, -, hok⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have ht : p.tags.typed = false := hRu.2.1
  obtain ⟨spec, hset⟩ := setup_of_tablesOf (tablesOf_of_parses hP)
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt p.tags p.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  generalize hst : p.sts[j] = st at hd ⊢
  have s := Hidden.setupHidden_spec' hd
  have hcst : HmOut.parseCommit p.tags p.I.circuitFile p.I.tables = .ok st.c :=
    (parseCommit_tagsAt ht _ _ _ _).symm.trans s.parse
  have hc0' : c0 = st.c := by
    unfold ProvedScope.circuitOf at hc0
    rw [hh] at hc0
    simp only [↓reduceIte] at hc0
    exact Except.ok.inj (hc0.symm.trans hcst)
  rw [hc0'] at hok
  have hrows : (Layout.v1Ports st.c || Layout.bitRows st.c) = true := by
    unfold FormOk at hok
    split at hok
    · unfold ProvedScope.scopeOkT at hok
      simp only [Bool.and_eq_true] at hok
      have e : ProvedScope.v2Row = Layout.v2Row := rfl
      have e' : ProvedScope.segRow = Layout.segRow := rfl
      have hr := hok.2
      simp only [ProvedScope.v1Ports, Layout.v1Ports, ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned,
        Layout.unpinned, ProvedScope.bitRows, Layout.bitRows, ProvedScope.bitRowsU, Layout.bitRowsU, e, e'] at hr ⊢
      exact hr
    · rw [hh] at hok
      simp only [↓reduceIte] at hok
      rw [FailClosed.scopeOkI_eq] at hok
      unfold Layout.scopeOkI at hok
      simp only [Bool.and_eq_true] at hok
      exact hok.2.2
  exact HiddenExec.rowsFit_of_committed ht' (Hidden.setupHidden_specCC hd).committed hrows

end RowsFit

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
  have hj := sts_pos hr
  have e0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hj
  have hsm := stmt_member hr
  obtain ⟨_, _, _, hx', -, -, -, hok, hu, hck, -, -, -, hset⟩ := hr
  obtain ⟨hpp, hR⟩ := relS_setupOf hset hck
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨p, hpb, -, -, hc, hh⟩ := parsed_of_stmtOkAt h (stmtOkAt_unit h hx' hok hu) hS
    obtain ⟨rfl, -⟩ := parsed_setupOf hset hpb
    obtain ⟨T, a, hst0, hk, hcirc, hsat⟩ := accepted_check hpp hc hh hj
    have hz := hsat r.wF hR
    have e1 : r.sts[0]! = a.st := e0.trans hst0.symm
    have hkA : T.kLog = a.st.c.kLog := by rw [hk, hst0]
    have hk' : T.kLog = r.p.sts[0]!.c.kLog := hkA.trans (congrArg (fun s : _root_.Flock.Stmt => s.c.kLog) e1).symm
    obtain ⟨hty, ⟨ls, hcc⟩, -⟩ := FlockSoundness.Discharge.Hidden.Committed.check a.ht a.committed
    have hg0 : 0 < a.st.c.g := FlockSoundness.Discharge.Hidden.check_g_pos hty hcc
    have hfs : firstStmt (unitStmt h r.m₂ r.u) = some a.st := by
      show (_root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) (unitStmt h r.m₂ r.u)).bind (·.sts[0]?) = _
      rw [hpb, ← e1, e0]
      exact getElem?_pos r.sts 0 hj
    have hβ : r.β h = r.n h / a.st.c.g % 2 ^ (a.st.m - a.st.c.kLog) := by
      show ((firstStmt (unitStmt h r.m₂ r.u)).map fun st =>
        rankAt h [] r.m₁ (unitStmt h r.m₂ r.u) r.u / st.c.g % 2 ^ (st.m - st.c.kLog)).getD 0 = _
      rw [hfs]
      rfl
    have hblkA : r.n h / a.st.c.g < 2 ^ (a.st.m - a.st.c.kLog) := by
      have hb := hblk
      unfold InBlock at hb
      rwa [e1] at hb
    have hblk' : r.n h / a.st.c.g < 2 ^ T.nbl := hblkA.trans_eq a.nbl_eq
    have hgi : r.n h % a.st.c.g < a.st.c.g := Nat.mod_lt _ hg0
    have hwire : r.wire h g = colAt a.st a.L.unit_built (r.n h % a.st.c.g) (r.pick h g) := by
      classical
      show ((firstStmt (unitStmt h r.m₂ r.u)).map fun st =>
        if hb : FlockSoundness.Built (FlockSoundness.lunit st.c) then
          colAt st hb (rankAt h [] r.m₁ (unitStmt h r.m₂ r.u) r.u % st.c.g)
            (pickAny h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) g)
        else 0).getD 0 = _
      rw [hfs]
      show (if hb : FlockSoundness.Built (FlockSoundness.lunit a.st.c) then
          colAt a.st hb (rankAt h [] r.m₁ (unitStmt h r.m₂ r.u) r.u % a.st.c.g)
            (pickAny h (templateOf h.ms r.u) (r.u - Properties.Spot.unitBase h.ms (templateOf h.ms r.u)) g)
        else 0) = _
      rw [dite_eq_left (show FlockSoundness.Built (FlockSoundness.lunit a.st.c) from a.L.unit_built)]
      rfl
    let o : Fin (2 ^ T.nbl) := ⟨r.n h / a.st.c.g, hblk'⟩
    rw [hβ, Nat.mod_eq_of_lt hblkA, hwire, e1]
    have key : ∀ (cx : Fin a.rows.w) (q' b t : ℕ), cx.val = r.pick h g →
        FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t < 2 ^ a.st.c.kLog →
        T.block (_root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m) o (a.col hgi cx) =
          _root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m (FlockSoundness.Discharge.HiddenExec.siteAddrH a o
            (FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t)) →
        _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / a.st.c.g)
            (FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t) =
          _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / a.st.c.g)
            (colAt a.st a.L.unit_built (r.n h % a.st.c.g) (r.pick h g)) := by
      intro cx q' b t hX hin hcp
      have hcol : (a.col hgi cx).val = colAt a.st a.L.unit_built (r.n h % a.st.c.g) (r.pick h g) := by
        rw [col_val a hgi cx, hX]
      have hy : FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t < 2 ^ T.kLog := by
        rw [hkA]; exact hin
      rw [← hcol]
      calc _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / a.st.c.g)
            (FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t)
          = FlockSoundness.bit (T.block (_root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m) o ⟨_, hy⟩) :=
            (bit_block_zOf r.p r.wF 0 hk' o ⟨_, hy⟩).symm
        _ = FlockSoundness.Discharge.Hm.blockBit T (_root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m) o
            (FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t) := by
            unfold FlockSoundness.Discharge.Hm.blockBit
            rw [dif_pos hy]
        _ = FlockSoundness.bit (_root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m
            (FlockSoundness.Discharge.HiddenExec.siteAddrH a o
              (FlockSoundness.Typed.msgCol a.st.c (r.n h % a.st.c.g) q' b t))) :=
            (FlockSoundness.Discharge.HiddenExec.bit_siteAddrH a _ o hin).symm
        _ = FlockSoundness.bit (T.block (_root_.Flock.Readable.zOf r.p.sts.size r.wF 0 T.m) o (a.col hgi cx)) := by
            rw [hcp]
        _ = _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / a.st.c.g) (a.col hgi cx) :=
            bit_block_zOf r.p r.wF 0 hk' o _
    rw [e1] at hx
    unfold Copies at hx
    rcases hx with ⟨hxl, hq, hkey⟩ | hout
    · have hnIn : a.rows.nIn = a.st.c.upv * (a.st.c.unit.inWords * _root_.Flock.WORD) := by
        show a.st.c.upv * (FlockSoundness.unitRows a.st a.L.unit_built).nIn = _
        rw [a.L.unitRows_nIn]
      let cx : Fin a.rows.w := ⟨r.pick h g, by show _ < a.rows.nIn + a.rows.nComp + 1; omega⟩
      have hin := FlockSoundness.Discharge.Hidden.Committed.msgCol_lt a.ht a.committed hgi hq (i / 8) (i % 8)
        (Nat.mod_lt _ (by norm_num))
      exact key cx q (i / 8) (i % 8) rfl hin
        (FlockSoundness.Discharge.HiddenExec.copy_siteH a hz o hgi cx (by rw [hnIn]; exact hxl) hkey hin)
    · obtain ⟨m', hm', hs⟩ := hsm
      rw [Option.some.inj (hm'.symm.trans hm)] at hs
      have hce : _root_.Flock.HmOut.unrolled a.st.c = m.1 := by
        rw [← hcirc]
        exact hs
      obtain ⟨nets, he⟩ := FlockSoundness.Discharge.Hidden.Committed.unrolled_eq a.ht a.committed
      have hm1 : m.1 = { a.st.c with nets := nets, tmpl := none } := hce.symm.trans he
      have hrfA : FlockSoundness.Discharge.Hidden.Any.RowsFitS a.st.c := by
        rw [hst0]
        exact rowsFitS_check hpp hc hh hj
      have hrf : FlockSoundness.Discharge.Hidden.Any.RowsFitS m.1 := by
        rw [← hce]
        exact (FlockSoundness.Discharge.HiddenExec.rowsFit_unrolled _).2 hrfA
      obtain ⟨ho, hb', u', j, t, jo, k, hu', hj', ht, h1, h2, hxe, hjo, hk2, hw, rfl, rfl⟩ := hout
      have hup : m.1.upv = a.st.c.upv := by rw [hm1]
      have hmul : (u' + 1) * ((FlockSoundness.unitRows a.st hb').nComp + 1) ≤
          a.st.c.upv * ((FlockSoundness.unitRows a.st hb').nComp + 1) :=
        Nat.mul_le_mul_right _ (by rw [← hup]; exact hu')
      rw [add_one_mul] at hmul
      let cx : Fin a.rows.w := ⟨r.pick h g, by
        show _ < a.st.c.upv * (FlockSoundness.unitRows a.st hb').nIn +
          a.st.c.upv * ((FlockSoundness.unitRows a.st hb').nComp + 1) + 1
        rw [hxe, hup]
        omega⟩
      have hjo' : jo < a.st.c.outPorts.size := by rw [hm1] at hjo; exact hjo
      have hin := FlockSoundness.Discharge.Hidden.Committed.msgCol_lt a.ht a.committed hgi
        (show _root_.Flock.HmOut.nIn m.1 + jo < a.st.c.ports.size by rw [hm1]; exact (hrfA.1.2 jo hjo').1)
        ((16 * k + t) / 8) ((16 * k + t) % 8) (Nat.mod_lt _ (by norm_num))
      exact key cx _ _ _ rfl hin
        (FlockSoundness.Discharge.HiddenExec.out_copyH a hce ho hrf hz o hgi cx hu' hj' ht h1 h2 hxe hjo hk2
          (by rw [hw]; exact getElem!_pos _ j hj'))

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
