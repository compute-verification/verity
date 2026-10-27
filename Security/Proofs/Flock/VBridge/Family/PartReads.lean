import Proofs.Flock.VBridge.Family.Registered

/-!
# VBridge G at V*'s family: the parts' row columns and the reads at them

`RowColKeys` and `SameRow` (`Family/Registered.lean`) split into what the parts' circuits key at their row columns and
what the part sessions' reads files and registered reads say there:

* `RowPort`: a port of part `p`'s circuit that one of its row columns `rowColOf` is keyed to;
* `rowColOf_lt`: each row column is an input (the verifier's `RecResiduals.check`);
* `PartRowCols`, `PartRowBits`: each row column is keyed by a row bit, and two parts' columns of one element of `w` key
  the same bit of rows of one name (the parse of the parts' circuits);
* `OwnNamesRows`, `PartPortsOne`: each part's reads file names its row ports, and two parts' entries at a row port of
  one name are one entry (`rec_vstage.py:356-357`, `rec_residuals.py:446-449`). Not at every port: each part's `v` and
  `out` entries are its own (`rec_residuals.py:451-461`);
* `ReadsOwn`: a part session's registered read is at `own`'s entry of its port's name, at the position it gives the
  unit (`OwnReads`, at a part's labels);
* `nv_part`, `un_val`: a part's statement has two instances (`Stmt.instances`), so rep `r`'s unit is instance `r`;
* `OwnNamesOne`, `readOne_of`: each part's reads file names each port once, so a read's port is its one entry of that
  name, read at the unit's position (`Recursion.vRows_tie`'s and `zero_tie`'s `hone` and `hpos`);
* `rowColKeys_of`, `sameRow_of`: `RowColKeys` and `SameRow` from those and `HCover`.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **A row port of part `p`**: some row column `rowColOf` of the unit of statement `j`, part `p`'s, is keyed (copy 0)
to message bit `k'` of port `q'`, within its row. -/
def RowPort (p : ℕ) (j : Fin F.stmts.size) (q' : Fin (F.Vs j).c.ports.size) : Prop :=
  ∃ q < 128 * (nsOf F.sh p).sum, ∃ k' < Layout.portRowW (F.Vs j).c q',
    Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p q) = .msg q'.val (k' / 8) (k' % 8)

/-- Obligation: discharged by the parse of the part's circuit (`parse_circuitFile` with `circuitFile`'s emit and parse
round trip, the lane that proves `HUnitName`), at V*'s parts the verifier's groups (`HCover`'s lemma). **Each row
column of a part is keyed by a row bit.** In the circuit of part `p`'s statement, the Δ key (`Aliased.srcKey`, copy 0)
of every row column `rowColOf` of the unit is message bit `k'` of a port `q'`, within its row. V*'s `InnerRepCheck`
circuit takes its row ports as its inputs from column 0 (`rowCol`), each a port of the statement under its name
(`rec_residuals.py:125`). That the column is an input is the verifier's check (`rowColOf_lt`). -/
def PartRowCols : Prop :=
  ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
    ∀ q < 128 * (nsOf F.sh p).sum, ∃ (q' : Fin (F.Vs j).c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW (F.Vs j).c q'),
      Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p q) = .msg q'.val (k' / 8) (k' % 8)

/-- Obligation: discharged by the parse of the parts' circuits (`parse_circuitFile` with `circuitFile`'s emit and parse
round trip, the lane that proves `HUnitName`), at V*'s parts the verifier's groups (`HCover`'s lemma). **Two parts'
columns of one element's bit key one bit of rows of one name.** Where parts `p` and `p'` hold the same element `i` of
`w` at positions `a` and `a'`, the keys of their columns of bit `e` are the same bit `b` of ports of one name. A part
reads whole row ports (`wIdx`), each under its name (`rec_residuals.py:125`), so element `i` is bit
`128 (i - start) + e` of one port's row in both. -/
def PartRowBits : Prop :=
  ∀ (p p' : ℕ) (j j' : Fin F.stmts.size) (nm : String) (pt : VStar.Part) (nm' : String) (pt' : VStar.Part),
    F.stmts[j] = .inner p nm pt → F.stmts[j'] = .inner p' nm' pt' →
    ∀ (a a' e : ℕ) (ha : a < (wIOf F.sh p).size) (ha' : a' < (wIOf F.sh p').size),
      (wIOf F.sh p)[a] = (wIOf F.sh p')[a'] → e < 128 →
    ∀ (q : Fin (F.Vs j).c.ports.size) (b : ℕ) (q' : Fin (F.Vs j').c.ports.size) (b' : ℕ),
      Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p (128 * a + e)) =
        .msg q.val (b / 8) (b % 8) →
      Aliased.srcKey (F.Vs j').c (0 * (Rows.ofNet (F.Vs j').hU).nIn + rowColOf F.sh p' (128 * a' + e)) =
        .msg q'.val (b' / 8) (b' % 8) →
      b < Layout.portRowW (F.Vs j).c q → b' < Layout.portRowW (F.Vs j').c q' →
      b = b' ∧ (F.Vs j).c.ports[q].name = (F.Vs j').c.ports[q'].name

/-- Obligation: discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind (lane
hv). **Each part's reads file names its row ports.** At every accepted outer proof, the reads file `own` of part `p`'s
session names every row port of its circuit (`RowPort`). V* writes it with an entry per port of the part
(`rec_vstage.py:356-363`; `rec_residuals.write` refuses reads of other ports, `rec_residuals.py:443`). -/
def OwnNamesRows : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ q' : Fin (F.Vs j).c.ports.size, F.RowPort p j q' →
        ∃ pt' ∈ (F.zs R ω t R₂ j).own, pt'.name = (F.Vs j).c.ports[q'].name

/-- Obligation: discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind (lane
hv). **Two parts' entries at a row port of one name are one entry.** At every accepted outer proof, an entry of part
`p`'s reads file named as a row port `q` of its circuit and an entry of part `p'`s named as a row port `q'` of the same
name are equal. V* writes both as the row's registration (or the verifier's own entry for it) with positions `[0, 1]`
(`rec_vstage.py:356-357`, `rec_residuals.py:449`), and `acc0`, `acc1` as `rec-acc` at `rep_ends`, the same in every
part. It is false at every port: each part's `v` entry is the registration of its own `v` elements and its `out` entry
is under its own salts (`rec_residuals.py:451-461`), which is why it is stated at row ports. -/
def PartPortsOne : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p p' : ℕ) (j j' : Fin F.stmts.size) (nm : String) (pt : VStar.Part) (nm' : String) (pt' : VStar.Part),
      F.stmts[j] = .inner p nm pt → F.stmts[j'] = .inner p' nm' pt' →
      ∀ (q : Fin (F.Vs j).c.ports.size) (q' : Fin (F.Vs j').c.ports.size), F.RowPort p j q → F.RowPort p' j' q' →
        (F.Vs j).c.ports[q].name = (F.Vs j').c.ports[q'].name →
        ∀ reg ∈ (F.zs R ω t R₂ j).own, reg.name = (F.Vs j).c.ports[q].name →
        ∀ reg' ∈ (F.zs R ω t R₂ j').own, reg'.name = (F.Vs j').c.ports[q'].name → reg = reg'

/-- Obligation: discharged by the family's construction of `rd`, with `RowsRegistered` (the transfer's extraction, else
record custody). **A part's registered read is at `own`'s entry of its port's name, at the position it gives the
unit.** At every accepted outer proof, a registered read `(u, q)` of part `p`'s session (at a row port, `v` or `out`)
has as its port an entry of `own` named as `q`, whose position for unit `u` is the read's. `ZkOuter.hown`
(`OwnReads`) says it at the drawn outcomes, at the drawn statement's label of `u`, which is `u` at a part's units;
acceptance of an arbitrary outcome does not give it, and `rd` is the family's field. -/
def ReadsOwn : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size),
        (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
        (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
          ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
          ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
            some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))

/-- **A part's statement has two instances** (`Stmt.instances`, `hnv`). -/
theorem nv_part {p : ℕ} {j : Fin F.stmts.size} {nm : String} {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) :
    (F.Vs j).nv = 2 := by
  rw [F.hnv j, hj]
  rfl

/-- **Rep `r`'s unit of part `p` is instance `r`** (`nv_part` at the statement `HCover` gives, `hun`). -/
theorem un_val (hcov : F.HCover) {p : ℕ} (hp : p < (partsOf F.sh).size) (r : Fin 2) : (F.un p r).val = r.val := by
  obtain ⟨j, nm, pt, hj⟩ := hcov p hp
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  exact F.hun p r (by rw [hsj, F.nv_part hj]; exact r.isLt)

/-- Obligation: discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind (lane
hv), as `OwnNamesRows`. **Each part's reads file names each port once.** At every accepted outer proof, two entries
of part `p`'s reads file of one name are one entry. V* writes one entry per port of the part (`rec_vstage.py:356-363`);
check 5 reads `find?`'s first entry of a name (`RecCoins.checkRows`), so it does not refuse a second. -/
def OwnNamesOne : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ reg ∈ (F.zs R ω t R₂ j).own, ∀ reg' ∈ (F.zs R ω t R₂ j).own, reg.name = reg'.name → reg = reg'

/-- **`Recursion.vRows_tie`'s and `zero_tie`'s `hone` and `hpos` at a part's read** (`ReadsOwn`, `OwnNamesOne`): the
read's port is `own`'s one entry of its name, read at the unit's position. -/
theorem readOne_of (hrd : F.ReadsOwn dg N k Rw) (hnd : F.OwnNamesOne dg N k Rw) {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o} (hacc : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {p : ℕ}
    {j : Fin F.stmts.size} {nm : String} {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) {u : Fin (F.Vs j).nv}
    {q : Fin (F.Vs j).c.ports.size} (hread : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    (∀ reg ∈ (F.zs R ω t R₂ j).own, reg.name = ((F.Vs j).c.ports[q.val]!).name →
      reg = (F.zs R ω t R₂ j).rd.port (.inl (u, q))) ∧
    ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q)))) := by
  obtain ⟨hm, hn, hr⟩ := hrd R ω t R₂ τ o hacc p j nm pt hj u q hread
  refine ⟨fun reg hreg hname => hnd R ω t R₂ τ o hacc p j nm pt hj reg hreg _ hm ?_, hr⟩
  simp only [getElem!_pos (F.Vs j).c.ports q.val q.isLt] at hname
  exact hname.trans hn.symm

/-- **A part's row columns are inputs**: its circuit passes `RecResiduals.check` (`ParseFacts.recResiduals`), so its
unit is the verifier's part `p` (`HUnitName`, `check_part`, `part_eq` at `HStruct`), whose row columns are below its
inputs' end (`rowCol_lt_nIn`). -/
theorem rowColOf_lt {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hname : F.HUnitName) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) {p : ℕ} {j : Fin F.stmts.size}
    {nm : String} {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) (hp : p < (partsOf F.sh).size) {q : ℕ}
    (hq : q < 128 * (nsOf F.sh p).sum) : rowColOf F.sh p q < (Rows.ofNet (F.Vs j).hU).nIn := by
  have hP : partsOf F.sh = gs := partsOf_eq hst.1 hst.2
  have hpg : p < gs.size := by rw [← hP]; exact hp
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname _ p nm pt hj).trans (statements_name F.hstmts _ hj)
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := (parse_facts (F.zs R ω t R₂ j).hc).recResiduals
  obtain ⟨S, ns, nv, hpart, hn⟩ :=
    FlockVBridge.OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hst.1 hst.2 hpg hpart
  have hgp : (partsOf F.sh)[p]! = gs[p] := by rw [hP, getElem!_pos gs p hpg]
  have hnsp : nsOf F.sh p = ns := by rw [nsOf, hgp, hns]
  have hnvp : nvOf F.sh p = nv := by rw [nvOf, hgp, hnv]
  rw [rowColOf, hnsp, hnvp]
  exact OfVStar.rowCol_lt_nIn (F.Vs j).hU hn (by rw [← hnsp]; exact hq)

/-- **`RowColKeys` from the parts' circuits and reads files** (`PartRowCols`, `OwnNamesRows`, at the part's statement
`HCover` gives; the columns inputs by `rowColOf_lt`). -/
theorem rowColKeys_of {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hname : F.HUnitName) (hcov : F.HCover) (hcols : F.PartRowCols) (hnames : F.OwnNamesRows dg N k Rw) :
    F.RowColKeys dg N k Rw := by
  intro R ω t R₂ τ o hacc p hp q hq
  obtain ⟨j, nm, pt, hj⟩ := hcov p hp
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  subst hsj
  obtain ⟨q', k', hk, hkey⟩ := hcols p _ nm pt hj q hq
  exact ⟨F.rowColOf_lt hst hname R ω t R₂ hj hp hq, q', k', hk, hkey,
    hnames R ω t R₂ τ o hacc p _ nm pt hj q' ⟨q, hq, k', hk, hkey⟩⟩

/-- **`SameRow` from the parts' circuits, reads files and registered reads** (`PartRowBits`, `ReadsOwn`,
`PartPortsOne`, at the parts' statements `HCover` gives): the two columns key one bit of rows of one name, both reads
are at the one entry of that name, and both units of rep `r` are instance `r` (`nv_part`, `hun`), so at its position
`r`. -/
theorem sameRow_of (hcov : F.HCover) (hbits : F.PartRowBits) (hrd : F.ReadsOwn dg N k Rw)
    (hone : F.PartPortsOne dg N k Rw) : F.SameRow dg N k Rw := by
  intro R ω t R₂ τ o hacc r p p' a a' e hp hp' ha ha' heq he q b q' b' hkey hkey' hb hb' hread hread'
  obtain ⟨j, nm, pt, hj⟩ := hcov p hp
  obtain ⟨j', nm', pt', hj'⟩ := hcov p' hp'
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  have hsj' : F.sess p' = j' := F.hsess p' j' ⟨nm', pt', hj'⟩
  subst hsj hsj'
  have hsz : (wIOf F.sh p).size = (nsOf F.sh p).sum := by rw [wIOf, nsOf, OfVStar.wIdx_size]
  have hsz' : (wIOf F.sh p').size = (nsOf F.sh p').sum := by rw [wIOf, nsOf, OfVStar.wIdx_size]
  have hrow : F.RowPort p (F.sess p) q := ⟨128 * a + e, by omega, b, hb, hkey⟩
  have hrow' : F.RowPort p' (F.sess p') q' := ⟨128 * a' + e, by omega, b', hb', hkey'⟩
  obtain ⟨rfl, hname⟩ := hbits p p' _ _ nm pt nm' pt' hj hj' a a' e ha ha' heq he q b q' b' hkey hkey' hb hb'
  obtain ⟨hm, hn, hr⟩ := hrd R ω t R₂ τ o hacc p _ nm pt hj (F.un p r) q hread
  obtain ⟨hm', hn', hr'⟩ := hrd R ω t R₂ τ o hacc p' _ nm' pt' hj' (F.un p' r) q' hread'
  have hport := hone R ω t R₂ τ o hacc p p' _ _ nm pt nm' pt' hj hj' q q' hrow hrow' hname _ hm hn _ hm' hn'
  refine ⟨rfl, hport, ?_⟩
  rw [F.un_val hcov hp r, hport] at hr
  rw [F.un_val hcov hp' r] at hr'
  exact Option.some.inj (Option.some.inj (hr.symm.trans hr'))

end VStarFamily

end family

end FlockVBridge.Family
