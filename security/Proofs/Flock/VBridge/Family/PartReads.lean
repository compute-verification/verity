module

public import Proofs.Flock.VBridge.Family.Registered
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! `RowColKeys` and `SameRow` from the parts' circuits, reads files and registered reads (`rowColKeys_of`,
`sameRow_of`). -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

def RowPort (p : ℕ) (j : Fin F.stmts.size) (q' : Fin (F.Vs j).c.ports.size) : Prop :=
  ∃ q < 128 * (nsOf F.sh p).sum, ∃ k' < Layout.portRowW (F.Vs j).c q',
    Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p q) = .msg q'.val (k' / 8) (k' % 8)

/-- Discharged by the parse of the part's circuit (`circuitFile`'s emit and parse round trip, as for `HUnitName`), at
V*'s parts the verifier's groups. -/
def PartRowCols : Prop :=
  ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
    ∀ q < 128 * (nsOf F.sh p).sum, ∃ (q' : Fin (F.Vs j).c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW (F.Vs j).c q'),
      Aliased.srcKey (F.Vs j).c (0 * (Rows.ofNet (F.Vs j).hU).nIn + rowColOf F.sh p q) = .msg q'.val (k' / 8) (k' % 8)

/-- Discharged by the parse of the parts' circuits (`circuitFile`'s emit and parse round trip, as for `HUnitName`),
at V*'s parts the verifier's groups. -/
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

/-- Discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind. -/
def OwnNamesRows : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ q' : Fin (F.Vs j).c.ports.size, F.RowPort p j q' →
        ∃ pt' ∈ (F.zs R ω t R₂ j).own, pt'.name = (F.Vs j).c.ports[q'].name

/-- Discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind. It is stated at
row ports because each part's `v` and `out` entries are its own. -/
def PartPortsOne : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p p' : ℕ) (j j' : Fin F.stmts.size) (nm : String) (pt : VStar.Part) (nm' : String) (pt' : VStar.Part),
      F.stmts[j] = .inner p nm pt → F.stmts[j'] = .inner p' nm' pt' →
      ∀ (q : Fin (F.Vs j).c.ports.size) (q' : Fin (F.Vs j').c.ports.size), F.RowPort p j q → F.RowPort p' j' q' →
        (F.Vs j).c.ports[q].name = (F.Vs j').c.ports[q'].name →
        ∀ reg ∈ (F.zs R ω t R₂ j).own, reg.name = (F.Vs j).c.ports[q].name →
        ∀ reg' ∈ (F.zs R ω t R₂ j').own, reg'.name = (F.Vs j').c.ports[q'].name → reg = reg'

/-- Discharged by the family's construction of `rd`, with `RowsRegistered`; acceptance of an arbitrary outcome does
not give it, because `rd` is the family's field. -/
def ReadsOwn : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size),
        (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
        (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
          ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
          ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
            some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))

theorem nv_part {p : ℕ} {j : Fin F.stmts.size} {nm : String} {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) :
    (F.Vs j).nv = 2 := by
  rw [F.hnv j, hj]
  rfl

theorem un_val (hcov : F.HCover) {p : ℕ} (hp : p < (partsOf F.sh).size) (r : Fin 2) : (F.un p r).val = r.val := by
  obtain ⟨j, nm, pt, hj⟩ := hcov p hp
  have hsj : F.sess p = j := F.hsess p j ⟨nm, pt, hj⟩
  exact F.hun p r (by rw [hsj, F.nv_part hj]; exact r.isLt)

/-- Discharged by a check that the verifier writes each part's reads file itself, of `HOwn`'s kind; check 5 reads
`find?`'s first entry of a name (`RecCoins.checkRows`), so it does not refuse a second. -/
def OwnNamesOne : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ reg ∈ (F.zs R ω t R₂ j).own, ∀ reg' ∈ (F.zs R ω t R₂ j).own, reg.name = reg'.name → reg = reg'

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
