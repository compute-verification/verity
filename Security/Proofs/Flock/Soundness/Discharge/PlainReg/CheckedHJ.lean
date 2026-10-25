import Proofs.Flock.Soundness.Discharge.PlainReg.Checked
import Proofs.Flock.Soundness.Discharge.PlainHJ.Bind
import Proofs.Flock.Soundness.Discharge.PlainHJ.Decode
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ

set_option Elab.async false

/-!
# The registered check at an accepted plain hidden-output outcome of `J` tables

`flock-verify verify --plain-tables --registered F` (no `--zk`) runs `checkRegistered` on every table's statement
(`PlainReg.AcceptsR`), and table `j`'s is setupHidden's under its tags at its part of the record's draw, `drawJ I d j`
(`PlainHJ.setupOf_tablePHJ`), over `HmOut.drawn c pub (drawJ I d j)`. A read `(u, q)` is at the table `u` is drawn into
(`ZkReg.OwnReadsHJ`), so that table's check is `opens_of_acceptsR`'s at its draw, as `ZkReg.checked_of_acceptsHJ` is
under `--zk`:

* `opens_of_acceptsRHJ`: each registered read the verifier holds opens the root `own` holds, through `ZkReg.regPathsHJ`;
* `checked_of_acceptsRHJ`, `checked_of_acceptsRHJI`: so the registered reads of the plain layout of `J` tables are
  checked at every accepted outcome, at the unit's key program (`PlainHJ.layPHJ`) and at the template instance's
  (`PlainHJ.layPHJI`). The rows `PlainHJ.rsPHJ` and `rsPHJI` read no `pub.tables = none`, so they are the rows at a
  public file with or without shared rows.
-/

namespace FlockSoundness.Discharge.PlainReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL Composed
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.ZkReg (OwnReadsHJ regPathsHJ checkRegistered_ok check_tables checkPort_at)
open FlockSoundness.Discharge.PlainJ (inputsPJ)

variable {mPts n : ℕ} {Reg : Type}

/-- **Each registered read the verifier holds opens at an accepted outcome of `J` tables.** At an outcome
`verify --plain-tables --registered F` accepts (`own` F's ports) over a hidden-output statement whose circuit parses to
`c` and whose population's public file loads to `pub`, each registered read that is the verifier's own at the draw of the
table its unit is drawn into (`OwnReadsHJ`) is at a row `(u, q)`, and the leaf of `pub`'s row there opens the root `own`
holds at the position `own` gives, through `regPathsHJ`. -/
theorem opens_of_acceptsRHJ (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hh : I.tags.hiddenOutputs = true)
    {t : Exec.Transcript} (hacc : AcceptsR I own t) {Gate : Type}
    {isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop} {port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port}
    {readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ} (hown : OwnReadsHJ I own c pub t isRead port readAt) :
    ∀ p, isRead p → ∃ (u : Fin n) (q : Fin c.ports.size), p = .inl (u, q) ∧
      Registered.opensLeaf (port p) (Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size)) (readAt p)
        (regPathsHJ I c pub t port p) = true := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  refine ⟨u, q, rfl, ?_⟩
  obtain ⟨spec, sts, hs, hreg, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := PlainHJ.setupOf_tablePHJ hh hd hs
  have hjs : (partOf I d u.val).val < sts.size := by rw [hsz]; exact (partOf I d u.val).isLt
  have hst := hstJ _ hjs
  have s := Hidden.setupHidden_spec' hst
  have hcst : sts[(partOf I d u.val).val].c = c := by
    have h1 := s.parse
    rw [parseH_tagsAt ht, hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, loadPublicH_tagsAt, hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub (drawJ I d (partOf I d u.val)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [hread] at hreads
  have hra : readAt (.inl (u, q)) =
      positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPathsHJ I c pub t port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsHJ, hd, hD, hdp, htb, hent, Except.toOption]
  have hget : D.proved[D.proved.idxOf u.val]'hi = u.val := by
    have := Array.findIdx_getElem (xs := D.proved) (p := (· == u.val)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  have hdig : pub.digest u.val q.val c.ports.size =
      Registered.tableRow dp tb q.val (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := Hidden.drawn_digestW (Hidden.loadPublic_nil hpub) hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  rw [hra, hpath, hdig]
  exact hopens

/-- **The registered reads are checked at every accepted plain hidden-output outcome of `J` tables, at the unit's
layout.** At an outcome `verify --plain-tables --registered F` accepts (`own` F's ports), each registered read of the
plain layout of `J` tables `PlainHJ.layPHJ` that is the verifier's own at the draw of the table its unit is drawn into
(`OwnReadsHJ`) has its registered leaf open the root `own` holds at the position `own` gives, through `regPathsHJ`. -/
theorem checked_of_acceptsRHJ (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hU : Built c.unit) (hscope : Layout.scopeOkH c = true)
    (hh : I.tags.hiddenOutputs = true) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsR I own o)
    (rd : FlockSoundness.Registered.ReadsX (PlainHJ.layPHJ (Reg := Reg) dj x₀ ht hc hU hscope (pub := pub))
      (PlainHJ.rsPHJ dj x₀ ht hc hpub hU hscope))
    (hown : OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, rfl, h⟩ := opens_of_acceptsRHJ I ht own hc hpub hh hacc hown p hp
  have hleaf : ((PlainHJ.layPHJ (Reg := Reg) dj x₀ ht hc hU hscope (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) :=
    regLeaf_bytes (pub.digest u.val q.val c.ports.size)
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf]
  exact h

/-- **The registered reads are checked at every accepted plain hidden-output outcome of `J` tables, at the template
instance's layout.** `checked_of_acceptsRHJ` at `PlainHJ.layPHJI` and `rsPHJI`. -/
theorem checked_of_acceptsRHJI (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c)
    {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hU : Built c.unit)
    (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (hh : I.tags.hiddenOutputs = true) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsR I own o)
    (rd : FlockSoundness.Registered.ReadsX (PlainHJ.layPHJI (Reg := Reg) dj x₀ ht hc hrf (pub := pub))
      (PlainHJ.rsPHJI dj x₀ ht hc hrf hpub hU hpr hw))
    (hown : OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, rfl, h⟩ := opens_of_acceptsRHJ I ht own hc hpub hh hacc hown p hp
  have hleaf : ((PlainHJ.layPHJI (Reg := Reg) dj x₀ ht hc hrf (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) :=
    regLeaf_bytes (pub.digest u.val q.val c.ports.size)
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf]
  exact h

end FlockSoundness.Discharge.PlainReg
