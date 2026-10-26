import Proofs.Flock.Soundness.Discharge.PlainReg.Defs
import Proofs.Flock.Soundness.Discharge.PlainReg.BindH
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedH
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst

/-!
# The registered check at an accepted hidden-output `verify --registered F` outcome, without `--zk`

`opens_of_acceptsR`: at an outcome `AcceptsR` accepts over hidden outputs at one table, each registered read that is the
verifier's own at the record's draw (`ZkReg.OwnReadsH`) opens the root `own` holds at the position `own` gives, through
`ZkReg.regPathsH`, at the leaf of the population's row. It is `ZkReg.checked_of_acceptsH`'s proof at `setupOf I` in
place of `setupOf (zkInputs I)`: the one table's statement is setupHidden's at the record's draw
(`HiddenExec.setupTables_okH`), `checkRegistered` is `Registered.check` (`ZkReg.checkRegistered_ok`), and the drawn
instance at `u`'s rank holds the population's row of `u` (`Hidden.drawn_digestW`).

The leaf of the population's row at `(u, q)` is the registered leaf of both hidden layouts, the unit's (`layAtH`) and
the template instance's (`layAtHI`), each the population's public file's (`regLeaf_bytes`), so:

* `checked_of_acceptsRH`: the registered reads of `layAtH`, at `rsAtHS`, are checked;
* `checked_of_acceptsRHI`: the registered reads of `layAtHI`, at `HiddenExec.rsAtHI`, are checked.
-/

namespace FlockSoundness.Discharge.PlainReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.ZkReg (OwnReadsH regPathsH checkRegistered_ok check_tables checkPort_at)

variable {mPts n : ℕ} {Reg : Type}

/-- The hm96 leaf of a registered row's bytes, as the hidden layouts compute it, is its `treeLeaf`. -/
theorem regLeaf_bytes (x : ByteArray) :
    (H512 (K512.leafPrefix ++ x.data.toList)).1 = Hm96.Default512.treeLeaf x := by
  rw [H512_val]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ x.data.toList).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++ x)
  rw [← data_append, bytes_of_list]

/-- **Each registered read the verifier holds opens at an accepted outcome.** At an outcome `verify --registered F`
accepts without `--zk` (`own` F's ports) over a hidden-output statement of one table, whose circuit parses to `c` and
whose population's public file loads to `pub`, each registered read that is the verifier's own at the outcome's draw
(`OwnReadsH`) is at a row `(u, q)`, and the leaf of `pub`'s row there opens the root `own` holds at the position `own`
gives, through `regPathsH`. -/
theorem opens_of_acceptsR (I : Inputs) (own : Array Registered.Port) {c : Flock.Circuit}
    (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1)
    {t : Exec.Transcript} (hacc : AcceptsR I own t) {Gate : Type}
    {isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop} {port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port}
    {readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ} (hown : OwnReadsH own c pub t isRead port readAt) :
    ∀ p, isRead p → ∃ (u : Fin n) (q : Fin c.ports.size), p = .inl (u, q) ∧
      Registered.opensLeaf (port p) (Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size)) (readAt p)
        (regPathsH c pub t port p) = true := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  refine ⟨u, q, rfl, ?_⟩
  obtain ⟨spec, sts, hs, hreg, -⟩ := hacc
  rcases setupTables_okH hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · change 1 < I.count at hc'
    omega
  change Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (recordDraw t.record)
    I.partition I.program = .ok st at hst
  rw [hd] at hst
  have s := Hidden.setupHidden_spec' hst
  have hcst : st.c = c := by
    have h1 := s.parseF (HmOut.parse_isTmpl hc)
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, hpub] at hpop'
  cases hpop'
  have hdr' := hdr d rfl
  rw [hcst, hdp] at hdr'
  have hpd : st.pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub d = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg 0 (by simp))
  simp only [List.getElem_toArray, List.getElem_cons_zero] at hck
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hnD, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hnD]; exact hi)
  rw [hread] at hreads
  have hra : readAt (.inl (u, q)) =
      positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPathsH c pub t port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsH, hd, hD, hdp, htb, hent, Except.toOption]
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

/-- **The registered reads are checked at every accepted hidden-output outcome, at the unit's layout.** At an outcome
`verify --registered F` accepts without `--zk` (`own` F's ports), each registered read of the hidden layout `layAtH`
that is the verifier's own at the outcome's draw (`OwnReadsH`) has its registered leaf open the root `own` holds at the
position `own` gives, through `regPathsH`. -/
theorem checked_of_acceptsRH (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1)
    (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsR I own o)
    (rd : FlockSoundness.Registered.ReadsX (layAtH (Reg := Reg) ht hc hscope dj x₀ hU (pub := pub))
      (rsAtHS ht hc hscope dj x₀ hU hpub))
    (hown : OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, rfl, h⟩ := opens_of_acceptsR I own hc hpub hh hn hacc hown p hp
  have hleaf : ((layAtH (Reg := Reg) ht hc hscope dj x₀ hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) :=
    regLeaf_bytes (pub.digest u.val q.val c.ports.size)
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf]
  exact h

/-- **The registered reads are checked at every accepted hidden-output outcome, at the template instance's layout.**
`checked_of_acceptsRH` at `layAtHI` and `rsAtHI`, the general form's. -/
theorem checked_of_acceptsRHI (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)
    (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1)
    (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsR I own o)
    (rd : FlockSoundness.Registered.ReadsX (layAtHI (Reg := Reg) ht hc hrf dj x₀ (pub := pub))
      (rsAtHI ht hc hrf dj x₀ hpub hU hpr hw))
    (hown : OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, rfl, h⟩ := opens_of_acceptsR I own hc hpub hh hn hacc hown p hp
  have hleaf : ((layAtHI (Reg := Reg) ht hc hrf dj x₀ (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) :=
    regLeaf_bytes (pub.digest u.val q.val c.ports.size)
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf]
  exact h

end FlockSoundness.Discharge.PlainReg
